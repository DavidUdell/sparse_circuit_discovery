# %%
"""Validate circuits by measuring faithfulness of top-k nodes."""

import os
import warnings
from collections import defaultdict
from contextlib import ExitStack

import numpy as np
import torch as t
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import wandb

from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
    prepare_autoencoder_and_indices,
)
from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    slice_to_range,
    load_preexisting_graph,
)

# %%
# Load constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
WANDB_MODE = config.get("WANDB_MODE")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
GRADS_DOT_FILE = config.get("GRADS_DOT_FILE")
LOGIT_TOKENS = config.get("LOGIT_TOKENS")
SEED = config.get("SEED")

if WANDB_MODE:
    os.environ["WANDB_MODE"] = WANDB_MODE

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config to wandb.
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)

# %%
# Load and prepare the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    token=HF_ACCESS_TOKEN,
)
accelerator: Accelerator = Accelerator()
model = accelerator.prepare(model)
model.eval()

layer_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)
ablate_layer_range: range = layer_range[:-1]

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

# %%
# Prepare all layer autoencoders and layer dim index lists up front.
layer_encoders, layer_dim_indices = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)
layer_decoders, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)

# %%
# Load circuit from dot file
graph = load_preexisting_graph(MODEL_DIR, GRADS_DOT_FILE, __file__)
if graph is None:
    raise ValueError("No circuit graph found. Please run circuit discovery first.")

# Collect all node weights and sort by magnitude
node_weights = {}
for edge in graph.edges():
    # Extract color from edge which encodes the weight
    color = graph.get_edge(edge[0], edge[1]).attr["color"]
    if color.startswith("#"):
        # Parse RGBA hex color
        r = int(color[1:3], 16)
        b = int(color[5:7], 16)
        a = int(color[7:9], 16) / 255.0
        # Reconstruct weight - positive if blue, negative if red
        weight = a * (1 if b > r else -1)

        # Store weight for both source and target nodes
        for node in edge:
            if "error" not in node:  # Skip error nodes
                layer_idx = int(node.split("_")[-1].split(".")[0])
                feat_idx = int(node.split(".")[-1])
                if layer_idx in node_weights:
                    node_weights[layer_idx][feat_idx] = max(
                        abs(weight), abs(node_weights[layer_idx].get(feat_idx, 0))
                    )
                else:
                    node_weights[layer_idx] = {feat_idx: abs(weight)}

# Sort nodes by weight magnitude for each layer
sorted_nodes = {}
for layer_idx, weights in node_weights.items():
    sorted_nodes[layer_idx] = sorted(weights.items(), key=lambda x: x[1], reverse=True)

# %%
# Validate the circuit with ablations for different k values
print("Base case (no ablation):")
outputs = model(**inputs)
base_logit = outputs.logits[:, -1, :]
base_loss = t.nn.functional.cross_entropy(base_logit, inputs["input_ids"][:, -1])
print(f"Base loss: {base_loss.item():.4f}")

k_values = [1, 5, 10, 20, 50, 100]
for k in k_values:
    print(f"\nAblating top {k} nodes by magnitude:")

    # Collect top k nodes per layer
    ablation_nodes = {}
    for layer_idx, nodes in sorted_nodes.items():
        ablation_nodes[layer_idx] = [n[0] for n in nodes[:k]]
        print(f"Layer {layer_idx}: {len(ablation_nodes[layer_idx])} nodes")

    # Ablate nodes and measure loss
    with ExitStack() as stack:
        for layer_idx, node_indices in ablation_nodes.items():
            if node_indices:  # Only add hooks for layers with nodes to ablate
                stack.enter_context(
                    hooks_manager(
                        layer_idx,
                        node_indices,
                        layer_range,
                        {layer_idx + 1: []},
                        model,
                        layer_encoders,
                        layer_decoders,
                        defaultdict(list),
                    )
                )

        outputs = model(**inputs)
        ablated_logit = outputs.logits[:, -1, :]
        ablated_loss = t.nn.functional.cross_entropy(
            ablated_logit, inputs["input_ids"][:, -1]
        )

        # Calculate and display metrics
        loss_diff = ablated_loss - base_loss
        print(f"Loss: {ablated_loss.item():.4f} (diff: {loss_diff.item():+.4f})")

        # Show probability changes for top tokens
        prob_diff = t.nn.functional.softmax(
            ablated_logit, dim=-1
        ) - t.nn.functional.softmax(base_logit, dim=-1)
        prob_diff = prob_diff.mean(dim=0)

        print("\nLargest probability changes:")
        positive_tokens = prob_diff.topk(LOGIT_TOKENS).indices
        negative_tokens = prob_diff.topk(LOGIT_TOKENS, largest=False).indices

        for token_id in positive_tokens:
            token = tokenizer.decode(token_id)
            print(f"{token}: +{prob_diff[token_id].item()*100:.1f}%")
        print("---")
        for token_id in negative_tokens:
            token = tokenizer.decode(token_id)
            print(f"{token}: {prob_diff[token_id].item()*100:.1f}%")

# %%
# Wrap up logging.
wandb.finish()
