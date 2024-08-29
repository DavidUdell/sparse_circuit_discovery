# %%
"""
Circuit discovery with gradient methods, with sparse autoencoders.

Implements the unsupervised circuit discovery algorithm in Baulab 2024.
"""


import re

import torch as t
from accelerate import Accelerator
from pygraphviz import AGraph
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb

from sparse_coding.utils.interface import (
    load_preexisting_graph,
    load_yaml_constants,
    parse_slice,
    sanitize_model_name,
    save_paths,
    slice_to_range,
)
from sparse_coding.interp_tools.utils.hooks import (
    grads_manager,
    prepare_autoencoder_and_indices,
)


# %%
# Load constants.
_, config = load_yaml_constants(__file__)

WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
ATTN_ENCODER_FILE = config.get("ATTN_ENCODER_FILE")
ATTN_ENC_BIASES_FILE = config.get("ATTN_ENC_BIASES_FILE")
ATTN_DECODER_FILE = config.get("ATTN_DECODER_FILE")
ATTN_DEC_BIASES_FILE = config.get("ATTN_DEC_BIASES_FILE")
MLP_ENCODER_FILE = config.get("MLP_ENCODER_FILE")
MLP_ENC_BIASES_FILE = config.get("MLP_ENC_BIASES_FILE")
MLP_DECODER_FILE = config.get("MLP_DECODER_FILE")
MLP_DEC_BIASES_FILE = config.get("MLP_DEC_BIASES_FILE")
RESID_TOKENS_FILE = config.get("TOP_K_INFO_FILE")
ATTN_TOKENS_FILE = config.get("ATTN_TOKEN_FILE")
MLP_TOKENS_FILE = config.get("MLP_TOKEN_FILE")
GRADS_FILE = config.get("GRADS_FILE")
GRADS_DOT_FILE = config.get("GRADS_DOT_FILE")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")

# Or some other means of thresholding approximated effects.
NUM_TOP_EFFECTS: int = 10

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Log to wandb.
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)

# %%
# Load and prepare the model.
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    output_hidden_states=True,
)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator: Accelerator = Accelerator()

model = accelerator.prepare(model)
layer_range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Prepare all layer range autoencoders.
# Residual autoencoders
res_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    RESID_TOKENS_FILE,
    accelerator,
    __file__,
)
res_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    RESID_TOKENS_FILE,
    accelerator,
    __file__,
)

# Attention autoencoders
attn_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ATTN_ENCODER_FILE,
    ATTN_ENC_BIASES_FILE,
    ATTN_TOKENS_FILE,
    accelerator,
    __file__,
)
attn_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ATTN_DECODER_FILE,
    ATTN_DEC_BIASES_FILE,
    ATTN_TOKENS_FILE,
    accelerator,
    __file__,
)

# MLP autoencoders
mlp_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    MLP_ENCODER_FILE,
    MLP_ENC_BIASES_FILE,
    MLP_TOKENS_FILE,
    accelerator,
    __file__,
)
mlp_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    MLP_DECODER_FILE,
    MLP_DEC_BIASES_FILE,
    MLP_TOKENS_FILE,
    accelerator,
    __file__,
)

# %%
# Load preexisting graph, if available.
graph = load_preexisting_graph(MODEL_DIR, GRADS_DOT_FILE, __file__)
if graph is None:
    graph = AGraph(directed=True)

save_graph_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_FILE}"
)
save_dot_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_DOT_FILE}"
)

# %%
# Define cross-entropy loss.
metric = t.nn.CrossEntropyLoss()

# %%
# Model passes.
print("Prompt:")
print()
print(PROMPT)

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
acts_dict: dict = None
grads_dict: dict = None
marginal_grads_dict: dict = {}

with grads_manager(
    model,
    layer_range,
    res_enc_and_biases,
    res_dec_and_biases,
    attn_enc_and_biases,
    attn_dec_and_biases,
    mlp_enc_and_biases,
    mlp_dec_and_biases,
) as acts_and_grads:

    # Forward pass installs all backward hooks.
    output = model(**inputs)

    # Metric backward pass.
    metric(
        output.logits.squeeze(),
        inputs["input_ids"].squeeze(),
    ).backward(retain_graph=True)

    acts_dict, grads_dict = acts_and_grads

    # Add model_dim activations to acts_dict, if needed.
    for grad in grads_dict:
        if "res_error" in grad:
            idx: int = int(grad.split("_")[-1])
            act: t.Tensor = output.hidden_states[idx]
            acts_dict[grad] = act

    # Compute Jacobian-vector products.
    for act in acts_dict:
        assert act in grads_dict

    for loc, grad in grads_dict.items():
        assert loc in acts_dict

        # The grad must be treated as a constant in the jvp calculation.
        grad = grad.squeeze().unsqueeze(0).detach()
        act = acts_dict[loc].squeeze().unsqueeze(0)

        # The jvp at an activation is a scalar with gradient tracking that
        # represents how well the model would do on the loss metric in that
        # forward pass, if all later modules were replaced with a best-fit
        # first-order Taylor approximation.
        jvp = t.einsum("bsd,bsd->bs", grad, act).squeeze()
        jvp[-1].backward(retain_graph=True)
        _, marginal_grads = acts_and_grads

        # The marginal grads are how much each activation scalar contributes to
        # the jvp estimation of the loss. We define our edges here where the
        # bottom node is the jvp position and the top node is the upstream
        # activation position of our choice.
        down_idx = int(loc.split("_")[-1])
        up_idx = down_idx - 1
        if up_idx not in layer_range:
            continue

        # res_x
        # mlp_x
        # attn_x
        # res_error_x
        # mlp_error_x
        # attn_error_x
        if "attn_" in loc:
            # Upstream res_
            marginal_grads_dict[f"res_{up_idx}_to_" + loc] = marginal_grads[
                f"res_{up_idx}"
            ]
            marginal_grads_dict[f"res_error_{up_idx}_to_" + loc] = (
                marginal_grads[f"res_error_{up_idx}"]
            )
        elif "mlp_" in loc:
            # Same-layer attn_
            marginal_grads_dict[f"attn_{down_idx}_to_" + loc] = marginal_grads[
                f"attn_{down_idx}"
            ]
            marginal_grads_dict[f"attn_error_{down_idx}_to_" + loc] = (
                marginal_grads[f"attn_error_{down_idx}"]
            )
        elif "res_" in loc:
            # Upstream res_
            marginal_grads_dict[f"res_{up_idx}_to_" + loc] = marginal_grads[
                f"res_{up_idx}"
            ]
            marginal_grads_dict[f"res_error_{up_idx}_to_" + loc] = (
                marginal_grads[f"res_error_{up_idx}"]
            )
            # Same-layer mlp_
            marginal_grads_dict[f"mlp_{down_idx}_to_" + loc] = marginal_grads[
                f"mlp_{down_idx}"
            ]
            marginal_grads_dict[f"mlp_error_{down_idx}_to_" + loc] = (
                marginal_grads[f"mlp_error_{down_idx}"]
            )


# %%
# Double-counting correction functionality.
def dedupe(overall_edge: str, val, edges_dict: dict):
    """
    Deduplicate effect sizes for GPT-2 edges.

    These cases specifically need to account for double-counting:
    res_error_ to res_error_
    res_ to res_
    """

    # Regex: start of string, "x_", minimal selection of any characters, then
    # "y_".
    regexes: list[str] = [
        "^res_error_.*?res_error_",
        "^res_\d+.*?res_\d+",
    ]

    for regex in regexes:
        if re.match(regex, overall_edge) is not None:
            print("Match:", overall_edge)
            edge_ends: tuple = overall_edge.split("_to_")
            res_up: str = edge_ends[0]
            res_same: str = edge_ends[-1]

            attn_same: str = res_same.replace("res_", "attn_")
            mlp_same: str = res_same.replace("res_", "mlp_")

            res_to_attn_edge: str = f"{res_up}_to_{attn_same}"
            attn_to_mlp_edge: str = f"{attn_same}_to_{mlp_same}"
            mlp_to_res_edge: str = f"{mlp_same}_to_{res_same}"
            print(
                "Confounds:",
                res_to_attn_edge,
                attn_to_mlp_edge,
                mlp_to_res_edge,
            )

            for confound in [
                res_to_attn_edge,
                attn_to_mlp_edge,
                mlp_to_res_edge,
            ]:
                assert confound in edges_dict

                val -= edges_dict[confound]

            # If it's error you don't need to also check projected.
            break

    return val


# %%
# Render the graph.
for i, v in marginal_grads_dict.items():
    v = dedupe(i, v, marginal_grads_dict)
    # Normal graphing can now take place using i and v. We'll plot only the
    # contributions of the final forward pass using the next line.
    v = v.detach().squeeze(0)[-1, :]

    print(i)
    values, indices = t.topk(v, NUM_TOP_EFFECTS)
    # Negative topk as well.
    values, indices = values.tolist(), indices.tolist()
    for value, idx in zip(values, indices):
        if round(value) == 0:
            continue
        print(f"{idx}: {value:.2f}")
    print("\n")
