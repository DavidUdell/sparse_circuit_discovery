# %%
"""Validate circuits with simultaneous ablation studies."""


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
)


# %%
# Load constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
SEQ_PER_DIM_CAP = config.get("SEQ_PER_DIM_CAP", 10)
# dict[int, list[int]]
VALIDATION_DIMS_PINNED = config.get("VALIDATION_DIMS_PINNED")
LOGIT_TOKENS = config.get("LOGIT_TOKENS")
SEED = config.get("SEED")

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
# layer_encoders: dict[int, tuple[t.Tensor]]
# layer_dim_indices: dict[int, list[int]]
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
# Sanity check the pinned circuit indices.
if VALIDATION_DIMS_PINNED is not None:
    for k, v in VALIDATION_DIMS_PINNED.items():
        assert (
            k in ablate_layer_range
        ), "Layer range should include one more layer after last pinned layer."
        for i in v:
            assert i in layer_dim_indices[k]

# %%
# Validate the pinned circuit with ablations. Base case first.
outputs = model(**inputs)
base_logit = outputs.logits[:, -1, :]

ALTERED_LOGITS = None
with ExitStack() as stack:
    for k, v in VALIDATION_DIMS_PINNED.items():
        stack.enter_context(
            hooks_manager(
                k,
                v,
                layer_range,
                {k + 1: []},
                model,
                layer_encoders,
                layer_decoders,
                defaultdict(list),
            )
        )

    outputs = model(**inputs)
    altered_logit = outputs.logits[:, -1, :]

# %%
# Compute and display logit diffs.
prob_diff = t.nn.functional.softmax(
    altered_logit, dim=-1
) - t.nn.functional.softmax(base_logit, dim=-1)

prob_diff = prob_diff.mean(dim=0)
positive_tokens = prob_diff.topk(LOGIT_TOKENS).indices
negative_tokens = prob_diff.topk(LOGIT_TOKENS, largest=False).indices
token_ids = t.cat((positive_tokens, negative_tokens), dim=0)

for meta_idx, token_id in enumerate(token_ids):
    if meta_idx == len(positive_tokens):
        print()

    token = tokenizer.convert_ids_to_tokens(token_id.item())
    token = tokenizer.convert_tokens_to_string([token])
    token = token.replace("\n", "\\n")

    print(
        token,
        str(round(prob_diff[token_id.item()].item() * 100, 2)) + "%",
        sep="  ",
    )

# %%
# Wrap up logging.
wandb.finish()
