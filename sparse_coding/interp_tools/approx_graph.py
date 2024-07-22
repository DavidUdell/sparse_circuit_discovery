# %%
"""A constant-time approximation of the causal graphing algorithm."""

import torch as t
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb

from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    slice_to_range,
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
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
JACOBIANS_FILE = config.get("JACOBIANS_FILE")
JACOBIANS_DOT_FILE = config.get("JACOBIANS_DOT_FILE")
THRESHOLD_EXP = config.get("THRESHOLD_EXP")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")

# Ensure THRESHOLD_EXP is behaving.
if THRESHOLD_EXP is None:
    THRESHOLD_EXP = 0.0
else:
    THRESHOLD_EXP = 2.0**THRESHOLD_EXP

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
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator: Accelerator = Accelerator()

model = accelerator.prepare(model)
layer_range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Forward pass with Jacobian hooks.
