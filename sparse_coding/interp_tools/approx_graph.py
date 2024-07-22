# %%
"""A constant-time approximation of the causal graphing algorithm."""

import torch as t
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from sparse_coding.utils.interface import load_yaml_constants


# %%
# Load constants.
_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
SEED = config.get("SEED")

LAYER: int = 1
TOP_K: int = 10
PROMPT = "Copyright(C"

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Load and prepare the model.
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# %%
# Forward pass with Jacobian hooks.
