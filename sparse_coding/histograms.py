# %%
"""
Histograms from autoencoder activations, to set distribution-aware thresholds.
"""


import warnings

import numpy as np
import torch as t
from transformers import AutoModelForCausalLM, PreTrainedModel
import wandb

from sparse_coding.interp_tools.utils.computations import Encoder
from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    slice_to_range,
)


# %%
# Set up constants
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
RESID_ACTS_FILE = config.get("ACTS_DATA_FILE")
ATTN_ACTS_FILE = config.get("ATTN_DATA_FILE")
MLP_ACTS_FILE = config.get("MLP_DATA_FILE")
RESID_ENCODER_FILE = config.get("ENCODER_FILE")
RESID_BIASES_FILE = config.get("ENC_BIASES_FILE")
ATTN_ENCODER_FILE = config.get("ATTN_ENCODER_FILE")
ATTN_BIASES_FILE = config.get("ATTN_ENC_BIASES_FILE")
MLP_ENCODER_FILE = config.get("MLP_ENCODER_FILE")
MLP_BIASES_FILE = config.get("MLP_ENC_BIASES_FILE")
SEED = config.get("SEED")

# %%
# Reproducibility
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config/run to wandb
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)


# %%
# Compute histograms functionality
def histograms():
    """Compute histograms with `t.histc`."""


# %%
# Loop over all the model layers in the slice.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
    )
seq_layer_indices: range = slice_to_range(model, ACTS_LAYERS_SLICE)

resid = {
    "acts": RESID_ACTS_FILE,
    "encoder": RESID_ENCODER_FILE,
    "biases": RESID_BIASES_FILE,
}
attn = {
    "acts": ATTN_ACTS_FILE,
    "encoder": ATTN_ENCODER_FILE,
    "biases": ATTN_BIASES_FILE,
}
mlp = {
    "acts": MLP_ACTS_FILE,
    "encoder": MLP_ENCODER_FILE,
    "biases": MLP_BIASES_FILE,
}

# %%
# Wrap up logging
wandb.finish()
