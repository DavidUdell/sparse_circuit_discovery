# %%
"""
Histograms from autoencoder activations, to set distribution-aware thresholds.
"""


import warnings

import numpy as np
import torch as t
from transformers import AutoModelForCausalLM, PreTrainedModel
import wandb

from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    slice_to_range,
)


# %%
# Set up constants
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
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
# Loop over all the model layers in the slice.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
    )
seq_layer_indices: range = slice_to_range(model, ACTS_LAYERS_SLICE)


# %%
# Compute histograms functionality
def histograms():
    """Compute histograms with `t.histc`."""


# %%
# Wrap up logging
wandb.finish()
