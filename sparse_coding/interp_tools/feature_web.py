# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


from contextlib import contextmanager

import accelerate
import torch as t
from transformers import AutoConfig

from sparse_coding.utils.configure import load_yaml_constants
from sparse_coding.utils.caching import parse_slice


# %%
# Import constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
SEED = config.get("SEED")
tsfm_config = AutoConfig.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
HIDDEN_DIM = tsfm_config.hidden_size
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
PROJECTION_DIM = int(HIDDEN_DIM * PROJECTION_FACTOR)

# %%
# Reproducibility.
t.manual_seed(SEED)

# %%
# Initialize and prepare the model.


# %%
# Ablations hook factory.


# %%
# Loop over every autoencoder dim and ablate, recording effects.
