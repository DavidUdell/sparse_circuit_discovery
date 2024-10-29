# %%
"""
Histograms from autoencoder activations, to set distribution-aware thresholds.
"""


import numpy as np
import torch as t
import wandb

from sparse_coding.utils.interface import load_yaml_constants


# %%
# Set up constants
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
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
# Compute histograms functionality
def histograms():
    """Compute histograms with `t.histc`."""


# %%
# Wrap up logging
wandb.finish()
