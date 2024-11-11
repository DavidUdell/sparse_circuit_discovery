# %%
"""
RASP toy model validation. This script does not depend on any prior setup: the
rasp model itself is initialized here.
"""


import os

import numpy as np
import torch as t
import wandb

from sparse_coding.interp_tools.utils.hooks import rasp_ablate_hook_fac
from sparse_coding.utils.interface import load_yaml_constants, save_paths
from sparse_coding.rasp.rasp_to_transformer_lens import transformer_lens_model
from sparse_coding.rasp.rasp_torch_tokenizer import tokenize
from sparse_coding.interp_tools.utils.graphs import graph_causal_effects


# %%
# Import logging and reproducibility constants.
_, config = load_yaml_constants(__file__)

WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
WANDB_MODE = config.get("WANDB_MODE")
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
# Validates against the RASP toy model.
MODEL_DIR = "rasp"
GRAPH_FILE = "rasp_feature_graph.png"
GRAPH_DOT_FILE = ""
TOP_K_INFO_FILE = ""

print("directed_graph_rasp.py is hardcoded for RASP, layers 0 and 1.")

# %%
# Record the differential downstream effects of ablating each dim.
prompt = ["BOS", "w", "w", "w", "w", "x", "x", "x", "z", "z"]
token_ids = tokenize(prompt)

base_activations = {}
ablated_activations = {}

# Cache base activations.
for residual_idx in range(0, 2):
    for neuron_idx in range(transformer_lens_model.cfg.d_model):
        (  # pylint: disable=unpacking-non-sequence
            _,
            base_activations[residual_idx, neuron_idx],
        ) = transformer_lens_model.run_with_cache(token_ids)
# Cache ablated activations.
for residual_idx in range(0, 2):
    for neuron_idx in range(transformer_lens_model.cfg.d_model):
        transformer_lens_model.add_perma_hook(
            "blocks.0.hook_resid_pre",
            rasp_ablate_hook_fac(neuron_idx),
        )

        (  # pylint: disable=unpacking-non-sequence
            _,
            ablated_activations[residual_idx, neuron_idx],
        ) = transformer_lens_model.run_with_cache(token_ids)

        transformer_lens_model.reset_hooks(including_permanent=True)

# Compute effects.
act_diffs = {}
for ablate_layer_idx, neuron_idx in ablated_activations:
    act_diffs[ablate_layer_idx, neuron_idx] = (
        base_activations[(ablate_layer_idx, neuron_idx)][
            "blocks.1.hook_resid_pre"
        ]
        .sum(axis=1)
        .squeeze()
        - ablated_activations[(ablate_layer_idx, neuron_idx)][
            "blocks.1.hook_resid_pre"
        ]
        .sum(axis=1)
        .squeeze()
    )

# %%
# Plot and save effects.
graph_causal_effects(
    act_diffs,
    MODEL_DIR,
    TOP_K_INFO_FILE,
    GRAPH_DOT_FILE,
    0.0,
    0.0,
    None,
    None,
    None,
    __file__,
    rasp=True,
).draw(
    save_paths(__file__, GRAPH_FILE),
    prog="dot",
)

image = wandb.Image(save_paths(__file__, GRAPH_FILE))
wandb.log({GRAPH_FILE: image})

# %%
# Wrap up logging.
wandb.finish()
