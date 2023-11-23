# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


from collections import defaultdict
from textwrap import dedent

import torch as t

from sparse_coding.interp_tools.utils.hooks import ablations_hook_fac
from sparse_coding.utils.configure import load_yaml_constants, save_paths
from sparse_coding.utils.caching import parse_slice, slice_to_seq
from sparse_coding.rasp.rasp_to_transformer_lens import transformer_lens_model
from sparse_coding.rasp.rasp_torch_tokenizer import tokenize
from sparse_coding.interp_tools.utils.graphs import graph_causal_effects


# %%
# Import constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
SEED = config.get("SEED")

# %%
# Reproducibility.
t.manual_seed(SEED)

# %%
# This implementation validates against just the rasp model. After validation,
# I will generalize to real-world autoencoded models.
if MODEL_DIR != "rasp":
    raise ValueError(
        dedent(
            f"""
            `rasp_cache.py` requires that MODEL_DIR be set to `rasp`, not
            {MODEL_DIR}.
            """
        )
    )

if ACTS_LAYERS_SLICE != slice(0, 2):
    raise ValueError(
        dedent(
            f"""
            `rasp_cache.py` requires that ACTS_LAYERS_SLICE be set to `slice(0,
            2)`, not {ACTS_LAYERS_SLICE}.
            """
        )
    )

# %%
# Record the differential downstream effects of ablating each dim.
prompt = ["BOS", "w", "w", "w", "w", "x", "x", "x", "z", "z"]
token_ids = tokenize(prompt)

base_activations = {}
ablated_activations = {}

# Cache the base activations.
for residual_idx in slice_to_seq(ACTS_LAYERS_SLICE):
    for neuron_idx in range(transformer_lens_model.cfg.d_model):
        _, base_activations[residual_idx, neuron_idx] = (  # pylint: disable=unpacking-non-sequence
            transformer_lens_model.run_with_cache(token_ids)
        )

# Cache the ablated activations.
for residual_idx in slice_to_seq(ACTS_LAYERS_SLICE):
    for neuron_idx in range(transformer_lens_model.cfg.d_model):
        transformer_lens_model.run_with_hooks(
            token_ids,
            fwd_hooks=[
                (
                    "blocks.0.hook_resid_pre",
                    ablations_hook_fac(neuron_idx)
                )
            ],
        )

        transformer_lens_model.reset_hooks()

# %%
# Graph the causal effects.
activation_diffs_by_tokens = {}
activation_diffs = defaultdict(float)

for i, j, tok in ablated_activations:
    activation_diffs_by_tokens[i, j, tok] = (
        ablated_activations[i, j, tok] - base_activations[i, j, tok]
    )

# Sum over tokens.
for i, j, tok in activation_diffs_by_tokens:
    activation_diffs[i, j] += activation_diffs_by_tokens[i, j, tok]

graph_causal_effects(activation_diffs).draw(
    save_paths(__file__, "feature_web.png"), prog="dot"
)
