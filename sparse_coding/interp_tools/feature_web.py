# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


from collections import defaultdict
from textwrap import dedent

import torch as t

from sparse_coding.utils.configure import load_yaml_constants, save_paths
from sparse_coding.utils.caching import parse_slice, slice_to_seq
from sparse_coding.rasp.rasp_to_torch import RaspModel
from sparse_coding.interp_tools.utils.graphs import graph_causal_effects
from sparse_coding.interp_tools.utils.hooks import (
    ablations_lifecycle,
    base_caching_lifecycle,
)


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

model = RaspModel()
model.eval()

# Record the differential downstream effects of ablating each dim.
base_activations = {}
ablated_activations = {}

for layer_index in slice_to_seq(ACTS_LAYERS_SLICE):
    for neuron_index in range(7):
        for prompt in [
            ["BOS", "w"],
            ["BOS", "x"],
            ["BOS", "y"],
            ["BOS", "z"],
        ]:
            tokens = model.haiku_model.input_encoder.encode(prompt)
            for token in tokens:
                for context, dictionary in (
                    (base_caching_lifecycle, base_activations),
                    (ablations_lifecycle, ablated_activations),
                ):
                    with context(
                        model,
                        neuron_index,
                        layer_index,
                        token,
                        dictionary,
                    ):
                        # Run inference.
                        model_input = t.tensor(token, dtype=t.int).unsqueeze(0)
                        model(model_input)

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
