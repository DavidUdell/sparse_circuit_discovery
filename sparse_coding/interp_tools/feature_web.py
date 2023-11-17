# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


import torch as t

from sparse_coding.utils.configure import load_yaml_constants
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

assert MODEL_DIR == "rasp", "MODEL_DIR must be 'rasp`, for now."
assert ACTS_LAYERS_SLICE == slice(
    0, 2
), "ACTS_LAYERS_SLICE must be 0:2, for now."

model = RaspModel()
model.eval()

# Record the differential downstream effects of ablating each dim.
base_activations: dict = {}
ablated_activations: dict = {}

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
print(base_activations.keys())
print(ablated_activations.keys())
print(base_activations.values())
print(ablated_activations.values())
graph_causal_effects(base_activations).draw(
    "../data/feature_web.png", prog="dot"
)
