# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


from contextlib import contextmanager

import torch as t

from sparse_coding.utils.configure import load_yaml_constants
from sparse_coding.utils.caching import parse_slice, slice_to_seq
from sparse_coding.rasp.rasp_to_torch import RaspModel
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
# Ablations context manager and factory.
@contextmanager
def ablations_lifecycle(  # pylint: disable=redefined-outer-name
    torch_model: t.nn.Module, neuron_idx: int
) -> None:
    """Define, register, then unregister hooks."""

    def ablations_hook(  # pylint: disable=unused-argument, redefined-builtin, redefined-outer-name
        module: t.nn.Module, input: tuple, output: t.Tensor
    ) -> None:
        """Zero out a particular neuron's activations."""
        output[:, neuron_idx] = 0.0

    def caching_hook(  # pylint: disable=unused-argument, redefined-builtin, redefined-outer-name
        module: t.nn.Module, input: tuple, output: t.Tensor
    ) -> None:
        """Cache downstream layer activations."""
        activations[(layer_index, neuron_idx, token)] = output.detach()

    # Register the hooks with `torch`. Note that `attn_1` and `attn_2` are
    # hardcoded for the rasp model for now.
    ablations_hook_handle = torch_model.attn_1.register_forward_hook(
        ablations_hook
    )
    caching_hook_handle = torch_model.attn_2.register_forward_hook(
        caching_hook
    )

    try:
        # Yield control flow to caller function.
        yield
    finally:
        # Unregister the ablations hook.
        ablations_hook_handle.remove()
        # Unregister the caching hook.
        caching_hook_handle.remove()


# %%
# This implementation validates against just the rasp model. After validation,
# I will generalize to real-world autoencoded models.

assert MODEL_DIR == "rasp", "MODEL_DIR must be 'rasp`, for now."
assert ACTS_LAYERS_SLICE == slice(
    0, 2
), "ACTS_LAYERS_SLICE must be 0:2, for now."

model = RaspModel()
model.eval()

# Loop over every dim and ablate, recording effects.
activations = {}

for layer_index in slice_to_seq(ACTS_LAYERS_SLICE):
    for neuron_idx in range(7):
        with ablations_lifecycle(model, neuron_idx):
            for literal in [
                ["BOS", "w"],
                ["BOS", "x"],
                ["BOS", "y"],
                ["BOS", "z"],
            ]:
                tokens = model.haiku_model.input_encoder.encode(literal)
                for token in tokens:
                    # Run inference on the ablated model.
                    token = t.tensor(token, dtype=t.int).unsqueeze(0)
                    model(token)

# %%
# Graph the causal effects.
print(activations.keys())
graph_causal_effects(activations).draw("../data/feature_web.png", prog="dot")
