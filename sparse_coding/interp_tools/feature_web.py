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
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
SEED = config.get("SEED")

# %%
# Reproducibility.
t.manual_seed(SEED)


# %%
# Ablations context manager and factory.
@contextmanager
def ablations_lifecycle(
    model: t.nn.Module, layer_idx: int, neuron_idx: int
) -> None:
    """Define, register, then unregister a specified ablations hook."""

    def ablations_hook(  # pylint: disable=unused-argument, redefined-builtin
        module: t.nn.Module, input: tuple, output: t.Tensor
    ) -> None:
        """Zero out a particular neuron's activations."""
        output[:, neuron_idx] = 0.0

    # Register the hook with `torch`.
    hook_handle = model[layer_idx].register_forward_hook(ablations_hook)

    try:
        # Yield control flow to caller function.
        yield
    finally:
        # Unregister the ablations hook.
        hook_handle.remove()


# %%
# This implementation validates against just the rasp model. After validation,
# I will generalize to real-world autoencoded models.

assert MODEL_DIR == "rasp", "MODEL_DIR must be 'rasp`, for now."
assert ACTS_LAYERS_SLICE == slice(
    0, 2
), "ACTS_LAYERS_SLICE must be 0:2, for now."

# Initialize and prepare the rasp model.
# TODO

# Loop over every dim and ablate, recording effects.
# TODO
