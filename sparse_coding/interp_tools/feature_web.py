# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


from contextlib import contextmanager

import accelerate
import torch as t

from sparse_coding.utils.configure import load_yaml_constants
from sparse_coding.utils.caching import parse_slice
from sparse_coding.rasp.rasp_to_torch import RaspModel


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
    torch_model: t.nn.Module, layer_index: int, neuron_idx: int
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

    # Register the hooks with `torch`.
    ablations_hook_handle = torch_model[layer_index].register_forward_hook(
        ablations_hook
    )
    caching_hook_handle = torch_model[layer_index + 1].register_forward_hook(
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
accelerator = accelerate.Accelerator()
model = accelerator.prepare(model)

# Loop over every dim and ablate, recording effects.
activations = {}

for layer_index in ACTS_LAYERS_SLICE:
    for neuron_idx in range(attention_dim):
        with ablations_lifecycle(model, layer_index, neuron_idx):
            for token in ["w", "x", "y", "z"]:
                # Run inference on the ablated model.
                output = model(token)
                # Record the downstream activations.
                activations[(layer_index, neuron_idx, token)] = output.hiddens

# %%
# Graph the causal effects.
print(activations.keys())
