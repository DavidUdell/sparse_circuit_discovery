# %%
"""A constant-time approximation of the causal graphing algorithm."""

import warnings

from nnsight import LanguageModel
import torch as t
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
)

from sparse_coding.utils.interface import load_yaml_constants


# %%
# Load constants.
_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
SEED = config.get("SEED")

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Load and prepare the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    model = LanguageModel(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator = Accelerator()
model = accelerator.prepare(model)


# %%
# Approximation function.
def approximate(
    base_model,
    sublayers,
):
    """Patch the activations of a model using its gradient."""
    acts = {}
    baseline = {}
    gradients = {}

    # Check all sublayer.output types.
    output_types = {}
    with model.trace(" "):
        for sublayer in sublayers:
            output_types[sublayer] = type(sublayer.output.shape)

    # Cache sublayer acts and gradients.
    with base_model.trace("The Eiddel Tower is in"):
        for sublayer in sublayers:
            if output_types[sublayer] == tuple:
                # Resolve tuple cases before proceeding.
                working_output = sublayer.output[0]
            else:
                working_output = sublayer.output

            activation = working_output
            gradient = working_output.grad

            acts[f"{sublayer}"] = activation
            baseline[f"{sublayer}"] = t.zeros_like(activation)
            gradients[f"{sublayer}"] = gradient

    effects = {}
    deltas = {}
    for sublayer in sublayers:
        key: str = f"{sublayer}"

        patch_state, clean_state, grad = (
            baseline[key],
            acts[key],
            gradients[key],
        )
        delta = (
            patch_state - clean_state.detach()
            if patch_state is not None
            else -clean_state.detach()
        )
        effect = delta @ grad
        effects[sublayer] = effect
        deltas[sublayer] = delta
        gradients[sublayer] = grad

    return acts, gradients


# %%
# Run approximation on the model.
activations, grads = approximate(
    model,
    model.transformer.h,
)
