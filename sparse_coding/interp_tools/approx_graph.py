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
    gradients_dict = {}

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
            gradients_dict[f"{sublayer}"] = gradient

    effects_dict = {}
    deltas_dict = {}
    for sublayer in sublayers:
        key: str = f"{sublayer}"

        base_state, changed_state, grad = (
            baseline[key],
            acts[key],
            gradients_dict[key],
        )

        delta = base_state - changed_state.detach()
        effect = delta @ grad

        effects_dict[key] = effect
        deltas_dict[key] = delta
        gradients_dict[key] = grad

    return effects_dict, deltas_dict, gradients_dict


# %%
# Run approximation on the model.
effects, deltas, gradients = approximate(
    model,
    model.transformer.h,
)
