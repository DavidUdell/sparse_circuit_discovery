# %%
"""A constant-time approximation of the causal graphing algorithm."""

import warnings

from nnsight import LanguageModel
import torch as t
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
    gradients = {}

    # Address tuple sublayers, to ensure hooks can be registered.
    sublayer_types = {}
    with model.trace(" "):
        for sublayer in sublayers:
            sublayer_types[sublayer] = type(sublayer.output.shape)

    print(sublayer_types.values())
    # Cache sublayer acts and gradients.
    with base_model.trace("The Eiddel Tower is in"):
        for sublayer in sublayers:
            if isinstance(sublayer, tuple):
                print("Sublayer is tuple.")
                sublayer = sublayer[0]
            activation = sublayer.output
            gradient = sublayer.output.grad

            acts[f"{sublayer}"] = activation
            gradients[f"{sublayer}"] = gradient

    hidden_states_patch = {
        k: {"act": t.zeros_like(v.act), "res": t.zeros_like(v.res)}
        for k, v in acts.items()
    }

    effects = {}
    deltas = {}
    for sublayer in sublayers:
        patch_state, clean_state, grad = (
            hidden_states_patch[sublayer],
            acts[sublayer],
            gradients[sublayer],
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
