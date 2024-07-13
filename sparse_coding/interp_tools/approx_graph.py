# %%
"""A constant-time approximation of the causal graphing algorithm."""

import warnings
from collections import namedtuple

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
# Function and classes.
EffectOut = namedtuple(
    "EffectOut", ["effects", "deltas", "grads", "total_effect"]
)


def patch_act(
    base_model,
    sublayers,
):
    """Patch the activations of a model using its gradient."""

    # Cache sublayer acts and gradients.
    with base_model.trace("The Eiddel Tower is in"):
        for sublayer in sublayers:
            activation = sublayer.output.save()
            gradient = sublayer.output.grad

    acts = {k: v.value for k, v in acts.items()}
    gradients = {k: v.value for k, v in gradients.items()}

    hidden_states_patch = {
        k: {"act": t.zeros_like(v.act), "res": t.zeros_like(v.res)}
        for k, v in acts.items()
    }
    total_effect = None

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
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, gradients, total_effect)


# %%
# Run approximation on the model.
patch_act(
    model,
    model.transformer.h,
)
