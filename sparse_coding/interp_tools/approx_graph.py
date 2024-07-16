# %%
"""A constant-time approximation of the causal graphing algorithm."""

from nnsight import LanguageModel
import torch as t
from transformers import AutoTokenizer, logging

from sparse_coding.utils.interface import load_yaml_constants


# %%
# Load constants.
_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
SEED = config.get("SEED")

LAYER: int = 2
PROMPT = "Copyright(C"

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Load and prepare the model.
wrapped_model = LanguageModel(MODEL_DIR, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Suppress horribly annoying usage warnings about NNSight's internals.
logging.set_verbosity_error()

# %%
# Approximate the causal effects.
with wrapped_model.trace(PROMPT):
    # Proxy elements saved in the computational graph.
    acts = wrapped_model.transformer.h[LAYER].output[0].detach().save()
    logits = wrapped_model.output.logits.save()
    grad = wrapped_model.transformer.h[LAYER].output[0].grad.save()

    logits.sum().backward()

print(acts.shape)
print(grad.shape)
print((-acts * grad).shape)
