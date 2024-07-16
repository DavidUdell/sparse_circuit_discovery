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

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Load and prepare the model.
wrapped_model = LanguageModel(MODEL_DIR, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
logging.set_verbosity_error()

with wrapped_model.trace("Copyright(C"):
    # Proxy elements saved in the computational graph.
    grad = wrapped_model.transformer.h[2].output[0].grad.save()
    logits = wrapped_model.output.logits.save()

    logits.sum().backward()

print(logits.detach())
print(grad)

# %%
# Approximation function.
# def approximate(
#     model,
#     sublayers,
# ):
#     """Patch the activations of a model using its gradient."""
#     acts = {}
#     grads = {}

#     # Check all sublayer.output types.
#     with model.trace("The Eiffel Tower is in"):
#         for sublayer in sublayers:
#             acts[f"{sublayer}"] = sublayer.output[0].save()
#             grads[f"{sublayer}"] = sublayer.output[0].grad.save()
#         metric = t.nn.functional.l1_loss(model)
#         metric.sum().backward()
#     return acts, grads


# # %%
# # Run approximation on the model.
# effects, gradients = approximate(
#     wrapped_model,
#     wrapped_model.transformer.h,
# )

# # Computations after inference.
# for value in gradients.values():
#     print(value)
