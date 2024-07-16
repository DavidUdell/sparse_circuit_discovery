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

LAYER: int = 1
TOP_K: int = 10
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
    # Proxy elements in the computational graph all need to be saved.
    acts = wrapped_model.transformer.h[LAYER].output[0].detach().save()
    grads = wrapped_model.transformer.h[LAYER].output[0].grad.save()
    logits = wrapped_model.output.logits.save()
    # Backwards pass occurs.
    logits.sum().backward()

    approximation = (-acts * grads).save()

final_pass_logits = approximation.squeeze(0)[-1, :]
effects, indices = t.topk(final_pass_logits, TOP_K)

plotted = t.sum(t.abs(effects)).item()
total = t.sum(t.abs(final_pass_logits)).item()

# %%
# Print effects.
print("Index", "Effect")
print()
for idx, effect in zip(indices, effects):
    print(f"{idx.item()}:", round(effect.item(), None))

print("\n")
print("Plotted:")
print(f"{round((plotted / total)*100.0, 2)}%")
