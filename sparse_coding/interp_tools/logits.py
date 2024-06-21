# %%
"""
Print base model logits for the set prompt.
"""

import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparse_coding.utils.interface import load_yaml_constants


_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
TOPK = 10

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# %%
# Top logits.
inputs = tokenizer(PROMPT, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
for idx in range(logits.shape[1]):
    print(f"Sequence position {idx}:")
    # t.return_types.topk
    seq_logits_data = t.topk(logits[:, idx, :], TOPK)
    seq_probs = t.softmax(logits[:, idx, :], dim=-1).detach()
    indices = seq_logits_data.indices.squeeze().tolist()

    for i in indices:
        token = tokenizer.decode(i)
        token = token.replace("\n", "\\n")
        probability = seq_probs[0, i] * 100

        print(f"{token}: {probability:.1f}%")

    print()
