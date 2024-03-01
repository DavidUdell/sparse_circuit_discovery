# %%
"""
Just chat with a model, with and without feature ablations.

This should help me check (1) whether my logit diffs are correct, and (2)
whether the model is manifestly too dumb to do a task.
"""


from collections import defaultdict

import torch as t
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
)
from sparse_coding.utils.interface import (
    load_yaml_constants,
    load_layer_tensors,
)

_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
PROMPT = "What do you call a chicken that crosses"
ABLATION_LAYER = 5
ABLATION_DIM: int = 883

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator = Accelerator()

# %%
# Just chat.
base_logits = []
sequence = PROMPT  # pylint: disable=invalid-name
for _ in range(40):
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = accelerator.prepare(inputs)
    outputs = model(**inputs)
    base_logits.append(outputs.logits[:, -1, :])
    next_token = t.argmax(outputs.logits[:, -1, :], dim=-1)
    sequence = sequence + tokenizer.decode(*next_token)
print(sequence)

# %%
# Chat with steering ablations.
layer_encoders = load_layer_tensors(
    MODEL_DIR,
    ABLATION_LAYER,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    __file__,
)

layer_decoders = load_layer_tensors(
    MODEL_DIR,
    ABLATION_LAYER,
    DECODER_FILE,
    DEC_BIASES_FILE,
    __file__,
)

activations_dict = defaultdict(list)
altered_logits = []
sequence = PROMPT  # pylint: disable=invalid-name
for _ in range(20):
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = accelerator.prepare(inputs)
    outputs = model(**inputs)
    altered_logits.append(outputs.logits[:, -1, :])
    next_token = t.argmax(outputs.logits[:, -1, :], dim=-1)
    sequence = sequence + tokenizer.decode(*next_token)
with hooks_manager(
    ABLATION_LAYER,
    ABLATION_DIM,
    range(ABLATION_LAYER, ABLATION_LAYER + 2),
    {ABLATION_LAYER + 1: []},
    model,
    {ABLATION_LAYER: layer_encoders, ABLATION_LAYER + 1: layer_encoders},
    {ABLATION_LAYER: layer_decoders, ABLATION_LAYER + 1: layer_decoders},
    activations_dict,
):
    for _ in range(20):
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = accelerator.prepare(inputs)
        outputs = model(**inputs)
        altered_logits.append(outputs.logits[:, -1, :])
        next_token = t.argmax(outputs.logits[:, -1, :], dim=-1)
        sequence = sequence + tokenizer.decode(*next_token)
print(sequence)

# %%
# Look at logits boosted.
assert len(base_logits) == len(altered_logits)
for seq_idx, (base, alt) in enumerate(zip(base_logits, altered_logits)):
    base = t.softmax(base, dim=-1)
    alt = t.softmax(alt, dim=-1)
    diff = alt - base
    if t.max(diff) < 0.0001:
        continue
    diff_ids = t.topk(diff, k=10, dim=-1)
    diff_tokens = tokenizer.decode(*diff_ids[-1])
    print(str(seq_idx) + ": ", diff_tokens.replace("\n", "\\n"))

# %%
# Look at logits suppressed.
for seq_idx, (base, alt) in enumerate(zip(base_logits, altered_logits)):
    base = t.softmax(base, dim=-1)
    alt = t.softmax(alt, dim=-1)
    diff = alt - base
    if t.max(-diff) < 0.0001:
        continue
    diff_ids = t.topk(-diff, k=10, dim=-1)
    diff_tokens = tokenizer.decode(*diff_ids[-1])
    print(str(seq_idx) + ": ", diff_tokens.replace("\n", "\\n"))
