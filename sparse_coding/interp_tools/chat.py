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
PROMPT = "News on the front is that"
ABLATION_LAYER = 2
ABLATION_DIM: int = 885

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator = Accelerator()

# %%
# Just chat.
sequence = PROMPT  # pylint: disable=invalid-name
for _ in range(50):
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = accelerator.prepare(inputs)
    outputs = model(**inputs)
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
sequence = PROMPT  # pylint: disable=invalid-name
for _ in range(20):
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = accelerator.prepare(inputs)
    outputs = model(**inputs)
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
    for _ in range(30):
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = accelerator.prepare(inputs)
        outputs = model(**inputs)
        next_token = t.argmax(outputs.logits[:, -1, :], dim=-1)
        sequence = sequence + tokenizer.decode(*next_token)
print(sequence)

# %%
# Look at logits affected.
