# %%
"""Collect model activations during inference on `openwebtext`."""


import gc
import warnings

import numpy as np
import torch as t
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sparse_coding.utils.interface import (
    parse_slice,
    validate_slice,
    cache_layer_tensor,
    slice_to_seq,
    load_yaml_constants,
    save_paths,
    pad_activations,
)


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers v4.31.0"

# %%
# Set up constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
ACTS_DATA_FILE = config.get("ACTS_DATA_FILE")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
NUM_BATCHES_EVALED = config.get("NUM_BATCHES_EVALED", 1000)
SEED = config.get("SEED")

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Model setup.
t.set_grad_enabled(False)
accelerator: Accelerator = Accelerator()
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True,
    token=HF_ACCESS_TOKEN,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
        output_hidden_states=True,
    )
model = accelerator.prepare(model)
model.eval()

validate_slice(model, ACTS_LAYERS_SLICE)
acts_layers_range = slice_to_seq(model, ACTS_LAYERS_SLICE)

# %%
# Dataset.
dataset: list[str] = load_dataset("Elriggs/openwebtext-100k", split="train")[
    "text"
]
dataset_array: np.ndarray = np.array(dataset)

all_indices: np.ndarray = np.random.choice(
    len(dataset_array), size=len(dataset_array), replace=False
)
train_indices: np.ndarray = all_indices[:NUM_BATCHES_EVALED]

# %%
# Tokenization and inference. The taut constraint here is how much memory you
# put into `activations`.
activations: list[t.Tensor] = []
prompt_ids_tensors: list[t.Tensor] = []
for idx, batch in enumerate(dataset_array[train_indices].tolist()):
    try:
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=4000
        ).to(model.device)
        outputs = model(**inputs)
        activations.append(outputs.hidden_states[ACTS_LAYERS_SLICE])
        prompt_ids_tensors.append(inputs["input_ids"].squeeze().cpu())
    except RuntimeError:
        # Manually clear memory and try again.
        gc.collect()
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=4000
        ).to(model.device)
        outputs = model(**inputs)
        activations.append(outputs.hidden_states[ACTS_LAYERS_SLICE])
        prompt_ids_tensors.append(inputs["input_ids"].squeeze().cpu())

# %%
# Save the prompt ids and activations.
prompt_ids_lists = []
for tensor in prompt_ids_tensors:
    prompt_ids_lists.append([tensor.tolist()])
prompt_ids_array: np.ndarray = np.array(prompt_ids_lists, dtype=object)
np.save(PROMPT_IDS_PATH, prompt_ids_array, allow_pickle=True)
# array of (x, 1)
# Each element along x is a list of ints, of seq len.

# Single layer case lacks outer tuple; this solves that.
if isinstance(activations, list) and isinstance(activations[0], t.Tensor):
    activations: list[tuple[t.Tensor]] = [(tensor,) for tensor in activations]
# Tensors are of classic shape: (batch, seq, hidden)

max_seq_length: int = max(
    tensor.size(1) for layers_tuple in activations for tensor in layers_tuple
)

for abs_idx, layer_idx in enumerate(acts_layers_range):
    layer_activations: list[t.Tensor] = [
        pad_activations(
            accelerator.prepare(layers_tuple[abs_idx]),
            max_seq_length,
            accelerator,
        )
        for layers_tuple in activations
    ]
    layer_activations: t.Tensor = t.cat(layer_activations, dim=0)

    cache_layer_tensor(
        layer_activations,
        layer_idx,
        ACTS_DATA_FILE,
        __file__,
        MODEL_DIR,
    )
