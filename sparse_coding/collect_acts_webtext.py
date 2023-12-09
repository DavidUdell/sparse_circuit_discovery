# %%
"""Collect model activations during inference on `openwebtext`."""


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
    MODEL_DIR, use_fast=True, token=HF_ACCESS_TOKEN
)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, token=HF_ACCESS_TOKEN
)
model = accelerator.prepare(model)
model.eval()

# Validate slice against model.
validate_slice(model, ACTS_LAYERS_SLICE)

# %%
# Dataset.
dataset: list[str] = load_dataset(
    "Elriggs/openwebtext-100k", split="train"
)["text"]

# %%
# Tokenization and inference.
for idx, batch in enumerate(dataset):
    inputs = tokenizer(batch, return_tensors="pt")
    # try:
    #     inputs = accelerator.prepare(inputs)
    #     outputs = model(**inputs)
    #     del outputs
    #     t.cuda.empty_cache()
    # except RuntimeError:
    inputs = inputs.to(model.device)
    inputs = accelerator.prepare(inputs)
    outputs = model(**inputs)

    del inputs
    del outputs
    t.cuda.empty_cache()
    accelerator.clear()
    print(f"Batch {idx} of {len(dataset)}", end="\r")
