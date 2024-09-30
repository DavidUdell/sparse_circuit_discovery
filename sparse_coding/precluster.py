# %%
"""Partition the dataset into k-clusters based on activation cosines."""


import sys
import warnings

import datasets
import numpy as np
import torch as t
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import wandb

from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_range,
    load_input_token_ids,
    load_yaml_constants,
    save_paths,
)


# %%
# Set up constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
DATASET = config.get("DATASET")
SEED = config.get("SEED")

if DATASET is None:
    print("DATASET not set; not partitioning forward passes.")
    sys.exit(0)
else:
    dataset_name: str = DATASET.split("/")[-1]
    print(f"Partitioning dataset: {dataset_name}")

# %%
# Reproducibility
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config to wandb.
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config)

# %%
# Model and accelerator
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
accelerator: Accelerator = Accelerator()
# Ranges are iterable while slices aren't.
layers_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Load token ids.
token_ids: list[list[int]] = load_input_token_ids(PROMPT_IDS_PATH)

# %%
# Main loop
for i in layers_range:
    print(i)
