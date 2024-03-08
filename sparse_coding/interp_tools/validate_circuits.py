# %%
"""Validate circuits with simultaneous ablation studies."""


import warnings
from collections import defaultdict
from contextlib import ExitStack

import numpy as np
import torch as t
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
    prepare_autoencoder_and_indices,
)
from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    slice_to_range,
)


# %%
# Load constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
VALIDATION_DIMS_PINNED: dict[int, int] = config.get("VALIDATION_DIMS_PINNED")
LOGIT_TOKENS = config.get("LOGIT_TOKENS")
SEED = config.get("SEED")

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config to wandb.
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)

# %%
# Load and prepare the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    token=HF_ACCESS_TOKEN,
)
accelerator: Accelerator = Accelerator()
model = accelerator.prepare(model)
model.eval()

layer_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Load the `openwetext` validation set.
dataset: list[list[str]] = load_dataset(
    "Elriggs/openwebtext-100k",
    split="train",
)["text"]
dataset_indices: np.ndarray = np.random.choice(
    len(dataset),
    size=len(dataset),
    replace=False,
)
STARTING_META_IDX: int = len(dataset) - NUM_SEQUENCES_INTERPED
eval_indices: np.ndarray = dataset_indices[STARTING_META_IDX:]
eval_set: list[list[str]] = [dataset[i] for i in eval_indices]

# %%
# Prepare all layer autoencoders and layer dim index lists up front.
# layer_encoders: dict[int, tuple[t.Tensor]]
# layer_dim_indices: dict[int, list[int]]
layer_encoders, layer_dim_indices = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)
layer_decoders, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)

# %%
# Validate the pinned circuit indices.
for i in VALIDATION_DIMS_PINNED:
    assert i in layer_range
    assert VALIDATION_DIMS_PINNED[i] in layer_dim_indices[i]

# %%
# Validate the pinned circuit with ablations. Base case first.
BASE_LOGITS = None
for seq in eval_set:
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_INTERPED_LEN,
    ).to(model.device)
    outputs = model(**inputs)
    logit = outputs.logits[:, -1, :]
    if BASE_LOGITS is None:
        BASE_LOGITS = logit
    else:
        BASE_LOGITS = t.cat((BASE_LOGITS, logit), dim=0)

ALTERED_LOGITS = None
with ExitStack() as stack:
    for k, v in VALIDATION_DIMS_PINNED.items():
        stack.enter_context(
            hooks_manager(
                k,
                v,
                layer_range,
                {k+1: []},
                model,
                layer_encoders,
                layer_decoders,
                defaultdict(list),
            )
        )

    for seq in eval_set:
        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_INTERPED_LEN,
        ).to(model.device)
        outputs = model(**inputs)
        logit = outputs.logits[:, -1, :]
        if ALTERED_LOGITS is None:
            ALTERED_LOGITS = logit
        else:
            ALTERED_LOGITS = t.cat((ALTERED_LOGITS, logit), dim=0)

# %%
# Compute and display logit diffs.
prob_diff = (
    t.nn.functional.softmax(ALTERED_LOGITS,dim=-1)
    - t.nn.functional.softmax(BASE_LOGITS, dim=-1)
)
