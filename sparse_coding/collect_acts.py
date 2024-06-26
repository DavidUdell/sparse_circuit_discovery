# %%
"""Collect model activations during inference on `openwebtext`."""


import gc
import warnings

import numpy as np
import torch as t
import transformers
import wandb
from accelerate import Accelerator
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
    slice_to_range,
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
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
ACTS_DATA_FILE = config.get("ACTS_DATA_FILE")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
NUM_SEQUENCES_EVALED = config.get("NUM_SEQUENCES_EVALED", 1000)
MAX_SEQ_LEN = config.get("MAX_SEQ_LEN", 1000)
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
acts_layers_range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Dataset. Poor man's fancy indexing.
training_set: list[list[int]] = [
        PROMPT,
]

# %%
# Tokenization and inference. The taut constraint here is how much memory you
# put into `activations`.
activations: list[t.Tensor] = []
prompt_ids_tensors: list[t.Tensor] = []
for idx, batch in enumerate(training_set):
    try:
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(model.device)
        outputs = model(**inputs)
        activations.append(outputs.hidden_states[ACTS_LAYERS_SLICE])
        prompt_ids_tensors.append(inputs["input_ids"].squeeze().cpu())
    except RuntimeError:
        # Manually clear memory and try again.
        gc.collect()
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
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

# %%
# Wrap up logging.
wandb.finish()
