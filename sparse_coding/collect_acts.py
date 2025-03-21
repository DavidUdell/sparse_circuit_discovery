# %%
"""
Collect model activations during inference.

If a dataset is specified, activations are collected for the dataset.
Otherwise, activations are collected for the prompt specified.
"""


import os
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
import wandb

from sparse_coding.utils.interface import (
    cache_layer_tensor,
    load_yaml_constants,
    pad_activations,
    parse_slice,
    sanitize_model_name,
    save_paths,
    slice_to_range,
    validate_slice,
)
from sparse_coding.interp_tools.utils.hooks import attn_mlp_acts_manager


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers v4.31.0"

# %%
# Set up constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
WANDB_MODE = config.get("WANDB_MODE")
DATASET = config.get("DATASET")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
RESID_DATA_FILE = config.get("ACTS_DATA_FILE")
ATTN_DATA_FILE = config.get("ATTN_DATA_FILE")
MLP_DATA_FILE = config.get("MLP_DATA_FILE")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
NUM_SEQUENCES_EVALED = config.get("NUM_SEQUENCES_EVALED", 1000)
MAX_SEQ_LEN = config.get("MAX_SEQ_LEN", 1000)
SEED = config.get("SEED")

if WANDB_MODE:
    os.environ["WANDB_MODE"] = WANDB_MODE

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
    # token=HF_ACCESS_TOKEN,
    clean_up_tokenization_spaces=True,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        # token=HF_ACCESS_TOKEN,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
model = accelerator.prepare(model)
model.eval()

validate_slice(model, ACTS_LAYERS_SLICE)
acts_layers_range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Dataset xor prompt.
if DATASET is not None:
    dataset: list[list[int]] = load_dataset(DATASET, split="train")["text"]
    # Indexing this way separates out a test set.
    dataset_indices = np.random.choice(
        len(dataset), size=len(dataset), replace=False
    )
    training_indices = dataset_indices[:NUM_SEQUENCES_EVALED]
    training_set: list[list[int]] = [dataset[idx] for idx in training_indices]
else:
    training_set: list[list[int]] = [
        PROMPT,
    ]

# %%
# Tokenization and inference. The taut constraint here is how much memory you
# put into `resid_acts`, `attn_acts`, `mlp_acts`.
prompt_ids_tensors: list[t.Tensor] = []
resid_acts: list[tuple[t.Tensor]] = []
attn_acts: list[tuple[t.Tensor]] = []
mlp_acts: list[tuple[t.Tensor]] = []

for idx, batch in enumerate(training_set):
    inputs = tokenizer(
        batch, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(model.device)

    with attn_mlp_acts_manager(model, list(acts_layers_range)) as a:
        outputs = model(**inputs)

        resid_acts.append(outputs.hidden_states[ACTS_LAYERS_SLICE])
        attn_acts.append(tuple(a[f"attn_{i}"] for i in acts_layers_range))
        # mlp acts save with a different convention.
        mlp_acts.append(
            tuple(a[f"mlp_{i}"].unsqueeze(0) for i in acts_layers_range)
        )

    prompt_ids_tensors.append(inputs["input_ids"].squeeze().cpu())

# Handle single layer resid case lacking outer tuples.
if isinstance(resid_acts, list) and isinstance(resid_acts[0], t.Tensor):
    # Tensors are of classic shape: (batch, seq, hidden)
    resid_acts: list[tuple[t.Tensor]] = [(tensor,) for tensor in resid_acts]

# %%
# Save prompt ids.
prompt_ids_lists = []
for tensor in prompt_ids_tensors:
    prompt_ids_lists.append([tensor.tolist()])

# array of (x, 1) Each element along x is a list of ints, of seq len.
prompt_ids_array: np.ndarray = np.array(prompt_ids_lists, dtype=object)
np.save(PROMPT_IDS_PATH, prompt_ids_array, allow_pickle=True)

# %%
# Save sublayer activations.
sublayers_acts = [resid_acts, attn_acts, mlp_acts]
sublayer_paths = [RESID_DATA_FILE, ATTN_DATA_FILE, MLP_DATA_FILE]

for sublayer_acts, sublayer_path in zip(sublayers_acts, sublayer_paths):
    max_seq_length: int = max(
        tensor.size(1)
        for layers_tuple in sublayer_acts
        for tensor in layers_tuple
    )

    # sublayer_acts: list[tuple[t.Tensor...]]: [batch]
    # layers_tuple: tuple[t.Tensor...]: [num_layers]
    # tensor: t.Tensor: [1, seq, hidden]
    for abs_idx, layer_idx in enumerate(acts_layers_range):
        layer_activations: list[t.Tensor] = [
            pad_activations(
                accelerator.prepare(layers_tuple[abs_idx]),
                max_seq_length,
                accelerator,
            )
            for layers_tuple in sublayer_acts
        ]
        layer_activations: t.Tensor = t.cat(layer_activations, dim=0)

        cache_layer_tensor(
            layer_activations,
            layer_idx,
            sublayer_path,
            __file__,
            MODEL_DIR,
        )

# Sanity check the saved tensors.
old_resid: t.Tensor = t.Tensor([])
old_attn: t.Tensor = t.Tensor([])
old_mlp: t.Tensor = t.Tensor([])

for layer_idx in acts_layers_range:
    resid_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{RESID_DATA_FILE}",
    )
    resid_acts: t.Tensor = t.load(resid_path, weights_only=True)
    attn_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{ATTN_DATA_FILE}",
    )
    attn_acts: t.Tensor = t.load(attn_path, weights_only=True)
    mlp_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{MLP_DATA_FILE}",
    )
    mlp_acts: t.Tensor = t.load(mlp_path, weights_only=True)

    assert not t.equal(resid_acts, attn_acts), f"{layer_idx}"
    assert not t.equal(attn_acts, mlp_acts), f"{layer_idx}"
    assert not t.equal(mlp_acts, resid_acts), f"{layer_idx}"

    old_resid = old_resid.to(resid_acts.device)
    old_attn = old_attn.to(attn_acts.device)
    old_mlp = old_mlp.to(mlp_acts.device)

    assert not t.equal(resid_acts, old_resid), f"{layer_idx}, {layer_idx - 1}"
    assert not t.equal(attn_acts, old_attn), f"{layer_idx}, {layer_idx - 1}"
    assert not t.equal(mlp_acts, old_mlp), f"{layer_idx}, {layer_idx - 1}"

    old_resid = resid_acts
    old_attn = attn_acts
    old_mlp = mlp_acts

# %%
# Wrap up logging.
wandb.finish()
