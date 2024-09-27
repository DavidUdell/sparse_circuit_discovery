# %%
"""
Collect model activations during inference.

If a dataset is specified, activations are collected for the dataset.
Otherwise, activations are collected for the prompt specified.
"""


import warnings

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
    validate_slice,
    cache_layer_tensor,
    slice_to_range,
    load_yaml_constants,
    save_paths,
    pad_activations,
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
DATASET = config.get("DATASET")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
ACTS_DATA_FILE = config.get("ACTS_DATA_FILE")
ATTN_DATA_FILE = config.get("ATTN_DATA_FILE")
MLP_DATA_FILE = config.get("MLP_DATA_FILE")
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
    clean_up_tokenization_spaces=True,
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
if DATASET is not None:
    print(DATASET)

training_set: list[list[int]] = [
    PROMPT,
]

# %%
# Tokenization and inference. The taut constraint here is how much memory you
# put into `activations`.
resid_acts: list[t.Tensor] = []
attn_acts: list[t.Tensor] = []
mlp_acts: list[t.Tensor] = []
prompt_ids_tensors: list[t.Tensor] = []

for idx, batch in enumerate(training_set):
    inputs = tokenizer(
        batch, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(model.device)

    with attn_mlp_acts_manager(model, list(acts_layers_range)) as a:
        outputs = model(**inputs)

        resid_acts.append(outputs.hidden_states[ACTS_LAYERS_SLICE])

        for i in acts_layers_range:
            attn_acts.append(a[f"attn_{i}"])
            mlp_acts.append(a[f"mlp_{i}"])

    prompt_ids_tensors.append(inputs["input_ids"].squeeze().cpu())

# %%
# Save prompt ids.
prompt_ids_lists = []
for tensor in prompt_ids_tensors:
    prompt_ids_lists.append([tensor.tolist()])

# array of (x, 1) Each element along x is a list of ints, of seq len.
prompt_ids_array: np.ndarray = np.array(prompt_ids_lists, dtype=object)
np.save(PROMPT_IDS_PATH, prompt_ids_array, allow_pickle=True)

# %%
# Save activations.
# Single layer resid case lacks outer tuple; this solves that.
if isinstance(resid_acts, list) and isinstance(resid_acts[0], t.Tensor):
    # Tensors are of classic shape: (batch, seq, hidden)
    resid_acts: list[tuple[t.Tensor]] = [(tensor,) for tensor in resid_acts]

max_seq_length: int = max(
    tensor.size(1) for layers_tuple in resid_acts for tensor in layers_tuple
)

for abs_idx, layer_idx in enumerate(acts_layers_range):
    layer_activations: list[t.Tensor] = [
        pad_activations(
            accelerator.prepare(layers_tuple[abs_idx]),
            max_seq_length,
            accelerator,
        )
        for layers_tuple in resid_acts
    ]
    layer_activations: t.Tensor = t.cat(layer_activations, dim=0)

    cache_layer_tensor(
        layer_activations,
        layer_idx,
        ACTS_DATA_FILE,
        __file__,
        MODEL_DIR,
    )

# Now attn-out and mlp-out.
# acts: list
for layer_acts in [attn_acts, mlp_acts]:
    for idx, act in zip(acts_layers_range, layer_acts):
        # In-place squeeze then unsqueeze, to regularize shapes.
        act.squeeze_()
        act.unsqueeze_(0)
        # Single-token prompt edge case.
        if act.dim() == 2:
            act.unsqueeze_(0)

        assert (
            act.shape == layer_activations.shape
        ), "Sublayer_out shapes should all match."

        # Only for single prompts, for now.
        cache_layer_tensor(
            act,
            idx,
            ATTN_DATA_FILE if layer_acts is attn_acts else MLP_DATA_FILE,
            __file__,
            MODEL_DIR,
        )

# %%
# Wrap up logging.
wandb.finish()
