# %%
"""
Print the top affected tokens per dimension of a learned decoder.

Requires a HF access token to get `Llama-2`'s tokenizer.
"""


import warnings
import csv
from collections import defaultdict
from math import isnan
from typing import Union

import numpy as np
import prettytable
import torch as t
import transformers
import wandb
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sparse_coding.utils import top_k
from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_range,
    load_input_token_ids,
    sanitize_model_name,
    load_yaml_constants,
    save_paths,
)


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# Set up constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
ACTS_DATA_FILE = config.get("ACTS_DATA_FILE")
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
SEED = config.get("SEED")
tsfm_config = AutoConfig.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
HIDDEN_DIM = tsfm_config.hidden_size
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
PROJECTION_DIM = int(HIDDEN_DIM * PROJECTION_FACTOR)
TOP_K = config.get("TOP_K", 6)
# None means "round to int", in SIG_FIGS.
SIG_FIGS = config.get("SIG_FIGS", None)
# DIMS_IN_BATCH is tunable, to fit in GPU memory.
DIMS_IN_BATCH = config.get("DIMS_IN_BATCH", 200)

if config.get("N_DIMS_PRINTED_OVERRIDE") is not None:
    N_DIMS_PRINTED = config.get("N_DIMS_PRINTED_OVERRIDE")
else:
    N_DIMS_PRINTED = PROJECTION_DIM

assert (
    0 < DIMS_IN_BATCH <= PROJECTION_DIM
), "DIMS_IN_BATCH must be at least 1 and at most PROJECTION_DIM."

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config to wandb.
wandb.init(
    project="sparse_circuit_discovery",
    entity="davidudell",
    config=config,
)

# %%
# We need the original tokenizer.
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    token=HF_ACCESS_TOKEN,
)
accelerator: Accelerator = Accelerator()

# %%
# Input token ids are constant across layers.
unpacked_prompts_ids: list[list[int]] = load_input_token_ids(PROMPT_IDS_PATH)


# %%
# Define the encoder class, taking imported_weights and biases as
# initialization args.
class Encoder(t.nn.Module):
    """Reconstruct an encoder as a callable linear layer."""

    def __init__(self, layer_weights: t.Tensor, layer_biases: t.Tensor):
        """Initialize the encoder."""
        super().__init__()
        self.encoder_layer = t.nn.Linear(HIDDEN_DIM, PROJECTION_DIM)
        self.encoder_layer.weight.data = layer_weights
        self.encoder_layer.bias.data = layer_biases

        self.encoder = t.nn.Sequential(self.encoder_layer, t.nn.ReLU())

    def forward(self, inputs):
        """Project to the sparse latent space."""

        # Apparently unneeded patch for `accelerate` with small models:
        # inputs = inputs.to(self.encoder_layer.weight.device)
        return self.encoder(inputs)


# %%
# Tabluation functionality.
def round_floats(num: Union[float, int]) -> Union[float, int]:
    """Round floats to number decimal places."""
    if isnan(num):
        print(f"{num} is NaN.")
        return num
    return round(num, SIG_FIGS)


def populate_table(
    _table, top_k_tokes, model_dir, top_k_info_file, layer_index
) -> None:
    """Put the results in the table _and_ save to csv."""
    csv_rows: list[list] = [
        ["Dimension", "Top Tokens", "Top-Token Activations"]
    ]

    for feature_dim, tokens_list in list(top_k_tokes.items())[:N_DIMS_PRINTED]:
        # Replace the tokenizer's special space char with a space literal.
        top_tokens = [str(t).replace("Ġ", " ") for t, _ in tokens_list[:TOP_K]]
        top_values = [round_floats(v) for _, v in tokens_list[:TOP_K]]

        # Skip the dimension if its activations are all zeroed out.
        if top_values[0] == 0:
            continue

        keeper_tokens = []
        keeper_values = []

        # Omit tokens _within a dimension_ with no activation.
        for top_t, top_v in zip(top_tokens, top_values):
            if top_v != 0:
                keeper_tokens.append(top_t)
                keeper_values.append(top_v)

        # Cast survivors to string.
        keeper_values = [str(v) for v in keeper_values]

        processed_row = [
            f"{feature_dim}",
            ", ".join(keeper_tokens),
            ", ".join(keeper_values),
        ]
        _table.add_row(processed_row)
        csv_rows.append(processed_row)

    top_k_info_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(model_dir)}/{layer_index}/{top_k_info_file}",
    )
    with open(top_k_info_path, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)


# %%
# Loop over all the sliced model layers.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
    )
seq_layer_indices: range = slice_to_range(model, ACTS_LAYERS_SLICE)

for layer_idx in seq_layer_indices:
    ENCODER_PATH = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{ENCODER_FILE}",
    )
    BIASES_PATH = save_paths(
        __file__, f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{BIASES_FILE}"
    )
    imported_weights: t.Tensor = t.load(ENCODER_PATH)
    imported_biases: t.Tensor = t.load(BIASES_PATH)

    # Initialize a concrete encoder for this layer.
    model: Encoder = Encoder(imported_weights, imported_biases)
    model = accelerator.prepare(model)

    # Load and parallelize activations.
    LAYER_ACTS_PATH = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{ACTS_DATA_FILE}",
    )
    layer_acts_data: t.Tensor = accelerator.prepare(t.load(LAYER_ACTS_PATH))

    # Note that activations are stored as a list of question tensors from this
    # function on out. Functions may internally unpack that into individual
    # activations, but that's the general protocol between functions.
    unpadded_acts: list[t.Tensor] = top_k.unpad_activations(
        layer_acts_data, unpacked_prompts_ids
    )

    # If you want to _directly_ interpret the model's activations, assign
    # `feature_acts` directly to `unpadded_acts` and ensure constants are set
    # to the model's embedding dimensionality.
    feature_acts: list[t.Tensor] = top_k.project_activations(
        unpadded_acts, model, accelerator
    )

    table = prettytable.PrettyTable()
    table.field_names = [
        "Dimension",
        "Top Tokens",
        "Top-Token Activations",
    ]
    # Calculate per-input-token summed activation, for each feature dimension.
    effects: defaultdict[
        int, defaultdict[str, float]
    ] = top_k.per_input_token_effects(
        unpacked_prompts_ids,
        feature_acts,
        model,
        tokenizer,
        accelerator,
        DIMS_IN_BATCH,
    )

    # Select just the top-k effects.
    truncated_effects: defaultdict[
        int, list[tuple[str, float]]
    ] = top_k.select_top_k_tokens(effects, TOP_K)

    populate_table(
        table, truncated_effects, MODEL_DIR, TOP_K_INFO_FILE, layer_idx
    )
    print(table)
    wandb.log({f"layer_{layer_idx}": wandb.Table(data=table)})

    accelerator.free_memory()

# %%
# Finish logging.
wandb.finish()
