# %%
"""
Mess with autoencoder activation dims during `webtext` and graph effects.

`directed_graph_webtext` identifies the sequence positions that most excited
each autoencoder dimension and plots ablation effects at those positions. It
relies on prior cached data from `pipe.py`.

You may need to have logged a HF access token, if applicable.
"""


import gc
import warnings
from collections import defaultdict
from textwrap import dedent

import numpy as np
import torch as t
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm

from sparse_coding.interp_tools.utils.computations import calc_act_diffs
from sparse_coding.interp_tools.utils.graphs import graph_and_log
from sparse_coding.interp_tools.utils.hooks import (
    hooks_manager,
    prepare_autoencoder_and_indices,
    prepare_dim_indices,
)
from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    slice_to_range,
)
from sparse_coding.utils.tasks import recursive_defaultdict


# %%
# Load constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
GRAPH_FILE = config.get("GRAPH_FILE")
GRAPH_DOT_FILE = config.get("GRAPH_DOT_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
COEFFICIENT = config.get("COEFFICIENT", 0.0)
INIT_THINNING_FACTOR = config.get("INIT_THINNING_FACTOR", None)
BRANCHING_FACTOR = config.get("BRANCHING_FACTOR")
DIMS_PINNED: dict[int, int] = config.get("DIMS_PINNED", None)
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED", 0)

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
# Load model, etc.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
accelerator: Accelerator = Accelerator()
model = accelerator.prepare(model)
model.eval()

layer_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)
ablate_layer_range: range = layer_range[:-1]

# %%
# Load the complementary `openwebtext` dataset subset, relative to
# `collect_acts_webtext`.
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
# layer_autoencoders: dict[int, tuple[t.Tensor]]
# layer_dim_indices: dict[int, list[int]]
layer_autoencoders, layer_dim_indices = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)

# %%
# Collect base case data.
base_activations_all_positions = defaultdict(recursive_defaultdict)
for ablate_layer_idx in ablate_layer_range:
    # Base run, to determine top activating sequence positions. I'm
    # repurposing the hooks_lifecycle to cache _at_ the would-be ablated
    # layer, by using its interface in a hacky way.
    with hooks_manager(
        ablate_layer_idx - 1,
        None,
        [ablate_layer_idx],
        layer_dim_indices,
        model,
        layer_autoencoders,
        base_activations_all_positions,
        ablate_during_run=False,
    ):
        for sequence in eval_set:
            _ = t.manual_seed(SEED)
            inputs = tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_INTERPED_LEN,
            ).to(model.device)

            try:
                model(**inputs)
            except RuntimeError:
                # Manually clear memory and retry.
                gc.collect()
                model(**inputs)

# %%
# Find each dim's top activation value, and select all positions in the range
# [activation/2, activation], a la Cunningham et al. 2023.
# base_activation indices:
# [ablate_layer_index][None][base_cache_dim_index]
favorite_sequence_positions: dict[tuple[int, int, int], list[int]] = {}

for i in base_activations_all_positions:
    for j in base_activations_all_positions[i]:
        assert j is None, f"Middle index {j} should have been None."
        for k in base_activations_all_positions[i][j]:
            # The t.argmax here finds the top sequence position for each dict
            # index tuple. # favorite_sequence_position indices are now the
            # tuple (ablate_layer_idx, None, base_cache_dim_index).
            activations_tensor = base_activations_all_positions[i][j][k]

            fave_seq_pos_flat: int = (
                t.argmax(activations_tensor, dim=1).squeeze().item()
            )
            max_val = activations_tensor[:, fave_seq_pos_flat, :].unsqueeze(1)
            min_val = max_val / 2.0
            mask = (activations_tensor >= min_val) & (
                activations_tensor <= max_val
            )

            top_indices: t.Tensor = t.nonzero(mask)[:, 1]
            favorite_sequence_positions[i, j, k] = top_indices.tolist()

# %%
# Run ablations at top sequence positions.
ablated_activations = defaultdict(recursive_defaultdict)
base_activations_top_positions = defaultdict(recursive_defaultdict)
keepers: dict[tuple[int, int], int] = {}
logit_diffs = {}

for ablate_layer_idx in ablate_layer_range:
    # Thin the first layer indices or fix any indices, when requested.
    if ablate_layer_idx == ablate_layer_range[0] or (
        DIMS_PINNED is not None
        and DIMS_PINNED.get(ablate_layer_idx) is not None
    ):
        layer_dim_indices[ablate_layer_idx]: list[int] = prepare_dim_indices(
            INIT_THINNING_FACTOR,
            DIMS_PINNED,
            layer_dim_indices[ablate_layer_idx],
            ablate_layer_idx,
            layer_range,
            SEED,
        )

    for ablate_dim_idx in tqdm(
        layer_dim_indices[ablate_layer_idx], desc="Dim Ablations Progress"
    ):
        # Truncated means truncated to MAX_SEQ_INTERPED_LEN. This block does
        # the work of further truncating to the top activating position length.
        truncated_tok_seqs = []

        for fav_seq_pos in favorite_sequence_positions[
            ablate_layer_idx - 1, None, ablate_dim_idx
        ]:
            for seq in eval_set:
                # The tokenizer also takes care of MAX_SEQ_INTERPED_LEN.
                tok_seq = tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_INTERPED_LEN,
                )
                # fav_seq_pos is the index for a flattened eval_set.
                if tok_seq["input_ids"].size(-1) < fav_seq_pos:
                    fav_seq_pos = fav_seq_pos - tok_seq["input_ids"].size(-1)
                    continue
                if tok_seq["input_ids"].size(-1) >= fav_seq_pos:
                    tok_seq = tokenizer(
                        seq,
                        return_tensors="pt",
                        truncation=True,
                        max_length=fav_seq_pos + 1,
                    )
                    truncated_tok_seqs.append(tok_seq)
                    break
                raise ValueError("fav_seq_pos out of range.")

        assert len(truncated_tok_seqs) > 0, dedent(
            f"No truncated sequences for {ablate_layer_idx}.{ablate_dim_idx}."
        )
        # This is a conventional use of hooks_lifecycle, but we're only passing
        # in as input to the model the top activating sequence, truncated. We
        # run one ablated and once not.
        with hooks_manager(
            ablate_layer_idx,
            ablate_dim_idx,
            layer_range,
            layer_dim_indices,
            model,
            layer_autoencoders,
            base_activations_top_positions,
            ablate_during_run=False,
        ):
            for seq in truncated_tok_seqs:
                top_input = seq.to(model.device)
                _ = t.manual_seed(SEED)

                try:
                    model(**top_input)
                except RuntimeError:
                    gc.collect()
                    model(**top_input)

        with hooks_manager(
            ablate_layer_idx,
            ablate_dim_idx,
            layer_range,
            layer_dim_indices,
            model,
            layer_autoencoders,
            ablated_activations,
            ablate_during_run=True,
            coefficient=COEFFICIENT,
        ):
            for seq in truncated_tok_seqs:
                top_input = seq.to(model.device)
                _ = t.manual_seed(SEED)

                try:
                    model(**top_input)
                except RuntimeError:
                    gc.collect()
                    model(**top_input)

    if BRANCHING_FACTOR is None:
        break

    # Keep just the most affected indices for the next layer's ablations.
    assert isinstance(BRANCHING_FACTOR, int)

    working_tensor = t.Tensor([[0.0]])
    top_layer_dims = []
    a = ablate_layer_idx

    for j in ablated_activations[a]:
        for k in ablated_activations[a][j]:
            working_tensor = t.abs(
                t.cat(
                    [
                        working_tensor,
                        ablated_activations[a][j][k][:, -1, :]
                        - base_activations_top_positions[a][j][k][:, -1, :],
                    ]
                )
            )

        _, ordered_dims = t.sort(
            working_tensor.squeeze(),
            descending=True,
        )
        ordered_dims = ordered_dims.tolist()
        top_dims = [
            idx for idx in ordered_dims if idx in layer_dim_indices[a + 1]
        ][:BRANCHING_FACTOR]

        assert len(top_dims) <= BRANCHING_FACTOR

        keepers[a, j] = top_dims
        top_layer_dims.extend(top_dims)

    layer_dim_indices[a + 1] = list(set(top_layer_dims))

# %%
# Compute ablated effects minus base effects.
act_diffs: dict[tuple[int, int, int], t.Tensor] = calc_act_diffs(
    ablated_activations,
    base_activations_top_positions,
)

# %%
# Graph effects.
graph_and_log(
    act_diffs,
    keepers,
    BRANCHING_FACTOR,
    MODEL_DIR,
    GRAPH_FILE,
    GRAPH_DOT_FILE,
    TOP_K_INFO_FILE,
    LOGIT_TOKENS,
    tokenizer,
    logit_diffs,
    __file__,
)

# %%
# Wrap up logging.
wandb.finish()
