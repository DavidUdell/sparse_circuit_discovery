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
DIMS_PLOTTED_DICT = config.get("DIMS_PLOTTED_DICT", None)
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

for ablate_layer_idx in ablate_layer_range:
    # Thin the first layer indices or fix any indices, when requested.
    if (
        ablate_layer_idx == ablate_layer_range[0]
        or DIMS_PLOTTED_DICT.get(ablate_layer_idx) is not None
    ):
        layer_dim_indices[ablate_layer_idx]: list[int] = prepare_dim_indices(
            INIT_THINNING_FACTOR,
            DIMS_PLOTTED_DICT,
            layer_dim_indices[ablate_layer_idx],
            ablate_layer_idx,
            layer_range,
            SEED,
        )

    for ablate_dim_idx in tqdm(
        layer_dim_indices[ablate_layer_idx], desc="Dim Ablations Progress"
    ):
        # Ablation run at top activating sequence positions. We use the -1
        # index from the initial top position collection.
        per_seq_positions: list[int] = favorite_sequence_positions[
            ablate_layer_idx - 1, None, ablate_dim_idx
        ]
        truncated_tok_seqs = []
        truncated_seqs_final_indices: list[int] = []
        for per_seq_position in per_seq_positions:
            # per_seq_position is an int idx for a flattened eval_set.
            for sequence_idx, seq in enumerate(eval_set):
                # The tokenizer also takes care of MAX_SEQ_INTERPED_LEN.
                tok_seq = tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_INTERPED_LEN,
                )
                if tok_seq["input_ids"].size(-1) < per_seq_position:
                    per_seq_position = per_seq_position - tok_seq[
                        "input_ids"
                    ].size(-1)
                    continue
                truncated_tok_seqs.append(tok_seq)
                truncated_seqs_final_indices.append(per_seq_position)
                break

        assert len(truncated_tok_seqs) > 0, dedent(
            f"No truncated sequences for {ablate_layer_idx}.{ablate_dim_idx}."
        )
        assert len(truncated_seqs_final_indices) > 0, dedent(
            "No truncated sequence final indices were found."
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
            for s in truncated_tok_seqs:
                top_input = s.to(model.device)
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
            for s in truncated_tok_seqs:
                top_input = s.to(model.device)
                _ = t.manual_seed(SEED)
                try:
                    model(**top_input)
                except RuntimeError:
                    gc.collect()
                    model(**top_input)

    # Keep just the most affected indices for the next layer's ablations.
    if BRANCHING_FACTOR is None:
        break
    assert isinstance(BRANCHING_FACTOR, int)

    working_dict = {}
    top_layer_dims = []
    a = ablate_layer_idx

    for j in ablated_activations[a]:
        for k in ablated_activations[a][j]:
            working_dict[a, j, k] = (
                ablated_activations[a][j][k]
                - base_activations_top_positions[a][j][k]
            )

            top_dims = t.topk(
                abs(working_dict[a, j, k]).squeeze(),
                BRANCHING_FACTOR,
            )
            top_layer_dims.extend(top_dims[1].tolist())

    top_layer_dims = list(set(top_layer_dims))
    print(
        dedent(
            f"""
            Number of dims independently found most affected in next layer:
            {len(top_layer_dims)}.
            """
        )
    )
    layer_dim_indices[a+1] = [
        x for x in top_layer_dims if x in layer_dim_indices[a+1]
    ]
    print(
        dedent(
            f"""
            Length of intersection of previously labeled and currently most
            affected dims lists: {len(layer_dim_indices[a+1])}.
            """
        )
    )

# %%
# Compute ablated effects minus base effects. Recursive defaultdict indices
# are: [ablation_layer_idx][ablated_dim_idx][downstream_dim]
act_diffs: dict[tuple[int, int, int], t.Tensor] = {}
for i in ablated_activations:
    for j in ablated_activations[i]:
        for k in ablated_activations[i][j]:
            ablate_vec = ablated_activations[i][j][k]
            base_vec = base_activations_top_positions[i][j][k]

            assert (
                ablate_vec.shape == base_vec.shape == (1, base_vec.size(1), 1)
            ), dedent(
                f"""
                Shape mismatch between ablated and base vectors for ablate
                layer {i}, ablate dim {j}, and cache dim {k}; ablate shape
                {ablate_vec.shape} and base shape {base_vec.shape}.
                """
            )

            # The truncated seqs were all flattened. Now we just want what
            # would be the last position of each sequence.
            act_diffs[i, j, k] = t.tensor([[0.0]])
            for x in truncated_seqs_final_indices:
                act_diffs[i, j, k] += (
                    ablate_vec[:, x - 1, :] - base_vec[:, x - 1, :]
                )

# Check that there was any effect.
OVERALL_EFFECTS = 0.0
for i, j, k in act_diffs:
    OVERALL_EFFECTS += abs(act_diffs[i, j, k].item())
assert OVERALL_EFFECTS != 0.0, "Ablate hook effects sum to exactly zero."

# %%
# Graph effects.
graph_and_log(
    act_diffs,
    layer_range,
    layer_dim_indices,
    BRANCHING_FACTOR,
    MODEL_DIR,
    GRAPH_FILE,
    GRAPH_DOT_FILE,
    TOP_K_INFO_FILE,
    OVERALL_EFFECTS,
    __file__,
)

# %%
# Wrap up logging.
wandb.finish()
