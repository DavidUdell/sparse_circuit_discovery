# %%
"""
Mess with autoencoder activation dims during `truthful_qa` and graph effects.

`cognition_graph_mc` in particular tries a model agains the multiple-choice
task on `truthful_qa`, where the model is teed up to answer a m/c question with
widely believed but false choices. The base task is compared to the task in
which autoencoder activations dimensions are surgically scaled during
inference, at the crucial last sequence position, where the model is answering.
Results are plotted as a causal graph, using cached data from the scripts in
`pipe.py`. You may either try ablating all feature dimensions or choose a
subset by index.

You may need to have logged a HF access token, if applicable.
"""


import os
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
from sparse_coding.utils.tasks import (
    multiple_choice_task,
    recursive_defaultdict,
)


# %%
# Load constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
WANDB_MODE = config.get("WANDB_MODE")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
GRAPH_FILE = config.get("GRAPH_FILE")
GRAPH_DOT_FILE = config.get("GRAPH_DOT_FILE")
NUM_QUESTIONS_INTERPED = config.get("NUM_QUESTIONS_INTERPED", 50)
NUM_SHOT = config.get("NUM_SHOT", 6)
# COEFFICIENT = config.get("COEFFICIENT", 0.0)
INIT_THINNING_FACTOR = config.get("INIT_THINNING_FACTOR", None)
BRANCHING_FACTOR = config.get("BRANCHING_FACTOR")
DIMS_PINNED: dict[int, list[int]] = config.get("DIMS_PINNED", None)
THRESHOLD_EXP = config.get("THRESHOLD_EXP")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")

if DIMS_PINNED is not None:
    for v in DIMS_PINNED.values():
        assert isinstance(v, list) and len(v) == 1, dedent(
            """
            In this script, DIMS_PINNED for ablations should be a dict of
            singleton index lists.
            """
        )

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
# Run a full-scale HF model using the repo's interface.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
accelerator = Accelerator()
model = accelerator.prepare(model)
model.eval()

layer_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)
ablate_layer_range: range = layer_range[:-1]

# %%
# Run on the validation dataset with and without ablations.
dataset: dict = load_dataset("truthful_qa", "multiple_choice")
dataset_indices: np.ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=len(dataset["validation"]["question"]),
    replace=False,
)
starting_index: int = len(dataset_indices) - NUM_QUESTIONS_INTERPED
validation_indices: list = dataset_indices[starting_index:].tolist()

base_activations = defaultdict(recursive_defaultdict)
ablated_activations = defaultdict(recursive_defaultdict)
keepers: dict[tuple[int, int], int] = {}
logit_diffs = {}

# %%
# Prepare all layer autoencoders and layer dim index lists up front.
# layer_autoencoders: dict[int, tuple[t.Tensor]]
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

for ablate_layer_meta_index, ablate_layer_idx in enumerate(ablate_layer_range):
    # Thin the first layer indices or fix any indices, when requested.
    if ablate_layer_idx == ablate_layer_range[0] or (
        DIMS_PINNED is not None
        and DIMS_PINNED.get(ablate_layer_idx) is not None
    ):
        # list[int]
        layer_dim_indices[ablate_layer_idx] = prepare_dim_indices(
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
        np.random.seed(SEED)
        # Base run.
        with hooks_manager(
            ablate_layer_idx,
            ablate_dim_idx,
            layer_range,
            layer_dim_indices,
            model,
            layer_encoders,
            layer_decoders,
            base_activations,
            ablate_during_run=False,
        ):
            base_logits = multiple_choice_task(
                dataset,
                validation_indices,
                model,
                tokenizer,
                accelerator,
                NUM_SHOT,
                return_logits=True,
            )

        np.random.seed(SEED)
        # Ablated run.
        with hooks_manager(
            ablate_layer_idx,
            ablate_dim_idx,
            layer_range,
            layer_dim_indices,
            model,
            layer_encoders,
            layer_decoders,
            ablated_activations,
            ablate_during_run=True,
        ):
            altered_logits = multiple_choice_task(
                dataset,
                validation_indices,
                model,
                tokenizer,
                accelerator,
                NUM_SHOT,
                return_logits=True,
            )

        logit_diff = altered_logits - base_logits
        logit_diffs[ablate_layer_idx, ablate_dim_idx] = logit_diff.cpu()

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
                        - base_activations[a][j][k][:, -1, :],
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
    base_activations,
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
    THRESHOLD_EXP,
    LOGIT_TOKENS,
    tokenizer,
    logit_diffs,
    __file__,
)

# %%
# Wrap up logging.
wandb.finish()
