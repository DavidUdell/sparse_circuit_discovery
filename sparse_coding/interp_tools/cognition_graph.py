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
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm
import wandb

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
PROMPT = config.get("PROMPT")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
TOP_K_INFO_FILE = config.get("TOP_K_INFO_FILE")
GRAPH_FILE = config.get("GRAPH_FILE")
GRAPH_DOT_FILE = config.get("GRAPH_DOT_FILE")
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
MAX_SEQ_INTERPED_LEN = config.get("MAX_SEQ_INTERPED_LEN")
SEQ_PER_DIM_CAP = config.get("SEQ_PER_DIM_CAP", 10)
# COEFFICIENT = config.get("COEFFICIENT", 0.0)
INIT_THINNING_FACTOR = config.get("INIT_THINNING_FACTOR", None)
BRANCHING_FACTOR = config.get("BRANCHING_FACTOR")
DIMS_PINNED: dict[int, list[int]] = config.get("DIMS_PINNED", None)
THRESHOLD = config.get("THRESHOLD", 0.0)
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED", 0)


if DIMS_PINNED is not None:
    for v in DIMS_PINNED.values():
        assert isinstance(v, list) and len(v) == 1, dedent(
            """
            In this script, DIMS_PINNED for ablations should be a dict of
            singleton index lists.
            """
        )

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
ablate_layer_range: range = layer_range[:-1]

# %%
# Fix the validation set.
eval_set: list[str] = [PROMPT]

print("Prompt is as follows:")
for i in eval_set:
    print(i)

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
# Run ablations at top sequence positions.
ablated_activations = defaultdict(recursive_defaultdict)
base_activations_top_positions = defaultdict(recursive_defaultdict)
keepers: dict[tuple[int, int], int] = {}
probability_diffs = {}

for ablate_layer_idx in ablate_layer_range:
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

    # Truncated means truncated to MAX_SEQ_INTERPED_LEN. This block does
    # the work of further truncating to the top activating position length.
    truncated_tok_seqs = [
        tokenizer(
            c,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_INTERPED_LEN,
        )
        for c in eval_set
    ]

    BASE_LOGITS = None
    with hooks_manager(
        ablate_layer_idx,
        None,
        layer_range,
        layer_dim_indices,
        model,
        layer_encoders,
        layer_decoders,
        base_activations_top_positions,
        ablate_during_run=False,
    ):
        for seq in truncated_tok_seqs:
            top_input = seq.to(model.device)
            _ = t.manual_seed(SEED)

            try:
                output = model(**top_input)
            except RuntimeError:
                gc.collect()
                output = model(**top_input)

            logit = output.logits[:, -1, :].cpu()
            if BASE_LOGITS is None:
                BASE_LOGITS = logit
            elif isinstance(BASE_LOGITS, t.Tensor):
                BASE_LOGITS = t.cat([BASE_LOGITS, logit], dim=0)

    for ablate_dim_idx in tqdm(
        layer_dim_indices[ablate_layer_idx], desc="Dim Ablations Progress"
    ):

        assert len(truncated_tok_seqs) > 0, dedent(
            f"No truncated sequences for {ablate_layer_idx}.{ablate_dim_idx}."
        )
        # This is a conventional use of hooks_lifecycle, but we're only passing
        # in as input to the model the top activating sequence, truncated. We
        # run one ablated and once not.
        ALTERED_LOGITS = None
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
            for seq in truncated_tok_seqs:
                top_input = seq.to(model.device)
                _ = t.manual_seed(SEED)

                try:
                    output = model(**top_input)
                except RuntimeError:
                    gc.collect()
                    output = model(**top_input)

                logit = output.logits[:, -1, :].cpu()
                if ALTERED_LOGITS is None:
                    ALTERED_LOGITS = logit
                elif isinstance(ALTERED_LOGITS, t.Tensor):
                    ALTERED_LOGITS = t.cat([ALTERED_LOGITS, logit], dim=0)

        log_prob_diff = -t.nn.functional.log_softmax(
            ALTERED_LOGITS,
            dim=-1,
        ) + t.nn.functional.log_softmax(
            BASE_LOGITS,
            dim=-1,
        )
        probability_diffs[ablate_layer_idx, ablate_dim_idx] = log_prob_diff

    if BRANCHING_FACTOR is None:
        break

    # Keep just the most affected indices for the next layer's ablations.
    assert isinstance(BRANCHING_FACTOR, int)

    top_layer_dims: set = set()
    a: int = ablate_layer_idx
    reference_set = set(layer_dim_indices[a + 1])

    # len(ablated_activations[a])) == NUM_ABLATION_DIMS
    for j in tqdm(ablated_activations[a], desc="Branchings Progress"):
        WORKING_TENSOR = None
        cache_indices = list(ablated_activations[a][j].keys())
        # len(ablated_activations[a][j]) == NUM_CACHE_DIMS
        for k in ablated_activations[a][j]:

            # ablated_activations[a][j][k].shape == (1, SEQ_LEN, 1). Take all
            # the cached activation diffs and concat their absolute values.
            if WORKING_TENSOR is None:
                WORKING_TENSOR = t.abs(
                    ablated_activations[a][j][k]
                    - base_activations_top_positions[a][None][k]
                ).mean(dim=1)
            else:
                WORKING_TENSOR = t.cat(
                    [
                        WORKING_TENSOR,
                        t.abs(
                            ablated_activations[a][j][k]
                            - base_activations_top_positions[a][None][k],
                        ).mean(dim=1),
                    ]
                )

        # ordered_meta_indices: t.LongTensor
        _, ordered_meta_indices = t.sort(
            WORKING_TENSOR.squeeze(),
            descending=True,
        )
        assert len(ordered_meta_indices) == len(cache_indices)
        top_dims = []
        for i in ordered_meta_indices:
            if cache_indices[i.item()] in reference_set:
                top_dims.append(cache_indices[i.item()])
                if len(top_dims) == BRANCHING_FACTOR:
                    break
            else:
                raise ValueError("OOD ablation effect registered.")
        keepers[a, j] = top_dims

        top_layer_dims.update(top_dims)
    # list
    layer_dim_indices[a + 1] = list(top_layer_dims)

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
    THRESHOLD,
    LOGIT_TOKENS,
    tokenizer,
    probability_diffs,
    __file__,
)

# %%
# Wrap up logging.
wandb.finish()
