# %%
"""
Mess with autoencoder activations dims during `truthful_qa` and graph effects.

`directed_graph_mc` in particular tries a model agains the multiple-choice task
on `truthful_qa`, where the model is teed up to answer a m/c question with
widely believed but false choices. The base task is compared to the task in
which autoencoder activations dimensions are surgically scaled during
inference, at the crucial last sequence position, where the model is answering.
Results are plotted as a causal graph, using cached data from the scripts in
`pipe.py`. You may either try ablating all feature dimensions or choose a
subset by index. You'll need to set a HF access token if needed.
"""


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

from sparse_coding.interp_tools.utils.hooks import hooks_manager
from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_range,
    load_yaml_constants,
    save_paths,
    sanitize_model_name,
    load_layer_tensors,
    load_layer_feature_indices,
)
from sparse_coding.utils.tasks import (
    multiple_choice_task,
    recursive_defaultdict,
)
from sparse_coding.interp_tools.utils.graphs import graph_causal_effects


# %%
# Import constants.
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
NUM_QUESTIONS_INTERPED = config.get("NUM_QUESTIONS_INTERPED", 50)
NUM_SHOT = config.get("NUM_SHOT", 6)
COEFFICIENT = config.get("COEFFICIENT", 0.0)
THINNING_FACTOR = config.get("THINNING_FACTOR", None)
BRANCHING_FACTOR = config.get("BRANCHING_FACTOR")
DIMS_PLOTTED_LIST = config.get("DIMS_PLOTTED_LIST", None)
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
ablate_range: range = layer_range[:-1]

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

# %%
# Prepare all layer autoencoders and layer dim index lists up front.
layer_autoencoders: dict[int, tuple[t.Tensor]] = {}
layer_dim_indices: dict[int, list[int]] = {}
for layer_idx in layer_range:
    layer_encoder, layer_bias = load_layer_tensors(
        MODEL_DIR,
        layer_idx,
        ENCODER_FILE,
        BIASES_FILE,
        __file__,
    )
    layer_encoder, layer_bias = accelerator.prepare(
        layer_encoder, layer_bias
    )
    layer_autoencoders[layer_idx] = (layer_encoder, layer_bias)
    layer_dim_list = load_layer_feature_indices(
        MODEL_DIR,
        layer_idx,
        TOP_K_INFO_FILE,
        __file__,
    )
    layer_dim_indices[layer_idx] = layer_dim_list

for ablate_layer_meta_index, ablate_layer_idx in enumerate(ablate_range):
    # Ablation layer feature-dim indices.
    ablate_dim_indices = layer_dim_indices[ablate_layer_idx]

    # THINNING_FACTOR pruning of ablate_dim_indices.
    if THINNING_FACTOR is not None:
        np.random.seed(SEED)
        ablate_dim_indices_thinned = np.random.choice(
            ablate_dim_indices,
            size=int(len(ablate_dim_indices) * THINNING_FACTOR),
            replace=False,
        ).tolist()

    # DIMS_PLOTTED_LIST will immediately override THINNING_FACTOR, if it was
    # set.
    if DIMS_PLOTTED_LIST is not None:
        for i in DIMS_PLOTTED_LIST:
            assert i in ablate_dim_indices, dedent(
                f"""Index {i} not in layer {ablate_layer_idx} feature
                    indices."""
            )
        ablate_dim_indices = []
        ablate_dim_indices.extend(DIMS_PLOTTED_LIST)
    else:
        ablate_dim_indices = ablate_dim_indices_thinned

    for ablate_dim_idx in tqdm(
        ablate_dim_indices, desc="Feature ablations progress"
    ):
        np.random.seed(SEED)
        # Base run.
        with hooks_manager(
            ablate_layer_idx,
            ablate_dim_idx,
            layer_range,
            layer_dim_indices,
            model,
            layer_autoencoders,
            base_activations,
            ablate_during_run=False,
        ):
            multiple_choice_task(
                dataset,
                validation_indices,
                model,
                tokenizer,
                accelerator,
                NUM_SHOT,
                ACTS_LAYERS_SLICE,
                return_outputs=False,
            )

        np.random.seed(SEED)
        # Ablated run.
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
            multiple_choice_task(
                dataset,
                validation_indices,
                model,
                tokenizer,
                accelerator,
                NUM_SHOT,
                ACTS_LAYERS_SLICE,
                return_outputs=False,
            )

# %%
# Compute ablated effects minus base effects. Recursive defaultdict indices
# are: [ablation_layer_idx][ablated_dim_idx][downstream_dim]
act_diffs = {}
for i in ablate_range:
    for j in base_activations[i].keys():
        for k in base_activations[i][j].keys():
            act_diffs[i, j, k] = (
                ablated_activations[i][j][k].sum(axis=1).squeeze()
                - base_activations[i][j][k].sum(axis=1).squeeze()
            )

# Check that there was any overall effect.
OVERALL_EFFECTS = 0.0
for i, j, k in act_diffs:
    OVERALL_EFFECTS += abs(act_diffs[i, j, k]).sum().item()
assert (
    OVERALL_EFFECTS != 0.0
), "Ablate hook effects sum to exactly zero."

# All other effects should be t.Tensors, but wandb plays nicer with floats.
diffs_table = wandb.Table(columns=["Ablated Dim->Cached Dim", "Effect"])
for i, j, k in act_diffs:
    key: str = f"{i}.{j}->{i+1}.{k}"
    value: float = act_diffs[i, j, k].item()
    diffs_table.add_data(key, value)
wandb.log({"Effects": diffs_table})

plotted_diffs = {}
if BRANCHING_FACTOR is not None:
    # Keep only the top effects per ablation site i, j across all
    # downstream indices k.
    working_dict = {}

    for key, effect in act_diffs.items():
        site = key[:2]
        if site not in working_dict:
            working_dict[site] = []
        working_dict[site].append((key, effect))

    for site, items in working_dict.items():
        sorted_items = sorted(
            items,
            key=lambda x: abs(x[-1].item()),
            reverse=True,
        )
        for k, v in sorted_items[:BRANCHING_FACTOR]:
            plotted_diffs[k] = v

else:
    plotted_diffs = act_diffs

# %%
# Graph effects.
save_path: str = save_paths(
    __file__,
    f"{sanitize_model_name(MODEL_DIR)}/{GRAPH_FILE}",    
)
save_pickle_path: str = save_paths(
    __file__,
    f"{sanitize_model_name(MODEL_DIR)}/{GRAPH_DOT_FILE}",
)

graph = graph_causal_effects(
    plotted_diffs,
    MODEL_DIR,
    TOP_K_INFO_FILE,
    GRAPH_DOT_FILE,
    OVERALL_EFFECTS,
    __file__,
)

# Save the graph .svg.
graph.draw(
    save_path,
    format="svg",
    prog="dot",
)

# Read the .svg into a `wandb` artifact.
artifact = wandb.Artifact("feature_graph", type="directed_graph")
artifact.add_file(save_path)
wandb.log_artifact(artifact)

# Save the AGraph object as a DOT file.
graph.write(save_pickle_path)

# %%
# Wrap up logging.
wandb.finish()
