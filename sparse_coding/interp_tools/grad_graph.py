# %%
"""
Circuit discovery with gradient methods, with sparse autoencoders.

Implements the unsupervised circuit discovery algorithm in Baulab 2024.
"""


import csv
import os
from copy import deepcopy
from math import isnan

import requests
import torch as t
from accelerate import Accelerator
from pygraphviz import AGraph
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb

from sparse_coding.utils.interface import (
    load_preexisting_graph,
    load_yaml_constants,
    parse_slice,
    sanitize_model_name,
    save_paths,
    slice_to_range,
)
from sparse_coding.utils.tasks import recursive_defaultdict
from sparse_coding.interp_tools.utils.computations import (
    ExactlyZeroEffectError,
)
from sparse_coding.interp_tools.utils.graphs import (
    get_local_logits,
    label_highlighting,
    logit_diffs_str,
    neuronpedia_api,
    prune_graph,
)
from sparse_coding.interp_tools.utils.hooks import (
    grads_manager,
    measure_confounds,
    prepare_autoencoder_and_indices,
)


# %%
# Load constants.
access, config = load_yaml_constants(__file__)

NEURONPEDIA_KEY = access.get("NEURONPEDIA_KEY")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
WANDB_MODE = config.get("WANDB_MODE")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT = config.get("PROMPT")
# Patch over ACTS_LAYERS_SLICE, if env_var is specified.
ACTS_LAYERS_SLICE = parse_slice(
    os.environ.get("ACTS_LAYERS_SLICE", config.get("ACTS_LAYERS_SLICE"))
)
ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
ATTN_ENCODER_FILE = config.get("ATTN_ENCODER_FILE")
ATTN_ENC_BIASES_FILE = config.get("ATTN_ENC_BIASES_FILE")
ATTN_DECODER_FILE = config.get("ATTN_DECODER_FILE")
ATTN_DEC_BIASES_FILE = config.get("ATTN_DEC_BIASES_FILE")
MLP_ENCODER_FILE = config.get("MLP_ENCODER_FILE")
MLP_ENC_BIASES_FILE = config.get("MLP_ENC_BIASES_FILE")
MLP_DECODER_FILE = config.get("MLP_DECODER_FILE")
MLP_DEC_BIASES_FILE = config.get("MLP_DEC_BIASES_FILE")
RESID_TOKENS_FILE = config.get("TOP_K_INFO_FILE")
ATTN_TOKENS_FILE = config.get("ATTN_TOKEN_FILE")
MLP_TOKENS_FILE = config.get("MLP_TOKEN_FILE")
GRADS_FILE = config.get("GRADS_FILE")
GRADS_DOT_FILE = config.get("GRADS_DOT_FILE")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")
TOP_K = config.get("TOP_K")
VIEW = config.get("VIEW")
NUM_DOWN_NODES = config.get("NUM_DOWN_NODES")
NUM_UP_NODES = config.get("NUM_UP_NODES")

# export WANDB_MODE, if set in config
if WANDB_MODE:
    os.environ["WANDB_MODE"] = WANDB_MODE

# %%
# Neuronpedia API test call.
test_url: str = (
    "https://www.neuronpedia.org/api/feature/gpt2-small/0-res-jb/14057"
)
test_response = requests.get(
    test_url,
    headers={
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "sparse_circuit_discovery",
        "X-Api-Key": NEURONPEDIA_KEY,
    },
    timeout=300,
)
http_status: int = test_response.status_code

assert isinstance(http_status, int)

if http_status == 404:
    raise ValueError("Neuronpedia API test connection failed: 404")

print("Neuronpedia API test connection successful:", http_status)

# %%
# Reproducibility.
_ = t.manual_seed(SEED)

# %%
# Log to wandb.
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)


# %%
# Load and prepare the model.
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    output_hidden_states=True,
    return_dict_in_generate=True,
)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, clean_up_tokenization_spaces=True
)
accelerator: Accelerator = Accelerator()

model = accelerator.prepare(model)
layer_range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Prepare all layer range autoencoders.
# Residual autoencoders
res_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    RESID_TOKENS_FILE,
    accelerator,
    __file__,
)
res_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    RESID_TOKENS_FILE,
    accelerator,
    __file__,
)

# Attention autoencoders
attn_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ATTN_ENCODER_FILE,
    ATTN_ENC_BIASES_FILE,
    ATTN_TOKENS_FILE,
    accelerator,
    __file__,
)
attn_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ATTN_DECODER_FILE,
    ATTN_DEC_BIASES_FILE,
    ATTN_TOKENS_FILE,
    accelerator,
    __file__,
)

# MLP autoencoders
mlp_enc_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    MLP_ENCODER_FILE,
    MLP_ENC_BIASES_FILE,
    MLP_TOKENS_FILE,
    accelerator,
    __file__,
)
mlp_dec_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    MLP_DECODER_FILE,
    MLP_DEC_BIASES_FILE,
    MLP_TOKENS_FILE,
    accelerator,
    __file__,
)

# %%
# Load percentile thresholds, if available.
percentiles: dict = {}

for layer_idx in layer_range:
    for basename in [
        "resid_percentile.csv",
        "attn_percentile.csv",
        "mlp_percentile.csv",
    ]:
        percentile_path: str = save_paths(
            __file__,
            f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{basename}",
        )
        sublayer: str = basename.split("_", maxsplit=1)[0]
        printable: str = f"Layer {layer_idx} {sublayer} percentile threshold"

        try:
            with open(percentile_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    percentile: float = float(row[0])
                    percentiles[f"{sublayer}_{layer_idx}"] = percentile
                    print(f"{printable} found:", round(percentile, 2))

        except FileNotFoundError:
            percentiles[f"{sublayer}_{layer_idx}"] = None
            print(f"{printable} not found; using top-k")

# %%
# Load preexisting graph, if available.
graph = load_preexisting_graph(MODEL_DIR, GRADS_DOT_FILE, __file__)

if graph is None:
    print("Graph status: No preexisting graph found; starting new graph.")
    graph = AGraph(directed=True)
else:
    print("Graph status: Preexisting graph loaded.")

print()

save_graph_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_FILE}"
)
save_dot_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_DOT_FILE}"
)

# %%
# Define cross-entropy loss.
metric = t.nn.CrossEntropyLoss()

# %%
# Model passes.
print("Prompt:")
print()
print(PROMPT)
print()
print("Backward passes:")

tokens = tokenizer(PROMPT, return_tensors="pt").to(model.device)
inputs = tokens.copy()

inputs["input_ids"] = inputs["input_ids"][:, :-1]
inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
target = tokens["input_ids"][:, -1].squeeze()

acts_dict: dict = None
grads_dict: dict = None
old_grads_dict: dict = None
marginal_grads_dict: dict = recursive_defaultdict()

with grads_manager(
    model,
    layer_range,
    res_enc_and_biases,
    res_dec_and_biases,
    attn_enc_and_biases,
    attn_dec_and_biases,
    mlp_enc_and_biases,
    mlp_dec_and_biases,
) as acts_and_grads:

    # Forward pass installs all backward hooks.
    output = model(**inputs)
    logits = output.logits[:, -1, :]
    flat_logits = logits.squeeze()

    loss = metric(
        flat_logits,
        target,
    )
    loss.backward(retain_graph=True)

    acts_dict, grads_dict, ripcord = acts_and_grads

    # This block brings edge grads later on into correspondence with the Bau
    # Lab implementation. The backward hooks need to change for succeeding
    # backward passes; this removes the gradient replacement hooks used above
    # that are now unwanted.
    for h in ripcord:
        h.remove()

    old_grads_dict = deepcopy(grads_dict)

    for loc, grad in old_grads_dict.items():
        grad = grad.squeeze().unsqueeze(0)
        act = acts_dict[loc].squeeze().unsqueeze(0)

        weighted_prod = t.einsum("...sd,...sd->...sd", grad, act)
        # Standardize weighted_prod shape
        if weighted_prod.dim() == 2:
            # Single-token prompt edge case.
            weighted_prod = weighted_prod[-1, :].squeeze()
        else:
            weighted_prod = weighted_prod[:, -1, :].squeeze()

        ####  Thresholding down-nodes -> indices  ####
        percentile: None | float = percentiles.get(loc, None)
        if percentile is None:
            if "error_" not in loc:
                ab_top_values, ab_top_indices = t.topk(
                    weighted_prod.abs(), NUM_DOWN_NODES
                )
                indices: list = ab_top_indices[ab_top_values > 0.0].tolist()
            elif "error_" in loc:
                # Sum across the error tensors, since we don't care about the
                # edges into the neuron basis.
                weighted_prod = weighted_prod.sum().unsqueeze(0)
                indices: list = [0]
            else:
                raise ValueError("Module location not recognized.")
        else:
            # elif percentile is float
            if acts_dict[loc].dim() == 2:
                acts_tensor = acts_dict[loc][-1, :].squeeze()
            else:
                acts_tensor = acts_dict[loc][:, -1, :].squeeze()
            thresh_tensor = t.full_like(acts_tensor, percentile)
            gt_tensor = t.nn.functional.relu(acts_tensor - thresh_tensor)
            indices: list | int = t.nonzero(gt_tensor).squeeze().tolist()
            if isinstance(indices, int):
                indices: list = [indices]
            assert len(indices) > 0
        ####  End thresholding down-nodes  ####

        # # Effect/down node regression test printouts
        # print(loc, "effects:")
        # print(weighted_prod.to("cpu").detach())
        # print()
        # print(loc, "select nodes:")
        # print(indices)
        # print()

        for dim_idx in tqdm(indices, desc=loc):
            # Edge-level backward passes
            weighted_prod[dim_idx].backward(retain_graph=True)
            _, marginal_grads, _ = acts_and_grads

            old_marginal_grads = deepcopy(marginal_grads)

            down_layer_idx = int(loc.split("_")[-1])
            up_layer_idx = down_layer_idx - 1
            if up_layer_idx not in layer_range:
                continue

            # Perpare confound effects for subtraction, per down-node
            confounds_grads = measure_confounds(
                loc,
                marginal_grads,
                acts_dict,
                acts_and_grads,
            )
            if isinstance(confounds_grads, tuple):
                x_mlp_resid_grads, resid_attn_resid_grads, full_path_grads = (
                    confounds_grads
                )
                x_mlp_resid_grads = deepcopy(x_mlp_resid_grads)
                resid_attn_resid_grads = deepcopy(resid_attn_resid_grads)
                full_path_grads = deepcopy(full_path_grads)
            if confounds_grads is not None:
                resid_attn_mlp_grads: dict = confounds_grads
                resid_attn_mlp_grads = deepcopy(resid_attn_mlp_grads)

            # Down-node keys are:
            # resid_x, mlp_x, attn_x, resid_error_x, mlp_error_x, attn_error_x
            # Deduplicate and store edges
            if "attn_" in loc:
                # resid-to-attn
                marginal_grads_dict[f"resid_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...sd",
                    old_marginal_grads[f"resid_{up_layer_idx}"],
                    -acts_dict[f"resid_{up_layer_idx}"],
                )[
                    :, -1, :
                ].cpu()

                marginal_grads_dict[f"resid_error_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...s",
                    old_marginal_grads[f"resid_error_{up_layer_idx}"],
                    -acts_dict[f"resid_error_{up_layer_idx}"],
                )[
                    :, -1
                ].cpu()

            elif "mlp_" in loc:
                # attn-to-mlp
                marginal_grads_dict[f"attn_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...sd",
                    old_marginal_grads[f"attn_{down_layer_idx}"],
                    -acts_dict[f"attn_{down_layer_idx}"],
                )[
                    :, -1, :
                ].cpu()

                marginal_grads_dict[f"attn_error_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...s",
                    old_marginal_grads[f"attn_error_{down_layer_idx}"],
                    -acts_dict[f"attn_error_{down_layer_idx}"],
                )[
                    :, -1
                ].cpu()

                # resid-to-mlp - (resid-attn-mlp)
                marginal_grads_dict[f"resid_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...sd",
                    old_marginal_grads[f"resid_{up_layer_idx}"]
                    - resid_attn_mlp_grads[  # pylint: disable=possibly-used-before-assignment
                        f"resid_{up_layer_idx}"
                    ],
                    -acts_dict[f"resid_{up_layer_idx}"],
                )[
                    :, -1, :
                ].cpu()

                marginal_grads_dict[f"resid_error_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...s",
                    old_marginal_grads[f"resid_error_{up_layer_idx}"]
                    - resid_attn_mlp_grads[f"resid_error_{up_layer_idx}"],
                    -acts_dict[f"resid_error_{up_layer_idx}"],
                )[
                    :, -1
                ].cpu()

            elif "resid_" in loc:
                # attn-to-resid - (attn-mlp-resid)
                marginal_grads_dict[f"attn_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...sd",
                    old_marginal_grads[f"attn_{down_layer_idx}"]
                    - x_mlp_resid_grads[  # pylint: disable=possibly-used-before-assignment
                        f"attn_{down_layer_idx}"
                    ],
                    -acts_dict[f"attn_{down_layer_idx}"],
                )[
                    :, -1, :
                ].cpu()

                marginal_grads_dict[f"attn_error_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...s",
                    old_marginal_grads[f"attn_error_{down_layer_idx}"]
                    - x_mlp_resid_grads[f"attn_error_{down_layer_idx}"],
                    -acts_dict[f"attn_error_{down_layer_idx}"],
                )[
                    :, -1
                ].cpu()

                # mlp-to-resid
                marginal_grads_dict[f"mlp_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...sd",
                    old_marginal_grads[f"mlp_{down_layer_idx}"],
                    -acts_dict[f"mlp_{down_layer_idx}"],
                )[
                    :, -1, :
                ].cpu()

                marginal_grads_dict[f"mlp_error_{down_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...s",
                    old_marginal_grads[f"mlp_error_{down_layer_idx}"],
                    -acts_dict[f"mlp_error_{down_layer_idx}"],
                )[
                    :, -1
                ].cpu()

                # resid-to-resid - (resid-attn-resid) - (resid-mlp-resid)
                # + (resid-attn-mlp-resid)
                marginal_grads_dict[f"resid_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...sd",
                    old_marginal_grads[f"resid_{up_layer_idx}"]
                    - resid_attn_resid_grads[  # pylint: disable=possibly-used-before-assignment
                        f"resid_{up_layer_idx}"
                    ]
                    - x_mlp_resid_grads[f"resid_{up_layer_idx}"]
                    + full_path_grads[f"resid_{up_layer_idx}"],
                    -acts_dict[f"resid_{up_layer_idx}"],
                )[
                    :, -1, :
                ].cpu()

                marginal_grads_dict[f"resid_error_{up_layer_idx}_to_" + loc][
                    dim_idx
                ] = t.einsum(
                    "...sd,...sd->...s",
                    old_marginal_grads[f"resid_error_{up_layer_idx}"]
                    - resid_attn_resid_grads[f"resid_error_{up_layer_idx}"]
                    - x_mlp_resid_grads[f"resid_error_{up_layer_idx}"]
                    + full_path_grads[f"resid_error_{up_layer_idx}"],
                    -acts_dict[f"resid_error_{up_layer_idx}"],
                )[
                    :, -1
                ].cpu()

            else:
                raise ValueError("Module location not recognized.")

# Here to have the newlines look nice in both the interactive notebooks and the
# shell.
print()

# # Act/grad regression test printouts.
# print("Activations:")
# for k, v in acts_dict.items():
#     print(k, v)
# # Note that grads_list is used here; grads_dict is not accurate anymore by
# # this point.
# print("Gradients:")
# for k, v in old_grads_dict.items():
#     print(k, v)

# # Marginal-effects regression test printouts.
# print("Marginal effects:")
# for k, v in marginal_grads_dict.items():
#     print(k)
#     for i, j in v.items():
#         print(i, j.detach())
#     print()

# %%
# Populate graph.
print("Computing top-k/bottom-k graph edges:")

explained_dict: dict = {}
unexplained_dict: dict = {}

effect_explained: float = 0.0
effect_unexplained: float = 0.0

for edges_str, down_nodes in marginal_grads_dict.items():
    node_types: tuple[str] = edges_str.split("_to_")
    up_layer_split: tuple = node_types[0].split("_")
    down_layer_split: tuple = node_types[1].split("_")

    down_layer_str: str = "".join(down_layer_split)

    up_layer_idx: int = int(up_layer_split[-1])
    down_layer_idx: int = int(down_layer_split[-1])

    up_layer_module: str = "".join(up_layer_split[:-1])
    down_layer_module: str = "".join(down_layer_split[:-1])

    up_layer_subgraph: str = up_layer_split[0] + up_layer_split[-1]
    down_layer_subgraph: str = down_layer_split[0] + down_layer_split[-1]

    up_subgraph = graph.add_subgraph(name=up_layer_subgraph, rank="same")
    down_subgraph = graph.add_subgraph(name=down_layer_subgraph, rank="same")

    sublayer_explained: float = 0.0

    for down_dim, up_values in tqdm(down_nodes.items(), desc=edges_str):
        up_values = up_values.squeeze()
        if up_values.dim() == 2:
            up_values = up_values[-1, :]
        elif up_values.dim() == 0:
            up_values = up_values.unsqueeze(dim=0)
        assert up_values.dim() == 1

        ###  Thresholding up-nodes  ###
        if "error" in up_layer_module:
            up_values = up_values.sum().unsqueeze(0)
            top_values = up_values
            bottom_values = up_values
            indices: list[int] = [0]
        else:
            _, ab_top_indices = t.topk(up_values.abs(), NUM_UP_NODES)
            indices: list = ab_top_indices.tolist()

        color_max_scalar: float = t.max(up_values)
        color_min_scalar: float = t.min(up_values)

        for up_dim, effect in enumerate(up_values):
            if up_dim not in indices:
                continue

            effect = effect.item()
            if effect == 0.0:
                continue

            if "error" in up_layer_module:
                effect_unexplained += effect
                info: str | None = None
                shape: str = "box"
                style: str = "dotted"
            elif "res" in up_layer_module:
                effect_explained += effect
                info: str = RESID_TOKENS_FILE
                shape: str = "box3d"
                style: str = "dashed"
                encoder = res_enc_and_biases
                decoder = res_dec_and_biases
            elif "attn" in up_layer_module:
                effect_explained += effect
                info: str = ATTN_TOKENS_FILE
                shape: str = "box3d"
                style: str = "solid"
                encoder = attn_enc_and_biases
                decoder = attn_dec_and_biases
            elif "mlp" in up_layer_module:
                effect_explained += effect
                info: str = MLP_TOKENS_FILE
                shape: str = "box3d"
                style: str = "solid"
                encoder = mlp_enc_and_biases
                decoder = mlp_dec_and_biases
            else:
                raise ValueError("Module location not recognized.")
            up_dim_name: str = f"{node_types[0]}.{up_dim}"

            if info is not None:
                # Autoencoder up nodes
                prob_diffs = get_local_logits(
                    inputs,
                    model,
                    logits,
                    up_layer_idx,
                    [up_dim],
                    encoder,
                    decoder,
                    up_layer_module,
                )
                try:
                    label = label_highlighting(
                        up_layer_idx,
                        up_dim,
                        MODEL_DIR,
                        info,
                        LOGIT_TOKENS,
                        tokenizer,
                        up_dim_name,
                        prob_diffs,
                        __file__,
                        neuronpedia=True,
                        sublayer_type=up_layer_module,
                        top_k=TOP_K,
                        view=VIEW,
                        neuronpedia_key=NEURONPEDIA_KEY,
                    )
                    graph.add_node(
                        up_dim_name,
                        label=label,
                        shape=shape,
                        style=style,
                    )
                except ValueError:
                    label: str = (
                        '<<table border="0" cellborder="0" cellspacing="0">'
                    )
                    label += '<tr><td><font point-size="16"><b>'
                    label += up_dim_name
                    label += "</b></font></td></tr>"
                    label += neuronpedia_api(
                        up_layer_idx,
                        up_dim,
                        NEURONPEDIA_KEY,
                        up_layer_module,
                        TOP_K,
                        VIEW,
                    )
                    label += logit_diffs_str(
                        up_layer_idx,
                        up_dim,
                        prob_diffs,
                        LOGIT_TOKENS,
                        tokenizer,
                    )
                    label += "</table>>"
                    graph.add_node(
                        up_dim_name, label=label, shape=shape, style=style
                    )
            else:
                # Error up nodes
                label: str = (
                    '<<table border="0" cellborder="0" cellspacing="0">'
                )
                label += '<tr><td><font point-size="16"><b>'
                label += up_dim_name
                label += "</b></font></td></tr>"
                label += "</table>>"
                graph.add_node(
                    up_dim_name, label=label, shape=shape, style=style
                )
            up_subgraph.add_node(up_dim_name)

            if "error" in down_layer_module:
                info: str | None = None
                shape: str = "box"
                style: str = "dotted"
            elif "res" in down_layer_module:
                info: str = RESID_TOKENS_FILE
                shape: str = "box3d"
                style: str = "dashed"
                encoder = res_enc_and_biases
                decoder = res_dec_and_biases
            elif "attn" in down_layer_module:
                info: str = ATTN_TOKENS_FILE
                shape: str = "box3d"
                style: str = "solid"
                encoder = attn_enc_and_biases
                decoder = attn_dec_and_biases
            elif "mlp" in down_layer_module:
                info: str = MLP_TOKENS_FILE
                shape: str = "box3d"
                style: str = "solid"
                encoder = mlp_enc_and_biases
                decoder = mlp_dec_and_biases
            else:
                raise ValueError("Module location not recognized.")
            down_dim_name: str = f"{node_types[1]}.{down_dim}"

            if info is not None:
                # Autoencoder down nodes
                prob_diffs = get_local_logits(
                    inputs,
                    model,
                    logits,
                    down_layer_idx,
                    [down_dim],
                    encoder,
                    decoder,
                    down_layer_module,
                )
                try:
                    label = label_highlighting(
                        down_layer_idx,
                        down_dim,
                        MODEL_DIR,
                        info,
                        LOGIT_TOKENS,
                        tokenizer,
                        down_dim_name,
                        prob_diffs,
                        __file__,
                        neuronpedia=True,
                        sublayer_type=down_layer_module,
                        top_k=TOP_K,
                        view=VIEW,
                        neuronpedia_key=NEURONPEDIA_KEY,
                    )
                    graph.add_node(
                        down_dim_name,
                        label=label,
                        shape=shape,
                        style=style,
                    )
                except ValueError:
                    label: str = (
                        '<<table border="0" cellborder="0" cellspacing="0">'
                    )
                    label += '<tr><td><font point-size="16"><b>'
                    label += down_dim_name
                    label += "</b></font></td></tr>"
                    label += neuronpedia_api(
                        down_layer_idx,
                        down_dim,
                        NEURONPEDIA_KEY,
                        down_layer_module,
                        TOP_K,
                        VIEW,
                    )
                    label += logit_diffs_str(
                        down_layer_idx,
                        down_dim,
                        prob_diffs,
                        LOGIT_TOKENS,
                        tokenizer,
                    )
                    label += "</table>>"
                    graph.add_node(
                        down_dim_name, label=label, shape=shape, style=style
                    )
            else:
                # Error down nodes
                label: str = (
                    '<<table border="0" cellborder="0" cellspacing="0">'
                )
                label += '<tr><td><font point-size="16"><b>'
                label += down_dim_name
                label += "</b></font></td></tr>"
                label += "</table>>"
                graph.add_node(
                    down_dim_name, label=label, shape=shape, style=style
                )
            down_subgraph.add_node(down_dim_name)

            # Edge coloration.
            if effect > 0.0:
                red, blue = 0, 255
            elif effect < 0.0:
                red, blue = 255, 0
            elif isnan(effect):
                raise ValueError(
                    f"Exploding/vanishing gradients?: {edges_str, effect}"
                )
            else:
                # Satisfies linter
                raise ValueError("Should be unreachable.")

            alpha: int = int(
                255
                * abs(effect)
                / max(abs(color_max_scalar), abs(color_min_scalar))
            )
            rgba: str = f"#{red:02X}00{blue:02X}{alpha:02X}"
            # Prioritize the residual stream during graph layout.
            weight: int = 1
            if "resid" in up_subgraph and "resid" in down_subgraph:
                weight: int = 5

            graph.add_edge(
                up_dim_name,
                down_dim_name,
                color=rgba,
                weight=weight,
            )

# Here to have the newlines look nice in both the interactive notebooks and the
# shell.
print()

# %%
# Graph annotation.
if effect_explained == 0.0:
    raise ExactlyZeroEffectError()

total_frac_explained = round(
    effect_explained / (effect_explained + effect_unexplained), 2
)

# Nuke singleton nodes.
for node in graph.nodes():
    if len(graph.edges(node)) == 0:
        graph.remove_node(node)
# Prune graph to source-to-sink subgraph.
graph = prune_graph(graph)

graph.add_node(
    f"Overall effect explained by autoencoders: ~{total_frac_explained*100}%"
)

# %%
# Render graph
graph.write(save_dot_path)
# Format (.svg, .png) is inferred from file extension.
graph.draw(save_graph_path, prog="dot")

print("Graph saved to:")
print(save_graph_path)

# Prevents an ugly exception ignored at cleanup time.
graph.close()

# %%
# Close wandb.
wandb.finish()
