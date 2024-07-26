# %%
"""A constant-time causal graphing algorithm."""


import torch as t
from accelerate import Accelerator
from pygraphviz import AGraph
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb

from sparse_coding.interp_tools.utils.graphs import label_highlighting
from sparse_coding.utils.interface import (
    load_preexisting_graph,
    load_yaml_constants,
    parse_slice,
    sanitize_model_name,
    save_paths,
    slice_to_range,
)
from sparse_coding.interp_tools.utils.hooks import (
    jacobians_manager,
    prepare_autoencoder_and_indices,
)


# %%
# Load constants.
_, config = load_yaml_constants(__file__)

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
JACOBIANS_FILE = config.get("JACOBIANS_FILE")
JACOBIANS_DOT_FILE = config.get("JACOBIANS_DOT_FILE")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")

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
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
accelerator: Accelerator = Accelerator()

model = accelerator.prepare(model)
layer_range = slice_to_range(model, ACTS_LAYERS_SLICE)
up_layer_idx = layer_range[0]

# %%
# Prepare all layer range autoencoders.
encoders_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)
decoders_and_biases, _ = prepare_autoencoder_and_indices(
    layer_range,
    MODEL_DIR,
    DECODER_FILE,
    DEC_BIASES_FILE,
    TOP_K_INFO_FILE,
    accelerator,
    __file__,
)

# %%
# Load preexisting graph, if available.
graph = load_preexisting_graph(MODEL_DIR, JACOBIANS_DOT_FILE, __file__)
if graph is None:
    graph = AGraph(directed=True)

save_graph_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{JACOBIANS_FILE}"
)
save_dot_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{JACOBIANS_DOT_FILE}"
)

total_effect: float = 0.0
graphed_effect: float = 0.0

# %%
# Forward pass with Jacobian hooks.
print("Prompt:")
print()
print(PROMPT)

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
jac_func_and_point = {}

with jacobians_manager(
    up_layer_idx,
    model,
    encoders_and_biases,
    decoders_and_biases,
) as func_and_point:
    _ = model(**inputs)

    jac_func_and_point = func_and_point

# %%
# Compute Jacobian.
jac_function, act = jac_func_and_point[up_layer_idx]
act = act[:, -1, :].unsqueeze(0)

jacobian = jac_function(act)
jacobian = jacobian.squeeze()
row_length: int = jacobian.shape[0]

# Weight raw Jacobian by the activations, to get an approximation for this
# forward pass.
act = act.squeeze(0)
jacobian = jacobian * act

# %%
# Reduce Jacobian to directed graph.
flat_jac = t.flatten(jacobian)
pos_values, pos_indices = t.topk(flat_jac, 10)
neg_values, neg_indices = t.topk(flat_jac, 10, largest=False)
# Color range scalars for later labeling.
color_max_scalar = pos_values.max().item()
color_min_scalar = neg_values.min().item()

# %%
# Populate graph.
indices = pos_indices.tolist() + neg_indices.tolist()
values = pos_values.tolist() + neg_values.tolist()

for i, effect in zip(indices, values):
    magnitude = abs(effect)
    total_effect += magnitude

    if 0.0 == effect:
        print("Item skipped.")
        continue

    graphed_effect += magnitude
    # Upper index is mod row_length; downstream index is floor row_length.
    up_dim_idx = i % row_length
    down_dim_idx = i // row_length

    graph.add_node(
        f"{up_layer_idx}.{up_dim_idx}",
        label=label_highlighting(
            up_layer_idx,
            up_dim_idx,
            MODEL_DIR,
            TOP_K_INFO_FILE,
            0,
            tokenizer,
            f"{up_layer_idx}.{up_dim_idx}",
            {},
            __file__,
        ),
        shape="box",
    )
    try:
        graph.add_node(
            f"{up_layer_idx + 1}.{down_dim_idx}",
            label=label_highlighting(
                up_layer_idx + 1,
                down_dim_idx,
                MODEL_DIR,
                TOP_K_INFO_FILE,
                0,
                tokenizer,
                f"{up_layer_idx + 1}.{down_dim_idx}",
                {},
                __file__,
            ),
            shape="box",
        )
    except ValueError:
        print(
            f"Node {up_layer_idx + 1}.{down_dim_idx} not recognized; skipped."
        )
        continue

    if effect > 0.0:
        red, blue = 0, 255
    elif effect < 0.0:
        red, blue = 255, 0
    else:
        raise ValueError("Should be unreachable.")

    alpha: int = int(
        255 * magnitude / max(abs(color_max_scalar), abs(color_min_scalar))
    )
    rgba_str: str = f"#{red:02X}00{blue:02X}{alpha:02X}"
    graph.add_edge(
        f"{up_layer_idx}.{up_dim_idx}",
        f"{up_layer_idx + 1}.{down_dim_idx}",
        color=rgba_str,
    )

# %%
# Graph cleanup.
edges = graph.edges()

assert len(edges) == len(set(edges))

unlinked_nodes: int = 0
for node in graph.nodes():
    if len(graph.edges(node)) == 0:
        graph.remove_node(node)
        unlinked_nodes += 1
print(f"{unlinked_nodes} unlinked node(s) were dropped from graph.")

if total_effect == 0.0:
    raise ValueError("Total effect graphed was 0.0")

fraction_included = round(graphed_effect / total_effect, 2)
graph.add_node(f"Jacobian graphed out of overall: ~{fraction_included*100}%.")

# %%
# Render and save graph.
graph.write(save_dot_path)
graph.draw(save_graph_path, format="svg", prog="dot")

artifact = wandb.Artifact("jacobian_graph", type="causal_graph")
artifact.add_file(save_graph_path)
wandb.log_artifact(artifact)

print(f"Graph saved to {save_graph_path}")

# %%
# End wandb logging.
wandb.finish()
