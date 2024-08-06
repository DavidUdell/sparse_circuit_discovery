# %%
"""
Classic circuit discovery with gradient methods, with sparse autoencoders.

I want to get a feel for this approach with an implementation here, in this
interface and infrastructure.
"""


import torch as t
from accelerate import Accelerator
from pygraphviz import AGraph
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
from sparse_coding.interp_tools.utils.hooks import (
    grads_manager,
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
GRADS_FILE = config.get("GRADS_FILE")
GRADS_DOT_FILE = config.get("GRADS_DOT_FILE")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")

# Or some other means of thresholding approximated effects.
NUM_TOP_EFFECTS: int = 10

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
graph = load_preexisting_graph(MODEL_DIR, GRADS_DOT_FILE, __file__)
if graph is None:
    graph = AGraph(directed=True)

save_graph_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_FILE}"
)
save_dot_path: str = save_paths(
    __file__, f"{sanitize_model_name(MODEL_DIR)}/{GRADS_DOT_FILE}"
)

# %%
# Define cross-entropy loss.


# %%
# Model passes.
print("Prompt:")
print()
print(PROMPT)

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
acts_dict: dict = None
grads_dict: dict = None

# Forward pass installs all hooks.
with grads_manager(
    model,
    layer_range,
    encoders_and_biases,
    decoders_and_biases,
) as grad:

    output = model(**inputs)
    acts_dict = output.hidden_states

    grads_dict = grad
