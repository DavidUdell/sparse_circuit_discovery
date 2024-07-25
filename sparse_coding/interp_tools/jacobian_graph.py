# %%
"""A constant-time causal graphing algorithm."""


import torch as t
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb

from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
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
THRESHOLD_EXP = config.get("THRESHOLD_EXP")
LOGIT_TOKENS = config.get("LOGIT_TOKENS", 10)
SEED = config.get("SEED")

# Ensures THRESHOLD_EXP will behave.
if THRESHOLD_EXP is None:
    THRESHOLD_EXP = 0.0
else:
    THRESHOLD_EXP = 2.0**THRESHOLD_EXP

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
# Forward pass with Jacobian hooks.
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
# `jacobians` will be overwritten by the context manager's value.
jacobians = {}

with jacobians_manager(
    layer_range[0],
    model,
    encoders_and_biases,
    decoders_and_biases,
) as jacobians_dict:
    _ = model(**inputs)
    jacobians = jacobians_dict

# %%
# Compute and print Jacobian.
jac_func, act = jacobians[layer_range[0]]

jac = jac_func(act)

print(jac)
