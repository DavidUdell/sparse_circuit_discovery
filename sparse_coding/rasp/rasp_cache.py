# %%
"""Create a rasp model and cache its weights and biases in the interface."""


from textwrap import dedent

import einops
import numpy as np
import torch as t
from tracr import compiler
from tracr.rasp import rasp
from tracr.compiler.lib import make_frac_prevs

from sparse_coding.utils.configure import load_yaml_constants
from sparse_coding.utils.caching import parse_slice, cache_layer_tensor


# %%
# Load up constants.
_, config = load_yaml_constants(__file__)

MODEL_DIR = config.get("MODEL_DIR")
ENCODER_FILE = config.get("ENCODER_FILE")
BIASES_FILE = config.get("BIASES_FILE")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))

# %%
# Errors ensure yaml files are configured for the subsequent ACDC run.
if MODEL_DIR != "rasp":
    raise ValueError(
        dedent(
            f"""
            `rasp_cache.py` requires that MODEL_DIR be set to `rasp`, not
            {MODEL_DIR}.
            """
        )
    )

if ACTS_LAYERS_SLICE != slice(0, 2):
    raise ValueError(
        dedent(
            f"""
            `rasp_cache.py` requires that ACTS_LAYERS_SLICE be set to `slice(0,
            2)`, not {ACTS_LAYERS_SLICE}.
            """
        )
    )

# %%
# Compile the rasp model.
model = compiler.compiling.compile_rasp_to_model(
    make_frac_prevs(rasp.tokens == "x"),
    vocab={"w", "x", "y", "z"},
    max_seq_len=5,
    compiler_bos="BOS",
)

# %%
# Tensorize the weights and biases of the rasp model.
# I don't yet fully understand this ACDC repo tensorization code.
num_heads = model.model_config.num_heads
num_layers = model.model_config.num_layers
attention_dim = model.model_config.key_size
hidden_dim = model.model_config.mlp_hidden_size
vocab_dim = model.params["token_embed"]["embeddings"].shape[0]
resid_width = model.params["token_embed"]["embeddings"].shape[1]
# Length of vocab minus BOS and PAD.
vocab_dim_out = model.params["token_embed"]["embeddings"].shape[0] - 2

# Reshape weights and biases everywhere to fit num_heads.
state_dict = {}
state_dict["pos_embed.W_pos"] = model.params["pos_embed"]["embeddings"]
state_dict["embed.W_E"] = model.params["token_embed"]["embeddings"]
state_dict["unembed.W_U"] = np.eye(resid_width, vocab_dim_out)

for layer_idx in range(num_layers):
    state_dict[f"blocks.{layer_idx}.attn.W_K"] = einops.rearrange(
        model.params[f"transformer/layer_{layer_idx}/attn/key"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.b_K"] = einops.rearrange(
        model.params[f"transformer/layer_{layer_idx}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.W_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{layer_idx}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.b_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{layer_idx}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.W_V"] = einops.rearrange(
        model.params[f"transformer/layer_{layer_idx}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.b_V"] = einops.rearrange(
        model.params[f"transformer/layer_{layer_idx}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.W_O"] = einops.rearrange(
        model.params[f"transformer/layer_{layer_idx}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.b_O"] = model.params[
        f"transformer/layer_{layer_idx}/attn/linear"
    ]["b"]

    state_dict[f"blocks.{layer_idx}.mlp.W_in"] = model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_1"
    ]["w"]
    state_dict[f"blocks.{layer_idx}.mlp.b_in"] = model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_1"
    ]["b"]
    state_dict[f"blocks.{layer_idx}.mlp.W_out"] = model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_2"
    ]["w"]
    state_dict[f"blocks.{layer_idx}.mlp.b_out"] = model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_2"
    ]["b"]

# Convert weights to tensors
for key, value in state_dict.items():
    # Jax array, to numpy array, to torch tensor.
    state_dict[key] = t.tensor(np.array(value))

# %%
# Cache the rasp model weights and biases. I use MLP-in weights and biases.
for layer_idx in range(num_layers):
    cache_layer_tensor(
        state_dict[f"blocks.{layer_idx}.mlp.W_in"],
        layer_idx,
        ENCODER_FILE,
        __file__,
        "rasp",
    )
    cache_layer_tensor(
        state_dict[f"blocks.{layer_idx}.mlp.b_in"],
        layer_idx,
        BIASES_FILE,
        __file__,
        "rasp",
    )
