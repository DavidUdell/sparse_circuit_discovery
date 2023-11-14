# %%
"""Ablate autoencoder dimensions during inference and graph causal effects."""


from contextlib import contextmanager

import accelerate
import einops
import numpy as np
import torch as t
from tracr import compiler
from tracr.rasp import rasp
from tracr.compiler.lib import make_frac_prevs

from sparse_coding.utils.configure import load_yaml_constants
from sparse_coding.utils.caching import parse_slice


# %%
# Import constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
SEED = config.get("SEED")

# %%
# Reproducibility.
t.manual_seed(SEED)

# %%
# Ablations context manager and factory.


@contextmanager
def ablations_lifecycle(  # pylint: disable=redefined-outer-name
    torch_model: t.nn.Module, layer_index: int, neuron_idx: int
) -> None:
    """Define, register, then unregister hooks."""

    def ablations_hook(  # pylint: disable=unused-argument, redefined-builtin, redefined-outer-name
        module: t.nn.Module, input: tuple, output: t.Tensor
    ) -> None:
        """Zero out a particular neuron's activations."""
        output[:, neuron_idx] = 0.0

    def caching_hook(  # pylint: disable=unused-argument, redefined-builtin, redefined-outer-name
        module: t.nn.Module, input: tuple, output: t.Tensor
    ) -> None:
        """Cache downstream layer activations."""
        activations[(layer_index, neuron_idx, token)] = output.detach()

    # Register the hooks with `torch`.
    ablations_hook_handle = torch_model[layer_index].register_forward_hook(
        ablations_hook
    )
    caching_hook_handle = torch_model[layer_index + 1].register_forward_hook(
        caching_hook
    )

    try:
        # Yield control flow to caller function.
        yield
    finally:
        # Unregister the ablations hook.
        ablations_hook_handle.remove()
        # Unregister the caching hook.
        caching_hook_handle.remove()


# %%
# This implementation validates against just the rasp model. After validation,
# I will generalize to real-world autoencoded models.

assert MODEL_DIR == "rasp", "MODEL_DIR must be 'rasp`, for now."
assert ACTS_LAYERS_SLICE == slice(
    0, 2
), "ACTS_LAYERS_SLICE must be 0:2, for now."

# Compile the rasp model.
haiku_model = compiler.compiling.compile_rasp_to_model(
    make_frac_prevs(rasp.tokens == "x"),
    vocab={"w", "x", "y", "z"},
    max_seq_len=5,
    compiler_bos="BOS",
)

# Transplant the model into a proper `torch` module.
num_heads = haiku_model.model_config.num_heads
num_layers = haiku_model.model_config.num_layers
attention_dim = haiku_model.model_config.key_size
hidden_dim = haiku_model.model_config.mlp_hidden_size
vocab_dim = haiku_model.params["token_embed"]["embeddings"].shape[0]
resid_width = haiku_model.params["token_embed"]["embeddings"].shape[1]
# Length of vocab minus BOS and PAD.
vocab_dim_out = haiku_model.params["token_embed"]["embeddings"].shape[0] - 2

# Reshape weights and biases everywhere to fit num_heads.
state_dict = {}
state_dict["pos_embed.weight"] = haiku_model.params["pos_embed"]["embeddings"]
state_dict["embed.weight"] = haiku_model.params["token_embed"]["embeddings"]
state_dict["unembed.weight"] = np.eye(resid_width, vocab_dim_out)

for layer_idx in range(num_layers):
    state_dict[f"blocks.{layer_idx}.attn.out_proj_weight"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/key"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.out_proj_bias"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.in_proj_weight"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.in_proj_bias"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.W_V"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.b_V"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.W_O"] = einops.rearrange(
        haiku_model.params[f"transformer/layer_{layer_idx}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head=attention_dim,
        n_heads=num_heads,
    )
    state_dict[f"blocks.{layer_idx}.attn.b_O"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/attn/linear"
    ]["b"]

    state_dict[f"blocks.{layer_idx}.mlp.0.weight"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_1"
    ]["w"]
    state_dict[f"blocks.{layer_idx}.mlp.0.bias"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_1"
    ]["b"]
    state_dict[f"blocks.{layer_idx}.mlp.2.weight"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_2"
    ]["w"]
    state_dict[f"blocks.{layer_idx}.mlp.2.bias"] = haiku_model.params[
        f"transformer/layer_{layer_idx}/mlp/linear_2"
    ]["b"]

for key, value in state_dict.items():
    # Jax array, to numpy array, to torch tensor.
    state_dict[key] = t.tensor(np.array(value))


class RaspModel(t.nn.Module):
    """A `torch` module that wraps the `rasp` weights and biases."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()
        self.pos_embed = t.nn.Embedding.from_pretrained(
            state_dict["pos_embed.weight"]
        )
        self.embed = t.nn.Embedding.from_pretrained(state_dict["embed.weight"])
        self.unembed = t.nn.Linear(
            in_features=vocab_dim_out, out_features=resid_width
        )
        self.blocks = t.nn.ModuleList()
        for (
            layer_idx  # pylint: disable=unused-variable, redefined-outer-name
        ) in range(num_layers):
            self.blocks.append(
                t.nn.ModuleDict(
                    {
                        "attn": t.nn.MultiheadAttention(
                            embed_dim=attention_dim,
                            num_heads=num_heads,
                            dropout=0.0,
                        ),
                        "mlp": t.nn.Sequential(
                            t.nn.Linear(
                                in_features=hidden_dim,
                                out_features=hidden_dim,
                            ),
                            t.nn.ReLU(),
                            t.nn.Linear(
                                in_features=hidden_dim,
                                out_features=hidden_dim,
                            ),
                        ),
                    }
                )
            )
        self.out = t.nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass."""
        # x: (batch_size, seq_len)
        # x: (batch_size, seq_len, resid_width)
        x = self.unembed(x)
        # x: (batch_size, seq_len, resid_width)
        x = x + self.pos_embed(t.arange(x.shape[1], device=x.device))
        # x: (seq_len, batch_size, resid_width)
        x = x.permute(1, 0, 2)
        # x: (seq_len, batch_size, resid_width)
        x = self.embed(x)
        # x: (seq_len, batch_size, resid_width)
        x = x.permute(1, 0, 2)
        # x: (batch_size, seq_len, resid_width)
        x = x + self.pos_embed(t.arange(x.shape[1], device=x.device))
        # x: (batch_size, seq_len, resid_width)
        x = x.permute(1, 0, 2)
        # x: (seq_len, batch_size, resid_width)
        x = x + t.randn_like(x) * 0.02
        # x: (seq_len, batch_size, resid_width)
        x = x.permute(1, 0, 2)
        # x: (batch_size, seq_len, resid_width)
        for (  # pylint: disable=unused-variable, redefined-outer-name
            layer_index
        ) in range(num_layers):
            # x: (batch_size, seq_len, resid_width)
            x = self.blocks[layer_index]["attn"](
                query=x, key=x, value=x, need_weights=False
            )[0]
            # x: (batch_size, seq_len, resid_width)
            x = x + t.randn_like(x) * 0.02
            # x: (batch_size, seq_len, resid_width)
            x = self.blocks[layer_index]["mlp"](x)
            # x: (batch_size, seq_len, resid_width)
            x = x + t.randn_like(x) * 0.02
        # x: (batch_size, seq_len, resid_width)
        x = x.permute(1, 0, 2)
        # x: (seq_len, batch_size, resid_width)
        x = self.out(x)


model = RaspModel()
model.load_state_dict(state_dict)
model.eval()
accelerator = accelerate.Accelerator()
model = accelerator.prepare(model)

# Loop over every dim and ablate, recording effects.
activations = {}

for layer_index in ACTS_LAYERS_SLICE:
    for neuron_idx in range(attention_dim):
        with ablations_lifecycle(model, layer_index, neuron_idx):
            for token in ["w", "x", "y", "z"]:
                # Run inference on the ablated model.
                output = model(token)
                # Record the downstream activations.
                activations[(layer_index, neuron_idx, token)] = output.hiddens

# %%
# Graph the causal effects.
print(activations.keys())
