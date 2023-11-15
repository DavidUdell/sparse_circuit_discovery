"""Convert a JAX rasp model into a torch module model."""


import einsum
import numpy as np
import torch as t
from tracr import compiler
from tracr.rasp import rasp
from tracr.compiler.lib import make_frac_prevs


class Attn(t.nn.Module):
    """A custom single-headed attention layer, loading from tensors."""

    def __init__(self, key: t.Tensor, query: t.Tensor, value: t.Tensor):
        """Initialize the layer."""
        super().__init__()
        self.key = key
        self.query = query
        self.value = value

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass."""


class RaspModel(t.nn.Module):
    """A torch module wrapping a rasp model."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        # Compile the Haiku version of the model.
        haiku_model = compiler.compiling.compile_rasp_to_model(
            make_frac_prevs(rasp.tokens == "x"),
            vocab={"w", "x", "y", "z"},
            max_seq_len=5,
            compiler_bos="BOS",
        )

        torch_tensors: dict = {}

        for layer in haiku_model.params:
            for matrix in haiku_model.params[layer]:
                tensor_name: str = f"{layer}_{matrix}"
                torch_tensors[tensor_name] = t.tensor(
                    np.array(haiku_model.params[layer][matrix])
                )
        # pos_embed_embeddings
        # token_embed_embeddings
        # transformer/layer_0/attn/key_b
        # transformer/layer_0/attn/key_w
        # transformer/layer_0/attn/linear_b
        # transformer/layer_0/attn/linear_w
        # transformer/layer_0/attn/query_b
        # transformer/layer_0/attn/query_w
        # transformer/layer_0/attn/value_b
        # transformer/layer_0/attn/value_w
        # transformer/layer_0/mlp/linear_1_b
        # transformer/layer_0/mlp/linear_1_w
        # transformer/layer_0/mlp/linear_2_b
        # transformer/layer_0/mlp/linear_2_w
        # transformer/layer_1/attn/key_b
        # transformer/layer_1/attn/key_w
        # transformer/layer_1/attn/linear_b
        # transformer/layer_1/attn/linear_w
        # transformer/layer_1/attn/query_b
        # transformer/layer_1/attn/query_w
        # transformer/layer_1/attn/value_b
        # transformer/layer_1/attn/value_w
        # transformer/layer_1/mlp/linear_1_b
        # transformer/layer_1/mlp/linear_1_w
        # transformer/layer_1/mlp/linear_2_b
        # transformer/layer_1/mlp/linear_2_w

        hidden_dim: int = haiku_model.model_config.mlp_hidden_size
        attention_dim: int = haiku_model.model_config.key_size

        self.pos_embed = t.nn.Embedding.from_pretrained(
            torch_tensors["pos_embed_embeddings"]
        )
        self.embed = t.nn.Embedding.from_pretrained(
            torch_tensors["token_embed_embeddings"]
        )
        self.attn_1 = Attn()
        self.mlp_1 = t.nn.Sequential(
            t.nn.Linear(hidden_dim, hidden_dim),
            t.nn.ReLU(),
            t.nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_2 = Attn()
        self.mlp_2 = t.nn.Sequential(
            t.nn.Linear(hidden_dim, hidden_dim),
            t.nn.ReLU(),
            t.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass."""
