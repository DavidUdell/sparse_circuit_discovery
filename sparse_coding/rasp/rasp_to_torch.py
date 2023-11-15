"""Convert a JAX rasp model into a torch module model."""


from math import sqrt

import numpy as np
import torch as t
from tracr import compiler
from tracr.rasp import rasp
from tracr.compiler.lib import make_frac_prevs


class Attn(t.nn.Module):
    """A custom single-headed attention layer, loading from tensors."""

    def __init__(
        self,
        key_weights: t.Tensor,
        key_bias: t.Tensor,
        query_weights: t.Tensor,
        query_bias: t.Tensor,
        value_weights: t.Tensor,
        value_bias: t.Tensor,
        out_weights: t.Tensor,
        out_bias: t.Tensor,
    ):
        """Initialize the layer."""
        super().__init__()
        self.key_weights = key_weights
        self.key_bias = key_bias
        self.query_weights = query_weights
        self.query_bias = query_bias
        self.value_weights = value_weights
        self.value_bias = value_bias
        self.out_weights = out_weights
        self.out_bias = out_bias

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass."""
        Q = t.einsum("ij,jk->ik", x, self.query_weights) + self.query_bias
        K = t.einsum("ij,jk->ik", x, self.key_weights) + self.key_bias
        V = t.einsum("ij,jk->ik", x, self.value_weights) + self.value_bias

        scores = t.matmul(Q, K.transpose(-2, -1) / sqrt(Q.size(-1)))
        attn = t.nn.functional.softmax(scores, dim=-1)

        # VO circuit
        context = t.matmul(attn, V)
        output = (
            t.einsum("ij,jk->ik", context, self.out_weights) + self.out_bias
        )
        return output


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
        self.haiku_model = haiku_model
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

        self.pos_embed = t.nn.Embedding.from_pretrained(
            torch_tensors["pos_embed_embeddings"]
        )
        self.embed = t.nn.Embedding.from_pretrained(
            torch_tensors["token_embed_embeddings"]
        )
        self.attn_1 = Attn(
            torch_tensors["transformer/layer_0/attn/key_w"],
            torch_tensors["transformer/layer_0/attn/key_b"],
            torch_tensors["transformer/layer_0/attn/query_w"],
            torch_tensors["transformer/layer_0/attn/query_b"],
            torch_tensors["transformer/layer_0/attn/value_w"],
            torch_tensors["transformer/layer_0/attn/value_b"],
            torch_tensors["transformer/layer_0/attn/linear_w"],
            torch_tensors["transformer/layer_0/attn/linear_b"],
        )
        self.mlp_1 = t.nn.Sequential(
            t.nn.Linear(hidden_dim, hidden_dim),
            t.nn.ReLU(),
            t.nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_1[0].weight = t.nn.Parameter(
            torch_tensors["transformer/layer_0/mlp/linear_1_w"]
        )
        self.mlp_1[0].bias = t.nn.Parameter(
            torch_tensors["transformer/layer_0/mlp/linear_1_b"]
        )
        self.mlp_1[2].weight = t.nn.Parameter(
            torch_tensors["transformer/layer_0/mlp/linear_2_w"]
        )
        self.mlp_1[2].bias = t.nn.Parameter(
            torch_tensors["transformer/layer_0/mlp/linear_2_b"]
        )
        self.attn_2 = Attn(
            torch_tensors["transformer/layer_1/attn/key_w"],
            torch_tensors["transformer/layer_1/attn/key_b"],
            torch_tensors["transformer/layer_1/attn/query_w"],
            torch_tensors["transformer/layer_1/attn/query_b"],
            torch_tensors["transformer/layer_1/attn/value_w"],
            torch_tensors["transformer/layer_1/attn/value_b"],
            torch_tensors["transformer/layer_1/attn/linear_w"],
            torch_tensors["transformer/layer_1/attn/linear_b"],
        )
        self.mlp_2 = t.nn.Sequential(
            t.nn.Linear(hidden_dim, hidden_dim),
            t.nn.ReLU(),
            t.nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_2[0].weight = t.nn.Parameter(
            torch_tensors["transformer/layer_1/mlp/linear_1_w"]
        )
        self.mlp_2[0].bias = t.nn.Parameter(
            torch_tensors["transformer/layer_1/mlp/linear_1_b"]
        )
        self.mlp_2[2].weight = t.nn.Parameter(
            torch_tensors["transformer/layer_1/mlp/linear_2_w"]
        )
        self.mlp_2[2].bias = t.nn.Parameter(
            torch_tensors["transformer/layer_1/mlp/linear_2_b"]
        )

    def forward(self, x: str) -> t.Tensor:
        """Forward pass."""
        x = self.embed(x) + self.pos_embed(x)

        x = self.attn_1(x)
        x = self.mlp_1(x)

        x = self.attn_2(x)
        x = self.mlp_2(x)

        return x
