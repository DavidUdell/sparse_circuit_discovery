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
        """Initialize the layer object."""
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

        scores = t.matmul(Q, K.transpose(0, 1) / sqrt(Q.size(-1)))
        normalized_scores = t.nn.functional.softmax(scores, dim=-1)

        # VO circuit
        scored_value = t.matmul(normalized_scores, V)
        output = (
            t.einsum("ij,jk->ik", scored_value, self.out_weights)
            + self.out_bias
        )
        return output


class MLP(t.nn.Module):
    """A custom MLP layer, loading from tensors."""

    def __init__(
        self,
        weights_1: t.Tensor,
        bias_1: t.Tensor,
        weights_2: t.Tensor,
        bias_2: t.Tensor,
    ):
        """Initialize the layer object."""
        super().__init__()
        self.weights_1 = weights_1
        self.bias_1 = bias_1
        self.weights_2 = weights_2
        self.bias_2 = bias_2

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass."""
        x = t.einsum("ij,jk->ik", x, self.weights_1) + self.bias_1
        x = t.nn.functional.relu(x)
        x = t.einsum("ij,jk->ik", x, self.weights_2) + self.bias_2

        return x


class Residual(t.nn.Module):
    """A minimal residual stream class, for easy hooking."""

    def __init__(self):
        """Initialize the layer object."""
        super().__init__()

    def forward(self, x: t.Tensor, other: t.Tensor) -> t.Tensor:
        """Just add x and other."""
        output = x + other

        return output


class RaspModel(t.nn.Module):
    """A torch module wrapping a rasp model."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        # Compile the Haiku version of the model.
        haiku_model = compiler.compiling.compile_rasp_to_model(
            make_frac_prevs(rasp.tokens == "x"),
            vocab={"w", "x", "y", "z", "q", "r", "s", "t"},
            max_seq_len=10,
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

        self.pos_embed = t.nn.Embedding.from_pretrained(
            torch_tensors["pos_embed_embeddings"]
        )
        self.embed = t.nn.Embedding.from_pretrained(
            torch_tensors["token_embed_embeddings"]
        )

        self.layer_norm_1 = t.nn.LayerNorm(23, bias=False)
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
        self.residual_1 = Residual()

        self.layer_norm_2 = t.nn.LayerNorm(23, bias=False)
        self.mlp_1 = MLP(
            torch_tensors["transformer/layer_0/mlp/linear_1_w"],
            torch_tensors["transformer/layer_0/mlp/linear_1_b"],
            torch_tensors["transformer/layer_0/mlp/linear_2_w"],
            torch_tensors["transformer/layer_0/mlp/linear_2_b"],
        )
        self.residual_2 = Residual()

        self.layer_norm_3 = t.nn.LayerNorm(23, bias=False)
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
        self.residual_3 = Residual()

        self.layer_norm_4 = t.nn.LayerNorm(23, bias=False)
        self.mlp_2 = MLP(
            torch_tensors["transformer/layer_1/mlp/linear_1_w"],
            torch_tensors["transformer/layer_1/mlp/linear_1_b"],
            torch_tensors["transformer/layer_1/mlp/linear_2_w"],
            torch_tensors["transformer/layer_1/mlp/linear_2_b"],
        )
        self.residual_4 = Residual()

        self.layer_norm_5 = t.nn.LayerNorm(23)

    def forward(self, x) -> t.Tensor:
        """Forward pass."""
        # Embedding
        x = self.embed(x) + self.pos_embed(x)

        # Transformer block 1
        layernorm_out_1 = self.layer_norm_1(x)
        attn_out_1 = self.attn_1(layernorm_out_1)
        x = self.residual_1(x, attn_out_1)
        layernorm_out_2 = self.layer_norm_2(x)
        mlp_out_1 = self.mlp_1(layernorm_out_2)
        x = self.residual_2(x, mlp_out_1)

        # Transformer block 2
        layernorm_out_3 = self.layer_norm_3(x)
        attn_out_2 = self.attn_2(layernorm_out_3)
        x = self.residual_3(x, attn_out_2)
        layernorm_out_4 = self.layer_norm_4(x)
        mlp_out_2 = self.mlp_2(layernorm_out_4)
        x = self.residual_4(x, mlp_out_2)

        # Final layernorm
        x = self.layer_norm_5(x)
        return x
