"""Tests for the rasp model functionality."""


from contextlib import contextmanager

import numpy as np
import torch as t

from sparse_coding.rasp.rasp_to_torch import RaspModel
from sparse_coding.rasp.rasp_tokenizer import rasp_encode


# Test determinism.
t.manual_seed(0)


def test_rasp_model_internals():
    """Compare JAX and rasp_to_torch model internal activation tensors."""

    @contextmanager
    def all_layer_hooks(model: t.nn.Module):
        """Pull all the layer out activations from the torch model."""
        layer_outs = []

        def hook(
            module, input, output
        ):  # pylint: disable=unused-argument, redefined-builtin
            layer_outs.append(output)

        hook_handles = []
        for layer_out in (
            model.attn_1,
            model.residual_1,
            model.mlp_1,
            model.residual_2,
            model.attn_2,
            model.residual_3,
            model.mlp_2,
            model.residual_4,
        ):
            hook_handles.append(layer_out.register_forward_hook(hook))
        yield layer_outs

        for hook in hook_handles:
            hook.remove()

    layer_names: tuple = (
        "Attention 1",
        "Residual 1",
        "MLP 1",
        "Residual 2",
        "Attention 2",
        "Residual 3",
        "MLP 2",
        "Residual 4"
        )
    model = RaspModel()
    model.eval()
    raw_tokens = ["BOS", "x"]
    torch_token_ids = model.haiku_model.input_encoder.encode(raw_tokens)

    jax_sublayer_tensors: list = model.haiku_model.apply(
        raw_tokens
    ).layer_outputs
    jax_residual_tensors: list = model.haiku_model.apply(raw_tokens).residuals
    jax_activation_tensors = []
    for sublayer, residual in zip(jax_sublayer_tensors, jax_residual_tensors):
        jax_activation_tensors.append(t.tensor(np.array(sublayer)))
        jax_activation_tensors.append(t.tensor(np.array(residual)))

    torch_model_activation_tensors = [None] * len(layer_names)

    with all_layer_hooks(model) as layer_outs:
        token_ids = t.tensor(torch_token_ids, dtype=t.int).unsqueeze(0)
        model(token_ids)
        for idx, layer_out in enumerate(layer_outs):
            if torch_model_activation_tensors[idx] is None:
                torch_model_activation_tensors[idx] = layer_out.detach()
            else:
                torch_model_activation_tensors[idx] = t.cat(
                    torch_model_activation_tensors[idx],
                    layer_out.detach(),
                    dim=0
                )

    for sublayer, jax_activation_tensor, torch_activation_tensor in zip(
        layer_names,
        jax_activation_tensors,
        torch_model_activation_tensors
    ):
        print(f"{sublayer} activations for JAX/torch:")
        print(jax_activation_tensor)
        print(f"{torch_activation_tensor}\n")

    for sublayer, jax_activation_tensor, torch_activation_tensor in zip(
        layer_names,
        jax_activation_tensors,
        torch_model_activation_tensors
    ):
        assert t.allclose(
            torch_activation_tensor,
            jax_activation_tensor,
            atol=0.0001,
        ), (
            f"{sublayer} tensors for JAX/torch differ:\n"
            f"{torch_activation_tensor}\n"
            f"{jax_activation_tensor}\n"
        )


def test_rasp_model_outputs():
    """Compare JAX and rasp_to_torch model outputs."""

    model = RaspModel()
    model.eval()
    raw_tokens = ["BOS"]

    ground_truths = model.haiku_model.apply(raw_tokens).transformer_output
    input_ids = rasp_encode(model, raw_tokens)
    tensorized_input_ids = t.tensor(input_ids, dtype=t.int).unsqueeze(0)

    outputs = model(tensorized_input_ids)

    for idx, output, ground_truth in zip(
        range(len(raw_tokens)), outputs, ground_truths
        ):

        assert t.isclose(
            output.sum(-1)[idx].detach(),
            t.tensor(np.array(ground_truth)).sum(-1)[idx],
            atol=0.00001,
        ), (
            f"JAX/torch mismatch at sequence index {idx}:\n"
            f"{t.tensor(np.array(ground_truth)).sum(-1)[idx]}\n"
            f"{output.sum(-1)[idx].detach()}\n"
        )

        print(f"Outputs match at index {idx}!")
    print(outputs)
