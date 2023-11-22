"""Tests for the rasp model functionality."""


from contextlib import contextmanager

import numpy as np
import torch as t

from sparse_coding.rasp.rasp_to_torch import RaspModel
from sparse_coding.rasp.rasp_tokenizer import rasp_encode


# Test determinism.
t.manual_seed(0)


def test_rasp_model_outputs():
    """Compare JAX and rasp_to_torch model outputs."""

    model = RaspModel()
    model.eval()
    raw_tokens = ["BOS", "x"]

    ground_truths = model.haiku_model.apply(raw_tokens).decoded
    input_ids = rasp_encode(model, raw_tokens)
    tensorized_input_ids = t.tensor(input_ids, dtype=t.int).unsqueeze(0)

    outputs = model(tensorized_input_ids)

    for idx, output_token, ground_truth in zip(
        range(2), outputs, ground_truths
        ):
        if isinstance(ground_truth, float):
            assert t.isclose(
                t.tensor(output_token),
                t.tensor(ground_truth),
                atol=0.00001,
            ), (
                f"Model output (sequence index {idx}) {output_token} "
                f"should be {ground_truth}."
            )
        print(f"Outputs match at index {idx}!")
    print(outputs)


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
            model.mlp_1,
            model.attn_2,
            model.mlp_2,
        ):
            hook_handles.append(layer_out.register_forward_hook(hook))
        yield layer_outs

        for hook in hook_handles:
            hook.remove()

    model = RaspModel()
    model.eval()
    raw_tokens = ["BOS", "x"]
    torch_token_ids = model.haiku_model.input_encoder.encode(raw_tokens)

    jax_model_activation_tensors: list = model.haiku_model.apply(
        raw_tokens
    ).layer_outputs

    assert isinstance(jax_model_activation_tensors, list)

    for layer_idx, _ in enumerate(jax_model_activation_tensors):
        jax_model_activation_tensors[layer_idx] = t.tensor(
            np.array(jax_model_activation_tensors[layer_idx])
        )

    torch_model_activation_tensors = [None] * 4

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
        ("Attention 1", "MLP 1", "Attention 2", "MLP 2"),
        jax_model_activation_tensors,
        torch_model_activation_tensors
    ):
        print(f"{sublayer} activations for JAX, torch:")
        print(jax_activation_tensor)
        print(f"{torch_activation_tensor}\n")

    for sublayer, jax_activation_tensor, torch_activation_tensor in zip(
        ("Attention 1", "MLP 1", "Attention 2", "MLP 2"),
        jax_model_activation_tensors,
        torch_model_activation_tensors
    ):
        assert t.allclose(
            torch_activation_tensor,
            jax_activation_tensor,
            atol=0.0001,
        ), (
            f"{sublayer} tensors for JAX, torch differ:"
            f"{torch_activation_tensor}"
            f"{jax_activation_tensor}\n"
        )
