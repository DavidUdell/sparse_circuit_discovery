"""Tests for the rasp model functionality."""


from contextlib import contextmanager
from textwrap import dedent

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

    baseline = model.haiku_model.apply(["BOS", "w", "x", "y", "z"]).decoded

    input_tokens = rasp_encode(model, ["BOS", "w", "x", "y", "z"])
    output_tokens = []

    for idx, input_token, ground_truth in zip(
        range(5), input_tokens, baseline
    ):
        input_token = t.tensor(input_token, dtype=t.int).unsqueeze(0)
        output = model(input_token)
        output_tokens.append(output.sum().item())

        assert isinstance(
            output_tokens[-1], float
        ), f"Model output {output_tokens[-1]} must be a float."
        if isinstance(ground_truth, float):
            assert t.isclose(
                t.tensor(output_tokens[-1]),
                t.tensor(ground_truth),
                atol=0.00001,
            ), (
                f"Model output (sequence index {idx}) {output_tokens[-1]} "
                f"should be {ground_truth}."
            )
    output_tokens = model.haiku_model.input_encoder.decode(output_tokens)
    print(output_tokens)


def test_rasp_model_internals():
    """Compare JAX and rasp_to_torch model internal activation tensors."""

    @contextmanager
    def all_layer_hooks():
        """Pull all the layer out activations from the torch model."""
        layer_outs = []

        def hook(
            module, input, output
        ):  # pylint: disable=unused-argument, redefined-builtin
            layer_outs.append(output)

        hook_handles = []
        for layer in model.modules():
            hook_handles.append(layer.register_forward_hook(hook))
        yield layer_outs

        for hook in hook_handles:
            hook.remove()

    model = RaspModel()
    model.eval()
    raw_tokens = ["BOS", "w", "x", "y", "z", "q", "q", "q", "q"]
    torch_token_ids = model.haiku_model.input_encoder.encode(raw_tokens)

    jax_model_activation_tensors: list = model.haiku_model.apply(
        raw_tokens
    ).layer_outputs

    assert isinstance(jax_model_activation_tensors, list)

    for layer_idx, _ in enumerate(jax_model_activation_tensors):
        jax_model_activation_tensors[layer_idx] = t.tensor(
            np.array(jax_model_activation_tensors[layer_idx])
        )

    torch_model_activation_tensors = [None] * 11

    for token_id in torch_token_ids:
        with all_layer_hooks() as layer_outs:
            token_id = t.tensor(token_id, dtype=t.int).unsqueeze(0)
            model(token_id)
            for idx, layer_out in enumerate(layer_outs):
                if torch_model_activation_tensors[idx] is None:
                    torch_model_activation_tensors[idx] = layer_out
                else:
                    torch_model_activation_tensors[idx] = t.cat(
                        (torch_model_activation_tensors[idx], layer_out), dim=0
                    )
    print(f"torch model sublayers: {len(torch_model_activation_tensors)}")
    print(f"JAX model sublayers: {len(jax_model_activation_tensors)}")

    for sublayer_idx, jax_activation_tensor, torch_activation_tensor in zip(
        range(11), jax_model_activation_tensors, torch_model_activation_tensors
    ):
        print(f"sublayer {sublayer_idx}:")
        print(f"torch: {torch_activation_tensor}")
        print(f"JAX: {jax_activation_tensor}")

    for sublayer_idx, jax_activation_tensor, torch_activation_tensor in zip(
        range(11), jax_model_activation_tensors, torch_model_activation_tensors
    ):
        assert t.allclose(
            torch_activation_tensor,
            jax_activation_tensor,
            atol=0.0001,
        ), dedent(
            f"""
            Sublayer {sublayer_idx} tensors {torch_activation_tensor} (torch)
            and {jax_activation_tensor} (JAX) differ.
            """
        )
