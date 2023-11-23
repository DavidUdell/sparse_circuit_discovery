"""Validate the `transformer_lens` RASP model against the Haiku version."""


import numpy as np
import pytest
import torch as t

from sparse_coding.rasp.haiku_rasp_model import haiku_model
from sparse_coding.rasp.rasp_to_transformer_lens import transformer_lens_model
from sparse_coding.rasp.rasp_torch_tokenizer import tokenize


# Test determinism.
t.manual_seed(0)


@pytest.fixture
def inference_setup():
    """Set up for all test inference."""

    prompt = ["BOS", "x", "y", "z", "x"]

    haiku_output = haiku_model.apply(prompt)
    lens_output, lens_activations = transformer_lens_model.run_with_cache(  # pylint: disable=unpacking-non-sequence
        tokenize(prompt)
    )

    return haiku_output, lens_output, lens_activations, prompt


def test_rasp_model_activations(inference_setup):  # pylint: disable=redefined-outer-name
    """Compare Haiku and `transformer_lens` RASP model activations."""

    haiku_output, _, lens_activations, __ = inference_setup

    # Compare activations.
    for layer_idx in range(transformer_lens_model.cfg.n_layers):

        # Attention outs.
        assert np.allclose(
            np.array(haiku_output.layer_outputs[2 * layer_idx]),
            lens_activations["attn_out", layer_idx].detach().numpy(),
        )

        # MLP outs.
        assert np.allclose(
            np.array(haiku_output.layer_outputs[2 * layer_idx + 1]),
            lens_activations["mlp_out", layer_idx].detach().numpy(),
        )


def test_rasp_model_outputs(inference_setup):  # pylint: disable=redefined-outer-name
    """Compare Haiku and `transformer_lens` RASP model outputs."""

    haiku_output, lens_output, _, prompt = inference_setup
    vocab_out_length = transformer_lens_model.cfg.d_vocab_out

    lens_proportions = lens_output.detach().numpy().sum(axis=2)

    haiku_output_array = np.array(haiku_output.transformer_output)
    haiku_proportions = []
    for token_idx, __ in enumerate(prompt):
        haiku_proportions.append(haiku_output_array[:, token_idx, :vocab_out_length])
    haiku_proportions = np.array(haiku_proportions).sum(axis=2).T

    # Compare outputs.
    assert np.allclose(
        haiku_proportions,
        lens_proportions,
        atol=0.0001,
    ), print(haiku_proportions, lens_proportions)
