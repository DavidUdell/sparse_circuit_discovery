"""Unit tests for the `sparse_coding` module."""


from collections import defaultdict

import pytest
import torch as t
import transformers
from accelerate import Accelerator

from sparse_coding.utils.top_contexts import (
    context_activations,
    project_activations,
    top_k_contexts,
)


# Test determinism.
t.manual_seed(0)


@pytest.fixture
def mock_autoencoder():
    """Return a mock model, its tokenizer, and its accelerator."""

    class MockEncoder:
        """Mock an encoder model."""

        def __init__(self):
            """Initialize the mock encoder."""
            self.encoder_layer = t.nn.Linear(512, 1024)
            t.nn.Sequential(self.encoder_layer, t.nn.ReLU())

        def __call__(self, inputs):
            """Mock projection behavior."""
            return self.encoder_layer(inputs)

    mock_encoder = MockEncoder()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m"
    )
    accelerator = Accelerator()

    return mock_encoder, tokenizer, accelerator


@pytest.fixture
def mock_data():
    """Return mock input token ids by q and encoder activations by q."""

    # "Just say, oops."
    # "Just say, hello world!"
    input_token_ids_by_q: list[list[int]] = [
        [6300, 1333, 13, 258, 2695, 15],
        [6300, 1333, 13, 23120, 1533, 2],
    ]
    encoder_activations_by_q_block: list[t.Tensor] = [
        (t.ones(6, 1024)) * 7,
        (t.ones(6, 1024)) * 11,
    ]

    return input_token_ids_by_q, encoder_activations_by_q_block


def test_context_activations(  # pylint: disable=redefined-outer-name
    mock_autoencoder, mock_data
):
    """Test `context_activations`."""

    # Pytest fixture injections.
    mock_encoder, _, _ = mock_autoencoder
    question_token_ids, feature_activations = mock_data

    mock_effects: defaultdict[
        int, list[tuple[list[int], list[float]]]
    ] = context_activations(
        question_token_ids,
        feature_activations,
        mock_encoder,
    )

    assert isinstance(mock_effects, defaultdict)
    assert isinstance(mock_effects[0], list)
    assert isinstance(mock_effects[0][0], tuple)
    assert isinstance(mock_effects[0][0][0], list)
    assert isinstance(mock_effects[0][0][0][0], int)
    assert isinstance(mock_effects[0][0][1], list)
    assert isinstance(mock_effects[0][0][1][0], float)
    assert len(mock_effects) == 1024  # 1024 encoder dims
    assert len(mock_effects[0]) == 2  # context and activations
    assert len(mock_effects[0][0][0]) == 6  # 6 tokens
    assert len(mock_effects[0][0][1]) == 6


def test_project_activations(  # pylint: disable=redefined-outer-name
    mock_autoencoder,
):
    """Test `project_activations`."""

    acts_list = [t.randn(5, 512) for _ in range(2)]
    mock_encoder, _, accelerator = mock_autoencoder

    mock_projections = project_activations(
        acts_list, mock_encoder, accelerator
    )

    assert isinstance(mock_projections, list)
    assert isinstance(mock_projections[0], t.Tensor)
    assert mock_projections[0].shape == (5, 1024)


def test_top_k_contexts():
    """Test `top_k_contexts`."""

    mock_effects: defaultdict[
        int, list[tuple[list[int], list[float]]]
    ] = defaultdict(list)
    mock_effects[0].append(([0, 1, 2], [0.1, 0.2, 0.3]))
    mock_effects[0].append(([1, 1, 2], [0.4, 0.5, 0.6]))
    mock_effects[0].append(([2, 2, 2], [0.7, 0.8, 0.9]))
    mock_effects[0].append(([3, 5, 3], [0.10, 0.11, 0.12]))
    mock_effects[0].append(([5, 4, 3], [0.13, 0.14, 0.15]))
    mock_effects[0].append(([5, 5, 2], [0.16, 0.17, 0.18]))
    mock_effects[0].append(([4, 2, 1], [0.19, 0.20, 0.21]))

    mock_effects[1].append(([12, 2, 5], [0.1, 0.2, 0.3]))
    mock_effects[1].append(([4, 5, 23], [0.4, 0.5, 0.6]))
    mock_effects[1].append(([12, 12, 5], [0.7, 0.8, 0.9]))
    mock_effects[1].append(([2, 1, 1], [0.10, 0.11, 0.12]))
    mock_effects[1].append(([11, 25, 43], [0.13, 0.14, 0.15]))
    mock_effects[1].append(([1, 1, 55], [0.16, 0.17, 0.18]))
    mock_effects[1].append(([5, 4, 90], [0.19, 0.20, 0.21]))

    view: int = 2
    top_k: int = 3

    mock_top_k_contexts = top_k_contexts(mock_effects, view, top_k)

    assert isinstance(mock_top_k_contexts, defaultdict)
    assert isinstance(mock_top_k_contexts[0], list)
    assert isinstance(mock_top_k_contexts[0][0], tuple)
    assert isinstance(mock_top_k_contexts[0][0][0], list)
    assert isinstance(mock_top_k_contexts[0][0][0][0], int)
    assert isinstance(mock_top_k_contexts[0][0][1], list)
    assert isinstance(mock_top_k_contexts[0][0][1][0], float)

    assert len(mock_top_k_contexts) == 2
    assert len(mock_top_k_contexts[0]) == 3
    assert len(mock_top_k_contexts[1]) == 3
