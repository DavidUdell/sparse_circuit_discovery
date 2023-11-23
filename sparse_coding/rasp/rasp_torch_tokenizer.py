"""Tokenization for the `transformer_lens` RASP model."""


import torch as t

from sparse_coding.rasp.haiku_rasp_model import haiku_model


def tokenize(prompt: list[str]) -> t.Tensor:
    """Tokenize prompt inputs for a torch model."""

    token_ids = haiku_model.input_encoder.encode(prompt)
    token_ids = t.tensor(token_ids).unsqueeze(0)

    return token_ids
