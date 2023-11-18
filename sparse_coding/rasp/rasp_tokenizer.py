"""Tokenization functionality for the rasp_to_torch model."""


import torch as t

from sparse_coding.rasp.rasp_to_torch import RaspModel


def rasp_encode(model: RaspModel, tokens: list[str]):
    """Encodings from string tokens."""
    input_encodings = model.haiku_model.input_encoder.encode(tokens)
    input_encodings = t.tensor(input_encodings)
    return input_encodings
