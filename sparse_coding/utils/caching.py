"""Help cache data and layer tensors from many model layers."""


import os
from textwrap import dedent

import numpy as np
import torch as t
from transformers import PreTrainedModel

from sparse_coding.utils.configure import save_paths


def parse_slice(slice_string: str) -> slice:
    """Parse any valid slice string into its slice object."""

    start = stop = step = None
    slice_parts: list = slice_string.split(":")

    if not 0 <= len(slice_parts) <= 3:
        raise ValueError(
            dedent(
                f"""
                Slice string {slice_string} is not well-formed.
                """
            )
        )

    # Remember that Python evaluates empty strings as falsy.
    if slice_parts[0]:
        start = int(slice_parts[0])
    if len(slice_parts) > 1 and slice_parts[1]:
        stop = int(slice_parts[1])
    if len(slice_parts) == 3 and slice_parts[2]:
        step = int(slice_parts[2])

    layers_slice = slice(start, stop, step)

    if layers_slice.start is not None and layers_slice.stop is not None:
        assert start < stop, dedent(
            f"""
            Slice start ({layers_slice.start}) must be less than stop
            ({layers_slice.stop})
            """
        )

    return layers_slice


def validate_slice(model: PreTrainedModel, layers_slice: slice) -> None:
    """
    See whether the layers slice fits in the model's layers.
    
    Note that this is unnecessary when the slice is preprocessed with
    `slice_to_seq`; only use this when you need to validate the _slice_ object,
    not the corresponding range.
    """

    if layers_slice.stop is None:
        return

    # num_hidden_layers is not inclusive.
    last_layer: int = model.config.num_hidden_layers - 1

    # slice.stop is not inclusive.
    if last_layer < layers_slice.stop - 1:
        raise ValueError(
            dedent(
                f"""
                The layers slice {layers_slice} is out of bounds for the
                model's layer count.
                """
            )
        )

    return


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model names for saving and loading."""

    return model_name.replace("/", "_")


def cache_layer_tensor(
    layer_tensor: t.Tensor,
    layer_idx: int,
    save_append: str,
    base_file: str,
    model_name: str,
) -> None:
    """
    Cache per layer tensors in appropriate subdirectories.

    Base file is `__file__` in the calling module. Save append should be _just_
    the file name and extension, not any additional path. Model name will be
    sanitized, so HF hub names are kosher.
    """

    assert isinstance(
        layer_idx, int
    ), f"Layer index {layer_idx} is not an int."
    # Python bools are an int subclass.
    assert not isinstance(
        layer_idx, bool
    ), f"Layer index {layer_idx} is a bool, not an int."

    save_dir_path: str = save_paths(base_file, "")
    safe_model_name = sanitize_model_name(model_name)

    # Subdirectory structure in the save directory is
    # data/models/layers/tensor.pt.
    save_subdir_path: str = save_dir_path + f"/{safe_model_name}/{layer_idx}"

    os.makedirs(save_subdir_path, exist_ok=True)
    t.save(layer_tensor, save_subdir_path + f"/{save_append}")


def slice_to_seq(model: PreTrainedModel, input_slice: slice) -> range:
    """Build a range corresponding to an input slice."""

    if input_slice.start is None:
        start = 0
    elif input_slice.start < 0:
        start: int = model.config.num_hidden_layers + input_slice.start
    else:
        start: int = input_slice.start

    if input_slice.stop is None:
        stop = model.config.num_hidden_layers
    elif input_slice.stop < 0:
        stop: int = model.config.num_hidden_layers + input_slice.stop
    else:
        stop: int = input_slice.stop

    step: int = 1 if input_slice.step is None else input_slice.step

    # Truncate final ranges to the model's size.
    output_range = range(
        max(start, 0),
        min(stop, model.config.num_hidden_layers),
        step,
    )

    return output_range


def load_input_token_ids(prompt_ids_path: str) -> list[list[int]]:
    """
    Load input ids.

    These are constant across layers, making this a simpler job.
    """
    prompts_ids: np.ndarray = np.load(prompt_ids_path, allow_pickle=True)
    prompts_ids_list = prompts_ids.tolist()
    unpacked_ids: list[list[int]] = [
        elem for question_list in prompts_ids_list for elem in question_list
    ]

    return unpacked_ids
