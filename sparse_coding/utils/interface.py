"""Read from and write to the module-level interface."""


import csv
import gc
import os
import pickle
from pathlib import Path
from textwrap import dedent

import yaml
import numpy as np
import torch as t
from transformers import PreTrainedModel
from pygraphviz import AGraph


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


def slice_to_range(model: PreTrainedModel, input_slice: slice) -> range:
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


def load_yaml_constants(base_file):
    """Load config files."""

    current_dir = Path(base_file).parent
    hf_access_file: str = "config/hf_access.yaml"
    central_config_file: str = "config/central_config.yaml"

    if current_dir.name == "sparse_coding":
        hf_access_path = current_dir / hf_access_file
        central_config_path = current_dir / central_config_file

    elif current_dir.name in ("interp_tools", "rasp"):
        hf_access_path = current_dir.parent / hf_access_file
        central_config_path = current_dir.parent / central_config_file

    else:
        raise ValueError(
            dedent(
                f"""
                Trying to access config files from an unfamiliar working
                directory: {current_dir}
                """
            )
        )

    try:
        with open(hf_access_path, "r", encoding="utf-8") as f:
            access = yaml.safe_load(f)
    except FileNotFoundError:
        print("hf_access.yaml not found. Creating it now.")
        with open(hf_access_path, "w", encoding="utf-8") as w:
            w.write('HF_ACCESS_TOKEN: ""\n')
        access = {}
    except yaml.YAMLError as e:
        print(e)

    with open(central_config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    return access, config


def save_paths(base_file, save_append: str) -> str:
    """Route to save paths from the current working directory."""

    assert isinstance(
        save_append, str
    ), f"`save_append` must be a string: {save_append}."

    current_dir = Path(base_file).parent

    if current_dir.name == "sparse_coding":
        save_path = current_dir / "data" / save_append
        return str(save_path)

    if current_dir.name in ("interp_tools", "rasp"):
        save_path = current_dir.parent / "data" / save_append
        return str(save_path)

    raise ValueError(
        dedent(
            f"""
            Trying to route to save directory from an unfamiliar working
            directory:
            {current_dir}
            """
        )
    )


def load_layer_tensors(
    model_dir: str,
    layer_idx: int,
    encoder_file: str,
    biases_file: str,
    base_file: str,
) -> t.Tensor:
    """
    Return the autoencoder, bias tensors for a model layer.

    `base_file should be __file__ in the calling module.
    """

    encoder = t.load(
        save_paths(
            base_file,
            (
                sanitize_model_name(model_dir)
                + "/"
                + str(layer_idx)
                + "/"
                + encoder_file
            ),
        )
    )

    bias = t.load(
        save_paths(
            base_file,
            (
                sanitize_model_name(model_dir)
                + "/"
                + str(layer_idx)
                + "/"
                + biases_file
            ),
        )
    )

    return encoder, bias


def load_layer_feature_indices(
    model_dir: str,
    layer_idx: int,
    top_k_info_file: str,
    base_file: str,
) -> list[int]:
    """
    Return the meaningful feature indices for a model layer.

    `base_file` should be `__file__` in the calling module.
    """

    indices = []

    with open(
        save_paths(
            base_file,
            (
                sanitize_model_name(model_dir)
                + "/"
                + str(layer_idx)
                + "/"
                + top_k_info_file
            ),
        ),
        mode="r",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(file)
        # Skip the header.
        next(reader)

        for row in reader:
            indices.append(int(row[0]))

    return indices


def load_layer_feature_labels(
    model_dir: str,
    layer_idx: int,
    feature_idx: int,
    top_k_info_file: str,
    base_file: str,
) -> list[str]:
    """
    Return the top-k input token labels for an encoder layer feature.

    `base_file` should be `__file__` in the calling module.
    """

    with open(
        save_paths(
            base_file,
            (
                sanitize_model_name(model_dir)
                + "/"
                + str(layer_idx)
                + "/"
                + top_k_info_file
            ),
        ),
        mode="r",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(file)
        # Skip the header.
        next(reader)

        for row in reader:
            if int(row[0]) == feature_idx:
                return row[1]

        raise ValueError(
            dedent(
                f"""
                Feature index {feature_idx} not found in layer {layer_idx}
                autoencoder.
                """
            )
        )


def load_preexisting_graph(
    model_dir: str,
    graph_dot_file: str,
    base_file: str,
) -> AGraph | None:
    """
    Load a preexisting graph from disk.

    `base_file` should be `__file__` in the calling module.
    """

    try:
        graph_rel_path = save_paths(
            base_file,
            (
                sanitize_model_name(model_dir)
                + "/"
                + graph_dot_file
            ),
        )
        graph = AGraph(graph_rel_path)
        assert graph is not None, "Newly loaded graph is None."
        return graph
    except Exception:
        return None


def pad_activations(
    tensor: t.Tensor, max_length: int, accelerator
) -> t.Tensor:
    """Pad activation tensors to a given sequence length."""

    complement_length: int = max_length - tensor.size(1)
    padding: t.Tensor = t.zeros(
        tensor.size(0), complement_length, tensor.size(2)
    ).to(tensor.device)
    padding = accelerator.prepare(padding)
    try:
        return t.cat([tensor, padding], dim=1)
    except RuntimeError:
        gc.collect()
        return t.cat([tensor, padding], dim=1)
