"""Ablation and caching hooks."""


from collections import defaultdict
from contextlib import contextmanager
from textwrap import dedent

import numpy as np
import torch as t

from sparse_coding.utils.interface import (
    load_layer_tensors,
    load_layer_feature_indices,
)


def prepare_autoencoder_and_indices(
        layer_range: range,
        model_dir: str,
        encoder_file: str,
        biases_file: str,
        top_k_info_file: str,
        accelerator,
        base_file,
):
    """Prepare all layer autoencoders and layer dim index lists up front."""

    layer_autoencoders: dict[int, tuple[t.Tensor]] = {}
    layer_dim_indices: dict[int, list[int]] = {}

    for layer_idx in layer_range:
        layer_encoder, layer_bias = load_layer_tensors(
            model_dir,
            layer_idx,
            encoder_file,
            biases_file,
            base_file,
        )
        layer_encoder, layer_bias = accelerator.prepare(
            layer_encoder, layer_bias
        )
        layer_autoencoders[layer_idx] = (layer_encoder, layer_bias)
        layer_dim_list = load_layer_feature_indices(
            model_dir,
            layer_idx,
            top_k_info_file,
            base_file,
        )
        layer_dim_indices[layer_idx] = layer_dim_list

    return layer_autoencoders, layer_dim_indices


def prepare_dim_indices(
        thinning_factor: float | None,
        dims_plotted_list: list[int] | None,
        ablate_dim_indices: list[int],
        ablate_layer_idx: int,
        seed: int,
) -> list[int]:
    """
    Apply THINNING_FACTOR and/or DIMS_PLOTTED_LIST to ablate_dim_indices.

    `dims_plotted_list` will override `thinning_factor`, if set.
    """

    if dims_plotted_list is not None:
        for i in dims_plotted_list:
            assert i in ablate_dim_indices, dedent(
                f"Index {i} not in `ablate_dim_indices`."
            )

        return dims_plotted_list

    if thinning_factor is not None:
        np.random.seed(seed)
        ablate_dim_indices_thinned: list[int] = np.random.choice(
            ablate_dim_indices,
            size=int(len(ablate_dim_indices) * thinning_factor),
            replace=False,
        ).tolist()

        for i in ablate_dim_indices_thinned:
            assert i in ablate_dim_indices, dedent(
                f"""Index {i} not in layer {ablate_layer_idx} feature
                    indices."""
            )

        return ablate_dim_indices_thinned

    return ablate_dim_indices


def rasp_ablate_hook_fac(neuron_index: int):
    """Factory for rasp ablations hooks, working at a neuron idx."""

    # All `transformer_lens` hook functions must have this interface.
    def ablate_hook(  # pylint: disable=unused-argument
        acts_tensor: t.Tensor, hook
    ) -> t.Tensor:
        """Zero out a particular neuron's activations."""

        acts_tensor[:, :, neuron_index] = 0.0

        return acts_tensor

    return ablate_hook


@contextmanager
def hooks_manager(
    ablate_layer_idx: int,
    ablate_dim_idx: int,
    model_layer_range: range,
    cache_dim_indices: dict[int, list[int]],
    model,
    tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    activations_dict: defaultdict,
    ablate_during_run: bool = True,
    coefficient: float = 0.0,
):
    """
    Context manager for the full-scale ablations and caching.

    Ablates the specified feature at `layer_idx` and caches the downstream
    effects.
    """

    def ablate_hook_fac(dim_idx: int, encoder: t.Tensor, biases: t.Tensor):
        """Create hooks that zero a projected neuron and project it back."""

        def ablate_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """Project activation vectors; ablate them; project them back."""

            # Project activations through the encoder/bias.
            projected_acts_unrec = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    output[0],
                    encoder.to(model.device),
                    bias=biases.to(model.device),
                )
            )
            projected_acts = projected_acts_unrec.to(model.device)
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=False
            )
            # Ablate the activation at dim_idx. Modify here to scale in
            # different ways besides ablation.
            projected_acts[:, -1, dim_idx] = (
                projected_acts[:, -1, dim_idx] * coefficient
            )
            # Project back to activation space.
            ablated_activations = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    projected_acts,
                    encoder.T.to(model.device),
                )
            )
            # We must preserve the attention data in `output[1]`.
            return (
                ablated_activations,
                output[1],
            )

        return ablate_hook

    def cache_hook_fac(
        ablate_dim_idx: int,
        cache_dims: list[int],
        ablate_layer_idx: int,
        encoder: t.Tensor,
        biases: t.Tensor,
        cache_dict: defaultdict,
    ):
        """Create hooks that cache the projected activations."""

        def cache_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """Cache projected activations."""

            # Project activations through the encoder/bias.
            projected_acts_unrec = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    output[0],
                    encoder.to(model.device),
                    bias=biases.to(model.device),
                )
            )
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=False
            )

            # Cache the activations.
            for cache_dim in cache_dims:
                extant_data = cache_dict[ablate_layer_idx][ablate_dim_idx][
                    cache_dim
                ]
                # A defaultdict here means no cached data yet.
                if isinstance(extant_data, defaultdict):
                    cache_dict[ablate_layer_idx][ablate_dim_idx][
                        cache_dim
                    ] = (
                        projected_acts[:, :, cache_dim]
                        .unsqueeze(-1)
                        .detach()
                        .cpu()
                    )
                # We concat if there's an existing tensor.
                elif isinstance(extant_data, t.Tensor):
                    cache_dict[ablate_layer_idx][ablate_dim_idx][
                        cache_dim
                    ] = t.cat(
                        (
                            extant_data,
                            projected_acts[:, :, cache_dim]
                            .unsqueeze(-1)
                            .detach()
                            .cpu(),
                        ),
                        dim=1,
                    )
                else:
                    raise ValueError(
                        f"Unexpected data type in cache: {type(extant_data)}"
                    )

        return cache_hook

    if ablate_layer_idx == model_layer_range[-1]:
        raise ValueError("Cannot ablate and cache from the last layer.")
    cache_layer_idx: int = ablate_layer_idx + 1
    # Just the Pythia layer syntax, for now.
    if ablate_during_run:
        ablate_encoder, ablate_bias = tensors_per_layer[ablate_layer_idx]
        ablate_hook_handle = model.gpt_neox.layers[
            ablate_layer_idx
        ].register_forward_hook(
            ablate_hook_fac(ablate_dim_idx, ablate_encoder, ablate_bias)
        )

    cache_encoder, cache_bias = tensors_per_layer[cache_layer_idx]
    cache_hook_handle = model.gpt_neox.layers[
        cache_layer_idx
    ].register_forward_hook(
        cache_hook_fac(
            ablate_dim_idx,
            cache_dim_indices[cache_layer_idx],
            ablate_layer_idx,
            cache_encoder,
            cache_bias,
            activations_dict,
        )
    )

    try:
        yield
    finally:
        cache_hook_handle.remove()
        if ablate_during_run:
            ablate_hook_handle.remove()
