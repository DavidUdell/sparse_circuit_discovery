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
    dims_plotted_dict: dict[int, int] | None,
    ablate_dim_indices: list[int],
    ablate_layer_idx: int,
    layer_range: range,
    seed: int,
) -> list[int]:
    """
    Apply DIMS_PLOTTED_LIST and/or THINNING_FACTOR to ablate_dim_indices.

    `dims_plotted_list` will override `thinning_factor`, if set.
    `thinning_factor` will only be applied to the first layer, if set, since
    layer plotted dims are already pruned to those that were affected upstream.
    """

    if dims_plotted_dict is not None:
        specified_dims: list[int] = []
        for k, v in dims_plotted_dict.items():
            if k == ablate_layer_idx:
                assert v in ablate_dim_indices, dedent(
                    f"Index {v} not in `ablate_dim_indices`."
                )
                specified_dims.append(v)

        return specified_dims

    if thinning_factor is not None and ablate_layer_idx == layer_range[0]:
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
    enc_tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    dec_tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    activations_dict: defaultdict,
    ablate_during_run: bool = True,
):
    """
    Context manager for the full-scale ablations and caching.

    Ablates the specified feature at `layer_idx` and caches the downstream
    effects.
    """

    def ablate_hook_fac(
        dim_idx: int,
        encoder: t.Tensor,
        enc_biases: t.Tensor,
        decoder,
        dec_biases,
    ):
        """Create hooks that zero a projected neuron and project it back."""

        def ablate_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """
            Project activation vectors; ablate them; project them back.
            """

            # Project through the encoder. Bias usage now corresponds to Joseph
            # Bloom's (and, by his way, Antropic's).
            projected_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    output[0] - dec_biases.to(model.device),
                    encoder.T.to(model.device),
                    bias=enc_biases.to(model.device),
                )
            ).to(model.device)

            t.nn.functional.relu(
                projected_acts,
                inplace=True,
            )

            # We will ablate the activation at dim_idx by subtracting only that
            # dim value.
            projected_acts[:, -1, :dim_idx] = t.zeros_like(
                projected_acts[:, -1, :dim_idx]
            )
            projected_acts[:, -1, dim_idx + 1 :] = t.zeros_like(
                projected_acts[:, -1, dim_idx + 1 :]
            )

            projected_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    projected_acts,
                    decoder.T.to(model.device),
                    bias=dec_biases.to(model.device),
                )
            )
            # Perform the ablation. We must also preserve the attention data in
            # `output[1]`.
            return (
                output[0] - projected_acts,
                output[1],
            )

        return ablate_hook

    def cache_hook_fac(
        ablate_dim_idx: int,
        cache_dims: list[int],
        ablate_layer_idx: int,
        encoder: t.Tensor,
        enc_biases: t.Tensor,
        dec_biases: t.Tensor,
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
                    output[0] + dec_biases.to(model.device),
                    encoder.T.to(model.device),
                    bias=enc_biases.to(model.device),
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
                    cache_dict[ablate_layer_idx][ablate_dim_idx][cache_dim] = (
                        projected_acts[:, :, cache_dim]
                        .unsqueeze(-1)
                        .detach()
                        .cpu()
                    )
                # We concat if there's an existing tensor.
                elif isinstance(extant_data, t.Tensor):
                    cache_dict[ablate_layer_idx][ablate_dim_idx][cache_dim] = (
                        t.cat(
                            (
                                extant_data,
                                projected_acts[:, :, cache_dim]
                                .unsqueeze(-1)
                                .detach()
                                .cpu(),
                            ),
                            dim=1,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unexpected data type in cache: {type(extant_data)}"
                    )

        return cache_hook

    if ablate_layer_idx == model_layer_range[-1]:
        raise ValueError("Cannot ablate and cache from the last layer.")
    cache_layer_idx: int = ablate_layer_idx + 1
    # Just the GPT-2 small layer syntax, for now.
    if ablate_during_run:
        ablate_encoder, ablate_enc_bias = enc_tensors_per_layer[
            ablate_layer_idx
        ]
        ablate_decoder, ablate_dec_bias = dec_tensors_per_layer[
            ablate_layer_idx
        ]

        ablate_hook_handle = model.transformer.h[
            ablate_layer_idx
        ].register_forward_hook(
            ablate_hook_fac(
                ablate_dim_idx,
                ablate_encoder,
                ablate_enc_bias,
                ablate_decoder,
                ablate_dec_bias,
            )
        )

    cache_encoder, cache_enc_bias = enc_tensors_per_layer[cache_layer_idx]
    _, cache_dec_bias = dec_tensors_per_layer[cache_layer_idx]
    cache_hook_handle = model.transformer.h[
        cache_layer_idx
    ].register_forward_hook(
        cache_hook_fac(
            ablate_dim_idx,
            cache_dim_indices[cache_layer_idx],
            ablate_layer_idx,
            cache_encoder,
            cache_enc_bias,
            cache_dec_bias,
            activations_dict,
        )
    )

    try:
        yield
    finally:
        cache_hook_handle.remove()
        if ablate_during_run:
            ablate_hook_handle.remove()
