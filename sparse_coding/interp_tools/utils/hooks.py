"""Ablation and caching hooks."""


from collections import defaultdict
from contextlib import contextmanager

import torch as t


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
def hooks_lifecycle(
    ablate_layer_idx: int,
    ablate_dim_idx: int,
    model_layer_range: range,
    cache_dim_indices: dict[int, list[int]],
    model,
    tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    activations_dict: defaultdict,
    ablate_during_run: bool = True,
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
            projected_acts = t.nn.functional.relu(
                projected_acts_unrec, inplace=False
            )
            # Zero out the activation at dim_idx.
            projected_acts[:, :, dim_idx] = 0.0
            # Project back to activation space.
            output = projected_acts.to(model.device)
            output = t.nn.functional.linear(  # pylint: disable=not-callable
                projected_acts.to(model.device),
                encoder.T.to(model.device),
            )
            output = (output,)

        return ablate_hook

    def cache_hook_fac(
        ablated_dim_idx: int,
        cache_dims: list[int],
        ablation_layer_idx: int,
        encoder: t.Tensor,
        biases: t.Tensor,
        cache: defaultdict,
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
            for downstream_dim in cache_dims:
                extant_data = cache[ablation_layer_idx][ablated_dim_idx][
                    downstream_dim
                ]
                # A defaultdict here means no cached data yet.
                if isinstance(extant_data, defaultdict):
                    cache[ablation_layer_idx][ablated_dim_idx][
                        downstream_dim
                    ] = (
                        projected_acts[:, :, downstream_dim]
                        .unsqueeze(-1)
                        .detach()
                        .cpu()
                    )
                # We concat if there's an existing tensor.
                elif isinstance(extant_data, t.Tensor):
                    cache[ablation_layer_idx][ablated_dim_idx][
                        downstream_dim
                    ] = t.cat(
                        (
                            extant_data,
                            projected_acts[:, :, downstream_dim]
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
    cache_range: range = range(ablate_layer_idx + 1, model_layer_range[-1] + 1)
    # Just the Pythia layer syntax, for now.
    if ablate_during_run:
        ablate_encoder, ablate_bias = tensors_per_layer[ablate_layer_idx]
        ablate_hook_handle = model.gpt_neox.layers[
            ablate_layer_idx
        ].register_forward_hook(
            ablate_hook_fac(ablate_dim_idx, ablate_encoder, ablate_bias)
        )

    cache_hook_handles = {}
    for cache_layer_idx in cache_range:
        cache_encoder, cache_bias = tensors_per_layer[cache_layer_idx]
        cache_hook_handles[cache_layer_idx] = model.gpt_neox.layers[
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
        if ablate_during_run:
            ablate_hook_handle.remove()
        for handle in cache_hook_handles.values():
            handle.remove()
