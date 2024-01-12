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
