"""Ablation and caching hooks."""

import re
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager
from textwrap import dedent
from typing import Generator

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
    dims_plotted_dict: dict[int, list[int]] | None,
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
                for i in v:
                    assert i in ablate_dim_indices, dedent(
                        f"Index {v} not in `ablate_dim_indices`."
                    )
                    specified_dims.append(i)

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
    ablate_dim_indices: list[int],
    model_layer_range: range,
    cache_dim_indices: dict[int, list[int]],
    model,
    enc_tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    dec_tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    activations_dict: defaultdict,
    ablate_during_run: bool = True,
    coefficient: float = 0.0,
):
    """
    Context manager for the full-scale ablations and caching.

    Ablates the specified feature at `layer_idx` and caches the downstream
    effects. `coefficient` can be set for other multiplicative pinnings, apart
    from ablation.
    """

    def ablate_hook_fac(
        dim_indices: list[int],
        encoder: t.Tensor,
        enc_biases: t.Tensor,
        decoder,
        dec_biases,
    ):
        """Create hooks that zero projected neurons and project them back."""

        def ablate_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """
            Project activation vectors; ablate them; project them back.
            """

            # Project through the encoder. Bias usage now corresponds to Joseph
            # Bloom's (and, by way of him, Antropic's).
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

            # Zero out or otherwise pin the column vectors specified.
            mask = t.ones(projected_acts.shape, dtype=t.bool).to(model.device)
            if coefficient == 0.0:
                mask[:, :, dim_indices] = False
            else:
                mask = mask.float()
                mask[:, :, dim_indices] *= coefficient
            ablated_acts = projected_acts * mask

            projected_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    projected_acts,
                    decoder.T.to(model.device),
                    bias=dec_biases.to(model.device),
                )
            )
            ablated_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    ablated_acts,
                    decoder.T.to(model.device),
                    bias=dec_biases.to(model.device),
                )
            )

            # Perform the ablation. The right term reflects just ablation
            # effects, hopefully canceling out autoencoder mangling. We must
            # also preserve the attention data in `output[1]`.
            return (
                output[0] + (ablated_acts - projected_acts),
                output[1],
            )

        return ablate_hook

    def cache_hook_fac(
        ablate_dim_idx: list[int],
        cache_dims: list[int],
        ablate_layer_idx: int,
        encoder: t.Tensor,
        enc_biases: t.Tensor,
        dec_biases: t.Tensor,
        cache_dict: defaultdict,
    ):
        """Create hooks that cache the projected activations."""
        if isinstance(ablate_dim_idx, list) and len(ablate_dim_idx) != 1:
            # Don't cache during multiablations.
            return lambda *args: None

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
                        projected_acts[:, -1, cache_dim]
                        .unsqueeze(-1)
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
                                projected_acts[:, -1, cache_dim]
                                .unsqueeze(-1)
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
                ablate_dim_indices,
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
            ablate_dim_indices,
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


@contextmanager
def jacobians_manager(
    upstream_layer_idx: int,
    model,
    enc_tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    dec_tensors_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
) -> Generator[dict, None, None]:
    """
    Context manager for Jacobian-hooking forward passes.

    Only residual stream autoencoders are supported here.
    """
    jac_dict: dict = {}

    def composite_module(current_module: t.nn.Module) -> t.nn.Sequential:
        """The relevant torch modules, composed."""

        class OffsetBy(t.nn.Module):
            """Subtract a bias from an input tensor."""

            def __init__(self, bias: t.Tensor):
                super().__init__()

                self.bias = bias

            def forward(self, x: t.Tensor) -> t.Tensor:
                """Just subtract the bias from the input tensor."""

                return x[0] - self.bias

        decoder_1, dec_bias_1 = dec_tensors_per_layer[upstream_layer_idx]
        _, dec_bias_2 = dec_tensors_per_layer[upstream_layer_idx + 1]
        encoder_2, enc_bias_2 = enc_tensors_per_layer[upstream_layer_idx + 1]
        # Recreates forward-pass section.
        composed_mod = t.nn.Sequential(
            t.nn.Linear(decoder_1.shape[1], decoder_1.shape[0]),
            current_module,
            OffsetBy(dec_bias_2),
            t.nn.Linear(encoder_2.shape[1], encoder_2.shape[0]),
            t.nn.ReLU(inplace=True),
        )

        # Assign weight and bias tensors to submodules.
        composed_mod[0].weight = t.nn.Parameter(decoder_1.T)
        composed_mod[0].bias = t.nn.Parameter(dec_bias_1)
        composed_mod[3].weight = t.nn.Parameter(encoder_2.T)
        composed_mod[3].bias = t.nn.Parameter(enc_bias_2)

        return composed_mod

    def splice_hook_fac(
        encoder: t.Tensor,
        enc_biases: t.Tensor,
        decoder: t.Tensor,
        dec_biases: t.Tensor,
    ):
        """
        Create hooks that interfere with gradients to get proper jacobians.
        """

        def splice_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """
            Splice zero tensors into a forward pass; divert them out; call a
            torch Jacobian method on them.
            """

            # Project activations through the encoder. Bias usage corresponds
            # to JBloom's.
            projected_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    output[0] - dec_biases.to(model.device),
                    encoder.T.to(model.device),
                    bias=enc_biases.to(model.device),
                ).to(model.device)
            )
            t.nn.functional.relu(
                projected_acts,
                inplace=True,
            )

            jac_dict["point"] = projected_acts.detach()

            projected_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    projected_acts,
                    decoder.T.to(model.device),
                    bias=dec_biases.to(model.device),
                )
            )

            return (projected_acts, output[1])

        return splice_hook

    def divert_hook_fac(
        encoder: t.Tensor,
        enc_biases: t.Tensor,
        dec_biases: t.Tensor,
    ):
        """
        Create the downstream hook that computes and caches the Jacobian.
        """

        def divert_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """
            Divert the spliced acts tensor; call a torch Jacobian method on it;
            put the Jacobian in a returned defaultdict with key data.
            """
            projected_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    output[0] - dec_biases.to(model.device),
                    encoder.T.to(model.device),
                    bias=enc_biases.to(model.device),
                ).to(model.device)
            )
            t.nn.functional.relu(
                projected_acts,
                inplace=True,
            )

            differentiable_mod = composite_module(module)
            # Functional. Note that a `chunk_size` can be set here other than
            # the full tensor.
            jacobian = t.func.jacfwd(differentiable_mod)
            jac_dict["function"] = jacobian

        return divert_hook

    splice_encoder, splice_enc_bias = enc_tensors_per_layer[upstream_layer_idx]
    splice_decoder, splice_dec_bias = dec_tensors_per_layer[upstream_layer_idx]
    splice_hook_handle = model.transformer.h[
        upstream_layer_idx
    ].register_forward_hook(
        splice_hook_fac(
            splice_encoder, splice_enc_bias, splice_decoder, splice_dec_bias
        )
    )

    divert_encoder, divert_enc_bias = enc_tensors_per_layer[
        upstream_layer_idx + 1
    ]
    _, divert_dec_bias = dec_tensors_per_layer[upstream_layer_idx + 1]
    divert_hook_handle = model.transformer.h[
        upstream_layer_idx + 1
    ].register_forward_hook(
        divert_hook_fac(divert_encoder, divert_enc_bias, divert_dec_bias)
    )

    try:
        yield jac_dict
    finally:
        splice_hook_handle.remove()
        divert_hook_handle.remove()


@contextmanager
def grads_manager(
    model: t.nn.Module,
    layer_indices: list[int],
    res_enc_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    res_dec_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    attn_enc_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    attn_dec_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    mlp_enc_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
    mlp_dec_per_layer: dict[int, tuple[t.Tensor, t.Tensor]],
) -> Generator[tuple[dict, dict], None, None]:
    """Context manager for backward hooks on autoencoder inserts."""

    acts_dict: dict = {}
    grads_dict: dict = {}
    handles = []
    ripcord = []

    def backwards_replace_fac(replace):
        """Replace a gradient with the gradient of an argument."""

        def backward_replace_hook(grad):  # pylint: disable=unused-argument
            """Replace a gradient with a closure object's gradient."""
            return replace.grad

        return backward_replace_hook

    def backwards_cache_fac(location: str):
        """Allow backward hooks to label their dictionary entries."""

        def backward_cache_hook(grad):
            """Label the gradient tensor with its location."""
            grads_dict[location] = grad

        return backward_cache_hook

    def forward_hooks_fac(
        layer_idx: int,
        encoder: t.Tensor,
        enc_biases: t.Tensor,
        decoder: t.Tensor,
        dec_biases: t.Tensor,
    ):
        """
        Pass activations through autoencoders where requested and register
        backward hooks at the autoencoder tensors.
        """

        def forward_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """
            Pass activations through autoencoder and register a backward hook
            at the autoencoder tensor and error residual.
            """

            # Module detection
            if "attention" in module.__class__.__name__.lower():
                # "gpt2attention"
                current_name: str = f"attn_{layer_idx}"
                error_name: str = f"attn_error_{layer_idx}"
            elif "mlp" in module.__class__.__name__.lower():
                # "gpt2mlp"
                current_name: str = f"mlp_{layer_idx}"
                error_name: str = f"mlp_error_{layer_idx}"
            elif "gpt2block" in module.__class__.__name__.lower():
                # "gpt2block"
                # Standardizing to "resid_"
                current_name: str = f"resid_{layer_idx}"
                error_name: str = f"resid_error_{layer_idx}"
            else:
                raise ValueError("Unexpected module name.")

            # Handle tuple cases
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                assert isinstance(output, t.Tensor)
                hidden = output

            # Project activations through the encoder. Bias usage corresponds
            # to JBloom's.
            projected_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    hidden - dec_biases.to(model.device),
                    encoder.T.to(model.device),
                    bias=enc_biases.to(model.device),
                ).to(model.device)
            )
            projected_acts = t.nn.functional.relu(projected_acts)

            handles.append(
                projected_acts.register_hook(backwards_cache_fac(current_name))
            )
            # Note that the cached projected acts are neither cloned nor
            # detached yet.
            acts_dict[current_name] = projected_acts

            # Decode projected acts.
            decoded_acts = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    projected_acts,
                    decoder.T.to(model.device),
                    bias=dec_biases.to(model.device),
                )
            )

            # Algebra for the error residual.
            error = hidden - decoded_acts

            # Note that the cached error acts are neither cloned nor detached
            # yet.
            acts_dict[error_name] = error
            # Stop grad must go here
            error = error.detach().requires_grad_(True)

            handles.append(
                error.register_hook(backwards_cache_fac(error_name))
            )

            # hidden = decoded_acts + error
            reconstructed: t.Tensor = decoded_acts + error

            # This block brings this repo's grads into alignment with the Bau
            # Lab implementation.
            reconstructed.retain_grad()
            ripcord.append(
                hidden.register_hook(backwards_replace_fac(reconstructed))
            )

            if isinstance(output, tuple):
                return reconstructed, output[1]
            if isinstance(output, t.Tensor):
                return reconstructed
            raise ValueError("Unexpected output type.")

        return forward_hook

    # The context manager registers the initial forward hooks.
    for layer_idx in layer_indices:
        # Residual stream
        res_enc, res_enc_bias = res_enc_per_layer[layer_idx]
        res_dec, res_dec_bias = res_dec_per_layer[layer_idx]
        handles.append(
            model.transformer.h[layer_idx].register_forward_hook(
                forward_hooks_fac(
                    layer_idx, res_enc, res_enc_bias, res_dec, res_dec_bias
                )
            )
        )

        # Attention
        attn_enc, attn_enc_bias = attn_enc_per_layer[layer_idx]
        attn_dec, attn_dec_bias = attn_dec_per_layer[layer_idx]
        handles.append(
            model.transformer.h[layer_idx].attn.register_forward_hook(
                forward_hooks_fac(
                    layer_idx, attn_enc, attn_enc_bias, attn_dec, attn_dec_bias
                )
            )
        )

        # MLP
        mlp_enc, mlp_enc_bias = mlp_enc_per_layer[layer_idx]
        mlp_dec, mlp_dec_bias = mlp_dec_per_layer[layer_idx]
        handles.append(
            model.transformer.h[layer_idx].mlp.register_forward_hook(
                forward_hooks_fac(
                    layer_idx, mlp_enc, mlp_enc_bias, mlp_dec, mlp_dec_bias
                )
            )
        )

    try:
        yield (acts_dict, grads_dict, ripcord)
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def attn_mlp_acts_manager(
    model: t.nn.Module,
    layer_indices: list[int],
) -> Generator[dict[str, t.Tensor], None, None]:
    """Retrieve select attn-out and MLP-out activations."""

    acts_dict: dict[str, t.Tensor] = {}
    handles = []

    def forward_hooks_fac(layer_idx: int):
        """Create attn-out and MLP-out forward act hooks for GPT-2-small."""

        def forward_hook(  # pylint: disable=unused-argument, redefined-builtin
            module, input, output
        ) -> None:
            """Cache activations."""

            if "attention" in module.__class__.__name__.lower():
                # "gpt2attention"
                current_name: str = f"attn_{layer_idx}"
            elif "mlp" in module.__class__.__name__.lower():
                # "gpt2mlp"
                current_name: str = f"mlp_{layer_idx}"
            else:
                raise ValueError("Unexpected module name.")

            if current_name not in acts_dict:
                acts_dict[current_name] = output[0]
            else:
                acts_dict[current_name] = t.cat(
                    (acts_dict[current_name], output[0]),
                    dim=0,
                )

        return forward_hook

    for layer_idx in layer_indices:
        handles.append(
            model.transformer.h[layer_idx].attn.register_forward_hook(
                forward_hooks_fac(layer_idx)
            )
        )
        handles.append(
            model.transformer.h[layer_idx].mlp.register_forward_hook(
                forward_hooks_fac(layer_idx)
            )
        )

    try:
        yield acts_dict
    finally:
        for handle in handles:
            handle.remove()


def measure_confounds(
    down_node_location: str,
    gradients: dict,
    activations: dict,
    acts_and_grads,
) -> tuple | dict | None:
    """
    Provide confound effect grads for each down-node.

    This function runs inside a backward-pass context manager which has a
    global `acts_and_grads`. The edges resid-to-attn, attn-to-mlp, mlp-to-resid
    have no confounds.
    """
    resid_pattern: str = "(resid_error_|resid_)"
    mlp_pattern: str = "(mlp_error_|mlp_)"

    # We don't want the input gradients to change dynamically if we need to
    # take several backward pass measurements:
    gradients = deepcopy(gradients)

    # Deal with two possible cases; we only know the down node is resid
    if re.match("resid_", down_node_location) is not None:
        # RESID: attn-to-resid - (attn-to-mlp-to-resid)
        # sub in mlp
        mlp_handle: str = re.sub(resid_pattern, "mlp_", down_node_location)
        # Weight mlp acts by effect on resid.
        mlp_affecting_resid = t.einsum(
            "...sd,...sd->...s",
            gradients[mlp_handle],
            activations[mlp_handle],
        ).squeeze()
        # Differentiate effects w/r/t attn
        if mlp_affecting_resid.dim() == 0:
            # Single-token prompt edge case.
            mlp_affecting_resid.backward(retain_graph=True)
        else:
            mlp_affecting_resid[-1].backward(retain_graph=True)
        _, x_to_mlp_to_resid_grads, _ = acts_and_grads
        x_to_mlp_to_resid_grads = deepcopy(x_to_mlp_to_resid_grads)

        # RESID:
        # resid-to-resid - (resid-to-attn-to-resid) - (resid-to-mlp-to-resid)
        # + (resid-to-attn-to-mlp-to-resid)
        # (sub in mlp case was already covered), sub in attn
        attn_handle: str = re.sub(resid_pattern, "attn_", down_node_location)
        attn_affecting_resid = t.einsum(
            "...sd,...sd->...s",
            gradients[attn_handle],
            activations[attn_handle],
        ).squeeze()
        # Differentiate effects w/r/t resid
        if attn_affecting_resid.dim() == 0:
            # Single-token prompt edge case.
            attn_affecting_resid.backward(retain_graph=True)
        else:
            attn_affecting_resid[-1].backward(retain_graph=True)
        _, resid_to_attn_to_resid_grads, _ = acts_and_grads
        resid_to_attn_to_resid_grads = deepcopy(resid_to_attn_to_resid_grads)

        attn_affecting_mlp_affecting_resid = t.einsum(
            "...sd,...sd->...s",
            x_to_mlp_to_resid_grads[attn_handle],
            activations[attn_handle],
        ).squeeze()
        if attn_affecting_mlp_affecting_resid.dim() == 0:
            attn_affecting_mlp_affecting_resid.backward(retain_graph=True)
        else:
            attn_affecting_mlp_affecting_resid[-1].backward(retain_graph=True)
        # up_resid affecting attn affecting mlp affecting resid
        _, full_confound_grads, _ = acts_and_grads

        return (
            x_to_mlp_to_resid_grads,
            resid_to_attn_to_resid_grads,
            full_confound_grads,
        )

    # MLP: resid-to-mlp - (resid-to-attn-to-mlp)
    if re.match("mlp_", down_node_location) is not None:
        # sub in attn
        attn_handle: str = re.sub(mlp_pattern, "attn_", down_node_location)
        attn_affecting_mlp = t.einsum(
            "...sd,...sd->...s",
            gradients[attn_handle],
            activations[attn_handle],
        ).squeeze()
        # Differentiate effects w/r/t resid
        if attn_affecting_mlp.dim() == 0:
            # Single-token prompt edge case.
            attn_affecting_mlp.backward(retain_graph=True)
        else:
            attn_affecting_mlp[-1].backward(retain_graph=True)
        _, resid_to_attn_to_mlp_grads, _ = acts_and_grads
        return resid_to_attn_to_mlp_grads

    assert re.match("attn_", down_node_location)
    return None
