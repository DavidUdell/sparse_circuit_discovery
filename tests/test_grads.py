"""Test the grads manager."""

from copy import deepcopy

import torch as t
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_coding.utils.interface import slice_to_range
from sparse_coding.utils.tasks import recursive_defaultdict
from sparse_coding.interp_tools.utils.hooks import (
    grads_manager,
    measure_confounds,
    prepare_autoencoder_and_indices,
)


def test_edge_level_effects():
    """Assert correctness of edge-level effects across layers 11-12."""
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, clean_up_tokenization_spaces=True
    )
    accelerator: Accelerator = Accelerator()

    model = accelerator.prepare(model)
    layer_range = slice_to_range(model, ACTS_LAYERS_SLICE)

    # %%
    # Prepare all layer range autoencoders.
    # Residual autoencoders
    res_enc_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        MODEL_DIR,
        ENCODER_FILE,
        ENC_BIASES_FILE,
        RESID_TOKENS_FILE,
        accelerator,
        __file__,
    )
    res_dec_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        MODEL_DIR,
        DECODER_FILE,
        DEC_BIASES_FILE,
        RESID_TOKENS_FILE,
        accelerator,
        __file__,
    )

    # Attention autoencoders
    attn_enc_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        MODEL_DIR,
        ATTN_ENCODER_FILE,
        ATTN_ENC_BIASES_FILE,
        ATTN_TOKENS_FILE,
        accelerator,
        __file__,
    )
    attn_dec_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        MODEL_DIR,
        ATTN_DECODER_FILE,
        ATTN_DEC_BIASES_FILE,
        ATTN_TOKENS_FILE,
        accelerator,
        __file__,
    )

    # MLP autoencoders
    mlp_enc_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        MODEL_DIR,
        MLP_ENCODER_FILE,
        MLP_ENC_BIASES_FILE,
        MLP_TOKENS_FILE,
        accelerator,
        __file__,
    )
    mlp_dec_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        MODEL_DIR,
        MLP_DECODER_FILE,
        MLP_DEC_BIASES_FILE,
        MLP_TOKENS_FILE,
        accelerator,
        __file__,
    )

    metric = t.nn.CrossEntropyLoss()

    tokens = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    inputs = tokens.copy()

    inputs["input_ids"] = inputs["input_ids"][:, :-1]
    inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
    target = tokens["input_ids"][:, -1].squeeze()

    acts_dict: dict = None
    grads_dict: dict = None
    old_grads_dict: dict = None
    marginal_grads_dict: dict = recursive_defaultdict()

    with grads_manager(
        model,
        layer_range,
        res_enc_and_biases,
        res_dec_and_biases,
        attn_enc_and_biases,
        attn_dec_and_biases,
        mlp_enc_and_biases,
        mlp_dec_and_biases,
    ) as acts_and_grads:

        # Forward pass installs all backward hooks.
        output = model(**inputs)
        logits = output.logits[:, -1, :].squeeze()

        loss = metric(
            logits,
            target,
        )
        loss.backward(retain_graph=True)

        acts_dict, grads_dict, ripcord = acts_and_grads

        # This block brings edge grads later on into correspondence with the
        # Bau Lab implementation. The backward hooks need to change for
        # succeeding backward passes; this removes the gradient replacement
        # hooks used above that are now unwanted.
        for h in ripcord:
            h.remove()

        old_grads_dict = deepcopy(grads_dict)

        for loc, grad in old_grads_dict.items():
            grad = grad.squeeze().unsqueeze(0)
            act = acts_dict[loc].squeeze().unsqueeze(0)

            weighted_prod = t.einsum("...sd,...sd->...sd", grad, act)
            # Standardize weighted_prod shape
            if weighted_prod.dim() == 2:
                # Single-token prompt edge case.
                weighted_prod = weighted_prod[-1, :].squeeze()
            else:
                weighted_prod = weighted_prod[:, -1, :].squeeze()

            ####  Thresholding down-nodes -> indices  ####
            percentile: None | float = percentiles.get(loc, None)
            if percentile is None:
                if "error_" not in loc:
                    ab_top_values, ab_top_indices = t.topk(
                        weighted_prod.abs(), NUM_DOWN_NODES
                    )
                    indices: list = ab_top_indices[
                        ab_top_values > 0.0
                    ].tolist()
                elif "error_" in loc:
                    # Sum across the error tensors, since we don't care about
                    # the edges into the neuron basis.
                    weighted_prod = weighted_prod.sum().unsqueeze(0)
                    indices: list = [0]
                else:
                    raise ValueError("Module location not recognized.")
            else:
                # elif percentile is float
                if acts_dict[loc].dim() == 2:
                    acts_tensor = acts_dict[loc][-1, :].squeeze()
                else:
                    acts_tensor = acts_dict[loc][:, -1, :].squeeze()
                thresh_tensor = t.full_like(acts_tensor, percentile)
                gt_tensor = t.nn.functional.relu(acts_tensor - thresh_tensor)
                indices: list | int = t.nonzero(gt_tensor).squeeze().tolist()
                if isinstance(indices, int):
                    indices: list = [indices]
                assert len(indices) > 0
            ####  End thresholding down-nodes  ####

            for dim_idx in indices:
                # Edge-level backward passes
                weighted_prod[dim_idx].backward(retain_graph=True)
                _, marginal_grads, _ = acts_and_grads

                old_marginal_grads = deepcopy(marginal_grads)

                down_layer_idx = int(loc.split("_")[-1])
                up_layer_idx = down_layer_idx - 1
                if up_layer_idx not in layer_range:
                    continue

                # Perpare confound effects for subtraction, per down-node
                confounds_grads = measure_confounds(
                    loc,
                    marginal_grads,
                    acts_dict,
                    acts_and_grads,
                )
                if isinstance(confounds_grads, tuple):
                    (
                        x_mlp_resid_grads,
                        resid_attn_resid_grads,
                        full_path_grads,
                    ) = confounds_grads
                    x_mlp_resid_grads = deepcopy(x_mlp_resid_grads)
                    resid_attn_resid_grads = deepcopy(resid_attn_resid_grads)
                    full_path_grads = deepcopy(full_path_grads)
                if confounds_grads is not None:
                    resid_attn_mlp_grads: dict = confounds_grads
                    resid_attn_mlp_grads = deepcopy(resid_attn_mlp_grads)

                # Deduplicate and store edges
                if "attn_" in loc:
                    # resid-to-attn
                    marginal_grads_dict[f"resid_{up_layer_idx}_to_" + loc][
                        dim_idx
                    ] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"resid_{up_layer_idx}"],
                        -acts_dict[f"resid_{up_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    marginal_grads_dict[
                        f"resid_error_{up_layer_idx}_to_" + loc
                    ][dim_idx] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"resid_error_{up_layer_idx}"],
                        -acts_dict[f"resid_error_{up_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                elif "mlp_" in loc:
                    # attn-to-mlp
                    marginal_grads_dict[f"attn_{down_layer_idx}_to_" + loc][
                        dim_idx
                    ] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"attn_{down_layer_idx}"],
                        -acts_dict[f"attn_{down_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    marginal_grads_dict[
                        f"attn_error_{down_layer_idx}_to_" + loc
                    ][dim_idx] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"attn_error_{down_layer_idx}"],
                        -acts_dict[f"attn_error_{down_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    # resid-to-mlp - (resid-attn-mlp)
                    marginal_grads_dict[f"resid_{up_layer_idx}_to_" + loc][
                        dim_idx
                    ] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"resid_{up_layer_idx}"]
                        - resid_attn_mlp_grads[  # pylint: disable=possibly-used-before-assignment
                            f"resid_{up_layer_idx}"
                        ],
                        -acts_dict[f"resid_{up_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    marginal_grads_dict[
                        f"resid_error_{up_layer_idx}_to_" + loc
                    ][dim_idx] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"resid_error_{up_layer_idx}"]
                        - resid_attn_mlp_grads[f"resid_error_{up_layer_idx}"],
                        -acts_dict[f"resid_error_{up_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                elif "resid_" in loc:
                    # attn-to-resid - (attn-mlp-resid)
                    marginal_grads_dict[f"attn_{down_layer_idx}_to_" + loc][
                        dim_idx
                    ] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"attn_{down_layer_idx}"]
                        - x_mlp_resid_grads[  # pylint: disable=possibly-used-before-assignment
                            f"attn_{down_layer_idx}"
                        ],
                        -acts_dict[f"attn_{down_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    marginal_grads_dict[
                        f"attn_error_{down_layer_idx}_to_" + loc
                    ][dim_idx] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"attn_error_{down_layer_idx}"]
                        - x_mlp_resid_grads[f"attn_error_{down_layer_idx}"],
                        -acts_dict[f"attn_error_{down_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    # mlp-to-resid
                    marginal_grads_dict[f"mlp_{down_layer_idx}_to_" + loc][
                        dim_idx
                    ] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"mlp_{down_layer_idx}"],
                        -acts_dict[f"mlp_{down_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    marginal_grads_dict[
                        f"mlp_error_{down_layer_idx}_to_" + loc
                    ][dim_idx] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"mlp_error_{down_layer_idx}"],
                        -acts_dict[f"mlp_error_{down_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    # resid-to-resid - (resid-attn-resid) - (resid-mlp-resid)
                    # + (resid-attn-mlp-resid)
                    marginal_grads_dict[f"resid_{up_layer_idx}_to_" + loc][
                        dim_idx
                    ] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"resid_{up_layer_idx}"]
                        - resid_attn_resid_grads[  # pylint: disable=possibly-used-before-assignment
                            f"resid_{up_layer_idx}"
                        ]
                        - x_mlp_resid_grads[f"resid_{up_layer_idx}"]
                        + full_path_grads[f"resid_{up_layer_idx}"],
                        -acts_dict[f"resid_{up_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                    marginal_grads_dict[
                        f"resid_error_{up_layer_idx}_to_" + loc
                    ][dim_idx] = t.einsum(
                        "...sd,...sd->...sd",
                        old_marginal_grads[f"resid_error_{up_layer_idx}"]
                        - resid_attn_resid_grads[f"resid_error_{up_layer_idx}"]
                        - x_mlp_resid_grads[f"resid_error_{up_layer_idx}"]
                        + full_path_grads[f"resid_error_{up_layer_idx}"],
                        -acts_dict[f"resid_error_{up_layer_idx}"],
                    )[
                        :, -1, :
                    ].cpu()

                else:
                    raise ValueError("Module location not recognized.")

    # Marginal-effects regression test printouts.
    print("Marginal effects:")
    for k, v in marginal_grads_dict.items():
        print(k)
        for i, j in v.items():
            print(i, j.detach())
        print()
