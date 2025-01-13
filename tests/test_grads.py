"""Test the grads manager."""

from copy import deepcopy
from runpy import run_module

import torch as t
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from sparse_coding.utils.interface import (
    slice_to_range,
    load_yaml_constants,
    parse_slice,
)
from sparse_coding.utils.tasks import recursive_defaultdict
from sparse_coding.interp_tools.utils.hooks import (
    grads_manager,
    measure_confounds,
    prepare_autoencoder_and_indices,
)
from tests.test_smoke_sparse_coding import (  # pylint: disable=unused-import
    mock_interface,
)


def test_edge_level_effects(
    mock_interface,
):  # pylint: disable=redefined-outer-name, unused-argument
    """Assert correctness of edge-level effects across layers 11-12."""

    # Self-contained preparations.
    scripts = [
        "collect_acts",
        "load_autoencoder",
        "interp_tools.contexts",
    ]
    for script in scripts:
        with wandb.init(mode="offline"):
            run_module(f"sparse_coding.{script}")

    # Test constants are patched over
    _, config = load_yaml_constants(__file__)

    model_dir = config.get("MODEL_DIR")
    prompt = config.get("PROMPT")
    acts_layers_slice = parse_slice(config.get("ACTS_LAYERS_SLICE"))
    encoder_file = config.get("ENCODER_FILE")
    enc_biases_file = config.get("ENC_BIASES_FILE")
    decoder_file = config.get("DECODER_FILE")
    dec_biases_file = config.get("DEC_BIASES_FILE")
    attn_encoder_file = config.get("ATTN_ENCODER_FILE")
    attn_enc_biases_file = config.get("ATTN_ENC_BIASES_FILE")
    attn_decoder_file = config.get("ATTN_DECODER_FILE")
    attn_dec_biases_file = config.get("ATTN_DEC_BIASES_FILE")
    mlp_encoder_file = config.get("MLP_ENCODER_FILE")
    mlp_enc_biases_file = config.get("MLP_ENC_BIASES_FILE")
    mlp_decoder_file = config.get("MLP_DECODER_FILE")
    mlp_dec_biases_file = config.get("MLP_DEC_BIASES_FILE")
    resid_tokens_file = config.get("TOP_K_INFO_FILE")
    attn_tokens_file = config.get("ATTN_TOKEN_FILE")
    mlp_tokens_file = config.get("MLP_TOKEN_FILE")
    num_down_nodes = config.get("NUM_DOWN_NODES")

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_dir,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        model_dir, clean_up_tokenization_spaces=True
    )
    accelerator: Accelerator = Accelerator()

    model = accelerator.prepare(model)
    layer_range = slice_to_range(model, acts_layers_slice)

    # %%
    # Prepare all layer range autoencoders.
    # Residual autoencoders
    res_enc_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        model_dir,
        encoder_file,
        enc_biases_file,
        resid_tokens_file,
        accelerator,
        __file__,
    )
    res_dec_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        model_dir,
        decoder_file,
        dec_biases_file,
        resid_tokens_file,
        accelerator,
        __file__,
    )

    # Attention autoencoders
    attn_enc_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        model_dir,
        attn_encoder_file,
        attn_enc_biases_file,
        attn_tokens_file,
        accelerator,
        __file__,
    )
    attn_dec_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        model_dir,
        attn_decoder_file,
        attn_dec_biases_file,
        attn_tokens_file,
        accelerator,
        __file__,
    )

    # MLP autoencoders
    mlp_enc_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        model_dir,
        mlp_encoder_file,
        mlp_enc_biases_file,
        mlp_tokens_file,
        accelerator,
        __file__,
    )
    mlp_dec_and_biases, _ = prepare_autoencoder_and_indices(
        layer_range,
        model_dir,
        mlp_decoder_file,
        mlp_dec_biases_file,
        mlp_tokens_file,
        accelerator,
        __file__,
    )

    metric = t.nn.CrossEntropyLoss()

    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
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
            if "error_" not in loc:
                ab_top_values, ab_top_indices = t.topk(
                    weighted_prod.abs(), num_down_nodes
                )
                indices: list = ab_top_indices[ab_top_values > 0.0].tolist()
            elif "error_" in loc:
                # Sum across the error tensors, since we don't care about
                # the edges into the neuron basis.
                weighted_prod = weighted_prod.sum().unsqueeze(0)
                indices: list = [0]
            else:
                raise ValueError("Module location not recognized.")
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
    raise ValueError()
