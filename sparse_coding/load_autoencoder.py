# %%
"""Load sparse autoencoders from HuggingFace."""


import torch as t
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    save_paths,
    sanitize_model_name,
)
from sae_training.utils import (  # pylint: disable=unused-import
    LMSparseAutoencoderSessionloader,
)

# %%
# Set up constants.
_, config = load_yaml_constants(__file__)

ENCODER_FILE = config.get("ENCODER_FILE")
ENC_BIASES_FILE = config.get("ENC_BIASES_FILE")
DECODER_FILE = config.get("DECODER_FILE")
DEC_BIASES_FILE = config.get("DEC_BIASES_FILE")
ATTN_ENCODER_FILE = config.get("ATTN_ENCODER_FILE")
ATTN_ENC_BIASES_FILE = config.get("ATTN_ENC_BIASES_FILE")
ATTN_DECODER_FILE = config.get("ATTN_DECODER_FILE")
ATTN_DEC_BIASES_FILE = config.get("ATTN_DEC_BIASES_FILE")
MLP_ENCODER_FILE = config.get("MLP_ENCODER_FILE")
MLP_ENC_BIASES_FILE = config.get("MLP_ENC_BIASES_FILE")
MLP_DECODER_FILE = config.get("MLP_DECODER_FILE")
MLP_DEC_BIASES_FILE = config.get("MLP_DEC_BIASES_FILE")
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
ACTS_LAYERS_RANGE = range(ACTS_LAYERS_SLICE.start, ACTS_LAYERS_SLICE.stop)

RESID_REPO: str = "jbloom/GPT2-Small-SAEs"
ATTN_REPO: str = "jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs"
MLP_REPO: str = "jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs"

assert (
    MODEL_DIR == "openai-community/gpt2"
), "Model directory must be openai-community/gpt2."

assert PROJECTION_FACTOR == 32, "Projection factor must be 32."


# %%
# Residual autopencoder loading functionality.
def load_resid_autoencoder(
    autoencoder_repository: str,
    encoder_file,
    enc_biases_file,
    decoder_file,
    dec_biases_file,
    model_dir,
    acts_layers_range,
    base_file: str,
) -> None:
    """Save a sparse autoencoder directly to disk."""

    prefix: str = "final_sparse_autoencoder_gpt2-small_blocks."
    endfix: str = ".hook_resid_pre_24576.pt"

    for idx in acts_layers_range:
        filename: str = f"{prefix}{idx}{endfix}"
        file_url = hf_hub_download(
            repo_id=autoencoder_repository,
            filename=filename,
        )

        # Solves a GPU availability issue with CI runners.
        if not t.cuda.is_available():
            tensors_dict = t.load(file_url, map_location="cpu")
        else:
            tensors_dict = t.load(file_url)

        encoder = tensors_dict["state_dict"]["W_enc"]
        enc_biases = tensors_dict["state_dict"]["b_enc"]
        decoder = tensors_dict["state_dict"]["W_dec"]
        dec_biases = tensors_dict["state_dict"]["b_dec"]

        safe_model_name = sanitize_model_name(model_dir)
        t.save(
            encoder,
            save_paths(base_file, f"{safe_model_name}/{idx}/{encoder_file}"),
        )
        t.save(
            enc_biases,
            save_paths(
                base_file, f"{safe_model_name}/{idx}/{enc_biases_file}"
            ),
        )
        t.save(
            decoder,
            save_paths(base_file, f"{safe_model_name}/{idx}/{decoder_file}"),
        )
        t.save(
            dec_biases,
            save_paths(
                base_file, f"{safe_model_name}/{idx}/{dec_biases_file}"
            ),
        )


# %%
# Call residual loading.
load_resid_autoencoder(
    RESID_REPO,
    ENCODER_FILE,
    ENC_BIASES_FILE,
    DECODER_FILE,
    DEC_BIASES_FILE,
    MODEL_DIR,
    ACTS_LAYERS_RANGE,
    __file__,
)


# %%
# Attn and MLP autoencoder loading functionality.
def load_sublayer_autoencoder(
    autoencoder_repo: str,
    encoder_file: str,
    enc_biases_file: str,
    decoder_file: str,
    dec_biases_file: str,
    model_dir: str,
    acts_layers_range: range,
    base_file: str,
):
    """
    Load sublayer autoencoders from HuggingFace.

    The HF Hub interface for these is rather different, so they are getting
    their own import function.
    """

    filename: str = "sae_weights.safetensors"

    for idx in acts_layers_range:
        subfolder: str = f"v5_128k_layer_{idx}"
        safe_model_name = sanitize_model_name(model_dir)

        file_url = hf_hub_download(
            repo_id=autoencoder_repo,
            filename=filename,
            subfolder=subfolder,
        )

        tensors_dict: dict = {}
        with safe_open(file_url, "pt") as f:
            for k in f.keys():
                tensors_dict[k] = f.get_tensor(k)

        encoder = tensors_dict["W_enc"]
        enc_biases = tensors_dict["b_enc"]
        decoder = tensors_dict["W_dec"]
        dec_biases = tensors_dict["b_dec"]

        t.save(
            encoder,
            save_paths(base_file, f"{safe_model_name}/{idx}/{encoder_file}"),
        )
        t.save(
            enc_biases,
            save_paths(
                base_file, f"{safe_model_name}/{idx}/{enc_biases_file}"
            ),
        )
        t.save(
            decoder,
            save_paths(base_file, f"{safe_model_name}/{idx}/{decoder_file}"),
        )
        t.save(
            dec_biases,
            save_paths(
                base_file, f"{safe_model_name}/{idx}/{dec_biases_file}"
            ),
        )


# %%
# Call attn and mlp loading.
load_sublayer_autoencoder(
    ATTN_REPO,
    ATTN_ENCODER_FILE,
    ATTN_ENC_BIASES_FILE,
    ATTN_DECODER_FILE,
    ATTN_DEC_BIASES_FILE,
    MODEL_DIR,
    ACTS_LAYERS_RANGE,
    __file__,
)

load_sublayer_autoencoder(
    MLP_REPO,
    MLP_ENCODER_FILE,
    MLP_ENC_BIASES_FILE,
    MLP_DECODER_FILE,
    MLP_DEC_BIASES_FILE,
    MODEL_DIR,
    ACTS_LAYERS_RANGE,
    __file__,
)
