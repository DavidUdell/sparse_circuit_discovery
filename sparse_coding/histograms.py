# %%
"""
Histograms from autoencoder activations, to set distribution-aware thresholds.
"""


import itertools
import warnings

import numpy as np
import torch as t
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from tqdm.auto import tqdm
import wandb

from sparse_coding.interp_tools.utils.computations import Encoder
from sparse_coding.utils.interface import (
    load_yaml_constants,
    parse_slice,
    sanitize_model_name,
    save_paths,
    slice_to_range,
)


# %%
# Set up constants
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
RESID_ACTS_FILE = config.get("ACTS_DATA_FILE")
ATTN_ACTS_FILE = config.get("ATTN_DATA_FILE")
MLP_ACTS_FILE = config.get("MLP_DATA_FILE")
RESID_ENCODER_FILE = config.get("ENCODER_FILE")
RESID_BIASES_FILE = config.get("ENC_BIASES_FILE")
ATTN_ENCODER_FILE = config.get("ATTN_ENCODER_FILE")
ATTN_BIASES_FILE = config.get("ATTN_ENC_BIASES_FILE")
MLP_ENCODER_FILE = config.get("MLP_ENCODER_FILE")
MLP_BIASES_FILE = config.get("MLP_ENC_BIASES_FILE")
tsfm_config = AutoConfig.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
HIDDEN_DIM = tsfm_config.hidden_size
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
PROJECTION_DIM = int(HIDDEN_DIM * PROJECTION_FACTOR)
SEED = config.get("SEED")

PERCENTILE: float = 99.99

# %%
# Reproducibility
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config/run to wandb
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=config,
)

# %%
# Loop over all the model layers in the slice.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
    )
seq_layer_indices: range = slice_to_range(model, ACTS_LAYERS_SLICE)

resid = {
    "acts": RESID_ACTS_FILE,
    "encoder": RESID_ENCODER_FILE,
    "biases": RESID_BIASES_FILE,
}
attn = {
    "acts": ATTN_ACTS_FILE,
    "encoder": ATTN_ENCODER_FILE,
    "biases": ATTN_BIASES_FILE,
}
mlp = {
    "acts": MLP_ACTS_FILE,
    "encoder": MLP_ENCODER_FILE,
    "biases": MLP_BIASES_FILE,
}

sublayer_iterator = itertools.product(seq_layer_indices, [resid, attn, mlp])
percentiles_dict: dict[str, float] = {}

for layer_idx, sublayer in tqdm(
    sublayer_iterator,
    desc="Sublayer Percentiles",
    total=len(seq_layer_indices) * 3,
):
    acts_file: str = sublayer["acts"]
    encoder_file: str = sublayer["encoder"]
    biases_file: str = sublayer["biases"]

    encoder_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{encoder_file}",
    )
    biases_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{biases_file}",
    )
    acts_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{acts_file}",
    )
    imported_weights: t.Tensor = t.load(encoder_path, weights_only=True).T
    imported_biases: t.Tensor = t.load(biases_path, weights_only=True)
    # Moved this to CPU to avoid cuda memory crashes.
    layer_acts_data: t.Tensor = t.load(acts_path, weights_only=True).cpu()

    module_type: str = acts_file.split("_")[0]
    append: str = f"{module_type}_percentile.csv"
    percentile_path: str = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{append}",
    )

    graph_title: str = f"Layer {layer_idx} {module_type}"
    # Reassigning model variable to tell the garbage collector we're done with
    # it now.
    model: Encoder = Encoder(
        imported_weights,
        imported_biases,
        HIDDEN_DIM,
        PROJECTION_DIM,
    ).to("cpu")

    projected_acts: np.ndarray = (
        model(layer_acts_data).squeeze().detach().numpy()
    )

    # np.percentile() is actually faster than explicit histogram and cdf
    # computations.
    percentile = np.percentile(projected_acts, PERCENTILE)
    percentile = round(percentile, 2)

    percentiles_dict[percentile_path] = percentile

# %%
# Save percentiles to csv
for path, threshold in percentiles_dict.items():
    np.savetxt(
        path,
        [threshold],
    )

# %%
# Wrap up logging
wandb.finish()
