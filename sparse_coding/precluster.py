# %%
"""Partition the dataset into k-clusters based on activation cosines."""


import sys
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch as t
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
)
import wandb

from sparse_coding.utils.interface import (
    load_input_token_ids,
    load_yaml_constants,
    parse_slice,
    sanitize_model_name,
    save_paths,
    slice_to_range,
)
from sparse_coding.utils.top_contexts import unpad_activations


# %%
# Set up constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
RESID_DATA_FILE = config.get("ACTS_DATA_FILE")
ATTN_DATA_FILE = config.get("ATTN_DATA_FILE")
MLP_DATA_FILE = config.get("MLP_DATA_FILE")
DATASET = config.get("DATASET")
NUM_CLUSTERS = config.get("NUM_CLUSTERS")
KEEPER_CLUSTER_INDEX = config.get("KEEPER_CLUSTER_INDEX")
SEED = config.get("SEED")

if DATASET is None:
    print("DATASET not set; not partitioning forward passes.")
    sys.exit(0)
else:
    dataset_name: str = DATASET.split("/")[-1]
    print(f"Partitioning dataset: {dataset_name}")

datafiles: list[str] = [
    RESID_DATA_FILE,
    ATTN_DATA_FILE,
    MLP_DATA_FILE,
]

# %%
# Reproducibility
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Log config to wandb.
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config)

# %%
# Load model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
# Ranges are iterable while slices aren't.
acts_layers_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)

# %%
# Load token ids.
token_ids: list[list[int]] = load_input_token_ids(PROMPT_IDS_PATH)

# %%
# Cluster into k-partitions.
print(f"Partitioning into {NUM_CLUSTERS} clusters.")

for layer_idx in acts_layers_range:
    for datafile in datafiles:
        acts_path: str = save_paths(
            __file__,
            f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{datafile}",
        )
        acts: t.Tensor = t.load(acts_path, weights_only=True)

        acts_list: list[t.Tensor] = unpad_activations(acts, token_ids)
        seq_by_hidden_acts: t.Tensor = t.cat(acts_list, dim=0).cpu()

        # Cluster using cosine similarity.
        normed_acts = normalize(seq_by_hidden_acts, norm="l2", axis=1)
        # Initialize repeatedly to maintain reproducibility with the seed.
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init=10)
        clusters_indices = kmeans.fit_predict(normed_acts)

        # Save select cluster.
        cluster = seq_by_hidden_acts[clusters_indices == KEEPER_CLUSTER_INDEX]
