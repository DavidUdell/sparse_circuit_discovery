# %%
"""Partition the dataset into k-clusters based on activation cosines."""


import sys
import warnings
from textwrap import dedent

import numpy as np
from accelerate import Accelerator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch as t
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
)

from sparse_coding.utils.interface import (
    load_input_token_ids,
    load_yaml_constants,
    pad_activations,
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
# Load model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, token=HF_ACCESS_TOKEN
    )
accelerator: Accelerator = Accelerator()

# Ranges are subscriptable while slices aren't.
acts_layers_range: range = slice_to_range(model, ACTS_LAYERS_SLICE)
target_layer: int = acts_layers_range[0]
print(f"Target layer: {target_layer} residual.")

# %%
# Load token ids.
token_ids: list[list[int]] = load_input_token_ids(PROMPT_IDS_PATH)

# %%
# Cluster into k-partitions.
print(
    dedent(
        f"""
        Partitioning into {NUM_CLUSTERS} clusters; keeping cluster
        {KEEPER_CLUSTER_INDEX}.
        """
    )
)

acts_path: str = save_paths(
    __file__,
    f"{sanitize_model_name(MODEL_DIR)}/{target_layer}/{RESID_DATA_FILE}",
)
acts: t.Tensor = t.load(acts_path, weights_only=True)

acts_list: list[t.Tensor] = unpad_activations(acts, token_ids)
seq_by_hidden_acts: t.Tensor = t.cat(acts_list, dim=0).cpu()

# Cluster using cosine similarity.
normed_acts = normalize(seq_by_hidden_acts, norm="l2", axis=1)
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init=10)
flat_clusters_indices = kmeans.fit_predict(normed_acts)

# Reassemble flat_cluster_indices into the original shape.
cluster_brick = []
for seq in token_ids:
    cluster_brick.append(flat_clusters_indices[: len(seq)])
    flat_clusters_indices = flat_clusters_indices[len(seq) :]

filtered_token_ids: list[list[int]] = []
for seq, tokens in zip(cluster_brick, token_ids):
    if not any(x == KEEPER_CLUSTER_INDEX for x in seq):
        continue

    filtered_seq = []
    for x, token in zip(seq, tokens):
        filtered_seq.append(token)
        if x == KEEPER_CLUSTER_INDEX:
            break

    filtered_token_ids.append(filtered_seq)

# Mirror the filtered_token_ids pattern in the activations.
for layer_idx in acts_layers_range:
    for datafile in datafiles:
        acts_path: str = save_paths(
            __file__,
            f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{datafile}",
        )
        acts = t.load(acts_path, weights_only=True)
        # Unpacked original activations
        acts_list = unpad_activations(acts, token_ids)

        # Filtered activations list
        filtered_acts_list = []
        for seq_clusters, seq_acts in zip(cluster_brick, acts_list):
            if not any(c == KEEPER_CLUSTER_INDEX for c in seq_clusters):
                continue

            filtered_seq_acts: list = []
            for c, act in zip(seq_clusters, seq_acts):
                # Restore seq dimension
                act = act.unsqueeze(0)
                filtered_seq_acts.append(act)
                if c == KEEPER_CLUSTER_INDEX:
                    break

            filtered_seq_acts: t.Tensor = t.cat(filtered_seq_acts, dim=0)
            # Restore batch dimension
            filtered_acts_list.append(filtered_seq_acts.unsqueeze(0))

        max_seq_len: int = max(seq.shape[1] for seq in filtered_acts_list)
        padded_acts_list: list[t.Tensor] = [
            pad_activations(act, max_seq_len, accelerator)
            for act in filtered_acts_list
        ]
        new_acts: t.Tensor = t.cat(padded_acts_list, dim=0)
        t.save(new_acts, acts_path)

# Save new token ids.
new_token_ids: list = []
for sublist in filtered_token_ids:
    new_token_ids.append([sublist])
new_token_ids = np.array(new_token_ids, dtype=object)
np.save(PROMPT_IDS_PATH, new_token_ids, allow_pickle=True)
