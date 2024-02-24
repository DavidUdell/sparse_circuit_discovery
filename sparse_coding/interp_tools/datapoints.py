"""
Print out the validation set datapoints.

The idea is that you want to get a hand feel for what the datapoints look like,
so that you understand the distribution the interp data is indexed by.
"""

import numpy as np
from datasets import load_dataset

from sparse_coding.utils.interface import load_yaml_constants


# %%
# Load constants
_, config = load_yaml_constants(__file__)
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")

# %%
# Load the validation set
dataset: list[list[str]] = load_dataset(
    "Elriggs/openwebtext-100k",
    split="train",
)["text"]
dataset_indices: np.ndarray = np.random.choice(
    len(dataset),
    size=len(dataset),
    replace=False,
)
STARTING_META_IDX: int = len(dataset) - NUM_SEQUENCES_INTERPED
eval_indices: np.ndarray = dataset_indices[STARTING_META_IDX:]
eval_set: list[list[str]] = [dataset[i] for i in eval_indices]

# %%
# Print the validation datapoints
for datapoint in eval_set:
    print(datapoint)
