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
NUM_SEQUENCES_EVALED = config.get("NUM_SEQUENCES_EVALED", 1000)
NUM_SEQUENCES_INTERPED = config.get("NUM_SEQUENCES_INTERPED")
SEED = config.get("SEED")

# %%
# Load the dataset
np.random.seed(SEED)
dataset: list[list[str]] = load_dataset(
    "Elriggs/openwebtext-100k",
    split="train",
)["text"]
dataset_indices: np.ndarray = np.random.choice(
    len(dataset),
    size=len(dataset),
    replace=False,
)

# %%
# Print the training datapoints
train_indices: np.ndarray = dataset_indices[:NUM_SEQUENCES_EVALED]
training_set: list[list[int]] = [dataset[i] for i in train_indices]

for datapoint in training_set:
    print(datapoint)

# %%
# Print the validation datapoints
STARTING_META_IDX: int = len(dataset) - NUM_SEQUENCES_INTERPED
eval_indices: np.ndarray = dataset_indices[STARTING_META_IDX:]
eval_set: list[list[str]] = [dataset[i] for i in eval_indices]

for datapoint in eval_set:
    print(datapoint)
