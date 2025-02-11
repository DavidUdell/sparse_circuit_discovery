# %%
"""
Dict learning on an activations dataset, with a basic autoencoder.

The script will save the trained encoder matrix to disk; that encoder matrix
is your learned dictionary.
"""


import os
import warnings

import numpy as np
import torch as t
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from sparse_coding.utils.interface import (
    parse_slice,
    slice_to_range,
    sanitize_model_name,
    cache_layer_tensor,
    load_input_token_ids,
    load_yaml_constants,
    save_paths,
)


assert t.__version__ >= "2.0.1", "`Lightning` requires newer `torch` versions."
# If your training runs are hanging, be sure to update `transformers` too. Just
# update everything the script uses and try again.

# %%
# Set up constants. Drive towards an L_0 of 10-100 at convergence.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
WANDB_PROJECT = config.get("WANDB_PROJECT")
WANDB_ENTITY = config.get("WANDB_ENTITY")
WANDB_MODE = config.get("WANDB_MODE")
SEED = config.get("SEED")
ACTS_DATA_FILE = config.get("ACTS_DATA_FILE")
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
BIASES_FILE = config.get("BIASES_FILE")
ENCODER_FILE = config.get("ENCODER_FILE")
MODEL_DIR = config.get("MODEL_DIR")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
# Float casts fix YAML bug with scientific notation.
LAMBDA_L1 = float(config.get("LAMBDA_L1"))
LEARNING_RATE = float(config.get("LEARNING_RATE"))
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
tsfm_config = AutoConfig.from_pretrained(MODEL_DIR, token=HF_ACCESS_TOKEN)
EMBEDDING_DIM = tsfm_config.hidden_size
PROJECTION_DIM = int(EMBEDDING_DIM * PROJECTION_FACTOR)
NUM_WORKERS = config.get("NUM_WORKERS")
LOG_EVERY_N_STEPS = config.get("LOG_EVERY_N_STEPS", 5)
EPOCHS = config.get("EPOCHS", 150)
SYNC_DIST_LOGGING = config.get("SYNC_DIST_LOGGING", True)
# For smaller autoencoders, larger batch sizes are possible.
TRAINING_BATCH_SIZE = config.get("TRAINING_BATCH_SIZE", 16)
ACCUMULATE_GRAD_BATCHES = config.get("ACCUMULATE_GRAD_BATCHES", 1)

if WANDB_MODE:
    os.environ["WANDB_MODE"] = WANDB_MODE

# %%
# Use available tensor cores.
t.set_float32_matmul_precision("medium")


# %%
# Create a padding mask.
def padding_mask(
    activations_block: t.Tensor, unpadded_prompts: list[list[str]]
) -> t.Tensor:
    """Create a padding mask for the activations block."""
    masks: list = []

    for unpadded_prompt in unpadded_prompts:
        original_stream_length: int = len(unpadded_prompt)
        # The mask will drop the embedding dimension.
        mask: t.Tensor = t.zeros(
            (activations_block.size(1),),
            dtype=t.bool,
        )
        mask[:original_stream_length] = True
        masks.append(mask)

    # `masks` is of shape (batch, stream_dim).
    masks: t.Tensor = t.stack(masks, dim=0)
    return masks


# %%
# Define a `torch` dataset.
class ActivationsDataset(Dataset):
    """Dataset of hidden states from a pretrained model."""

    def __init__(self, tensor_data: t.Tensor, mask: t.Tensor):
        """Constructor; inherits from `torch.utils.data.Dataset` class."""
        self.data = tensor_data
        self.mask = mask

    def __len__(self):
        """Return the dataset length."""
        return len(self.data)

    def __getitem__(self, indx):
        """Return the item at the passed index."""
        return self.data[indx], self.mask[indx]


# %%
# Set up and run training and validation.
def train_autoencoder() -> None:
    """Train an autoencoder on activations, from constants."""

    training_loader: DataLoader = DataLoader(
        dataset,
        batch_size=TRAINING_BATCH_SIZE,
        sampler=training_sampler,
        num_workers=NUM_WORKERS,
    )
    validation_loader: DataLoader = DataLoader(
        dataset,
        batch_size=TRAINING_BATCH_SIZE,
        sampler=validation_sampler,
        num_workers=NUM_WORKERS,
    )
    # The `accumulate_grad_batches` argument helps with memory on the largest
    # autoencoders. I don't currently do anything about "dead neurons," as
    # Bricken et al. 2023 found and discussed.
    trainer: L.Trainer = L.Trainer(
        accelerator="auto",
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        callbacks=early_stop,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=logger,
        max_epochs=EPOCHS,
    )

    trainer.fit(
        model,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
    )


# %%
# Input token ids are constant across model layers.
unpacked_prompts_ids: list[list[int]] = load_input_token_ids(PROMPT_IDS_PATH)

# %%
# Loop over the layer_idx values in the model slice.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        token=HF_ACCESS_TOKEN,
    )
seq_layer_indices: range = slice_to_range(hf_model, ACTS_LAYERS_SLICE)

for layer_idx in seq_layer_indices:
    # Load, preprocess, and split an activations dataset.
    DATASET_PATH = save_paths(
        __file__,
        f"{sanitize_model_name(MODEL_DIR)}/{layer_idx}/{ACTS_DATA_FILE}",
    )
    padded_acts_block = t.load(DATASET_PATH)
    pad_mask: t.Tensor = padding_mask(padded_acts_block, unpacked_prompts_ids)

    dataset: ActivationsDataset = ActivationsDataset(
        padded_acts_block,
        pad_mask,
    )

    training_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=SEED,
    )

    training_sampler = t.utils.data.SubsetRandomSampler(training_indices)
    validation_sampler = t.utils.data.SubsetRandomSampler(val_indices)

    # Define a tied autoencoder, with `lightning`.
    class Autoencoder(L.LightningModule):
        """An autoencoder architecture."""

        def __init__(
            self, lr=LEARNING_RATE
        ):  # pylint: disable=unused-argument
            super().__init__()
            self.save_hyperparameters()
            self.encoder = t.nn.Sequential(
                t.nn.Linear(EMBEDDING_DIM, PROJECTION_DIM, bias=True),
                t.nn.ReLU(),
            )

            # Orthogonal initialization.
            t.nn.init.orthogonal_(self.encoder[0].weight.data)

            self.total_activity = t.zeros(PROJECTION_DIM)

        def forward(self, state):  # pylint: disable=arguments-differ
            """The forward pass of an autoencoder for activations."""
            encoded_state = self.encoder(state)

            # Decode the sampled state.
            decoder_weights = self.encoder[0].weight.data.T
            output_state = (
                t.nn.functional.linear(  # pylint: disable=not-callable
                    encoded_state, decoder_weights
                )
            )

            return encoded_state, output_state

        def training_step(self, batch):  # pylint: disable=arguments-differ
            """Train the autoencoder."""
            data, mask = batch
            data_mask = mask.unsqueeze(-1).expand_as(data)
            masked_data = data * data_mask

            encoded_state, output_state = self.forward(masked_data)

            pass_activity: t.Tensor = encoded_state.sum((0, 1))
            self.total_activity += pass_activity.to(self.total_activity.device)
            total_inactive = (self.total_activity == 0.0).sum().item()

            pass_frac_inactive = (
                pass_activity == 0.0
            ).sum().item() / PROJECTION_DIM
            total_frac_inactive: float = total_inactive / PROJECTION_DIM

            print(f"pass_frac_inactive: {round(pass_frac_inactive, 2)}\n")
            print(f"total_frac_inactive: {round(total_frac_inactive, 2)}\n")
            self.log(
                "fraction neurons inactive during pass",
                pass_frac_inactive,
                sync_dist=SYNC_DIST_LOGGING,
            )
            self.log(
                "fraction neurons never activated",
                total_frac_inactive,
                sync_dist=SYNC_DIST_LOGGING,
            )

            # The mask excludes the padding tokens from consideration.
            mse_loss = t.nn.functional.mse_loss(output_state, masked_data)
            l1_loss = t.nn.functional.l1_loss(
                encoded_state,
                t.zeros_like(encoded_state),
            )

            training_loss = mse_loss + (LAMBDA_L1 * l1_loss)
            l0_sparsity = (
                (encoded_state != 0).float().sum(dim=-1).mean().item()
            )
            print(f"L^0: {round(l0_sparsity, 2)}\n")
            self.log(
                "training loss", training_loss, sync_dist=SYNC_DIST_LOGGING
            )
            print(f"t_loss: {round(training_loss.item(), 2)}\n")
            self.log(
                "L1 component",
                LAMBDA_L1 * l1_loss,
                sync_dist=SYNC_DIST_LOGGING,
            )
            self.log("MSE component", mse_loss, sync_dist=SYNC_DIST_LOGGING)
            self.log("L0 sparsity", l0_sparsity, sync_dist=SYNC_DIST_LOGGING)
            return training_loss

        # Unused import resolves `lightning` bug.
        def validation_step(
            self, batch, batch_idx
        ):  # pylint: disable=unused-argument,arguments-differ
            """Validate the autoencoder."""
            data, mask = batch
            data_mask = mask.unsqueeze(-1).expand_as(data)
            masked_data = data * data_mask

            encoded_state, output_state = self.forward(masked_data)

            mse_loss = t.nn.functional.mse_loss(output_state, masked_data)
            l1_loss = t.nn.functional.l1_loss(
                encoded_state,
                t.zeros_like(encoded_state),
            )
            validation_loss = mse_loss + (LAMBDA_L1 * l1_loss)

            self.log(
                "validation loss", validation_loss, sync_dist=SYNC_DIST_LOGGING
            )
            return validation_loss

        def configure_optimizers(self):
            """Configure the `Adam` optimizer."""
            return t.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # Validation-loss-based early stopping.
    early_stop = L.pytorch.callbacks.EarlyStopping(
        monitor="validation loss",
        min_delta=1e-5,
        patience=3,
        verbose=False,
        mode="min",
    )

    # Train the autoencoder. Note that `lightning` does its own
    # parallelization.
    model: Autoencoder = Autoencoder()
    logger = L.pytorch.loggers.WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=config,
    )

    try:
        train_autoencoder()

    except RuntimeError:
        # `accelerate` does not degrade gracefully as you scale down to small
        # models. This global change fixes that from here on in the loop.
        NUM_WORKERS: int = 0
        ACCUMULATE_GRAD_BATCHES: int = 1

        train_autoencoder()

    # Save the trained encoder weights and biases.
    cache_layer_tensor(
        model.encoder[0].weight.data,
        layer_idx,
        ENCODER_FILE,
        __file__,
        MODEL_DIR,
    )

    cache_layer_tensor(
        model.encoder[0].bias.data,
        layer_idx,
        BIASES_FILE,
        __file__,
        MODEL_DIR,
    )
