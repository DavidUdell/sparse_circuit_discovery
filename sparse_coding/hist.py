"""Run the histograms pipeline in one command."""

import os

from subprocess import run

from sparse_coding.utils.interface import load_yaml_constants

# export WANDB_MODE, if set in config
_, config = load_yaml_constants(__file__)
WANDB_MODE = config.get("WANDB_MODE")
if WANDB_MODE:
    os.environ["WANDB_MODE"] = WANDB_MODE

for script in [
    "collect_acts",
    "load_autoencoder",
    "histograms",
]:
    run(["python3", script], check=True)
