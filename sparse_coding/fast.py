"""Run the constant-time graph pipeline in one command."""

import os

from subprocess import run

from sparse_coding.utils.interface import load_yaml_constants

os.environ["WANDB_SILENT"] = "true"

# export WANDB_MODE, if set in config
_, config = load_yaml_constants(__file__)
WANDB_MODE = config.get("WANDB_MODE")
if WANDB_MODE:
    os.environ["WANDB_MODE"] = WANDB_MODE

for script in [
    "collect_acts.py",
    "precluster.py",
    "load_autoencoder.py",
    "interp_tools/contexts.py",
    "interp_tools/grad_graph.py",
]:
    run(["python3", script], check=True)
