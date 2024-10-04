"""Run the constant-time graph pipeline in one command."""

import os

from subprocess import run

os.environ["WANDB_SILENT"] = "true"

for script in [
    "collect_acts.py",
    "precluster.py",
    "load_autoencoder.py",
    "interp_tools/contexts.py",
    "interp_tools/grad_graph.py",
]:
    run(["python3", script], check=True)
