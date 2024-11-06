"""Run the histograms pipeline in one command."""

import os

from subprocess import run


os.environ["WANDB_SILENT"] = "true"

# Run from any pwd
dirname: list[str] = __file__.split("/")[:-1]
dirname: str = "/".join(dirname)

for basename in [
    "collect_acts.py",
    "load_autoencoder.py",
    "histograms.py",
]:
    path: str = f"{dirname}/{basename}"
    run(["python3", path], check=True)
