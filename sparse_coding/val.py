# %%
"""Validate circuits with one command."""

import gc
import os
import sys

from runpy import run_module


os.environ["WANDB_SILENT"] = "true"

# Run from any pwd
path_to_dir: str = os.path.dirname(__file__)
if path_to_dir not in sys.path:
    sys.path.insert(0, path_to_dir)

for submodule in [
    "collect_acts",
    "load_autoencoder",
    "interp_tools.contexts",
]:
    run_module(f"sparse_coding.{submodule}")

# Run final script separately, after manually clearing memory
gc.collect()
run_module("sparse_coding.interp_tools.validate_circuits")
