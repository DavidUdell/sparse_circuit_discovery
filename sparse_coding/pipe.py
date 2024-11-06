"""Run the main sparse coding pipeline in one command."""

import os
import sys
from textwrap import dedent

from runpy import run_module


os.environ["WANDB_SILENT"] = "true"

# Run from any pwd
path_to_dir: str = os.path.dirname(__file__)
if path_to_dir not in sys.path:
    sys.path.insert(0, path_to_dir)

print(
    dedent(
        """
        For the time being,
        1. `model_dir` must be `openai-community/gpt2`,
        2. `projection_factor` must be 32, and
        3. only ablation studies are performed and measured.
        """
    )
)

for submodule in [
    "collect_acts",
    "load_autoencoder",
    "interp_tools.contexts",
    "interp_tools.cognition_graph",
]:
    run_module(f"sparse_coding.{submodule}")
