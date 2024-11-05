"""Run the main sparse coding pipeline in one command."""

import os
from textwrap import dedent

from runpy import run_module

from sparse_coding.utils.interface import load_yaml_constants


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

# export WANDB_MODE, if set in config
_, config = load_yaml_constants(__file__)
WANDB_MODE = config.get("WANDB_MODE")
if WANDB_MODE:
    os.environ["WANDB_MODE"] = WANDB_MODE

for script in [
    "collect_acts",
    "load_autoencoder",
    "interp_tools.contexts",
    "interp_tools.cognition_graph",
]:
    try:
        run_module(f"sparse_coding.{script}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error at script {script}: {e}")
