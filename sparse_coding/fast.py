"""Run the constant-time graph pipeline in one command."""

from runpy import run_module

for script in [
    "collect_acts",
    "load_autoencoder",
    "interp_tools.contexts",
    "interp_tools.grad_graph",
]:
    run_module(f"sparse_coding.{script}")
