# %%
"""Validate circuits with one command."""


from runpy import run_module


for script in [
    "collect_acts",
    "load_autoencoder",
    "interp_tools.contexts",
    "interp_tools.validate_circuits",
]:
    run_module(f"sparse_coding.{script}")
