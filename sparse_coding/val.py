# %%
"""Validate circuits with one command."""


from runpy import run_module


for script in [
    "load_autoencoder",
    "interp_tools.validate_circuits",
]:
    run_module(f"sparse_coding.{script}")
