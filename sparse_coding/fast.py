"""Run the constant-time graph pipeline in one command."""

import os

from subprocess import run


os.environ["WANDB_SILENT"] = "true"

# Run from any pwd
prepend: list[str] = __file__.split("/")[:-1]
prepend: str = "/".join(prepend)

for append in [
    "collect_acts.py",
    "precluster.py",
    "load_autoencoder.py",
    "interp_tools/contexts.py",
    "interp_tools/grad_graph.py",
]:
    path = f"{prepend}/{append}"
    run(["python3", path], check=True)
