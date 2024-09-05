"""Run the constant-time graph pipeline in one command."""

from subprocess import run

for script in [
    "collect_acts.py",
    "load_autoencoder.py",
    "interp_tools/contexts.py",
    "interp_tools/grad_graph.py",
]:
    run(["python3", script], check=True)
