"""Run the constant-time graph pipeline in one command."""

import os
from math import floor
from subprocess import run
from textwrap import dedent

import pynvml

from sparse_coding.utils.interface import load_yaml_constants, parse_slice


os.environ["WANDB_SILENT"] = "true"


# Run from any pwd
dirname: list[str] = __file__.split("/")[:-1]
dirname: str = "/".join(dirname)

# Check NVIDIA GPU VRAM
pynvml.nvmlInit()
num_devices = pynvml.nvmlDeviceGetCount()
stats = []
for idx in range(num_devices):
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    vram_total: int = memory.total
    stats.append(vram_total)
pynvml.nvmlShutdown()

total: float = 0.0
for m in stats:
    MiB: float = m / 1024**2
    total += MiB
print("NVIDIA VRAM:", round(total), "MiB")

# 30712 was my benched VRAM draw for the full model slice "0:12", with 6538 for
# the slice "10:12" and 8924 for the slice "9:12". So there's an initial
# intercept here of ~4600 and a pair-wise coefficient of ~2200, approaching
# from above.
PER_PAIRING: float = 2200.0
INTERCEPT: float = 4600.0

_, config = load_yaml_constants(__file__)
layers_slice = parse_slice(config.get("ACTS_LAYERS_SLICE"))
pairings: int = (layers_slice.stop - 1) - layers_slice.start
vram_req: float = (pairings * PER_PAIRING) + INTERCEPT
adequate: bool = True
if total < vram_req:
    adequate: bool = False

if adequate:
    print("VRAM available sufficient for specified run.")

else:
    # Dividing into halves always suffices
    halfway: int = floor((layers_slice.start + layers_slice.stop) / 2)
    print(
        dedent(
            f"""
            Dividing model slice into {layers_slice.start}:{halfway + 1} and
            {halfway}:{layers_slice.stop} to economize on VRAM use.
            """
        )
    )
    os.environ["ACTS_LAYERS_SLICE"] = f"{halfway}:{layers_slice.stop}"

for basename in [
    "collect_acts.py",
    "precluster.py",
    "load_autoencoder.py",
    "interp_tools/contexts.py",
    "interp_tools/grad_graph.py",
]:
    path = f"{dirname}/{basename}"
    run(["python3", path], check=True)

# Run second pass when needed
if not adequate:
    path = f"{dirname}/interp_tools/grad_graph.py"
    os.environ["ACTS_LAYERS_SLICE"] = f"{layers_slice.start}:{halfway + 1}"
    run(["python3", path], check=True)

# Cleanup env variable
_ = os.environ.pop("ACTS_LAYERS_SLICE", None)
