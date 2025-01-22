"""Run the constant-time graph pipeline in one command."""

import os
from subprocess import run

import pynvml


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

# 30712 was my benched VRAM draw for the full model slice "0:12", with 6538
# for the slice "10:12" and 8924 for the slice "9:12". So there's an
# initial intercept here of ~4600 and a layer-wise coefficient of ~2200,
# approaching from above.
PER_LAYER: float = 2200.0
PER_MODEL: float = 4600.0

layers: int = 12
vram_req: float = (layers * PER_LAYER) + PER_MODEL
adequate: bool = True
if total < vram_req:
    adequate: bool = False

if adequate:
    print("VRAM available sufficient for specified run.")
else:
    print("Dividing model slice to economize on VRAM use.")

for basename in [
    # "collect_acts.py",
    # "precluster.py",
    # "load_autoencoder.py",
    # "interp_tools/contexts.py",
    # "interp_tools/grad_graph.py",
]:
    path = f"{dirname}/{basename}"
    run(["python3", path], check=True)
