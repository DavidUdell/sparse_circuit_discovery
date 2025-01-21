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
print("NVIDIA VRAM:", total)
vram_adequate: bool = True
if total < 31000:
    # 30712 was my benched VRAM draw for the full model slice "0:12".
    vram_adequate: bool = False
if not vram_adequate:
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
