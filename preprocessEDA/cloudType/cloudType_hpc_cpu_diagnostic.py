
#!/usr/bin/env python3
"""
Diagnose HPC CPU resources and the Python environment for CPU-based TabM.

Outputs are saved only under:
eda_results/cloudType/hpc_diagnostics/
"""

import os
import sys
import json
import platform
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("eda_results/cloudType/hpc_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report = {
    "diagnostic_time": datetime.now().isoformat(),
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_version": sys.version,
    "python_executable": sys.executable,
    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    "cpu": {},
    "memory": {},
    "scheduler": {},
    "software": {},
    "tabm": {},
}


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def run_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------
# CPU information
# ---------------------------------------------------------

report["cpu"]["logical_cpu_count"] = os.cpu_count()

try:
    report["cpu"]["available_cpu_affinity"] = sorted(
        os.sched_getaffinity(0)
    )
    report["cpu"]["affinity_count"] = len(
        report["cpu"]["available_cpu_affinity"]
    )
except (AttributeError, OSError):
    report["cpu"]["available_cpu_affinity"] = None
    report["cpu"]["affinity_count"] = None

cpuinfo = read_text("/proc/cpuinfo")
if cpuinfo:
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model name"):
            report["cpu"]["model_name"] = line.split(":", 1)[1].strip()
            break

# ---------------------------------------------------------
# Memory information
# ---------------------------------------------------------

meminfo = read_text("/proc/meminfo")
if meminfo:
    for line in meminfo.splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:")):
            key, value = line.split(":", 1)
            report["memory"][key] = value.strip()

# Check cgroup memory limits where available
for path in [
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
]:
    value = read_text(path)
    if value:
        report["memory"]["cgroup_memory_limit"] = value
        break

# ---------------------------------------------------------
# HPC scheduler information
# ---------------------------------------------------------

scheduler_variables = [
    "PBS_JOBID",
    "PBS_NODEFILE",
    "PBS_O_WORKDIR",
    "PBS_QUEUE",
    "PBS_NP",
    "PBS_NUM_PPN",
    "SLURM_JOB_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_MEM_PER_NODE",
]

for variable in scheduler_variables:
    value = os.environ.get(variable)
    if value:
        report["scheduler"][variable] = value

nodefile = os.environ.get("PBS_NODEFILE")
if nodefile and Path(nodefile).exists():
    try:
        nodes = Path(nodefile).read_text().splitlines()
        report["scheduler"]["pbs_nodefile_entries"] = len(nodes)
        report["scheduler"]["pbs_allocated_nodes"] = sorted(set(nodes))
    except Exception as exc:
        report["scheduler"]["pbs_nodefile_error"] = str(exc)

# ---------------------------------------------------------
# Software versions
# ---------------------------------------------------------

try:
    import numpy as np
    report["software"]["numpy"] = np.__version__
except Exception as exc:
    report["software"]["numpy_error"] = str(exc)

try:
    import pandas as pd
    report["software"]["pandas"] = pd.__version__
except Exception as exc:
    report["software"]["pandas_error"] = str(exc)

try:
    import torch

    report["software"]["pytorch"] = torch.__version__
    report["software"]["torch_threads"] = torch.get_num_threads()
    report["software"]["torch_interop_threads"] = (
        torch.get_num_interop_threads()
    )
    report["software"]["cuda_available"] = torch.cuda.is_available()
    report["software"]["cuda_build"] = torch.version.cuda

    # Do not initialize or use a GPU.
    torch.set_num_threads(
        report["cpu"]["affinity_count"]
        or report["cpu"]["logical_cpu_count"]
        or 1
    )

    report["software"]["torch_threads_after_check"] = (
        torch.get_num_threads()
    )

except Exception as exc:
    report["software"]["pytorch_error"] = str(exc)

# ---------------------------------------------------------
# TabM package and API inspection
# ---------------------------------------------------------

try:
    spec = importlib.util.find_spec("tabm")
    report["tabm"]["package_found"] = spec is not None

    if spec is not None:
        import tabm

        report["tabm"]["module_file"] = getattr(
            tabm, "__file__", None
        )
        report["tabm"]["module_members"] = [
            name for name in dir(tabm)
            if not name.startswith("_")
        ]

        try:
            from importlib.metadata import version
            report["tabm"]["package_version"] = version("tabm")
        except Exception as exc:
            report["tabm"]["version_lookup"] = str(exc)

        if hasattr(tabm, "TabM"):
            import inspect
            report["tabm"]["TabM_make_signature"] = str(
                inspect.signature(tabm.TabM.make)
            )
        else:
            report["tabm"]["TabM_class_found"] = False

except Exception as exc:
    report["tabm"]["import_error"] = str(exc)

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

json_path = OUTPUT_DIR / "hpc_cpu_diagnostic.json"
text_path = OUTPUT_DIR / "hpc_cpu_diagnostic.txt"

with open(json_path, "w") as f:
    json.dump(report, f, indent=4, default=str)

with open(text_path, "w") as f:
    for section, values in report.items():
        f.write(f"\n[{section}]\n")
        if isinstance(values, dict):
            for key, value in values.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(f"{values}\n")

print("\n========== HPC CPU DIAGNOSTIC ==========")
print(json.dumps(report, indent=4, default=str))
print("\n========================================")
print(f"JSON report: {json_path}")
print(f"Text report: {text_path}")
