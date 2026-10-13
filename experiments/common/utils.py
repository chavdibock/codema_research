"""Small helpers: seeding, run directories, markers, atomic writes, provenance."""

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]

DONE = "DONE"
RUNNING = "RUNNING"
FAILED = "FAILED"


def seed_all(seed: int) -> None:
    """Same seeding as ``osrl.common.exp_util.seed_all`` (kept identical for fidelity)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_dir_for(out: str, exp: str, task: str, model_id: str, variant: Optional[str],
                seed: int) -> Path:
    """results/<exp>/<task>/<model>[__<variant>]/seed_<s>/ (spec §8.4)."""
    name = model_id if not variant else f"{model_id}__{variant}"
    return Path(out) / exp / task / name / f"seed_{seed}"


def is_done(run_dir: Path) -> bool:
    return (Path(run_dir) / DONE).exists()


def write_marker(run_dir: Path, name: str, content: str = "") -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_text(Path(run_dir) / name, content)


def remove_marker(run_dir: Path, name: str) -> None:
    try:
        (Path(run_dir) / name).unlink()
    except FileNotFoundError:
        pass


def job_tag() -> str:
    """Host + SLURM ids, written into RUNNING/FAILED markers."""
    job = os.environ.get("SLURM_JOB_ID", "-")
    arr = os.environ.get("SLURM_ARRAY_TASK_ID", "-")
    return f"host={socket.gethostname()} pid={os.getpid()} job={job} array={arr}"


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def atomic_write_yaml(path: Path, obj: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(_to_builtin(obj), sort_keys=False))


def atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _to_builtin(o: Any) -> Any:
    if isinstance(o, str):
        return str(o)  # e.g. torch.__version__ is a str subclass yaml cannot represent
    if isinstance(o, bool) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_builtin(v) for v in o]
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return o


def stable_hash(obj: Any, n: int = 16) -> str:
    blob = json.dumps(_to_builtin(obj), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:n]


def experiments_code_hash() -> str:
    """git hash of experiments/ if it is a repository, else a content hash of *.py files.

    NOTE: the spec asks for the git hash of experiments/; the folder is not a git
    repository yet, so we fall back to a sha1 over all tracked source files.
    """
    try:
        out = subprocess.run(["git", "-C", str(EXPERIMENTS_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            dirty = subprocess.run(
                ["git", "-C", str(EXPERIMENTS_ROOT), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    h = hashlib.sha1()
    for p in sorted(EXPERIMENTS_ROOT.rglob("*")):
        if p.suffix in {".py", ".sh", ".sbatch"} and "results" not in p.parts:
            h.update(str(p.relative_to(EXPERIMENTS_ROOT)).encode())
            h.update(p.read_bytes())
    return "content-" + h.hexdigest()[:16]


_PIP_FREEZE_CACHE: Optional[str] = None


def pip_freeze() -> str:
    global _PIP_FREEZE_CACHE
    if _PIP_FREEZE_CACHE is None:
        try:
            _PIP_FREEZE_CACHE = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                timeout=120).stdout
        except (OSError, subprocess.SubprocessError):
            _PIP_FREEZE_CACHE = "unavailable"
    return _PIP_FREEZE_CACHE


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpus_allowed": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    try:
        import mamba_ssm  # noqa

        info["mamba_ssm"] = mamba_ssm.__version__
    except Exception as e:  # noqa: BLE001 - any import failure is just recorded
        info["mamba_ssm"] = f"unavailable ({type(e).__name__})"
    return info


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)
