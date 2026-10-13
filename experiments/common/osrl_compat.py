"""Access to the unmodified OSRL checkout: per-task configs, commit hash.

OSRL is installed with ``pip install -e OSRL --no-deps``. Its ``examples`` package
(which holds the per-task hyperparameters) is excluded from the installed packages,
so we put the checkout root on ``sys.path`` to import it. Nothing in OSRL is modified.
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

OSRL_ROOT = Path(os.environ.get("OSRL_ROOT",
                                Path(__file__).resolve().parents[2] / "OSRL"))

# algo id -> (config module, default-config dict name)
_ALGO_CONFIGS = {
    "cdt": ("examples.configs.cdt_configs", "CDT_DEFAULT_CONFIG"),
    "bc": ("examples.configs.bc_configs", "BC_DEFAULT_CONFIG"),
    "bcql": ("examples.configs.bcql_configs", "BCQL_DEFAULT_CONFIG"),
    "bearl": ("examples.configs.bearl_configs", "BEARL_DEFAULT_CONFIG"),
    "cpq": ("examples.configs.cpq_configs", "CPQ_DEFAULT_CONFIG"),
    "coptidice": ("examples.configs.coptidice_configs", "COptiDICE_DEFAULT_CONFIG"),
}


def ensure_osrl_examples_importable() -> None:
    """Make ``import examples.configs...`` resolve to the OSRL checkout."""
    root = str(OSRL_ROOT)
    if not (OSRL_ROOT / "examples" / "configs").is_dir():
        raise FileNotFoundError(
            f"OSRL checkout not found at {OSRL_ROOT}; set OSRL_ROOT to the OSRL directory.")
    if root not in sys.path:
        sys.path.append(root)


def osrl_task_config(algo: str, task: str) -> Dict[str, Any]:
    """Return OSRL's default per-task config for ``algo`` as a plain dict."""
    import importlib

    ensure_osrl_examples_importable()
    module_name, dict_name = _ALGO_CONFIGS[algo]
    module = importlib.import_module(module_name)
    table = getattr(module, dict_name)
    if task not in table:
        raise KeyError(f"OSRL has no {algo} config for task {task}")
    return asdict(table[task]())


def osrl_commit() -> str:
    """Commit hash of the OSRL checkout, read from .git without calling git."""
    git_dir = OSRL_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        ref_file = git_dir / ref
        if ref_file.exists():
            return ref_file.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(ref):
                    return line.split(" ", 1)[0]
    except OSError:
        pass
    return "unknown"
