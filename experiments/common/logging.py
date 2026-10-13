"""CSV/JSON logging (replaces ``fsrl.WandbLogger``); wandb optional in offline mode."""

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class CSVLog:
    """Append-only CSV with a fixed header (header taken from the first row).

    On resume, rows with ``step > resume_step`` are dropped so a run resumed from a
    checkpoint does not duplicate steps.
    """

    def __init__(self, path: Path, resume_step: Optional[int] = None):
        self.path = Path(path)
        self.fields: Optional[List[str]] = None
        if self.path.exists():
            if resume_step is None:
                self.path.unlink()
            else:
                self._truncate(resume_step)

    def _truncate(self, resume_step: int) -> None:
        with open(self.path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            self.path.unlink()
            return
        self.fields = list(rows[0].keys())
        keep = [r for r in rows if "step" not in r or float(r["step"]) <= resume_step]
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            w.writerows(keep)

    def write(self, row: Dict[str, Any]) -> None:
        row = {k: _fmt(v) for k, v in row.items()}
        new = not self.path.exists()
        if self.fields is None:
            self.fields = list(row.keys())
        else:
            extra = [k for k in row if k not in self.fields]
            if extra:  # widen the header (rare): rewrite file
                self._rewrite_with(self.fields + extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, restval="")
            if new:
                w.writeheader()
            w.writerow(row)

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        for r in rows:
            self.write(r)

    def _rewrite_with(self, fields: List[str]) -> None:
        rows = []
        if self.path.exists():
            with open(self.path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.fields = fields
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)


def _fmt(v: Any) -> Any:
    if hasattr(v, "item") and callable(v.item):
        try:
            v = v.item()
        except (ValueError, RuntimeError):
            pass
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return v


class StoreLogger:
    """Minimal ``fsrl`` logger interface (``store``) used by the OSRL trainers.

    Values are accumulated and averaged over a logging interval.
    """

    def __init__(self):
        self.data: Dict[str, List[float]] = defaultdict(list)

    def store(self, tab: Optional[str] = None, **kwargs) -> None:
        for k, v in kwargs.items():
            key = f"{tab}/{k}" if tab else k
            self.data[key].append(float(v.item() if hasattr(v, "item") else v))

    def means_and_reset(self) -> Dict[str, float]:
        out = {k.replace("/", "_"): float(np.mean(v)) for k, v in self.data.items() if v}
        self.data.clear()
        return out

    # fsrl compatibility no-ops
    def write(self, *a, **k) -> None:
        pass

    def write_without_reset(self, *a, **k) -> None:
        pass


class WandbOffline:
    """Optional wandb mirror (offline mode only; enabled with CDX_WANDB=1)."""

    def __init__(self, run_dir: Path, name: str, config: Dict[str, Any]):
        self.run = None
        if os.environ.get("CDX_WANDB", "0") != "1":
            return
        try:
            import wandb

            os.environ.setdefault("WANDB_MODE", "offline")
            self.run = wandb.init(project="cdx", name=name, dir=str(run_dir), config=config,
                                  mode="offline", reinit=True)
        except Exception:  # noqa: BLE001 - logging must never kill a run
            self.run = None

    def log(self, row: Dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log({k: v for k, v in row.items() if isinstance(v, (int, float))},
                         step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
