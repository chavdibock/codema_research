"""The E0 gate as a file: $CDX_RESULTS/decisions.yaml (spec §6 E0, §4.8 S3).

    python -m experiments.snelius.decisions --out $CDX_RESULTS            # print the proposal
    python -m experiments.snelius.decisions --out $CDX_RESULTS --write    # and write it

The file is written once, before the first non-E0 run starts, and never changed after
that (``--force`` overwrites; do that only before any main run has started). Every
non-E0 run applies it in ``resolve_run_config``, so ``config.yaml`` records the values.

Rules (all from the E0 runs; nothing else is read):
  * precision (E0prec, cdt Car-Run seed 0, kappa=10): keep bf16 if the bf16 - fp32
    difference of the mean R_norm and of the mean C_norm are both within 2 standard errors
    of the paired per-episode differences (same env seeds); otherwise fp32.
    NOTE: the spec says "within seed-to-seed noise", which one seed cannot measure. The
    paired episode-level noise is smaller than seed-to-seed noise, so this rule falls back
    to fp32 more often than the spec's rule would (conservative).
  * learning rate per backbone (E0tune, validation seeds): among the lr variants with
    C_norm <= 1 the highest R_norm; if none is safe, the lowest C_norm (spec E0).
  * pack size (E0pack): the pack size with the highest total training steps/s among the
    sizes whose 9 runs all finished.
The gate fails (main runs do not start) if any run of E0, E0prec, E0tune or E0pack is not
DONE. E0fid is informative only and is not part of the gate.

$CDX_RESULTS/platform.yaml (written by bootstrap.py) holds per-model overrides that apply
to every experiment, e.g. the Mamba-3 reference fallback when the CUDA kernels do not run.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DECISIONS_FILE = "decisions.yaml"
PLATFORM_FILE = "platform.yaml"
GATE_EXPS = ("E0", "E0prec", "E0tune", "E0pack")


# ------------------------------------------------------------------ files
def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    import yaml

    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_decisions(out: str) -> Optional[Dict[str, Any]]:
    return _read_yaml(Path(out) / DECISIONS_FILE)


def load_platform(out: str) -> Optional[Dict[str, Any]]:
    return _read_yaml(Path(out) / PLATFORM_FILE)


def overrides_for(out: str, exp: str, model_id: str, model_kind: str) -> Dict[str, Any]:
    """Overrides from platform.yaml (every experiment) and decisions.yaml (non-E0 only)."""
    ov: Dict[str, Any] = {}
    plat = load_platform(out)
    if plat:
        ov.update((plat.get("model_overrides") or {}).get(model_id, {}))
    dec = load_decisions(out)
    if dec and not exp.startswith("E0"):
        if model_kind == "sequence" and dec.get("precision"):
            ov["train.precision"] = dec["precision"]
        lr = (dec.get("learning_rate") or {}).get(model_id)
        if lr is not None:
            ov["train.learning_rate"] = float(lr)
    return ov


# ------------------------------------------------------------------ rules
def _episodes(run_dir: Path):
    import pandas as pd

    p = run_dir / "eval_episodes.csv"
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def _rdir(out: Path, exp: str, task: str, model: str, variant: Optional[str], seed: int) -> Path:
    from experiments.common.utils import run_dir_for

    return run_dir_for(str(out), exp, task, model, variant, seed)


def gate_status(out: Path) -> Tuple[bool, List[str]]:
    """(all gate runs DONE, list of run dirs that are not)."""
    from experiments.common.expdefs import enumerate_runs
    from experiments.common.utils import is_done

    missing = []
    for exp in GATE_EXPS:
        for r in enumerate_runs(exp):
            d = _rdir(out, r.exp, r.task, r.model, r.variant, r.seed)
            if not is_done(d):
                missing.append(str(d))
    return not missing, missing


def decide_precision(out: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    from experiments.common.expdefs import get_exp

    e = get_exp("E0prec")
    task, model, seed = e.tasks[0], e.models[0], e.seeds[0]
    ep = {v: _episodes(_rdir(out, "E0prec", task, model, v, seed)) for v in ("fp32", "bf16")}
    if any(len(d) == 0 for d in ep.values()):
        return None, {"reason": "E0prec episodes missing"}
    a, b = (ep[v][(ep[v]["target_C"] == 10) & (ep[v]["split"] == "test")]
            .set_index("env_seed") for v in ("fp32", "bf16"))
    common = a.index.intersection(b.index)
    ev: Dict[str, Any] = {"n_paired_episodes": int(len(common))}
    within = True
    for m in ("R_norm", "C_norm"):
        d = (b.loc[common, m] - a.loc[common, m]).to_numpy(dtype=float)
        mean = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        ok = bool(abs(mean) <= 2 * se) if np.isfinite(se) else mean == 0.0
        ev[m] = {"fp32": float(a.loc[common, m].mean()), "bf16": float(b.loc[common, m].mean()),
                 "diff": mean, "se": se, "within_2se": ok}
        within &= ok
    return ("bf16" if within else "fp32"), ev


def decide_learning_rates(out: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    from experiments.common.expdefs import get_exp

    e = get_exp("E0tune")
    task, seed = e.tasks[0], e.seeds[0]
    picks, ev = {}, {}
    for m in e.models:
        rows = []
        for v, ov in e.variants.items():
            d = _episodes(_rdir(out, "E0tune", task, m, v, seed))
            if len(d):
                rows.append((v, float(ov["train.learning_rate"]), float(d["R_norm"].mean()),
                             float(d["C_norm"].mean())))
        if not rows:
            continue
        safe = [r for r in rows if r[3] <= 1.0]
        best = max(safe, key=lambda r: r[2]) if safe else min(rows, key=lambda r: r[3])
        picks[m] = best[1]
        ev[m] = {r[0]: {"R_norm": r[2], "C_norm": r[3]} for r in rows}
        ev[m]["picked"] = best[0]
    return picks, ev


def decide_pack(out: Path) -> Tuple[Optional[int], Dict[str, Any]]:
    import pandas as pd

    from experiments.common.expdefs import get_exp
    from experiments.common.utils import is_done

    e = get_exp("E0pack")
    task, model = e.tasks[0], e.models[0]
    ev = {}
    for v in e.variants:
        dirs = [_rdir(out, "E0pack", task, model, v, s) for s in e.seeds]
        if not all(is_done(d) for d in dirs):
            ev[v] = {"complete": False}
            continue
        steps, wall = 0, []
        for d in dirs:
            steps += int(pd.read_csv(d / "train_log.csv")["step"].max())
            with open(d / "run_summary.json") as f:
                wall.append(float(json.load(f).get("train_wall_s", np.nan)))
        # p concurrent runs: total throughput = total steps / mean wall time of one run
        ev[v] = {"complete": True, "total_steps_per_s": steps / float(np.nanmean(wall))}
    ok = {v: x["total_steps_per_s"] for v, x in ev.items() if x.get("complete")}
    if not ok:
        return None, ev
    best = max(ok, key=ok.get)
    return int(str(best).lstrip("p")), ev


def propose(out: Path) -> Dict[str, Any]:
    from experiments.common.config import OUR_MODELS

    complete, missing = gate_status(out)
    precision, ev_p = decide_precision(out)
    lrs, ev_lr = decide_learning_rates(out)
    pack, ev_pack = decide_pack(out)
    return {
        "gate_passed": bool(complete and precision and pack
                            and all(m in lrs for m in OUR_MODELS)),
        "missing_runs": missing[:50], "n_missing_runs": len(missing),
        "precision": precision, "learning_rate": lrs, "pack_seq": pack,
        "evidence": {"E0prec": ev_p, "E0tune": ev_lr, "E0pack": ev_pack},
        "rules": __doc__.split("Rules")[1].split("$CDX_RESULTS/platform")[0].strip(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
    }


def write(out: Path, proposal: Dict[str, Any], force: bool = False) -> Path:
    from experiments.common.utils import atomic_write_yaml

    path = Path(out) / DECISIONS_FILE
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists (decisions are fixed once written; --force)")
    atomic_write_yaml(path, proposal)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    import yaml

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get("CDX_RESULTS", "results"))
    p.add_argument("--write", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    prop = propose(Path(a.out))
    print(yaml.safe_dump({k: v for k, v in prop.items() if k != "rules"}, sort_keys=False))
    if not prop["gate_passed"]:
        print("GATE NOT PASSED: main runs must not start (see missing_runs / evidence)")
        return 1
    if a.write:
        print(f"written: {write(Path(a.out), prop, a.force)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
