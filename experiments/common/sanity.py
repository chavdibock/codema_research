"""E0 helpers: tiny-batch overfit check and the E0 report.

    python -m experiments.common.sanity --overfit [--device cuda] [--steps 500]
    python -m experiments.common.sanity --e0-report --out $CDX_RESULTS

--overfit: every backbone must overfit one fixed batch of 64 windows (loss falls).
--e0-report reads finished E0* runs and prints (nothing is decided automatically):
  * E0prec: final normalized reward/cost of cdt Car-Run seed 0, fp32 vs bf16;
  * E0tune: per backbone, the lr selected on *validation* seeds (rule of spec E0);
  * E0pack: total training steps/s per pack size (pick the highest);
  * E0fid:  original OSRL train_cdt.py vs our pipeline (E1 cdt Car-Run seed 0);
  * E0:     every model x task, whether the loss decreased and the rollout finished.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from experiments.common.config import OUR_MODELS


def overfit(device: str, steps: int, task: str) -> int:
    from experiments.common.config import TrainConfig
    from experiments.common.data import env_constants
    from experiments.common.profiling import build_fresh, random_batch
    from experiments.common.trainer import SequenceTrainer

    dims = env_constants(task)
    failed = 0
    for m in OUR_MODELS:
        pol, _, impl = build_fresh(m, task, device, dims=dims)
        pol.train()
        tc = TrainConfig(batch_size=64, precision="fp32", lr_warmup_steps=10,
                         learning_rate=1e-3, fused_adamw=False)
        tr = SequenceTrainer(pol, tc, device)
        batch = random_batch(pol, 64, pol.seq_len, device)
        losses = [float(tr.train_one_step(batch)["act_loss"]) for _ in range(steps)]
        first, last = np.mean(losses[:10]), np.mean(losses[-10:])
        ok = last < first - 1.0  # NLL of a memorized batch keeps falling
        failed += not ok
        print(f"{m:<10} impl={impl:<4} act_loss {first:8.3f} -> {last:8.3f}  "
              f"{'OK' if ok else 'DID NOT OVERFIT'}")
    return failed


def _load_runs(out: Path, exp: str) -> pd.DataFrame:
    import tempfile

    from experiments.analysis.aggregate import aggregate, load

    with tempfile.TemporaryDirectory(prefix="cdx_e0_") as tmp:
        aggregate(out, Path(tmp))
        runs = load(Path(tmp) / "runs.parquet")
    return runs[runs["exp"] == exp] if not runs.empty else runs


def e0_report(out: Path) -> None:
    from experiments.analysis.aggregate import done_run_dirs, parse_run_dir

    runs = _load_runs(out, "E0prec")
    print("== E0prec: cdt Car-Run seed 0, kappa=10 (keep bf16 only if within seed noise) ==")
    if len(runs):
        print(runs[runs["target_C"] == 10][["variant", "R_norm", "C_norm"]].to_string(index=False))
    else:
        print("  no finished runs")

    print("\n== E0tune: lr selection on validation seeds (safe first, then reward) ==")
    tune = _load_runs(out, "E0tune")
    for m in OUR_MODELS:
        d = tune[tune["model"] == m] if len(tune) else tune
        if not len(d):
            print(f"  {m}: no finished runs")
            continue
        d = d.groupby("variant")[["R_norm", "C_norm"]].mean().reset_index()
        safe = d[d["C_norm"] <= 1.0]
        pick = (safe.sort_values("R_norm", ascending=False) if len(safe)
                else d.sort_values("C_norm")).iloc[0]
        print(f"  {m}: " + ", ".join(f"{r.variant}: R={r.R_norm:.1f} C={r.C_norm:.2f}"
                                     for r in d.itertuples()) + f"  -> {pick['variant']}")

    print("\n== E0pack: total training throughput per pack size ==")
    rows = []
    for d in done_run_dirs(out):
        k = parse_run_dir(d)
        if k["exp"] != "E0pack":
            continue
        with open(d / "run_summary.json") as f:
            s = json.load(f)
        log = pd.read_csv(d / "train_log.csv")
        rows.append({"variant": k["variant"], "steps": int(log["step"].max()),
                     "train_s": s.get("train_wall_s", np.nan)})
    if rows:
        df = pd.DataFrame(rows)
        g = df.groupby("variant").agg(runs=("steps", "size"), steps=("steps", "sum"),
                                      mean_train_s=("train_s", "mean"))
        # concurrent runs: total throughput = total steps / mean wall time of a run
        g["total_steps_per_s"] = g["steps"] / g["mean_train_s"]
        print(g.to_string())
    else:
        print("  no finished runs")

    print("\n== E0fid: OSRL train_cdt.py vs our pipeline (Car-Run seed 0, kappa=10) ==")
    fid = _load_runs(out, "E0fid")
    e1 = _load_runs(out, "E1")
    ours = e1[(e1["model"] == "cdt") & (e1["task"] == "OfflineCarRun-v0") & (e1["seed"] == 0)
              & (e1["target_C"] == 10)] if len(e1) else e1
    for name, d in (("OSRL script", fid[fid["target_C"] == 10] if len(fid) else fid),
                    ("our pipeline", ours)):
        if len(d):
            print(f"  {name:<13} R_norm={d['R_norm'].iloc[0]:.2f} C_norm={d['C_norm'].iloc[0]:.2f}")
        else:
            print(f"  {name:<13} no finished run")

    print("\n== E0 pilot: loss decreased and rollout finished ==")
    for d in done_run_dirs(out):
        k = parse_run_dir(d)
        if k["exp"] != "E0":
            continue
        log = pd.read_csv(d / "train_log.csv")
        col = "all_loss" if "all_loss" in log else next(
            (c for c in log.columns if "loss" in c), None)
        dec = col is not None and log[col].iloc[-1] < log[col].iloc[0]
        ep = d / "eval_episodes.csv"
        n_ep = len(pd.read_csv(ep)) if ep.exists() and ep.stat().st_size else 0
        print(f"  {k['task']:<22} {k['model']:<11} loss decreased: {dec!s:<5} "
              f"eval episodes: {n_ep}")


def main() -> None:
    import os

    from experiments.common.utils import resolve_device

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--e0-report", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--task", default="OfflineCarRun-v0")
    p.add_argument("--out", default=os.environ.get("CDX_RESULTS", "results"))
    a = p.parse_args()
    if a.overfit:
        raise SystemExit(overfit(resolve_device(a.device), a.steps, a.task))
    if a.e0_report:
        e0_report(Path(a.out))


if __name__ == "__main__":
    main()
