"""Walk results/ and write tidy tables (spec §8.5).

    python -m experiments.analysis.aggregate --results results --summary summary

Writes summary/episodes.parquet (one row per test episode), summary/runs.parquet (one
row per run x target: per-seed means), summary/val_logs.parquet (learning curves) and
summary/efficiency.parquet (E6). Only runs with a DONE marker are included.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

METRICS = ["R", "C", "R_norm", "C_norm", "violation", "excess", "len"]


def parse_run_dir(d: Path) -> Dict[str, Any]:
    seed_dir, mv, task, exp = d, d.parent, d.parent.parent, d.parent.parent.parent
    model, _, variant = mv.name.partition("__")
    return {"exp": exp.name, "task": task.name, "model": model, "variant": variant,
            "seed": int(seed_dir.name.split("_", 1)[1])}


def done_run_dirs(results: Path) -> List[Path]:
    return sorted(p.parent for p in results.glob("*/*/*/seed_*/DONE"))


def _read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def save(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError):
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return alt


def load(path: Path) -> pd.DataFrame:
    """Read a summary table (parquet, or the csv fallback); empty frame if missing."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    return pd.DataFrame()


def aggregate(results: Path, summary: Path) -> Dict[str, int]:
    episodes, runs, vals = [], [], []
    for d in done_run_dirs(results):
        key = parse_run_dir(d)
        info = _read_json(d / "env_info.json")
        rs = _read_json(d / "run_summary.json")
        ep_path = d / "eval_episodes.csv"
        if ep_path.exists() and ep_path.stat().st_size > 0:
            ep = pd.read_csv(ep_path)
            ep = ep.assign(**{k: v for k, v in key.items()})
            episodes.append(ep)
            g = ep.groupby(["target_R", "target_C"], as_index=False)
            per = g[[m for m in METRICS if m in ep.columns]].mean()
            per["n_episodes"] = g.size()["size"].values
            for k, v in key.items():
                per[k] = v
            per["data_c_max"] = info.get("data_c_max", np.nan)
            per["data_r_max"] = info.get("data_r_max", np.nan)
            per["eps_hat"] = (info.get("eps_data") or {}).get("eps_hat", np.nan)
            for f in ("train_wall_s", "val_wall_s", "eval_wall_s", "total_wall_s",
                      "peak_train_mem_mb", "params_total", "params_backbone"):
                per[f] = rs.get(f, np.nan)
            roll = rs.get("rollout_indicative") or rs.get("rollout") or {}
            per["rollout_act_ms_median"] = roll.get("act_ms_median", np.nan)
            runs.append(per)
        vp = d / "val_log.csv"
        if vp.exists() and vp.stat().st_size > 0:
            v = pd.read_csv(vp).assign(**key)
            vals.append(v)
    out = {}
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    for name, frames in (("episodes", episodes), ("runs", runs), ("val_logs", vals)):
        df = cat(frames)
        save(df, summary / f"{name}.parquet")
        out[name] = len(df)
    eff = results / "E6" / "efficiency.csv"
    if eff.exists():
        df = pd.read_csv(eff)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        save(df, summary / "efficiency.parquet")
        out["efficiency"] = len(df)
    hw = results / "E6" / "hardware.json"
    if hw.exists():
        (summary / "hardware.json").write_text(hw.read_text())
    return out


def main() -> None:
    import os

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=os.environ.get("CDX_RESULTS", "results"))
    p.add_argument("--summary", default="summary")
    a = p.parse_args()
    counts = aggregate(Path(a.results), Path(a.summary))
    print(f"aggregated from {a.results}: {counts}")


if __name__ == "__main__":
    main()
