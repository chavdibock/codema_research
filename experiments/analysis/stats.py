"""Seed-level aggregation, bootstrap CIs and the [CDT] ranking rule (spec §5.1)."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# model id -> display name (order = row order in tables)
OUR_MODELS = ["cdt", "cdt_light", "cdmamba", "cdconv"]
BASELINES = ["bc_safe", "dt_cost", "bcq_lag", "bear_lag", "cpq", "coptidice"]
DISPLAY = {
    "cdt": "CDT", "cdt_light": "Lightweight CDT", "cdmamba": "CDMamba",
    "cdconv": "CDConvFormer", "bc_safe": "BC-Safe", "dt_cost": "DT-Cost",
    "bcq_lag": "BCQ-Lag", "bear_lag": "BEAR-Lag", "cpq": "CPQ", "coptidice": "COptiDICE",
    "mt_bc": "MT-BC", "bc_all": "BC-All", "bc_risky": "BC-Risky",
    "bc_frontier": "BC-Frontier", "bc_boundary": "BC-Boundary", "dt": "DT", "bcq": "BCQ",
    "bear": "BEAR",
}
TASK_ORDER = ["OfflineAntRun-v0", "OfflineCarCircle-v0", "OfflineCarRun-v0",
              "OfflineDroneCircle-v0", "OfflineDroneRun-v0"]
TASK_SHORT = {"OfflineAntRun-v0": "Ant-Run", "OfflineCarCircle-v0": "Car-Circle",
              "OfflineCarRun-v0": "Car-Run", "OfflineDroneCircle-v0": "Drone-Circle",
              "OfflineDroneRun-v0": "Drone-Run"}


def seed_summary(runs: pd.DataFrame, by: Sequence[str],
                 metrics: Sequence[str] = ("R_norm", "C_norm")) -> pd.DataFrame:
    """Per-seed values (already episode means) -> mean and std over seeds."""
    if runs.empty:
        return pd.DataFrame()
    g = runs.groupby(list(by))
    out = g[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def bootstrap_ci(episodes: pd.DataFrame, metric: str, n_boot: int = 2000,
                 alpha: float = 0.05, rng: Optional[np.random.Generator] = None):
    """Two-level bootstrap: resample seeds, then episodes within each seed."""
    rng = rng or np.random.default_rng(0)
    groups = [g[metric].to_numpy() for _, g in episodes.groupby("seed")]
    if not groups:
        return (np.nan, np.nan)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, len(groups), len(groups))
        stats[b] = np.mean([rng.choice(groups[s], len(groups[s])).mean() for s in seeds])
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def rank_key(r_norm: float, c_norm: float):
    """[CDT] §5: safe beats unsafe; unsafe by lower cost; safe by higher reward."""
    if np.isnan(r_norm) or np.isnan(c_norm):
        return (2, 0.0)
    return (0, -r_norm) if c_norm <= 1.0 else (1, c_norm)


def cell_style(c_norm: float, best: bool) -> str:
    """'best' (blue), 'safe' (bold), 'unsafe' (gray) or 'missing'."""
    if np.isnan(c_norm):
        return "missing"
    if c_norm > 1.0:
        return "unsafe"
    return "best" if best else "safe"
