"""Normalization, violation rate, excess cost (spec §5.1).

R_norm = 100 * (R - r_min) / (r_max - r_min) with DSRL's per-task constants
(``max_episode_reward``/``min_episode_reward``, the ones ``get_normalized_score`` uses).
C_norm = C / (kappa + eps), eps = 1e-6, where kappa is the budget *requested in that
evaluation* (OSRL's eval_cdt.py always divides by cost_limit=10; we do not).
"""

from typing import Dict, Iterable

import numpy as np

COST_EPS = 1e-6


def normalized_reward(R, r_min: float, r_max: float):
    return 100.0 * (np.asarray(R, dtype=np.float64) - r_min) / (r_max - r_min)


def normalized_cost(C, kappa, eps: float = COST_EPS):
    return np.asarray(C, dtype=np.float64) / (np.asarray(kappa, dtype=np.float64) + eps)


def violation(C, kappa):
    return (np.asarray(C, dtype=np.float64) > np.asarray(kappa, dtype=np.float64)).astype(float)


def excess_cost(C, kappa):
    return np.maximum(0.0, np.asarray(C, dtype=np.float64) - np.asarray(kappa, dtype=np.float64))


def episode_metrics(R, C, kappa, r_min: float, r_max: float) -> Dict[str, np.ndarray]:
    return {
        "R_norm": normalized_reward(R, r_min, r_max),
        "C_norm": normalized_cost(C, kappa),
        "violation": violation(C, kappa),
        "excess": excess_cost(C, kappa),
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "n": int(v.size)}


def budget_tracking_error(kappas, costs, data_c_max: float) -> float:
    """E4: mean over kappa <= dataset-max of max(0, C - kappa)."""
    k = np.asarray(kappas, dtype=np.float64)
    c = np.asarray(costs, dtype=np.float64)
    sel = k <= data_c_max
    return float(np.mean(np.maximum(0.0, c[sel] - k[sel]))) if sel.any() else float("nan")


def is_safe(c_norm_mean: float) -> bool:
    """[CDT] §5: an agent is safe if its normalized cost is <= 1."""
    return bool(c_norm_mean <= 1.0)
