"""Experiment definitions E0-E11 (spec §6): models, tasks, seeds, overrides, targets.

A run is identified by (exp, model, variant, task, seed). ``resolve_run_config`` builds
its fully resolved ``RunConfig``; ``enumerate_runs`` lists the runs of an experiment
for the manifest generator.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from experiments.common.config import (OUR_MODELS, TASKS, DataConfig, RunConfig,
                                       apply_overrides, get_model_spec, task_defaults)

SEEDS5 = [0, 1, 2, 3, 4]
SEEDS3 = [0, 1, 2]
E2_BASELINES = ["bc_safe", "dt_cost", "bcq_lag", "bear_lag", "cpq", "coptidice"]
E9_BC = ["bc_all", "bc_risky", "bc_frontier", "bc_boundary"]
E4_KAPPAS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
E5_NUM_REWARDS = 8
E5_REWARD_RANGE = (0.5, 1.3)  # x r_max of the dataset
E8_ALPHAS = [0.1, 0.4, 1.0, 1.5, 2.0]  # percent
# NOTE: DSRL DENSITY_CFG exists for densities 0.5 and 0.75; we use 0.5 (see NOTES.md)
E8_DENSITY = 0.5
E11_DIMS = [32, 64, 128, 256]

# Unconstrained sequence models for E7 (DT without the cost token, [CDT] App. A).
UNCONSTRAINED = {"model.use_cost": False, "model.stochastic": False,
                 "data.augment_percent": 0.0, "data.cost_sample": False,
                 "train.loss_cost_weight": 0.0}


@dataclass
class RunSpec:
    exp: str
    model: str
    variant: Optional[str]
    task: str
    seed: int
    kind: str  # "seq" | "mlp" | "eval" | "osrl"
    ckpt_from: Optional[Tuple[str, Optional[str]]] = None  # (exp, variant) for eval-only


@dataclass
class ExpDef:
    name: str
    priority: str
    description: str
    models: List[str]
    tasks: List[str]
    seeds: List[int]
    variants: Dict[Optional[str], Dict[str, Any]] = field(default_factory=lambda: {None: {}})
    overrides: Dict[str, Any] = field(default_factory=dict)
    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_set: str = "kappa10"
    model_target_set: Dict[str, str] = field(default_factory=dict)
    # eval-only experiments: variant -> (source exp, source variant, seeds)
    eval_sources: Dict[Optional[str], Tuple[str, Optional[str], List[int]]] = field(
        default_factory=dict)
    eval_models: List[str] = field(default_factory=list)
    extra_runs: Optional[Callable[[], List[RunSpec]]] = None


def _variants_e8() -> Dict[Optional[str], Dict[str, Any]]:
    out = {}
    for a in E8_ALPHAS:
        base = {"data.outliers_percent": a / 100.0}
        out[f"a{a:g}_assoc"] = dict(base)
        out[f"a{a:g}_density"] = dict(base, **{"data.post_density": E8_DENSITY})
    return out


def _e0fid_runs() -> List[RunSpec]:
    # original OSRL train_cdt.py on Car-Run, seed 0 (compared with E1 cdt Car-Run seed 0)
    return [RunSpec("E0fid", "osrl_cdt", None, "OfflineCarRun-v0", 0, "osrl")]


EXPERIMENTS: Dict[str, ExpDef] = {
    "E0": ExpDef("E0", "P0", "pilot: every model and task, 2k steps, 1 seed",
                 models=OUR_MODELS + E2_BASELINES + ["mt_bc"], tasks=TASKS, seeds=[0],
                 overrides={"train.update_steps": 2000, "train.val_every": 1000,
                            "train.val_episodes": 2, "train.ckpt_every": 1000,
                            "eval.eval_episodes": 4},
                 model_target_set={"mt_bc": "kappa10"}),
    "E0prec": ExpDef("E0prec", "P0", "bf16 vs fp32 check, cdt Car-Run seed 0, 100k steps",
                     models=["cdt"], tasks=["OfflineCarRun-v0"], seeds=[0],
                     variants={"fp32": {"train.precision": "fp32"},
                               "bf16": {"train.precision": "bf16"}}),
    "E0tune": ExpDef("E0tune", "P0", "lr in {1e-4, 3e-4}, 30k steps, Car-Circle, val seeds",
                     models=OUR_MODELS, tasks=["OfflineCarCircle-v0"], seeds=[0],
                     variants={"lr1e-4": {"train.learning_rate": 1e-4},
                               "lr3e-4": {"train.learning_rate": 3e-4}},
                     overrides={"train.update_steps": 30_000, "eval.seed_split": "val"}),
    "E0pack": ExpDef("E0pack", "P0", "pack-size sweep: 9 cdt runs x 2k steps per pack size",
                     models=["cdt"], tasks=["OfflineCarCircle-v0"], seeds=list(range(9)),
                     variants={f"p{p}": {} for p in (1, 3, 6, 9)},
                     overrides={"train.update_steps": 2000, "train.val_every": 0,
                                "eval.eval_episodes": 0}),
    "E0fid": ExpDef("E0fid", "P0", "fidelity: original OSRL train_cdt.py, Car-Run seed 0",
                    models=[], tasks=["OfflineCarRun-v0"], seeds=[0],
                    extra_runs=_e0fid_runs),
    "E1": ExpDef("E1", "P0", "main comparison, kappa=10 (+20/40 for Table 4)",
                 models=OUR_MODELS, tasks=TASKS, seeds=SEEDS5, target_set="multi"),
    "E2": ExpDef("E2", "P1", "baselines", models=E2_BASELINES, tasks=TASKS, seeds=SEEDS5,
                 target_set="kappa10", model_target_set={"dt_cost": "multi"}),
    "E3": ExpDef("E3", "P1", "component ablations for every backbone", models=OUR_MODELS,
                 tasks=TASKS, seeds=SEEDS3,
                 variants={"noaug": {"data.augment_percent": 0.0},
                           "noent": {"train.no_entropy": True},
                           "det": {"model.stochastic": False}}),
    "E4": ExpDef("E4", "P0", "zero-shot cost-budget adaptation (eval of E1 + MT-BC)",
                 models=["mt_bc"], tasks=TASKS, seeds=SEEDS5, target_set="kappa_sweep",
                 eval_models=OUR_MODELS, eval_sources={None: ("E1", None, SEEDS5)}),
    "E5": ExpDef("E5", "P1", "conflicting reward targets (eval of E1 and E3 noaug)",
                 models=[], tasks=TASKS, seeds=SEEDS5, target_set="reward_sweep",
                 eval_models=OUR_MODELS,
                 eval_sources={None: ("E1", None, SEEDS5), "noaug": ("E3", "noaug", SEEDS3)}),
    "E6full": ExpDef("E6full", "P0", "dedicated unpacked full runs for M1 ground truth",
                     models=OUR_MODELS, tasks=["OfflineCarCircle-v0"], seeds=[0],
                     target_set="kappa10"),
    "E6long": ExpDef("E6long", "P3", "E6.7: performance at K=50",
                     models=["cdt", "cdmamba", "cdconv"],
                     tasks=["OfflineCarCircle-v0", "OfflineDroneRun-v0"], seeds=SEEDS3,
                     variants={"K50": {"data.seq_len": 50}}),
    "E7": ExpDef("E7", "P2", "epsilon-reducibility: full vs reduced datasets",
                 models=["dt", "bcq", "bear"], tasks=["OfflineAntRun-v0", "OfflineDroneRun-v0"],
                 seeds=SEEDS3,
                 variants={"full": {}, "reduced": {"data.reduce_rule": "cdt"}}),
    "E7ext": ExpDef("E7ext", "P3", "E7 extension: our backbones in unconstrained mode",
                    models=["cdt_light", "cdmamba", "cdconv"],
                    tasks=["OfflineAntRun-v0", "OfflineDroneRun-v0"], seeds=SEEDS3,
                    variants={"full_unc": dict(UNCONSTRAINED),
                              "reduced_unc": dict(UNCONSTRAINED, **{"data.reduce_rule": "cdt"})}),
    "E8": ExpDef("E8", "P2", "outlier robustness on Drone-Run", models=OUR_MODELS,
                 tasks=["OfflineDroneRun-v0"], seeds=SEEDS3, variants=_variants_e8()),
    "E9": ExpDef("E9", "P2", "BC variants", models=E9_BC, tasks=TASKS, seeds=SEEDS3),
    "E10": ExpDef("E10", "P0", "dataset visualizations (no training)", models=[], tasks=TASKS,
                  seeds=[0]),
    "E11": ExpDef("E11", "P3", "capacity sweep", models=["cdt", "cdmamba", "cdconv"],
                  tasks=["OfflineCarCircle-v0", "OfflineDroneRun-v0"], seeds=SEEDS3,
                  variants={f"d{d}": {"model.embedding_dim": d} for d in E11_DIMS}),
}


def get_exp(name: str) -> ExpDef:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name}; known: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def enumerate_runs(exp_name: str, models: Optional[List[str]] = None,
                   tasks: Optional[List[str]] = None, seeds: Optional[List[int]] = None,
                   variants: Optional[List[str]] = None) -> List[RunSpec]:
    """All runs of an experiment, grouped by task (packs then share the data cache)."""
    e = get_exp(exp_name)
    runs: List[RunSpec] = []
    for task in e.tasks:
        if tasks and task not in tasks:
            continue
        for model in e.models:
            if models and model not in models:
                continue
            kind = get_model_spec(model).pack_kind
            for variant in e.variants:
                if variants and (variant or "") not in variants:
                    continue
                for seed in e.seeds:
                    if seeds is not None and seed not in seeds:
                        continue
                    runs.append(RunSpec(exp_name, model, variant, task, seed, kind))
        for variant, (src_exp, src_var, src_seeds) in e.eval_sources.items():
            if variants and (variant or "") not in variants:
                continue
            for model in e.eval_models:
                if models and model not in models:
                    continue
                for seed in src_seeds:
                    if seeds is not None and seed not in seeds:
                        continue
                    runs.append(RunSpec(exp_name, model, variant, task, seed, "eval",
                                        ckpt_from=(src_exp, src_var)))
    if e.extra_runs is not None:
        runs.extend(r for r in e.extra_runs() if not tasks or r.task in tasks)
    return runs


def resolve_run_config(exp_name: str, model_id: str, task: str, seed: int,
                       variant: Optional[str] = None, out: str = "results",
                       cli_overrides: Optional[Dict[str, Any]] = None,
                       device: str = "auto", cache_dir: str = "") -> RunConfig:
    """defaults -> OSRL per-task -> model spec -> experiment -> variant -> CLI."""
    e = get_exp(exp_name)
    cfg = RunConfig(exp=exp_name, model_id=model_id, variant=variant, task=task, seed=seed,
                    out=out, device=device, cache_dir=cache_dir)
    td = task_defaults(task)
    cfg.data.task = task
    for k, v in td["data"].items():
        setattr(cfg.data, k, copy.deepcopy(v))
    cfg.meta["osrl_episode_len"] = td["osrl_episode_len"]
    cfg.meta["task_targets"] = td["targets"]

    spec = get_model_spec(model_id)
    cfg.model = copy.deepcopy(spec.model)
    apply_overrides(cfg, spec.overrides)
    cfg.eval.target_set = e.model_target_set.get(model_id, e.target_set)
    apply_overrides(cfg, e.overrides)
    apply_overrides(cfg, e.model_overrides.get(model_id, {}))
    if variant not in e.variants and variant not in e.eval_sources:
        raise KeyError(f"{exp_name} has no variant {variant!r}")
    apply_overrides(cfg, e.variants.get(variant, {}))
    apply_overrides(cfg, cli_overrides or {})
    if cfg.data.episode_len == 0:
        from experiments.common.data import env_spec

        cfg.data.episode_len = int(env_spec(task).max_episode_steps)  # D1
    cfg.eval.targets = resolve_targets(cfg.eval.target_set, td["targets"])
    return cfg


def resolve_targets(target_set: str, task_targets: List[List[float]],
                    data_r_max: Optional[float] = None) -> List[List[float]]:
    """(reward, cost) target pairs for a target set (spec §5.2, E4, E5)."""
    by_kappa = {float(c): float(r) for r, c in task_targets}
    r10 = by_kappa[10.0]
    if target_set in ("kappa10", "none"):
        return [[r10, 10.0]]
    if target_set == "multi":
        return [[by_kappa[k], k] for k in (10.0, 20.0, 40.0)]
    if target_set == "kappa_sweep":
        return [[r10, float(k)] for k in E4_KAPPAS]
    if target_set == "reward_sweep":
        if data_r_max is None:
            return []  # resolved at evaluation time from the checkpoint's env_info.json
        lo, hi = E5_REWARD_RANGE
        return [[float(r), 10.0]
                for r in np.linspace(lo * data_r_max, hi * data_r_max, E5_NUM_REWARDS)]
    raise ValueError(f"unknown target_set {target_set}")


def data_jobs_for(exp_name: str) -> List[Tuple[DataConfig, int, str]]:
    """(data config, seed, cache kind) needed by an experiment (for prepare_data)."""
    jobs = []
    if exp_name == "E10":
        # full datasets, E7 reduced datasets, E8 2 % outliers (both mitigations)
        for task in TASKS:
            jobs.append((resolve_run_config("E1", "cdt", task, 0).data, 0, "seq"))
        for task in get_exp("E7").tasks:
            jobs.append((resolve_run_config("E7", "dt", task, 0, "reduced").data, 0, "pre"))
            jobs.append((resolve_run_config("E7", "dt", task, 0, "full").data, 0, "pre"))
        for v in ("a2_assoc", "a2_density"):
            jobs.append((resolve_run_config("E8", "cdt", "OfflineDroneRun-v0", 0, v).data, 0,
                         "seq"))
        return jobs
    for r in enumerate_runs(exp_name):
        if r.kind in ("seq", "mlp"):
            cfg = resolve_run_config(exp_name, r.model, r.task, r.seed, r.variant)
            jobs.append((cfg.data, r.seed, "seq" if r.kind == "seq" else "pre"))
    return jobs
