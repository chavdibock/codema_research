"""Single entry point: one run = one command (spec §8.2).

    python -m experiments.common.run --exp E1 --model cdmamba \
        --task OfflineCarRun-v0 --seed 3 [--variant V] [--override key=value ...] \
        --out results/

Trains, evaluates every target the experiment requests, writes the results tree of
spec §8.4 and finally ``DONE``. Idempotent (exits if ``DONE`` exists) and resumable
(checkpoint every ``train.ckpt_every`` steps; restarts resume from the latest one).
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from experiments.common import osrl_compat
from experiments.common.config import RunConfig, parse_override, to_dict
from experiments.common.logging import CSVLog, WandbOffline
from experiments.common.utils import (DONE, FAILED, RUNNING, atomic_torch_save,
                                      atomic_write_json, atomic_write_yaml,
                                      experiments_code_hash, hardware_info, is_done, job_tag,
                                      pip_freeze, remove_marker, resolve_device, run_dir_for,
                                      seed_all, write_marker)


def provenance() -> Dict[str, Any]:
    return {"experiments_hash": experiments_code_hash(), "osrl_commit": osrl_compat.osrl_commit(),
            "hardware": hardware_info(), "pip_freeze": pip_freeze().splitlines(),
            "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def build_env_info(cfg: RunConfig, data_meta: Dict[str, Any]) -> Dict[str, Any]:
    from experiments.common.data import env_constants

    info = env_constants(cfg.task)
    info["episode_len"] = cfg.data.episode_len  # D1: == env.spec.max_episode_steps
    info["osrl_episode_len"] = cfg.meta.get("osrl_episode_len")
    info["normalization"] = {"r_min": info["min_episode_reward"],
                             "r_max": info["max_episode_reward"], "cost_eps": 1e-6}
    info.update({k: v for k, v in data_meta.items() if not isinstance(v, dict) or k.startswith("eps")})
    return info


# ------------------------------------------------------------ sequence models
def run_sequence(cfg: RunConfig, run_dir: Path, device: str) -> Dict[str, Any]:
    from experiments.common.data import (GPUWindowSampler, OSRLLoaderSampler,
                                         build_osrl_sequence_dataset, get_sequence_data,
                                         load_preprocessed_uncached)
    from experiments.common.evaluate import evaluate_actor, mean_row, policy_actor_factory
    from experiments.common.policy import build_policy
    from experiments.common.trainer import SequenceTrainer, fit, setup_precision

    setup_precision(cfg.train)
    seed_all(cfg.seed)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    sd = get_sequence_data(cfg.data, cfg.seed, cache_dir)
    env_info = build_env_info(cfg, sd.meta)
    atomic_write_json(run_dir / "env_info.json", env_info)

    # torch RNG is untouched by data preparation, so the init matches OSRL's for a seed
    seed_all(cfg.seed)
    policy = build_policy(cfg.model, env_info["state_dim"], env_info["action_dim"],
                          env_info["max_action"], cfg.data.seq_len, cfg.data.episode_len)
    policy = policy.to(device)
    trainer = SequenceTrainer(policy, cfg.train, device)
    if cfg.train.sampler == "gpu":
        sampler = GPUWindowSampler(sd, cfg.data.seq_len, cfg.data.reward_scale,
                                   cfg.data.cost_scale, cfg.train.batch_size, device, cfg.seed)
    else:
        data, _ = load_preprocessed_uncached(cfg.data, cfg.seed)
        sampler = OSRLLoaderSampler(build_osrl_sequence_dataset(data, cfg.data),
                                    cfg.train.batch_size, device, cfg.train.num_workers)
    make_actor = policy_actor_factory(cfg, policy, device)
    k10 = [t for t in cfg.meta["task_targets"] if float(t[1]) == 10.0][:1]

    def validate(step: int) -> Dict[str, Any]:
        policy.eval()
        rows, _ = evaluate_actor(cfg, make_actor, k10, cfg.train.val_episodes, "val",
                                 env_info, device)
        policy.train()
        return mean_row(rows)

    validate_fn = validate if cfg.seed in cfg.train.val_seeds and cfg.train.val_every > 0 else None
    final_path = run_dir / "ckpt_final.pt"
    train_stats: Dict[str, Any] = {}
    if final_path.exists():  # training finished earlier, evaluation did not
        ck = torch.load(final_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model_state"])
        train_stats = ck.get("train_stats", {})
    else:
        policy.train()
        train_stats = fit(trainer.train_one_step, sampler, cfg.train, run_dir, device,
                          trainer.state_dict, trainer.load_state_dict, trainer.lr,
                          validate_fn, tokens_per_sample=policy.seq_repeat * cfg.data.seq_len)
        atomic_torch_save({"model_state": policy.state_dict(), "config": to_dict(cfg),
                           "env_info": env_info, "step": cfg.train.update_steps,
                           "train_stats": train_stats}, final_path)
    policy.eval()
    t_eval = time.perf_counter()
    rows, roll = [], {}
    if cfg.eval.eval_episodes > 0 and cfg.eval.targets:
        rows, roll = evaluate_actor(cfg, make_actor, cfg.eval.targets, cfg.eval.eval_episodes,
                                    cfg.eval.seed_split, env_info, device)
    CSVLog(run_dir / "eval_episodes.csv").write_rows(rows)
    return {**train_stats, "eval_wall_s": time.perf_counter() - t_eval,
            "rollout_indicative": roll, "params_total": policy.num_parameters(),
            "params_backbone": policy.num_parameters(backbone_only=True),
            "sampler_mb": sampler.nbytes() / 2**20 if hasattr(sampler, "nbytes") else None}


# ------------------------------------------------------------ MLP baselines
def run_mlp(cfg: RunConfig, run_dir: Path, device: str) -> Dict[str, Any]:
    from experiments.baselines.mlp import (build_mlp, capture_state, coptidice_stats,
                                           make_step_fn, mlp_actor_factory,
                                           prepare_transitions, resolve_osrl_config,
                                           restore_state)
    from experiments.common.data import GPUTransitionSampler, get_transition_data
    from experiments.common.evaluate import evaluate_actor, mean_row
    from experiments.common.trainer import fit, setup_precision

    setup_precision(cfg.train)
    seed_all(cfg.seed)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    raw, meta = get_transition_data(cfg.data, cfg.seed, cache_dir)
    env_info = build_env_info(cfg, meta)
    atomic_write_json(run_dir / "env_info.json", env_info)
    algo = cfg.model.algo
    osrl_cfg = resolve_osrl_config(cfg.model, cfg.task, cfg.data.episode_len)
    osrl_cfg["update_steps"] = cfg.train.update_steps  # our protocol: same steps for all
    data = prepare_transitions(algo, osrl_cfg, raw)
    extra = coptidice_stats(data) if algo == "coptidice" else {}
    seed_all(cfg.seed)
    model, trainer, logger = build_mlp(algo, osrl_cfg, env_info["state_dim"],
                                       env_info["action_dim"], env_info["max_action"],
                                       device, extra)
    # NOTE: OSRL's per-algorithm batch size (512) is kept for the MLP baselines.
    tcfg = type(cfg.train)(**{**to_dict(cfg.train), "batch_size": osrl_cfg["batch_size"]})
    sampler = GPUTransitionSampler(data, osrl_cfg.get("reward_scale", 1.0),
                                   osrl_cfg.get("cost_scale", 1.0), tcfg.batch_size, device,
                                   cfg.seed, state_init=(algo == "coptidice"))
    bc_mode = osrl_cfg.get("bc_mode", "")
    make_actor = mlp_actor_factory(model, algo, bc_mode, device)
    k10 = [t for t in cfg.meta["task_targets"] if float(t[1]) == 10.0][:1]

    def validate(step: int) -> Dict[str, Any]:
        model.eval()
        rows, _ = evaluate_actor(cfg, make_actor, k10, cfg.train.val_episodes, "val",
                                 env_info, device)
        model.train()
        return mean_row(rows)

    validate_fn = validate if cfg.seed in cfg.train.val_seeds and cfg.train.val_every > 0 else None
    final_path = run_dir / "ckpt_final.pt"
    train_stats: Dict[str, Any] = {}
    if final_path.exists():
        ck = torch.load(final_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        train_stats = ck.get("train_stats", {})
    else:
        model.train()
        train_stats = fit(make_step_fn(algo, trainer, cfg.train.precision, device), sampler,
                          tcfg, run_dir, device, lambda: capture_state(model),
                          lambda st: restore_state(model, st), lambda: float("nan"),
                          validate_fn, extra_log_fn=logger.means_and_reset)
        atomic_torch_save({"model_state": model.state_dict(), "config": to_dict(cfg),
                           "osrl_cfg": osrl_cfg, "extra_kwargs": extra, "env_info": env_info,
                           "step": cfg.train.update_steps, "train_stats": train_stats},
                          final_path)
    model.eval()
    t_eval = time.perf_counter()
    rows, roll = [], {}
    if cfg.eval.eval_episodes > 0 and cfg.eval.targets:
        rows, roll = evaluate_actor(cfg, make_actor, cfg.eval.targets, cfg.eval.eval_episodes,
                                    cfg.eval.seed_split, env_info, device)
    CSVLog(run_dir / "eval_episodes.csv").write_rows(rows)
    return {**train_stats, "eval_wall_s": time.perf_counter() - t_eval,
            "rollout_indicative": roll,
            "params_total": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "osrl_config": {k: v for k, v in osrl_cfg.items() if not k.startswith("_")}}


# ------------------------------------------------------------------ main
def execute(cfg: RunConfig, run_dir: Path) -> int:
    if is_done(run_dir):
        print(f"skip (DONE): {run_dir}")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    write_marker(run_dir, RUNNING, job_tag())
    remove_marker(run_dir, FAILED)
    try:
        device = resolve_device(cfg.device)
        cfg.meta.update(provenance())
        atomic_write_yaml(run_dir / "config.yaml", to_dict(cfg))
        wb = WandbOffline(run_dir, f"{cfg.exp}-{cfg.model_id}-{cfg.task}-{cfg.seed}", to_dict(cfg))
        t0 = time.perf_counter()
        if cfg.model.kind == "sequence":
            summary = run_sequence(cfg, run_dir, device)
        else:
            summary = run_mlp(cfg, run_dir, device)
        summary["total_wall_s"] = time.perf_counter() - t0
        summary["device"] = device
        summary["packed_job"] = os.environ.get("CDX_PACK", "1")
        summary["note"] = "timings from packed jobs are indicative only (spec §4.6.5)"
        atomic_write_json(run_dir / "run_summary.json", summary)
        wb.finish()
        write_marker(run_dir, DONE, job_tag())
        ck = run_dir / "ckpt_latest.pt"
        if ck.exists():
            ck.unlink()
    except BaseException:
        write_marker(run_dir, FAILED, job_tag() + "\n" + traceback.format_exc())
        raise
    finally:
        remove_marker(run_dir, RUNNING)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.common.expdefs import resolve_run_config

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", default=None)
    p.add_argument("--override", nargs="+", action="extend", default=[],
                   help="section.key=value (YAML-parsed value)")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--cache-dir", default=os.environ.get("CDX_CACHE", ""))
    p.add_argument("--dry-run", action="store_true", help="print the resolved config only")
    args = p.parse_args(argv)
    overrides = dict(parse_override(o) for o in args.override)
    cfg = resolve_run_config(args.exp, args.model, args.task, args.seed, args.variant,
                             args.out, overrides, args.device, args.cache_dir)
    run_dir = run_dir_for(args.out, args.exp, args.task, args.model, args.variant, args.seed)
    if args.dry_run:
        import yaml

        print(f"# run_dir: {run_dir}")
        print(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        return 0
    return execute(cfg, run_dir)


if __name__ == "__main__":
    sys.exit(main())
