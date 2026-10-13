"""Evaluation helpers and eval-only experiments (E4, E5; S8).

    python -m experiments.common.evaluate --exp E4 --ckpt-from E1 \
        --model cdmamba --task OfflineCarRun-v0 --seed 3 --out results/
    python -m experiments.common.evaluate --exp E5 --ckpt-from E3 --src-variant noaug \
        --variant noaug --model cdt --task OfflineCarRun-v0 --seed 0 --out results/
    python -m experiments.common.evaluate --exp E0fid --model osrl_cdt \
        --task OfflineCarRun-v0 --seed 0 --osrl-ckpt <OSRL logdir> --out results/

Each checkpoint is loaded once and all (target x episode) jobs run in one vectorized
pass with a batched forward per time step.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from experiments.common.config import RunConfig, from_dict, to_dict
from experiments.common.logging import CSVLog
from experiments.common.metrics import episode_metrics
from experiments.common.rollout import make_jobs, run_rollouts, sequence_actor_factory
from experiments.common.utils import (DONE, FAILED, RUNNING, atomic_write_json,
                                      atomic_write_yaml, is_done, job_tag, remove_marker,
                                      resolve_device, run_dir_for, write_marker)


def seed_base(cfg: RunConfig, split: Optional[str] = None) -> int:
    split = split or cfg.eval.seed_split
    return cfg.eval.val_seed_base if split == "val" else cfg.eval.test_seed_base


def score_rows(rows: List[Dict[str, Any]], env_info: Dict[str, Any],
               extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Add R_norm, C_norm (requested kappa), violation, excess to per-episode rows."""
    if not rows:
        return rows
    R = np.array([r["R"] for r in rows])
    C = np.array([r["C"] for r in rows])
    k = np.array([r["target_C"] for r in rows])
    m = episode_metrics(R, C, k, env_info["min_episode_reward"], env_info["max_episode_reward"])
    out = []
    for i, r in enumerate(rows):
        row = dict(extra)
        row.update(r)
        row.update({key: float(v[i]) for key, v in m.items()})
        out.append(row)
    return out


def evaluate_actor(cfg: RunConfig, make_actor, targets: List[List[float]], episodes: int,
                   split: str, env_info: Dict[str, Any], device: str
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Seeded rollouts for every (target, episode); returns scored rows + stats."""
    jobs = make_jobs(targets, episodes, seed_base(cfg, split))
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(cfg.eval.threads)
    try:
        rows, stats = run_rollouts(cfg.task, jobs, make_actor, cfg.data.episode_len,
                                   vectorized=cfg.eval.vectorized,
                                   num_workers=cfg.eval.num_workers, device=device,
                                   context=cfg.eval.vec_context)
    finally:
        torch.set_num_threads(prev_threads)
    extra = {"exp": cfg.exp, "model": cfg.model_id, "variant": cfg.variant or "",
             "task": cfg.task, "seed": cfg.seed, "split": split,
             "mode": cfg.eval.inference_mode}
    return score_rows(rows, env_info, extra), stats


def policy_actor_factory(cfg: RunConfig, policy, device: str):
    return sequence_actor_factory(policy, cfg.data.reward_scale, cfg.data.cost_scale, device,
                                  sample=cfg.eval.eval_sample, mode=cfg.eval.inference_mode,
                                  batch_invariant=cfg.eval.batch_invariant)


def mean_row(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ("R", "C", "R_norm", "C_norm")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# ----------------------------------------------------------- checkpoints
def load_run(run_dir: Path, device: str):
    """(cfg, env_info, model-or-policy) from a finished run directory."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = from_dict(yaml.safe_load(f))
    with open(run_dir / "env_info.json") as f:
        env_info = json.load(f)
    ck = torch.load(run_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)
    if cfg.model.kind == "sequence":
        from experiments.common.policy import build_policy

        policy = build_policy(cfg.model, env_info["state_dim"], env_info["action_dim"],
                              env_info["max_action"], cfg.data.seq_len, cfg.data.episode_len)
        policy.load_state_dict(ck["model_state"])
        return cfg, env_info, policy.to(device).eval()
    from experiments.baselines.mlp import build_mlp

    model, _, _ = build_mlp(cfg.model.algo, ck["osrl_cfg"], env_info["state_dim"],
                            env_info["action_dim"], env_info["max_action"], device,
                            ck["extra_kwargs"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    return cfg, env_info, model


def actor_for_loaded(cfg: RunConfig, model, device: str):
    if cfg.model.kind == "sequence":
        return policy_actor_factory(cfg, model, device)
    from experiments.baselines.mlp import mlp_actor_factory

    return mlp_actor_factory(model, cfg.model.algo, cfg.model.algo_kwargs.get("bc_mode", ""),
                             device)


def load_osrl_cdt(logdir: Path, task: str, device: str):
    """Policy from an original OSRL train_cdt.py log directory (E0 fidelity check)."""
    from osrl.common.exp_util import load_config_and_model

    from experiments.common.config import ModelConfig
    from experiments.common.data import env_constants
    from experiments.common.policy import build_policy, load_osrl_cdt_state

    ocfg, ck = load_config_and_model(str(logdir), best=False)
    info = env_constants(task)
    mc = ModelConfig(
        name="osrl_cdt", backbone="transformer", embedding_dim=ocfg["embedding_dim"],
        embedding_dropout=ocfg["embedding_dropout"], time_emb=ocfg["time_emb"],
        use_rew=ocfg["use_rew"], use_cost=ocfg["use_cost"],
        cost_transform=ocfg["cost_transform"], add_cost_feat=ocfg["add_cost_feat"],
        mul_cost_feat=ocfg["mul_cost_feat"], cat_cost_feat=ocfg["cat_cost_feat"],
        action_head_layers=ocfg["action_head_layers"], cost_prefix=ocfg["cost_prefix"],
        stochastic=ocfg["stochastic"], init_temperature=ocfg["init_temperature"],
        backbone_kwargs=dict(num_layers=ocfg["num_layers"], num_heads=ocfg["num_heads"],
                             attention_dropout=ocfg["attention_dropout"],
                             residual_dropout=ocfg["residual_dropout"]))
    policy = build_policy(mc, info["state_dim"], info["action_dim"], info["max_action"],
                          ocfg["seq_len"], ocfg["episode_len"])
    load_osrl_cdt_state(policy, ck["model_state"])
    return policy.to(device).eval(), ocfg, info


# ------------------------------------------------------------------ CLI
def main(argv: Optional[List[str]] = None) -> int:
    from experiments.common.config import parse_override
    from experiments.common.expdefs import resolve_run_config, resolve_targets

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True)
    p.add_argument("--ckpt-from", default=None, help="source experiment of the checkpoint")
    p.add_argument("--src-variant", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--osrl-ckpt", default=None, help="OSRL train_cdt.py log directory")
    p.add_argument("--override", nargs="+", action="extend", default=[])
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    run_dir = run_dir_for(args.out, args.exp, args.task, args.model, args.variant, args.seed)
    if is_done(run_dir):
        print(f"skip (DONE): {run_dir}")
        return 0
    write_marker(run_dir, RUNNING, job_tag())
    remove_marker(run_dir, FAILED)
    try:
        t0 = time.perf_counter()
        overrides = dict(parse_override(o) for o in args.override)
        if args.osrl_ckpt:
            model, ocfg, env_info = load_osrl_cdt(Path(args.osrl_ckpt), args.task, device)
            cfg = resolve_run_config(args.exp, "cdt", args.task, args.seed, None, args.out,
                                     overrides, device)
            cfg.model_id = args.model
            cfg.data.episode_len = int(ocfg["episode_len"])
            src = {"osrl_logdir": str(args.osrl_ckpt), "osrl_config": ocfg}
            env_info = dict(env_info, episode_len=cfg.data.episode_len)
            make_actor = policy_actor_factory(cfg, model, device)
        else:
            if not args.ckpt_from:
                raise SystemExit("--ckpt-from is required unless --osrl-ckpt is given")
            src_dir = run_dir_for(args.out, args.ckpt_from, args.task, args.model,
                                  args.src_variant, args.seed)
            if not is_done(src_dir):
                raise FileNotFoundError(f"source run not finished: {src_dir}")
            src_cfg, env_info, model = load_run(src_dir, device)
            cfg = resolve_run_config(args.exp, args.model, args.task, args.seed, args.variant,
                                     args.out, overrides, device)
            # model/data settings come from the checkpoint; evaluation from this exp
            cfg.data, cfg.model = src_cfg.data, src_cfg.model
            src = {"ckpt_dir": str(src_dir), "ckpt_exp": args.ckpt_from,
                   "ckpt_variant": args.src_variant}
            make_actor = actor_for_loaded(cfg, model, device)
        targets = resolve_targets(cfg.eval.target_set, cfg.meta["task_targets"],
                                  env_info.get("data_r_max"))
        cfg.eval.targets = targets
        cfg.meta.update(src)
        atomic_write_yaml(run_dir / "config.yaml", to_dict(cfg))
        atomic_write_json(run_dir / "env_info.json", env_info)
        rows, stats = evaluate_actor(cfg, make_actor, targets, cfg.eval.eval_episodes,
                                     cfg.eval.seed_split, env_info, device)
        log = CSVLog(run_dir / "eval_episodes.csv")
        log.write_rows(rows)
        atomic_write_json(run_dir / "run_summary.json",
                          {"eval_wall_s": time.perf_counter() - t0, "train_wall_s": 0.0,
                           "rollout": stats, "n_targets": len(targets), "source": src})
        write_marker(run_dir, DONE, job_tag())
    except BaseException:
        write_marker(run_dir, FAILED, job_tag() + "\n" + traceback.format_exc())
        raise
    finally:
        remove_marker(run_dir, RUNNING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
