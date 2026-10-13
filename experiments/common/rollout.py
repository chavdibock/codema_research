"""Seeded evaluation: sequential and vectorized (S2/S8), window and recurrent modes.

Seeding. Bullet-Safety-Gym ignores ``reset(seed=...)``: its reset randomness comes from
the global ``np.random``. Moreover, a second episode on the same env instance is not
reproducible (pybullet's ``restoreState`` does not restore all agent state). Verified
locally: the first episode after construction is bit-identical across instances, later
ones are not. ``SeededReset`` therefore rebuilds the env (0.01-0.3 s) and seeds
``np.random``/``random`` on every seeded reset, so episode ``i`` is fully determined by
``eval_seed_base + i`` (identical across models, and between sequential and vectorized
evaluation: test T11).

Rewards. We step the raw env and keep returns-to-go in scaled units
(``R * reward_scale``), which is numerically what OSRL does with its reward-scaled
wrapper; reported episode rewards are raw.
"""

import os
import random
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from experiments.common.utils import available_cpus

Job = Tuple[float, float, int]  # (target reward, target cost, env seed)


def _import_envs() -> None:
    import bullet_safety_gym  # noqa: F401
    import dsrl  # noqa: F401


def _make_raw_env(task: str):
    import gymnasium as gym

    _import_envs()
    return gym.make(task)


def make_env(task: str):
    return SeededReset(_make_raw_env(task), partial(_make_raw_env, task))


try:
    import gymnasium as _gym

    class SeededReset(_gym.Wrapper):
        """``reset(seed=s)``: fresh env instance + seeded global numpy/python RNGs."""

        def __init__(self, env, env_fn: Callable):
            super().__init__(env)
            self._env_fn = env_fn

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            if seed is not None:
                self.env.close()
                self.env = self._env_fn()
                np.random.seed(seed)
                random.seed(seed)
            return self.env.reset(seed=seed, options=options)

except ImportError:  # pragma: no cover - gymnasium is a hard dependency at runtime
    SeededReset = None  # type: ignore


def make_jobs(targets: Sequence[Sequence[float]], episodes: int, seed_base: int) -> List[Job]:
    """targets x episodes; episode i of every target uses env seed seed_base + i."""
    return [(float(r), float(c), seed_base + i) for (r, c) in targets for i in range(episodes)]


# --------------------------------------------------------------------- actors
class WindowSequenceActor:
    """Batched version of OSRL's ``CDTTrainer.rollout`` (window of the last K steps)."""

    def __init__(self, policy, target_R: np.ndarray, target_C: np.ndarray,
                 reward_scale: float, cost_scale: float, device: str, sample: bool = False):
        n = len(target_R)
        ep, dev = policy.episode_len, device
        self.policy, self.device, self.sample = policy, dev, sample
        self.rs, self.cs = reward_scale, cost_scale
        self.K = policy.seq_len
        self.states = torch.zeros(n, ep + 1, policy.state_dim, device=dev)
        self.actions = torch.zeros(n, ep, policy.action_dim, device=dev)
        self.returns = torch.zeros(n, ep + 1, device=dev)
        self.costs = torch.zeros(n, ep + 1, device=dev)
        self.time_steps = torch.arange(ep, dtype=torch.long, device=dev).view(1, -1).expand(n, -1)
        self.returns[:, 0] = torch.as_tensor(target_R * reward_scale, device=dev)
        self.costs[:, 0] = torch.as_tensor(target_C * cost_scale, device=dev)
        self.epi_cost = torch.as_tensor(target_C * cost_scale, dtype=torch.float, device=dev)

    def reset(self, obs: np.ndarray) -> None:
        self.states[:, 0] = torch.as_tensor(obs, device=self.device)

    def window(self, t: int):
        sl = slice(max(0, t + 1 - self.K), t + 1)
        return (self.states[:, sl], self.actions[:, sl], self.returns[:, sl],
                self.costs[:, sl], self.time_steps[:, sl])

    @torch.no_grad()
    def act(self, t: int) -> torch.Tensor:
        s, a, r, c, ts = self.window(t)
        acts, _, _ = self.policy(s, a, r, c, ts, None, self.epi_cost)
        if self.policy.stochastic:
            acts = acts.sample() if self.sample else acts.mean
        acts = acts.clamp(-self.policy.max_action, self.policy.max_action)
        return acts[:, -1]

    def observe(self, t: int, act: torch.Tensor, obs_next: np.ndarray, reward: np.ndarray,
                cost: np.ndarray) -> None:
        dev = self.device
        self.actions[:, t] = act
        self.states[:, t + 1] = torch.as_tensor(obs_next, device=dev)
        self.returns[:, t + 1] = self.returns[:, t] - torch.as_tensor(reward * self.rs,
                                                                      device=dev)
        self.costs[:, t + 1] = self.costs[:, t] - torch.as_tensor(cost * self.cs, device=dev)


class RecurrentSequenceActor:
    """Recurrent mode (E6.3): carry the backbone state across env steps."""

    def __init__(self, policy, target_R: np.ndarray, target_C: np.ndarray,
                 reward_scale: float, cost_scale: float, device: str, sample: bool = False):
        self.policy, self.device, self.sample = policy, device, sample
        self.rs, self.cs = reward_scale, cost_scale
        n = len(target_R)
        self.R = torch.as_tensor(target_R * reward_scale, dtype=torch.float, device=device)
        self.C = torch.as_tensor(target_C * cost_scale, dtype=torch.float, device=device)
        self.rstate = policy.init_recurrent(n, device=device)
        self.obs = None
        self.n = n

    def reset(self, obs: np.ndarray) -> None:
        self.obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def act(self, t: int) -> torch.Tensor:
        tt = torch.full((self.n, ), t, dtype=torch.long, device=self.device)
        dist, self.rstate = self.policy.recurrent_observe(self.obs, self.R, self.C, tt,
                                                          self.rstate)
        if self.policy.stochastic:
            acts = dist.sample() if self.sample else dist.mean
        else:
            acts = dist
        return acts.clamp(-self.policy.max_action, self.policy.max_action)

    @torch.no_grad()
    def observe(self, t: int, act: torch.Tensor, obs_next: np.ndarray, reward: np.ndarray,
                cost: np.ndarray) -> None:
        tt = torch.full((self.n, ), t, dtype=torch.long, device=self.device)
        self.rstate = self.policy.recurrent_act(act, tt, self.rstate)
        self.R = self.R - torch.as_tensor(reward * self.rs, dtype=torch.float, device=self.device)
        self.C = self.C - torch.as_tensor(cost * self.cs, dtype=torch.float, device=self.device)
        self.obs = torch.as_tensor(obs_next, dtype=torch.float, device=self.device)


class PerEnvActor:
    """Batch-invariant mode: n independent batch-1 actors (same buffers and kernels as a
    sequential rollout, hence bit-identical results; fp32 matmuls round differently for
    different batch sizes and memory layouts)."""

    def __init__(self, actors: List[Any]):
        self.actors = actors

    def reset(self, obs: np.ndarray) -> None:
        for i, a in enumerate(self.actors):
            a.reset(obs[i:i + 1])

    def act(self, t: int):
        outs = [a.act(t) for a in self.actors]
        if torch.is_tensor(outs[0]):
            return torch.cat(outs, dim=0)
        return np.concatenate(outs, axis=0)

    def observe(self, t: int, act, obs_next: np.ndarray, reward: np.ndarray,
                cost: np.ndarray) -> None:
        for i, a in enumerate(self.actors):
            a.observe(t, act[i:i + 1], obs_next[i:i + 1], reward[i:i + 1], cost[i:i + 1])


def per_env(make_actor: Callable) -> Callable:
    """Wrap an actor factory so that every env gets its own batch-1 actor."""

    def make(target_R: np.ndarray, target_C: np.ndarray):
        return PerEnvActor([make_actor(target_R[i:i + 1], target_C[i:i + 1])
                            for i in range(len(target_R))])

    return make


def sequence_actor_factory(policy, reward_scale: float, cost_scale: float, device: str,
                           sample: bool = False, mode: str = "window",
                           batch_invariant: bool = False) -> Callable:

    cls = WindowSequenceActor if mode == "window" else RecurrentSequenceActor

    def make(target_R: np.ndarray, target_C: np.ndarray):
        return cls(policy, target_R, target_C, reward_scale, cost_scale, device, sample)

    return per_env(make) if batch_invariant else make


# ------------------------------------------------------------------- envs
class _SingleVec:
    """One env with the batched interface used below (no autoreset)."""

    def __init__(self, env):
        self.env = env
        self.num_envs = 1

    def reset(self, seed: List[int]):
        obs, info = self.env.reset(seed=int(seed[0]))
        return obs[None], info

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, info = self.env.step(actions[0])
        infos = {"cost": np.array([info.get("cost", 0.0)]), "_cost": np.array([True])}
        return (obs[None], np.array([rew], dtype=np.float64), np.array([term]),
                np.array([trunc]), infos)

    def close(self) -> None:
        self.env.close()


def _extract_cost(infos: Dict[str, Any], n: int) -> np.ndarray:
    """Per-env cost of this step; with gymnasium<1 autoreset the terminal step's info is
    under ``final_info``."""
    c = np.zeros(n, dtype=np.float64)
    if "cost" in infos:
        mask = np.asarray(infos.get("_cost", np.ones(n, dtype=bool)), dtype=bool)
        c[mask] = np.asarray(infos["cost"], dtype=np.float64)[mask]
    if "final_info" in infos:
        fmask = np.asarray(infos.get("_final_info", np.ones(n, dtype=bool)), dtype=bool)
        for i in np.where(fmask)[0]:
            fi = infos["final_info"][i]
            if fi is not None and "cost" in fi:
                c[i] = float(fi["cost"])
    return c


def default_num_workers() -> int:
    env = os.environ.get("CDX_EVAL_WORKERS")
    return max(1, int(env)) if env else max(1, min(available_cpus(), 32))


def run_rollouts(task: str,
                 jobs: List[Job],
                 make_actor: Callable,
                 episode_len: int,
                 vectorized: bool = True,
                 num_workers: int = 0,
                 device: str = "cpu",
                 context: Optional[str] = "fork",
                 record_latency: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run every job once; return per-episode rows and timing stats.

    Vectorized: ``W`` AsyncVectorEnv workers step together with one batched policy
    forward per time step; jobs are processed in chunks of ``W`` (padded chunks are
    ignored). Sequential: one env, one episode at a time (OSRL behaviour).
    """
    if not jobs:
        return [], {}
    if vectorized and len(jobs) > 1:
        from gymnasium.vector import AsyncVectorEnv

        W = min(len(jobs), num_workers or default_num_workers())
        venv = AsyncVectorEnv([partial(make_env, task) for _ in range(W)], context=context)
    else:
        W = 1
        venv = _SingleVec(make_env(task))
    sync = device.startswith("cuda") and torch.cuda.is_available()
    rows: List[Dict[str, Any]] = []
    act_times: List[float] = []
    t_start = time.perf_counter()
    try:
        for c0 in range(0, len(jobs), W):
            chunk = jobs[c0:c0 + W]
            m = len(chunk)
            padded = chunk + [chunk[-1]] * (W - m)
            tR = np.array([j[0] for j in padded], dtype=np.float64)
            tC = np.array([j[1] for j in padded], dtype=np.float64)
            actor = make_actor(tR, tC)
            obs, _ = venv.reset(seed=[int(j[2]) for j in padded])
            actor.reset(obs)
            ret = np.zeros(W)
            cst = np.zeros(W)
            length = np.zeros(W, dtype=np.int64)
            done = np.zeros(W, dtype=bool)
            for t in range(episode_len):
                t0 = time.perf_counter()
                act = actor.act(t)
                act_np = act.detach().float().cpu().numpy() if torch.is_tensor(act) else act
                if sync:
                    torch.cuda.synchronize()
                if record_latency:
                    act_times.append(time.perf_counter() - t0)
                obs, rew, term, trunc, infos = venv.step(act_np.astype(np.float32))
                cost = _extract_cost(infos, W)
                rew = np.asarray(rew, dtype=np.float64)
                active = ~done
                ret += rew * active
                cst += cost * active
                length += active
                actor.observe(t, act, obs, rew, cost)
                done |= np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)
                if done[:m].all():
                    break
            for i in range(m):
                rows.append({"target_R": chunk[i][0], "target_C": chunk[i][1],
                             "env_seed": chunk[i][2], "R": float(ret[i]), "C": float(cst[i]),
                             "len": int(length[i])})
    finally:
        venv.close()
    stats = {"rollout_wall_s": time.perf_counter() - t_start, "num_workers": W,
             "n_episodes": len(jobs)}
    if act_times:
        a = np.asarray(act_times) * 1e3
        stats.update({"act_ms_median": float(np.median(a)), "act_ms_p95": float(np.percentile(a, 95)),
                      "act_batch": W, "act_calls": int(a.size)})
    return rows, stats
