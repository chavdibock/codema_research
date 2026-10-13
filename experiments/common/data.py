"""Thin wrapper around DSRL ``pre_process_data`` + OSRL ``SequenceDataset`` with a
preprocessing cache (S5) and a GPU-resident window sampler (S1).

CLI:
    python -m experiments.common.data --download --tasks all        # login node only
    python -m experiments.common.data --prepare --exp E1 [--workers 8]
"""

import argparse
import contextlib
import copy
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")  # DSRL's pre_process_data saves a figure

import numpy as np
import torch

from experiments.common.config import TASKS, DataConfig
from experiments.common.utils import atomic_write_json, seed_all, stable_hash

CACHE_VERSION = "v1"


# ----------------------------------------------------------------------------- env
def _import_envs() -> None:
    import bullet_safety_gym  # noqa: F401
    import dsrl  # noqa: F401  (registers Offline*-v0)


def env_spec(task: str):
    import gymnasium as gym

    _import_envs()
    return gym.spec(task)


def env_constants(task: str) -> Dict[str, Any]:
    """Normalization constants and dims (DSRL registration + one env construction)."""
    import gymnasium as gym

    spec = env_spec(task)
    kw = spec.kwargs
    env = gym.make(task)
    try:
        info = {
            "task": task,
            "max_episode_steps": int(spec.max_episode_steps),
            "max_episode_reward": float(kw["max_episode_reward"]),
            "min_episode_reward": float(kw["min_episode_reward"]),
            "max_episode_cost": float(kw["max_episode_cost"]),
            "min_episode_cost": float(kw["min_episode_cost"]),
            "state_dim": int(env.observation_space.shape[0]),
            "action_dim": int(env.action_space.shape[0]),
            "max_action": float(env.action_space.high[0]),
            "dataset_url": kw["dataset_url"],
        }
    finally:
        env.close()
    return info


def dataset_path(task: str) -> Path:
    from dsrl.offline_env import filepath_from_url

    return Path(filepath_from_url(env_spec(task).kwargs["dataset_url"]))


def download_datasets(tasks: List[str]) -> None:
    """Login node only: fetch the DSRL hdf5 files into $DSRL_DATASET_DIR."""
    from dsrl.offline_env import download_dataset_from_url

    for task in tasks:
        url = env_spec(task).kwargs["dataset_url"]
        path = download_dataset_from_url(url)
        print(f"{task}: {path} ({Path(path).stat().st_size / 2**20:.1f} MiB)")


def _require_local_dataset(task: str) -> None:
    path = dataset_path(task)
    if not path.exists() and os.environ.get("CDX_ALLOW_DOWNLOAD", "0") != "1":
        raise FileNotFoundError(
            f"{path} missing. Download on the login node first "
            "(python -m experiments.common.data --download), compute nodes must not "
            "download; DSRL_DATASET_DIR=" + os.environ.get("DSRL_DATASET_DIR", "<unset>"))


@contextlib.contextmanager
def _in_tmp_cwd():
    """DSRL's pre_process_data writes a png into the CWD; keep it out of the tree."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="cdx_pre_") as tmp:
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(old)


# ------------------------------------------------------------ trajectory helpers
def split_trajectories(data: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
    done = np.where((data["terminals"] == 1) | (data["timeouts"] == 1))[0]
    starts = np.concatenate([[0], done[:-1] + 1])
    return list(zip(starts.tolist(), (done + 1).tolist()))


def trajectory_returns(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    seg = split_trajectories(data)
    c = np.array([data["costs"][s:e].sum() for s, e in seg], dtype=np.float64)
    r = np.array([data["rewards"][s:e].sum() for s, e in seg], dtype=np.float64)
    return c, r


def epsilon_hat(cost_ret: np.ndarray, rew_ret: np.ndarray, kappa: float) -> Dict[str, float]:
    """[CDT] Def. 1: PF(k) = max R s.t. C <= k, IPF(k) = max R s.t. C >= k,
    eps = PF - IPF, normalized by the maximum reward return of the dataset."""
    pf = rew_ret[cost_ret <= kappa].max() if np.any(cost_ret <= kappa) else np.nan
    ipf = rew_ret[cost_ret >= kappa].max() if np.any(cost_ret >= kappa) else np.nan
    eps = pf - ipf
    return {"PF": float(pf), "IPF": float(ipf), "eps": float(eps),
            "eps_hat": float(eps / rew_ret.max())}


def _select_trajectories(data: Dict[str, np.ndarray], keep: np.ndarray) -> Dict[str, np.ndarray]:
    seg = split_trajectories(data)
    idx = np.concatenate([np.arange(seg[i][0], seg[i][1]) for i in np.where(keep)[0]])
    return {k: v[idx] for k, v in data.items()}


def reduce_dataset_cdt(data: Dict[str, np.ndarray], kappa: float) -> Dict[str, np.ndarray]:
    """[CDT] App. A reduced dataset: remove trajectories whose cost exceeds kappa and
    whose reward is higher than that of the trajectories with cost == kappa.

    NOTE: "reward of the trajectories with cost == kappa" = RF(kappa) = max reward among
    trajectories with C == kappa; if none has C == kappa exactly, we fall back to
    PF(kappa) = max reward with C <= kappa. Documented in NOTES.md.
    """
    c, r = trajectory_returns(data)
    at_k = np.isclose(c, kappa)
    ref = r[at_k].max() if at_k.any() else r[c <= kappa].max()
    keep = ~((c > kappa) & (r > ref))
    return _select_trajectories(data, keep)


def post_density_filter(data: Dict[str, np.ndarray], task: str,
                        density: float) -> Dict[str, np.ndarray]:
    """E8 density mitigation: DSRL's grid filter with its DENSITY_CFG for this task,
    applied after the outlier injection. NOTE: DSRL's own ``density`` option runs
    *before* outliers are injected and therefore cannot remove them."""
    from dsrl.infos import DENSITY_CFG
    from dsrl.offline_env import grid_filter

    cfg = DENSITY_CFG[f"{task}_density{density}"]
    c, r = trajectory_returns(data)
    idx = grid_filter(c, r, xbins=cfg["cbins"], ybins=cfg["rbins"],
                      max_num_per_bin=cfg["max_npb"], min_num_per_bin=cfg["min_npb"])
    keep = np.zeros(len(c), dtype=bool)
    keep[np.asarray(idx, dtype=int)] = True
    return _select_trajectories(data, keep)


# ------------------------------------------------------------ preprocessing (S5)
def load_preprocessed_uncached(cfg: DataConfig, seed: int) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Same call sequence as OSRL's train_cdt.py: seed_all -> get_dataset ->
    set_target_cost -> pre_process_data (+ our E7/E8 options)."""
    import gymnasium as gym
    from dsrl.infos import DENSITY_CFG

    _import_envs()
    _require_local_dataset(cfg.task)
    seed_all(seed)
    env = gym.make(cfg.task)
    raw = env.get_dataset()
    env.set_target_cost(cfg.cost_limit)
    c0, r0 = trajectory_returns(raw)
    cbins, rbins, max_npb, min_npb = None, None, None, None
    if cfg.density != 1.0:
        dcfg = DENSITY_CFG[cfg.task + "_density" + str(cfg.density)]
        cbins, rbins = dcfg["cbins"], dcfg["rbins"]
        max_npb, min_npb = dcfg["max_npb"], dcfg["min_npb"]
    inpaint = None if cfg.inpaint_ranges is None else tuple(map(tuple, cfg.inpaint_ranges))
    with _in_tmp_cwd():
        data = env.unwrapped.pre_process_data(raw, cfg.outliers_percent, cfg.noise_scale,
                                              inpaint, cfg.epsilon, cfg.density,
                                              cbins=cbins, rbins=rbins, max_npb=max_npb,
                                              min_npb=min_npb)
    env.close()
    if cfg.reduce_rule == "cdt":
        data = reduce_dataset_cdt(data, cfg.cost_limit)
    elif cfg.reduce_rule != "none":
        raise ValueError(f"unknown reduce_rule {cfg.reduce_rule}")
    if cfg.post_density != 1.0:
        data = post_density_filter(data, cfg.task, cfg.post_density)
    c1, r1 = trajectory_returns(data)
    meta = {
        "raw_n_traj": int(len(c0)), "raw_n_transitions": int(len(raw["rewards"])),
        "raw_r_max": float(r0.max()), "raw_r_min": float(r0.min()),
        "raw_c_max": float(c0.max()), "raw_c_min": float(c0.min()),
        "n_traj": int(len(c1)), "n_transitions": int(len(data["rewards"])),
        "data_r_max": float(r1.max()), "data_r_min": float(r1.min()),
        "data_c_max": float(c1.max()), "data_c_min": float(c1.min()),
        "eps_raw": epsilon_hat(c0, r0, cfg.cost_limit),
        "eps_data": epsilon_hat(c1, r1, cfg.cost_limit),
    }
    return data, meta


def _sequence_dataset_kwargs(cfg: DataConfig) -> Dict[str, Any]:
    ct = (lambda x: 70 - x) if cfg.linear else (lambda x: 1 / (x + 10))
    return dict(seq_len=cfg.seq_len, reward_scale=cfg.reward_scale, cost_scale=cfg.cost_scale,
                deg=cfg.deg, pf_sample=cfg.pf_sample, max_rew_decrease=cfg.max_rew_decrease,
                beta=cfg.beta, augment_percent=cfg.augment_percent,
                cost_reverse=cfg.cost_reverse, max_reward=cfg.max_reward,
                min_reward=cfg.min_reward, pf_only=cfg.pf_only, rmin=cfg.rmin,
                cost_bins=cfg.cost_bins, npb=cfg.npb, cost_sample=cfg.cost_sample,
                cost_transform=ct, start_sampling=cfg.start_sampling, prob=cfg.prob,
                random_aug=cfg.random_aug, aug_rmin=cfg.aug_rmin, aug_rmax=cfg.aug_rmax,
                aug_cmin=cfg.aug_cmin, aug_cmax=cfg.aug_cmax, cgap=cfg.cgap,
                rstd=cfg.rstd, cstd=cfg.cstd)


def build_osrl_sequence_dataset(data: Dict[str, np.ndarray], cfg: DataConfig):
    """OSRL's SequenceDataset with the per-task OSRL arguments (unchanged code)."""
    from osrl.common import SequenceDataset

    return SequenceDataset(data, **_sequence_dataset_kwargs(cfg))


@dataclass
class SequenceData:
    """Flattened trajectories (original + augmented) as produced by SequenceDataset."""
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    cost: np.ndarray
    rtg: np.ndarray  # unscaled reward-to-go (relabelled for augmented trajectories)
    ctg: np.ndarray  # unscaled cost-to-go
    offsets: np.ndarray  # [N + 1]
    sample_prob: np.ndarray  # [N] or empty (uniform)
    is_aug: np.ndarray  # [N] bool
    aug_source: np.ndarray  # [N] index of the original trajectory (-1 for originals)
    orig_cost_ret: np.ndarray  # per original trajectory
    orig_rew_ret: np.ndarray
    pareto_coef: np.ndarray  # poly1d coefficients of the fitted frontier (or empty)
    meta: Dict[str, Any] = field(default_factory=dict)

    ARRAYS = ("obs", "act", "rew", "cost", "rtg", "ctg", "offsets", "sample_prob", "is_aug",
              "aug_source", "orig_cost_ret", "orig_rew_ret", "pareto_coef")

    @property
    def num_traj(self) -> int:
        return len(self.offsets) - 1

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


def sequence_data_from_osrl(ds, meta: Dict[str, Any]) -> SequenceData:
    trajs = ds.dataset
    n_orig = len(ds.original_data)
    lens = np.array([len(t["rewards"]) for t in trajs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    cat = lambda k: np.concatenate([t[k] for t in trajs]).astype(np.float32)  # noqa: E731
    is_aug = np.zeros(len(trajs), dtype=bool)
    is_aug[n_orig:] = True
    aug_source = -np.ones(len(trajs), dtype=np.int64)
    pareto = np.zeros(0)
    if len(ds.aug_data) and hasattr(ds, "indices"):
        # ds.idx indexes the grid-filtered list; ds.indices maps it to the original list
        aug_source[n_orig:] = np.asarray(ds.indices)[np.asarray(ds.idx)]
        pareto = np.asarray(ds.pareto_frontier.coefficients, dtype=np.float64)
    orig_c = np.array([t["cost_returns"][0] for t in ds.original_data], dtype=np.float64)
    orig_r = np.array([t["returns"][0] for t in ds.original_data], dtype=np.float64)
    sp = np.zeros(0) if ds.sample_prob is None else np.asarray(ds.sample_prob, dtype=np.float64)
    meta = dict(meta)
    meta.update({"n_orig_traj": n_orig, "n_aug_traj": len(ds.aug_data),
                 "n_traj_total": len(trajs), "max_traj_len": int(lens.max())})
    return SequenceData(obs=cat("observations"), act=cat("actions"), rew=cat("rewards"),
                        cost=cat("costs"), rtg=cat("returns"), ctg=cat("cost_returns"),
                        offsets=offsets, sample_prob=sp, is_aug=is_aug, aug_source=aug_source,
                        orig_cost_ret=orig_c, orig_rew_ret=orig_r, pareto_coef=pareto,
                        meta=meta)


PREPROCESS_FIELDS = ("task", "cost_limit", "outliers_percent", "noise_scale", "inpaint_ranges",
                     "epsilon", "density", "reduce_rule", "post_density")


def data_cache_key(cfg: DataConfig, seed: int, kind: str) -> str:
    """Key = task + data-config + seed (T12). 'pre' caches depend only on the
    pre-processing fields; 'seq' caches on the whole data config."""
    d = asdict(cfg)
    d.pop("episode_len", None)  # does not change the data
    if kind == "pre":
        d = {k: d[k] for k in PREPROCESS_FIELDS}
    return f"{kind}_{stable_hash({'cfg': d, 'seed': seed, 'v': CACHE_VERSION})}"


def default_cache_dir() -> Path:
    return Path(os.environ.get("CDX_CACHE", "cache"))


def _save_npz(path: Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    atomic_write_json(path.with_suffix(".json"), meta)


def _load_meta(path: Path) -> Dict[str, Any]:
    import json

    with open(path.with_suffix(".json")) as f:
        return json.load(f)


def build_sequence_data(cfg: DataConfig, seed: int) -> SequenceData:
    """Recompute from scratch (always re-runs pre-processing so the RNG stream matches
    OSRL's: seed_all -> pre_process_data -> SequenceDataset)."""
    data, meta = load_preprocessed_uncached(cfg, seed)
    ds = build_osrl_sequence_dataset(data, cfg)
    return sequence_data_from_osrl(ds, meta)


def get_sequence_data(cfg: DataConfig, seed: int, cache_dir: Optional[Path] = None,
                      use_cache: bool = True) -> SequenceData:
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    key = data_cache_key(cfg, seed, "seq")
    path = cache_dir / cfg.task / f"{key}.npz"
    if use_cache and path.exists():
        with np.load(path) as z:
            arrays = {k: z[k] for k in SequenceData.ARRAYS}
        return SequenceData(**arrays, meta=_load_meta(path))
    sd = build_sequence_data(cfg, seed)
    meta = dict(sd.meta, cache_key=key, seed=seed, data_cfg=asdict(cfg))
    sd.meta = meta
    if use_cache:
        _save_npz(path, {k: getattr(sd, k) for k in SequenceData.ARRAYS}, meta)
    return sd


TRANSITION_KEYS = ("observations", "next_observations", "actions", "rewards", "costs",
                   "terminals", "timeouts")


def get_transition_data(cfg: DataConfig, seed: int, cache_dir: Optional[Path] = None,
                        use_cache: bool = True) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Pre-processed transitions (for the MLP baselines)."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    key = data_cache_key(cfg, seed, "pre")
    path = cache_dir / cfg.task / f"{key}.npz"
    if use_cache and path.exists():
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        return data, _load_meta(path)
    data, meta = load_preprocessed_uncached(cfg, seed)
    data = {k: data[k] for k in TRANSITION_KEYS}
    meta = dict(meta, cache_key=key, seed=seed, data_cfg=asdict(cfg))
    if use_cache:
        _save_npz(path, data, meta)
    return data, meta


# ------------------------------------------------------------ GPU sampler (S1)
class GPUWindowSampler:
    """Distributionally identical replacement for ``SequenceDataset.__iter__`` (T10).

    Trajectories are padded to ``T_max + K`` and stored as device tensors. One call
    draws ``batch_size`` windows with ``torch.multinomial`` over OSRL's ``sample_prob``,
    uniform start indices in ``[0, len_i - 1]`` and one advanced-indexing gather.
    """

    def __init__(self, sd: SequenceData, seq_len: int, reward_scale: float,
                 cost_scale: float, batch_size: int, device: str, seed: int):
        self.K = seq_len
        self.B = batch_size
        self.device = torch.device(device)
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        lens = sd.lengths()
        n, t_pad = len(lens), int(lens.max()) + seq_len

        def pad(flat: np.ndarray) -> torch.Tensor:
            shape = (n, t_pad) + flat.shape[1:]
            out = np.zeros(shape, dtype=np.float32)
            for i in range(n):
                s, e = sd.offsets[i], sd.offsets[i + 1]
                out[i, :e - s] = flat[s:e]
            return torch.from_numpy(out).to(self.device)

        self.obs = pad(sd.obs)
        self.act = pad(sd.act)
        self.rtg = pad(sd.rtg)
        self.ctg = pad(sd.ctg)
        self.cost = pad(sd.cost)
        self.len = torch.as_tensor(lens, device=self.device)
        self.ep_cost = torch.as_tensor(sd.ctg[sd.offsets[:-1]], device=self.device)
        if len(sd.sample_prob):
            self.prob = torch.as_tensor(sd.sample_prob, dtype=torch.float64, device=self.device)
        else:
            self.prob = None
        self.arange = torch.arange(seq_len, device=self.device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.obs, self.act, self.rtg, self.ctg, self.cost))

    def sample_indices(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prob is None:
            traj = torch.randint(0, self.len.numel(), (n, ), generator=self.gen,
                                 device=self.device)
        else:
            traj = torch.multinomial(self.prob, n, replacement=True, generator=self.gen)
        u = torch.rand(n, generator=self.gen, device=self.device, dtype=torch.float64)
        start = torch.minimum((u * self.len[traj]).long(), self.len[traj] - 1)
        return traj, start

    def gather(self, traj: torch.Tensor, start: torch.Tensor):
        """Windows for given (traj, start) pairs in the order of OSRL's batch tuple:
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs."""
        idx = start[:, None] + self.arange[None, :]
        t = traj[:, None]
        mask = (self.arange[None, :] < (self.len[traj] - start)[:, None]).float()
        return (self.obs[t, idx], self.act[t, idx], self.rtg[t, idx] * self.reward_scale,
                self.ctg[t, idx] * self.cost_scale, idx, mask,
                self.ep_cost[traj] * self.cost_scale, self.cost[t, idx])

    def sample(self):
        return self.gather(*self.sample_indices(self.B))

    def state_dict(self) -> Dict[str, Any]:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gen.set_state(state["gen"].cpu())


class OSRLLoaderSampler:
    """Reference path (``train.sampler=osrl``): OSRL SequenceDataset + DataLoader."""

    def __init__(self, ds, batch_size: int, device: str, num_workers: int):
        from torch.utils.data import DataLoader

        self.device = device
        self.it = iter(DataLoader(ds, batch_size=batch_size, pin_memory=True,
                                  num_workers=num_workers))

    def sample(self):
        b = next(self.it)
        out = [x.to(self.device, non_blocking=True) for x in b]
        out[2], out[3], out[5], out[6] = (out[2].float(), out[3].float(), out[5].float(),
                                          out[6].float())
        return tuple(out)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass


class GPUTransitionSampler:
    """Uniform transition sampler (== OSRL TransitionDataset with sample_prob=None)."""

    def __init__(self, data: Dict[str, np.ndarray], reward_scale: float, cost_scale: float,
                 batch_size: int, device: str, seed: int, state_init: bool = False):
        self.device = torch.device(device)
        self.B = batch_size
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)  # noqa
        done = np.logical_or(data["terminals"], data["timeouts"]).astype(np.float32)
        self.obs = t(data["observations"])
        self.next_obs = t(data["next_observations"])
        self.act = t(data["actions"])
        self.rew = t(data["rewards"]) * reward_scale
        self.cost = t(data["costs"]) * cost_scale
        self.done = t(done)
        self.state_init = state_init
        if state_init:
            is_init = done.copy()
            is_init[1:] = is_init[:-1]
            is_init[0] = 1.0
            self.is_init = t(is_init)
        self.n = self.obs.shape[0]
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    def sample(self):
        i = torch.randint(0, self.n, (self.B, ), generator=self.gen, device=self.device)
        out = (self.obs[i], self.next_obs[i], self.act[i], self.rew[i], self.cost[i],
               self.done[i])
        return out + (self.is_init[i], ) if self.state_init else out

    def state_dict(self) -> Dict[str, Any]:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gen.set_state(state["gen"].cpu())


# ------------------------------------------------------------------------- CLI
def _prepare_one(args: Tuple[Dict[str, Any], int, str, str]) -> str:
    cfg_dict, seed, kind, cache_dir = args
    cfg = DataConfig(**cfg_dict)
    if kind == "seq":
        sd = get_sequence_data(cfg, seed, Path(cache_dir))
        return f"seq {cfg.task} seed={seed} traj={sd.num_traj}"
    data, _ = get_transition_data(cfg, seed, Path(cache_dir))
    return f"pre {cfg.task} seed={seed} transitions={len(data['rewards'])}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--tasks", nargs="*", default=["all"])
    p.add_argument("--exp", nargs="*", default=["E1"])
    p.add_argument("--cache-dir", default=str(default_cache_dir()))
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    tasks = TASKS if args.tasks == ["all"] else args.tasks
    if args.download:
        download_datasets(tasks)
    if args.prepare:
        from experiments.common.expdefs import data_jobs_for

        jobs = []
        for exp in args.exp:
            for cfg, seed, kind in data_jobs_for(exp):
                if cfg.task in tasks:
                    jobs.append((asdict(cfg), seed, kind, args.cache_dir))
        uniq = {stable_hash(j[:3]): j for j in jobs}
        jobs = list(uniq.values())
        print(f"preparing {len(jobs)} cache entries with {args.workers} workers")
        if args.workers > 1:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(args.workers) as pool:
                for msg in pool.imap_unordered(_prepare_one, jobs):
                    print(msg, flush=True)
        else:
            for j in jobs:
                print(_prepare_one(j), flush=True)


if __name__ == "__main__":
    main()
