"""Synthetic DSRL-format datasets for CPU tests and smoke runs (no download needed).

The files have the keys/shapes of the real hdf5 files and are written where DSRL looks
for them (``DSRL_DATASET_DIR``). Their content is random: never use them for results.
"""

import os
from pathlib import Path

import h5py
import numpy as np


def write_synthetic_dataset(task: str, out_dir: Path, n_traj: int = 60, seed: int = 0) -> Path:
    import gymnasium as gym

    import bullet_safety_gym  # noqa: F401
    import dsrl  # noqa: F401
    from dsrl.offline_env import filepath_from_url, set_dataset_path

    out_dir = Path(out_dir)
    set_dataset_path(str(out_dir))
    os.environ["DSRL_DATASET_DIR"] = str(out_dir)
    spec = gym.spec(task)
    path = Path(filepath_from_url(spec.kwargs["dataset_url"]))
    if path.exists():
        return path
    env = gym.make(task)
    ds, da = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    T = int(spec.max_episode_steps)
    rng = np.random.default_rng(seed)
    obs, nobs, act, rew, cost, term, tout = [], [], [], [], [], [], []
    for i in range(n_traj):
        L = T if i % 5 else int(rng.integers(T // 2, T))  # some short episodes
        p_cost = rng.uniform(0.0, 0.3)  # spread of cost returns
        o = rng.normal(size=(L + 1, ds)).astype(np.float32)
        obs.append(o[:-1])
        nobs.append(o[1:])
        act.append(rng.uniform(-1, 1, size=(L, da)).astype(np.float32))
        rew.append((rng.uniform(0, 2, size=L) + 4 * p_cost).astype(np.float32))
        cost.append((rng.uniform(size=L) < p_cost).astype(np.float32))
        t = np.zeros(L, dtype=np.float32)
        to = np.zeros(L, dtype=np.float32)
        if L < T:
            t[-1] = 1
        else:
            to[-1] = 1
        term.append(t)
        tout.append(to)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in (("observations", obs), ("next_observations", nobs), ("actions", act),
                     ("rewards", rew), ("costs", cost), ("terminals", term),
                     ("timeouts", tout)):
            f.create_dataset(k, data=np.concatenate(v))
    return path
