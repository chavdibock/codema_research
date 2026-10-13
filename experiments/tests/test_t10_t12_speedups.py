"""T10 GPU sampler vs OSRL SequenceDataset, T11 vectorized == sequential evaluation,
T12 preprocessing cache."""

import dataclasses

import numpy as np
import pytest
import torch
from scipy.stats import chisquare

from experiments.common.data import (GPUWindowSampler, build_osrl_sequence_dataset,
                                     data_cache_key, get_sequence_data,
                                     load_preprocessed_uncached, sequence_data_from_osrl)
from experiments.common.utils import seed_all


@pytest.fixture(scope="module")
def osrl_ds(data_cfg, synthetic_data):
    data, meta = load_preprocessed_uncached(data_cfg, 0)
    ds = build_osrl_sequence_dataset(data, data_cfg)
    return ds, sequence_data_from_osrl(ds, meta)


# ------------------------------------------------------------------------ T10
def test_t10_identical_windows(osrl_ds, data_cfg):
    ds, sd = osrl_ds
    assert sd.meta["n_aug_traj"] > 0  # augmentation ran on the synthetic data
    s = GPUWindowSampler(sd, data_cfg.seq_len, data_cfg.reward_scale, data_cfg.cost_scale,
                         64, "cpu", seed=0)
    rng = np.random.default_rng(0)
    lens = sd.lengths()
    pairs = [(int(i), int(rng.integers(0, lens[i]))) for i in rng.integers(0, sd.num_traj, 300)]
    pairs += [(0, int(lens[0]) - 1), (sd.num_traj - 1, 0)]  # padded end / first step
    traj = torch.tensor([p[0] for p in pairs])
    start = torch.tensor([p[1] for p in pairs])
    ours = s.gather(traj, start)
    for k, (i, st) in enumerate(pairs):
        ref = ds._SequenceDataset__prepare_sample(i, st)
        for j, (a, b) in enumerate(zip(ours, ref)):
            np.testing.assert_allclose(a[k].numpy(), np.asarray(b, dtype=np.float64),
                                       rtol=1e-6, atol=1e-6, err_msg=f"field {j}")


def test_t10_same_sampling_distribution(osrl_ds, data_cfg):
    ds, sd = osrl_ds
    assert len(sd.sample_prob)
    np.testing.assert_allclose(sd.sample_prob, ds.sample_prob)
    s = GPUWindowSampler(sd, data_cfg.seq_len, data_cfg.reward_scale, data_cfg.cost_scale,
                         64, "cpu", seed=1)
    n = 200_000
    traj, start = s.sample_indices(n)
    counts = np.bincount(traj.numpy(), minlength=sd.num_traj)
    expected = n * sd.sample_prob
    keep = expected >= 5
    obs, exp = counts[keep], expected[keep]
    if (~keep).any() and expected[~keep].sum() > 0:  # pool the rare trajectories
        obs = np.append(obs, counts[~keep].sum())
        exp = np.append(exp, expected[~keep].sum())
    assert chisquare(obs, exp * obs.sum() / exp.sum()).pvalue > 1e-3
    # start index uniform on [0, len - 1] (OSRL: random.randint(0, len - 1))
    lens = sd.lengths()[traj.numpy()]
    st = start.numpy()
    assert (st >= 0).all() and (st <= lens - 1).all()
    full = lens == lens.max()
    c = np.bincount(st[full], minlength=int(lens.max()))
    assert chisquare(c).pvalue > 1e-3


# ------------------------------------------------------------------------ T11
@pytest.mark.slow
def test_t11_vectorized_equals_sequential():
    from experiments.common.rollout import run_rollouts, make_jobs, sequence_actor_factory
    from experiments.tests.conftest import small_policy

    torch.manual_seed(0)
    pol = small_policy().eval()
    jobs = make_jobs([(575.0, 10.0), (575.0, 20.0)], 3, 1000)
    seq, _ = run_rollouts("OfflineCarRun-v0", jobs,
                          sequence_actor_factory(pol, 0.1, 1.0, "cpu"), 200, vectorized=False)
    vec, _ = run_rollouts("OfflineCarRun-v0", jobs,
                          sequence_actor_factory(pol, 0.1, 1.0, "cpu", batch_invariant=True),
                          200, vectorized=True, num_workers=4)
    for a, b in zip(seq, vec):
        assert (a["env_seed"], a["target_C"]) == (b["env_seed"], b["target_C"])
        assert a["R"] == b["R"] and a["C"] == b["C"] and a["len"] == b["len"]


def test_t11_env_reset_is_seed_determined():
    """Same seed => identical episode, independent of the env's history."""
    from experiments.common.rollout import make_env

    def episode(env, seed):
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for t in range(30):
            obs, *_ = env.step(np.array([np.sin(t / 3), np.cos(t / 5)], dtype=np.float32))
            out.append(obs)
        return np.array(out)

    e1, e2 = make_env("OfflineCarRun-v0"), make_env("OfflineCarRun-v0")
    a = episode(e1, 7)
    episode(e1, 8)
    b = episode(e1, 7)  # same env, after another episode
    c = episode(e2, 7)  # fresh env
    assert np.array_equal(a, b) and np.array_equal(a, c)


# ------------------------------------------------------------------------ T12
def test_t12_cache_byte_identical(data_cfg, synthetic_data, tmp_path):
    from experiments.common.data import SequenceData

    sd1 = get_sequence_data(data_cfg, 0, tmp_path)  # computes + writes cache
    files = list((tmp_path / data_cfg.task).glob("seq_*.npz"))
    assert len(files) == 1
    sd2 = get_sequence_data(data_cfg, 0, tmp_path)  # loads from cache
    sd3 = get_sequence_data(data_cfg, 0, tmp_path, use_cache=False)  # recomputes
    for k in SequenceData.ARRAYS:
        a, b, c = getattr(sd1, k), getattr(sd2, k), getattr(sd3, k)
        assert a.dtype == b.dtype == c.dtype, k
        assert a.tobytes() == b.tobytes() == c.tobytes(), k


def test_t12_cache_key_covers_task_config_seed(data_cfg):
    base = data_cache_key(data_cfg, 0, "seq")
    assert data_cache_key(data_cfg, 1, "seq") != base
    assert data_cache_key(dataclasses.replace(data_cfg, task="OfflineAntRun-v0"), 0, "seq") != base
    assert data_cache_key(dataclasses.replace(data_cfg, augment_percent=0.0), 0, "seq") != base
    assert data_cache_key(dataclasses.replace(data_cfg, outliers_percent=0.01), 0, "pre") != \
        data_cache_key(data_cfg, 0, "pre")
    # episode_len does not change the data
    assert data_cache_key(dataclasses.replace(data_cfg, episode_len=123), 0, "seq") == base
