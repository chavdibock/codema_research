"""T6 rollout target updates, T7 determinism, T8 metrics."""

import numpy as np
import pytest
import torch

from experiments.common.metrics import (budget_tracking_error, episode_metrics,
                                        normalized_cost, normalized_reward)
from experiments.common.rollout import (RecurrentSequenceActor, WindowSequenceActor, make_jobs)
from experiments.tests.conftest import small_policy


# ------------------------------------------------------------------------- T6
def test_t6_target_updates_window_actor():
    pol = small_policy().eval()
    rs, cs = 0.1, 1.0
    actor = WindowSequenceActor(pol, np.array([300.0]), np.array([2.0]), rs, cs, "cpu")
    actor.reset(np.zeros((1, 7)))
    rewards, costs = [10.0, 20.0, 5.0], [1.0, 1.0, 1.0]  # budget 2 goes negative at t=3
    for t in range(3):
        act = actor.act(t)
        actor.observe(t, act, np.full((1, 7), t + 1.0), np.array([rewards[t]]),
                      np.array([costs[t]]))
    R = actor.returns[0, :4].numpy()
    C = actor.costs[0, :4].numpy()
    np.testing.assert_allclose(R, [30.0, 29.0, 27.0, 26.5], rtol=1e-6)  # R_{t+1}=R_t-r_t (scaled)
    np.testing.assert_allclose(C, [2.0, 1.0, 0.0, -1.0], rtol=1e-6)  # C_{t+1}=C_t-c_t
    assert torch.equal(actor.states[0, 3], torch.full((7, ), 3.0))
    s, a, r, c, ts = actor.window(3)
    assert ts[0].tolist() == [0, 1, 2, 3] and r.shape[1] == 4


def test_t6_target_updates_recurrent_actor():
    pol = small_policy("conv").eval()
    actor = RecurrentSequenceActor(pol, np.array([300.0]), np.array([2.0]), 0.1, 1.0, "cpu")
    actor.reset(np.zeros((1, 7)))
    for t, (r, c) in enumerate([(10.0, 1.0), (20.0, 1.0), (5.0, 1.0)]):
        act = actor.act(t)
        actor.observe(t, act, np.zeros((1, 7)), np.array([r]), np.array([c]))
    assert actor.R.item() == pytest.approx(26.5)
    assert actor.C.item() == pytest.approx(-1.0)


# ------------------------------------------------------------------------- T7
def _first_losses(data_cfg, synthetic_data, n=50):
    from experiments.common.config import TrainConfig
    from experiments.common.data import GPUWindowSampler, get_sequence_data
    from experiments.common.policy import build_policy
    from experiments.common.trainer import SequenceTrainer
    from experiments.common.utils import seed_all
    from experiments.common.expdefs import resolve_run_config

    cfg = resolve_run_config("E1", "cdt", synthetic_data["task"], 0)
    sd = get_sequence_data(cfg.data, 0, synthetic_data["cache"])
    seed_all(0)
    cfg.model.backbone_kwargs.update(num_layers=1)
    pol = build_policy(cfg.model, 7, 2, 1.0, 10, 200)
    tc = TrainConfig(batch_size=32, precision="fp32", fused_adamw=False)
    tr = SequenceTrainer(pol, tc, "cpu")
    sampler = GPUWindowSampler(sd, 10, 0.1, 1.0, 32, "cpu", seed=0)
    return [float(tr.train_one_step(sampler.sample())["all_loss"]) for _ in range(n)]


@pytest.mark.slow
def test_t7_same_seed_same_losses(data_cfg, synthetic_data):
    a = _first_losses(data_cfg, synthetic_data)
    b = _first_losses(data_cfg, synthetic_data)
    assert a == b


def test_t7_same_eval_seeds_across_models():
    from experiments.common.config import OUR_MODELS
    from experiments.common.expdefs import resolve_run_config

    jobs = []
    for m in OUR_MODELS + ["dt_cost", "bc_safe", "cpq"]:
        cfg = resolve_run_config("E1", m, "OfflineCarRun-v0", 3)
        cfg.eval.targets = [[575.0, 10.0]]
        jobs.append(make_jobs(cfg.eval.targets, cfg.eval.eval_episodes, cfg.eval.test_seed_base))
    assert all(j == jobs[0] for j in jobs)
    assert [s for _, _, s in jobs[0]] == list(range(1_000_000, 1_000_020))
    # validation and test seeds are disjoint
    cfg = resolve_run_config("E1", "cdt", "OfflineCarRun-v0", 0)
    assert cfg.eval.val_seed_base + 100 < cfg.eval.test_seed_base


# ------------------------------------------------------------------------- T8
def test_t8_metrics_hand_computed():
    # r_min = 100, r_max = 300: R = 200 -> 50
    assert normalized_reward(200.0, 100.0, 300.0) == pytest.approx(50.0)
    assert normalized_reward(100.0, 100.0, 300.0) == pytest.approx(0.0)
    # C_norm uses the requested kappa (not a fixed 10)
    assert normalized_cost(20.0, 40.0) == pytest.approx(0.5, rel=1e-6)
    assert normalized_cost(20.0, 10.0) == pytest.approx(2.0, rel=1e-6)
    assert normalized_cost(0.0, 0.0) == 0.0
    assert normalized_cost(1.0, 0.0) == pytest.approx(1e6)
    m = episode_metrics(np.array([200.0, 300.0]), np.array([5.0, 30.0]), np.array([10.0, 20.0]),
                        100.0, 300.0)
    np.testing.assert_allclose(m["R_norm"], [50.0, 100.0])
    np.testing.assert_allclose(m["C_norm"], [0.5, 1.5], rtol=1e-6)
    np.testing.assert_allclose(m["violation"], [0.0, 1.0])
    np.testing.assert_allclose(m["excess"], [0.0, 10.0])
    # E4 budget-tracking error: only kappa <= dataset max
    assert budget_tracking_error([10, 20, 80], [15, 10, 100], data_c_max=50) == pytest.approx(2.5)


def test_t8_matches_dsrl_reward_normalization():
    """Our R_norm equals 100 x DSRL get_normalized_score's reward part."""
    import gymnasium as gym

    import bullet_safety_gym  # noqa: F401
    import dsrl  # noqa: F401

    env = gym.make("OfflineCarRun-v0")
    env.unwrapped.set_target_cost(10)
    r_dsrl, _ = env.unwrapped.get_normalized_score(400.0, 5.0)
    u = env.unwrapped
    assert normalized_reward(400.0, u.min_episode_reward, u.max_episode_reward) == \
        pytest.approx(100 * r_dsrl)
    env.close()
