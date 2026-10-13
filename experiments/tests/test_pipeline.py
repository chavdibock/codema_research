"""Pipeline properties beyond T1-T12: resumable/idempotent runs, manifests, no side
effects on import, fairness of the model specs."""

import importlib
import subprocess
import sys

import pytest
import torch

from experiments.common.config import OUR_MODELS, get_model_spec


def test_fairness_only_backbone_differs():
    """cdt/cdt_light/cdmamba/cdconv differ only in backbone, its kwargs and d_model."""
    specs = {m: get_model_spec(m) for m in OUR_MODELS}
    ref = specs["cdt"].model
    for m, s in specs.items():
        assert s.overrides == {}, m
        for f in ("embedding_dropout", "time_emb", "use_rew", "use_cost", "cost_transform",
                  "stochastic", "init_temperature", "action_head_layers", "cost_prefix"):
            assert getattr(s.model, f) == getattr(ref, f), (m, f)


def test_imports_have_no_side_effects():
    code = ("import experiments.snelius.make_manifest, experiments.common.run, "
            "experiments.common.evaluate, experiments.common.profiling, "
            "experiments.analysis.tables; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ":".join(sys.path)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_manifest_counts_match_spec(tmp_path):
    from experiments.common.expdefs import enumerate_runs

    n = lambda e, k=None: len([r for r in enumerate_runs(e) if k is None or r.kind == k])  # noqa
    assert n("E1") == 100
    assert n("E2") == 150 and n("E2", "mlp") == 125
    assert n("E3") == 180
    assert n("E4", "mlp") == 25 and n("E4", "eval") == 100
    assert n("E8") == 120
    assert n("E9") == 60
    assert n("E7", "seq") == 12 and n("E7", "mlp") == 24


@pytest.mark.slow
def test_resume_equals_uninterrupted(data_cfg, synthetic_data, tmp_path):
    """A run interrupted after a checkpoint resumes to the same final weights."""
    from experiments.common.config import TrainConfig
    from experiments.common.data import GPUWindowSampler, get_sequence_data
    from experiments.common.trainer import SequenceTrainer, fit
    from experiments.common.utils import seed_all
    from experiments.tests.conftest import small_policy

    sd = get_sequence_data(data_cfg, 0, synthetic_data["cache"])
    tc = TrainConfig(update_steps=30, batch_size=16, precision="fp32", fused_adamw=False,
                     log_every=10, ckpt_every=10, val_every=0)

    def make():
        seed_all(0)
        pol = small_policy()
        tr = SequenceTrainer(pol, tc, "cpu")
        s = GPUWindowSampler(sd, 10, 0.1, 1.0, 16, "cpu", seed=0)
        return pol, tr, s

    pol_a, tr_a, s_a = make()
    fit(tr_a.train_one_step, s_a, tc, tmp_path / "a", "cpu", tr_a.state_dict,
        tr_a.load_state_dict, tr_a.lr)

    pol_b, tr_b, s_b = make()
    calls = {"n": 0}

    def crashing(batch):
        calls["n"] += 1
        if calls["n"] == 25:
            raise RuntimeError("simulated preemption")
        return tr_b.train_one_step(batch)

    with pytest.raises(RuntimeError):
        fit(crashing, s_b, tc, tmp_path / "b", "cpu", tr_b.state_dict, tr_b.load_state_dict,
            tr_b.lr)
    pol_c, tr_c, s_c = make()  # fresh process state, resumes from step 20
    stats = fit(tr_c.train_one_step, s_c, tc, tmp_path / "b", "cpu", tr_c.state_dict,
                tr_c.load_state_dict, tr_c.lr)
    assert stats["resumed_from_step"] == 20
    for (k, a), (_, c) in zip(pol_a.state_dict().items(), pol_c.state_dict().items()):
        if torch.is_tensor(a):
            assert torch.equal(a, c), k
    lines = (tmp_path / "b" / "train_log.csv").read_text().strip().splitlines()
    assert [l.split(",")[0] for l in lines[1:]] == ["10", "20", "30"]


@pytest.mark.slow
def test_run_is_idempotent(synthetic_data, tmp_path):
    from experiments.common.run import main

    args = ["--exp", "E0", "--model", "cdconv", "--task", synthetic_data["task"], "--seed", "0",
            "--out", str(tmp_path), "--cache-dir", str(synthetic_data["cache"]), "--override",
            "train.update_steps=20", "train.batch_size=16", "train.log_every=10",
            "train.val_every=0", "eval.eval_episodes=1", "eval.vectorized=false"]
    assert main(args) == 0
    run_dir = tmp_path / "E0" / synthetic_data["task"] / "cdconv" / "seed_0"
    for f in ("config.yaml", "env_info.json", "train_log.csv", "eval_episodes.csv",
              "ckpt_final.pt", "DONE", "run_summary.json"):
        assert (run_dir / f).exists(), f
    mtime = (run_dir / "DONE").stat().st_mtime
    assert main(args) == 0  # skipped
    assert (run_dir / "DONE").stat().st_mtime == mtime


@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["coptidice", "bcq_lag", "bear_lag", "cpq"])
def test_mlp_resume_equals_uninterrupted(model_id, synthetic_data, tmp_path, monkeypatch):
    """An MLP baseline interrupted after a checkpoint resumes to the same final weights
    (optimizers, plain tensors such as log_alpha/lmbda/_lmbda, PID controller)."""
    import experiments.baselines.mlp as mlp
    from experiments.common.expdefs import resolve_run_config
    from experiments.common.run import run_mlp

    overrides = {"train.update_steps": 30, "train.ckpt_every": 10, "train.log_every": 10,
                 "train.val_every": 0, "eval.eval_episodes": 0}

    def run(run_dir):
        cfg = resolve_run_config("E2", model_id, synthetic_data["task"], 1,
                                 out=str(tmp_path), cli_overrides=overrides,
                                 device="cpu", cache_dir=str(synthetic_data["cache"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        run_mlp(cfg, run_dir, "cpu")
        return torch.load(run_dir / "ckpt_final.pt", map_location="cpu",
                          weights_only=False)["model_state"]

    ref = run(tmp_path / "a")
    make_step_fn = mlp.make_step_fn

    def crashing_factory(*a, **kw):
        step, calls = make_step_fn(*a, **kw), {"n": 0}

        def crashing(batch):
            calls["n"] += 1
            if calls["n"] == 25:
                raise RuntimeError("simulated preemption")
            return step(batch)
        return crashing

    monkeypatch.setattr(mlp, "make_step_fn", crashing_factory)
    with pytest.raises(RuntimeError):
        run(tmp_path / "b")
    monkeypatch.setattr(mlp, "make_step_fn", make_step_fn)
    out = run(tmp_path / "b")  # fresh model, resumes from step 20
    for k, a in ref.items():
        assert torch.equal(a, out[k]), k
