"""Environment check (spec §8.1): imports, versions, one GPU forward/backward per
backbone, Mamba-3 CUDA vs reference agreement, one env step per task, dataset files and
the items the spec marks "verify". Writes env_report.txt.

    python experiments/snelius/check_env.py [--report env_report.txt] [--skip-gpu]
"""

import argparse
import importlib
import inspect
import os
import platform
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the Experiments/ dir

LINES = []
STATUS = {"ok": 0, "warn": 0, "fail": 0}


def log(kind: str, msg: str) -> None:
    STATUS[kind] += 1
    line = f"[{kind.upper():4}] {msg}"
    LINES.append(line)
    print(line, flush=True)


def check(name: str, fn) -> object:
    try:
        out = fn()
        log("ok", f"{name}: {out}")
        return out
    except Exception as e:  # noqa: BLE001
        log("fail", f"{name}: {type(e).__name__}: {e}")
        LINES.append(traceback.format_exc())
        return None


def version(mod: str) -> str:
    m = importlib.import_module(mod)
    return getattr(m, "__version__", "?")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="env_report.txt")
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    log("ok", f"python {platform.python_version()} on {platform.node()}")
    log("ok", f"modules: {os.environ.get('LOADEDMODULES', '<none>')}")

    # ---- imports and versions
    for mod in ("numpy", "scipy", "torch", "triton", "gymnasium", "pybullet",
                "bullet_safety_gym", "dsrl", "fsrl", "osrl", "pyrallis", "oapackage", "h5py",
                "numba", "yaml", "pandas", "matplotlib", "pyarrow", "psutil"):
        check(f"import {mod}", lambda mod=mod: version(mod))
    import torch

    log("ok", f"torch cuda={torch.version.cuda} available={torch.cuda.is_available()} "
        f"devices={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    from experiments.common import osrl_compat

    log("ok", f"OSRL commit {osrl_compat.osrl_commit()} at {osrl_compat.OSRL_ROOT}")

    # ---- verify items (spec "verify")
    import dsrl.offline_env as oe

    src = inspect.getsource(oe)
    log("ok" if "DSRL_DATASET_DIR" in src else "fail",
        "DSRL dataset env var is DSRL_DATASET_DIR (dsrl/offline_env.py)")
    log("ok", f"DSRL_DATASET_DIR={os.environ.get('DSRL_DATASET_DIR', '<unset>')} "
        f"(DSRL uses {oe.DATASET_PATH})")
    pre = inspect.getsource(oe.OfflineEnv.pre_process_data)
    log("ok" if "outliers_percent" in pre and "1.5 * trajs[i][\"rewards\"]" in pre else "warn",
        "DSRL outliers_percent: picks high-cost/high-reward trajectories, relabels cost < kappa "
        "and reward x1.5 ('lucky' trajectories, [CDT] App. D.2); value is a fraction")
    log("ok" if "if epsilon > 0" in pre and "if epsilon < 0" in pre else "warn",
        "DSRL epsilon: >0 removes top-reward SAFE trajectories, <0 removes top-reward UNSAFE "
        "ones; neither is [CDT] App. A's rule, so E7 uses data.reduce_rule=cdt")
    import osrl.algorithms.bc as obc

    log("ok" if "np.append(obs, self.cost_limit)" in inspect.getsource(obc) else "warn",
        "MT-BC conditions on kappa at evaluation (obs + cost_limit, osrl/algorithms/bc.py)")
    check("Mamba-3 import path mamba_ssm.modules.mamba3.Mamba3",
          lambda: importlib.import_module("mamba_ssm.modules.mamba3").Mamba3.__name__)
    check("mamba_ssm version", lambda: version("mamba_ssm"))

    # ---- tasks: episode length (D1), one env step, dataset file present
    import gymnasium as gym

    from experiments.common.config import TASKS
    from experiments.common.data import dataset_path, env_spec
    from experiments.common.rollout import make_env

    for task in TASKS:
        spec = env_spec(task)
        msg = f"{task}: max_episode_steps={spec.max_episode_steps}"
        if task == "OfflineDroneRun-v0":
            msg += " ([CDT] Tab. 4 says 100; we use the env spec, D1)"
        log("ok", msg)

        def step(task=task):
            env = make_env(task)
            obs, _ = env.reset(seed=0)
            out = env.step(env.action_space.sample())
            env.close()
            return f"obs {obs.shape}, cost key in info: {'cost' in out[4]}"

        check(f"{task} env step", step)
        path = dataset_path(task)
        log("ok" if path.exists() else "warn", f"{task} dataset {path} exists={path.exists()}")

    # ---- backbones: forward/backward on GPU (and Mamba-3 CUDA vs reference)
    if not args.skip_gpu and torch.cuda.is_available():
        from experiments.common.policy import SequencePolicy

        for bb, kw in (("transformer", {}), ("conv", {}), ("mamba3", {"mamba_impl": "cuda"}),
                       ("mamba3", {"mamba_impl": "ref"})):

            def fb(bb=bb, kw=kw):
                torch.manual_seed(0)
                pol = SequencePolicy(state_dim=7, action_dim=2, max_action=1.0, seq_len=10,
                                     episode_len=200, use_rew=True, use_cost=True,
                                     cost_transform=True, stochastic=True, target_entropy=-2,
                                     backbone=bb, backbone_kwargs=kw).cuda()
                B, K = 64, 10
                x = (torch.randn(B, K, 7), torch.rand(B, K, 2), torch.randn(B, K),
                     torch.randn(B, K), torch.arange(K).repeat(B, 1))
                x = [t.cuda() for t in x]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a, c, s = pol(*x)
                loss = -a.log_prob(x[1]).mean() + c.float().mean()
                loss.backward()
                torch.cuda.synchronize()
                return f"ok, loss={loss.item():.4f}, params={pol.num_parameters()}"

            check(f"GPU forward/backward {bb} {kw}", fb)

        def t5():
            from experiments.cdmamba.mamba3_ref import mamba3_ref_forward
            from experiments.cdmamba.mamba_block import import_official_mamba3

            M = import_official_mamba3()
            torch.manual_seed(0)
            m = M(d_model=128, d_state=64, expand=2, headdim=32, chunk_size=64).cuda()
            u = torch.randn(4, 40, 128, device="cuda") * 0.5
            with torch.no_grad():
                y_cuda = m(torch.nn.functional.pad(u, (0, 0, 0, 24)))[:, :40].float()
                y_ref = mamba3_ref_forward(m, u).float()
            err = (y_cuda - y_ref).abs().max().item()
            if err > 1e-3:
                raise AssertionError(f"max |cuda - ref| = {err:.2e} > 1e-3 (kernel runs bf16)")
            return f"max |cuda - ref| = {err:.2e}"

        check("T5 Mamba-3 CUDA vs reference", t5)
    else:
        log("warn", "GPU checks skipped")

    hdr = (f"env_report: {STATUS['ok']} ok, {STATUS['warn']} warn, {STATUS['fail']} fail\n")
    Path(args.report).write_text(hdr + "\n".join(LINES) + "\n")
    print(hdr + f"written to {args.report}")
    return 1 if STATUS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
