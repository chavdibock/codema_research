"""Bootstrap checks on a GPU node, before E0 (run by bootstrap.sbatch).

    python -m experiments.snelius.bootstrap --out $CDX_RESULTS --cache-dir $CDX_CACHE

Steps; the first fatal one stops the pipeline (exit 1, so dependent jobs never start):
  1. check_env.py without GPU checks (package pins, DSRL/OSRL source semantics, datasets);
  2. Mamba-3 CUDA kernels (NOTES V4). If they do not run here, platform.yaml switches
     cdmamba to the pure-PyTorch reference for every experiment (spec §4.4; E6 labels
     those timings impl=ref) instead of stopping the pipeline;
  3. one bf16 training step of every model on the GPU;
  4. T5 Mamba-3 CUDA vs reference (NOTES V5): <= 1e-3 ok; <= 1e-2 accepted as the
     kernel's internal bf16 rounding and recorded; larger = semantic mismatch, fatal;
  5. pytest experiments/tests (T5 is step 4, so it is deselected here);
  6. tiny-batch overfit check of every backbone on the GPU;
  7. preprocessing cache (S5) for the given experiments.
Writes $CDX_RESULTS/bootstrap.json (and platform.yaml when step 3 falls back).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

T5_OK, T5_BF16 = 1e-3, 1e-2


def sh(cmd: List[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def mamba_cuda_check() -> Dict[str, Any]:
    import torch

    from experiments.common.policy import SequencePolicy

    torch.manual_seed(0)
    pol = SequencePolicy(state_dim=7, action_dim=2, max_action=1.0, seq_len=10,
                         episode_len=200, use_rew=True, use_cost=True, cost_transform=True,
                         stochastic=True, target_entropy=-2, backbone="mamba3",
                         backbone_kwargs={"mamba_impl": "cuda"}).cuda()
    x = [t.cuda() for t in (torch.randn(8, 10, 7), torch.rand(8, 10, 2), torch.randn(8, 10),
                            torch.randn(8, 10), torch.arange(10).repeat(8, 1))]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, c, _ = pol(*x)
    (-a.log_prob(x[1]).mean() + c.float().mean()).backward()
    torch.cuda.synchronize()
    return {"ok": True}


def t5_error() -> float:
    import torch

    from experiments.cdmamba.mamba3_ref import Mamba3Ref
    from experiments.cdmamba.mamba_block import import_official_mamba3

    M = import_official_mamba3()
    worst = 0.0
    for L in (4, 28, 40, 64, 100):
        torch.manual_seed(0)
        off = M(d_model=128, d_state=64, expand=2, headdim=32, chunk_size=64).cuda()
        ref = Mamba3Ref(d_model=128, d_state=64, expand=2, headdim=32).cuda()
        missing, unexpected = ref.load_state_dict(off.state_dict(), strict=False)
        if missing or unexpected:
            raise AssertionError(f"state dict mismatch: {missing} {unexpected}")
        u = 0.5 * torch.randn(3, L, 128, device="cuda")
        pad = (64 - L % 64) % 64
        with torch.no_grad():
            y_cuda = off(torch.nn.functional.pad(u, (0, 0, 0, pad)))[:, :L].float()
            y_ref = ref(u).float()
        worst = max(worst, (y_cuda - y_ref).abs().max().item())
    return worst


def main(argv: Optional[List[str]] = None) -> int:
    import torch

    from experiments.common.utils import atomic_write_json, atomic_write_yaml
    from experiments.snelius.decisions import PLATFORM_FILE

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get("CDX_RESULTS", "results"))
    p.add_argument("--cache-dir", default=os.environ.get("CDX_CACHE", "cache"))
    p.add_argument("--prepare-exps", nargs="*",
                   default=["E0", "E0prec", "E0tune", "E0pack", "E1", "E2", "E3", "E4",
                            "E6full", "E7", "E8", "E9", "E10"])
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: Dict[str, Any] = {"started": datetime.datetime.now().isoformat(timespec="seconds"),
                           "host": os.uname().nodename, "steps": {}}

    def done(ok: bool, step: str, **info) -> bool:
        rep["steps"][step] = {"ok": ok, **info}
        atomic_write_json(out / "bootstrap.json", rep)
        print(f"== {step}: {'ok' if ok else 'FAILED'} {info or ''}", flush=True)
        return ok

    # 1. environment and source checks (the GPU part is below)
    if not done(sh([sys.executable, "experiments/snelius/check_env.py", "--skip-gpu"]) == 0,
                "check_env"):
        return 1
    if not torch.cuda.is_available():
        done(False, "gpu", error="no CUDA device in the bootstrap job")
        return 1

    # 2. Mamba-3 CUDA kernels; fall back to the reference implementation if they fail
    try:
        mamba_cuda_check()
        mamba_ok = True
        done(True, "mamba3_cuda")
    except Exception as e:  # noqa: BLE001
        mamba_ok = False
        atomic_write_yaml(out / PLATFORM_FILE, {
            "model_overrides": {"cdmamba": {"model.backbone_kwargs.mamba_impl": "ref"}},
            "reason": f"Mamba-3 CUDA kernels failed on {torch.cuda.get_device_name()}: {e!r}",
            "note": "spec §4.4: train with the pure-PyTorch reference; label its timings"})
        done(True, "mamba3_cuda", fallback="ref", error=repr(e))

    # 3. one bf16 training step of every model on the GPU (with the impl chosen above)
    from experiments.common.config import OUR_MODELS, TrainConfig
    from experiments.common.policy import build_policy
    from experiments.common.profiling import model_cfg_for, random_batch, task_dims
    from experiments.common.trainer import SequenceTrainer

    try:
        dims = task_dims("OfflineCarCircle-v0")
        for m in OUR_MODELS:
            mc, impl = model_cfg_for(m, "cuda")
            if mc.backbone == "mamba3" and not mamba_ok:
                mc.backbone_kwargs["mamba_impl"] = "ref"
            torch.manual_seed(0)
            pol = build_policy(mc, dims["state_dim"], dims["action_dim"], dims["max_action"],
                               10, dims["max_episode_steps"]).cuda()
            tr = SequenceTrainer(pol, TrainConfig(batch_size=32, precision="bf16"), "cuda")
            tr.train_one_step(random_batch(pol, 32, pol.seq_len, "cuda"))
        done(True, "gpu_train_step")
    except Exception as e:  # noqa: BLE001
        done(False, "gpu_train_step", error=repr(e))
        return 1

    # 4. T5
    if mamba_ok:
        try:
            err = t5_error()
        except Exception as e:  # noqa: BLE001
            done(False, "t5", error=repr(e))
            return 1
        if err > T5_BF16:
            done(False, "t5", max_abs_err=err,
                 error="larger than bf16 rounding: semantic mismatch in mamba3_ref.py")
            return 1
        done(True, "t5", max_abs_err=err,
             note="within 1e-3" if err <= T5_OK else "bf16 rounding of the kernel (NOTES V5)")

    # 5. test suite (CUDA tests included, T5 handled above)
    sel = ["-k", "not test_t5_cuda_matches_reference"] if mamba_ok else ["-m", "not mamba_cuda"]
    if not done(sh([sys.executable, "-m", "pytest", "experiments/tests", "-q", *sel]) == 0,
                "pytest"):
        return 1

    # 6. overfit check on the GPU
    if not done(sh([sys.executable, "-m", "experiments.common.sanity", "--overfit",
                    "--device", "cuda"]) == 0, "overfit"):
        return 1

    # 7. preprocessing cache
    if not done(sh([sys.executable, "-m", "experiments.common.data", "--prepare", "--exp",
                    *a.prepare_exps, "--cache-dir", a.cache_dir, "--workers",
                    str(a.workers)]) == 0, "prepare_cache"):
        return 1
    rep["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    rep["ok"] = True
    atomic_write_json(out / "bootstrap.json", rep)
    print("bootstrap ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
