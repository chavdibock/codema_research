"""Efficiency measurements (E6) and parameter counting/matching (spec §4.3).

    python -m experiments.common.profiling --count-params [--task OfflineCarCircle-v0]
    python -m experiments.common.profiling --match-light [--write]
    python -m experiments.common.profiling --e6 --device cuda --rep 0 \
        [--models cdt cdt_light cdmamba cdconv] [--tasks ...] [--what size train infer mem scaling]
    CUDA_VISIBLE_DEVICES="" taskset -c 0 python -m experiments.common.profiling --e6 \
        --device cpu --threads 1 --what infer mem --rep 0

Four core metrics (Table 2, K=10): M1 training time, M2 inference time per prediction,
M3 training memory, M4 inference memory; plus size (E6.1) and context scaling (E6.6).
Rows go to results/E6/efficiency.csv (one row per model, task, metric, device, mode, K,
batch, precision, repetition). Run each repetition in a separate process.
"""

import argparse
import copy
import fcntl
import gc
import io
import itertools
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.common.config import OUR_MODELS, TASKS, get_model_spec
from experiments.common.utils import atomic_write_json, hardware_info, resolve_device, run_dir_for

E6_KS = [10, 20, 50, 100, 200]
CSV_FIELDS = ["model", "task", "metric", "stat", "value", "unit", "device", "mode", "K",
              "batch", "precision", "repetition", "threads", "impl", "weights", "host"]


# ------------------------------------------------------------------ output
class EfficiencyCSV:
    """Append rows to efficiency.csv under an exclusive file lock."""

    def __init__(self, path: Path, common: Dict[str, Any]):
        self.path = Path(path)
        self.common = common
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, **row) -> None:
        r = {**self.common, **row}
        line = ",".join(_csv_cell(r.get(k, "")) for k in CSV_FIELDS) + "\n"
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if f.tell() == 0:
                f.write(",".join(CSV_FIELDS) + "\n")
            f.write(line)
            fcntl.flock(f, fcntl.LOCK_UN)
        print({k: r.get(k) for k in ("model", "metric", "stat", "value", "unit", "K", "batch",
                                     "precision", "device", "mode")}, flush=True)


def _csv_cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    s = str(v)
    return f'"{s}"' if "," in s else s


# ------------------------------------------------------------------ models
def task_dims(task: str) -> Dict[str, Any]:
    from experiments.common.data import env_constants

    return env_constants(task)


def model_cfg_for(model_id: str, device: str, seq_len: int = 10):
    """ModelConfig with the mamba impl forced to 'ref' on CPU or if kernels are missing."""
    mc = copy.deepcopy(get_model_spec(model_id).model)
    impl = "-"
    if mc.backbone == "mamba3":
        impl = mc.backbone_kwargs.get("mamba_impl", "cuda")
        if device == "cpu" or not _mamba_cuda_available():
            impl = "ref"
        mc.backbone_kwargs["mamba_impl"] = impl
    return mc, impl


_MAMBA_OK: Optional[bool] = None


def _mamba_cuda_available() -> bool:
    global _MAMBA_OK
    if _MAMBA_OK is None:
        try:
            from experiments.cdmamba.mamba_block import import_official_mamba3

            import_official_mamba3()
            _MAMBA_OK = torch.cuda.is_available()
        except ImportError:
            _MAMBA_OK = False
    return _MAMBA_OK


def build_fresh(model_id: str, task: str, device: str, seq_len: int = 10, dims=None,
                episode_len: Optional[int] = None):
    from experiments.common.policy import build_policy

    dims = dims or task_dims(task)
    mc, impl = model_cfg_for(model_id, device, seq_len)
    ep = episode_len or dims["max_episode_steps"]
    torch.manual_seed(0)
    pol = build_policy(mc, dims["state_dim"], dims["action_dim"], dims["max_action"], seq_len,
                       ep)
    return pol.to(device), mc, impl


def load_trained(model_id: str, task: str, device: str, out: str, ckpt_from: str):
    """Trained checkpoint (seed 0) if available; mamba forced to an impl that runs here."""
    from experiments.common.policy import build_policy

    run_dir = run_dir_for(out, ckpt_from, task, model_id, None, 0)
    ck_path = run_dir / "ckpt_final.pt"
    if not ck_path.exists():
        return None
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    from experiments.common.config import from_dict

    cfg = from_dict(ck["config"])
    mc = cfg.model
    impl = "-"
    if mc.backbone == "mamba3":
        impl = "ref" if device == "cpu" or not _mamba_cuda_available() else "cuda"
        mc.backbone_kwargs["mamba_impl"] = impl
    info = ck["env_info"]
    pol = build_policy(mc, info["state_dim"], info["action_dim"], info["max_action"],
                       cfg.data.seq_len, cfg.data.episode_len)
    pol.load_state_dict(ck["model_state"])
    return pol.to(device), cfg, impl, ck_path.stat().st_size


def count_params(model_ids: List[str], task: str) -> List[Dict[str, Any]]:
    dims = task_dims(task)
    rows = []
    for m in model_ids:
        spec = get_model_spec(m)
        if spec.model.kind != "sequence":
            continue
        pol, _, _ = build_fresh(m, task, "cpu", dims=dims)
        rows.append({"model": m, "total": pol.num_parameters(),
                     "backbone": pol.num_parameters(backbone_only=True)})
    return rows


def static_bytes(model: torch.nn.Module) -> int:
    return (sum(p.numel() * p.element_size() for p in model.parameters()) +
            sum(b.numel() * b.element_size() for b in model.buffers()))


def state_dict_bytes(model: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


# ------------------------------------------------------------------ inputs
def random_batch(pol, B: int, K: int, device: str):
    ds, da, ep = pol.state_dim, pol.action_dim, pol.episode_len
    g = torch.Generator(device="cpu").manual_seed(0)
    start = torch.randint(0, ep, (B, 1), generator=g)
    batch = (torch.randn(B, K, ds, generator=g), torch.rand(B, K, da, generator=g) * 2 - 1,
             torch.randn(B, K, generator=g) * 50, torch.rand(B, K, generator=g) * 40,
             start + torch.arange(K)[None], torch.ones(B, K),
             torch.rand(B, generator=g) * 40, (torch.rand(B, K, generator=g) < 0.1).float())
    return tuple(x.to(device) for x in batch)


def rollout_windows(pol, task: str, device: str, targets: Tuple[float, float],
                    reward_scale: float, cost_scale: float, seed: int = 1_000_000):
    """Real context windows from one evaluation rollout (window mode inputs per step)."""
    from experiments.common.rollout import WindowSequenceActor, make_env

    env = make_env(task)
    actor = WindowSequenceActor(pol, np.array([targets[0]]), np.array([targets[1]]),
                                reward_scale, cost_scale, device)
    obs, _ = env.reset(seed=seed)
    actor.reset(obs[None])
    windows = []
    for t in range(pol.episode_len):
        windows.append(tuple(x.clone() for x in actor.window(t)))
        act = actor.act(t)
        obs, r, term, trunc, info = env.step(act[0].float().cpu().numpy())
        actor.observe(t, act, obs[None], np.array([r]), np.array([info.get("cost", 0.0)]))
        if term or trunc:
            break
    env.close()
    return windows


# ------------------------------------------------------------------ timing
def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_calls(fn, n_warm: int, n_timed: int, device: str) -> np.ndarray:
    for _ in range(n_warm):
        fn()
    _sync(device)
    out = np.empty(n_timed)
    for i in range(n_timed):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out[i] = time.perf_counter() - t0
    return out


def emit_stats(csv: EfficiencyCSV, arr_s: np.ndarray, unit_scale: float, unit: str, **kw):
    a = arr_s * unit_scale
    for stat, v in (("median", np.median(a)), ("p95", np.percentile(a, 95)),
                    ("mean", a.mean()), ("std", a.std())):
        csv.add(stat=stat, value=float(v), unit=unit, **kw)


def rss_bytes() -> int:
    import psutil

    return psutil.Process().memory_info().rss


# ------------------------------------------------------------- measurements
def measure_train(csv, pol, cfg_train, device, K, B, precision, sampler=None, warm=200,
                  timed=2000, tokens=None, label="train_step"):
    """M1/M3: s/step (sampler excluded or included) and peak memory of one full step."""
    from experiments.common.trainer import SequenceTrainer, setup_precision

    tcfg = copy.deepcopy(cfg_train)
    tcfg.precision = precision
    tcfg.batch_size = B
    tcfg.compile = False
    setup_precision(tcfg)
    snapshot = copy.deepcopy(pol.state_dict())  # timed steps must not change the weights
    pol.train()
    tr = SequenceTrainer(pol, tcfg, device)
    pre = [random_batch(pol, B, K, device) if sampler is None else sampler.sample()
           for _ in range(4)]
    cyc = itertools.cycle(pre)
    # M3: one full step (forward, backward, AdamW, temperature update)
    if device.startswith("cuda"):
        tr.train_one_step(next(cyc))
        _sync(device)
        torch.cuda.reset_peak_memory_stats()
        tr.train_one_step(next(cyc))
        _sync(device)
        csv.add(metric="train_mem_allocated", stat="peak", unit="MB", K=K, batch=B,
                precision=precision, mode="window",
                value=torch.cuda.max_memory_allocated() / 2**20)
        csv.add(metric="train_mem_reserved", stat="peak", unit="MB", K=K, batch=B,
                precision=precision, mode="window",
                value=torch.cuda.max_memory_reserved() / 2**20)
        csv.add(metric="train_mem_static", stat="value", unit="MB", K=K, batch=B,
                precision=precision, mode="window", value=static_bytes(pol) / 2**20)
    else:
        base = rss_bytes()
        tr.train_one_step(next(cyc))
        csv.add(metric="train_mem_rss_increase", stat="value", unit="MB", K=K, batch=B,
                precision=precision, mode="window", value=(rss_bytes() - base) / 2**20)
    t_ex = time_calls(lambda: tr.train_one_step(next(cyc)), warm, timed, device)
    s_step = float(np.median(t_ex))
    csv.add(metric=f"{label}_s", stat="median", unit="s/step", K=K, batch=B,
            precision=precision, mode="sampler_excluded", value=s_step)
    csv.add(metric=f"{label}_s", stat="mean", unit="s/step", K=K, batch=B,
            precision=precision, mode="sampler_excluded", value=float(t_ex.mean()))
    csv.add(metric="steps_per_s", stat="value", unit="1/s", K=K, batch=B, precision=precision,
            mode="sampler_excluded", value=1.0 / s_step)
    if tokens:
        csv.add(metric="tokens_per_s", stat="value", unit="tokens/s", K=K, batch=B,
                precision=precision, mode="sampler_excluded", value=B * tokens / s_step)
    csv.add(metric="train_time_100k_projected", stat="value", unit="h", K=K, batch=B,
            precision=precision, mode="sampler_excluded", value=s_step * 1e5 / 3600)
    if sampler is not None:
        t_in = time_calls(lambda: tr.train_one_step(sampler.sample()), warm, timed, device)
        s_in = float(np.median(t_in))
        csv.add(metric=f"{label}_s", stat="median", unit="s/step", K=K, batch=B,
                precision=precision, mode="sampler_included", value=s_in)
        csv.add(metric="steps_per_s", stat="value", unit="1/s", K=K, batch=B,
                precision=precision, mode="sampler_included", value=1.0 / s_in)
        csv.add(metric="train_time_100k_projected", stat="value", unit="h", K=K, batch=B,
                precision=precision, mode="sampler_included", value=s_in * 1e5 / 3600)
    del tr
    pol.load_state_dict(snapshot)
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@torch.inference_mode()
def measure_infer(csv, pol, device, windows, K, warm=100, timed=1000, label="window"):
    """M2 (a) policy forward only, batch 1; M4 inference memory."""
    pol.eval()
    full = [w for w in windows if w[0].shape[1] == K] or windows
    for subset, mode in ((full, f"{label}_full"), (windows, f"{label}_episode_avg")):
        cyc = itertools.cycle([tuple(x.to(device) for x in w) for w in subset])

        def fwd():
            s, a, r, c, t = next(cyc)
            pol(s, a, r, c, t, None, None)

        arr = time_calls(fwd, warm, timed, device)
        emit_stats(csv, arr, 1e3, "ms", metric="infer_latency_forward", K=K, batch=1,
                   precision="fp32", mode=mode)
    w = tuple(x.to(device) for x in full[0])
    if device.startswith("cuda"):
        _sync(device)
        torch.cuda.reset_peak_memory_stats()
        pol(*w, None, None)
        _sync(device)
        total = torch.cuda.max_memory_allocated()
        csv.add(metric="infer_mem_allocated", stat="peak", unit="MB", K=K, batch=1,
                precision="fp32", mode=label, value=total / 2**20)
        csv.add(metric="infer_mem_activation", stat="peak", unit="MB", K=K, batch=1,
                precision="fp32", mode=label, value=(total - static_bytes(pol)) / 2**20)
    else:
        base = rss_bytes()
        peak = base
        for _ in range(1000):
            pol(*w, None, None)
            peak = max(peak, rss_bytes())
        csv.add(metric="infer_mem_rss_increase", stat="peak", unit="MB", K=K, batch=1,
                precision="fp32", mode=label, value=(peak - base) / 2**20)
    window_bytes = sum(x.numel() * x.element_size() for x in w)
    csv.add(metric="context_window_bytes", stat="value", unit="bytes", K=K, batch=1,
            precision="fp32", mode=label, value=float(window_bytes))


def measure_control_step(csv, pol, task, device, targets, reward_scale, cost_scale, K,
                         mode="window", n_steps=1000):
    """M2 nested latencies: (a) forward, (b) + pre/post-processing and transfers,
    (c) + env.step, measured on real rollouts (batch 1)."""
    from experiments.common.rollout import (RecurrentSequenceActor, WindowSequenceActor,
                                            make_env)

    cls = WindowSequenceActor if mode == "window" else RecurrentSequenceActor
    env = make_env(task)
    a_t, b_t, c_t = [], [], []
    seed = 1_000_000
    with torch.inference_mode():
        while len(c_t) < n_steps:
            actor = cls(pol, np.array([targets[0]]), np.array([targets[1]]), reward_scale,
                        cost_scale, device)
            obs, _ = env.reset(seed=seed)
            seed += 1
            actor.reset(obs[None])
            for t in range(pol.episode_len):
                t0 = time.perf_counter()
                if mode == "window":
                    win = actor.window(t)
                    _sync(device)
                    ta = time.perf_counter()
                    out, _, _ = pol(*win, None, actor.epi_cost)
                    _sync(device)
                    tb = time.perf_counter()
                    act = (out.mean if pol.stochastic else out).clamp(
                        -pol.max_action, pol.max_action)[:, -1]
                else:
                    ta = time.perf_counter()
                    act = actor.act(t)
                    _sync(device)
                    tb = time.perf_counter()
                act_np = act[0].float().cpu().numpy()
                t1 = time.perf_counter()
                obs, r, term, trunc, info = env.step(act_np)
                t2 = time.perf_counter()
                tp = time.perf_counter()
                actor.observe(t, act, obs[None], np.array([r]), np.array([info.get("cost", 0.0)]))
                _sync(device)
                t3 = time.perf_counter()
                a_t.append(tb - ta)
                b_t.append((t1 - t0) + (t3 - tp))
                c_t.append((t1 - t0) + (t3 - tp) + (t2 - t1))
                if term or trunc or len(c_t) >= n_steps:
                    break
    env.close()
    for name, arr in (("infer_latency_a_forward", a_t), ("infer_latency_b_plus_io", b_t),
                      ("infer_latency_c_plus_env", c_t)):
        emit_stats(csv, np.asarray(arr), 1e3, "ms", metric=name, K=K, batch=1,
                   precision="fp32", mode=f"{mode}_rollout")


@torch.inference_mode()
def measure_recurrent(csv, pol, device, K, warm=100, timed=1000):
    """Recurrent mode (cdmamba, cdconv): latency of one env step = R, C, s tokens -> action
    plus the a-token update; size of the carried state."""
    pol.eval()
    if csv.common.get("impl") not in (None, "-"):
        # the Mamba-3 step always runs the pure-PyTorch reference (mamba3_ref_step)
        csv = EfficiencyCSV(csv.path, {**csv.common, "impl": "ref_step"})
    st = pol.init_recurrent(1, device=device)
    s = torch.randn(1, pol.state_dim, device=device)
    r = torch.zeros(1, device=device)
    c = torch.zeros(1, device=device)
    a = torch.zeros(1, pol.action_dim, device=device)
    t = torch.zeros(1, dtype=torch.long, device=device)
    holder = {"st": st}

    def one():
        _, holder["st"] = pol.recurrent_observe(s, r, c, t, holder["st"])

    def full_step():
        one()
        holder["st"] = pol.recurrent_act(a, t, holder["st"])

    emit_stats(csv, time_calls(one, warm, timed, device), 1e3, "ms",
               metric="infer_latency_forward", K=K, batch=1, precision="fp32",
               mode="recurrent_predict")
    emit_stats(csv, time_calls(full_step, warm, timed, device), 1e3, "ms",
               metric="infer_latency_forward", K=K, batch=1, precision="fp32",
               mode="recurrent_full_step")
    csv.add(metric="recurrent_state_bytes", stat="value", unit="bytes", K=K, batch=1,
            precision="fp32", mode="recurrent",
            value=float(pol.backbone.state_nbytes(holder["st"])))
    if device.startswith("cuda"):
        _sync(device)
        torch.cuda.reset_peak_memory_stats()
        one()
        _sync(device)
        csv.add(metric="infer_mem_allocated", stat="peak", unit="MB", K=K, batch=1,
                precision="fp32", mode="recurrent",
                value=torch.cuda.max_memory_allocated() / 2**20)


# ------------------------------------------------------------------ E6 driver
def run_e6(args) -> None:
    from experiments.common.config import TrainConfig
    from experiments.common.data import GPUWindowSampler, get_sequence_data
    from experiments.common.expdefs import resolve_run_config

    device = resolve_device(args.device)
    if device == "cpu":
        torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    out = Path(args.out) / "E6"
    hw = hardware_info()
    hw.update({"torch_threads": torch.get_num_threads(), "device": device,
               "mamba_cuda_kernels": _mamba_cuda_available()})
    _merge_hardware(out / "hardware.json", hw)
    per_task = set(args.what) & {"size", "train", "infer", "mem"}
    for task in (args.tasks if per_task else []):
        for m in args.models:
            loaded = load_trained(m, task, device, args.out, args.ckpt_from)
            if loaded is None:
                if not args.allow_fresh:
                    print(f"skip {m} {task}: no {args.ckpt_from} seed-0 checkpoint "
                          "(use --allow-fresh for freshly initialized weights)")
                    continue
                pol, _, impl = build_fresh(m, task, device)
                rcfg = resolve_run_config(args.ckpt_from, m, task, 0)
                weights, ckpt_bytes = "fresh", None
            else:
                pol, rcfg, impl, ckpt_bytes = loaded
                weights = "trained"
            K = rcfg.data.seq_len
            csv = EfficiencyCSV(out / "efficiency.csv",
                                {"model": m, "task": task, "device": device,
                                 "repetition": args.rep, "threads": torch.get_num_threads(),
                                 "impl": impl, "weights": weights, "host": hw["hostname"]})
            if "size" in args.what:
                csv.add(metric="params_total", stat="value", unit="count", K=K,
                        value=float(pol.num_parameters()))
                csv.add(metric="params_backbone", stat="value", unit="count", K=K,
                        value=float(pol.num_parameters(backbone_only=True)))
                csv.add(metric="static_model_mem", stat="value", unit="MB", K=K,
                        value=static_bytes(pol) / 2**20)
                csv.add(metric="checkpoint_bytes", stat="value", unit="bytes", K=K,
                        value=float(ckpt_bytes or state_dict_bytes(pol)))
            tokens = pol.seq_repeat * K
            if "train" in args.what:
                sd = get_sequence_data(rcfg.data, 0)
                for prec in args.precisions:
                    for B in args.batches:
                        sampler = GPUWindowSampler(sd, K, rcfg.data.reward_scale,
                                                   rcfg.data.cost_scale, B, device, 0)
                        measure_train(csv, pol, rcfg.train, device, K, B, prec, sampler,
                                      args.warmup_steps, args.timed_steps, tokens)
            targets = [t for t in rcfg.meta["task_targets"] if float(t[1]) == 10.0][0]
            if "infer" in args.what or "mem" in args.what:
                windows = rollout_windows(pol, task, device, targets, rcfg.data.reward_scale,
                                          rcfg.data.cost_scale)
                measure_infer(csv, pol, device, windows, K)
                measure_control_step(csv, pol, task, device, targets, rcfg.data.reward_scale,
                                     rcfg.data.cost_scale, K, "window")
                if pol.backbone.supports_recurrent:
                    measure_recurrent(csv, pol, device, K)
                    measure_control_step(csv, pol, task, device, targets,
                                         rcfg.data.reward_scale, rcfg.data.cost_scale, K,
                                         "recurrent")
            del pol
            gc.collect()
    if "scaling" in args.what:
        run_scaling(args, device, hw)


def run_scaling(args, device: str, hw: Dict[str, Any]) -> None:
    """E6.6: freshly initialized models, K in {10..200}; computational scaling only."""
    from experiments.common.config import TrainConfig

    out = Path(args.out) / "E6"
    task = args.scaling_task
    dims = task_dims(task)
    for m in args.models:
        for K in args.scaling_ks:
            pol, mc, impl = build_fresh(m, task, device, seq_len=K, dims=dims)
            csv = EfficiencyCSV(out / "efficiency.csv",
                                {"model": m, "task": task, "device": device,
                                 "repetition": args.rep, "threads": torch.get_num_threads(),
                                 "impl": impl, "weights": "fresh_scaling",
                                 "host": hw["hostname"]})
            for prec in args.precisions:
                for B in args.batches:
                    try:
                        measure_train(csv, pol, TrainConfig(), device, K, B, prec, None,
                                      args.scaling_warmup, args.scaling_steps,
                                      pol.seq_repeat * K, label="scaling_train_step")
                    except torch.cuda.OutOfMemoryError:
                        csv.add(metric="scaling_train_step_s", stat="OOM", value="", unit="",
                                K=K, batch=B, precision=prec, mode="sampler_excluded")
                        torch.cuda.empty_cache()
            b = random_batch(pol, 1, K, device)
            windows = [(b[0], b[1], b[2], b[3], b[4])]  # one full window of length K
            measure_infer(csv, pol, device, windows, K, warm=20, timed=args.scaling_steps,
                          label="scaling_window")
            if pol.backbone.supports_recurrent:
                measure_recurrent(csv, pol, device, K, warm=20, timed=args.scaling_steps)
            del pol
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


def _merge_hardware(path: Path, hw: Dict[str, Any]) -> None:
    import json

    data = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    data[f"{hw['hostname']}:{hw['device']}:threads{hw['torch_threads']}"] = hw
    atomic_write_json(path, data)


# ------------------------------------------------------------------ §4.3 matching
LIGHT_GRID = dict(layers=(1, 2, 3), d=(48, 64, 96, 128), heads=(2, 4, 8))


def match_light(task: str) -> Dict[str, Any]:
    """Transformer config from the grid closest to min(params(cdmamba), params(cdconv))."""
    from experiments.common.policy import build_policy

    dims = task_dims(task)
    counts = {r["model"]: r["total"] for r in count_params(["cdmamba", "cdconv"], task)}
    target = min(counts.values())
    base = copy.deepcopy(get_model_spec("cdt").model)
    cands = []
    for L, d, h in itertools.product(*LIGHT_GRID.values()):
        if d % h:
            continue
        mc = copy.deepcopy(base)
        mc.embedding_dim = d
        mc.backbone_kwargs.update(num_layers=L, num_heads=h)
        pol = build_policy(mc, dims["state_dim"], dims["action_dim"], dims["max_action"], 10,
                           dims["max_episode_steps"])
        n = pol.num_parameters()
        cands.append({"layers": L, "d": d, "heads": h, "params": n,
                      "rel_gap": (n - target) / target})
    # closest count; ties -> fewer layers, then more heads (deterministic)
    best = min(cands, key=lambda c: (abs(c["rel_gap"]), c["layers"], -c["heads"]))
    return {"task": task, "target": target, "counts": counts, "best": best,
            "within_10pct": abs(best["rel_gap"]) <= 0.10, "candidates": cands}


def write_light_config(res: Dict[str, Any]) -> None:
    b = res["best"]
    path = Path(__file__).resolve().parents[1] / "cdt_light" / "config.py"
    path.write_text(f'''"""Lightweight CDT: parameter-matched transformer (capacity control, spec §4.3).

Chosen with ``python -m experiments.common.profiling --match-light`` on {res["task"]}:
grid layers {{1,2,3}} x d {{48,64,96,128}} x heads {{2,4,8}}, closest total trainable
parameter count (embeddings and heads included) to min(params(cdmamba), params(cdconv)).
    target   = {res["target"]} (cdmamba {res["counts"]["cdmamba"]}, cdconv {res["counts"]["cdconv"]})
    achieved = {b["params"]} ({100 * b["rel_gap"]:+.1f} %, within +-10 %: {res["within_10pct"]})
"""

from experiments.common.config import ModelConfig, ModelSpec


def get_spec(model_id: str = "cdt_light") -> ModelSpec:
    return ModelSpec(model=ModelConfig(
        name="cdt_light",
        backbone="transformer",
        embedding_dim={b["d"]},
        backbone_kwargs=dict(num_layers={b["layers"]}, num_heads={b["heads"]}, attention_dropout=0.1,
                             residual_dropout=0.1),
    ))
''')


# ------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count-params", action="store_true")
    p.add_argument("--match-light", action="store_true")
    p.add_argument("--write", action="store_true", help="write cdt_light/config.py")
    p.add_argument("--task", default="OfflineCarCircle-v0")
    p.add_argument("--e6", action="store_true")
    p.add_argument("--models", nargs="+", default=OUR_MODELS)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--ckpt-from", default="E1")
    p.add_argument("--allow-fresh", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--what", nargs="+", default=["size", "train", "infer", "mem", "scaling"])
    p.add_argument("--precisions", nargs="+", default=["fp32", "bf16"])
    p.add_argument("--batches", nargs="*", type=int, default=[2048, 256])
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--timed-steps", type=int, default=2000)
    p.add_argument("--scaling-task", default="OfflineCarCircle-v0")
    p.add_argument("--scaling-ks", nargs="+", type=int, default=E6_KS)
    p.add_argument("--scaling-warmup", type=int, default=20)
    p.add_argument("--scaling-steps", type=int, default=200)
    args = p.parse_args()
    if args.count_params:
        from experiments.baselines.config import get_spec as _  # noqa: F401

        rows = count_params(OUR_MODELS + ["dt_cost"], args.task)
        print(f"{'model':<12}{'total':>12}{'backbone':>12}   ({args.task})")
        for r in rows:
            print(f"{r['model']:<12}{r['total']:>12}{r['backbone']:>12}")
    if args.match_light:
        res = match_light(args.task)
        b = res["best"]
        print(f"target={res['target']} counts={res['counts']}")
        print(f"best: layers={b['layers']} d={b['d']} heads={b['heads']} params={b['params']} "
              f"gap={100 * b['rel_gap']:+.1f}% within10={res['within_10pct']}")
        if args.write:
            write_light_config(res)
            print("wrote experiments/cdt_light/config.py")
    if args.e6:
        run_e6(args)


if __name__ == "__main__":
    main()
