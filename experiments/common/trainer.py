"""Copy of OSRL's ``CDTTrainer.train_one_step`` + the fixes of spec §4.6 and the
performance engineering of §4.8 (S3 bf16/TF32/fused AdamW, S4 compile, S6 cheap
validation/checkpointing), plus a generic resumable training loop.
"""

import contextlib
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import functional as F

from experiments.common.config import TrainConfig
from experiments.common.logging import CSVLog
from experiments.common.utils import atomic_torch_save


def setup_precision(cfg: TrainConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32


def autocast_ctx(precision: str, device: str):
    if precision == "bf16":
        dev_type = "cuda" if device.startswith("cuda") else "cpu"
        return torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
    if precision != "fp32":
        raise ValueError(f"precision must be fp32 or bf16, got {precision}")
    return contextlib.nullcontext()


class SequenceTrainer:
    """Losses identical to OSRL's CDTTrainer (NLL + entropy reg. with learnable
    temperature, or MSE if deterministic; cost classification aux loss; state loss)."""

    def __init__(self, model, cfg: TrainConfig, device: str):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.clip_grad = cfg.clip_grad
        self.cost_weight = cfg.loss_cost_weight
        self.state_weight = cfg.loss_state_weight
        self.no_entropy = cfg.no_entropy
        self.stochastic = model.stochastic
        fused = bool(cfg.fused_adamw and device.startswith("cuda"))
        self.optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                       weight_decay=cfg.weight_decay, betas=tuple(cfg.betas),
                                       **({"fused": True} if fused else {}))
        warmup = cfg.lr_warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lambda steps: min((steps + 1) / warmup, 1))
        if self.stochastic:
            self.log_temperature_optimizer = torch.optim.Adam([model.log_temperature],
                                                              lr=cfg.temperature_lr,
                                                              betas=[0.9, 0.999])
        self.forward = model
        if cfg.compile and model.backbone_name != "mamba3":
            # S4: compiled training forward; evaluation uses the eager module
            self.forward = torch.compile(model)

    def compute_loss(self, batch):
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = batch
        padding_mask = ~mask.to(torch.bool)
        with autocast_ctx(self.cfg.precision, self.device):
            action_preds, cost_preds, state_preds = self.forward(
                states=states,
                actions=actions,
                returns_to_go=returns,
                costs_to_go=costs_return,
                time_steps=time_steps,
                padding_mask=padding_mask,
                episode_cost=episode_cost,
            )
        stats = {}
        if self.stochastic:
            dist = Normal(action_preds.loc.float(), action_preds.scale.float())
            log_likelihood = dist.log_prob(actions)[mask > 0].mean()
            entropy = dist.entropy()[mask > 0].mean()
            entropy_reg = self.model.temperature().detach()
            if self.no_entropy:
                entropy_reg = 0.0
            act_loss = -(log_likelihood + entropy_reg * entropy)
            stats.update(nll=-log_likelihood.detach(), ent=entropy.detach(),
                         ent_reg=torch.as_tensor(entropy_reg).detach())
        else:
            entropy = None
            act_loss = F.mse_loss(action_preds.float(), actions.detach(), reduction="none")
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()

        cost_preds = cost_preds.float().reshape(-1, 2)
        costs = costs.flatten().long().detach()
        cost_loss = F.nll_loss(cost_preds, costs, reduction="none")
        cost_loss = (cost_loss * mask.flatten()).mean()
        pred = cost_preds.data.max(dim=1)[1]
        correct = (pred.eq(costs.data.view_as(pred)) * mask.flatten()).sum()
        acc = correct / mask.sum()

        state_loss = F.mse_loss(state_preds[:, :-1].float(), states[:, 1:].detach(),
                                reduction="none")
        state_loss = (state_loss * mask[:, :-1].unsqueeze(-1)).mean()

        loss = act_loss + self.cost_weight * cost_loss + self.state_weight * state_loss
        stats.update(all_loss=loss.detach(), act_loss=act_loss.detach(),
                     cost_loss=cost_loss.detach(), cost_acc=acc.detach(),
                     state_loss=state_loss.detach())
        return loss, entropy, stats

    def train_one_step(self, batch) -> Dict[str, torch.Tensor]:
        loss, entropy, stats = self.compute_loss(batch)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()

        if self.stochastic:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() *
                                (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()
            stats["temperature"] = self.model.temperature().detach()

        self.scheduler.step()
        return stats

    def lr(self) -> float:
        return self.scheduler.get_last_lr()[0]

    def state_dict(self) -> Dict[str, Any]:
        sd = {"model": self.model.state_dict(), "optim": self.optim.state_dict(),
              "scheduler": self.scheduler.state_dict()}
        if self.stochastic:
            sd["temp_optim"] = self.log_temperature_optimizer.state_dict()
            sd["log_temperature"] = float(self.model.log_temperature.detach().cpu())
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.model.load_state_dict(sd["model"])
        self.optim.load_state_dict(sd["optim"])
        self.scheduler.load_state_dict(sd["scheduler"])
        if self.stochastic:
            with torch.no_grad():
                self.model.log_temperature.fill_(sd["log_temperature"])
            self.log_temperature_optimizer.load_state_dict(sd["temp_optim"])


# ------------------------------------------------------------------ RNG state
def rng_state() -> Dict[str, Any]:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
          "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ------------------------------------------------------------------ loop
def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def fit(step_fn: Callable[[Any], Dict[str, Any]],
        sampler,
        cfg: TrainConfig,
        run_dir: Path,
        device: str,
        state_fn: Callable[[], Dict[str, Any]],
        load_fn: Callable[[Dict[str, Any]], None],
        lr_fn: Callable[[], float],
        validate_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
        tokens_per_sample: int = 0,
        extra_log_fn: Optional[Callable[[], Dict[str, float]]] = None) -> Dict[str, Any]:
    """Resumable training loop (checkpoint every ``ckpt_every`` steps, resume from the
    latest). Writes train_log.csv / val_log.csv and returns timing/memory stats."""
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "ckpt_latest.pt"
    start, elapsed_train, elapsed_val, val_history = 0, 0.0, 0.0, []
    peak_mem = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_fn(ck["trainer"])
        sampler.load_state_dict(ck["sampler"])
        set_rng_state(ck["rng"])
        start = ck["step"]
        elapsed_train, elapsed_val = ck["elapsed_train"], ck["elapsed_val"]
        val_history = ck.get("val_history", [])
        peak_mem = ck.get("peak_mem", 0)
        print(f"resumed from {ckpt_path} at step {start}", flush=True)
    resumed = start > 0
    train_log = CSVLog(run_dir / "train_log.csv", resume_step=start if resumed else None)
    val_log = CSVLog(run_dir / "val_log.csv", resume_step=start if resumed else None)

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    acc: Dict[str, Any] = {}
    n_acc = 0
    timing = {"data": [], "compute": []}
    t_interval = time.perf_counter()
    steps_interval = 0
    total = cfg.update_steps

    def save_ckpt(step: int) -> None:
        atomic_torch_save({"step": step, "trainer": state_fn(), "sampler": sampler.state_dict(),
                           "rng": rng_state(), "elapsed_train": elapsed_train,
                           "elapsed_val": elapsed_val, "val_history": val_history,
                           "peak_mem": peak_mem}, ckpt_path)

    for step in range(start, total):
        timed = cfg.timing_every > 0 and step % cfg.timing_every == 0
        if timed:
            _sync(device)
            t0 = time.perf_counter()
        batch = sampler.sample()
        if timed:
            _sync(device)
            t1 = time.perf_counter()
        stats = step_fn(batch)
        if timed:
            _sync(device)
            t2 = time.perf_counter()
            timing["data"].append(t1 - t0)
            timing["compute"].append(t2 - t1)
        for k, v in stats.items():
            acc[k] = acc[k] + v if k in acc else v
        n_acc += 1
        steps_interval += 1
        last = step == total - 1

        if (step + 1) % cfg.log_every == 0 or last:
            _sync(device)
            now = time.perf_counter()
            dt = now - t_interval
            elapsed_train += dt
            row = {"step": step + 1, "wall_time": round(elapsed_train + elapsed_val, 3),
                   "train_time": round(elapsed_train, 3)}
            row.update({k: float(v) / n_acc for k, v in acc.items()})
            if extra_log_fn is not None:
                row.update(extra_log_fn())
            row["lr"] = lr_fn()
            row["steps_per_s"] = steps_interval / dt
            if tokens_per_sample:
                row["tokens_per_s"] = steps_interval * cfg.batch_size * tokens_per_sample / dt
            if timing["data"]:
                row["data_ms"] = 1e3 * float(np.mean(timing["data"]))
                row["compute_ms"] = 1e3 * float(np.mean(timing["compute"]))
            if cuda:
                mem = torch.cuda.max_memory_allocated()
                peak_mem = max(peak_mem, mem)
                row["gpu_mem_mb"] = mem / 2**20
                torch.cuda.reset_peak_memory_stats()
            train_log.write(row)
            acc, n_acc, steps_interval = {}, 0, 0
            timing = {"data": [], "compute": []}
            t_interval = time.perf_counter()

        if validate_fn is not None and cfg.val_every > 0 and ((step + 1) % cfg.val_every == 0
                                                              or last):
            tv = time.perf_counter()
            vrow = validate_fn(step + 1)
            elapsed_val += time.perf_counter() - tv
            vrow = {"step": step + 1, "wall_time": round(elapsed_train + elapsed_val, 3),
                    **vrow}
            val_log.write(vrow)
            val_history.append(vrow)
            t_interval = time.perf_counter()

        if cfg.ckpt_every > 0 and (step + 1) % cfg.ckpt_every == 0 and not last:
            tc = time.perf_counter()
            save_ckpt(step + 1)
            t_interval += time.perf_counter() - tc  # exclude checkpoint I/O from train time

    # OSRL's "best by eval cost" selection, kept only as a logged diagnostic (§4.6.3)
    best = None
    for v in val_history:
        if best is None or (v["C"], -v["R"]) < (best["C"], -best["R"]):
            best = v
    return {"train_wall_s": elapsed_train, "val_wall_s": elapsed_val,
            "peak_train_mem_mb": peak_mem / 2**20 if peak_mem else None,
            "resumed_from_step": start, "best_val_step_diagnostic": best["step"] if best else None}
