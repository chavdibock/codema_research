"""OSRL's MLP baselines (BC, BCQ-L, BEAR-L, CPQ, COptiDICE) under our protocol.

The algorithms and their ``train_one_step`` are OSRL's, unchanged. The wrapper supplies
the cached pre-processed data through a GPU transition sampler, our step count and seeds,
seeded vectorized evaluation (S2), resumable checkpoints and our output format.
Constructor arguments are exactly those passed by OSRL's ``examples/train/train_*.py``.
"""

import contextlib
import copy
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.common import osrl_compat
from experiments.common.logging import StoreLogger

MODEL_ARGS = {
    "bc": ["a_hidden_sizes", "episode_len"],
    "bcql": ["a_hidden_sizes", "c_hidden_sizes", "vae_hidden_sizes", "sample_action_num",
             "PID", "gamma", "tau", "lmbda", "beta", "phi", "num_q", "num_qc", "cost_limit",
             "episode_len"],
    "bearl": ["a_hidden_sizes", "c_hidden_sizes", "vae_hidden_sizes", "sample_action_num",
              "gamma", "tau", "beta", "lmbda", "mmd_sigma", "target_mmd_thresh",
              "start_update_policy_step", "num_q", "num_qc", "PID", "cost_limit",
              "episode_len"],
    "cpq": ["a_hidden_sizes", "c_hidden_sizes", "vae_hidden_sizes", "sample_action_num",
            "gamma", "tau", "beta", "num_q", "num_qc", "qc_scalar", "cost_limit",
            "episode_len"],
    "coptidice": ["f_type", "a_hidden_sizes", "c_hidden_sizes", "gamma", "alpha",
                  "cost_ub_epsilon", "num_nu", "num_chi", "cost_limit", "episode_len"],
}
TRAINER_ARGS = {
    "bc": ["actor_lr", "bc_mode", "cost_limit"],
    "bcql": ["actor_lr", "critic_lr", "vae_lr", "reward_scale", "cost_scale"],
    "bearl": ["actor_lr", "critic_lr", "vae_lr", "reward_scale", "cost_scale"],
    "cpq": ["actor_lr", "critic_lr", "alpha_lr", "vae_lr", "reward_scale", "cost_scale"],
    "coptidice": ["actor_lr", "critic_lr", "scalar_lr", "reward_scale", "cost_scale"],
}


def _classes(algo: str):
    from osrl import algorithms as A

    return {
        "bc": (A.BC, A.BCTrainer),
        "bcql": (A.BCQL, A.BCQLTrainer),
        "bearl": (A.BEARL, A.BEARLTrainer),
        "cpq": (A.CPQ, A.CPQTrainer),
        "coptidice": (A.COptiDICE, A.COptiDICETrainer),
    }[algo]


def resolve_osrl_config(model_cfg, task: str, episode_len: int) -> Dict[str, Any]:
    """OSRL per-task config + our algo_kwargs; episode_len from the env spec (D1)."""
    osrl_cfg = osrl_compat.osrl_task_config(model_cfg.algo, task)
    osrl_cfg.update(copy.deepcopy(model_cfg.algo_kwargs))
    osrl_cfg["episode_len"] = episode_len
    return osrl_cfg


def prepare_transitions(algo: str, osrl_cfg: Dict[str, Any],
                        data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    data = {k: np.array(v, copy=True) for k, v in data.items()}
    if algo == "bc":
        from osrl.common.dataset import process_bc_dataset

        process_bc_dataset(data, osrl_cfg["cost_limit"], osrl_cfg["gamma"],
                           osrl_cfg["bc_mode"])
    return data


def coptidice_stats(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Same statistics as OSRL's TransitionDataset(state_init=True).get_dataset_states()."""
    done = np.logical_or(data["terminals"], data["timeouts"]).astype(np.float32)
    is_init = done.copy()
    is_init[1:] = is_init[:-1]
    is_init[0] = 1.0
    return {"init_state_propotion": is_init.mean(),
            "observations_std": data["observations"].std(0, keepdims=True),
            "actions_std": data["actions"].std(0, keepdims=True)}


def build_mlp(algo: str, osrl_cfg: Dict[str, Any], state_dim: int, action_dim: int,
              max_action: float, device: str, extra_kwargs: Dict[str, Any]):
    """``extra_kwargs``: coptidice_stats(data) for COptiDICE, {} otherwise."""
    model_cls, trainer_cls = _classes(algo)
    kw = {k: osrl_cfg[k] for k in MODEL_ARGS[algo]}
    if algo == "bc" and osrl_cfg.get("bc_mode") == "multi-task":
        state_dim += 1
    kw.update(extra_kwargs)
    model = model_cls(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                      device=device, **kw).to(device)
    logger = StoreLogger()
    tkw = {k: osrl_cfg[k] for k in TRAINER_ARGS[algo]}
    trainer = trainer_cls(model, None, logger=logger, device=device, **tkw)
    return model, trainer, logger


def make_step_fn(algo: str, trainer, precision: str, device: str) -> Callable:
    if precision == "bf16":
        dev_type = "cuda" if device.startswith("cuda") else "cpu"
        ctx = lambda: torch.autocast(device_type=dev_type, dtype=torch.bfloat16)  # noqa
    else:
        ctx = contextlib.nullcontext

    def step(batch) -> Dict[str, Any]:
        with ctx():
            if algo == "bc":
                obs, _, act, _, _, _ = batch
                trainer.train_one_step(obs, act)
            elif algo == "coptidice":
                trainer.train_one_step(list(batch))
            else:
                trainer.train_one_step(*batch)
        return {}

    return step


# ------------------------------------------------------------ checkpointing
def capture_state(model: nn.Module) -> Dict[str, Any]:
    """state_dict + optimizers + plain tensor attributes (e.g. log_alpha, lmbda) +
    PID controller state + step counters, so OSRL models resume exactly."""
    from osrl.common.net import LagrangianPIDController

    st: Dict[str, Any] = {"model": model.state_dict(), "optims": {}, "tensors": {},
                          "scalars": {}, "controller": None}
    for name, v in vars(model).items():
        if isinstance(v, torch.optim.Optimizer):
            st["optims"][name] = v.state_dict()
        elif torch.is_tensor(v) and not isinstance(v, nn.Parameter):
            st["tensors"][name] = v.detach().clone()
        elif name == "n_train_steps":
            st["scalars"][name] = v
        elif isinstance(v, LagrangianPIDController):
            st["controller"] = {k: (x.detach().clone() if torch.is_tensor(x) else x)
                                for k, x in vars(v).items()}
    return st


def restore_state(model: nn.Module, st: Dict[str, Any]) -> None:
    from osrl.common.net import LagrangianPIDController

    model.load_state_dict(st["model"])
    for name, sd in st["optims"].items():
        getattr(model, name).load_state_dict(sd)
    for name, t in st["tensors"].items():
        cur = getattr(model, name, None)
        if torch.is_tensor(cur):
            with torch.no_grad():
                cur.data.copy_(t.to(cur.device))
        else:
            # transient tensors that only exist after the first update() (COptiDICE's
            # _lmbda = softplus(lmbda)); update() recomputes them before use
            setattr(model, name, t.to(model.device if hasattr(model, "device") else "cpu"))
    for name, v in st["scalars"].items():
        setattr(model, name, v)
    if st["controller"] is not None:
        for name, v in vars(model).items():
            if isinstance(v, LagrangianPIDController):
                for k, x in st["controller"].items():
                    dev = model.device if hasattr(model, "device") else "cpu"
                    setattr(v, k, x.to(dev) if torch.is_tensor(x) else x)


# ------------------------------------------------------------ evaluation
class MLPActor:
    """Batched deterministic actions, exactly as each OSRL trainer's rollout()."""

    def __init__(self, model, algo: str, bc_mode: str, target_C: np.ndarray, device: str):
        self.model, self.algo, self.device = model, algo, device
        self.multi_task = algo == "bc" and bc_mode == "multi-task"
        self.kappa = torch.as_tensor(target_C, dtype=torch.float32, device=device)[:, None]
        self.obs = None

    def reset(self, obs: np.ndarray) -> None:
        self.obs = obs

    @torch.no_grad()
    def act(self, t: int) -> torch.Tensor:
        o = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        m = self.model
        if self.multi_task:
            o = torch.cat([o, self.kappa], dim=-1)  # OSRL: np.append(obs, cost_limit)
        if self.algo == "bc":
            return m.actor(o)
        if self.algo == "bcql":
            return m.actor(o, m.vae.decode(o))
        if self.algo in ("bearl", "cpq"):
            return m._actor_forward(o, True, True)[0]
        if self.algo == "coptidice":
            return m.actor(o, True, True)[0]
        raise ValueError(self.algo)

    def observe(self, t: int, act, obs_next: np.ndarray, reward: np.ndarray,
                cost: np.ndarray) -> None:
        self.obs = obs_next


def mlp_actor_factory(model, algo: str, bc_mode: str, device: str) -> Callable:

    def make(target_R: np.ndarray, target_C: np.ndarray):
        return MLPActor(model, algo, bc_mode, target_C, device)

    return make
