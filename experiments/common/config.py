"""pyrallis-compatible dataclass configs: base + per-task + per-model + per-experiment.

Resolution order (later wins):
    dataclass defaults  ->  per-task defaults copied from OSRL's cdt_configs
    ->  model spec (approach folder)  ->  experiment definition  ->  variant
    ->  command-line ``--override section.key=value``.
"""

import copy
import importlib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from experiments.common import osrl_compat

TASKS: List[str] = [
    "OfflineAntRun-v0",
    "OfflineCarCircle-v0",
    "OfflineCarRun-v0",
    "OfflineDroneCircle-v0",
    "OfflineDroneRun-v0",
]
TASK_SHORT = {
    "OfflineAntRun-v0": "Ant-Run",
    "OfflineCarCircle-v0": "Car-Circle",
    "OfflineCarRun-v0": "Car-Run",
    "OfflineDroneCircle-v0": "Drone-Circle",
    "OfflineDroneRun-v0": "Drone-Run",
}
KAPPAS: Tuple[int, ...] = (10, 20, 40)

# D2: OSRL's Car-Circle targets are (450,10) (500,20) (550,50). We use kappa in
# {10, 20, 40} for every task.
# NOTE: rule for the kappa=40 reward target on Car-Circle: keep OSRL's third reward
# target (550) and set its cost to 40, as the other tasks pair their largest reward
# target with kappa=40. Documented in NOTES.md.
TARGET_OVERRIDES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "OfflineCarCircle-v0": ((450.0, 10.0), (500.0, 20.0), (550.0, 40.0)),
}

# Fields of OSRL's CDTTrainConfig that define the data pipeline (copied per task).
_OSRL_DATA_FIELDS = (
    "seq_len", "reward_scale", "cost_scale", "outliers_percent", "noise_scale",
    "inpaint_ranges", "epsilon", "density", "deg", "pf_sample", "beta", "augment_percent",
    "max_reward", "min_reward", "max_rew_decrease", "cost_reverse", "pf_only", "rmin",
    "cost_bins", "npb", "cost_sample", "linear", "start_sampling", "prob", "random_aug",
    "aug_rmin", "aug_rmax", "aug_cmin", "aug_cmax", "cgap", "rstd", "cstd", "cost_limit")


@dataclass
class DataConfig:
    task: str = "OfflineCarCircle-v0"
    seq_len: int = 10  # K
    # 0 -> env.spec.max_episode_steps (D1); recorded in env_info.json
    episode_len: int = 0
    reward_scale: float = 0.1
    cost_scale: float = 1.0
    # kappa used for env.set_target_cost before pre_process_data (outliers/epsilon/reduce)
    cost_limit: float = 10.0
    # --- DSRL pre_process_data options ---
    outliers_percent: Optional[float] = None  # fraction, e.g. 0.01 == 1 %
    noise_scale: Optional[float] = None
    inpaint_ranges: Optional[List[List[float]]] = None
    epsilon: Optional[float] = None
    density: float = 1.0
    # --- our additions (E7/E8), applied after pre_process_data ---
    reduce_rule: str = "none"  # "none" | "cdt": [CDT] App. A reduced dataset
    post_density: float = 1.0  # <1: DSRL grid filter (DENSITY_CFG) after outlier injection
    # --- OSRL SequenceDataset / augmentation (per-task defaults from cdt_configs) ---
    deg: int = 4
    pf_sample: bool = False
    beta: float = 1.0
    augment_percent: float = 0.2
    max_reward: float = 600.0
    min_reward: float = 1.0
    max_rew_decrease: float = 100.0
    cost_reverse: bool = False
    pf_only: bool = False
    rmin: float = 300.0
    cost_bins: int = 60
    npb: int = 5
    cost_sample: bool = True
    linear: bool = True
    start_sampling: bool = False
    prob: float = 0.2
    random_aug: float = 0.0
    aug_rmin: float = 400.0
    aug_rmax: float = 500.0
    aug_cmin: float = -2.0
    aug_cmax: float = 25.0
    cgap: float = 5.0
    rstd: float = 1.0
    cstd: float = 0.2


@dataclass
class ModelConfig:
    name: str = "cdt"
    kind: str = "sequence"  # "sequence" | "mlp"
    # ---- sequence models (SequencePolicy) ----
    backbone: str = "transformer"
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)
    embedding_dim: int = 128
    embedding_dropout: float = 0.1
    time_emb: bool = True
    use_rew: bool = True
    use_cost: bool = True
    cost_transform: bool = True
    add_cost_feat: bool = False
    mul_cost_feat: bool = False
    cat_cost_feat: bool = False
    action_head_layers: int = 1
    cost_prefix: bool = False
    stochastic: bool = True
    init_temperature: float = 0.1
    # ---- MLP baselines (OSRL algorithms) ----
    algo: Optional[str] = None  # bc | bcql | bearl | cpq | coptidice
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    update_steps: int = 100_000
    batch_size: int = 2048
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    lr_warmup_steps: int = 500
    loss_cost_weight: float = 0.02
    loss_state_weight: float = 0.0
    no_entropy: bool = False
    temperature_lr: float = 1e-4
    # --- performance engineering (spec §4.8) ---
    # NOTE: bf16 is the default; E0 checks bf16 vs fp32 on Car-Run and the paper must
    # switch the default to fp32 if the check fails (spec §4.8, last paragraph).
    precision: str = "bf16"  # "fp32" | "bf16"
    tf32: bool = True
    fused_adamw: bool = True
    compile: bool = False  # S4; ignored for the mamba3 backbone
    sampler: str = "gpu"  # "gpu" (S1) | "osrl" (reference SequenceDataset + DataLoader)
    num_workers: int = 2  # only for sampler == "osrl"
    # --- logging / checkpointing / validation ---
    log_every: int = 100
    # NOTE: §8.2 says every 10k steps, S6 says 25k. Checkpoints of these tiny models
    # cost milliseconds, so we keep the more robust 10k (configurable).
    ckpt_every: int = 10_000
    val_every: int = 5_000
    val_episodes: int = 5
    val_seeds: List[int] = field(default_factory=lambda: [0])  # S6: seed 0 only
    timing_every: int = 100  # synchronized data/compute timing once per N steps


@dataclass
class EvalConfig:
    eval_episodes: int = 20
    test_seed_base: int = 1_000_000
    val_seed_base: int = 500_000
    seed_split: str = "test"  # "test" | "val" (E0 tuning selects on validation seeds)
    eval_sample: bool = False  # False: Gaussian mean action, as OSRL
    vectorized: bool = True
    num_workers: int = 0  # 0 -> auto (CDX_EVAL_WORKERS or available cores)
    threads: int = 4  # torch.set_num_threads during rollouts (spec §5.3)
    vec_context: str = "fork"
    # "kappa10" | "multi" (kappa 10/20/40) | "none" (fixed-threshold MLP baselines)
    target_set: str = "multi"
    targets: List[List[float]] = field(default_factory=list)  # resolved (reward, cost)
    inference_mode: str = "window"  # "window" | "recurrent"
    # NOTE: False = one batched forward per time step (S2). Batched fp32 matmuls round
    # differently from batch-1 matmuls, which the chaotic dynamics amplify; True gives
    # bit-identical results to sequential evaluation (used by T11). See NOTES.md.
    batch_invariant: bool = False


@dataclass
class RunConfig:
    exp: str = "E1"
    model_id: str = "cdt"
    variant: Optional[str] = None
    task: str = "OfflineCarCircle-v0"
    seed: int = 0
    out: str = "results"
    device: str = "auto"
    cache_dir: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """What an approach folder exports: its model config plus required overrides."""
    model: ModelConfig
    overrides: Dict[str, Any] = field(default_factory=dict)
    pack_kind: str = "seq"  # "seq" (6 per slice) | "mlp" (12 per slice)


# model id -> module exporting ``get_spec(model_id) -> ModelSpec``
MODEL_MODULES: Dict[str, str] = {
    "cdt": "experiments.cdt.config",
    "cdt_light": "experiments.cdt_light.config",
    "cdmamba": "experiments.cdmamba.config",
    "cdconv": "experiments.cdconvformer.config",
}
OUR_MODELS = ["cdt", "cdt_light", "cdmamba", "cdconv"]


def get_model_spec(model_id: str) -> ModelSpec:
    module_name = MODEL_MODULES.get(model_id, "experiments.baselines.config")
    module = importlib.import_module(module_name)
    return module.get_spec(model_id)


def task_defaults(task: str) -> Dict[str, Any]:
    """Per-task data defaults and target pairs copied from OSRL's cdt_configs."""
    osrl_cfg = osrl_compat.osrl_task_config("cdt", task)
    data = {k: osrl_cfg[k] for k in _OSRL_DATA_FIELDS if k in osrl_cfg}
    targets = TARGET_OVERRIDES.get(task, tuple(tuple(map(float, t))
                                               for t in osrl_cfg["target_returns"]))
    return {"data": data, "targets": [list(t) for t in targets],
            "osrl_episode_len": osrl_cfg["episode_len"]}


def kappa_targets(task: str) -> Dict[float, Tuple[float, float]]:
    """Map kappa -> (reward target, kappa) for the task (spec §5.2)."""
    return {float(c): (float(r), float(c)) for r, c in task_defaults(task)["targets"]}


def set_by_path(cfg: Any, path: str, value: Any) -> None:
    """Set ``cfg.section.key`` (dataclasses) or nested dict keys from a dotted path."""
    parts = path.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        if not hasattr(obj, last):
            raise KeyError(f"unknown config key: {path}")
        setattr(obj, last, value)


def parse_override(s: str) -> Tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override must look like key=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def apply_overrides(cfg: RunConfig, overrides: Dict[str, Any]) -> None:
    for k, v in overrides.items():
        set_by_path(cfg, k, copy.deepcopy(v))


def to_dict(cfg: Any) -> Dict[str, Any]:
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


def from_dict(d: Dict[str, Any]) -> RunConfig:
    """Inverse of ``to_dict`` for RunConfig (used when loading config.yaml)."""
    d = copy.deepcopy(d)
    kw = {}
    for f in fields(RunConfig):
        if f.name not in d:
            continue
        if f.name == "data":
            kw[f.name] = DataConfig(**_known(DataConfig, d[f.name]))
        elif f.name == "model":
            kw[f.name] = ModelConfig(**_known(ModelConfig, d[f.name]))
        elif f.name == "train":
            t = _known(TrainConfig, d[f.name])
            if "betas" in t:
                t["betas"] = tuple(t["betas"])
            kw[f.name] = TrainConfig(**t)
        elif f.name == "eval":
            kw[f.name] = EvalConfig(**_known(EvalConfig, d[f.name]))
        else:
            kw[f.name] = d[f.name]
    return RunConfig(**kw)


def _known(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}
