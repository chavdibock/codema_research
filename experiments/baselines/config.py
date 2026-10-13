"""Baseline specs (spec §4.7). OSRL implementations, our protocol.

Sequence baselines (DT-Cost, unconstrained DT) use SequencePolicy with the CDT
transformer. MLP baselines wrap OSRL's BC/BCQL/BEARL/CPQ/COptiDICE; their
hyperparameters come from OSRL's per-task configs, except ``update_steps`` (100k for
every method, our protocol) and ``episode_len`` (env spec, D1).
"""

import copy

from experiments.common.config import ModelConfig, ModelSpec

_CDT_TRANSFORMER = dict(num_layers=3, num_heads=8, attention_dropout=0.1,
                        residual_dropout=0.1)

# NOTE: our approximation of [CDT]'s DT-Cost ("vanilla DT + cost token"): CDT network
# with a deterministic head (MSE), no Pareto augmentation, uniform trajectory sampling
# and no auxiliary cost loss. Documented in NOTES.md / the paper.
DT_COST_OVERRIDES = {"data.augment_percent": 0.0, "data.cost_sample": False,
                     "train.loss_cost_weight": 0.0}

# MLP baselines: model id -> (OSRL algo, algo kwargs overriding OSRL's per-task config)
MLP_BASELINES = {
    "bc_safe": ("bc", {"bc_mode": "safe"}),
    "bc_all": ("bc", {"bc_mode": "all"}),
    "bc_risky": ("bc", {"bc_mode": "risky"}),
    "bc_frontier": ("bc", {"bc_mode": "frontier"}),
    "bc_boundary": ("bc", {"bc_mode": "boundary"}),
    # MT-BC: OSRL appends the episode cost return to the observation for training and the
    # target kappa at evaluation (verified in osrl/algorithms/bc.py, BCTrainer.rollout).
    "mt_bc": ("bc", {"bc_mode": "multi-task"}),
    "bcq_lag": ("bcql", {}),
    "bear_lag": ("bearl", {}),
    "cpq": ("cpq", {}),
    "coptidice": ("coptidice", {}),
    # E7: Lagrangian disabled (PID gains 0 => multiplier 0 => no cost penalty)
    "bcq": ("bcql", {"PID": [0.0, 0.0, 0.0]}),
    "bear": ("bearl", {"PID": [0.0, 0.0, 0.0]}),
}


def get_spec(model_id: str) -> ModelSpec:
    if model_id == "dt_cost":
        return ModelSpec(model=ModelConfig(name="dt_cost", backbone="transformer",
                                           embedding_dim=128, stochastic=False,
                                           backbone_kwargs=dict(_CDT_TRANSFORMER)),
                         overrides=dict(DT_COST_OVERRIDES))
    if model_id == "dt":
        # E7 unconstrained DT: DT-Cost without the cost token
        return ModelSpec(model=ModelConfig(name="dt", backbone="transformer",
                                           embedding_dim=128, stochastic=False,
                                           use_cost=False,
                                           backbone_kwargs=dict(_CDT_TRANSFORMER)),
                         overrides=dict(DT_COST_OVERRIDES))
    if model_id in MLP_BASELINES:
        algo, kw = MLP_BASELINES[model_id]
        # NOTE: S3 for the MLP baselines = TF32 only. bf16 autocast inside OSRL's
        # Q-learning/DICE updates risks numerical instability, and OSRL creates its own
        # (non-fused) Adam optimizers internally. bf16 can still be enabled with
        # --override train.precision=bf16.
        return ModelSpec(model=ModelConfig(name=model_id, kind="mlp", algo=algo,
                                           algo_kwargs=copy.deepcopy(kw)),
                         overrides={"train.precision": "fp32"},
                         pack_kind="mlp")
    raise KeyError(f"unknown model id {model_id!r}")
