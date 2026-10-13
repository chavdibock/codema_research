"""Lightweight CDT: parameter-matched transformer (capacity control, spec §4.3).

Chosen with ``python -m experiments.common.profiling --match-light`` on OfflineCarCircle-v0:
grid layers {1,2,3} x d {48,64,96,128} x heads {2,4,8}, closest total trainable
parameter count (embeddings and heads included) to min(params(cdmamba), params(cdconv)).
    target   = 454542 (cdmamba 803646, cdconv 454542)
    achieved = 440590 (-3.1 %, within +-10 %: True)
"""

from experiments.common.config import ModelConfig, ModelSpec


def get_spec(model_id: str = "cdt_light") -> ModelSpec:
    return ModelSpec(model=ModelConfig(
        name="cdt_light",
        backbone="transformer",
        embedding_dim=128,
        backbone_kwargs=dict(num_layers=2, num_heads=8, attention_dropout=0.1,
                             residual_dropout=0.1),
    ))
