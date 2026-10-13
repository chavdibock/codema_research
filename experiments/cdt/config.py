"""CDT (reference): OSRL's transformer backbone, [CDT] Table 4 hyperparameters."""

from experiments.common.config import ModelConfig, ModelSpec


def get_spec(model_id: str = "cdt") -> ModelSpec:
    return ModelSpec(model=ModelConfig(
        name="cdt",
        backbone="transformer",
        embedding_dim=128,
        backbone_kwargs=dict(
            num_layers=3,
            num_heads=8,
            attention_dropout=0.1,
            residual_dropout=0.1,
        ),
    ))
