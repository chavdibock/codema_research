"""CDConvFormer config: everything else (embeddings, heads, losses, data) is shared."""

from experiments.common.config import ModelConfig, ModelSpec


def get_spec(model_id: str = "cdconv") -> ModelSpec:
    return ModelSpec(model=ModelConfig(
        name="cdconv",
        backbone="conv",
        embedding_dim=128,
        backbone_kwargs=dict(
            num_layers=3,
            conv_window=2,  # timesteps; L_conv = 4 * 2 = 8 tokens (DC default window)
            residual_dropout=0.1,
        ),
    ))
