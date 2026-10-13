"""CDMamba config: everything else (embeddings, heads, losses, data) is shared."""

from experiments.common.config import ModelConfig, ModelSpec


def get_spec(model_id: str = "cdmamba") -> ModelSpec:
    return ModelSpec(model=ModelConfig(
        name="cdmamba",
        backbone="mamba3",
        embedding_dim=128,  # d_model
        backbone_kwargs=dict(
            num_layers=3,
            d_state=64,
            expand=2,
            headdim=32,  # d_inner = 256 -> 8 heads
            d_conv=4,  # recorded only: Mamba-3 has no short convolution
            rope_fraction=0.5,  # mamba_ssm default
            is_mimo=False,  # SISO by default; MIMO optional (cuda impl only)
            mimo_rank=4,
            chunk_size=64,  # mamba_ssm default for SISO
            mamba_mlp=True,
            residual_dropout=0.1,
            mamba_impl="cuda",  # "ref" = pure-PyTorch fallback / CPU path
            pad_to_chunk=True,
        ),
    ))
