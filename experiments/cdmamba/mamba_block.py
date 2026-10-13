"""CDMamba backbone: Decision-Mamba style MetaFormer blocks with a Mamba-3 mixer.

    x = x + Dropout(Mamba3(LayerNorm(x)))
    x = x + MLP(LayerNorm(x))            # if mamba_mlp (4d hidden, GELU, dropout)

``mamba_impl="cuda"`` uses the official ``mamba_ssm.modules.mamba3.Mamba3`` (Triton
kernels); ``mamba_impl="ref"`` uses our pure-PyTorch ``Mamba3Ref`` (CPU tests, CPU
latency, and the training fallback if the kernels do not work on Snelius).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.common.backbones import Backbone, cdt_init_weights, register_backbone
from experiments.cdmamba.mamba3_ref import (Mamba3Ref, mamba3_ref_alloc_state,
                                            mamba3_ref_step)

# Verified in the mamba_ssm 2.3.2.post1 source (see NOTES.md).
MAMBA3_IMPORT_PATH = "mamba_ssm.modules.mamba3.Mamba3"


def import_official_mamba3():
    """Import the official Mamba-3 module; raises ImportError with a clear message."""
    try:
        from mamba_ssm.modules.mamba3 import Mamba3
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"cannot import {MAMBA3_IMPORT_PATH} ({type(e).__name__}: {e}). Install "
            "mamba_ssm>=2.3.2 built against the loaded CUDA, or run with "
            "--override model.backbone_kwargs.mamba_impl=ref") from e
    return Mamba3


def make_mixer(impl: str, d_model: int, **mixer_kwargs) -> nn.Module:
    if impl == "cuda":
        return import_official_mamba3()(d_model=d_model, **mixer_kwargs)
    if impl == "ref":
        return Mamba3Ref(d_model=d_model, **mixer_kwargs)
    raise ValueError(f"mamba_impl must be 'cuda' or 'ref', got {impl!r}")


class MambaBlock(nn.Module):

    def __init__(self, d_model: int, impl: str, mixer_kwargs: Dict[str, Any],
                 residual_dropout: float, use_mlp: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = make_mixer(impl, d_model, **mixer_kwargs)
        self.drop = nn.Dropout(residual_dropout)
        self.use_mlp = use_mlp
        if use_mlp:
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(residual_dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        if self.use_mlp:
            x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_t: torch.Tensor, state: Dict[str, torch.Tensor]):
        y, state = mamba3_ref_step(self.mixer, self.norm1(x_t), state)
        x_t = x_t + self.drop(y)
        if self.use_mlp:
            x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, state


@register_backbone("mamba3")
class Mamba3Backbone(Backbone):
    """Stack of MambaBlocks. Ignores ``padding_mask``: right padding cannot affect
    earlier positions of a causal model (spec §4.1, test T3)."""

    supports_recurrent = True

    def __init__(self,
                 d_model: int,
                 max_tokens: int,
                 num_modalities: int,
                 num_layers: int = 3,
                 d_state: int = 64,
                 expand: int = 2,
                 headdim: int = 32,
                 d_conv: int = 4,
                 rope_fraction: float = 0.5,
                 is_mimo: bool = False,
                 mimo_rank: int = 4,
                 chunk_size: int = 64,
                 mamba_mlp: bool = True,
                 residual_dropout: float = 0.1,
                 mamba_impl: str = "cuda",
                 pad_to_chunk: bool = True):
        super().__init__(d_model, max_tokens, num_modalities)
        # NOTE: d_conv is recorded for completeness; Mamba-3 has no short convolution.
        self.d_conv = d_conv
        self.impl = mamba_impl
        self.chunk_size = chunk_size
        self.pad_to_chunk = pad_to_chunk
        mixer_kwargs = dict(d_state=d_state, expand=expand, headdim=headdim,
                            rope_fraction=rope_fraction, is_mimo=is_mimo,
                            mimo_rank=mimo_rank, chunk_size=chunk_size)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, mamba_impl, mixer_kwargs, residual_dropout, mamba_mlp)
            for _ in range(num_layers)
        ])

    def init_weights(self) -> None:
        """CDT init for LayerNorm/MLP only; the Mamba-3 mixer keeps its own init
        (dt_bias, B/C biases, D, in/out projections) - spec §4.1."""
        for block in self.blocks:
            for name, child in block.named_children():
                if name != "mixer":
                    child.apply(cdt_init_weights)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        L = x.shape[1]
        pad = 0
        if self.impl == "cuda" and self.pad_to_chunk and L % self.chunk_size != 0:
            # right-pad to a chunk multiple, run, slice back (causal => exact)
            pad = self.chunk_size - L % self.chunk_size
            x = F.pad(x, (0, 0, 0, pad))
        for block in self.blocks:
            x = block(x)
        return x[:, :L] if pad else x

    def allocate_state(self, batch_size: int, device=None, dtype=None) -> List[Dict]:
        return [mamba3_ref_alloc_state(b.mixer, batch_size, device) for b in self.blocks]

    def step(self, x_t: torch.Tensor, state: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
        """Recurrent step (pure PyTorch, any device). NOTE: the official CuteDSL step
        kernel is H100-only per its docstring, so recurrent latency uses this path."""
        new_state = []
        for block, s in zip(self.blocks, state):
            x_t, s = block.step(x_t, s)
            new_state.append(s)
        return x_t, new_state
