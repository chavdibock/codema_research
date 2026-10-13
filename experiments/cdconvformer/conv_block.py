"""CDConvFormer backbone: per-modality causal depthwise convolution (Decision ConvFormer,
Kim et al., ICLR 2024) adapted from 3 to 4 modalities.

    x = x + Dropout(ConvMixer(LayerNorm(x)))
    x = x + MLP(LayerNorm(x))

ConvMixer: one depthwise causal 1-D filter per modality (R, C, s, a), kernel length
L_conv = num_modalities * conv_window tokens, left-only padding L_conv - 1. The output
at position i uses the filter of the modality of token i (4 parallel Conv1d calls plus
a modality-index gather).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.common.backbones import Backbone, cdt_init_weights, register_backbone


class ModalityCausalConv(nn.Module):

    def __init__(self, d_model: int, num_modalities: int, kernel_size: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=0)
            for _ in range(num_modalities)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        L = x.shape[1]
        xt = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))  # left-only padding
        ys = torch.stack([conv(xt) for conv in self.convs], dim=0)  # [M, B, D, L]
        pos = torch.arange(L, device=x.device)
        out = ys[pos % self.num_modalities, :, :, pos]  # [L, B, D]
        return out.permute(1, 0, 2)

    def step(self, x_t: torch.Tensor, buf: torch.Tensor, pos: int):
        """x_t [B, D]; buf [B, k-1, D] holds the previous k-1 inputs (zeros at start)."""
        window = torch.cat([buf, x_t.unsqueeze(1)], dim=1)  # [B, k, D]
        conv = self.convs[pos % self.num_modalities]
        y = conv(window.transpose(1, 2)).squeeze(-1)  # [B, D]
        return y, window[:, 1:]


class ConvBlock(nn.Module):

    def __init__(self, d_model: int, num_modalities: int, kernel_size: int,
                 residual_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = ModalityCausalConv(d_model, num_modalities, kernel_size)
        self.drop = nn.Dropout(residual_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(residual_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_t: torch.Tensor, buf: torch.Tensor, pos: int):
        y, buf = self.mixer.step(self.norm1(x_t), buf, pos)
        x_t = x_t + self.drop(y)
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, buf


@register_backbone("conv")
class ConvBackbone(Backbone):
    """Ignores ``padding_mask`` (causal, right padding only)."""

    supports_recurrent = True  # rolling buffer of the last L_conv - 1 inputs per layer

    def __init__(self,
                 d_model: int,
                 max_tokens: int,
                 num_modalities: int,
                 num_layers: int = 3,
                 conv_window: int = 2,
                 residual_dropout: float = 0.1):
        super().__init__(d_model, max_tokens, num_modalities)
        if conv_window not in (1, 2, 3):
            raise ValueError("conv_window must be in {1, 2, 3} timesteps")
        self.conv_window = conv_window
        self.kernel_size = num_modalities * conv_window  # L_conv = 4 * w for 4 modalities
        self.blocks = nn.ModuleList([
            ConvBlock(d_model, num_modalities, self.kernel_size, residual_dropout)
            for _ in range(num_layers)
        ])

    def init_weights(self) -> None:
        # NOTE: CDT init for Linear/LayerNorm; Conv1d keeps PyTorch's default init
        # (CDT's _init_weights does not touch Conv1d either).
        self.blocks.apply(cdt_init_weights)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def allocate_state(self, batch_size: int, device=None, dtype=None) -> Dict[str, Any]:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else torch.float32
        bufs = [
            torch.zeros(batch_size, self.kernel_size - 1, self.d_model, device=device,
                        dtype=dtype) for _ in self.blocks
        ]
        return {"bufs": bufs, "pos": 0}

    def step(self, x_t: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        pos = state["pos"]
        new_bufs: List[torch.Tensor] = []
        for block, buf in zip(self.blocks, state["bufs"]):
            x_t, buf = block.step(x_t, buf, pos)
            new_bufs.append(buf)
        return x_t, {"bufs": new_bufs, "pos": pos + 1}
