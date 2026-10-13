"""Backbone interface + registry (transformer, mamba3, conv).

Every backbone maps ``[B, L, D] -> [B, L, D]`` and must be strictly causal. Approach
folders register their backbone; the registry imports them lazily by name.
"""

import importlib
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from osrl.algorithms.cdt import CDT
from osrl.common.net import TransformerBlock

# CDT's generic init: N(0, 0.02) for Linear/Embedding, zeros/ones for LayerNorm.
cdt_init_weights: Callable[[nn.Module], None] = CDT._init_weights


class Backbone(nn.Module):
    """Token mixer stack. Subclasses set ``supports_recurrent`` if ``step`` exists."""

    supports_recurrent: bool = False

    def __init__(self, d_model: int, max_tokens: int, num_modalities: int):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.num_modalities = num_modalities

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def init_weights(self) -> None:
        """Backbone-specific initialization (called once by SequencePolicy)."""

    def allocate_state(self, batch_size: int, device=None, dtype=None) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no recurrent mode")

    def step(self, x_t: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError(f"{type(self).__name__} has no recurrent mode")

    def state_nbytes(self, state: Any) -> int:
        """Size of the carried recurrent state in bytes (E6.5)."""
        total = 0
        stack = [state]
        while stack:
            s = stack.pop()
            if torch.is_tensor(s):
                total += s.numel() * s.element_size()
            elif isinstance(s, dict):
                stack.extend(s.values())
            elif isinstance(s, (list, tuple)):
                stack.extend(s)
        return total


_REGISTRY: Dict[str, Type[Backbone]] = {}
_LAZY_MODULES = {
    "transformer": "experiments.common.backbones",
    "mamba3": "experiments.cdmamba.mamba_block",
    "conv": "experiments.cdconvformer.conv_block",
}


def register_backbone(name: str):

    def deco(cls: Type[Backbone]) -> Type[Backbone]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get_backbone_class(name: str) -> Type[Backbone]:
    if name not in _REGISTRY and name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[name])
    if name not in _REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; known: {sorted(_LAZY_MODULES)}")
    return _REGISTRY[name]


def build_backbone(name: str, d_model: int, max_tokens: int, num_modalities: int,
                   **kwargs) -> Backbone:
    return get_backbone_class(name)(d_model=d_model, max_tokens=max_tokens,
                                    num_modalities=num_modalities, **kwargs)


@register_backbone("transformer")
class TransformerBackbone(Backbone):
    """OSRL's CDT blocks (``osrl.common.net.TransformerBlock``), unchanged."""

    def __init__(self,
                 d_model: int,
                 max_tokens: int,
                 num_modalities: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 attention_dropout: float = 0.1,
                 residual_dropout: float = 0.1):
        super().__init__(d_model, max_tokens, num_modalities)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=max_tokens,
                embedding_dim=d_model,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            ) for _ in range(num_layers)
        ])

    def init_weights(self) -> None:
        # identical traversal order to OSRL's self.apply(self._init_weights) on `blocks`
        self.blocks.apply(cdt_init_weights)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return x
