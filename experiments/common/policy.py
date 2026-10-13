"""SequencePolicy: ``osrl.algorithms.CDT`` with a pluggable backbone (spec §4.1).

The only change w.r.t. OSRL's CDT is ``self.blocks`` -> ``self.backbone``. Module
construction order, initialization order and the forward pass are kept identical so
that ``SequencePolicy(backbone="transformer")`` reproduces ``osrl.algorithms.CDT``
numerically for the same seed (test T1).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from osrl.common.net import DiagGaussianActor, mlp
from experiments.common.backbones import Backbone, build_backbone, cdt_init_weights


class SequencePolicy(nn.Module):
    """CDT embeddings + heads around a ``Backbone``.

    Args are those of ``osrl.algorithms.CDT`` plus ``backbone`` (registry name) and
    ``backbone_kwargs``. ``num_layers``/``num_heads``/dropouts of the transformer are
    backbone kwargs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        embedding_dropout: float = 0.0,
        time_emb: bool = True,
        use_rew: bool = False,
        use_cost: bool = False,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
        cat_cost_feat: bool = False,
        action_head_layers: int = 1,
        cost_prefix: bool = False,
        stochastic: bool = False,
        init_temperature: float = 0.1,
        target_entropy: Optional[float] = None,
        backbone: str = "transformer",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        self.use_cost_transform = cost_transform
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = stochastic
        self.backbone_name = backbone

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.seq_repeat = 2
        self.use_rew = use_rew
        self.use_cost = use_cost
        if self.use_cost:
            self.cost_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew:
            self.return_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1

        dt_seq_len = self.seq_repeat * seq_len

        self.cost_prefix = cost_prefix
        if self.cost_prefix:
            self.prefix_emb = nn.Linear(1, embedding_dim)
            dt_seq_len += 1

        # the one change w.r.t. OSRL: self.blocks -> self.backbone (same position)
        self.backbone: Backbone = build_backbone(backbone,
                                                 d_model=embedding_dim,
                                                 max_tokens=dt_seq_len,
                                                 num_modalities=self.seq_repeat,
                                                 **(backbone_kwargs or {}))

        action_emb_dim = 2 * embedding_dim if self.cat_cost_feat else embedding_dim

        if self.stochastic:
            if action_head_layers >= 2:
                self.action_head = nn.Sequential(
                    nn.Linear(action_emb_dim, action_emb_dim), nn.GELU(),
                    DiagGaussianActor(action_emb_dim, action_dim))
            else:
                self.action_head = DiagGaussianActor(action_emb_dim, action_dim)
        else:
            self.action_head = mlp([action_emb_dim] * action_head_layers + [action_dim],
                                   activation=nn.GELU,
                                   output_activation=nn.Identity)
        self.state_pred_head = nn.Linear(embedding_dim, state_dim)
        # a classification problem
        self.cost_pred_head = nn.Linear(embedding_dim, 2)

        if self.stochastic:
            # plain tensor, as in OSRL (not a Parameter: excluded from AdamW / counts)
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

        self._init_all_weights()

    # ------------------------------------------------------------------ init
    def _init_all_weights(self) -> None:
        """Equivalent to OSRL's ``self.apply(self._init_weights)`` except that the
        backbone initializes itself (spec §4.1: never overwrite Mamba's A_log/dt_bias/D).
        Children are visited in registration order, as ``nn.Module.apply`` does."""
        for child in self.children():
            if child is self.backbone:
                child.init_weights()
            else:
                child.apply(cdt_init_weights)
        cdt_init_weights(self)

    def cost_transform(self, x: torch.Tensor) -> torch.Tensor:
        return 50 - x

    def temperature(self):
        if self.stochastic:
            return self.log_temperature.exp()
        return None

    # keep log_temperature on the module's device and inside state_dict
    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        if self.stochastic:
            with torch.no_grad():
                t = fn(self.log_temperature.detach())
            self.log_temperature = t.detach().clone().requires_grad_(True)
        return self

    def get_extra_state(self) -> Dict[str, Any]:
        if self.stochastic:
            return {"log_temperature": float(self.log_temperature.detach().cpu())}
        return {}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if self.stochastic and state and "log_temperature" in state:
            with torch.no_grad():
                self.log_temperature.fill_(state["log_temperature"])

    def num_parameters(self, backbone_only: bool = False) -> int:
        module = self.backbone if backbone_only else self
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    # --------------------------------------------------------------- forward
    def _embed(self, states, actions, returns_to_go, costs_to_go, time_steps):
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]

        if self.use_cost_transform:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        costs_emb = None
        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)
        return seq_list, costs_emb

    def _heads(self, state_feat, action_feature, costs_emb):
        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=-1)

        action_preds = self.action_head(state_feat)
        cost_preds = self.cost_pred_head(action_feature)
        cost_preds = F.log_softmax(cost_preds, dim=-1)
        state_preds = self.state_pred_head(action_feature)
        return action_preds, cost_preds, state_preds

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            costs_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            episode_cost: torch.Tensor = None,  # [batch_size, ]
    ):
        batch_size, seq_len = states.shape[0], states.shape[1]
        seq_list, costs_emb = self._embed(states, actions, returns_to_go, costs_to_go,
                                          time_steps)

        # [batch_size, seq_len, 2-4, emb_dim], (c_0 s_0, a_0, c_1, s_1, a_1, ...)
        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len,
                                    self.embedding_dim)

        if padding_mask is not None:
            # [batch_size, seq_len * self.seq_repeat], stack mask identically
            padding_mask = torch.stack([padding_mask] * self.seq_repeat,
                                       dim=1).permute(0, 2, 1).reshape(batch_size, -1)

        if self.cost_prefix:
            episode_cost = episode_cost.unsqueeze(-1).unsqueeze(-1)
            episode_cost = episode_cost.to(states.dtype)
            episode_cost_emb = self.prefix_emb(episode_cost)
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        # LayerNorm and Dropout as in OSRL
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        out = self.backbone(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        if self.cost_prefix:
            out = out[:, 1:]

        # [batch_size, seq_len, self.seq_repeat, embedding_dim]
        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        # [batch_size, self.seq_repeat, seq_len, embedding_dim]
        out = out.permute(0, 2, 1, 3)

        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]
        return self._heads(state_feat, action_feature, costs_emb)

    # ------------------------------------------------------- recurrent mode
    def token_order(self) -> List[str]:
        """Per-timestep token order, e.g. ['R', 'C', 's', 'a']."""
        order = (["R"] if self.use_rew else []) + (["C"] if self.use_cost else [])
        return order + ["s", "a"]

    def init_recurrent(self, batch_size: int, device=None, dtype=None) -> Any:
        if self.cost_prefix:
            raise NotImplementedError("recurrent mode does not support cost_prefix")
        if not self.backbone.supports_recurrent:
            raise NotImplementedError(f"{self.backbone_name} has no recurrent mode")
        return self.backbone.allocate_state(batch_size, device=device, dtype=dtype)

    def _recurrent_feed(self, emb: torch.Tensor, state: Any):
        x = self.emb_drop(self.emb_norm(emb))
        y, state = self.backbone.step(x, state)
        return self.out_norm(y), state

    def recurrent_observe(self, state_obs: torch.Tensor, return_to_go: torch.Tensor,
                          cost_to_go: torch.Tensor, t: torch.Tensor, rstate: Any):
        """Feed (R_t, C_t, s_t) one token at a time; return action prediction for t.

        state_obs [B, ds], return_to_go/cost_to_go [B], t [B] (long).
        """
        te = self.timestep_emb(t) if self.time_emb else 0.0
        costs_emb = None
        if self.use_rew:
            r_emb = self.return_emb(return_to_go.unsqueeze(-1)) + te
            _, rstate = self._recurrent_feed(r_emb, rstate)
        if self.use_cost:
            c = self.cost_transform(cost_to_go) if self.use_cost_transform else cost_to_go
            costs_emb = self.cost_emb(c.unsqueeze(-1)) + te
            _, rstate = self._recurrent_feed(costs_emb, rstate)
        s_emb = self.state_emb(state_obs) + te
        state_feat, rstate = self._recurrent_feed(s_emb, rstate)
        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=-1)
        return self.action_head(state_feat), rstate

    def recurrent_act(self, action: torch.Tensor, t: torch.Tensor, rstate: Any):
        """Feed the executed action token a_t (its output is not needed)."""
        te = self.timestep_emb(t) if self.time_emb else 0.0
        a_emb = self.action_emb(action) + te
        _, rstate = self._recurrent_feed(a_emb, rstate)
        return rstate


def build_policy(model_cfg, state_dim: int, action_dim: int, max_action: float,
                 seq_len: int, episode_len: int) -> SequencePolicy:
    """Build a SequencePolicy from a ``ModelConfig``."""
    return SequencePolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        seq_len=seq_len,
        episode_len=episode_len,
        embedding_dim=model_cfg.embedding_dim,
        embedding_dropout=model_cfg.embedding_dropout,
        time_emb=model_cfg.time_emb,
        use_rew=model_cfg.use_rew,
        use_cost=model_cfg.use_cost,
        cost_transform=model_cfg.cost_transform,
        add_cost_feat=model_cfg.add_cost_feat,
        mul_cost_feat=model_cfg.mul_cost_feat,
        cat_cost_feat=model_cfg.cat_cost_feat,
        action_head_layers=model_cfg.action_head_layers,
        cost_prefix=model_cfg.cost_prefix,
        stochastic=model_cfg.stochastic,
        init_temperature=model_cfg.init_temperature,
        target_entropy=-action_dim,
        backbone=model_cfg.backbone,
        backbone_kwargs=dict(model_cfg.backbone_kwargs),
    )


def load_osrl_cdt_state(policy: SequencePolicy, osrl_state: Dict[str, torch.Tensor]) -> None:
    """Load an ``osrl.algorithms.CDT`` state_dict (``blocks.*`` -> ``backbone.blocks.*``)."""
    mapped = {}
    for k, v in osrl_state.items():
        mapped["backbone." + k if k.startswith("blocks.") else k] = v
    missing, unexpected = policy.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if not m.endswith("_extra_state")]
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing} unexpected={unexpected}")
