"""Pure-PyTorch reference implementation of the Mamba-3 (SISO) mixer.

Written against ``mamba_ssm==2.3.2.post1`` (``mamba_ssm/modules/mamba3.py`` and the
Triton kernel ``ops/triton/mamba3/mamba3_siso_fwd.py``). The official package has no
pure-PyTorch path for the full Mamba-3 recurrence (only kernel-level references), so we
write our own (spec §4.4). ``mambapy`` is not used: it implements Mamba-1.

``Mamba3Ref`` has exactly the parameter names/shapes of ``mamba_ssm.modules.mamba3.Mamba3``
(SISO, ngroups=1, is_outproj_norm=False), so state_dicts are interchangeable (test T5).
The functional forms below also work directly on an official ``Mamba3`` module.

Recurrence per head h (trapezoidal discretization, data-dependent RoPE on B/C):
    theta_t = theta_{t-1} + tanh(angle_t) * pi * dt_t
    q_t = rot(C_t + C_bias, theta_t),  k_t = rot(B_t + B_bias, theta_t)
    alpha_t = exp(A_t dt_t),  gamma_t = sigmoid(trap_t) dt_t,
    beta_t = (1 - sigmoid(trap_t)) dt_t alpha_t
    S_t = alpha_t S_{t-1} + gamma_t x_t k_t^T + beta_t x_{t-1} k_{t-1}^T
    y_t = (S_t q_t + D x_t) * silu(z_t)
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormRef(nn.Module):
    """Same parameters as ``mamba_ssm.ops.triton.layernorm_gated.RMSNorm`` (no gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rmsnorm(x, self.weight, self.eps)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


class Mamba3Ref(nn.Module):
    """Mamba-3 SISO mixer with the official parameterization and initialization."""

    def __init__(self,
                 d_model: int,
                 d_state: int = 128,
                 expand: int = 2,
                 headdim: int = 64,
                 ngroups: int = 1,
                 rope_fraction: float = 0.5,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4,
                 A_floor: float = 1e-4,
                 is_outproj_norm: bool = False,
                 is_mimo: bool = False,
                 mimo_rank: int = 4,
                 chunk_size: int = 64,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()
        if is_mimo or is_outproj_norm or ngroups != 1:
            raise NotImplementedError(
                "reference path covers SISO, ngroups=1, is_outproj_norm=False only")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.A_floor = A_floor
        self.is_outproj_norm = False
        self.is_mimo = False
        self.mimo_rank = 1
        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        d_in_proj = (2 * self.d_inner + 2 * d_state * self.num_bc_heads * self.mimo_rank +
                     3 * self.nheads + self.num_rope_angles)
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32) *
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, d_state), dtype=torch.float32,
                            device=device))
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, d_state), dtype=torch.float32,
                            device=device))
        self.B_norm = RMSNormRef(d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormRef(d_state, eps=1e-5, **factory_kwargs)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    def forward(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return mamba3_ref_forward(self, u)


# ----------------------------------------------------------------- functional
def _split(m, zxbc: torch.Tensor):
    return torch.split(zxbc, [
        m.d_inner, m.d_inner, m.d_state * m.num_bc_heads * m.mimo_rank,
        m.d_state * m.num_bc_heads * m.mimo_rank, m.nheads, m.nheads, m.nheads,
        m.num_rope_angles
    ],
                       dim=-1)


def _norm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """RMSNorm with the module's weight (works for RMSNormRef and the Triton RMSNorm)."""
    return _rmsnorm(x, module.weight, getattr(module, "eps", 1e-5))


def _rotate_pairwise(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary on pairs (2i, 2i+1), i < theta.shape[-1]; rest unrotated."""
    s = theta.shape[-1]
    vr = v[..., :2 * s].reshape(*v.shape[:-1], s, 2)
    v0, v1 = vr[..., 0], vr[..., 1]
    c, sn = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([v0 * c - v1 * sn, v0 * sn + v1 * c], dim=-1).reshape(
        *v.shape[:-1], 2 * s)
    return torch.cat([rot, v[..., 2 * s:]], dim=-1)


def _project(m, u: torch.Tensor):
    """Shared preprocessing; u [..., d_model]. Returns fp32 tensors with a head axis."""
    z, x, Bm, Cm, dd_dt, dd_A, trap, angles = _split(m, m.in_proj(u))
    H, P = m.nheads, m.headdim
    z = z.reshape(*z.shape[:-1], H, P).float()
    x = x.reshape(*x.shape[:-1], H, P).float()
    Bm = _norm(m.B_norm, Bm).float()  # [..., N] (ngroups=1, rank 1)
    Cm = _norm(m.C_norm, Cm).float()
    A = torch.clamp(-F.softplus(dd_A.float()), max=-m.A_floor)  # [..., H]
    DT = F.softplus(dd_dt.float() + m.dt_bias.float())  # [..., H]
    trap = torch.sigmoid(trap.float())  # [..., H]
    ang = torch.tanh(angles.float())  # [..., S]
    q = Cm.unsqueeze(-2) + m.C_bias[:, 0, :].float()  # [..., H, N]
    k = Bm.unsqueeze(-2) + m.B_bias[:, 0, :].float()
    return z, x, q, k, A, DT, trap, ang


def mamba3_ref_forward(m, u: torch.Tensor) -> torch.Tensor:
    """Parallel (quadratic, exact) form of the recurrence. u [B, L, d_model]."""
    z, x, q, k, A, DT, trap, ang = _project(m, u)
    # rotary angles: cumulative sum over time of tanh(angle) * pi * dt (per head)
    theta = torch.cumsum(ang.unsqueeze(-2) * math.pi * DT.unsqueeze(-1), dim=1)  # [B,L,H,S]
    qk_diag = (q * k).sum(-1)  # [B, L, H]
    qr = _rotate_pairwise(q, theta)
    kr = _rotate_pairwise(k, theta)

    gamma = DT * trap  # [B, L, H]
    shifted = torch.zeros_like(gamma)
    shifted[:, :-1] = DT[:, 1:] * (1.0 - trap[:, 1:])
    scale = gamma + shifted
    cum = torch.cumsum(A * DT, dim=1)  # [B, L, H]

    L = u.shape[1]
    scores = torch.einsum("blhn,bshn->bhls", qr, kr)  # [B, H, L, L]
    cum_h = cum.transpose(1, 2)  # [B, H, L]
    seg = cum_h.unsqueeze(-1) - cum_h.unsqueeze(-2)  # [B, H, t, s]
    strict = torch.ones(L, L, dtype=torch.bool, device=u.device).tril(-1)
    decay = torch.exp(seg.masked_fill(~strict, float("-inf")))
    w = scores * decay * scale.transpose(1, 2).unsqueeze(-2)
    y = torch.einsum("bhls,bshp->blhp", w, x)
    y = y + (m.D.float().view(1, 1, -1, 1) + (gamma * qk_diag).unsqueeze(-1)) * x
    y = y * F.silu(z)
    y = y.reshape(*y.shape[:2], m.nheads * m.headdim)
    return m.out_proj(y.to(u.dtype))


def mamba3_ref_alloc_state(m, batch_size: int, device=None) -> Dict[str, torch.Tensor]:
    dev = device if device is not None else m.in_proj.weight.device
    f32 = torch.float32
    return {
        "angle": torch.zeros(batch_size, m.nheads, m.num_rope_angles, device=dev, dtype=f32),
        "ssm": torch.zeros(batch_size, m.nheads, m.headdim, m.d_state, device=dev, dtype=f32),
        "k": torch.zeros(batch_size, m.nheads, m.d_state, device=dev, dtype=f32),
        "v": torch.zeros(batch_size, m.nheads, m.headdim, device=dev, dtype=f32),
    }


def mamba3_ref_step(m, u_t: torch.Tensor,
                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """One recurrent step. u_t [B, d_model]; returns ([B, d_model], new state)."""
    z, x, q, k, A, DT, trap, ang = _project(m, u_t)  # x [B,H,P], q/k [B,H,N], A [B,H]
    theta = state["angle"] + ang.unsqueeze(-2) * math.pi * DT.unsqueeze(-1)  # [B,H,S]
    qr = _rotate_pairwise(q, theta)
    kr = _rotate_pairwise(k, theta)
    alpha = torch.exp(A * DT)
    beta = (1.0 - trap) * DT * alpha
    gamma = trap * DT
    ssm = (state["ssm"] * alpha[..., None, None] +
           gamma[..., None, None] * x.unsqueeze(-1) * kr.unsqueeze(-2) +
           beta[..., None, None] * state["v"].unsqueeze(-1) * state["k"].unsqueeze(-2))
    y = torch.einsum("bhpn,bhn->bhp", ssm, qr) + m.D.float().view(1, -1, 1) * x
    y = y * F.silu(z)
    out = m.out_proj(y.reshape(y.shape[0], -1).to(u_t.dtype))
    return out, {"angle": theta, "ssm": ssm, "k": kr, "v": x}
