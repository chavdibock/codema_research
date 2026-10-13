"""T1 fidelity, T2 causality, T3 padding, T4 action alignment."""

import pytest
import torch
from osrl.algorithms import CDT

from experiments.common.backbones import build_backbone
from experiments.common.policy import SequencePolicy, load_osrl_cdt_state
from experiments.common.utils import seed_all
from experiments.tests.conftest import random_inputs, small_policy

BACKBONES_CPU = [("transformer", dict(num_layers=2, num_heads=4)), ("conv", dict(num_layers=2)),
                 ("conv", dict(num_layers=2, conv_window=3)),
                 ("mamba3", dict(num_layers=2, d_state=16, headdim=16, mamba_impl="ref"))]


# ------------------------------------------------------------------------- T1
@pytest.mark.parametrize("stochastic,use_cost,heads", [(True, True, 1), (False, True, 1),
                                                       (True, False, 2)])
def test_t1_equals_osrl_cdt(stochastic, use_cost, heads):
    kw = dict(state_dim=7, action_dim=2, max_action=1.0, seq_len=10, episode_len=200,
              embedding_dim=128, embedding_dropout=0.1, time_emb=True, use_rew=True,
              use_cost=use_cost, cost_transform=True, stochastic=stochastic,
              init_temperature=0.1, target_entropy=-2, action_head_layers=heads)
    tb = dict(num_layers=3, num_heads=8, attention_dropout=0.1, residual_dropout=0.1)
    seed_all(0)
    ref = CDT(**kw, **tb).eval()
    seed_all(0)
    ours = SequencePolicy(**kw, backbone="transformer", backbone_kwargs=tb).eval()
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in ours.parameters())
    x = random_inputs(4, 10, start=37)
    pad = torch.zeros(4, 10, dtype=torch.bool)
    pad[1, 6:] = True
    with torch.no_grad():
        for mask in (None, pad):
            a_ref, c_ref, s_ref = ref(*x, padding_mask=mask)
            a, c, s = ours(*x, padding_mask=mask)
            if stochastic:
                a_ref, a = a_ref.mean, a.mean
            assert torch.equal(a_ref, a) and torch.equal(c_ref, c) and torch.equal(s_ref, s)
    # loading an OSRL state_dict gives the same outputs too
    seed_all(1)
    other = SequencePolicy(**kw, backbone="transformer", backbone_kwargs=tb).eval()
    load_osrl_cdt_state(other, ref.state_dict())
    with torch.no_grad():
        a2 = other(*x)[0]
        a1 = ref(*x)[0]
    assert torch.equal(a1.mean if stochastic else a1, a2.mean if stochastic else a2)


def test_t1_mamba_init_not_overwritten():
    """CDT's N(0, .02) init must not touch the Mamba-3 parameters (spec §4.1)."""
    pol = small_policy("mamba3")
    mixer = pol.backbone.blocks[0].mixer
    assert torch.all(mixer.D == 1.0)
    assert torch.all(mixer.B_bias == 1.0) and torch.all(mixer.C_bias == 1.0)
    dt = torch.nn.functional.softplus(mixer.dt_bias)
    assert dt.min() >= 1e-4 - 1e-7 and dt.max() <= 0.1 + 1e-6
    # in_proj keeps nn.Linear's default init, not N(0, 0.02)
    assert mixer.in_proj.weight.std() > 0.05


# ------------------------------------------------------------------------- T2
@pytest.mark.parametrize("name,kw", BACKBONES_CPU)
def test_t2_causality(name, kw):
    torch.manual_seed(0)
    bb = build_backbone(name, d_model=32, max_tokens=40, num_modalities=4, **kw).eval()
    x = torch.randn(3, 40, 32)
    with torch.no_grad():
        y = bb(x)
        for i in (0, 5, 17, 38):
            x2 = x.clone()
            x2[:, i + 1:] += torch.randn_like(x2[:, i + 1:]) * 3
            y2 = bb(x2)
            assert torch.allclose(y[:, :i + 1], y2[:, :i + 1], atol=1e-6), (name, i)
            assert not torch.allclose(y[:, i + 1:], y2[:, i + 1:], atol=1e-6)


@pytest.mark.cuda
@pytest.mark.mamba_cuda
def test_t2_causality_mamba_cuda():
    torch.manual_seed(0)
    bb = build_backbone("mamba3", d_model=128, max_tokens=40, num_modalities=4, num_layers=2,
                        d_state=64, headdim=32, mamba_impl="cuda").cuda().eval()
    x = torch.randn(3, 40, 128, device="cuda")
    with torch.no_grad():
        y = bb(x)
        for i in (0, 17, 38):
            x2 = x.clone()
            x2[:, i + 1:] += 3 * torch.randn_like(x2[:, i + 1:])
            assert torch.allclose(y[:, :i + 1], bb(x2)[:, :i + 1], atol=1e-6)


# ------------------------------------------------------------------------- T3
def _check_padding(pol, device="cpu"):
    from torch.distributions import Normal

    pol = pol.to(device).eval()
    K = pol.seq_len
    for L in range(1, K + 1):  # 4..4K tokens
        x = [t.to(device) for t in random_inputs(2, K, seed=L)]
        mask = torch.zeros(2, K, device=device)
        mask[:, :L] = 1
        xp = [t.clone() for t in x]
        for t in xp[:4]:
            t[:, L:] = 0  # right padding as in OSRL
        with torch.no_grad():
            a_pad, c_pad, _ = pol(*xp, padding_mask=~mask.bool())
            a_un, c_un, _ = pol(*[t[:, :L] for t in x])
        assert torch.isfinite(a_pad.mean).all() and torch.isfinite(c_pad).all()
        assert torch.allclose(a_pad.mean[:, :L], a_un.mean, atol=1e-5)
        assert torch.allclose(c_pad[:, :L], c_un, atol=1e-5)
        # masked loss == loss on the unpadded windows
        acts = x[1]
        nll_pad = -Normal(a_pad.loc, a_pad.scale).log_prob(acts)[mask > 0].mean()
        nll_un = -Normal(a_un.loc, a_un.scale).log_prob(acts[:, :L]).mean()
        assert torch.allclose(nll_pad, nll_un, atol=1e-5)


@pytest.mark.parametrize("name,kw", BACKBONES_CPU)
def test_t3_right_padding(name, kw):
    torch.manual_seed(0)
    _check_padding(small_policy(name, **kw))


@pytest.mark.cuda
@pytest.mark.mamba_cuda
def test_t3_right_padding_mamba_cuda_chunking():
    torch.manual_seed(0)
    pol = small_policy("mamba3", d_state=64, headdim=32, mamba_impl="cuda")
    _check_padding(pol, "cuda")


# ------------------------------------------------------------------------- T4
@pytest.mark.parametrize("name,kw", BACKBONES_CPU)
def test_t4_action_alignment(name, kw):
    torch.manual_seed(0)
    pol = small_policy(name, **kw).eval()
    x = random_inputs(2, 10)
    with torch.no_grad():
        base = pol(*x)[0].mean
        for t in (0, 4, 9):
            xa = [v.clone() for v in x]
            xa[1][:, t:] += 1.0  # a_t, a_{t+1}, ...
            assert torch.allclose(pol(*xa)[0].mean[:, :t + 1], base[:, :t + 1], atol=1e-6)
            xs = [v.clone() for v in x]
            xs[0][:, t] += 1.0  # s_t must matter for the action at t
            assert not torch.allclose(pol(*xs)[0].mean[:, t], base[:, t], atol=1e-6)
