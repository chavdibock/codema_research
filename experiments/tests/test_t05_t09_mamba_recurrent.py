"""T5 Mamba-3 CUDA vs reference, T9 recurrent mode == window mode."""

import pytest
import torch

from experiments.cdmamba.mamba3_ref import (Mamba3Ref, mamba3_ref_alloc_state,
                                            mamba3_ref_forward, mamba3_ref_step)
from experiments.tests.conftest import random_inputs, small_policy


def test_ref_step_matches_parallel_form():
    torch.manual_seed(0)
    m = Mamba3Ref(d_model=32, d_state=16, headdim=16).double()
    u = torch.randn(2, 13, 32, dtype=torch.float64)
    with torch.no_grad():
        y = mamba3_ref_forward(m, u)
        st = mamba3_ref_alloc_state(m, 2)
        outs = []
        for t in range(13):
            o, st = mamba3_ref_step(m, u[:, t], st)
            outs.append(o)
    assert torch.allclose(y, torch.stack(outs, 1).to(y.dtype), atol=1e-5)


@pytest.mark.cuda
@pytest.mark.mamba_cuda
@pytest.mark.parametrize("L", [4, 28, 40, 64, 100])
def test_t5_cuda_matches_reference(L):
    from mamba_ssm.modules.mamba3 import Mamba3

    torch.manual_seed(0)
    off = Mamba3(d_model=128, d_state=64, expand=2, headdim=32, chunk_size=64).cuda()
    ref = Mamba3Ref(d_model=128, d_state=64, expand=2, headdim=32).cuda()
    missing, unexpected = ref.load_state_dict(off.state_dict(), strict=False)
    assert not missing and not unexpected
    u = 0.5 * torch.randn(3, L, 128, device="cuda")
    pad = (64 - L % 64) % 64
    with torch.no_grad():
        y_cuda = off(torch.nn.functional.pad(u, (0, 0, 0, pad)))[:, :L].float()
        y_ref = ref(u).float()
    # NOTE: the Triton kernel computes Q/K/V in bf16 internally; see NOTES.md if this fails
    assert (y_cuda - y_ref).abs().max().item() < 1e-3


@pytest.mark.parametrize("name,kw", [("mamba3", dict(num_layers=2, d_state=16, headdim=16,
                                                     mamba_impl="ref")),
                                     ("conv", dict(num_layers=2)),
                                     ("conv", dict(num_layers=2, conv_window=1))])
def test_t9_recurrent_equals_window(name, kw):
    torch.manual_seed(0)
    pol = small_policy(name, seq_len=10, **kw).eval()
    T = 8  # window covers the full history
    s, a, r, c, t = random_inputs(1, T)
    with torch.no_grad():
        win = pol(s, a, r, c, t)[0].mean[0]  # [T, act_dim]
        st = pol.init_recurrent(1)
        rec = []
        for k in range(T):
            dist, st = pol.recurrent_observe(s[:, k], r[:, k], c[:, k], t[:, k], st)
            rec.append(dist.mean[0])
            st = pol.recurrent_act(a[:, k], t[:, k], st)
    assert torch.allclose(win, torch.stack(rec), atol=1e-5)


def test_recurrent_state_is_constant_size():
    pol = small_policy("mamba3").eval()
    st = pol.init_recurrent(1)
    n0 = pol.backbone.state_nbytes(st)
    x = random_inputs(1, 10)
    with torch.no_grad():
        for k in range(10):
            _, st = pol.recurrent_observe(x[0][:, k], x[2][:, k], x[3][:, k], x[4][:, k], st)
            st = pol.recurrent_act(x[1][:, k], x[4][:, k], st)
    assert pol.backbone.state_nbytes(st) == n0
