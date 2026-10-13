"""Shared fixtures. Tests marked ``cuda`` are skipped when no GPU is present."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

# bullet_safety_gym hides pybullet's banner with ``RedirectStream``, which looks up the C
# stream named after sys.stdout/sys.stderr and fails under pytest's output capture. In the
# test session only, import it with the real streams and make the helper a no-op.
_streams = sys.stdout, sys.stderr
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
try:
    import bullet_safety_gym.envs.builder as _bsg_builder  # noqa: E402
finally:
    sys.stdout, sys.stderr = _streams


class _NoRedirect:

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_bsg_builder.RedirectStream = _NoRedirect

TASK = "OfflineCarRun-v0"


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a CUDA GPU (skipped otherwise)")
    config.addinivalue_line("markers", "mamba_cuda: needs the official Mamba-3 CUDA kernels")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def _mamba_kernels() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from mamba_ssm.modules.mamba3 import Mamba3  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    no_gpu = pytest.mark.skip(reason="no CUDA GPU")
    no_mamba = pytest.mark.skip(reason="official Mamba-3 kernels not importable")
    has_mamba = _mamba_kernels()
    for item in items:
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(no_gpu)
        if "mamba_cuda" in item.keywords and not has_mamba:
            item.add_marker(no_mamba)


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    """DSRL-format random dataset for Car-Run + an isolated cache directory."""
    from experiments.tests.synthetic import write_synthetic_dataset

    root = tmp_path_factory.mktemp("dsrl")
    path = write_synthetic_dataset(TASK, root, n_traj=300, seed=0)
    cache = tmp_path_factory.mktemp("cache")
    os.environ["CDX_CACHE"] = str(cache)
    return {"task": TASK, "path": path, "cache": cache, "dsrl_dir": root}


@pytest.fixture(scope="session")
def data_cfg(synthetic_data):
    from experiments.common.expdefs import resolve_run_config

    return resolve_run_config("E1", "cdt", TASK, 0).data


def small_policy(backbone: str = "transformer", seq_len: int = 10, stochastic: bool = True,
                 use_cost: bool = True, **bk):
    from experiments.common.policy import SequencePolicy

    defaults = {"transformer": dict(num_layers=2, num_heads=4),
                "conv": dict(num_layers=2),
                "mamba3": dict(num_layers=2, d_state=16, headdim=16, mamba_impl="ref")}
    kw = dict(defaults[backbone])
    kw.update(bk)
    return SequencePolicy(state_dim=7, action_dim=2, max_action=1.0, seq_len=seq_len,
                          episode_len=200, embedding_dim=64, use_rew=True, use_cost=use_cost,
                          cost_transform=True, stochastic=stochastic, target_entropy=-2,
                          backbone=backbone, backbone_kwargs=kw)


def random_inputs(B: int, K: int, seed: int = 0, start: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, K, 7, generator=g), torch.rand(B, K, 2, generator=g) * 2 - 1,
            torch.randn(B, K, generator=g) * 50, torch.rand(B, K, generator=g) * 40,
            (start + torch.arange(K)).repeat(B, 1))
