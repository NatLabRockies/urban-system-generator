"""The single place torch gets imported for inference.

URBANopt calls the CLI from a Ruby subprocess; multi-threaded BLAS/OpenMP pools can hang there, so inference
defaults to one thread (override with ``USG_NUM_THREADS`` or ``--threads``). Training code passes
``threads=None`` to keep torch's own defaults.
"""

from __future__ import annotations

import os

from usg.constants import ENV_NUM_THREADS

_configured = False


def get_torch(threads: int | None = 1):
    """Import torch, configuring thread pools once. ``threads=None`` leaves torch defaults untouched."""
    global _configured
    env = os.environ.get(ENV_NUM_THREADS)
    if env:
        threads = int(env)
    if not _configured and threads is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, str(threads))
    import torch

    if not _configured:
        if threads is not None:
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:  # already set, or parallel work has started
                pass
        _configured = True
    return torch


def resolve_device(device: str = "cpu") -> str:
    torch = get_torch(threads=None) if _configured else get_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
