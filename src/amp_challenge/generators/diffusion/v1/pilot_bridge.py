"""The single explicit float64-to-float32 C0 bridge used by each pilot fit."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from amp_challenge.generators.diffusion.v1.pilot_data import AuthenticatedCountPrior
from amp_challenge.generators.diffusion.v1.pilot_model import (
    assert_r128_deterministic_runtime,
)


def count_prior_training_bridge(
    artifact: AuthenticatedCountPrior,
    *,
    device: str | torch.device,
) -> Tensor:
    """Reauthenticate C0, then apply the one frozen float32 conversion."""

    if type(artifact) is not AuthenticatedCountPrior:
        raise TypeError("artifact must be an AuthenticatedCountPrior")
    prior = artifact.revalidate()
    assert_r128_deterministic_runtime()
    source = prior.log_relative_position_probability
    if source.dtype != np.dtype("<f8") or source.shape != (5, 10, 20):
        raise TypeError("C0 bridge source must be exact little-endian float64 [5,10,20]")
    if not source.flags.c_contiguous or not bool(np.isfinite(source).all()):
        raise ValueError("C0 bridge source must be finite and C contiguous")
    source_tensor = torch.from_numpy(np.ascontiguousarray(source).copy())
    if source_tensor.dtype != torch.float64:
        raise RuntimeError("NumPy-to-Torch C0 bridge did not preserve float64")
    result = source_tensor.to(
        device=torch.device(device),
        dtype=torch.float32,
        non_blocking=False,
        copy=True,
        memory_format=torch.contiguous_format,
    )
    if result.dtype != torch.float32 or result.shape != (5, 10, 20):
        raise RuntimeError("C0 training bridge did not produce exact float32 [5,10,20]")
    if not result.is_contiguous() or not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("C0 training bridge produced invalid values")
    return result
