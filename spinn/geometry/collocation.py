"""Generate reproducible interior and boundary collocation batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .four_rod import FourRodTrap


@dataclass(frozen=True)
class CollocationBatch:
    """Tensor inputs and targets used by the PDE and boundary losses."""

    interior: torch.Tensor
    electrode_points: torch.Tensor
    electrode_voltages: torch.Tensor
    outer_points: torch.Tensor
    outer_voltages: torch.Tensor


def _as_tensor(
    array: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype).requires_grad_(
        requires_grad
    )


def sample_collocation_batch(
    trap: FourRodTrap,
    rng: np.random.Generator,
    n_interior: int,
    n_electrode_per: int,
    n_outer: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CollocationBatch:
    """Sample one deterministic batch using the supplied NumPy generator."""
    device = torch.device(device)
    interior = trap.sample_interior(n_interior, rng)
    electrode_points, electrode_voltages = trap.sample_electrode_boundaries(
        n_electrode_per, rng
    )
    outer_points, outer_voltages = trap.sample_outer_boundary(n_outer, rng)
    return CollocationBatch(
        interior=_as_tensor(interior, device=device, dtype=dtype, requires_grad=True),
        electrode_points=_as_tensor(electrode_points, device=device, dtype=dtype),
        electrode_voltages=_as_tensor(electrode_voltages, device=device, dtype=dtype),
        outer_points=_as_tensor(outer_points, device=device, dtype=dtype),
        outer_voltages=_as_tensor(outer_voltages, device=device, dtype=dtype),
    )
