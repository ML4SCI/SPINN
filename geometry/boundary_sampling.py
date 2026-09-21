from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .four_rod import FourRodTrap


@dataclass
class Batch:
    interior: torch.Tensor  # (Ni, 2), requires_grad
    electrode_points: torch.Tensor  # (Ne, 2)
    electrode_voltages: torch.Tensor  # (Ne, 1)
    outer_points: torch.Tensor  # (No, 2)
    outer_voltages: torch.Tensor  # (No, 1)


def _t(array: np.ndarray, requires_grad: bool = False, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(array), dtype=dtype, requires_grad=requires_grad)


def sample_batch(
    trap: FourRodTrap,
    rng: np.random.Generator,
    n_interior: int,
    n_electrode_per: int,
    n_outer: int,
    dtype=torch.float32,
) -> Batch:
    interior = trap.sample_interior(n_interior, rng)
    el_p, el_v = trap.sample_electrode_boundaries(n_electrode_per, rng)
    out_p, out_v = trap.sample_outer_boundary(n_outer, rng)
    return Batch(
        interior=_t(interior, requires_grad=True, dtype=dtype),
        electrode_points=_t(el_p, dtype=dtype),
        electrode_voltages=_t(el_v, dtype=dtype),
        outer_points=_t(out_p, dtype=dtype),
        outer_voltages=_t(out_v, dtype=dtype),
    )
