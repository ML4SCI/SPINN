"""Design and anchor masks for the residual coordinate-projection network.

Plan (sec. 3.1): ``s_design(z)`` is close to 1 near the movable electrode
boundaries and close to 0 near the outer boundary and trap center. The residual
deformation ``x = z + alpha * s_design(z) * Delta_phi(z)`` therefore localizes
movement to a band around the rods, leaving the center and the far field
anchored. The anchor mask ``1 - s_design`` identifies the points whose deviation
from identity ``L_anchor`` penalizes.

These operate on torch tensors so the masks differentiate through the map and on
numpy arrays for plotting/anchoring point selection.
"""

from __future__ import annotations

import numpy as np
import torch

from .four_rod import FourRodTrap


def _rod_centers_tensor(trap: FourRodTrap, like: torch.Tensor) -> torch.Tensor:
    centers = np.array([[e.cx, e.cy] for e in trap.electrodes()], dtype=float)
    return torch.as_tensor(centers, dtype=like.dtype, device=like.device)


def design_mask(
    points: torch.Tensor,
    trap: FourRodTrap,
    band_width: float | None = None,
) -> torch.Tensor:
    """Smooth mask ~1 in a band around each rod center, ~0 elsewhere.

    A Gaussian bump centered on each rod, summed and clamped to ``[0, 1]``. The
    band width defaults to the rod radius, so the design region covers the rod
    surface and its immediate neighborhood. The center and outer boundary fall
    off to zero because they are far from every rod center.
    """
    if band_width is None:
        band_width = trap.rod_radius_abs
    centers = _rod_centers_tensor(trap, points)  # (4, 2)
    diff = points.unsqueeze(1) - centers.unsqueeze(0)  # (N, 4, 2)
    dist2 = torch.sum(diff**2, dim=-1)  # (N, 4)
    # distance from each rod *surface*: shift so the bump peaks on the ring.
    surface = torch.sqrt(torch.clamp(dist2, min=1e-12)) - trap.rod_radius_abs
    bumps = torch.exp(-(surface**2) / (band_width**2))
    mask = torch.clamp(torch.sum(bumps, dim=1), max=1.0)
    return mask.unsqueeze(-1)  # (N, 1)


def design_mask_np(points: np.ndarray, trap: FourRodTrap, band_width: float | None = None) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(np.atleast_2d(points), dtype=torch.float64)
        return design_mask(t, trap, band_width).squeeze(-1).numpy()


def anchor_mask(points: torch.Tensor, trap: FourRodTrap, band_width: float | None = None) -> torch.Tensor:
    """Complement of the design mask: where identity should be preserved."""
    return 1.0 - design_mask(points, trap, band_width)
