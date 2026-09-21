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
    """C2 electrode-surface collar with exact center and enclosure anchors."""
    if band_width is None:
        band_width = 0.35 * trap.r0
    centers = _rod_centers_tensor(trap, points)  # (4, 2)
    diff = points.unsqueeze(1) - centers.unsqueeze(0)  # (N, 4, 2)
    dist2 = torch.sum(diff**2, dim=-1)  # (N, 4)

    # Smooth algebraic approximation to signed distance from each rod surface.
    # Unlike sqrt(dist2), this has finite second derivatives at rod centers.
    surface = (dist2 - trap.rod_radius_abs**2) / (2.0 * trap.rod_radius_abs)
    bumps = torch.exp(-(surface / band_width) ** 2)
    local_surface_collars = 1.0 - torch.prod(1.0 - bumps, dim=1)

    def smootherstep(value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, 0.0, 1.0)
        return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)

    radius2 = torch.sum(points**2, dim=1)
    center_fixed_radius = 0.40 * trap.r0
    center_release_radius = 0.85 * trap.r0
    center_anchor = smootherstep(
        (radius2 - center_fixed_radius**2)
        / (center_release_radius**2 - center_fixed_radius**2)
    )

    outer_fixed_width = 0.50 * trap.r0
    outer_release_radius = trap.outer_radius - outer_fixed_width
    outer_anchor = smootherstep(
        (trap.outer_radius**2 - radius2)
        / (trap.outer_radius**2 - outer_release_radius**2)
    )

    mask = local_surface_collars * center_anchor * outer_anchor
    return mask.unsqueeze(-1)  # (N, 1)


def design_mask_np(points: np.ndarray, trap: FourRodTrap, band_width: float | None = None) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(np.atleast_2d(points), dtype=torch.float64)
        return design_mask(t, trap, band_width).squeeze(-1).numpy()


def anchor_mask(points: torch.Tensor, trap: FourRodTrap, band_width: float | None = None) -> torch.Tensor:
    """Complement of the design mask: where identity should be preserved."""
    return 1.0 - design_mask(points, trap, band_width)
