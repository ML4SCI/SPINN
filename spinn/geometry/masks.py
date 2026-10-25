"""Smooth deformation and anchor masks for coordinate projection."""

from __future__ import annotations

import torch

from .four_rod import FourRodTrap


def _smootherstep(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def design_mask(
    points: torch.Tensor,
    trap: FourRodTrap,
    band_width: float | None = None,
) -> torch.Tensor:
    """C2 electrode-surface collar with exact center and enclosure anchors, ``(N, 1)``."""
    if band_width is None:
        band_width = 0.35 * trap.r0
    centers = points.new_tensor([[e.cx, e.cy] for e in trap.electrodes()])
    dist2 = torch.sum((points.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=-1)

    # Smooth algebraic approximation to signed distance from each rod surface.
    # Unlike sqrt(dist2), this has finite second derivatives at rod centers.
    surface = (dist2 - trap.rod_radius_abs**2) / (2.0 * trap.rod_radius_abs)
    bumps = torch.exp(-((surface / band_width) ** 2))
    surface_collars = 1.0 - torch.prod(1.0 - bumps, dim=1)

    radius2 = torch.sum(points**2, dim=1)
    center_fixed = 0.40 * trap.r0
    center_release = 0.85 * trap.r0
    center_anchor = _smootherstep(
        (radius2 - center_fixed**2) / (center_release**2 - center_fixed**2)
    )

    outer_release = trap.outer_radius - 0.50 * trap.r0
    outer_anchor = _smootherstep(
        (trap.outer_radius**2 - radius2) / (trap.outer_radius**2 - outer_release**2)
    )

    return (surface_collars * center_anchor * outer_anchor).unsqueeze(-1)


def anchor_mask(
    points: torch.Tensor, trap: FourRodTrap, band_width: float | None = None
) -> torch.Tensor:
    """Complement of the design mask: where identity should be preserved."""
    return 1.0 - design_mask(points, trap, band_width)
