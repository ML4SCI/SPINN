"""Geometry / fabrication constraint losses on the shape network ``NN_phi``.
Curvature of the deformed boundary curve is computed from the first and second
derivatives of the mapped electrode ring with respect to its parameter angle.
"""

from __future__ import annotations

import numpy as np
import torch

from ..geometry.four_rod import FourRodTrap
from ..geometry.masks import anchor_mask
from ..physics.laplace import map_jacobian


def jacobian_loss(shape_network, z: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    _, jac = map_jacobian(shape_network, z)
    det = torch.linalg.det(jac)
    return torch.mean(torch.relu(eps - det) ** 2)


def min_det_jacobian(shape_network, z: torch.Tensor) -> float:
    _, jac = map_jacobian(shape_network, z)
    return float(torch.linalg.det(jac).min().detach())


def anchor_loss(shape_network, trap: FourRodTrap, z: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from identity, weighted toward the anchored region."""
    x = shape_network(z)
    weight = anchor_mask(z, trap)  # (N,1) ~1 away from rods
    displacement_sq = torch.sum((x - z) ** 2, dim=1, keepdim=True)
    return torch.mean(weight * displacement_sq)


def _electrode_ring(trap: FourRodTrap, n: int, device, dtype) -> list[torch.Tensor]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rings = []
    for e in trap.electrodes():
        ref = np.column_stack([e.cx + e.radius * np.cos(theta), e.cy + e.radius * np.sin(theta)])
        rings.append(torch.as_tensor(ref, device=device, dtype=dtype))
    return rings


def min_gap(shape_network, trap: FourRodTrap, n: int = 200) -> float:
    """Smallest surface gap between any two deformed electrodes (diagnostic)."""
    device = next(shape_network.parameters()).device
    dtype = next(shape_network.parameters()).dtype
    rings = [shape_network(r) for r in _electrode_ring(trap, n, device, dtype)]
    best = float("inf")
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            best = min(best, float(torch.cdist(rings[i], rings[j]).min().detach()))
    return best


def max_curvature(shape_network, trap: FourRodTrap, n: int = 256) -> float:
    """Largest absolute boundary curvature over the deformed electrodes."""
    device = next(shape_network.parameters()).device
    dtype = next(shape_network.parameters()).dtype
    worst = 0.0
    for ring in _electrode_ring(trap, n, device, dtype):
        c = shape_network(ring)
        d1 = (torch.roll(c, -1, dims=0) - torch.roll(c, 1, dims=0)) / 2.0
        d2 = torch.roll(c, -1, dims=0) - 2.0 * c + torch.roll(c, 1, dims=0)
        num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        denom = torch.clamp((d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5, min=1e-9)
        worst = max(worst, float(torch.abs(num / denom).max().detach()))
    return worst


def gap_loss(shape_network, trap: FourRodTrap, g_min: float, n: int = 200) -> torch.Tensor:
    """Hinge on the minimum surface gap between adjacent deformed electrodes."""
    device = next(shape_network.parameters()).device
    dtype = next(shape_network.parameters()).dtype
    rings = [shape_network(r) for r in _electrode_ring(trap, n, device, dtype)]
    total = torch.zeros((), device=device, dtype=dtype)
    count = 0
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            # min pairwise distance between the two deformed boundary point sets
            d = torch.cdist(rings[i], rings[j]).min()
            total = total + torch.relu(g_min - d) ** 2
            count += 1
    return total / max(count, 1)


def curvature_loss(shape_network, trap: FourRodTrap, kappa_max: float, n: int = 256) -> torch.Tensor:
    """Hinge on boundary curvature ``|kappa| - kappa_max`` of deformed rings.

    For a closed curve ``c(t)`` sampled uniformly in the ring angle, the signed
    curvature is ``(x' y'' - y' x'') / (x'^2 + y'^2)^{3/2}`` with derivatives
    approximated by periodic finite differences.
    """
    device = next(shape_network.parameters()).device
    dtype = next(shape_network.parameters()).dtype
    penalties = []
    for ring in _electrode_ring(trap, n, device, dtype):
        c = shape_network(ring)  # (n, 2)
        d1 = (torch.roll(c, -1, dims=0) - torch.roll(c, 1, dims=0)) / 2.0
        d2 = torch.roll(c, -1, dims=0) - 2.0 * c + torch.roll(c, 1, dims=0)
        num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        denom = torch.clamp((d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5, min=1e-9)
        # With uniform angular step dt = 2*pi/n, the dt factors in c' and c''
        # cancel exactly in kappa = (c' x c'') / |c'|^3, so no n-scaling is needed.
        kappa = num / denom
        penalties.append(torch.mean(torch.relu(torch.abs(kappa) - kappa_max) ** 2))
    return torch.stack(penalties).mean()
