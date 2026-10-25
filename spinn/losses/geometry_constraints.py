"""Geometry / fabrication constraint losses and diagnostics on the shape network.

Curvature of a deformed rod boundary uses periodic finite differences of the
mapped electrode ring with respect to its parameter angle.
"""

from __future__ import annotations

import itertools

import torch

from ..geometry.four_rod import FourRodTrap
from ..geometry.masks import anchor_mask
from ..physics.laplace import batch_jacobian, map_jacobian


def _jacobian_det(shape_network, z, mapped_points=None) -> torch.Tensor:
    if mapped_points is None:
        _, jac = map_jacobian(shape_network, z)
    else:
        jac = batch_jacobian(mapped_points, z)
    return torch.linalg.det(jac)


def jacobian_loss(
    shape_network,
    z: torch.Tensor,
    eps: float = 0.05,
    *,
    mapped_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hinge ``mean(relu(eps - det J)^2)`` against folding."""
    det = _jacobian_det(shape_network, z, mapped_points)
    return torch.mean(torch.relu(eps - det) ** 2)


def min_det_jacobian(shape_network, z: torch.Tensor) -> float:
    return float(_jacobian_det(shape_network, z).min().detach())


def displacement_loss(
    shape_network,
    z: torch.Tensor,
    *,
    mapped_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared coordinate displacement ``||NN_phi(z) - z||^2``."""
    x = shape_network(z) if mapped_points is None else mapped_points
    return torch.mean(torch.sum((x - z) ** 2, dim=1))


def anchor_loss(
    shape_network,
    trap: FourRodTrap,
    z: torch.Tensor,
    *,
    mapped_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize deviation from identity, weighted toward the anchored region."""
    x = shape_network(z) if mapped_points is None else mapped_points
    displacement_sq = torch.sum((x - z) ** 2, dim=1, keepdim=True)
    return torch.mean(anchor_mask(z, trap) * displacement_sq)


# --- deformed electrode boundaries ------------------------------------------
def _deformed_rings(shape_network, trap: FourRodTrap, n: int) -> list[torch.Tensor]:
    like = next(shape_network.parameters())
    return [
        shape_network(torch.as_tensor(e.ring(n), device=like.device, dtype=like.dtype))
        for e in trap.electrodes()
    ]


def _pairwise_gaps(rings: list[torch.Tensor]) -> list[torch.Tensor]:
    """Minimum point-to-point distance between each pair of deformed rings."""
    return [torch.cdist(a, b).min() for a, b in itertools.combinations(rings, 2)]


def _curvature(ring: torch.Tensor) -> torch.Tensor:
    """Signed curvature ``(x' y'' - y' x'') / |c'|^3`` of a closed sampled curve.

    With uniform angular step the step-size factors in ``c'`` and ``c''``
    cancel exactly, so no ``n``-scaling is needed.
    """
    forward, backward = torch.roll(ring, -1, dims=0), torch.roll(ring, 1, dims=0)
    d1 = (forward - backward) / 2.0
    d2 = forward - 2.0 * ring + backward
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    denom = torch.clamp((d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5, min=1e-9)
    return num / denom


def gap_loss(
    shape_network, trap: FourRodTrap, g_min: float, n: int = 200
) -> torch.Tensor:
    """Hinge on the minimum surface gap between each pair of deformed electrodes."""
    penalties = [
        torch.relu(g_min - gap) ** 2
        for gap in _pairwise_gaps(_deformed_rings(shape_network, trap, n))
    ]
    return sum(penalties) / len(penalties)


def curvature_loss(
    shape_network, trap: FourRodTrap, kappa_max: float, n: int = 256
) -> torch.Tensor:
    """Hinge on boundary curvature ``|kappa| - kappa_max`` of deformed rings."""
    penalties = [
        torch.mean(torch.relu(torch.abs(_curvature(ring)) - kappa_max) ** 2)
        for ring in _deformed_rings(shape_network, trap, n)
    ]
    return torch.stack(penalties).mean()


@torch.no_grad()
def min_gap(shape_network, trap: FourRodTrap, n: int = 200) -> float:
    """Smallest surface gap between any two deformed electrodes (diagnostic)."""
    return min(
        float(gap) for gap in _pairwise_gaps(_deformed_rings(shape_network, trap, n))
    )


@torch.no_grad()
def max_curvature(shape_network, trap: FourRodTrap, n: int = 256) -> float:
    """Largest absolute boundary curvature over the deformed electrodes."""
    return max(
        float(torch.abs(_curvature(ring)).max())
        for ring in _deformed_rings(shape_network, trap, n)
    )
