"""Map-quality diagnostics for a coordinate map on the trap domain.

The instructor model produces a coordinate map, not a potential, so it cannot
emit the shape derivative ``dJ/dd``. What it can be measured on is the quality
of the map itself: how close it is to the identity, and whether its Jacobian is
well-behaved. The Jacobian matters because in the full composed model
``NN_theta(NN_phi(z))`` the shape derivative flows through ``NN_phi``'s Jacobian;
a map with a near-singular or sign-flipped Jacobian would corrupt any shape
gradient built on top of it.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from PINN.Coordinate_Projection.constraints import jacobian_constraints

from .model import CoordinateProjectionNet, TrapGeometry, TrapPointSampler


class _ParameterIgnoringMap(torch.nn.Module):
    """Adapt a single-argument coordinate map to the ``(points, parameter)`` API.

    The repo's ``map_jacobian`` and ``jacobian_constraints`` call a map as
    ``map(points, parameter)``. The instructor map takes only ``points``; this
    wrapper drops the unused parameter so the existing utilities can be reused
    without copying their logic.
    """

    def __init__(self, coordinate_map: CoordinateProjectionNet) -> None:
        super().__init__()
        self.coordinate_map = coordinate_map

    def forward(self, points: torch.Tensor, parameter=None) -> torch.Tensor:
        return self.coordinate_map(points)


def map_diagnostics(
    model: CoordinateProjectionNet,
    geometry: TrapGeometry,
    *,
    n_points: int = 4096,
    seed: int = 0,
) -> dict[str, float]:
    """Identity fidelity and Jacobian statistics on interior trap points."""

    rng = np.random.default_rng(seed)
    sampler = TrapPointSampler(geometry, rng)
    points = torch.tensor(
        sampler.interior(n_points),
        dtype=torch.float64,
        requires_grad=True,
    )

    # Evaluate on a float64 copy so the caller's model and dtype are untouched;
    # the Jacobian determinant is sensitive enough to warrant double precision.
    double_model = copy.deepcopy(model).double()
    adapted = _ParameterIgnoringMap(double_model)
    constraints = jacobian_constraints(adapted, points)
    determinant = constraints["determinant"].detach()
    condition = constraints["condition"].detach()

    with torch.no_grad():
        displacement = adapted(points) - points
        displacement_magnitude = torch.linalg.vector_norm(displacement, dim=1)

    return {
        "rms_identity_error": float(torch.sqrt(torch.mean(displacement_magnitude**2))),
        "max_displacement": float(displacement_magnitude.max()),
        "det_jacobian_min": float(determinant.min()),
        "det_jacobian_max": float(determinant.max()),
        "det_jacobian_mean": float(determinant.mean()),
        "jacobian_determinant_all_positive": bool(determinant.min() > 0),
        "condition_number_max": float(condition.max()),
        "condition_number_mean": float(condition.mean()),
    }
