"""Coordinate-projection PINNs for differentiable Paul-trap geometry."""

from .maps import (
    AffineMap,
    ElectrodeTranslationMap,
    IdentityMap,
    RadialDeformationMap,
    ResidualCoordinateMap,
)
from .operators import (
    cartesian_laplacian,
    physical_gradient,
    physical_hessian,
    pullback_laplacian,
)
from .model import CoordinateProjectionPINN, PhysicalPotentialMLP

__all__ = [
    "AffineMap",
    "ElectrodeTranslationMap",
    "IdentityMap",
    "RadialDeformationMap",
    "ResidualCoordinateMap",
    "CoordinateProjectionPINN",
    "PhysicalPotentialMLP",
    "cartesian_laplacian",
    "physical_gradient",
    "physical_hessian",
    "pullback_laplacian",
]
