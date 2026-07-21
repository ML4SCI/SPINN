"""Loss terms for the coupled shape-optimization objective.

The total objective is

    L = lambda_pde * L_PDE + lambda_pseudo * L_pseudo + lambda_bc * L_BC
        (+ optional deformation, smoothness, and Jacobian regularizers)

The PDE and pseudopotential terms are evaluated in physical coordinates via the
pullback operators, so their gradients reach both the PINN and the CPN. The
boundary term prevents the trivial ``phi = const`` solution that would otherwise
satisfy Laplace's equation and collapse the pseudopotential. See README.
"""

from __future__ import annotations

import torch

from PINN.Coordinate_Projection.constraints import jacobian_constraints
from PINN.Coordinate_Projection.operators import pullback_laplacian

from .config import PhysicsConfig
from .pseudopotential import psi_ideal, psi_model


def laplace_residual_loss(potential: torch.Tensor, reference_points: torch.Tensor, coordinate_map) -> torch.Tensor:
    """Mean squared physical Laplacian ``<(Delta_x phi)^2>``."""

    residual = pullback_laplacian(potential, reference_points, coordinate_map)
    return torch.mean(residual**2)


def pseudopotential_target_loss(
    potential: torch.Tensor,
    reference_points: torch.Tensor,
    physical_points: torch.Tensor,
    coordinate_map,
    k: float,
    physics: PhysicsConfig,
    fit_radius: float,
) -> torch.Tensor:
    """Shape mismatch between the model and ideal pseudopotential near center.

    Both pseudopotentials are normalized by their own near-center mean before
    comparison, so the loss measures the *shape* of the well (whether it is
    quadratic), not its absolute depth. This is intentional: the trap's voltage
    sets an arbitrary overall scale, and a 2 MHz-style secular target is far
    deeper in eV than this 300 V trap produces, so an absolute-eV comparison is
    dominated by that scale mismatch rather than by the geometry. See README.

    The ideal target is evaluated on detached physical points so the optimizer
    moves the model toward a stationary target rather than dragging the target.
    Only points within ``fit_radius`` of the center are scored.
    """

    model_psi = psi_model(potential, reference_points, coordinate_map, physics)
    target_psi = psi_ideal(physical_points.detach(), k, physics)

    radius = torch.linalg.vector_norm(physical_points.detach(), dim=1, keepdim=True)
    near_center = radius <= fit_radius
    if torch.sum(near_center) < 1:
        near_center = torch.ones_like(near_center)

    model_scale = torch.sum(model_psi * near_center) / torch.sum(near_center)
    target_scale = torch.sum(target_psi * near_center) / torch.sum(near_center)
    model_normalized = model_psi / torch.clamp(model_scale, min=1e-30)
    target_normalized = target_psi / torch.clamp(target_scale, min=1e-30)

    squared_error = (model_normalized - target_normalized) ** 2
    return torch.sum(squared_error * near_center) / torch.sum(near_center)


def boundary_loss(
    coupled_model,
    world_points: torch.Tensor,
    world_values: torch.Tensor,
    electrode_points: torch.Tensor,
    electrode_values: torch.Tensor,
    voltage_scale: float = 1.0,
) -> torch.Tensor:
    """Voltage mismatch on the world and electrode boundaries.

    Boundaries are prescribed on reference coordinates and pushed through the
    same composed model, so the boundary condition is enforced on the physical
    geometry the map produces. The mismatch is normalized by ``voltage_scale^2``
    so it is dimensionless and comparable to the shape-normalized pseudopotential
    term.
    """

    world_potential, _ = coupled_model(world_points)
    electrode_potential, _ = coupled_model(electrode_points)
    world_term = torch.mean((world_potential - world_values) ** 2)
    electrode_term = torch.mean((electrode_potential - electrode_values) ** 2)
    return (world_term + electrode_term) / (voltage_scale**2)


def deformation_loss(cpn, reference_points: torch.Tensor) -> torch.Tensor:
    """Penalize the size of the learned displacement (anchor toward identity)."""

    displacement = cpn(reference_points) - reference_points
    return torch.mean(torch.sum(displacement**2, dim=1))


def smoothness_loss(cpn, reference_points: torch.Tensor) -> torch.Tensor:
    """Penalize spatial variation of the displacement field."""

    points = reference_points.detach().clone().requires_grad_(True)
    displacement = cpn(points) - points
    gradients = []
    for axis in range(displacement.shape[1]):
        grad = torch.autograd.grad(
            displacement[:, axis],
            points,
            grad_outputs=torch.ones_like(displacement[:, axis]),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients.append(grad)
    jacobian = torch.stack(gradients, dim=1)
    return torch.mean(torch.sum(jacobian**2, dim=(1, 2)))


def jacobian_regularizer(coordinate_map, reference_points: torch.Tensor) -> torch.Tensor:
    """Positive-determinant and low-distortion penalty from the shared constraints."""

    constraints = jacobian_constraints(coordinate_map, reference_points)
    return constraints["determinant_loss"] + constraints["distortion_loss"]
