"""Differential operators under a learned or parameterized coordinate map."""

from __future__ import annotations

import torch


def cartesian_laplacian(
    potential: torch.Tensor,
    physical_points: torch.Tensor,
) -> torch.Tensor:
    """Laplacian of ``NN_theta(x)`` with respect to projected coordinates."""

    gradient = torch.autograd.grad(
        potential,
        physical_points,
        grad_outputs=torch.ones_like(potential),
        create_graph=True,
        retain_graph=True,
    )[0]
    laplacian = torch.zeros_like(potential)
    for axis in range(physical_points.shape[1]):
        second = torch.autograd.grad(
            gradient[:, axis],
            physical_points,
            grad_outputs=torch.ones_like(gradient[:, axis]),
            create_graph=True,
            retain_graph=True,
        )[0][:, axis:axis + 1]
        laplacian = laplacian + second
    return laplacian


def batch_jacobian(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Return per-sample Jacobians for pointwise batched functions."""

    columns = []
    for output_axis in range(outputs.shape[1]):
        derivative = torch.autograd.grad(
            outputs[:, output_axis],
            inputs,
            grad_outputs=torch.ones_like(outputs[:, output_axis]),
            create_graph=True,
            retain_graph=True,
        )[0]
        columns.append(derivative)
    return torch.stack(columns, dim=1)


def map_jacobian(
    coordinate_map,
    reference_points: torch.Tensor,
    parameter: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_points = coordinate_map(reference_points, parameter)
    jacobian = batch_jacobian(physical_points, reference_points)
    return physical_points, jacobian


def physical_gradient(
    reference_potential: torch.Tensor,
    reference_points: torch.Tensor,
    coordinate_map,
    parameter: float | torch.Tensor | None = None,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Transform ``grad_z u`` into ``grad_x phi = J^{-T} grad_z u``."""

    grad_reference = torch.autograd.grad(
        reference_potential,
        reference_points,
        grad_outputs=torch.ones_like(reference_potential),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    _, jacobian = map_jacobian(coordinate_map, reference_points, parameter)
    inverse_transpose = torch.linalg.inv(jacobian).transpose(1, 2)
    return torch.bmm(inverse_transpose, grad_reference[:, :, None]).squeeze(-1)


def physical_hessian(
    reference_potential: torch.Tensor,
    reference_points: torch.Tensor,
    coordinate_map,
    parameter: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the Hessian with respect to physical coordinates."""

    gradient_physical = physical_gradient(
        reference_potential,
        reference_points,
        coordinate_map,
        parameter,
        create_graph=True,
    )
    derivative_wrt_reference = batch_jacobian(gradient_physical, reference_points)
    _, jacobian = map_jacobian(coordinate_map, reference_points, parameter)
    inverse = torch.linalg.inv(jacobian)
    hessian = torch.bmm(derivative_wrt_reference, inverse)
    return 0.5 * (hessian + hessian.transpose(1, 2))


def pullback_laplacian(
    reference_potential: torch.Tensor,
    reference_points: torch.Tensor,
    coordinate_map,
    parameter: float | torch.Tensor | None = None,
    *,
    determinant_floor: float = 1e-8,
) -> torch.Tensor:
    r"""Evaluate the physical Laplacian on the reference domain.

    For ``x = F(z)`` and ``u(z) = phi(F(z))``:

    ``Delta_x phi = 1/det(J) div_z(det(J) J^-1 J^-T grad_z u)``.
    """

    grad_reference = torch.autograd.grad(
        reference_potential,
        reference_points,
        grad_outputs=torch.ones_like(reference_potential),
        create_graph=True,
        retain_graph=True,
    )[0]
    _, jacobian = map_jacobian(coordinate_map, reference_points, parameter)
    determinant = torch.linalg.det(jacobian)
    inverse = torch.linalg.inv(jacobian)
    metric = determinant[:, None, None] * torch.bmm(
        inverse,
        inverse.transpose(1, 2),
    )
    flux = torch.bmm(metric, grad_reference[:, :, None]).squeeze(-1)

    divergence = torch.zeros_like(determinant)
    for axis in range(reference_points.shape[1]):
        derivative = torch.autograd.grad(
            flux[:, axis],
            reference_points,
            grad_outputs=torch.ones_like(flux[:, axis]),
            create_graph=True,
            retain_graph=True,
        )[0][:, axis]
        divergence = divergence + derivative
    safe_determinant = torch.clamp(determinant, min=determinant_floor)
    return (divergence / safe_determinant)[:, None]
