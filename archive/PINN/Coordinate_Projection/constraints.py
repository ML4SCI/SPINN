"""Validity and fabrication constraints for two-dimensional coordinate maps."""

from __future__ import annotations

import torch

from .operators import map_jacobian


def jacobian_constraints(
    coordinate_map,
    points: torch.Tensor,
    parameter: float | torch.Tensor | None = None,
    *,
    minimum_determinant: float = 0.25,
    maximum_condition: float = 5.0,
) -> dict[str, torch.Tensor]:
    _, jacobian = map_jacobian(coordinate_map, points, parameter)
    determinant = torch.linalg.det(jacobian)
    singular_values = torch.linalg.svdvals(jacobian)
    condition = singular_values[:, 0] / torch.clamp(singular_values[:, -1], min=1e-8)
    return {
        "determinant": determinant,
        "condition": condition,
        "determinant_loss": torch.mean(torch.relu(minimum_determinant - determinant) ** 2),
        "distortion_loss": torch.mean(torch.relu(condition - maximum_condition) ** 2),
    }


def anchor_loss(
    coordinate_map,
    boundary_points: torch.Tensor,
    parameter: float | torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.mean((coordinate_map(boundary_points, parameter) - boundary_points) ** 2)


def symmetry_loss(
    coordinate_map,
    points: torch.Tensor,
    parameter: float | torch.Tensor | None = None,
) -> torch.Tensor:
    mapped = coordinate_map(points, parameter)
    reflect_x = points * points.new_tensor([-1.0, 1.0])
    reflect_y = points * points.new_tensor([1.0, -1.0])
    swap = points.flip(dims=(1,))
    error_x = coordinate_map(reflect_x, parameter) - mapped * points.new_tensor([-1.0, 1.0])
    error_y = coordinate_map(reflect_y, parameter) - mapped * points.new_tensor([1.0, -1.0])
    error_swap = coordinate_map(swap, parameter) - mapped.flip(dims=(1,))
    return torch.mean(error_x**2) + torch.mean(error_y**2) + torch.mean(error_swap**2)


def curve_curvature(
    mapped_curve: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    """Curvature of a differentiable planar curve parameterized pointwise."""

    dx = torch.autograd.grad(
        mapped_curve[:, 0],
        parameter,
        grad_outputs=torch.ones_like(mapped_curve[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0]
    dy = torch.autograd.grad(
        mapped_curve[:, 1],
        parameter,
        grad_outputs=torch.ones_like(mapped_curve[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0]
    ddx = torch.autograd.grad(
        dx,
        parameter,
        grad_outputs=torch.ones_like(dx),
        create_graph=True,
        retain_graph=True,
    )[0]
    ddy = torch.autograd.grad(
        dy,
        parameter,
        grad_outputs=torch.ones_like(dy),
        create_graph=True,
        retain_graph=True,
    )[0]
    numerator = torch.abs(dx * ddy - dy * ddx)
    denominator = torch.clamp(dx**2 + dy**2, min=1e-12) ** 1.5
    return numerator / denominator


def curvature_bound_loss(
    curvature: torch.Tensor,
    *,
    maximum_curvature: float,
) -> torch.Tensor:
    return torch.mean(torch.relu(curvature - maximum_curvature) ** 2)
