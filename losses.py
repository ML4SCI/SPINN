"""Laplace and boundary-condition losses for potential PINNs."""

from __future__ import annotations

import torch


def laplacian(potential: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        potential,
        points,
        grad_outputs=torch.ones_like(potential),
        create_graph=True,
    )[0]
    phi_x = gradient[:, 0:1]
    phi_y = gradient[:, 1:2]
    phi_xx = torch.autograd.grad(
        phi_x,
        points,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True,
    )[0][:, 0:1]
    phi_yy = torch.autograd.grad(
        phi_y,
        points,
        grad_outputs=torch.ones_like(phi_y),
        create_graph=True,
    )[0][:, 1:2]
    return phi_xx + phi_yy


def laplace_residual_loss(model, interior_points: torch.Tensor) -> torch.Tensor:
    potential = model(interior_points)
    return torch.mean(laplacian(potential, interior_points) ** 2)


def boundary_loss(
    model,
    world_points: torch.Tensor,
    world_values: torch.Tensor,
    electrode_points: torch.Tensor,
    electrode_values: torch.Tensor,
) -> torch.Tensor:
    world_error = model(world_points) - world_values
    electrode_error = model(electrode_points) - electrode_values
    return torch.mean(world_error**2) + torch.mean(electrode_error**2)


def total_loss(model, batch, bc_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    interior, world_points, world_values, electrode_points, electrode_values = batch
    pde = laplace_residual_loss(model, interior)
    bc = boundary_loss(model, world_points, world_values, electrode_points, electrode_values)
    total = pde + bc_weight * bc
    return total, {"pde": pde.detach(), "bc": bc.detach(), "total": total.detach()}

