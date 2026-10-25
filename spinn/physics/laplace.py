"""Autograd differential operators for Cartesian and mapped coordinates."""

from __future__ import annotations

import torch


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Pointwise-batched gradient of ``outputs`` w.r.t. ``inputs``, graph retained."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def cartesian_laplacian(potential: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``Delta_x V`` where ``points`` (= x) is a leaf carrying ``requires_grad``."""
    grad = _grad(potential, points)
    laplacian = torch.zeros_like(potential)
    for axis in range(points.shape[1]):
        laplacian = laplacian + _grad(grad[:, axis], points)[:, axis : axis + 1]
    return laplacian


def batch_jacobian(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Per-sample Jacobian of a pointwise-batched map, shaped ``(N, out, in)``."""
    return torch.stack(
        [_grad(outputs[:, axis], inputs) for axis in range(outputs.shape[1])], dim=1
    )


def map_jacobian(shape_network, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = shape_network(z)
    return x, batch_jacobian(x, z)


def pullback_laplacian(
    potential: torch.Tensor,
    z: torch.Tensor,
    shape_network,
    *,
    determinant_floor: float = 1e-8,
    mapped_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Physical Laplacian ``Delta_x V`` evaluated on the reference domain.

    ``Delta_x V = (1/det J) div_z(det J * J^-1 J^-T grad_z V)``.
    """
    grad_z = _grad(potential, z)
    x = shape_network(z) if mapped_points is None else mapped_points
    jac = batch_jacobian(x, z)
    det = torch.linalg.det(jac)
    inv = torch.linalg.inv(jac)
    metric = det[:, None, None] * torch.bmm(inv, inv.transpose(1, 2))
    flux = torch.bmm(metric, grad_z[:, :, None]).squeeze(-1)
    divergence = torch.zeros_like(det)
    for axis in range(z.shape[1]):
        divergence = divergence + _grad(flux[:, axis], z)[:, axis]
    safe_det = torch.clamp(det, min=determinant_floor)
    return (divergence / safe_det)[:, None]
