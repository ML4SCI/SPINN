from __future__ import annotations

import torch


def cartesian_laplacian(potential: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``Delta_x V`` where ``points`` (= x) is a leaf carrying ``requires_grad``."""
    grad = torch.autograd.grad(
        potential, points, grad_outputs=torch.ones_like(potential),
        create_graph=True, retain_graph=True,
    )[0]
    laplacian = torch.zeros_like(potential)
    for axis in range(points.shape[1]):
        second = torch.autograd.grad(
            grad[:, axis], points, grad_outputs=torch.ones_like(grad[:, axis]),
            create_graph=True, retain_graph=True,
        )[0][:, axis : axis + 1]
        laplacian = laplacian + second
    return laplacian


def batch_jacobian(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Per-sample Jacobian of a pointwise-batched map, shaped ``(N, out, in)``."""
    columns = []
    for axis in range(outputs.shape[1]):
        derivative = torch.autograd.grad(
            outputs[:, axis], inputs, grad_outputs=torch.ones_like(outputs[:, axis]),
            create_graph=True, retain_graph=True,
        )[0]
        columns.append(derivative)
    return torch.stack(columns, dim=1)


def map_jacobian(shape_network, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = shape_network(z)
    jac = batch_jacobian(x, z)
    return x, jac


def physical_gradient(
    potential: torch.Tensor,
    z: torch.Tensor,
    shape_network,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """``grad_x V = J^{-T} grad_z u``."""
    grad_z = torch.autograd.grad(
        potential, z, grad_outputs=torch.ones_like(potential),
        create_graph=create_graph, retain_graph=True,
    )[0]
    _, jac = map_jacobian(shape_network, z)
    inv_t = torch.linalg.inv(jac).transpose(1, 2)
    return torch.bmm(inv_t, grad_z[:, :, None]).squeeze(-1)


def physical_hessian(potential: torch.Tensor, z: torch.Tensor, shape_network) -> torch.Tensor:
    """Hessian of ``V`` with respect to physical coordinates ``x``."""
    grad_x = physical_gradient(potential, z, shape_network, create_graph=True)
    d_grad_dz = batch_jacobian(grad_x, z)
    _, jac = map_jacobian(shape_network, z)
    inv = torch.linalg.inv(jac)
    hessian = torch.bmm(d_grad_dz, inv)
    return 0.5 * (hessian + hessian.transpose(1, 2))


def pullback_laplacian(
    potential: torch.Tensor,
    z: torch.Tensor,
    shape_network,
    *,
    determinant_floor: float = 1e-8,
) -> torch.Tensor:
    """Physical Laplacian ``Delta_x V`` evaluated on the reference domain."""
    grad_z = torch.autograd.grad(
        potential, z, grad_outputs=torch.ones_like(potential),
        create_graph=True, retain_graph=True,
    )[0]
    _, jac = map_jacobian(shape_network, z)
    det = torch.linalg.det(jac)
    inv = torch.linalg.inv(jac)
    metric = det[:, None, None] * torch.bmm(inv, inv.transpose(1, 2))
    flux = torch.bmm(metric, grad_z[:, :, None]).squeeze(-1)
    divergence = torch.zeros_like(det)
    for axis in range(z.shape[1]):
        d = torch.autograd.grad(
            flux[:, axis], z, grad_outputs=torch.ones_like(flux[:, axis]),
            create_graph=True, retain_graph=True,
        )[0][:, axis]
        divergence = divergence + d
    safe_det = torch.clamp(det, min=determinant_floor)
    return (divergence / safe_det)[:, None]
