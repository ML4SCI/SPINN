"""Eta objective and the aggregated total loss.

``L_eta`` drives the design toward a more ideal quadrupole. Offering two
forms: ``-eta`` for direct maximization, or ``(1 - eta)^2`` for targeting the
ideal value. We use the differentiable autograd curvature estimate of eta at the
trap center so gradients reach both the physics backend and (through the
deformation) the shape network.

The aggregate ``compute_total_loss`` returns the scalar total plus a per-term
dictionary so each term is logged separately
"""

from __future__ import annotations

import torch

from ..geometry.boundary_sampling import Batch
from ..geometry.four_rod import FourRodTrap
from ..models.composed import ComposedField
from .boundary import boundary_loss
from .geometry_constraints import (
    anchor_loss,
    curvature_loss,
    gap_loss,
    jacobian_loss,
)
from .pde import pde_loss


def differentiable_eta(model: ComposedField, r0: float, V0: float) -> torch.Tensor:
    """Shape-aware ``eta = (r0^2/V0) d2[V_theta(NN_phi(z))]/dz_x^2`` at center.

    Earlier versions evaluated ``model.backend(0)`` directly, which made
    ``L_eta`` structurally disconnected from the shape network. This composed
    version keeps the eta term on the ``z -> NN_phi(z) -> V_theta`` graph. The
    central quadrupole-shape loss remains the preferred shape objective; eta is
    intentionally kept low-weight because this pullback curvature is still a
    local diagnostic and can be noisy.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    z = torch.zeros(1, 2, device=device, dtype=dtype, requires_grad=True)
    v, _ = model(z)
    grad = torch.autograd.grad(v, z, torch.ones_like(v), create_graph=True)[0]
    d2vdx2 = torch.autograd.grad(
        grad[:, 0], z, torch.ones_like(grad[:, 0]), create_graph=True
    )[0][:, 0]
    return (r0**2 / V0) * d2vdx2.squeeze()


def eta_loss(model: ComposedField, r0: float, V0: float, mode: str = "maximize") -> torch.Tensor:
    eta = differentiable_eta(model, r0, V0)
    if mode == "maximize":
        return -eta
    if mode == "target":
        return (1.0 - eta) ** 2
    raise ValueError(f"unknown eta loss mode {mode!r}")


def central_quadrupole_loss(
    model: ComposedField,
    r0: float,
    V0: float,
    *,
    radius: float = 0.25,
    grid: int = 17,
) -> torch.Tensor:
    """Match the ideal quadrupole field over a small center disk.

    This supplements the single-point curvature objective with a local field
    shape check, which is less vulnerable to a learned derivative spike at the
    origin.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    axis = torch.linspace(-radius, radius, grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    z = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    mask = torch.sum(z**2, dim=1) <= radius**2
    z = z[mask]
    x = model.shape_network(z)
    pred = model.backend(x).squeeze(-1)
    target = (V0 / (2.0 * r0**2)) * (x[:, 0] ** 2 - x[:, 1] ** 2)
    scale = torch.mean(target.detach() ** 2).clamp_min(1.0e-12)
    return torch.mean((pred - target) ** 2) / scale


def displacement_loss(shape_network, z: torch.Tensor) -> torch.Tensor:
    """Mean squared coordinate displacement ``||NN_phi(z) - z||^2``."""
    x = shape_network(z)
    return torch.mean(torch.sum((x - z) ** 2, dim=1))


def compute_total_loss(
    model: ComposedField,
    batch: Batch,
    trap: FourRodTrap,
    config,
    *,
    include_geometry: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Aggregate physics, objective, and geometry-regularization losses."""
    w = config.loss_weights
    exp = config.experiment
    z = batch.interior

    potential, _ = model(z)
    l_pde = pde_loss(potential, z, model.shape_network)
    l_bc = boundary_loss(model, batch)
    l_eta = eta_loss(model, exp.r0, exp.V0)
    l_quad = central_quadrupole_loss(
        model,
        exp.r0,
        exp.V0,
        radius=config.sampling.quadrupole_loss_radius,
        grid=config.sampling.quadrupole_loss_grid,
    )

    total = w.pde * l_pde + w.bc * l_bc + w.eta * l_eta + w.quadrupole * l_quad
    terms = {
        "pde": float(l_pde.detach()),
        "bc": float(l_bc.detach()),
        "eta_obj": float(l_eta.detach()),
        "quadrupole": float(l_quad.detach()),
    }

    if include_geometry:
        eps = config.shape_network.jacobian_eps
        # plan: g_min = 0.05-0.1 r0; kappa_max generous in normalized units
        g_min = 0.05 * exp.r0
        kappa_max = 1.0 / (0.5 * trap.rod_radius_abs)  # ~2x nominal rod curvature
        l_jac = jacobian_loss(model.shape_network, z, eps=eps)
        l_disp = displacement_loss(model.shape_network, z)
        l_gap = gap_loss(model.shape_network, trap, g_min)
        l_curv = curvature_loss(model.shape_network, trap, kappa_max)
        l_anchor = anchor_loss(model.shape_network, trap, z)
        total = (
            total
            + w.jacobian * l_jac
            + w.displacement * l_disp
            + w.gap * l_gap
            + w.curvature * l_curv
            + w.anchor * l_anchor
        )
        terms.update(
            {
                "jacobian": float(l_jac.detach()),
                "displacement": float(l_disp.detach()),
                "gap": float(l_gap.detach()),
                "curvature": float(l_curv.detach()),
                "anchor": float(l_anchor.detach()),
            }
        )

    terms["total"] = float(total.detach())
    return total, terms
