"""Eta objective, center-quadrupole objective, and the aggregated total loss.

``L_eta = -eta`` uses the differentiable autograd curvature of the composed
field at the trap center, so gradients reach both the physics backend and
(through the deformation) the shape network. ``compute_total_loss`` returns the
scalar total plus a per-term dictionary so each term is logged separately.
"""

from __future__ import annotations

import torch

from ..geometry.collocation import CollocationBatch
from ..geometry.four_rod import FourRodTrap
from ..models.composed import ComposedField
from ..physics.eta import center_d2v_dx2
from .boundary import boundary_loss
from .geometry_constraints import (
    anchor_loss,
    curvature_loss,
    displacement_loss,
    gap_loss,
    jacobian_loss,
)
from .pde import pde_loss


def differentiable_eta(model: ComposedField, r0: float, V0: float) -> torch.Tensor:
    """Shape-aware ``eta = (r0^2/V0) d2[V_theta(NN_phi(z))]/dz_x^2`` at center.

    Kept on the ``z -> NN_phi(z) -> V_theta`` graph so ``L_eta`` reaches the
    shape network. Intentionally low-weight: this pullback curvature is a local
    diagnostic and can be noisy; the central quadrupole loss is the preferred
    shape objective.
    """
    like = next(model.parameters())
    d2 = center_d2v_dx2(
        lambda z: model(z)[0], device=like.device, dtype=like.dtype, create_graph=True
    )
    return (r0**2 / V0) * d2.squeeze()


def central_quadrupole_loss(
    model: ComposedField,
    r0: float,
    V0: float,
    *,
    radius: float = 0.25,
    grid: int = 17,
) -> torch.Tensor:
    """Normalized misfit to the ideal quadrupole over a small center disk.

    Supplements the single-point curvature objective with a local field-shape
    check that is less vulnerable to a learned derivative spike at the origin.
    """
    like = next(model.parameters())
    axis = torch.linspace(-radius, radius, grid, device=like.device, dtype=like.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    z = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    z = z[torch.sum(z**2, dim=1) <= radius**2]
    pred, x = model(z)
    target = (V0 / (2.0 * r0**2)) * (x[:, 0] ** 2 - x[:, 1] ** 2)
    scale = torch.mean(target.detach() ** 2).clamp_min(1.0e-12)
    return torch.mean((pred.squeeze(-1) - target) ** 2) / scale


def compute_total_loss(
    model: ComposedField,
    batch: CollocationBatch,
    trap: FourRodTrap,
    config,
    *,
    include_geometry: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of physics, objective, and geometry-regularization losses.

    PDE and BC terms are always evaluated (and logged); every other term is
    evaluated only when its weight is nonzero.
    """
    w = config.loss_weights
    r0, V0 = config.experiment.r0, config.experiment.V0
    shape = model.shape_network
    z = batch.interior

    potential, x = model(z)
    l_pde = pde_loss(potential, z, shape, mapped_points=x)
    l_bc = boundary_loss(model, batch)
    total = w.pde * l_pde + w.bc * l_bc
    terms = {"pde": l_pde, "bc": l_bc}

    optional = {
        "eta_obj": (w.eta, lambda: -differentiable_eta(model, r0, V0)),
        "quadrupole": (
            w.quadrupole,
            lambda: central_quadrupole_loss(
                model,
                r0,
                V0,
                radius=config.sampling.quadrupole_loss_radius,
                grid=config.sampling.quadrupole_loss_grid,
            ),
        ),
    }
    if include_geometry:
        # plan: g_min = 0.05-0.1 r0; kappa_max ~2x nominal rod curvature
        g_min = 0.05 * r0
        kappa_max = 1.0 / (0.5 * trap.rod_radius_abs)
        eps = config.shape_network.jacobian_eps
        optional |= {
            "jacobian": (
                w.jacobian,
                lambda: jacobian_loss(shape, z, eps, mapped_points=x),
            ),
            "displacement": (
                w.displacement,
                lambda: displacement_loss(shape, z, mapped_points=x),
            ),
            "gap": (w.gap, lambda: gap_loss(shape, trap, g_min)),
            "curvature": (w.curvature, lambda: curvature_loss(shape, trap, kappa_max)),
            "anchor": (w.anchor, lambda: anchor_loss(shape, trap, z, mapped_points=x)),
        }

    zero = potential.new_zeros(())
    for name, (weight, compute) in optional.items():
        value = compute() if weight else zero
        total = total + weight * value
        terms[name] = value

    logged = {name: float(value.detach()) for name, value in terms.items()}
    logged["total"] = float(total.detach())
    return total, logged
