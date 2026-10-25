"""PDE residual loss: ``L_pde = mean(|Delta_x V|^2)`` over interior collocation.

The Laplacian is taken in physical coordinates. When the shape network is the
identity (fixed-geometry stage) the pullback reduces to the Cartesian Laplacian;
when it deforms the domain, the pullback routes gradients to both networks.
"""

from __future__ import annotations

import torch

from ..physics.laplace import cartesian_laplacian, pullback_laplacian


def pde_loss(
    potential: torch.Tensor,
    z: torch.Tensor,
    shape_network,
    *,
    mapped_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the mean-squared physical Laplace residual.

    The identity-map fast path avoids constructing map Jacobians and inverting
    one matrix per collocation point during fixed-geometry training.
    """
    if getattr(shape_network, "is_identity", False):
        residual = cartesian_laplacian(potential, z)
    else:
        residual = pullback_laplacian(
            potential, z, shape_network, mapped_points=mapped_points
        )
    return torch.mean(residual**2)
