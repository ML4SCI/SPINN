"""PDE residual loss: ``L_pde = mean(|Delta_x V|^2)`` over interior collocation.

The Laplacian is taken in physical coordinates. When the shape network is the
identity (fixed-geometry stage) the pullback reduces to the Cartesian Laplacian;
when it deforms the domain, the pullback routes gradients to both networks.
"""

from __future__ import annotations

import torch

from ..physics.laplace import pullback_laplacian


def pde_loss(potential: torch.Tensor, z: torch.Tensor, shape_network) -> torch.Tensor:
    residual = pullback_laplacian(potential, z, shape_network)
    return torch.mean(residual**2)
