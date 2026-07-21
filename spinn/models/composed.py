"""Composed field ``V_theta(NN_phi(z))`` and an identity shape map.

The forward pass returns ``(V, x)`` so the loss layer can take physical-space
derivatives. The reference points ``z`` must carry ``requires_grad`` and ``x``
must stay attached to the graph, so one backward pass reaches both networks.

For fixed-geometry training (plan stage 1) the shape network is replaced by
:class:`IdentityShapeNetwork`, which is exactly the identity, so the pullback
operators reduce to ordinary Cartesian derivatives in ``x = z``.
"""

from __future__ import annotations

import torch
from torch import nn


class IdentityShapeNetwork(nn.Module):
    """Exact identity map with the ``(points, parameter)`` call signature."""

    def forward(self, z: torch.Tensor, parameter=None) -> torch.Tensor:  # noqa: D401
        del parameter
        return z


class ComposedField(nn.Module):
    def __init__(self, shape_network: nn.Module, backend: nn.Module) -> None:
        super().__init__()
        self.shape_network = shape_network
        self.backend = backend

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shape_network(z)
        v = self.backend(x)
        return v, x

    def potential_at(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the backend directly on physical points (no deformation)."""
        return self.backend(x)

    def shape_parameters(self):
        return self.shape_network.parameters()

    def backend_parameters(self):
        return self.backend.parameters()
