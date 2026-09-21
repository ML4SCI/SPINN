from __future__ import annotations

import torch
from torch import nn


class IdentityShapeNetwork(nn.Module):
    """exact id map w/ ``(points, parameter)`` call signature."""

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
