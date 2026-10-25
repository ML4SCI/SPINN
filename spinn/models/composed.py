"""Composition of a reference-to-physical shape map and a field backend."""

from __future__ import annotations

import torch
from torch import nn


class IdentityShapeNetwork(nn.Module):
    """Exact identity map used by fixed-geometry experiments."""

    is_identity = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class ComposedField(nn.Module):
    """Evaluate ``V_theta(NN_phi(z))`` while retaining mapped coordinates."""

    def __init__(self, shape_network: nn.Module, backend: nn.Module) -> None:
        super().__init__()
        self.shape_network = shape_network
        self.backend = backend

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shape_network(z)
        return self.backend(x), x
