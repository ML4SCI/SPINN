"""Vanilla residual-MLP coordinate projection.

This mirrors ``CoordinateProjectionNet``/``MLP`` in
``PINN/Coordinate_Projection_Network/CPN.py``. CPN.py runs a full training
job at import time, so it cannot be imported as a library without side
effects; to keep that file untouched the same small architecture is
re-declared here. Keep the two in sync if CPN.py's network changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import CoordinateProjectionModel


class _DisplacementMLP(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VanillaCoordinateProjection(CoordinateProjectionModel):
    def __init__(self, width: int = 50, depth: int = 6, zero_last_layer: bool = True) -> None:
        super().__init__()
        self.mlp = _DisplacementMLP(width=width, depth=depth)
        if zero_last_layer:
            last = self.mlp.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def displacement(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)
