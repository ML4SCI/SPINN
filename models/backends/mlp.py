"""MLP physics backend (vanilla PINN baseline). Plan sec. 3.2: 6 hidden layers,
width 128, tanh, scalar output V, Xavier initialization."""

from __future__ import annotations

import torch
from torch import nn


class MLPBackend(nn.Module):
    kind = "mlp"

    def __init__(self, *, hidden_layers: int = 6, width: int = 128, activation: str = "tanh") -> None:
        super().__init__()
        if activation.lower() != "tanh":
            raise ValueError("MLP backend uses tanh for a second-order strong-form PINN")
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points)

    def param_groups(self, lr_main: float, **_unused) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr_main}]
