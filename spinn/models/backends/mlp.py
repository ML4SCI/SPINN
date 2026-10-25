"""MLP physics backend (vanilla PINN baseline). Plan sec. 3.2: 6 hidden layers,
width 128, tanh, scalar output V, Xavier initialization."""

from __future__ import annotations

import torch
from torch import nn

from ..layers import mlp, xavier_init_


class MLPBackend(nn.Module):
    def __init__(self, *, hidden_layers: int = 6, width: int = 128) -> None:
        super().__init__()
        self.network = mlp(2, 1, width, hidden_layers)
        xavier_init_(self.network)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points)

    def param_groups(self, lr_main: float, **_unused) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr_main}]
