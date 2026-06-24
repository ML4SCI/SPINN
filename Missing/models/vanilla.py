"""Vanilla MLP potential PINN, matching the fixed-geometry baseline role."""

from __future__ import annotations

import torch
from torch import nn


class _Sine(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(inputs)


def _activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "tanh":
        return nn.Tanh()
    if lowered == "gelu":
        return nn.GELU()
    if lowered in {"sine", "sin", "sinusoidal"}:
        return _Sine()
    if lowered in {"relu", "leaky_relu", "hardtanh"}:
        raise ValueError(
            f"{name} is unsuitable for a second-order strong-form PINN"
        )
    raise ValueError(f"unsupported activation: {name}")


class VanillaPotentialPINN(nn.Module):
    """MLP ``(x, y) -> phi`` with smooth activation."""

    def __init__(
        self,
        *,
        width: int = 128,
        depth: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), _activation(activation)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), _activation(activation)])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points)

