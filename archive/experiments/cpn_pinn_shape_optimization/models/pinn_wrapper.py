"""Vanilla PINN potential network.

The ``MLP`` and ``make_activation`` definitions are copied verbatim from
``PINN/Vanilla_Paul_Trap_PINN/Paul_Trap_PINN.py:320-348`` rather than imported,
because importing that module runs script-level side effects (``matplotlib.use``
and a global ``config``). Copying the small, stable network keeps this
experiment self-contained; the provenance is recorded here.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in {"sin", "sine", "sinusoidal"}:
        return Sine()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 5, activation: str = "tanh") -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), make_activation(activation)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), make_activation(activation)]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


def build_pinn(config: ModelConfig) -> MLP:
    return MLP(width=config.pinn_width, depth=config.pinn_depth, activation=config.pinn_activation)
