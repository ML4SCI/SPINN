"""Small building blocks shared by the shape network and physics backends."""

from __future__ import annotations

import torch
from torch import nn


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATIONS = {"tanh": nn.Tanh, "sine": Sine, "sin": Sine}


def mlp(
    in_dim: int, out_dim: int, width: int, hidden_layers: int, activation: str = "tanh"
) -> nn.Sequential:
    """``hidden_layers`` activated linear layers of size ``width``, then a linear head."""
    act = ACTIVATIONS[activation.lower()]
    layers: list[nn.Module] = [nn.Linear(in_dim, width), act()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(width, width), act()]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


def xavier_init_(network: nn.Module) -> None:
    for layer in network.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)


class BoxNormalized(nn.Module):
    """Maps physical points in ``[-R_out, R_out]^2`` to the ``[-1, 1]^2`` reference box."""

    def __init__(self, outer_radius: float) -> None:
        super().__init__()
        bound = float(outer_radius)
        self.register_buffer("lower", torch.tensor([-bound, -bound]))
        self.register_buffer("upper", torch.tensor([bound, bound]))

    def unit_coords(self, points: torch.Tensor) -> torch.Tensor:
        return 2.0 * ((points - self.lower) / (self.upper - self.lower)) - 1.0
