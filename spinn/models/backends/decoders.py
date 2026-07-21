"""Shared decoder head for the feature-grid backends (PIXEL, PIG).

A small tanh MLP mapping local PIXEL/PIG feature vectors to the scalar potential
``V``. The defaults mirror the compact official PIXEL/PIG examples: 2 layers,
width 16, tanh.
"""

from __future__ import annotations

import torch
from torch import nn


class DecoderMLP(nn.Module):
    def __init__(self, in_dim: int, layers: int = 2, width: int = 16) -> None:
        super().__init__()
        seq: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(width, width), nn.Tanh()]
        seq.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*seq)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
