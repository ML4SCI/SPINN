"""Shared decoder head for the feature-grid backends (PIXEL, PIG).

A small tanh MLP mapping local PIXEL/PIG feature vectors to the scalar potential
``V``. The defaults mirror the compact official PIXEL/PIG examples: 2 layers,
width 16, tanh.
"""

from __future__ import annotations

import torch
from torch import nn

from ..layers import mlp, xavier_init_


class DecoderMLP(nn.Module):
    def __init__(self, in_dim: int, layers: int = 2, width: int = 16) -> None:
        super().__init__()
        self.net = mlp(in_dim, 1, width, layers)
        xavier_init_(self.net)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
