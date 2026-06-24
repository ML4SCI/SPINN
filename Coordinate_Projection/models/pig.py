"""PIG potential PINN with learnable Gaussian feature embedding."""

from __future__ import annotations

import math

import torch
from torch import nn


class PigPotentialPINN(nn.Module):
    """Physics-Informed Gaussian representation ``(x, y) -> phi``."""

    def __init__(
        self,
        *,
        domain_bounds: tuple[float, float, float, float],
        num_gaussians: int = 512,
        feature_dim: int = 4,
        hidden_width: int = 16,
        init_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        xmin, xmax, ymin, ymax = domain_bounds
        self.register_buffer("lower", torch.tensor([xmin, ymin], dtype=torch.float32))
        self.register_buffer("upper", torch.tensor([xmax, ymax], dtype=torch.float32))

        means = torch.rand(num_gaussians, 2)
        self.means = nn.Parameter(means)
        self.log_sigma = nn.Parameter(torch.full((num_gaussians, 2), math.log(init_sigma)))
        self.features_parameter = nn.Parameter(
            torch.empty(num_gaussians, feature_dim).uniform_(-1.0, 1.0)
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _unit_coordinates(self, points: torch.Tensor) -> torch.Tensor:
        return torch.clamp((points - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def features(self, points: torch.Tensor) -> torch.Tensor:
        unit = self._unit_coordinates(points)
        diff = unit.unsqueeze(1) - self.means.unsqueeze(0)
        inv_var = torch.exp(-2.0 * self.log_sigma).unsqueeze(0)
        exponent = -0.5 * torch.sum(diff**2 * inv_var, dim=-1)
        weights = torch.exp(torch.clamp(exponent, min=-60.0, max=0.0))
        return weights @ self.features_parameter

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(points))

