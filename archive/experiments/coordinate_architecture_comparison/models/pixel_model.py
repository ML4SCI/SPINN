"""PIXEL-style grid-feature coordinate projection.

Faithful to the PIXEL representation (Kang et al., AAAI 2023): a learnable
2D grid of feature vectors queried by bilinear interpolation, followed by a
shallow MLP. Adapted here so the MLP outputs a 2D displacement instead of a
scalar field, and wrapped in the residual ``z + phi_hat(z)`` structure. The
query coordinates are normalized to ``[-1, 1]`` so ``grid_sample`` indexes
the feature grid; normalization bounds come from the reference domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CoordinateProjectionModel


class PixelCoordinateProjection(CoordinateProjectionModel):
    def __init__(
        self,
        domain_bounds: tuple[float, float, float, float],
        grid_resolution: int = 32,
        feature_dim: int = 4,
        hidden_width: int = 16,
    ) -> None:
        super().__init__()
        xmin, xmax, ymin, ymax = domain_bounds
        self.register_buffer("_lower", torch.tensor([xmin, ymin]))
        self.register_buffer("_upper", torch.tensor([xmax, ymax]))

        self.feature_grid = nn.Parameter(
            torch.zeros(1, feature_dim, grid_resolution, grid_resolution)
        )
        nn.init.normal_(self.feature_grid, std=1e-2)

        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _to_grid_coords(self, z: torch.Tensor) -> torch.Tensor:
        unit = (z - self._lower) / (self._upper - self._lower)
        return 2.0 * unit - 1.0

    def displacement(self, z: torch.Tensor) -> torch.Tensor:
        grid = self._to_grid_coords(z).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            self.feature_grid,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        features = sampled.view(self.feature_grid.shape[1], -1).t()
        return self.head(features)
