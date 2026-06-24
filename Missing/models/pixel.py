"""PIXEL potential PINN with smooth cosine grid interpolation."""

from __future__ import annotations

import torch
from torch import nn


class PixelPotentialPINN(nn.Module):
    """PIXEL-style learnable cell representation for scalar potential.

    This follows PIXEL's role as a PDE solution representation:
    coordinates query learnable feature grids, then a shallow MLP predicts
    ``phi``. It does not output a coordinate displacement.
    """

    def __init__(
        self,
        *,
        domain_bounds: tuple[float, float, float, float],
        grid_resolution: int = 32,
        feature_dim: int = 4,
        hidden_width: int = 16,
        num_grids: int = 4,
    ) -> None:
        super().__init__()
        if grid_resolution < 2:
            raise ValueError("grid_resolution must be at least 2")
        xmin, xmax, ymin, ymax = domain_bounds
        self.register_buffer("lower", torch.tensor([xmin, ymin], dtype=torch.float32))
        self.register_buffer("upper", torch.tensor([xmax, ymax], dtype=torch.float32))
        self.grid_resolution = grid_resolution
        self.num_grids = num_grids
        self.feature_grids = nn.Parameter(
            0.01 * torch.randn(num_grids, feature_dim, grid_resolution, grid_resolution)
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )

    @staticmethod
    def _cosine_kernel(weight: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 - torch.cos(torch.pi * weight))

    def _unit_coordinates(self, points: torch.Tensor) -> torch.Tensor:
        return torch.clamp((points - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def _interpolate_one_grid(self, grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        resolution = self.grid_resolution
        scaled = coords * (resolution - 1)
        base = torch.floor(scaled).to(torch.long)
        frac = scaled - base.to(scaled.dtype)
        base = torch.clamp(base, 0, resolution - 1)
        upper = torch.clamp(base + 1, 0, resolution - 1)

        wx0 = self._cosine_kernel(1.0 - frac[:, 0:1])
        wx1 = self._cosine_kernel(frac[:, 0:1])
        wy0 = self._cosine_kernel(1.0 - frac[:, 1:2])
        wy1 = self._cosine_kernel(frac[:, 1:2])

        x0, y0 = base[:, 0], base[:, 1]
        x1, y1 = upper[:, 0], upper[:, 1]
        f00 = grid[:, y0, x0].T
        f10 = grid[:, y0, x1].T
        f01 = grid[:, y1, x0].T
        f11 = grid[:, y1, x1].T
        return (
            wx0 * wy0 * f00
            + wx1 * wy0 * f10
            + wx0 * wy1 * f01
            + wx1 * wy1 * f11
        )

    def features(self, points: torch.Tensor) -> torch.Tensor:
        unit = self._unit_coordinates(points)
        summed = torch.zeros(
            points.shape[0],
            self.feature_grids.shape[1],
            dtype=points.dtype,
            device=points.device,
        )
        cell_shift = 1.0 / max(self.num_grids, 1) / max(self.grid_resolution - 1, 1)
        for index, grid in enumerate(self.feature_grids):
            shifted = torch.clamp(unit + index * cell_shift, 0.0, 1.0)
            summed = summed + self._interpolate_one_grid(grid, shifted)
        return summed

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(points))

