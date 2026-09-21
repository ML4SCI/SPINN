"""PIXEL physics backend: learnable multigrid cell features + decoder MLP.

This follows the official PIXEL reference implementation in
``external/PIXEL/network.py``: coordinates query learnable feature cells by
smooth cosine interpolation, the multigrid/cell dimension is summed, and the
resulting feature vector alone is decoded to the scalar field. The only SPINN
adaptation is the coordinate normalization from Paul-trap physical coordinates
``[-R_out, R_out]^2`` to PIXEL's expected ``[-1, 1]^2`` query coordinates.
"""

from __future__ import annotations

import torch
from torch import nn

from .decoders import DecoderMLP


class PixelBackend(nn.Module):
    kind = "pixel"

    def __init__(
        self,
        *,
        outer_radius: float,
        grid_resolution: int = 16,
        feature_dim: int = 4,
        num_grids: int = 16,
        decoder_layers: int = 2,
        decoder_width: int = 16,
    ) -> None:
        super().__init__()
        if grid_resolution < 2:
            raise ValueError("grid_resolution must be at least 2")
        self.grid_resolution = grid_resolution
        self.num_grids = num_grids
        bound = float(outer_radius)
        self.register_buffer("lower", torch.tensor([-bound, -bound], dtype=torch.float32))
        self.register_buffer("upper", torch.tensor([bound, bound], dtype=torch.float32))
        self.feature_grids = nn.Parameter(
            torch.empty(num_grids, feature_dim, grid_resolution, grid_resolution).uniform_(-1e-5, 1e-5)
        )
        self.decoder = DecoderMLP(feature_dim, layers=decoder_layers, width=decoder_width)

    @staticmethod
    def _cosine_kernel(weight: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 - torch.cos(torch.pi * weight))

    def _pixel_coords(self, points: torch.Tensor) -> torch.Tensor:
        unit = (points - self.lower) / (self.upper - self.lower)
        return 2.0 * unit - 1.0

    def _interpolate_one(self, grid: torch.Tensor, coords: torch.Tensor, offset: float) -> torch.Tensor:
        """Official-style 2D cosine grid sampling for one cell grid."""
        res = self.grid_resolution
        scaled = ((coords + 1.0) / 2.0) * (res - 2) + offset
        ix = scaled[:, 0]
        iy = scaled[:, 1]

        with torch.no_grad():
            ix_left = torch.floor(ix)
            ix_right = ix_left + 1
            iy_top = torch.floor(iy)
            iy_bottom = iy_top + 1

        dx_right = self._cosine_kernel(ix_right - ix)
        dx_left = 1.0 - dx_right
        dy_bottom = self._cosine_kernel(iy_bottom - iy)
        dy_top = 1.0 - dy_bottom

        x0 = torch.clamp(ix_left, 0, res - 1).to(torch.long)
        x1 = torch.clamp(ix_right, 0, res - 1).to(torch.long)
        y0 = torch.clamp(iy_top, 0, res - 1).to(torch.long)
        y1 = torch.clamp(iy_bottom, 0, res - 1).to(torch.long)

        f00 = grid[:, y0, x0].T
        f10 = grid[:, y0, x1].T
        f01 = grid[:, y1, x0].T
        f11 = grid[:, y1, x1].T
        return (
            dx_right[:, None] * dy_bottom[:, None] * f00
            + dx_left[:, None] * dy_bottom[:, None] * f10
            + dx_right[:, None] * dy_top[:, None] * f01
            + dx_left[:, None] * dy_top[:, None] * f11
        )

    def features(self, points: torch.Tensor) -> torch.Tensor:
        coords = self._pixel_coords(points)
        summed = torch.zeros(
            points.shape[0], self.feature_grids.shape[1], dtype=points.dtype, device=points.device
        )
        denom = max(self.num_grids, 1)
        for index, grid in enumerate(self.feature_grids):
            offset = index / denom
            summed = summed + self._interpolate_one(grid, coords, offset)
        return summed

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.features(points))

    def feature_norm_grid(self) -> torch.Tensor:
        """Per-cell feature norm summed over grids, shaped ``(res, res)``."""
        with torch.no_grad():
            return torch.linalg.vector_norm(self.feature_grids, dim=1).sum(dim=0)

    def param_groups(self, lr_main: float, **_unused) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr_main}]
