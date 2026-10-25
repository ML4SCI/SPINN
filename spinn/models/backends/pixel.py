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

from ..layers import BoxNormalized
from .decoders import DecoderMLP


def _cosine_kernel(weight: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 - torch.cos(torch.pi * weight))


class PixelBackend(BoxNormalized):
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
        super().__init__(outer_radius)
        if grid_resolution < 2:
            raise ValueError("grid_resolution must be at least 2")
        self.grid_resolution = grid_resolution
        self.num_grids = num_grids
        self.feature_grids = nn.Parameter(
            torch.empty(
                num_grids, feature_dim, grid_resolution, grid_resolution
            ).uniform_(-1e-5, 1e-5)
        )
        self.decoder = DecoderMLP(
            feature_dim, layers=decoder_layers, width=decoder_width
        )

    def _interpolate_one(
        self, grid: torch.Tensor, coords: torch.Tensor, offset: float
    ) -> torch.Tensor:
        """Official-style 2D cosine grid sampling for one cell grid."""
        res = self.grid_resolution
        scaled = ((coords + 1.0) / 2.0) * (res - 2) + offset
        ix, iy = scaled[:, 0], scaled[:, 1]

        with torch.no_grad():
            ix_left = torch.floor(ix)
            ix_right = ix_left + 1
            iy_top = torch.floor(iy)
            iy_bottom = iy_top + 1

        dx_right = _cosine_kernel(ix_right - ix)[:, None]
        dx_left = 1.0 - dx_right
        dy_bottom = _cosine_kernel(iy_bottom - iy)[:, None]
        dy_top = 1.0 - dy_bottom

        def cell(index: torch.Tensor) -> torch.Tensor:
            return torch.clamp(index, 0, res - 1).to(torch.long)

        x0, x1, y0, y1 = cell(ix_left), cell(ix_right), cell(iy_top), cell(iy_bottom)
        return (
            dx_right * dy_bottom * grid[:, y0, x0].T
            + dx_left * dy_bottom * grid[:, y0, x1].T
            + dx_right * dy_top * grid[:, y1, x0].T
            + dx_left * dy_top * grid[:, y1, x1].T
        )

    def features(self, points: torch.Tensor) -> torch.Tensor:
        coords = self.unit_coords(points)
        summed = points.new_zeros(points.shape[0], self.feature_grids.shape[1])
        for index, grid in enumerate(self.feature_grids):
            summed = summed + self._interpolate_one(
                grid, coords, index / self.num_grids
            )
        return summed

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.features(points))

    def feature_norm_grid(self) -> torch.Tensor:
        """Per-cell feature norm summed over grids, shaped ``(res, res)``."""
        with torch.no_grad():
            return torch.linalg.vector_norm(self.feature_grids, dim=1).sum(dim=0)

    def param_groups(self, lr_main: float, **_unused) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr_main}]
