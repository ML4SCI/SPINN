"""Residual coordinate map with localization and symmetry projection."""

from __future__ import annotations

import torch
from torch import nn

from ..geometry.four_rod import FourRodTrap
from ..geometry.masks import design_mask
from .layers import mlp

_IDENTITY = [[1.0, 0.0], [0.0, 1.0]]
_REFLECT_X = [[-1.0, 0.0], [0.0, 1.0]]
_REFLECT_Y = [[1.0, 0.0], [0.0, -1.0]]
_ROTATE_180 = [[-1.0, 0.0], [0.0, -1.0]]
_SWAP = [[0.0, 1.0], [1.0, 0.0]]
_ROTATE_90 = [[0.0, -1.0], [1.0, 0.0]]
_ROTATE_270 = [[0.0, 1.0], [-1.0, 0.0]]
_ANTI_SWAP = [[0.0, -1.0], [-1.0, 0.0]]

SYMMETRY_GROUPS = {
    "none": [_IDENTITY],
    "x_mirror": [_IDENTITY, _REFLECT_X],
    "y_mirror": [_IDENTITY, _REFLECT_Y],
    "xy_mirror": [_IDENTITY, _REFLECT_X, _REFLECT_Y, _ROTATE_180],
    "d4": [
        _IDENTITY,
        _REFLECT_X,
        _REFLECT_Y,
        _ROTATE_180,
        _SWAP,
        _ROTATE_90,
        _ROTATE_270,
        _ANTI_SWAP,
    ],
}


class ResidualShapeNetwork(nn.Module):
    """``x = z + alpha * s_design(z) * Delta_phi(z)``, starting at the exact identity."""

    def __init__(
        self,
        trap: FourRodTrap,
        *,
        width: int = 64,
        depth: int = 5,
        activation: str = "tanh",
        alpha: float = 0.10,
        symmetry: str = "xy_mirror",
    ) -> None:
        super().__init__()
        self.trap = trap
        self.alpha = alpha
        self.symmetry = (
            "d4" if symmetry.lower() == "d4_quadrupole" else symmetry.lower()
        )
        if self.symmetry not in SYMMETRY_GROUPS:
            raise ValueError(
                f"unknown shape symmetry {self.symmetry!r}; choose from {sorted(SYMMETRY_GROUPS)}"
            )
        self.network = mlp(2, 2, width, depth, activation)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        # Derived from ``symmetry``, so not persisted: checkpoints stay loadable
        # across symmetry choices and predate this buffer.
        self.register_buffer(
            "symmetry_matrices",
            torch.tensor(SYMMETRY_GROUPS[self.symmetry]),
            persistent=False,
        )

    def _raw_displacement(self, z: torch.Tensor) -> torch.Tensor:
        # Reynolds projection for a vector field: g^-1 u(g z). For these
        # orthogonal matrices and row-vector tensors this is raw @ g.
        projected = [self.network(z @ g.T) @ g for g in self.symmetry_matrices]
        return torch.stack(projected).mean(dim=0)

    def displacement(self, z: torch.Tensor) -> torch.Tensor:
        """The masked, scaled residual ``alpha * s_design(z) * Delta_phi(z)``."""
        return self.alpha * design_mask(z, self.trap) * self._raw_displacement(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.displacement(z)


def build_shape_network(config, trap: FourRodTrap) -> ResidualShapeNetwork:
    sn = config.shape_network
    return ResidualShapeNetwork(
        trap,
        width=sn.width,
        depth=sn.hidden_layers,
        activation=sn.activation,
        alpha=sn.alpha,
        symmetry=sn.symmetry,
    )
