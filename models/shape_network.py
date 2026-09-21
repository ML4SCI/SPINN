from __future__ import annotations

import torch
from torch import nn

from ..geometry.four_rod import FourRodTrap
from ..geometry.masks import design_mask


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name in {"sine", "sin"}:
        class _Sine(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sin(x)

        return _Sine()
    raise ValueError(f"unsupported shape-network activation: {name}")


def _build_mlp(width: int, depth: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(2, width), _activation(activation)]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), _activation(activation)]
    final = nn.Linear(width, 2)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ResidualShapeNetwork(nn.Module):
    def __init__(
        self,
        trap: FourRodTrap,
        *,
        width: int = 64,
        depth: int = 5,
        activation: str = "tanh",
        alpha: float = 0.10,
        symmetric: bool = True,
        symmetry: str | None = None,
    ) -> None:
        super().__init__()
        self.trap = trap
        self.alpha = alpha
        self.symmetry = ("xy_mirror" if symmetric else "none") if symmetry is None else symmetry.lower()
        if self.symmetry == "d4_quadrupole":
            self.symmetry = "d4"
        supported = {"none", "x_mirror", "y_mirror", "xy_mirror", "d4"}
        if self.symmetry not in supported:
            raise ValueError(f"unknown shape symmetry {self.symmetry!r}; choose from {sorted(supported)}")
        self.symmetric = self.symmetry != "none"
        self.network = _build_mlp(width, depth, activation)

        identity = torch.eye(2)
        reflect_x = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
        reflect_y = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        rotate_180 = -identity
        swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        rotate_90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        rotate_270 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        anti_swap = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
        groups = {
            "none": torch.stack([identity]),
            "x_mirror": torch.stack([identity, reflect_x]),
            "y_mirror": torch.stack([identity, reflect_y]),
            "xy_mirror": torch.stack([identity, reflect_x, reflect_y, rotate_180]),
            "d4": torch.stack([identity, reflect_x, reflect_y, rotate_180, swap, rotate_90, rotate_270, anti_swap]),
        }
        self.register_buffer("symmetry_matrices", groups[self.symmetry])

    def _raw_displacement(self, z: torch.Tensor) -> torch.Tensor:
        projected = []
        for matrix in self.symmetry_matrices:
            transformed_points = z @ matrix.T
            # Reynolds projection for a vector field: g^-1 u(g z). For these
            # orthogonal matrices and row-vector tensors this is raw @ g.
            projected.append(self.network(transformed_points) @ matrix)
        return torch.stack(projected).mean(dim=0)

    def displacement(self, z: torch.Tensor) -> torch.Tensor:
        """The masked, scaled residual ``alpha * s_design(z) * Delta_phi(z)``."""
        mask = design_mask(z, self.trap)
        return self.alpha * mask * self._raw_displacement(z)

    def forward(
        self, z: torch.Tensor, parameter: float | torch.Tensor | None = None
    ) -> torch.Tensor:
        del parameter
        return z + self.displacement(z)

    def reset_to_identity(self) -> None:
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


def build_shape_network(config, trap: FourRodTrap) -> ResidualShapeNetwork:
    sn = config.shape_network
    return ResidualShapeNetwork(
        trap,
        width=sn.width,
        depth=sn.hidden_layers,
        activation=sn.activation,
        alpha=sn.alpha,
        symmetric=sn.symmetric,
        symmetry=sn.symmetry,
    )
