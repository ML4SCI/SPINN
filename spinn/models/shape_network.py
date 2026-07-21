"""NN_phi: residual coordinate-projection network (the shape network).

Plan sec. 3.1 / deck p.14:

    x = NN_phi(z) = z + alpha * s_design(z) * Delta_phi(z)

* ``Delta_phi`` is a tanh MLP ``(z1, z2) -> (dx1, dx2)`` whose final layer is
  initialized to zero, so the complete map starts as the exact identity;
* ``s_design`` localizes the deformation to a band around the movable rods;
* ``alpha`` is the residual scale (0.05-0.15 r0 initially);
* optional x/y mirror symmetry constrains the deformation to respect the
  quadrupole's two mirror planes (first main benchmark, plan sec. 3.1).

The Jacobian ``J_phi = d NN_phi / dz`` is exposed for the positive-Jacobian
constraint and for det(J) diagnostics.
"""

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
    ) -> None:
        super().__init__()
        self.trap = trap
        self.alpha = alpha
        self.symmetric = symmetric
        self.network = _build_mlp(width, depth, activation)

    def _raw_displacement(self, z: torch.Tensor) -> torch.Tensor:
        if not self.symmetric:
            return self.network(z)
        # Enforce x/y mirror symmetry of the deformation field by averaging the
        # network over the four reflections with the matching sign flips. A
        # deformation that respects both mirror planes satisfies
        #   dx(-x, y) = -dx(x, y),  dx(x, -y) = dx(x, y)   (and y analogously),
        # so the quadrupole symmetry of the trap is preserved.
        x, y = z[:, 0:1], z[:, 1:2]
        d_pp = self.network(torch.cat([x, y], dim=1))
        d_mp = self.network(torch.cat([-x, y], dim=1))
        d_pm = self.network(torch.cat([x, -y], dim=1))
        d_mm = self.network(torch.cat([-x, -y], dim=1))
        dx = 0.25 * (d_pp[:, 0:1] - d_mp[:, 0:1] + d_pm[:, 0:1] - d_mm[:, 0:1])
        dy = 0.25 * (d_pp[:, 1:2] + d_mp[:, 1:2] - d_pm[:, 1:2] - d_mm[:, 1:2])
        return torch.cat([dx, dy], dim=1)

    def displacement(self, z: torch.Tensor) -> torch.Tensor:
        """The masked, scaled residual ``alpha * s_design(z) * Delta_phi(z)``."""
        mask = design_mask(z, self.trap)
        return self.alpha * mask * self._raw_displacement(z)

    def forward(
        self, z: torch.Tensor, parameter: float | torch.Tensor | None = None
    ) -> torch.Tensor:
        del parameter  # accepted for pullback-operator compatibility
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
    )
