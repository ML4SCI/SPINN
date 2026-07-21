"""The composed model ``phi(z) = NN_theta(NN_phi(z))``.

The forward pass returns both the potential and the physical coordinates so the
loss layer can evaluate the pullback physics. The reference points ``z`` must
carry ``requires_grad`` and the physical points ``x`` must stay attached to the
graph; the pullback operators differentiate the potential with respect to ``z``
and reuse the map's Jacobian, so detaching either would silently drop gradient
terms for one of the networks.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch import nn

from .cpn_wrapper import as_two_arg_map
from .pinn_wrapper import MLP
from benchmark.experiments.instructor_coordinate_projection.model import (
    CoordinateProjectionNet,
)


class CoupledCPNPINN(nn.Module):
    def __init__(self, pinn: MLP, cpn: CoordinateProjectionNet) -> None:
        super().__init__()
        self.pinn = pinn
        self.cpn = cpn
        self.map = as_two_arg_map(cpn)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.map(z)
        u = self.pinn(x)
        return u, x

    def cpn_parameters(self) -> Iterator[nn.Parameter]:
        return self.cpn.parameters()

    def pinn_parameters(self) -> Iterator[nn.Parameter]:
        return self.pinn.parameters()
