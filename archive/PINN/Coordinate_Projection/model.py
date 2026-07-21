"""Parametric potential model used by the coordinate-projection experiment."""

from __future__ import annotations

import torch
from torch import nn

from .layers import build_mlp, resolve_activation, validate_second_order_activation


class ParametricPotentialMLP(nn.Module):
    """Predict normalized potential from reference coordinates and geometry."""

    def __init__(
        self,
        *,
        width: int = 128,
        depth: int = 4,
        activation: str = "tanh",
        distance_min: float = 0.11,
        distance_max: float = 0.145,
    ) -> None:
        super().__init__()
        validate_second_order_activation(activation)
        self.network = build_mlp(
            3, 1,
            width=width,
            depth=depth,
            activation_factory=resolve_activation(activation),
        )
        self.distance_min = distance_min
        self.distance_max = distance_max

    def normalized_distance(self, distance: torch.Tensor) -> torch.Tensor:
        return 2.0 * (distance - self.distance_min) / (
            self.distance_max - self.distance_min
        ) - 1.0

    def forward(self, points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        if distance.ndim == 1:
            distance = distance[:, None]
        features = torch.cat([points, self.normalized_distance(distance)], dim=1)
        return self.network(features)


class PhysicalPotentialMLP(nn.Module):
    """Paper-aligned physical-field network ``NN_theta(x)``."""

    def __init__(
        self,
        *,
        width: int = 50,
        depth: int = 6,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        validate_second_order_activation(activation)
        if activation.lower() != "tanh":
            raise ValueError("the paper-aligned physical network currently uses tanh")
        self.network = build_mlp(
            2, 1,
            width=width,
            depth=depth,
            activation_factory=resolve_activation(activation),
        )

    def forward(self, physical_points: torch.Tensor) -> torch.Tensor:
        return self.network(physical_points)


class CoordinateProjectionPINN(nn.Module):
    r"""Compose the shape and field networks as ``NN_theta(NN_phi(z))``."""

    def __init__(
        self,
        shape_network: nn.Module,
        field_network: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.shape_network = shape_network
        self.field_network = field_network or PhysicalPotentialMLP()

    def project(self, reference_points: torch.Tensor) -> torch.Tensor:
        return self.shape_network(reference_points)

    def forward(
        self,
        reference_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        physical_points = self.project(reference_points)
        potential = self.field_network(physical_points)
        return physical_points, potential
