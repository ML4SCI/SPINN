"""Differentiable coordinate maps from a fixed reference trap domain."""

from __future__ import annotations

import math

import torch
from torch import nn

from .layers import build_mlp


def _as_batch_parameter(
    value: float | torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=points.dtype, device=points.device)
    if result.ndim == 0:
        result = result.expand(len(points), 1)
    elif result.ndim == 1:
        result = result[:, None]
    if result.shape != (len(points), 1):
        raise ValueError("map parameter must be scalar or have one value per point")
    return result


class IdentityMap(nn.Module):
    def forward(
        self,
        points: torch.Tensor,
        parameter: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del parameter
        return points


class AffineMap(nn.Module):
    def __init__(self, matrix: torch.Tensor, offset: torch.Tensor | None = None) -> None:
        super().__init__()
        matrix = torch.as_tensor(matrix)
        if matrix.shape != (2, 2):
            raise ValueError("matrix must have shape (2, 2)")
        if offset is None:
            offset = torch.zeros(2, dtype=matrix.dtype)
        self.register_buffer("matrix", matrix)
        self.register_buffer("offset", torch.as_tensor(offset, dtype=matrix.dtype))

    def forward(
        self,
        points: torch.Tensor,
        parameter: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del parameter
        return points @ self.matrix.T + self.offset


class RadialDeformationMap(nn.Module):
    """Smooth radial deformation that decays away from the origin."""

    def __init__(self, amplitude: float = 0.08, length_scale: float = 0.7) -> None:
        super().__init__()
        self.amplitude = amplitude
        self.length_scale = length_scale

    def forward(
        self,
        points: torch.Tensor,
        parameter: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        amplitude = self.amplitude if parameter is None else parameter
        amplitude = _as_batch_parameter(amplitude, points)
        radius2 = torch.sum(points**2, dim=1, keepdim=True)
        scale = 1.0 + amplitude * torch.exp(-radius2 / self.length_scale**2)
        return scale * points


def _smootherstep(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


class ElectrodeTranslationMap(nn.Module):
    """Move four circular electrodes while anchoring the rest of the world.

    The map is exactly a translation in a plateau surrounding each reference
    electrode and smoothly returns to identity outside its compact support.
    """

    def __init__(
        self,
        reference_distance: float = 0.125,
        radius: float = 0.05,
        plateau_margin: float = 0.005,
        support_radius: float = 0.12,
    ) -> None:
        super().__init__()
        if support_radius <= radius + plateau_margin:
            raise ValueError("support_radius must exceed radius + plateau_margin")
        self.reference_distance = reference_distance
        self.radius = radius
        self.plateau_radius = radius + plateau_margin
        self.support_radius = support_radius
        centers = torch.tensor([
            [reference_distance, 0.0],
            [-reference_distance, 0.0],
            [0.0, reference_distance],
            [0.0, -reference_distance],
        ])
        directions = torch.tensor([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        self.register_buffer("centers", centers)
        self.register_buffer("directions", directions)

    def forward(
        self,
        points: torch.Tensor,
        distance: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if distance is None:
            distance = self.reference_distance
        distance = _as_batch_parameter(distance, points)
        displacement_size = distance - self.reference_distance

        relative = points[:, None, :] - self.centers.to(points)[None, :, :]
        radius = torch.linalg.vector_norm(relative, dim=2)
        transition = (radius - self.plateau_radius) / (
            self.support_radius - self.plateau_radius
        )
        weights = 1.0 - _smootherstep(transition)
        displacement = torch.sum(
            weights[:, :, None] * self.directions.to(points)[None, :, :],
            dim=1,
        )
        return points + displacement_size * displacement

    def minimum_electrode_gap(self, distance: torch.Tensor) -> torch.Tensor:
        """Minimum surface gap among the four translated circular electrodes."""

        adjacent_gap = math.sqrt(2.0) * distance - 2.0 * self.radius
        opposing_gap = 2.0 * distance - 2.0 * self.radius
        return torch.minimum(adjacent_gap, opposing_gap)


class ResidualCoordinateMap(nn.Module):
    r"""Equation-13 residual shape network ``NN_phi(z) = z + NN_hat(z)``.

    The final layer is initialized to zero, so the complete map starts as the
    exact identity without requiring a separate supervised warm-up. World
    anchoring, symmetry, volume, and Jacobian bounds are imposed as losses,
    matching the separation between architecture and constraints in the paper.
    """

    def __init__(
        self,
        width: int = 50,
        depth: int = 6,
        max_displacement: float | None = None,
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            2, 2,
            width=width,
            depth=depth,
            activation_factory=nn.Tanh,
            zero_initialize_output=zero_initialize,
        )
        self.max_displacement = max_displacement

    def forward(
        self,
        points: torch.Tensor,
        parameter: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del parameter
        displacement = self.network(points)
        if self.max_displacement is not None:
            displacement = self.max_displacement * torch.tanh(displacement)
        return points + displacement

    def residual(self, points: torch.Tensor) -> torch.Tensor:
        """Return the learned deformation ``NN_hat(z)``."""

        mapped = self(points)
        return mapped - points

    def reset_to_identity(self) -> None:
        """Reset the output layer so the complete map is exactly identity."""

        output = self.network[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("residual projection output layer must be linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
