"""Common interfaces for evaluating PINN and numerical-solver fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class FieldBackend(ABC):
    """A scalar potential field with first- and second-derivative access."""

    dimension: int

    @abstractmethod
    def potential(self, points: np.ndarray) -> np.ndarray:
        """Return potential in volts at points shaped ``(n, dimension)``."""

    @abstractmethod
    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Return coordinate gradients in V per geometry unit."""

    @abstractmethod
    def hessian(self, points: np.ndarray) -> np.ndarray:
        """Return Hessians in V per geometry-unit squared."""


@dataclass
class TorchFieldBackend(FieldBackend):
    """Autograd-backed adapter for a scalar PyTorch model."""

    model: torch.nn.Module
    dimension: int = 2
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"

    def _tensor(self, points: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(points),
            dtype=self.dtype,
            device=self.device,
        ).requires_grad_(True)

    def potential(self, points: np.ndarray) -> np.ndarray:
        x = self._tensor(points)
        return self.model(x).detach().cpu().numpy().reshape(-1)

    def gradient(self, points: np.ndarray) -> np.ndarray:
        x = self._tensor(points)
        phi = self.model(x)
        grad = torch.autograd.grad(
            phi,
            x,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
        )[0]
        return grad.detach().cpu().numpy()

    def hessian(self, points: np.ndarray) -> np.ndarray:
        x = self._tensor(points)
        phi = self.model(x)
        grad = torch.autograd.grad(
            phi,
            x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]
        rows = []
        for axis in range(self.dimension):
            row = torch.autograd.grad(
                grad[:, axis],
                x,
                grad_outputs=torch.ones_like(grad[:, axis]),
                create_graph=False,
                retain_graph=axis + 1 < self.dimension,
            )[0]
            rows.append(row)
        hessian = torch.stack(rows, dim=1)
        return hessian.detach().cpu().numpy()


class ScatteredFieldBackend(FieldBackend):
    """Interpolated field adapter for DEVSIM or other scattered samples."""

    def __init__(
        self,
        points: np.ndarray,
        potential: np.ndarray,
        gradient: np.ndarray | None = None,
    ) -> None:
        points = np.asarray(points, dtype=float)
        values = np.asarray(potential, dtype=float).reshape(-1)
        if points.ndim != 2 or len(points) != len(values):
            raise ValueError("points and potential must have matching leading dimensions")

        self.dimension = points.shape[1]
        self.points = points
        self.values = values
        self._potential_linear = LinearNDInterpolator(points, values)
        self._potential_nearest = NearestNDInterpolator(points, values)

        self.gradient_values = None if gradient is None else np.asarray(gradient, dtype=float)
        self._gradient_linear = None
        self._gradient_nearest = None
        if self.gradient_values is not None:
            if self.gradient_values.shape != points.shape:
                raise ValueError("gradient must have the same shape as points")
            self._gradient_linear = [
                LinearNDInterpolator(points, self.gradient_values[:, axis])
                for axis in range(self.dimension)
            ]
            self._gradient_nearest = [
                NearestNDInterpolator(points, self.gradient_values[:, axis])
                for axis in range(self.dimension)
            ]

    @staticmethod
    def _fill(linear, nearest, points: np.ndarray) -> np.ndarray:
        values = np.asarray(linear(points), dtype=float)
        missing = ~np.isfinite(values)
        if np.any(missing):
            values[missing] = nearest(points[missing])
        return values

    def potential(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return self._fill(self._potential_linear, self._potential_nearest, points)

    def gradient(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if self._gradient_linear is not None and self._gradient_nearest is not None:
            return np.column_stack([
                self._fill(linear, nearest, points)
                for linear, nearest in zip(self._gradient_linear, self._gradient_nearest)
            ])

        scale = np.ptp(self.points, axis=0)
        step = np.maximum(scale * 1e-4, 1e-7)
        columns = []
        for axis in range(self.dimension):
            offset = np.zeros(self.dimension)
            offset[axis] = step[axis]
            columns.append(
                (self.potential(points + offset) - self.potential(points - offset))
                / (2.0 * step[axis])
            )
        return np.column_stack(columns)

    def hessian(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        scale = np.ptp(self.points, axis=0)
        step = np.maximum(scale * 5e-4, 1e-6)
        result = np.empty((len(points), self.dimension, self.dimension), dtype=float)
        for axis in range(self.dimension):
            offset = np.zeros(self.dimension)
            offset[axis] = step[axis]
            result[:, :, axis] = (
                self.gradient(points + offset) - self.gradient(points - offset)
            ) / (2.0 * step[axis])
        return 0.5 * (result + np.swapaxes(result, 1, 2))
