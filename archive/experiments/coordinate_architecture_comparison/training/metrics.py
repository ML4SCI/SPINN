"""Evaluation metrics for predicted coordinate pairs.

``evaluate_coordinate_error`` reports both the mean Euclidean coordinate
error (the training objective) and the mean squared coordinate error (for
the MSE comparison plots), reusing ``PINN.common.metrics.error_statistics``
for the underlying rmse so the experiment stays consistent with the rest of
the repository.
"""

from __future__ import annotations

import numpy as np
import torch

from PINN.common.metrics import error_statistics


@torch.no_grad()
def evaluate_coordinate_error(
    model: torch.nn.Module, z: torch.Tensor
) -> dict[str, float]:
    predicted = model(z)
    diff = (predicted - z).cpu().numpy()

    per_point_sq = np.sum(diff**2, axis=1)
    stats = error_statistics(predicted.cpu().numpy(), z.cpu().numpy())

    return {
        "mean_euclidean_error": float(np.mean(np.sqrt(per_point_sq))),
        "mse": float(np.mean(per_point_sq)),
        "rmse_coordinate": stats["rmse"],
        "max_euclidean_error": float(np.max(np.sqrt(per_point_sq))),
    }


@torch.no_grad()
def spatial_error_grid(
    model: torch.nn.Module, grid_x: np.ndarray, grid_y: np.ndarray, mask: np.ndarray, device: str
) -> np.ndarray:
    """Per-grid-point Euclidean error, NaN inside electrodes for plotting."""

    points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    z = torch.as_tensor(points, dtype=torch.float32, device=device)
    predicted = model(z).cpu().numpy()
    error = np.sqrt(np.sum((predicted - points) ** 2, axis=1)).reshape(grid_x.shape)
    return np.where(mask, np.nan, error)
