"""Reference-geometry sampling for the coordinate-projection comparison.

The geometry (a rectangular world with four circular electrodes) and the
interior/world-boundary/electrode sampling mirror
``PINN/Coordinate_Projection_Network/CPN.py``. CPN.py trains at import time
so it cannot be imported as a library; this module re-implements the same
sampling so the experiment stays isolated and CPN.py is left untouched.

All three architectures draw from the same generators, so they see the same
data distribution. The supervised target is the identity map: a model that
recovers ``x = z`` has learned the reference geometry exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import ExperimentConfig


def electrode_centers(config: ExperimentConfig) -> dict[str, tuple[float, float, float]]:
    g = config.geometry
    return {
        "North": (0.0, g.north_distance, g.radius),
        "South": (0.0, -g.south_distance, g.radius),
        "West": (-g.west_distance, 0.0, g.radius),
        "East": (g.east_distance, 0.0, g.radius),
    }


def _inside_any_electrode(xy: np.ndarray, electrodes: dict) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    for cx, cy, r in electrodes.values():
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r**2
    return inside


def sample_interior(config: ExperimentConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    xmin, xmax, ymin, ymax = config.domain_bounds()
    electrodes = electrode_centers(config)
    pts: list[list[float]] = []
    while len(pts) < n:
        xy = np.column_stack([rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)])
        xy = xy[~_inside_any_electrode(xy, electrodes)]
        pts.extend(xy.tolist())
    return np.array(pts[:n], dtype=np.float32)


def sample_world_boundary(config: ExperimentConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    xmin, xmax, ymin, ymax = config.domain_bounds()
    n_each = max(1, n // 4)
    bottom = np.column_stack([rng.uniform(xmin, xmax, n_each), np.full(n_each, ymin)])
    top = np.column_stack([rng.uniform(xmin, xmax, n_each), np.full(n_each, ymax)])
    left = np.column_stack([np.full(n_each, xmin), rng.uniform(ymin, ymax, n_each)])
    right = np.column_stack([np.full(n_each, xmax), rng.uniform(ymin, ymax, n_each)])
    return np.vstack([bottom, top, left, right]).astype(np.float32)


def sample_electrode_boundaries(
    config: ExperimentConfig, n_per_electrode: int, rng: np.random.Generator
) -> np.ndarray:
    pts = []
    for cx, cy, r in electrode_centers(config).values():
        theta = rng.uniform(0, 2 * np.pi, n_per_electrode)
        pts.append(np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]))
    return np.vstack(pts).astype(np.float32)


def sample_reference_points(
    config: ExperimentConfig, rng: np.random.Generator
) -> np.ndarray:
    """One training batch: interior plus world and electrode boundaries."""

    t = config.training
    return np.vstack(
        [
            sample_interior(config, t.batch_interior, rng),
            sample_world_boundary(config, t.batch_world, rng),
            sample_electrode_boundaries(config, t.batch_electrode, rng),
        ]
    ).astype(np.float32)


def to_tensor(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def grid_points(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense evaluation grid plus a mask of points inside an electrode."""

    xmin, xmax, ymin, ymax = config.domain_bounds()
    nx, ny = config.evaluation.nx, config.evaluation.ny
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    mask = _inside_any_electrode(points, electrode_centers(config)).reshape(ny, nx)
    return grid_x, grid_y, mask
