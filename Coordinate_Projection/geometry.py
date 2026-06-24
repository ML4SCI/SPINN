"""Circular Paul-trap sampling shared by all potential architectures."""

from __future__ import annotations

import numpy as np
import torch

from .config import BoundaryConfig, ExperimentConfig, GeometryConfig


def electrode_centers(geometry: GeometryConfig) -> dict[str, tuple[float, float, float]]:
    return {
        "North": (0.0, geometry.north_distance, geometry.radius),
        "South": (0.0, -geometry.south_distance, geometry.radius),
        "West": (-geometry.west_distance, 0.0, geometry.radius),
        "East": (geometry.east_distance, 0.0, geometry.radius),
    }


def electrode_voltages(boundary: BoundaryConfig) -> dict[str, float]:
    return {
        "North": boundary.north_voltage,
        "South": boundary.south_voltage,
        "West": boundary.west_voltage,
        "East": boundary.east_voltage,
    }


def inside_any_electrode(points: np.ndarray, geometry: GeometryConfig) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for cx, cy, radius in electrode_centers(geometry).values():
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    return inside


def sample_interior(config: ExperimentConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    xmin, xmax, ymin, ymax = config.domain_bounds()
    points: list[list[float]] = []
    while len(points) < n:
        candidates = np.column_stack([
            rng.uniform(xmin, xmax, n),
            rng.uniform(ymin, ymax, n),
        ])
        accepted = candidates[~inside_any_electrode(candidates, config.geometry)]
        points.extend(accepted.tolist())
    return np.asarray(points[:n], dtype=np.float32)


def sample_world_boundary(
    config: ExperimentConfig,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = config.domain_bounds()
    n_each = max(1, n // 4)
    bottom = np.column_stack([rng.uniform(xmin, xmax, n_each), np.full(n_each, ymin)])
    top = np.column_stack([rng.uniform(xmin, xmax, n_each), np.full(n_each, ymax)])
    left = np.column_stack([np.full(n_each, xmin), rng.uniform(ymin, ymax, n_each)])
    right = np.column_stack([np.full(n_each, xmax), rng.uniform(ymin, ymax, n_each)])
    points = np.vstack([bottom, top, left, right]).astype(np.float32)
    values = np.full((len(points), 1), config.boundary.world_voltage, dtype=np.float32)
    return points, values


def sample_electrode_boundaries(
    config: ExperimentConfig,
    n_per_electrode: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    values = []
    voltages = electrode_voltages(config.boundary)
    for name, (cx, cy, radius) in electrode_centers(config.geometry).items():
        theta = rng.uniform(0.0, 2.0 * np.pi, n_per_electrode)
        circle = np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
        points.append(circle)
        values.append(np.full((n_per_electrode, 1), voltages[name], dtype=np.float32))
    return np.vstack(points).astype(np.float32), np.vstack(values).astype(np.float32)


def training_batch(config: ExperimentConfig, rng: np.random.Generator, device: str = "cpu"):
    t = config.training
    interior = torch.tensor(
        sample_interior(config, t.n_interior, rng),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    world_points, world_values = sample_world_boundary(config, t.n_world, rng)
    electrode_points, electrode_values = sample_electrode_boundaries(
        config,
        max(1, t.n_electrode // 4),
        rng,
    )
    return (
        interior,
        torch.tensor(world_points, dtype=torch.float32, device=device),
        torch.tensor(world_values, dtype=torch.float32, device=device),
        torch.tensor(electrode_points, dtype=torch.float32, device=device),
        torch.tensor(electrode_values, dtype=torch.float32, device=device),
    )

