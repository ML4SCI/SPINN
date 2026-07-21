"""Geometry and sampling, wrapping the instructor module's trap utilities.

The reference circular trap, electrode layout, and point sampler are reused
directly from the instructor coordinate-projection module so this experiment
operates on an identical domain. Electrode boundary voltages follow the project
convention: East/West driven at the RF voltage, North/South and the world
boundary grounded.
"""

from __future__ import annotations

import numpy as np

from benchmark.experiments.instructor_coordinate_projection.model import (
    TrapGeometry,
    TrapPointSampler,
)

from .config import GeometryConfig, PhysicsConfig


def build_geometry(config: GeometryConfig) -> TrapGeometry:
    return TrapGeometry(
        world_size_x=config.world_size_x,
        world_size_y=config.world_size_y,
        radius=config.radius,
        north_distance=config.north_distance,
        south_distance=config.south_distance,
        west_distance=config.west_distance,
        east_distance=config.east_distance,
    )


def build_sampler(geometry: TrapGeometry, seed: int) -> TrapPointSampler:
    return TrapPointSampler(geometry, np.random.default_rng(seed))


def electrode_boundary_points_and_voltages(
    geometry: TrapGeometry,
    physics: PhysicsConfig,
    n_per_electrode: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Electrode boundary samples with their prescribed RF voltages.

    East/West are driven; North/South are grounded. Returns ``(points, values)``
    with ``values`` shaped ``(n, 1)``.
    """

    points = []
    values = []
    driven = {"East", "West"}
    for name, (cx, cy, r) in geometry.electrodes().items():
        theta = rng.uniform(0.0, 2.0 * np.pi, n_per_electrode)
        circle = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])
        points.append(circle)
        voltage = physics.rf_voltage if name in driven else 0.0
        values.append(np.full((n_per_electrode, 1), voltage))
    return (
        np.vstack(points).astype(np.float32),
        np.vstack(values).astype(np.float32),
    )


def world_boundary_points_and_voltages(
    geometry: TrapGeometry,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """World boundary samples held at ground (0 V)."""

    sampler = TrapPointSampler(geometry, rng)
    points = sampler.world_boundary(n)
    values = np.zeros((len(points), 1), dtype=np.float32)
    return points, values
