from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .four_rod import FourRodTrap


@dataclass
class DeformedGeometry:
    outer_radius: float
    V0: float
    # one entry per electrode: (closed polygon (M,2), voltage)
    electrodes: list[tuple[np.ndarray, float]]


def deformed_electrode_polygons(
    shape_network,
    trap: FourRodTrap,
    n_per_electrode: int = 400,
    *,
    device="cpu",
    dtype=torch.float32,
) -> DeformedGeometry:
    polygons: list[tuple[np.ndarray, float]] = []
    theta = np.linspace(0.0, 2.0 * np.pi, n_per_electrode, endpoint=False)
    for e in trap.electrodes():
        ref = np.column_stack([e.cx + e.radius * np.cos(theta), e.cy + e.radius * np.sin(theta)])
        with torch.no_grad():
            z = torch.as_tensor(ref, device=device, dtype=dtype)
            x = shape_network(z).cpu().numpy()
        polygons.append((x, trap.electrode_voltage(e)))
    return DeformedGeometry(trap.outer_radius, trap.V0, polygons)


def reference_geometry(trap: FourRodTrap, n_per_electrode: int = 400) -> DeformedGeometry:
    """Undeformed geometry (identity map) for sanity/baseline FEM solves."""
    polygons: list[tuple[np.ndarray, float]] = []
    theta = np.linspace(0.0, 2.0 * np.pi, n_per_electrode, endpoint=False)
    for e in trap.electrodes():
        ref = np.column_stack([e.cx + e.radius * np.cos(theta), e.cy + e.radius * np.sin(theta)])
        polygons.append((ref, trap.electrode_voltage(e)))
    return DeformedGeometry(trap.outer_radius, trap.V0, polygons)
