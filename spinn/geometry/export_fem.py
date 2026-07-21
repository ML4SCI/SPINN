"""Export the deformed electrode geometry for the FD/FEM validator.

The shape network deforms reference coordinates ``z`` into physical coordinates
``x = NN_phi(z)``. To validate eta independently of the PINN we need the *moved*
electrode boundaries in physical space. Each rod surface is sampled in reference
coordinates and pushed through the map; the resulting closed polygons define the
Dirichlet electrodes for the finite-difference solver.

The outer boundary is anchored by construction (the design mask is ~0 there), so
it stays the reference circle of radius ``R_out``.
"""

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
