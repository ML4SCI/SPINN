"""Electrode polygons handed to the finite-difference validator."""

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
    """Rod surfaces pushed through ``shape_network`` (``None`` = identity map)."""
    polygons = []
    for e in trap.electrodes():
        ring = e.ring(n_per_electrode)
        if shape_network is not None:
            with torch.no_grad():
                z = torch.as_tensor(ring, device=device, dtype=dtype)
                ring = shape_network(z).cpu().numpy()
        polygons.append((ring, trap.electrode_voltage(e)))
    return DeformedGeometry(trap.outer_radius, trap.V0, polygons)


def reference_geometry(
    trap: FourRodTrap, n_per_electrode: int = 400
) -> DeformedGeometry:
    """Undeformed geometry (identity map) for sanity/baseline FEM solves."""
    return deformed_electrode_polygons(None, trap, n_per_electrode)
