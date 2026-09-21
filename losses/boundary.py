"""Boundary loss: Dirichlet voltage mismatch on electrodes and outer boundary. Boundary points are prescribed on reference coordinates and pushed through the same composed model, so the condition is enforced on the geometry the map produces. 
"""

from __future__ import annotations

import torch

from ..models.composed import ComposedField
from ..geometry.boundary_sampling import Batch


def boundary_loss(model: ComposedField, batch: Batch) -> torch.Tensor:
    el_v, _ = model(batch.electrode_points)
    out_v, _ = model(batch.outer_points)
    electrode_term = torch.mean((el_v - batch.electrode_voltages) ** 2)
    outer_term = torch.mean((out_v - batch.outer_voltages) ** 2)
    return electrode_term + outer_term
