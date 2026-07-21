"""Shared coordinate-projection interface.

Every architecture in this experiment maps a batch of 2D reference
coordinates ``z`` to a batch of 2D predicted coordinates and preserves the
residual structure of Zhang et al.,

    phi(z) = z + phi_hat(z),

so a subclass only implements the learned displacement ``phi_hat``. At
identity initialization the displacement should be (close to) zero and the
model passes the input through.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CoordinateProjectionModel(nn.Module, ABC):
    @abstractmethod
    def displacement(self, z: torch.Tensor) -> torch.Tensor:
        """Return the learned residual ``phi_hat(z)`` with the same shape as ``z``."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.displacement(z)
