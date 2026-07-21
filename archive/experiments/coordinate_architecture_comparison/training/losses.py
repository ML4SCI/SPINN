"""Coordinate-projection losses.

``MeanEuclideanCoordinateLoss`` is the default training loss and matches
``mean_identity_distance_loss`` in CPN.py (mean L2 distance between predicted
and target coordinates). ``CoordinateMSELoss`` is the mean squared coordinate
error, used for evaluation and for the MSE comparison plots.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanEuclideanCoordinateLoss(nn.Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared = torch.sum((predicted - target) ** 2, dim=1)
        return torch.mean(torch.sqrt(squared + self.eps))


class CoordinateMSELoss(nn.Module):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum((predicted - target) ** 2, dim=1))
