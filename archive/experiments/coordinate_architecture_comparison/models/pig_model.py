"""PIG-style learnable-Gaussian coordinate projection.

Faithful to the PIG feature embedding (Kang et al., ICLR 2025):

    FE(x) = sum_i f_i * exp(-0.5 * (x - mu_i)^T Sigma_i^{-1} (x - mu_i)),

with learnable means ``mu_i``, diagonal covariances (the paper's efficient
choice), and feature vectors ``f_i``, followed by a single-hidden-layer MLP.
Adapted here so the MLP outputs a 2D displacement instead of a scalar field,
and wrapped in the residual ``z + phi_hat(z)`` structure. Means are
initialized uniformly over the reference domain; covariances start
isotropic. ``log_sigma`` is the learnable parameter so the standard
deviation stays positive under unconstrained gradient descent.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import CoordinateProjectionModel


class PigCoordinateProjection(CoordinateProjectionModel):
    def __init__(
        self,
        domain_bounds: tuple[float, float, float, float],
        num_gaussians: int = 256,
        feature_dim: int = 4,
        hidden_width: int = 16,
        init_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        xmin, xmax, ymin, ymax = domain_bounds
        lower = torch.tensor([xmin, ymin])
        upper = torch.tensor([xmax, ymax])

        means = lower + (upper - lower) * torch.rand(num_gaussians, 2)
        self.means = nn.Parameter(means)
        self.log_sigma = nn.Parameter(torch.full((num_gaussians, 2), math.log(init_sigma)))
        self.features = nn.Parameter(torch.randn(num_gaussians, feature_dim) * 1e-2)

        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _gaussian_weights(self, z: torch.Tensor) -> torch.Tensor:
        diff = z.unsqueeze(1) - self.means.unsqueeze(0)
        inv_var = torch.exp(-2.0 * self.log_sigma)
        exponent = -0.5 * torch.sum(diff**2 * inv_var.unsqueeze(0), dim=-1)
        return torch.exp(exponent)

    def displacement(self, z: torch.Tensor) -> torch.Tensor:
        weights = self._gaussian_weights(z)
        features = weights @ self.features
        return self.head(features)
