"""PIG physics backend: official-style channel-wise Gaussian features.

This follows the diagonal-covariance path in
``external/Physics-Informed-Gaussians/Helmholtz2d/network.py``. PIG builds
``feature_dim`` independent Gaussian channels; each channel has ``K`` learnable
centers, diagonal sigmas, and scalar weights. Each channel sums its Gaussians,
then a small MLP decodes the resulting ``feature_dim`` vector to the scalar
field. The only SPINN adaptation is mapping Paul-trap physical coordinates from
``[-R_out, R_out]^2`` into the reference code's ``[-1, 1]^2`` coordinates.
"""

from __future__ import annotations

import torch
from torch import nn

from .decoders import DecoderMLP


class PigBackend(nn.Module):
    kind = "pig"

    def __init__(
        self,
        *,
        outer_radius: float,
        num_gaussians: int = 800,
        feature_dim: int = 4,
        init_sigma: float = 0.10,
        decoder_layers: int = 2,
        decoder_width: int = 16,
        symmetry: str = "none",
    ) -> None:
        super().__init__()
        bound = float(outer_radius)
        self.register_buffer("lower", torch.tensor([-bound, -bound], dtype=torch.float32))
        self.register_buffer("upper", torch.tensor([bound, bound], dtype=torch.float32))
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.symmetry = "d4_quadrupole" if symmetry == "d4" else symmetry
        supported = {"none", "x_mirror", "y_mirror", "xy_mirror", "d4_quadrupole"}
        if self.symmetry not in supported:
            raise ValueError(f"unknown PIG symmetry {symmetry!r}; choose from {sorted(supported | {'d4'})}")
        self.means = nn.Parameter(torch.empty(feature_dim, num_gaussians, 2).uniform_(-1.0, 1.0))
        self.sigmas = nn.Parameter(torch.ones(feature_dim, num_gaussians, 2) * init_sigma)
        self.weights = nn.Parameter(torch.normal(0.0, 0.01, (feature_dim, num_gaussians)))
        self.decoder = DecoderMLP(feature_dim, layers=decoder_layers, width=decoder_width)

    def _pig_coords(self, points: torch.Tensor) -> torch.Tensor:
        unit = (points - self.lower) / (self.upper - self.lower)
        return 2.0 * unit - 1.0

    def features(self, points: torch.Tensor) -> torch.Tensor:
        coords = self._pig_coords(points)
        # Official layout: (feature_dim, points, gaussians, xy)
        diff = coords.unsqueeze(0).unsqueeze(2) - self.means.unsqueeze(1)
        exponent = torch.sum((diff / self.sigmas.unsqueeze(1)) ** 2, dim=-1)
        gaussians = torch.exp(torch.clamp(-0.5 * exponent, min=-60.0, max=0.0))
        return (gaussians * self.weights.unsqueeze(1)).sum(dim=-1).T

    def raw_forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.features(points))

    def _axis_even(self, points: torch.Tensor) -> torch.Tensor:
        reflected_x = points * points.new_tensor([-1.0, 1.0])
        reflected_y = points * points.new_tensor([1.0, -1.0])
        return 0.25 * (
            self.raw_forward(points)
            + self.raw_forward(reflected_x)
            + self.raw_forward(reflected_y)
            + self.raw_forward(-points)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if self.symmetry == "none":
            return self.raw_forward(points)
        if self.symmetry == "x_mirror":
            reflected = points * points.new_tensor([-1.0, 1.0])
            return 0.5 * (self.raw_forward(points) + self.raw_forward(reflected))
        if self.symmetry == "y_mirror":
            reflected = points * points.new_tensor([1.0, -1.0])
            return 0.5 * (self.raw_forward(points) + self.raw_forward(reflected))
        even = self._axis_even(points)
        if self.symmetry == "xy_mirror":
            return even
        swapped_even = self._axis_even(points[:, [1, 0]])
        return 0.5 * (even - swapped_even)

    def gaussian_diagnostics(self) -> dict:
        """Centers and sigmas in physical coordinates for all channel/Gaussian pairs."""
        with torch.no_grad():
            span = (self.upper - self.lower).cpu()
            lower = self.lower.cpu()
            mu = ((self.means.detach().cpu() + 1.0) / 2.0) * span + lower
            sigma = self.sigmas.detach().cpu().abs() * (span / 2.0)
            feature_norm = self.weights.detach().cpu().abs()
        return {
            "mu_x": mu[..., 0].reshape(-1).numpy(),
            "mu_y": mu[..., 1].reshape(-1).numpy(),
            "sigma_x": sigma[..., 0].reshape(-1).numpy(),
            "sigma_y": sigma[..., 1].reshape(-1).numpy(),
            "feature_norm": feature_norm.reshape(-1).numpy(),
        }

    def param_groups(
        self,
        lr_main: float,
        *,
        lr_centers: float | None = None,
        lr_covariances: float | None = None,
        **_unused,
    ) -> list[dict]:
        lr_centers = lr_main if lr_centers is None else lr_centers
        lr_covariances = lr_main if lr_covariances is None else lr_covariances
        decoder_and_weights = list(self.decoder.parameters()) + [self.weights]
        return [
            {"params": decoder_and_weights, "lr": lr_main},
            {"params": [self.means], "lr": lr_centers},
            {"params": [self.sigmas], "lr": lr_covariances},
        ]
