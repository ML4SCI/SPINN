"""The instructor-generated coordinate projection model, as an importable module.

The network architecture, the residual form ``NN_phi(z) = z + NNhat_phi(z)``,
the mean-Euclidean-distance identity loss, and the zero-last-layer option are
taken verbatim from the instructor script ``PINN/Coordinate_Projection_Network/
CPN.py``. Only the script-level execution, global configuration, and plotting
have been removed so that importing this module has no side effects. See the
experiment README for how this differs from the existing coordinate-projection
PINN.

This model maps coordinates to coordinates. It has no potential field and no
electrode-distance parameter, so it does not by itself produce the shape
derivative ``dJ/dd``; it is benchmarked on identity fidelity and on the
Jacobian of its map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class TrapGeometry:
    """Reference Paul-trap geometry the map is trained and evaluated on."""

    world_size_x: float = 2.5
    world_size_y: float = 2.0
    radius: float = 0.5e-1
    north_distance: float = 1.25e-1
    south_distance: float = 1.25e-1
    west_distance: float = 1.25e-1
    east_distance: float = 1.25e-1

    def bounds(self) -> tuple[float, float, float, float]:
        return (
            -self.world_size_x / 2,
            self.world_size_x / 2,
            -self.world_size_y / 2,
            self.world_size_y / 2,
        )

    def electrodes(self) -> dict[str, tuple[float, float, float]]:
        return {
            "North": (0.0, self.north_distance, self.radius),
            "South": (0.0, -self.south_distance, self.radius),
            "West": (-self.west_distance, 0.0, self.radius),
            "East": (self.east_distance, 0.0, self.radius),
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 2, width: int = 50, depth: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoordinateProjectionNet(nn.Module):
    """Residual coordinate map ``NN_phi(z) = z + NNhat_phi(z)``.

    The network learns a displacement field; for the identity map the residual
    should approach zero.
    """

    def __init__(self, width: int = 50, depth: int = 6, zero_last_layer: bool = False) -> None:
        super().__init__()
        self.displacement = MLP(2, 2, width=width, depth=depth)
        if zero_last_layer:
            last = self.displacement.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.displacement(z)


class TrapPointSampler:
    """Sample reference points on the trap interior and boundaries.

    Mirrors the instructor sampler. Interior points exclude electrode disks, so
    the map is only trained where the field problem is actually defined.
    """

    def __init__(self, geometry: TrapGeometry, rng: np.random.Generator) -> None:
        self.geometry = geometry
        self.rng = rng

    def _inside_any_electrode(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        inside = np.zeros(len(xy), dtype=bool)
        for cx, cy, r in self.geometry.electrodes().values():
            inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        return inside

    def interior(self, n: int) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.geometry.bounds()
        kept: list[list[float]] = []
        while len(kept) < n:
            xy = np.column_stack([
                self.rng.uniform(xmin, xmax, n),
                self.rng.uniform(ymin, ymax, n),
            ])
            xy = xy[~self._inside_any_electrode(xy)]
            kept.extend(xy.tolist())
        return np.array(kept[:n], dtype=np.float32)

    def world_boundary(self, n: int) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.geometry.bounds()
        n_each = n // 4
        bottom = np.column_stack([self.rng.uniform(xmin, xmax, n_each), np.full(n_each, ymin)])
        top = np.column_stack([self.rng.uniform(xmin, xmax, n_each), np.full(n_each, ymax)])
        left = np.column_stack([np.full(n_each, xmin), self.rng.uniform(ymin, ymax, n_each)])
        right = np.column_stack([np.full(n_each, xmax), self.rng.uniform(ymin, ymax, n_each)])
        return np.vstack([bottom, top, left, right]).astype(np.float32)

    def electrode_boundaries(self, n_per_electrode: int) -> np.ndarray:
        pts = []
        for cx, cy, r in self.geometry.electrodes().values():
            theta = self.rng.uniform(0, 2 * np.pi, n_per_electrode)
            pts.append(np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]))
        return np.vstack(pts).astype(np.float32)

    def training_batch(self, n_interior: int, n_world: int, n_electrode: int) -> np.ndarray:
        return np.vstack([
            self.interior(n_interior),
            self.world_boundary(n_world),
            self.electrode_boundaries(n_electrode),
        ]).astype(np.float32)


def mean_identity_distance_loss(model: nn.Module, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean Euclidean distance between ``NN_phi(z)`` and ``z`` (instructor loss)."""

    x_pred = model(z)
    distance = torch.sqrt(torch.sum((x_pred - z) ** 2, dim=1) + eps)
    return torch.mean(distance)


def rms_identity_error(model: nn.Module, z: torch.Tensor) -> float:
    with torch.no_grad():
        x_pred = model(z)
        return float(torch.sqrt(torch.mean(torch.sum((x_pred - z) ** 2, dim=1))))


@dataclass
class TrainingConfig:
    steps: int = 10000
    lr: float = 1e-3
    n_interior: int = 4096
    n_world: int = 1024
    n_electrode: int = 2048
    grad_clip_max_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 500
    scheduler_min_lr: float = 1e-6
    history_every: int = 100


@dataclass
class TrainingResult:
    final_rms_identity_error: float
    train_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_identity_map(
    model: CoordinateProjectionNet,
    geometry: TrapGeometry,
    config: TrainingConfig,
    *,
    seed: int = 0,
    device: str = "cpu",
) -> TrainingResult:
    """Train the residual map toward identity using the instructor recipe.

    Uses the same optimizer, plateau scheduler, gradient clipping, and loss as
    the instructor script. Resampling every step matches the original.
    """

    import time

    torch.manual_seed(seed)
    model = model.float()  # training samples are float32 regardless of global default
    rng = np.random.default_rng(seed)
    sampler = TrapPointSampler(geometry, rng)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(config.steps):
        batch = sampler.training_batch(config.n_interior, config.n_world, config.n_electrode)
        z = torch.tensor(batch, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        loss = mean_identity_distance_loss(model, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
        optimizer.step()
        scheduler.step(loss.item())

        if step % config.history_every == 0:
            history.append({
                "step": step,
                "loss": float(loss.item()),
                "rms_identity_error": rms_identity_error(model, z),
                "lr": optimizer.param_groups[0]["lr"],
            })

    train_seconds = time.perf_counter() - start
    final_batch = sampler.training_batch(config.n_interior, config.n_world, config.n_electrode)
    final_z = torch.tensor(final_batch, dtype=torch.float32, device=device)
    return TrainingResult(
        final_rms_identity_error=rms_identity_error(model, final_z),
        train_seconds=train_seconds,
        history=history,
    )
