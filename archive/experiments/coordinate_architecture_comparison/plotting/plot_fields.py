"""Per-model spatial plots: residual displacement field and error heatmap.

``plot_displacement_field`` is the analogue of CPN.py's
``displacement_field.png`` (quiver of ``phi(z) - z`` over a colormap of its
magnitude). ``plot_error_heatmap`` shows where the predicted coordinate error
is largest, which for the identity target coincides with displacement
magnitude but is kept separate so it stays meaningful if the target changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

from ..config import ExperimentConfig
from ..training.data import electrode_centers, grid_points
from ..training.metrics import spatial_error_grid


def _add_geometry(ax, config: ExperimentConfig) -> None:
    xmin, xmax, ymin, ymax = config.domain_bounds()
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1.5))
    for cx, cy, r in electrode_centers(config).values():
        ax.add_patch(Circle((cx, cy), r, fill=False, linewidth=1.5))


@torch.no_grad()
def plot_displacement_field(
    path: str | Path, config: ExperimentConfig, model: torch.nn.Module, device: str
) -> None:
    grid_x, grid_y, mask = grid_points(config)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    predicted = model(torch.as_tensor(points, dtype=torch.float32, device=device)).cpu().numpy()

    u = np.where(mask, np.nan, (predicted[:, 0] - points[:, 0]).reshape(grid_x.shape))
    v = np.where(mask, np.nan, (predicted[:, 1] - points[:, 1]).reshape(grid_x.shape))
    magnitude = np.ma.masked_invalid(np.sqrt(u**2 + v**2))
    stride = config.evaluation.quiver_stride

    fig, ax = plt.subplots()
    contour = ax.contourf(grid_x, grid_y, magnitude, levels=100)
    fig.colorbar(contour, ax=ax, label=r"$|\phi(z) - z|$")
    ax.quiver(
        grid_x[::stride, ::stride], grid_y[::stride, ::stride],
        u[::stride, ::stride], v[::stride, ::stride],
        angles="xy", scale_units="xy", scale=1, width=0.002,
    )
    _add_geometry(ax, config)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Residual displacement field ({config.model.kind})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def plot_error_heatmap(
    path: str | Path, config: ExperimentConfig, model: torch.nn.Module, device: str
) -> None:
    grid_x, grid_y, mask = grid_points(config)
    error = spatial_error_grid(model, grid_x, grid_y, mask, device)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(grid_x, grid_y, np.ma.masked_invalid(error), shading="auto")
    fig.colorbar(mesh, ax=ax, label="Euclidean coordinate error")
    _add_geometry(ax, config)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Spatial coordinate error ({config.model.kind})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
