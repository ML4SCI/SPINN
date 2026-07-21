"""Figures for a shape-optimization run.

All plots use the Agg backend and write PNGs into a run directory. Geometry is
drawn with the same circular-electrode layout the model was trained on so the
learned deformation can be read against the reference shape.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

from PINN.Coordinate_Projection.operators import map_jacobian

from .config import ExperimentConfig
from .geometry import build_geometry
from .pseudopotential import psi_ideal, psi_model

from benchmark.experiments.instructor_coordinate_projection.model import TrapPointSampler


def _grid(config: ExperimentConfig, nx: int = 200, ny: int = 160):
    xmin, xmax, ymin, ymax = config.domain_bounds()
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    return grid_x, grid_y, points


def _electrode_mask(config: ExperimentConfig, points: np.ndarray) -> np.ndarray:
    geometry = build_geometry(config.geometry)
    inside = np.zeros(len(points), dtype=bool)
    for cx, cy, r in geometry.electrodes().values():
        inside |= (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 <= r**2
    return inside


def _draw_electrodes(ax, config: ExperimentConfig) -> None:
    geometry = build_geometry(config.geometry)
    for cx, cy, r in geometry.electrodes().values():
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="black", linewidth=0.8))


def _heatmap(ax, grid_x, grid_y, values, mask, title, cmap="viridis"):
    field = np.where(mask, np.nan, values).reshape(grid_x.shape)
    mesh = ax.pcolormesh(grid_x, grid_y, np.ma.masked_invalid(field), shading="auto", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title(title)
    return mesh


def save_all(result: dict, config: ExperimentConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model = result["model"]
    k = result["k_quadratic_eV_per_m2"]
    grid_x, grid_y, points = _grid(config)
    mask = _electrode_mask(config, points)

    reference = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential, physical = model(reference)
    model_psi = psi_model(potential, reference, model.map, config.physics)
    ideal_psi = psi_ideal(physical.detach(), k, config.physics)
    _, jacobian = map_jacobian(model.map, reference)
    determinant = torch.linalg.det(jacobian)

    potential_np = potential.detach().numpy().ravel()
    model_psi_np = model_psi.detach().numpy().ravel()
    ideal_psi_np = ideal_psi.detach().numpy().ravel()
    determinant_np = determinant.detach().numpy().ravel()

    _save_potential(grid_x, grid_y, potential_np, mask, config, run_dir)
    _save_pseudopotentials(grid_x, grid_y, model_psi_np, ideal_psi_np, mask, config, run_dir)
    _save_pseudopotential_error(grid_x, grid_y, model_psi_np, ideal_psi_np, mask, config, run_dir)
    _save_determinant(grid_x, grid_y, determinant_np, mask, config, run_dir)
    _save_deformation(model, config, run_dir)
    _save_loss_curves(result["history"], run_dir)


def _save_potential(grid_x, grid_y, potential, mask, config, run_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    mesh = _heatmap(ax, grid_x, grid_y, potential, mask, r"Physical potential $\phi(x)$ [V]")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    _draw_electrodes(ax, config)
    fig.tight_layout()
    fig.savefig(run_dir / "potential.png", dpi=150)
    plt.close(fig)


def _save_pseudopotentials(grid_x, grid_y, model_psi, ideal_psi, mask, config, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    vmax = float(np.nanpercentile(np.where(mask, np.nan, model_psi), 99))
    mesh0 = _heatmap(axes[0], grid_x, grid_y, model_psi, mask, r"$\Psi_{\mathrm{model}}$ [eV]")
    mesh0.set_clim(0, vmax)
    fig.colorbar(mesh0, ax=axes[0], fraction=0.046, pad=0.04)
    mesh1 = _heatmap(axes[1], grid_x, grid_y, ideal_psi, mask, r"$\Psi_{\mathrm{ideal}}=k(x^2+y^2)$ [eV]")
    mesh1.set_clim(0, vmax)
    fig.colorbar(mesh1, ax=axes[1], fraction=0.046, pad=0.04)
    for ax in axes:
        _draw_electrodes(ax, config)
    fig.tight_layout()
    fig.savefig(run_dir / "pseudopotential.png", dpi=150)
    plt.close(fig)


def _save_pseudopotential_error(grid_x, grid_y, model_psi, ideal_psi, mask, config, run_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    mesh = _heatmap(ax, grid_x, grid_y, model_psi - ideal_psi, mask,
                    r"$\Psi_{\mathrm{model}}-\Psi_{\mathrm{ideal}}$ [eV]", cmap="coolwarm")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    _draw_electrodes(ax, config)
    fig.tight_layout()
    fig.savefig(run_dir / "pseudopotential_error.png", dpi=150)
    plt.close(fig)


def _save_determinant(grid_x, grid_y, determinant, mask, config, run_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    mesh = _heatmap(ax, grid_x, grid_y, determinant, mask, r"Jacobian $\det J$", cmap="coolwarm")
    mesh.set_clim(0, 2)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    _draw_electrodes(ax, config)
    fig.tight_layout()
    fig.savefig(run_dir / "jacobian_determinant.png", dpi=150)
    plt.close(fig)


def _save_deformation(model, config, run_dir):
    geometry = build_geometry(config.geometry)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    theta = np.linspace(0, 2 * np.pi, 200)
    for cx, cy, r in geometry.electrodes().values():
        circle = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]).astype(np.float32)
        with torch.no_grad():
            mapped = model.map(torch.tensor(circle)).numpy()
        ax.plot(circle[:, 0], circle[:, 1], color="0.6", linestyle="--", linewidth=1.0)
        ax.plot(mapped[:, 0], mapped[:, 1], color="C3", linewidth=1.4)
    ax.set_aspect("equal")
    ax.set_title("Electrode deformation (dashed=reference, red=mapped)")
    fig.tight_layout()
    fig.savefig(run_dir / "deformation.png", dpi=150)
    plt.close(fig)


def _save_loss_curves(history, run_dir):
    if not history:
        return
    steps = [row["step"] for row in history]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for key in ("total", "pde", "pseudo", "bc"):
        ax.semilogy(steps, [row[key] for row in history], marker="o", label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "losses.png", dpi=150)
    plt.close(fig)
