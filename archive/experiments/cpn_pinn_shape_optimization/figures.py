"""Notebook-facing figure builders for a shape-optimization run.

Each function trains (or re-trains) a variation and returns a single matplotlib
Figure, so a notebook cell can render one plot by calling one function. The
heavy lifting (pullback pseudopotential, Jacobian) is reused from the experiment
modules; these functions only arrange it onto axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from PINN.Coordinate_Projection.operators import map_jacobian

from .config import ExperimentConfig, make_preset
from .geometry import build_geometry
from .pseudopotential import psi_ideal, psi_model
from .train import train


@dataclass
class RunArtifacts:
    """A trained run plus the fields a notebook needs to plot it."""

    config: ExperimentConfig
    result: dict
    grid_x: np.ndarray
    grid_y: np.ndarray
    mask: np.ndarray
    potential: np.ndarray
    psi_model: np.ndarray
    psi_ideal: np.ndarray
    det_jacobian: np.ndarray


def run_variation(variation: str = "C", steps: int = 2000, seed: int = 0) -> RunArtifacts:
    """Train one variation and evaluate the fields on a dense grid."""

    torch.set_default_dtype(torch.float32)
    config = make_preset(variation)
    config.seed = seed
    config.training.steps = steps
    result = train(config)
    return _evaluate_on_grid(config, result)


def _evaluate_on_grid(config: ExperimentConfig, result: dict, nx: int = 200, ny: int = 160) -> RunArtifacts:
    model = result["model"]
    k = result["k_quadratic_eV_per_m2"]
    xmin, xmax, ymin, ymax = config.domain_bounds()
    grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

    geometry = build_geometry(config.geometry)
    mask = np.zeros(len(points), dtype=bool)
    for cx, cy, r in geometry.electrodes().values():
        mask |= (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 <= r**2

    reference = torch.tensor(points, requires_grad=True)
    potential, physical = model(reference)
    model_psi = psi_model(potential, reference, model.map, config.physics)
    ideal_psi = psi_ideal(physical.detach(), k, config.physics)
    _, jacobian = map_jacobian(model.map, reference)
    determinant = torch.linalg.det(jacobian)

    return RunArtifacts(
        config=config,
        result=result,
        grid_x=grid_x,
        grid_y=grid_y,
        mask=mask.reshape(grid_x.shape),
        potential=potential.detach().numpy().reshape(grid_x.shape),
        psi_model=model_psi.detach().numpy().reshape(grid_x.shape),
        psi_ideal=ideal_psi.detach().numpy().reshape(grid_x.shape),
        det_jacobian=determinant.detach().numpy().reshape(grid_x.shape),
    )


def _draw_electrodes(ax, config: ExperimentConfig) -> None:
    for cx, cy, r in build_geometry(config.geometry).electrodes().values():
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="black", linewidth=0.8))


def _masked(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.ma.masked_invalid(np.where(mask, np.nan, field))


def plot_potential(run: RunArtifacts) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    mesh = ax.pcolormesh(run.grid_x, run.grid_y, _masked(run.potential, run.mask), shading="auto")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    _draw_electrodes(ax, run.config)
    ax.set_aspect("equal")
    ax.set_title(r"Physical potential $\phi(x)$ [V]")
    return fig


def plot_pseudopotential(run: RunArtifacts) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    vmax = float(np.nanpercentile(_masked(run.psi_model, run.mask).filled(np.nan), 99))
    for ax, field, title in (
        (axes[0], run.psi_model, r"$\Psi_{\mathrm{model}}$ [eV]"),
        (axes[1], run.psi_ideal, r"$\Psi_{\mathrm{ideal}}=k(x^2+y^2)$ [eV]"),
    ):
        mesh = ax.pcolormesh(run.grid_x, run.grid_y, _masked(field, run.mask), shading="auto")
        mesh.set_clim(0, vmax)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        _draw_electrodes(ax, run.config)
        ax.set_aspect("equal")
        ax.set_title(title)
    return fig


def plot_pseudopotential_error(run: RunArtifacts) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    mesh = ax.pcolormesh(
        run.grid_x, run.grid_y, _masked(run.psi_model - run.psi_ideal, run.mask),
        shading="auto", cmap="coolwarm",
    )
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    _draw_electrodes(ax, run.config)
    ax.set_aspect("equal")
    ax.set_title(r"$\Psi_{\mathrm{model}}-\Psi_{\mathrm{ideal}}$ [eV]")
    return fig


def plot_jacobian_determinant(run: RunArtifacts) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    mesh = ax.pcolormesh(
        run.grid_x, run.grid_y, _masked(run.det_jacobian, run.mask),
        shading="auto", cmap="coolwarm",
    )
    mesh.set_clim(-1, 2)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    _draw_electrodes(ax, run.config)
    ax.set_aspect("equal")
    minimum = float(np.nanmin(run.det_jacobian))
    ax.set_title(rf"Jacobian $\det J$ (min ${minimum:.2f}$; folds where $<0$)")
    return fig


def plot_deformation(run: RunArtifacts) -> Figure:
    model = run.result["model"]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    theta = np.linspace(0, 2 * np.pi, 200)
    for cx, cy, r in build_geometry(run.config.geometry).electrodes().values():
        circle = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]).astype(np.float32)
        with torch.no_grad():
            mapped = model.map(torch.tensor(circle)).numpy()
        ax.plot(circle[:, 0], circle[:, 1], color="0.6", linestyle="--", linewidth=1.0)
        ax.plot(mapped[:, 0], mapped[:, 1], color="C3", linewidth=1.4)
    ax.set_aspect("equal")
    ax.set_title("Electrode deformation (dashed=reference, red=mapped)")
    return fig


def plot_loss_curves(run: RunArtifacts) -> Figure:
    history = run.result["history"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    steps = [row["step"] for row in history]
    for key in ("total", "pde", "pseudo", "bc"):
        ax.semilogy(steps, [row[key] for row in history], marker="o", label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return fig
