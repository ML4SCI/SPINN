"""Per-run diagnostic plots from a saved joint model (plan sec. 6)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from ..config import load_config
from ..geometry.export_fem import deformed_electrode_polygons, reference_geometry
from ..geometry.four_rod import FourRodTrap, build_trap
from ..models.backends import PigBackend, PixelBackend, build_backend
from ..models.composed import ComposedField
from ..models.shape_network import build_shape_network
from ..physics.laplace import cartesian_laplacian, map_jacobian


def load_model(config, model_path: Path) -> ComposedField:
    trap = build_trap(config)
    model = ComposedField(
        build_shape_network(config, trap), build_backend(config)
    ).float()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _square_grid(
    trap: FourRodTrap, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-trap.outer_radius, trap.outer_radius, n)
    gx, gy = np.meshgrid(axis, axis)
    return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])


def _draw_rods(ax, trap: FourRodTrap, style: str) -> None:
    for e in trap.electrodes():
        ring = e.boundary(np.linspace(0, 2 * np.pi, 100))
        ax.plot(ring[:, 0], ring[:, 1], style, lw=0.6)


def _finish(fig, ax, path: Path, title: str, coords: str = "x") -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("z1 / r0" if coords == "z" else "x / r0")
    ax.set_ylabel("z2 / r0" if coords == "z" else "y / r0")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def potential_and_residual(model, trap, out: Path, name: str) -> None:
    gx, gy, pts = _square_grid(trap, 200)
    outside = ~trap.in_vacuum(pts, margin=1e-3)
    x = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
    v = model.backend(x)
    residual = cartesian_laplacian(v, x).detach().numpy().reshape(-1) ** 2
    field = v.detach().numpy().reshape(-1)
    field[outside] = np.nan
    residual[outside] = np.nan

    fig, ax = plt.subplots(figsize=(5.5, 5))
    pc = ax.contourf(gx, gy, field.reshape(gx.shape), levels=40, cmap="RdBu_r")
    fig.colorbar(pc, ax=ax, label="V")
    _finish(fig, ax, out / f"potential_{name}.png", f"Potential - {name.upper()}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    pc = ax.contourf(
        gx, gy, np.log10(residual.reshape(gx.shape) + 1e-20), levels=40, cmap="magma"
    )
    fig.colorbar(pc, ax=ax, label=r"$\log_{10}|\Delta V|^2$")
    _finish(fig, ax, out / f"residual_{name}.png", f"PDE residual - {name.upper()}")


def deformation_and_detj(model, trap, out: Path, name: str) -> None:
    gx, gy, pts = _square_grid(trap, 26)
    z = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
    x, jac = map_jacobian(model.shape_network, z)
    disp = (x - z).detach().numpy()
    detj = torch.linalg.det(jac).detach().numpy().reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.quiver(pts[:, 0], pts[:, 1], disp[:, 0], disp[:, 1], angles="xy")
    _draw_rods(ax, trap, "k-")
    _finish(
        fig,
        ax,
        out / f"deformation_{name}.png",
        f"Deformation field - {name.upper()}",
        "z",
    )

    fig, ax = plt.subplots(figsize=(5.5, 5))
    pc = ax.contourf(gx, gy, detj, levels=40, cmap="coolwarm")
    fig.colorbar(pc, ax=ax, label=r"$\det J_\phi$")
    ax.contour(gx, gy, detj, levels=[0.0], colors="black", linewidths=1.0)
    _finish(fig, ax, out / f"detJ_{name}.png", f"det(J_phi) - {name.upper()}", "z")


def contours(model, trap, out: Path, name: str) -> None:
    initial = reference_geometry(trap).electrodes
    optimized = deformed_electrode_polygons(model.shape_network, trap).electrodes
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for index, ((poly0, _), (poly1, _)) in enumerate(zip(initial, optimized)):
        c0 = np.vstack([poly0, poly0[:1]])
        c1 = np.vstack([poly1, poly1[:1]])
        ax.plot(c0[:, 0], c0[:, 1], "b--", lw=1, label="initial" if index == 0 else "")
        ax.plot(
            c1[:, 0], c1[:, 1], "r-", lw=1.5, label="optimized" if index == 0 else ""
        )
    outer = trap.outer_radius * np.column_stack(
        [np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200))]
    )
    ax.plot(outer[:, 0], outer[:, 1], "k-", lw=0.8)
    ax.legend()
    _finish(
        fig,
        ax,
        out / f"contours_{name}.png",
        f"Initial vs optimized electrodes - {name.upper()}",
    )


def _pig_diagnostics(backend: PigBackend, trap, out: Path) -> None:
    diag = backend.gaussian_diagnostics()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(
        diag["mu_x"], diag["mu_y"], c=diag["feature_norm"], s=8, cmap="viridis"
    )
    fig.colorbar(sc, ax=ax, label="feature norm")
    _draw_rods(ax, trap, "r-")
    _finish(fig, ax, out / "pig_gaussian_centers.png", "PIG Gaussian centers")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    strongest = np.argsort(diag["feature_norm"])[-120:]
    for idx in strongest:
        ax.add_patch(
            Ellipse(
                (diag["mu_x"][idx], diag["mu_y"][idx]),
                width=2.0 * diag["sigma_x"][idx],
                height=2.0 * diag["sigma_y"][idx],
                facecolor="none",
                edgecolor="tab:green",
                alpha=0.35,
                lw=0.7,
            )
        )
    ax.scatter(
        diag["mu_x"][strongest],
        diag["mu_y"][strongest],
        s=5,
        color="tab:green",
        alpha=0.7,
    )
    _draw_rods(ax, trap, "r-")
    ax.set_xlim(-trap.outer_radius, trap.outer_radius)
    ax.set_ylim(-trap.outer_radius, trap.outer_radius)
    _finish(
        fig,
        ax,
        out / "pig_gaussian_covariances.png",
        "PIG Gaussian covariance ellipses",
    )


def _pixel_diagnostics(backend: PixelBackend, trap, out: Path) -> None:
    R = trap.outer_radius
    fig, ax = plt.subplots(figsize=(5.5, 5))
    pc = ax.imshow(
        backend.feature_norm_grid().cpu().numpy(),
        origin="lower",
        cmap="viridis",
        extent=[-R, R, -R, R],
    )
    fig.colorbar(pc, ax=ax, label="feature norm")
    _finish(fig, ax, out / "pixel_feature_activity.png", "PIXEL feature activity")


def evaluate_run(config, model_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    trap = build_trap(config)
    model = load_model(config, model_path)
    name = config.physics_backend.type
    potential_and_residual(model, trap, out, name)
    deformation_and_detj(model, trap, out, name)
    contours(model, trap, out, name)
    if isinstance(model.backend, PigBackend):
        _pig_diagnostics(model.backend, trap, out)
    elif isinstance(model.backend, PixelBackend):
        _pixel_diagnostics(model.backend, trap, out)
    print(f"[evaluate] wrote diagnostic plots for {name} to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    evaluate_run(load_config(args.config), Path(args.model), Path(args.out))


if __name__ == "__main__":
    main()
