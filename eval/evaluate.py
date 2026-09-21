"""Per-run diagnostic plots from a saved joint model (plan sec. 6)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

from ..config import load_config
from ..geometry.export_fem import deformed_electrode_polygons, reference_geometry
from ..geometry.four_rod import build_trap
from ..models.backends import PigBackend, PixelBackend, build_backend
from ..models.composed import ComposedField
from ..models.shape_network import build_shape_network
from ..physics.laplace import cartesian_laplacian, map_jacobian


def _grid(trap, n=200):
    R = trap.outer_radius
    axis = np.linspace(-R, R, n)
    gx, gy = np.meshgrid(axis, axis)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mask = trap.in_vacuum(pts, margin=1e-3)
    return gx, gy, pts, mask


def load_model(config, model_path: Path) -> ComposedField:
    trap = build_trap(config)
    model = ComposedField(build_shape_network(config, trap), build_backend(config)).float()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def potential_and_residual(model, trap, out: Path, backend_name: str) -> None:
    gx, gy, pts, mask = _grid(trap)
    x = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
    v = model.backend(x)
    lap = cartesian_laplacian(v, x).detach().numpy().reshape(-1)
    field = v.detach().numpy().reshape(-1)
    field[~mask] = np.nan
    res = lap**2
    res[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(5.5, 5))
    pc = ax.contourf(gx, gy, field.reshape(gx.shape), levels=40, cmap="RdBu_r")
    fig.colorbar(pc, ax=ax, label="V")
    ax.set_aspect("equal")
    ax.set_xlabel("x / r0")
    ax.set_ylabel("y / r0")
    ax.set_title(f"Potential - {backend_name.upper()}")
    fig.tight_layout()
    fig.savefig(out / f"potential_{backend_name}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    pc = ax.contourf(gx, gy, np.log10(res.reshape(gx.shape) + 1e-20), levels=40, cmap="magma")
    fig.colorbar(pc, ax=ax, label=r"$\log_{10}|\Delta V|^2$")
    ax.set_aspect("equal")
    ax.set_xlabel("x / r0")
    ax.set_ylabel("y / r0")
    ax.set_title(f"PDE residual - {backend_name.upper()}")
    fig.tight_layout()
    fig.savefig(out / f"residual_{backend_name}.png", dpi=300)
    plt.close(fig)


def deformation_and_detj(model, trap, out: Path, backend_name: str) -> None:
    R = trap.outer_radius
    axis = np.linspace(-R, R, 26)
    gx, gy = np.meshgrid(axis, axis)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    z = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
    x, jac = map_jacobian(model.shape_network, z)
    disp = (x - z).detach().numpy()
    detj = torch.linalg.det(jac).detach().numpy()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.quiver(pts[:, 0], pts[:, 1], disp[:, 0], disp[:, 1], angles="xy")
    for e in trap.electrodes():
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(e.cx + e.radius * np.cos(th), e.cy + e.radius * np.sin(th), "k-", lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("z1 / r0")
    ax.set_ylabel("z2 / r0")
    ax.set_title(f"Deformation field - {backend_name.upper()}")
    fig.tight_layout()
    fig.savefig(out / f"deformation_{backend_name}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    pc = ax.contourf(gx, gy, detj.reshape(gx.shape), levels=40, cmap="coolwarm")
    fig.colorbar(pc, ax=ax, label=r"$\det J_\phi$")
    ax.contour(gx, gy, detj.reshape(gx.shape), levels=[0.0], colors="black", linewidths=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("z1 / r0")
    ax.set_ylabel("z2 / r0")
    ax.set_title(f"det(J_phi) - {backend_name.upper()}")
    fig.tight_layout()
    fig.savefig(out / f"detJ_{backend_name}.png", dpi=300)
    plt.close(fig)


def contours(model, trap, out: Path, backend_name: str) -> None:
    ref = reference_geometry(trap)
    deformed = deformed_electrode_polygons(model.shape_network, trap)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (poly0, _), (poly1, _) in zip(ref.electrodes, deformed.electrodes):
        c0 = np.vstack([poly0, poly0[:1]])
        c1 = np.vstack([poly1, poly1[:1]])
        ax.plot(c0[:, 0], c0[:, 1], "b--", lw=1, label="initial" if "initial" not in ax.get_legend_handles_labels()[1] else "")
        ax.plot(c1[:, 0], c1[:, 1], "r-", lw=1.5, label="optimized" if "optimized" not in ax.get_legend_handles_labels()[1] else "")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(trap.outer_radius * np.cos(th), trap.outer_radius * np.sin(th), "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x / r0")
    ax.set_ylabel("y / r0")
    ax.legend()
    ax.set_title(f"Initial vs optimized electrodes - {backend_name.upper()}")
    fig.tight_layout()
    fig.savefig(out / f"contours_{backend_name}.png", dpi=300)
    plt.close(fig)


def backend_diagnostics(model, trap, out: Path, backend_name: str) -> None:
    backend = model.backend
    if isinstance(backend, PigBackend):
        diag = backend.gaussian_diagnostics()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sc = ax.scatter(diag["mu_x"], diag["mu_y"], c=diag["feature_norm"], s=8, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="feature norm")
        for e in trap.electrodes():
                th = np.linspace(0, 2 * np.pi, 100)
                ax.plot(e.cx + e.radius * np.cos(th), e.cy + e.radius * np.sin(th), "r-", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("x / r0")
        ax.set_ylabel("y / r0")
        ax.set_title("PIG Gaussian centers")
        fig.tight_layout()
        fig.savefig(out / "pig_gaussian_centers.png", dpi=300)
        fig.savefig(out / "pig_gaussians.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        order = np.argsort(diag["feature_norm"])[-120:]
        for idx in order:
            ellipse = Ellipse(
                (diag["mu_x"][idx], diag["mu_y"][idx]),
                width=2.0 * diag["sigma_x"][idx],
                height=2.0 * diag["sigma_y"][idx],
                facecolor="none",
                edgecolor="tab:green",
                alpha=0.35,
                lw=0.7,
            )
            ax.add_patch(ellipse)
        ax.scatter(diag["mu_x"][order], diag["mu_y"][order], s=5, color="tab:green", alpha=0.7)
        for e in trap.electrodes():
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(e.cx + e.radius * np.cos(th), e.cy + e.radius * np.sin(th), "r-", lw=0.6)
        ax.set_xlim(-trap.outer_radius, trap.outer_radius)
        ax.set_ylim(-trap.outer_radius, trap.outer_radius)
        ax.set_aspect("equal")
        ax.set_xlabel("x / r0")
        ax.set_ylabel("y / r0")
        ax.set_title("PIG Gaussian covariance ellipses")
        fig.tight_layout()
        fig.savefig(out / "pig_gaussian_covariances.png", dpi=300)
        plt.close(fig)
    elif isinstance(backend, PixelBackend):
        norm = backend.feature_norm_grid().cpu().numpy()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        R = trap.outer_radius
        pc = ax.imshow(norm, origin="lower", cmap="viridis", extent=[-R, R, -R, R])
        fig.colorbar(pc, ax=ax, label="feature norm")
        ax.set_xlabel("x / r0")
        ax.set_ylabel("y / r0")
        ax.set_title("PIXEL feature activity")
        fig.tight_layout()
        fig.savefig(out / "pixel_feature_activity.png", dpi=300)
        fig.savefig(out / "pixel_feature_norm.png", dpi=300)
        plt.close(fig)


def evaluate_run(config, model_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    trap = build_trap(config)
    model = load_model(config, model_path)
    backend_name = config.physics_backend.type
    potential_and_residual(model, trap, out, backend_name)
    deformation_and_detj(model, trap, out, backend_name)
    contours(model, trap, out, backend_name)
    backend_diagnostics(model, trap, out, backend_name)
    print(f"[evaluate] wrote diagnostic plots for {backend_name} to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    evaluate_run(config, Path(args.model), Path(args.out))


if __name__ == "__main__":
    main()
