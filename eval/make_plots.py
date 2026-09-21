"""Produces resulting plots from logged CSVs and exported runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = {"mlp": "tab:blue", "pixel": "tab:orange", "pig": "tab:green"}
_ORDER = ["mlp", "pixel", "pig"]


def _label(backend: str) -> str:
    return backend.upper()


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def eta_pinn_vs_epoch(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for backend in _ORDER:
        g = df[df["backend"] == backend]
        if g.empty:
            continue
        g = g.sort_values("step")
        ax.plot(g["step"], g["eta_pinn"], label=_label(backend), color=_COLORS.get(backend))
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\eta_{\mathrm{PINN}}$")
    ax.set_title("PINN-predicted eta vs epoch")
    ax.legend()
    _save(fig, out / "eta_pinn_vs_epoch.png")


def eta_fem_vs_checkpoint(df: pd.DataFrame, out: Path) -> None:
    fem = df[df["eta_fem"].notna()]
    fig, ax = plt.subplots(figsize=(6, 4))
    for backend in _ORDER:
        g = fem[fem["backend"] == backend]
        if g.empty:
            continue
        g = g.sort_values("step")
        ax.plot(g["step"], g["eta_fem"], marker="o", label=_label(backend), color=_COLORS.get(backend))
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel(r"$\eta_{\mathrm{FEM}}$")
    ax.set_title("FEM-validated eta vs checkpoint (headline)")
    ax.legend()
    _save(fig, out / "eta_fem_vs_checkpoint.png")


def eta_pinn_vs_fem_scatter(df: pd.DataFrame, out: Path) -> None:
    fem = df[df["eta_fem"].notna()]
    if fem.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for backend in _ORDER:
        g = fem[fem["backend"] == backend]
        if g.empty:
            continue
        ax.scatter(g["eta_pinn"], g["eta_fem"], label=_label(backend), color=_COLORS.get(backend), alpha=0.75)
    lo = float(np.nanmin([fem["eta_fem"].min(), fem["eta_pinn"].min()]))
    hi = float(np.nanmax([fem["eta_fem"].max(), fem["eta_pinn"].max()]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(r"$\eta_{\mathrm{PINN}}$")
    ax.set_ylabel(r"$\eta_{\mathrm{FEM}}$")
    ax.set_title("Trustworthiness: PINN eta vs FEM eta")
    ax.legend()
    _save(fig, out / "pinn_eta_vs_fem_eta_scatter.png")


def constraints_vs_epoch(df: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("min_detJ", r"$\min \det J_\phi$", 0.0, "detJ = 0"),
        ("min_gap", "minimum electrode gap", None, None),
        ("max_curv", "maximum boundary curvature", None, None),
        ("eta_fem", r"$\eta_{\mathrm{FEM}}$", None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, (col, ylabel, threshold, threshold_label) in zip(axes.ravel(), metrics):
        if col not in df:
            ax.text(0.5, 0.5, f"missing {col}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for backend in _ORDER:
            g = df[(df["backend"] == backend) & df[col].notna()].sort_values("step")
            if g.empty:
                continue
            ax.plot(g["step"], g[col], marker="o" if col == "eta_fem" else None,
                    label=_label(backend), color=_COLORS.get(backend))
        if threshold is not None:
            ax.axhline(threshold, color="red", ls=":", lw=1, label=threshold_label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend()
    fig.suptitle("Constraint and validated-eta health")
    _save(fig, out / "constraints_vs_epoch.png")


def loss_terms_vs_epoch(df: pd.DataFrame, out: Path) -> None:
    cols = [
        ("loss_pde", "PDE loss"),
        ("loss_bc", "BC loss"),
        ("loss_eta", "eta objective loss"),
        ("loss_quad", "center quadrupole loss"),
        ("loss_J", "Jacobian loss"),
        ("loss_disp", "displacement loss"),
        ("loss_gap", "gap loss"),
        ("loss_curv", "curvature loss"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, (col, title) in zip(axes.ravel(), cols):
        if col not in df:
            ax.text(0.5, 0.5, f"missing {col}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for backend in _ORDER:
            g = df[(df["backend"] == backend) & df[col].notna()].sort_values("step")
            if g.empty:
                continue
            y = g[col].astype(float)
            if col != "loss_eta" and np.nanmax(np.abs(y)) > 0:
                ax.set_yscale("symlog", linthresh=1e-8)
            ax.plot(g["step"], y, label=_label(backend), color=_COLORS.get(backend))
        ax.set_title(title)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(cols):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend()
    fig.suptitle("Loss terms vs epoch")
    _save(fig, out / "loss_terms_vs_epoch.png")


def shape_coupling_vs_epoch(df: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("grad_phi_eta", r"$||\nabla_\phi L_\eta||$", "Eta gradient into shape network"),
        ("grad_phi_quad", r"$||\nabla_\phi L_{\mathrm{quad}}||$", "Quadrupole gradient into shape network"),
        ("disp_mean", r"mean $||NN_\phi(z)-z||$", "Mean deformation magnitude"),
        ("disp_max", r"max $||NN_\phi(z)-z||$", "Max deformation magnitude"),
    ]
    if not any(col in df for col, _, _ in metrics):
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (col, ylabel, title) in zip(axes.ravel(), metrics):
        if col not in df:
            ax.text(0.5, 0.5, f"missing {col}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for backend in _ORDER:
            g = df[(df["backend"] == backend) & df[col].notna()].sort_values("step")
            if g.empty:
                continue
            y = g[col].astype(float)
            if col.startswith("grad_") and np.nanmax(np.abs(y)) > 0:
                ax.set_yscale("symlog", linthresh=1e-12)
            ax.plot(g["step"], y, marker="o", label=_label(backend), color=_COLORS.get(backend))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend()
    fig.suptitle("Shape-gradient coupling and deformation")
    _save(fig, out / "shape_coupling_vs_epoch.png")


def initial_vs_optimized_electrodes(results_root: Path, out: Path) -> str | None:
    """Build a compact geometry comparison from exported checkpoint ``.npz`` files."""
    joint_root = results_root / "joint"
    if not joint_root.exists():
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    wrote_any = False
    for ax, backend in zip(axes, _ORDER):
        run_dir = joint_root / f"{backend}_seed0" / "geometry"
        initial = run_dir / "checkpoint_0.npz"
        checkpoints = sorted(run_dir.glob("checkpoint_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
        if not initial.exists() or not checkpoints:
            ax.text(0.5, 0.5, "missing geometry", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(_label(backend))
            continue
        final = checkpoints[-1]
        with np.load(initial) as init_data, np.load(final) as final_data:
            electrode_keys = sorted(k for k in init_data.files if k.startswith("electrode_"))
            for index, key in enumerate(electrode_keys):
                p0 = init_data[key]
                p1 = final_data[key]
                p0 = np.vstack([p0, p0[:1]])
                p1 = np.vstack([p1, p1[:1]])
                ax.plot(p0[:, 0], p0[:, 1], "--", color="0.35", lw=1.0,
                        label="initial" if index == 0 else None)
                ax.plot(p1[:, 0], p1[:, 1], color=_COLORS.get(backend), lw=1.4,
                        label="optimized" if index == 0 else None)
            wrote_any = True
        ax.plot(0.0, 0.0, "k+", ms=8, mew=1.5, label="trap center")
        ax.set_aspect("equal")
        ax.set_title(_label(backend))
        ax.set_xlabel("x / r0")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y / r0")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Initial vs optimized electrode boundaries")
    if not wrote_any:
        plt.close(fig)
        return None
    filename = "initial_vs_optimized_electrodes.png"
    _save(fig, out / filename)
    return filename


def montage_diagnostics(out: Path, pattern_template: str, filename: str, title: str) -> str | None:
    """Combine per-backend diagnostic PNGs into one checklist-level figure."""
    paths = []
    for backend in _ORDER:
        path = out / f"{backend}_seed0" / pattern_template.format(backend=backend)
        paths.append(path if path.exists() else None)
    if not any(paths):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, backend, path in zip(axes, _ORDER, paths):
        if path is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.imshow(plt.imread(path))
        ax.set_title(_label(backend))
        ax.set_axis_off()
    fig.suptitle(title)
    _save(fig, out / filename)
    return filename


def make_joint_plots(metrics_csv: Path, out: Path) -> list[str]:
    df = pd.read_csv(metrics_csv)
    eta_pinn_vs_epoch(df, out)
    constraints_vs_epoch(df, out)
    loss_terms_vs_epoch(df, out)
    shape_coupling_vs_epoch(df, out)
    written = [
        "eta_pinn_vs_epoch.png",
        "constraints_vs_epoch.png",
        "loss_terms_vs_epoch.png",
        "shape_coupling_vs_epoch.png",
    ]
    if "eta_fem" in df and df["eta_fem"].notna().any():
        eta_fem_vs_checkpoint(df, out)
        eta_pinn_vs_fem_scatter(df, out)
        written += ["eta_fem_vs_checkpoint.png", "pinn_eta_vs_fem_eta_scatter.png"]
    geometry_plot = initial_vs_optimized_electrodes(metrics_csv.parents[1], out)
    if geometry_plot:
        written.append(geometry_plot)
    for plot in [
        montage_diagnostics(out, "deformation_{backend}.png", "deformation_field.png", "Deformation fields"),
        montage_diagnostics(out, "detJ_{backend}.png", "jacobian_determinant_heatmap.png", "Jacobian determinant heatmaps"),
    ]:
        if plot:
            written.append(plot)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="spinn/results/joint/metrics.csv")
    parser.add_argument("--out", default="spinn/results/figures")
    args = parser.parse_args(argv)
    written = make_joint_plots(Path(args.metrics), Path(args.out))
    print(f"[make_plots] wrote {len(written)} figures to {args.out}: {written}")


if __name__ == "__main__":
    main()
