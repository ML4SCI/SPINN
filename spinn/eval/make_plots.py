"""Cross-backend figures from the joint metrics CSV and exported geometries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import BACKENDS

_COLORS = {"mlp": "tab:blue", "pixel": "tab:orange", "pig": "tab:green"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _mark_missing(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _per_backend(df: pd.DataFrame, column: str):
    """Yield ``(backend, rows)`` with non-null ``column`` values, sorted by step."""
    for backend in BACKENDS:
        rows = df[(df["backend"] == backend) & df[column].notna()].sort_values("step")
        if not rows.empty:
            yield backend, rows


def _plot_lines(ax, df, column, *, marker=None, symlog=None) -> None:
    for backend, rows in _per_backend(df, column):
        y = rows[column].astype(float)
        if symlog is not None and np.nanmax(np.abs(y)) > 0:
            ax.set_yscale("symlog", linthresh=symlog)
        ax.plot(
            rows["step"],
            y,
            marker=marker,
            label=backend.upper(),
            color=_COLORS[backend],
        )


def _metric_grid(
    df, panels, shape, figsize, title, path, *, marker=None, symlog=None, threshold=None
):
    """One subplot per ``(column, ylabel, panel_title)``; missing columns are marked.

    ``marker``/``symlog`` may be callables of the column name; ``threshold`` is an
    optional ``(column, value, label)`` reference line.
    """
    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True)
    for ax, (column, ylabel, panel_title) in zip(axes.ravel(), panels):
        if column not in df:
            _mark_missing(ax, f"missing {column}")
            continue
        _plot_lines(
            ax,
            df,
            column,
            marker=marker(column) if callable(marker) else marker,
            symlog=symlog(column) if callable(symlog) else symlog,
        )
        if threshold is not None and column == threshold[0]:
            ax.axhline(threshold[1], color="red", ls=":", lw=1, label=threshold[2])
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if panel_title is not None:
            ax.set_title(panel_title)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(panels) :]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend()
    fig.suptitle(title)
    _save(fig, path)


def eta_pinn_vs_epoch(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_lines(ax, df, "eta_pinn")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\eta_{\mathrm{PINN}}$")
    ax.set_title("PINN-predicted eta vs epoch")
    ax.legend()
    _save(fig, out / "eta_pinn_vs_epoch.png")


def eta_fem_vs_checkpoint(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_lines(ax, df, "eta_fem", marker="o")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel(r"$\eta_{\mathrm{FEM}}$")
    ax.set_title("FEM-validated eta vs checkpoint (headline)")
    ax.legend()
    _save(fig, out / "eta_fem_vs_checkpoint.png")


def eta_pinn_vs_fem_scatter(df: pd.DataFrame, out: Path) -> None:
    fem = df[df["eta_fem"].notna()]
    fig, ax = plt.subplots(figsize=(5, 5))
    for backend, rows in _per_backend(fem, "eta_fem"):
        ax.scatter(
            rows["eta_pinn"],
            rows["eta_fem"],
            label=backend.upper(),
            color=_COLORS[backend],
            alpha=0.75,
        )
    lo = float(np.nanmin([fem["eta_fem"].min(), fem["eta_pinn"].min()]))
    hi = float(np.nanmax([fem["eta_fem"].max(), fem["eta_pinn"].max()]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(r"$\eta_{\mathrm{PINN}}$")
    ax.set_ylabel(r"$\eta_{\mathrm{FEM}}$")
    ax.set_title("Trustworthiness: PINN eta vs FEM eta")
    ax.legend()
    _save(fig, out / "pinn_eta_vs_fem_eta_scatter.png")


def constraints_vs_epoch(df: pd.DataFrame, out: Path) -> None:
    panels = [
        ("min_detJ", r"$\min \det J_\phi$", None),
        ("min_gap", "minimum electrode gap", None),
        ("max_curv", "maximum boundary curvature", None),
        ("eta_fem", r"$\eta_{\mathrm{FEM}}$", None),
    ]
    _metric_grid(
        df,
        panels,
        (2, 2),
        (9, 7),
        "Constraint and validated-eta health",
        out / "constraints_vs_epoch.png",
        marker=lambda column: "o" if column == "eta_fem" else None,
        threshold=("min_detJ", 0.0, "detJ = 0"),
    )


def loss_terms_vs_epoch(df: pd.DataFrame, out: Path) -> None:
    panels = [
        ("loss_pde", None, "PDE loss"),
        ("loss_bc", None, "BC loss"),
        ("loss_eta", None, "eta objective loss"),
        ("loss_quad", None, "center quadrupole loss"),
        ("loss_J", None, "Jacobian loss"),
        ("loss_disp", None, "displacement loss"),
        ("loss_gap", None, "gap loss"),
        ("loss_curv", None, "curvature loss"),
    ]
    _metric_grid(
        df,
        panels,
        (2, 4),
        (14, 6),
        "Loss terms vs epoch",
        out / "loss_terms_vs_epoch.png",
        symlog=lambda column: None if column == "loss_eta" else 1e-8,
    )


def shape_coupling_vs_epoch(df: pd.DataFrame, out: Path) -> bool:
    panels = [
        (
            "grad_phi_eta",
            r"$||\nabla_\phi L_\eta||$",
            "Eta gradient into shape network",
        ),
        (
            "grad_phi_quad",
            r"$||\nabla_\phi L_{\mathrm{quad}}||$",
            "Quadrupole gradient into shape network",
        ),
        ("disp_mean", r"mean $||NN_\phi(z)-z||$", "Mean deformation magnitude"),
        ("disp_max", r"max $||NN_\phi(z)-z||$", "Max deformation magnitude"),
    ]
    if not any(column in df for column, _, _ in panels):
        return False
    _metric_grid(
        df,
        panels,
        (2, 2),
        (10, 7),
        "Shape-gradient coupling and deformation",
        out / "shape_coupling_vs_epoch.png",
        marker="o",
        symlog=lambda column: 1e-12 if column.startswith("grad_") else None,
    )
    return True


def initial_vs_optimized_electrodes(results_root: Path, out: Path) -> bool:
    """Seed-0 geometry comparison from the exported checkpoint ``.npz`` files."""
    joint_root = results_root / "joint"
    if not joint_root.exists():
        return False
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    wrote_any = False
    for ax, backend in zip(axes, BACKENDS):
        geometry_dir = joint_root / f"{backend}_seed0" / "geometry"
        initial = geometry_dir / "checkpoint_0.npz"
        checkpoints = sorted(
            geometry_dir.glob("checkpoint_*.npz"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        ax.set_title(backend.upper())
        if not initial.exists() or not checkpoints:
            ax.text(
                0.5,
                0.5,
                "missing geometry",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        with np.load(initial) as init_data, np.load(checkpoints[-1]) as final_data:
            keys = sorted(k for k in init_data.files if k.startswith("electrode_"))
            for index, key in enumerate(keys):
                p0 = np.vstack([init_data[key], init_data[key][:1]])
                p1 = np.vstack([final_data[key], final_data[key][:1]])
                ax.plot(
                    p0[:, 0],
                    p0[:, 1],
                    "--",
                    color="0.35",
                    lw=1.0,
                    label="initial" if index == 0 else None,
                )
                ax.plot(
                    p1[:, 0],
                    p1[:, 1],
                    color=_COLORS[backend],
                    lw=1.4,
                    label="optimized" if index == 0 else None,
                )
        wrote_any = True
        ax.plot(0.0, 0.0, "k+", ms=8, mew=1.5, label="trap center")
        ax.set_aspect("equal")
        ax.set_xlabel("x / r0")
        ax.grid(alpha=0.2)
    if not wrote_any:
        plt.close(fig)
        return False
    axes[0].set_ylabel("y / r0")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Initial vs optimized electrode boundaries")
    _save(fig, out / "initial_vs_optimized_electrodes.png")
    return True


def montage_diagnostics(out: Path, pattern: str, filename: str, title: str) -> bool:
    """Combine per-backend seed-0 diagnostic PNGs (from ``evaluate``) into one figure."""
    paths = [out / f"{b}_seed0" / pattern.format(backend=b) for b in BACKENDS]
    if not any(p.exists() for p in paths):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, backend, path in zip(axes, BACKENDS, paths):
        if path.exists():
            ax.imshow(plt.imread(path))
            ax.set_axis_off()
        else:
            _mark_missing(ax, "missing")
        ax.set_title(backend.upper())
    fig.suptitle(title)
    _save(fig, out / filename)
    return True


def make_joint_plots(metrics_csv: Path, out: Path) -> list[str]:
    df = pd.read_csv(metrics_csv)
    eta_pinn_vs_epoch(df, out)
    constraints_vs_epoch(df, out)
    loss_terms_vs_epoch(df, out)
    written = [
        "eta_pinn_vs_epoch.png",
        "constraints_vs_epoch.png",
        "loss_terms_vs_epoch.png",
    ]
    if shape_coupling_vs_epoch(df, out):
        written.append("shape_coupling_vs_epoch.png")
    if "eta_fem" in df and df["eta_fem"].notna().any():
        eta_fem_vs_checkpoint(df, out)
        eta_pinn_vs_fem_scatter(df, out)
        written += ["eta_fem_vs_checkpoint.png", "pinn_eta_vs_fem_eta_scatter.png"]
    if initial_vs_optimized_electrodes(metrics_csv.parents[1], out):
        written.append("initial_vs_optimized_electrodes.png")
    montages = [
        ("deformation_{backend}.png", "deformation_field.png", "Deformation fields"),
        (
            "detJ_{backend}.png",
            "jacobian_determinant_heatmap.png",
            "Jacobian determinant heatmaps",
        ),
    ]
    written += [
        name
        for pattern, name, title in montages
        if montage_diagnostics(out, pattern, name, title)
    ]
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
