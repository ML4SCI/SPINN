#!/usr/bin/env python3
"""Aggregate PINN hyperparameter runs into tables and summary plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "SPINN" / "Hyperparameter" / "results" / "runs"
DEFAULT_OUT = PROJECT_ROOT / "SPINN" / "Hyperparameter" / "results" / "summary"
SWEEP_COLORS = {
    "baseline": "#2f3a45",
    "width": "#2f6fbb",
    "depth": "#8a5fbf",
    "activation": "#d64f4f",
    "bc_weight": "#2e8b57",
    "optimizer": "#c47f00",
    "learning_rate": "#008b8b",
}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_history(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


def first_step_below(steps, values, threshold):
    idx = np.where(values <= threshold)[0]
    if len(idx) == 0:
        return None
    return int(steps[idx[0]])


def summarize_run(run_dir):
    cfg = read_json(run_dir / "config.json")
    hist = read_history(run_dir / "training_history.csv")

    steps = np.asarray(hist["step"], dtype=float)
    total = np.asarray(hist["total_loss"], dtype=float)
    pde = np.asarray(hist["pde_loss"], dtype=float)
    bc = np.asarray(hist["bc_loss"], dtype=float)
    initial = float(total[0]) if len(total) else np.nan
    normalized = total / initial if initial != 0 else np.full_like(total, np.nan)

    row = {
        "run_dir": str(run_dir),
        "sweep_name": cfg.get("sweep_name", ""),
        "variant_name": cfg.get("variant_name", run_dir.name),
        "seed": cfg.get("seed", ""),
        "width": cfg["model"].get("width", ""),
        "depth": cfg["model"].get("depth", ""),
        "activation": cfg["model"].get("activation", ""),
        "optimizer": cfg["training"].get("optimizer", "adam"),
        "lr": cfg["training"].get("lr", ""),
        "bc_weight": cfg["training"].get("bc_weight", ""),
        "steps_logged": int(len(hist)),
        "final_step": int(steps[-1]) if len(steps) else "",
        "initial_total_loss": initial,
        "final_total_loss": float(total[-1]) if len(total) else np.nan,
        "final_pde_loss": float(pde[-1]) if len(pde) else np.nan,
        "final_bc_loss": float(bc[-1]) if len(bc) else np.nan,
        "final_normalized_loss": float(normalized[-1]) if len(normalized) else np.nan,
        "auc_normalized_loss": float(np.trapezoid(normalized, steps)) if len(steps) > 1 else np.nan,
        "step_norm_loss_le_1e-1": first_step_below(steps, normalized, 1e-1),
        "step_norm_loss_le_1e-2": first_step_below(steps, normalized, 1e-2),
        "step_norm_loss_le_1e-3": first_step_below(steps, normalized, 1e-3),
    }

    metrics_path = run_dir / "comparison_metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        phi = metrics.get("phi_rf", {})
        psi = metrics.get("psi_rf_eV", {})
        row.update({
            "phi_mse": phi.get("mse", np.nan),
            "phi_rmse": phi.get("rmse", np.nan),
            "phi_mae": phi.get("mae", np.nan),
            "phi_p95_abs": phi.get("p95_abs", np.nan),
            "phi_mae_percent_of_300V": phi.get("mean_abs_percent_of_norm", np.nan),
            "psi_mse": psi.get("mse", np.nan),
            "psi_rmse": psi.get("rmse", np.nan),
            "psi_mae": psi.get("mae", np.nan),
            "psi_p95_abs": psi.get("p95_abs", np.nan),
        })
    else:
        row.update({
            "phi_mse": np.nan,
            "phi_rmse": np.nan,
            "phi_mae": np.nan,
            "phi_p95_abs": np.nan,
            "phi_mae_percent_of_300V": np.nan,
            "psi_mse": np.nan,
            "psi_rmse": np.nan,
            "psi_mae": np.nan,
            "psi_p95_abs": np.nan,
        })

    return row, steps, normalized


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_normalized_losses(curves, out_path):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for label, sweep_name, steps, normalized in curves:
        color = SWEEP_COLORS.get(sweep_name, "#666666")
        linewidth = 2.2 if label in {"baseline", "bc_weight_1000", "activation_relu"} else 1.45
        alpha = 1.0 if label in {"baseline", "bc_weight_1000", "activation_relu"} else 0.72
        ax.semilogy(steps, normalized, label=label, linewidth=linewidth, color=color, alpha=alpha)
    ax.set_xlabel("step")
    ax.set_ylabel("normalized total loss")
    ax.set_title("Normalized PINN training convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bar(rows, key, ylabel, out_path, *, title, log=True, sort_by_value=True):
    rows = [r for r in rows if np.isfinite(float(r.get(key, np.nan)))]
    if not rows:
        return
    if sort_by_value:
        rows = sorted(rows, key=lambda r: float(r[key]))
    else:
        rows = sorted(rows, key=lambda r: (r["sweep_name"], r["variant_name"]))
    labels = [r["variant_name"] for r in rows]
    values = [float(r[key]) for r in rows]
    colors = [SWEEP_COLORS.get(r["sweep_name"], "#666666") for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 6.1))
    ax.bar(np.arange(len(labels)), values, color=colors, width=0.72)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.margins(x=0.01)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2g}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_vs_error(rows, out_path):
    rows = [
        r for r in rows
        if np.isfinite(float(r.get("final_normalized_loss", np.nan)))
        and np.isfinite(float(r.get("phi_mse", np.nan)))
        and float(r.get("final_normalized_loss", 0)) > 0
        and float(r.get("phi_mse", 0)) > 0
    ]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for sweep_name in sorted({r["sweep_name"] for r in rows}):
        group = [r for r in rows if r["sweep_name"] == sweep_name]
        ax.scatter(
            [float(r["final_normalized_loss"]) for r in group],
            [float(r["phi_mse"]) for r in group],
            s=70,
            label=sweep_name,
            color=SWEEP_COLORS.get(sweep_name, "#666666"),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )

    for r in rows:
        if r["variant_name"] in {"activation_relu", "bc_weight_1000", "baseline", "width_256", "optimizer_rmsprop"}:
            ax.annotate(
                r["variant_name"],
                (float(r["final_normalized_loss"]), float(r["phi_mse"])),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("final normalized training loss")
    ax.set_ylabel("MSE(Phi_PINN - Phi_DEVSIM) [V^2]")
    ax.set_title("Training loss does not always predict physical accuracy")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate PINN hyperparameter run outputs.")
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.manifest:
        manifest = read_json(args.manifest)
        run_dirs = [Path(exp["run_dir"]) for exp in manifest["experiments"] if exp.get("run_dir")]
    else:
        run_dirs = sorted(p for p in args.run_root.iterdir() if (p / "config.json").exists())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    for run_dir in run_dirs:
        hist_path = run_dir / "training_history.csv"
        if not hist_path.exists():
            continue
        row, steps, normalized = summarize_run(run_dir)
        rows.append(row)
        curves.append((row["variant_name"], row["sweep_name"], steps, normalized))

    summary_csv = args.out_dir / "summary.csv"
    summary_json = args.out_dir / "summary.json"
    write_csv(rows, summary_csv)
    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    plot_normalized_losses(curves, args.out_dir / "normalized_loss_curves.png")
    plot_bar(
        rows,
        "phi_mse",
        "MSE(Phi_PINN - Phi_DEVSIM) [V^2]",
        args.out_dir / "phi_mse_by_variant.png",
        title="RF potential error by variant",
    )
    plot_bar(
        rows,
        "psi_mse",
        "MSE(psi_PINN - psi_DEVSIM) [eV^2]",
        args.out_dir / "psi_mse_by_variant.png",
        title="RF pseudopotential error by variant",
    )
    plot_bar(
        rows,
        "final_normalized_loss",
        "final normalized loss",
        args.out_dir / "final_normalized_loss_by_variant.png",
        title="Final normalized training loss by variant",
    )
    plot_loss_vs_error(rows, args.out_dir / "loss_vs_phi_mse.png")

    print(f"Aggregated {len(rows)} runs")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
