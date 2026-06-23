"""Plots for the potential architecture comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_history(history: list[dict[str, float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    steps = [row["step"] for row in history]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.semilogy(steps, [row["total_loss"] for row in history], label="total")
    ax.semilogy(steps, [row["pde_loss"] for row in history], label="PDE")
    ax.semilogy(steps, [row["bc_loss"] for row in history], label="BC")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def read_history(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open() as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def comparison_plots(results_root: str | Path, kinds: list[str]) -> None:
    import json

    root = Path(results_root)
    histories = {}
    metrics = {}
    for kind in kinds:
        run_dir = root / kind
        history_path = run_dir / "training_history.csv"
        metrics_path = run_dir / "comparison_metrics.json"
        if history_path.exists():
            histories[kind] = read_history(history_path)
        if metrics_path.exists():
            metrics[kind] = json.loads(metrics_path.read_text())["phi_rf"]

    if histories:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for kind, history in histories.items():
            ax.semilogy(
                [row["step"] for row in history],
                [row["total_loss"] for row in history],
                label=kind,
            )
        ax.set_xlabel("step")
        ax.set_ylabel("total loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(root / "convergence_comparison.png", dpi=180)
        plt.close(fig)

    if metrics:
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        labels = list(metrics)
        values = [metrics[kind]["mse"] for kind in labels]
        ax.bar(labels, values)
        ax.set_yscale("log")
        ax.set_ylabel("phi MSE vs DEVSIM [V^2]")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(root / "phi_mse_comparison.png", dpi=180)
        plt.close(fig)

