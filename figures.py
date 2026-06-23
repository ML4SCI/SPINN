"""Notebook-facing helpers for potential architecture comparison plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .config import make_config
from .models import MODEL_KINDS

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def available_results(kinds: tuple[str, ...] = MODEL_KINDS) -> dict[str, bool]:
    """Return whether each model has both training and DEVSIM metrics."""

    return {
        kind: (RESULTS_ROOT / kind / "training_history.csv").exists()
        and (RESULTS_ROOT / kind / "comparison_metrics.json").exists()
        for kind in kinds
    }


def run_comparison(
    *,
    kinds: tuple[str, ...] = MODEL_KINDS,
    steps: int = 5000,
    seed: int = 0,
    skip_devsim: bool = False,
    overrides: list[str] | None = None,
    device: str = "cpu",
) -> None:
    """Train selected models and regenerate comparison plots."""

    from .plotting import comparison_plots
    from .train import train_one

    overrides = list(overrides or [])
    overrides.extend([f"training.steps={steps}", f"seed={seed}"])
    for kind in kinds:
        config = make_config(kind, overrides)
        train_one(
            config,
            device=device,
            output_dir=RESULTS_ROOT / kind,
            run_devsim_compare=not skip_devsim,
        )
    comparison_plots(RESULTS_ROOT, list(kinds))


def _missing_figure(message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.axis("off")
    ax.text(0.02, 0.75, message, ha="left", va="top", fontsize=11)
    fig.tight_layout()
    return fig


def _read_history(kind: str) -> list[dict[str, float]]:
    path = RESULTS_ROOT / kind / "training_history.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _read_metrics(kind: str) -> dict | None:
    path = RESULTS_ROOT / kind / "comparison_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["phi_rf"]


def plot_convergence(kinds: tuple[str, ...] = MODEL_KINDS) -> Figure:
    """Overlay total training loss curves for available models."""

    histories = {kind: _read_history(kind) for kind in kinds}
    histories = {kind: rows for kind, rows in histories.items() if rows}
    if not histories:
        return _missing_figure(
            "No training histories found. Run the setup cell with RUN_TRAINING=True "
            "or run the CLI comparison first."
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for kind, history in histories.items():
        ax.semilogy(
            [row["step"] for row in history],
            [row["total_loss"] for row in history],
            label=kind,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("total loss")
    ax.set_title("Convergence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_phi_mse(kinds: tuple[str, ...] = MODEL_KINDS) -> Figure:
    """Bar plot of RF potential MSE against DEVSIM."""

    metrics = {kind: _read_metrics(kind) for kind in kinds}
    metrics = {kind: value for kind, value in metrics.items() if value is not None}
    if not metrics:
        return _missing_figure(
            "No comparison_metrics.json files found. Run with DEVSIM evaluation enabled."
        )

    labels = list(metrics)
    values = [metrics[kind]["mse"] for kind in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel("phi MSE vs DEVSIM [V^2]")
    ax.set_title("RF potential error")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


def metrics_markdown(kinds: tuple[str, ...] = MODEL_KINDS) -> str:
    """Markdown table of DEVSIM RF-potential metrics."""

    rows = []
    for kind in kinds:
        metrics = _read_metrics(kind)
        if metrics is None:
            continue
        rows.append(
            [
                kind,
                f"{metrics['mse']:.4g}",
                f"{metrics['rmse']:.4g}",
                f"{metrics['mae']:.4g}",
                f"{metrics['mean_abs_percent_of_norm']:.3f}%",
            ]
        )
    if not rows:
        return "No DEVSIM comparison metrics found yet."
    lines = [
        "| model | phi MSE [V^2] | phi RMSE [V] | phi MAE [V] | MAE / 300 V |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def show_phi_comparison(kind: str) -> Figure:
    """Display the saved DEVSIM/model/error triptych for one architecture."""

    path = RESULTS_ROOT / kind / "comparison_phi_rf.png"
    if not path.exists():
        return _missing_figure(f"No phi comparison image found for {kind!r}: {path}")
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=(12.0, 4.0))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{kind}: DEVSIM vs model RF potential")
    fig.tight_layout()
    return fig
