"""Cross-architecture comparison plots and summary CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_convergence_comparison(path: str | Path, results: Iterable) -> None:
    fig, ax = plt.subplots()
    for result in results:
        ax.semilogy(result.history["step"], result.history["val_loss"], label=result.kind)
    ax.set_xlabel("step")
    ax.set_ylabel("validation mean Euclidean error")
    ax.set_title("Convergence comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mse_comparison(path: str | Path, results: Iterable) -> None:
    results = list(results)
    kinds = [r.kind for r in results]
    mse = [r.final_metrics["mse"] for r in results]

    fig, ax = plt.subplots()
    ax.bar(kinds, mse)
    ax.set_ylabel("final validation MSE")
    ax.set_yscale("log")
    ax.set_title("Final MSE by architecture")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_metrics_summary(path: str | Path, results: Iterable) -> None:
    results = list(results)
    fieldnames = ["model"] + sorted(results[0].final_metrics)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({"model": result.kind, **result.final_metrics})
