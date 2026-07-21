"""Per-model loss-history plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_history(path: str | Path, history: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots()
    steps = history["step"]
    ax.semilogy(steps, history["train_loss"], label="train mean Euclidean")
    ax.semilogy(steps, history["val_loss"], label="val mean Euclidean")
    ax.semilogy(steps, history["val_mse"], linestyle="--", label="val MSE")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
