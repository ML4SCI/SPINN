"""Evaluate trained potential PINNs against circular-trap DEVSIM RF data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from PINN.common.tecplot import read_devsim_tecplot


def error_stats(prediction: np.ndarray, truth: np.ndarray, norm: float = 300.0) -> dict[str, float]:
    error = prediction - truth
    abs_error = np.abs(error)
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "p50_abs": float(np.percentile(abs_error, 50)),
        "p95_abs": float(np.percentile(abs_error, 95)),
        "max_abs": float(np.max(abs_error)),
        "mean_abs_percent_of_norm": float(100.0 * np.mean(abs_error) / norm),
        "p95_abs_percent_of_norm": float(100.0 * np.percentile(abs_error, 95) / norm),
    }


def evaluate_model(model, coords: np.ndarray, chunk_size: int = 8192, device: str = "cpu") -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(coords), chunk_size):
            points = torch.tensor(coords[start:start + chunk_size], dtype=torch.float32, device=device)
            parts.append(model(points).detach().cpu().numpy().reshape(-1))
    return np.concatenate(parts)


def scatter_triptych(
    x: np.ndarray,
    y: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    title: str,
    max_points: int = 20000,
) -> None:
    rng = np.random.default_rng(0)
    if len(x) > max_points:
        indices = rng.choice(len(x), max_points, replace=False)
    else:
        indices = np.arange(len(x))
    x_s, y_s = x[indices], y[indices]
    truth_s = truth[indices]
    pred_s = prediction[indices]
    err_s = pred_s - truth_s
    vmin = float(min(np.nanmin(truth_s), np.nanmin(pred_s)))
    vmax = float(max(np.nanmax(truth_s), np.nanmax(pred_s)))
    err_abs = float(max(abs(np.nanmin(err_s)), abs(np.nanmax(err_s)), 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    panels = [
        (truth_s, f"DEVSIM {title}", "jet", vmin, vmax),
        (pred_s, f"Model {title}", "jet", vmin, vmax),
        (err_s, "Model - DEVSIM", "bwr", -err_abs, err_abs),
    ]
    for ax, (values, panel_title, cmap, lo, hi) in zip(axes, panels):
        scatter = ax.scatter(x_s, y_s, c=values, cmap=cmap, s=3, vmin=lo, vmax=hi)
        fig.colorbar(scatter, ax=ax, label="V")
        ax.set_title(panel_title)
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compare_to_devsim(
    model,
    devsim_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int = 8192,
    max_points_plot: int = 20000,
    device: str = "cpu",
) -> dict:
    output_dir = Path(output_dir)
    data = read_devsim_tecplot(devsim_path)
    coords = np.column_stack([data["x"], data["y"]]).astype(np.float32)
    truth = data["Potential"].astype(np.float32)
    prediction = evaluate_model(model, coords, chunk_size=chunk_size, device=device)
    metrics = {
        "devsim_rf": str(devsim_path),
        "n_points": int(len(coords)),
        "phi_rf": error_stats(prediction, truth, norm=300.0),
    }
    scatter_triptych(
        data["x"],
        data["y"],
        truth,
        prediction,
        output_dir / "comparison_phi_rf.png",
        "Phi_RF",
        max_points=max_points_plot,
    )
    (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

