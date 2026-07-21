"""Train a single coordinate-projection model and save its results.

The training and evaluation pipeline is shared across all three
architectures: same sampler, seed, optimizer, learning rate, step count, and
metrics. Only the model differs, selected by ``config.model.kind``. The
default training loss is the mean Euclidean coordinate distance; MSE is
tracked alongside it for the comparison plots.

Run directly:

    python -m experiments.coordinate_architecture_comparison.training.train_one_model \
        --model vanilla
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig, load_config
from ..models import MODEL_KINDS, build_model
from ..plotting.plot_fields import plot_displacement_field, plot_error_heatmap
from ..plotting.plot_losses import plot_loss_history
from .data import grid_points, sample_reference_points, to_tensor
from .losses import CoordinateMSELoss, MeanEuclideanCoordinateLoss
from .metrics import evaluate_coordinate_error

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class TrainingResult:
    """In-memory record of a finished run, consumed by the comparison plots."""

    def __init__(self, kind: str, history: dict[str, list[float]], final_metrics: dict[str, float]):
        self.kind = kind
        self.history = history
        self.final_metrics = final_metrics


def _make_validation_batch(config: ExperimentConfig, device: str) -> torch.Tensor:
    rng = np.random.default_rng(config.seed + 1)
    points = sample_reference_points(config, rng)
    if len(points) > config.training.val_batch:
        idx = rng.choice(len(points), config.training.val_batch, replace=False)
        points = points[idx]
    return to_tensor(points, device=device)


def train(config: ExperimentConfig, device: str = "cpu") -> tuple[torch.nn.Module, TrainingResult]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    train_loss_fn = MeanEuclideanCoordinateLoss()
    mse_fn = CoordinateMSELoss()

    val_z = _make_validation_batch(config, device)
    history: dict[str, list[float]] = {"step": [], "train_loss": [], "train_mse": [], "val_loss": [], "val_mse": []}

    for step in range(config.training.steps):
        z = to_tensor(sample_reference_points(config, rng), device=device)
        x_pred = model(z)
        loss = train_loss_fn(x_pred, z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.training.eval_every == 0 or step == config.training.steps - 1:
            with torch.no_grad():
                train_mse = mse_fn(x_pred, z).item()
                val_pred = model(val_z)
                val_loss = train_loss_fn(val_pred, val_z).item()
                val_mse = mse_fn(val_pred, val_z).item()
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["train_mse"].append(train_mse)
            history["val_loss"].append(val_loss)
            history["val_mse"].append(val_mse)

            if step % config.training.print_every == 0:
                print(
                    f"[{config.model.kind}] step {step:5d} | "
                    f"train {loss.item():.3e} | val {val_loss:.3e} | val_mse {val_mse:.3e}"
                )

    final_metrics = evaluate_coordinate_error(model, val_z)
    return model, TrainingResult(config.model.kind, history, final_metrics)


def _save_outputs(config: ExperimentConfig, model: torch.nn.Module, result: TrainingResult, device: str) -> Path:
    out_dir = RESULTS_DIR / config.model.kind
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_history(out_dir / "loss_history.png", result.history)
    plot_displacement_field(out_dir / "displacement_field.png", config, model, device)
    plot_error_heatmap(out_dir / "error_heatmap.png", config, model, device)

    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(
            {"model": config.model.kind, "config": config.to_dict(), "final_metrics": result.final_metrics},
            handle,
            indent=2,
        )
    return out_dir


def run_model(kind: str, device: str = "cpu") -> TrainingResult:
    """Train one architecture end to end and write its result directory."""

    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model {kind!r}, expected one of {MODEL_KINDS}")
    config = load_config(CONFIG_DIR / f"{kind}.yaml")
    model, result = train(config, device=device)
    out_dir = _save_outputs(config, model, result, device)
    print(f"[{kind}] saved results to {out_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one coordinate-projection architecture.")
    parser.add_argument("--model", required=True, choices=MODEL_KINDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_model(args.model, device=args.device)


if __name__ == "__main__":
    main()
