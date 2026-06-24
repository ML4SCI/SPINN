"""Shared training loop for Vanilla, PIXEL, and PIG potential PINNs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, write_config
from .devsim_compare import compare_to_devsim
from .geometry import training_batch
from .losses import total_loss
from .models import build_model
from .plotting import plot_loss_history

SPINN_ROOT = Path(__file__).resolve().parents[2]


def _optimizer(model: torch.nn.Module, config: ExperimentConfig):
    name = config.training.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.training.lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    if name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=config.training.lr,
            line_search_fn="strong_wolfe",
            max_iter=1,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm(2).item() ** 2)
    return total ** 0.5


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "total_loss", "pde_loss", "bc_loss", "grad_norm"],
        )
        writer.writeheader()
        writer.writerows(history)


def train_one(
    config: ExperimentConfig,
    *,
    device: str = "cpu",
    output_dir: Path | None = None,
    run_devsim_compare: bool = True,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = build_model(config).to(device)
    optimizer = _optimizer(model, config)
    history: list[dict[str, float]] = []

    def closure():
        optimizer.zero_grad()
        batch = training_batch(config, rng, device=device)
        loss, terms = total_loss(model, batch, config.training.bc_weight)
        loss.backward()
        return loss, terms

    for step in range(config.training.steps):
        if config.training.optimizer.lower() == "lbfgs":
            loss, terms = closure()
            if config.training.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step(lambda: closure()[0])
        else:
            loss, terms = closure()
            if config.training.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()

        if step % config.training.eval_every == 0 or step + 1 == config.training.steps:
            row = {
                "step": float(step),
                "total_loss": float(terms["total"]),
                "pde_loss": float(terms["pde"]),
                "bc_loss": float(terms["bc"]),
                "grad_norm": _grad_norm(model),
            }
            history.append(row)
        if step % config.training.print_every == 0:
            print(
                f"[{config.model.kind}] {step:6d} "
                f"total={float(terms['total']):.3e} "
                f"pde={float(terms['pde']):.3e} "
                f"bc={float(terms['bc']):.3e}",
                flush=True,
            )

    if output_dir is None:
        output_dir = Path(config.output_root) / config.model.kind
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config(output_dir / "config.json", config)
    _write_history(output_dir / "training_history.csv", history)
    plot_loss_history(history, output_dir / "loss_history.png")
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")

    result = {
        "model": config.model.kind,
        "config": config.to_dict(),
        "history": history,
    }
    if run_devsim_compare:
        devsim_path = Path(config.evaluation.devsim_rf)
        if not devsim_path.is_absolute():
            devsim_path = SPINN_ROOT / devsim_path
        result["comparison"] = compare_to_devsim(
            model,
            devsim_path,
            output_dir,
            chunk_size=config.evaluation.chunk_size,
            max_points_plot=config.evaluation.max_points_plot,
            device=device,
        )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return model, result
