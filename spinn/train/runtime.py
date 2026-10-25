"""Shared runtime for training jobs: CLI, seeding, device, stepping, artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..config import Config, apply_overrides, load_config
from ..geometry.collocation import CollocationBatch, sample_collocation_batch
from ..geometry.four_rod import FourRodTrap

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


# --- configuration and environment -----------------------------------------
def parse_training_config(
    description: str | None, argv: list[str] | None = None
) -> tuple[Config, int | None]:
    """Parse ``--config/--seed/--steps/--override`` into a validated config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.override)
    if args.seed is not None:
        config.experiment.seed = args.seed
    return config, args.steps


def seed_everything(config: Config) -> np.random.Generator:
    """Seed NumPy and PyTorch and configure deterministic execution."""
    seed = config.experiment.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(config.runtime.deterministic, warn_only=True)
    return np.random.default_rng(seed)


def resolve_device(config: Config) -> torch.device:
    """Resolve ``cpu``, ``cuda``, ``mps``, or ``auto`` with clear failures."""
    requested = config.runtime.device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("runtime.device requests MPS, but MPS is unavailable")
    return device


def resolve_dtype(config: Config) -> torch.dtype:
    return _DTYPES[config.runtime.dtype]


# --- optimization -----------------------------------------------------------
def build_scheduler(optimizer: torch.optim.Optimizer, kind: str, total_steps: int):
    """Plan optimizer table: cosine or exponential decay (to ~1% of the initial rate)."""
    kind = (kind or "none").lower()
    if kind == "none":
        return None
    if kind in {"cosine", "cosine_decay"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1)
        )
    if kind in {"exp", "exponential", "exponential_decay"}:
        gamma = math.exp(math.log(0.01) / max(total_steps, 1))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    raise ValueError(f"unknown scheduler {kind!r}")


def training_batches(
    config: Config,
    trap: FourRodTrap,
    rng: np.random.Generator,
    total_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    """Yield ``(step, batch)``, drawing fresh collocation every ``resample_every`` steps."""
    s = config.sampling
    for step in range(total_steps):
        if step % s.resample_every == 0:
            batch: CollocationBatch = sample_collocation_batch(
                trap,
                rng,
                s.interior_points,
                s.electrode_boundary_points_per_electrode,
                s.outer_boundary_points,
                device=device,
                dtype=dtype,
            )
        yield step, batch


def apply_gradient_step(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Config,
    *,
    label: str,
    step: int,
    terms: dict | None = None,
) -> bool:
    """Backprop, clip, and step. A non-finite loss raises or is skipped (returns False)."""
    if not torch.isfinite(loss):
        detail = terms if terms is not None else loss.detach().item()
        message = f"non-finite loss at step {step}: {detail}"
        if config.runtime.fail_on_nonfinite:
            raise FloatingPointError(message)
        print(f"[{label}] WARNING {message}", flush=True)
        return False
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        config.runtime.gradient_clip_norm,
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return True


# --- artifacts --------------------------------------------------------------
def save_checkpoint(
    model: torch.nn.Module, path: Path, extra: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row, refusing to mix rows into a file with a different header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    if not new_file:
        with path.open(newline="") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != fieldnames:
            raise ValueError(
                f"refusing to append incompatible row to {path}: "
                f"expected {fieldnames}, found {existing_header}"
            )
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, data: dict) -> None:
    """Atomically write JSON (via a temporary sibling file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=float) + "\n")
    temporary.replace(path)


def _git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass
class RunContext:
    """Paths, timing, and provenance (``manifest.json``) for one fixed or joint run."""

    mode: str
    directory: Path
    started_at: float
    manifest_path: Path

    @classmethod
    def create(
        cls, mode: str, config: Config, results_root: Path = RESULTS_ROOT
    ) -> RunContext:
        tag = f"{config.physics_backend.type}_seed{config.experiment.seed}"
        directory = Path(results_root) / mode / tag
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "mode": mode,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "git_commit": _git_value("rev-parse", "HEAD"),
                "git_status": _git_value("status", "--porcelain"),
                "config": config.to_dict(),
            },
        )
        return cls(mode, directory, time.perf_counter(), manifest_path)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def finish(self, **summary) -> None:
        """Mark the manifest complete and attach final scalar summaries."""
        manifest = json.loads(self.manifest_path.read_text())
        manifest.update(
            status="complete",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_time_sec=self.elapsed_seconds,
            **summary,
        )
        write_json(self.manifest_path, manifest)
