"""Fixed-geometry physics-solver benchmark (plan stage 1).

The shape network is the exact identity, so this trains a physics backend
(MLP / PIXEL / PIG) to solve Laplace on the *fixed* circular four-rod geometry.
It answers "which backend is the better PDE solver before any deformation" and
produces the fixed-geometry eta-error comparison (PINN eta vs FEM eta).

Outputs (under ``results/fixed/<backend>_seed<k>/``):
  metrics.csv, result.json, model.pt
and appends a row to ``results/fixed/metrics.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config, apply_overrides, load_config
from ..geometry.boundary_sampling import sample_batch
from ..geometry.export_fem import reference_geometry
from ..geometry.four_rod import build_trap
from ..losses.boundary import boundary_loss
from ..losses.pde import pde_loss
from ..models.backends import build_backend
from ..models.composed import ComposedField, IdentityShapeNetwork
from ..physics import eta as eta_mod
from ..physics.fem_reference import fem_eta
from .checkpoint import append_csv, save_checkpoint, write_json
from .schedules import build_scheduler

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

FIXED_FIELDS = [
    "backend", "seed", "step", "loss_pde", "loss_bc",
    "eta_pinn", "eta_fem", "eta_abs_error", "wall_time_sec",
]


def _fixed_lr(config: Config) -> dict:
    opt = config.optimizer
    kind = config.physics_backend.type.lower()
    if kind == "mlp":
        return {"lr_main": opt.lr_fixed_mlp}
    if kind == "pixel":
        return {"lr_main": opt.lr_fixed_pixel}
    return {
        "lr_main": opt.lr_fixed_pig_decoder,
        "lr_centers": opt.lr_fixed_pig_centers,
        "lr_covariances": opt.lr_fixed_pig_centers,
    }


def train_fixed(config: Config, steps: int | None = None, results_root: Path = RESULTS_ROOT) -> dict:
    torch.manual_seed(config.experiment.seed)
    rng = np.random.default_rng(config.experiment.seed)

    trap = build_trap(config)
    backend = build_backend(config)
    model = ComposedField(IdentityShapeNetwork(), backend).float()

    lrs = _fixed_lr(config)
    optimizer = torch.optim.Adam(backend.param_groups(**lrs))
    total_steps = steps if steps is not None else config.optimizer.steps
    scheduler = build_scheduler(optimizer, config.optimizer.scheduler, total_steps)

    s = config.sampling
    exp = config.experiment
    center_grid = trap.center_probe_grid(s.center_probe_radius, s.center_probe_grid)

    backend_name = config.physics_backend.type
    tag = f"{backend_name}_seed{exp.seed}"
    run_dir = results_root / "fixed" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    history = []
    for step in range(total_steps):
        if step % s.resample_every == 0:
            batch = sample_batch(
                trap, rng, s.interior_points,
                s.electrode_boundary_points_per_electrode, s.outer_boundary_points,
            )
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        l_pde = pde_loss(potential, batch.interior, model.shape_network)
        l_bc = boundary_loss(model, batch)
        loss = config.loss_weights.pde * l_pde + config.loss_weights.bc * l_bc
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % 200 == 0 or step + 1 == total_steps:
            row = {
                "backend": backend_name, "seed": exp.seed, "step": step,
                "loss_pde": float(l_pde.detach()), "loss_bc": float(l_bc.detach()),
                "wall_time_sec": time.time() - start,
            }
            history.append(row)
            print(f"[fixed:{backend_name}] step {step:6d} pde={row['loss_pde']:.3e} bc={row['loss_bc']:.3e}", flush=True)

    # final eta: PINN vs FEM on the fixed reference geometry
    eta_pinn = eta_mod.eta_from_backend_on_grid(
        backend, center_grid, exp.r0, exp.V0, s.center_probe_radius
    )
    geom = reference_geometry(trap)
    fem = fem_eta(geom, exp.r0, exp.V0, grid=config.validation.fem_grid, center_radius=s.center_probe_radius)
    eta_abs_error = abs(eta_pinn["eta_fit"] - fem["eta_fem"])

    final_row = {
        "backend": backend_name, "seed": exp.seed, "step": total_steps,
        "loss_pde": history[-1]["loss_pde"] if history else float("nan"),
        "loss_bc": history[-1]["loss_bc"] if history else float("nan"),
        "eta_pinn": eta_pinn["eta_fit"], "eta_fem": fem["eta_fem"],
        "eta_abs_error": eta_abs_error, "wall_time_sec": time.time() - start,
    }
    append_csv(results_root / "fixed" / "metrics.csv", final_row, FIXED_FIELDS)
    save_checkpoint(model, run_dir / "model.pt")
    result = {
        "backend": backend_name, "seed": exp.seed,
        "eta_pinn": eta_pinn, "eta_fem": fem["eta_fem"],
        "eta_abs_error": eta_abs_error, "history": history,
        "config": config.to_dict(),
    }
    fem.pop("_solved", None)
    result["fem"] = fem
    write_json(run_dir / "result.json", result)
    print(f"[fixed:{backend_name}] eta_pinn={eta_pinn['eta_fit']:.4f} eta_fem={fem['eta_fem']:.4f} |err|={eta_abs_error:.4f}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    if args.seed is not None:
        config.experiment.seed = args.seed
    train_fixed(config, steps=args.steps)


if __name__ == "__main__":
    main()
