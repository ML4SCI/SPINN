"""Main joint shape-optimization benchmark (plan stage 2).

Trains the shape network ``NN_phi`` and a physics backend ``NN_theta`` together
to maximize the geometric efficiency ``eta`` while solving Laplace and respecting
the geometry constraints. Every ``fem_checkpoint_every`` steps the current
deformed geometry is exported and validated with the finite-difference solver, so
the headline ``eta_fem vs checkpoint`` curve and the ``eta_pinn vs eta_fem``
trustworthiness scatter can be built from the logged CSVs.

Outputs (under ``results/joint/<backend>_seed<k>/``):
  result.json, model.pt, geometry/checkpoint_<step>.npz
and appends rows to ``results/joint/metrics.csv`` and ``checkpoints.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config, apply_overrides, load_config
from ..geometry.boundary_sampling import sample_batch
from ..geometry.export_fem import deformed_electrode_polygons
from ..geometry.four_rod import build_trap
from ..losses.geometry_constraints import max_curvature, min_det_jacobian, min_gap
from ..losses.objective import central_quadrupole_loss, compute_total_loss, eta_loss
from ..models.backends import build_backend
from ..models.composed import ComposedField
from ..models.shape_network import build_shape_network
from ..physics import eta as eta_mod
from ..physics.fem_reference import fem_eta
from .checkpoint import append_csv, save_checkpoint, write_json
from .schedules import build_scheduler

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

METRIC_FIELDS = [
    "backend", "seed", "step", "eta_pinn", "eta_fem",
    "loss_pde", "loss_bc", "loss_eta", "loss_quad", "loss_J", "loss_disp", "loss_gap", "loss_curv",
    "grad_phi_eta", "grad_phi_quad", "disp_mean", "disp_max",
    "min_detJ", "min_gap", "max_curv", "wall_time_sec",
]
CHECKPOINT_FIELDS = [
    "backend", "seed", "step", "checkpoint_path", "geometry_path",
    "fem_result_path", "valid_geometry", "notes",
]


def _build_optimizer(model: ComposedField, config: Config) -> torch.optim.Optimizer:
    opt = config.optimizer
    backend = model.backend
    backend_groups = backend.param_groups(
        opt.lr_theta, lr_centers=opt.lr_pig_centers, lr_covariances=opt.lr_pig_covariances
    )
    groups = list(backend_groups)
    groups.append({"params": list(model.shape_network.parameters()), "lr": opt.lr_phi})
    return torch.optim.Adam(groups)


def _identity_pretrain(model: ComposedField, config: Config) -> None:
    """Warm-start the shape network toward identity (plan stage 0).

    With the zero-initialized final layer the residual map already starts at
    identity, so this only needs to keep it there; we take a few steps anchoring
    ``NN_phi(z) ~ z`` on random interior points for robustness when the final
    layer is not zero (e.g. sine activation experiments).
    """
    steps = config.shape_network.identity_pretrain_steps
    if steps <= 0:
        return
    trap = build_trap(config)
    rng = np.random.default_rng(config.experiment.seed + 11)
    optimizer = torch.optim.Adam(model.shape_network.parameters(), lr=1e-3)
    for _ in range(min(steps, 2000)):  # cap for practicality; identity is already near-exact
        z = torch.tensor(trap.sample_interior(1024, rng), dtype=torch.float32)
        optimizer.zero_grad()
        loss = torch.mean((model.shape_network(z) - z) ** 2)
        loss.backward()
        optimizer.step()
        if float(loss.detach()) < 1e-10:
            break


def _fem_validate(model: ComposedField, trap, config: Config, run_dir: Path, step: int):
    geom = deformed_electrode_polygons(model.shape_network, trap)
    fem = fem_eta(
        geom, config.experiment.r0, config.experiment.V0,
        grid=config.validation.fem_grid, center_radius=config.sampling.center_probe_radius,
    )
    geom_dir = run_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    geom_path = geom_dir / f"checkpoint_{step}.npz"
    np.savez(
        geom_path,
        **{f"electrode_{i}": poly for i, (poly, _) in enumerate(geom.electrodes)},
        voltages=np.array([v for _, v in geom.electrodes]),
    )
    return fem, geom_path


def _shape_grad_norm(loss: torch.Tensor, model: ComposedField) -> float:
    params = [p for p in model.shape_network.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    total = None
    for grad in grads:
        if grad is None:
            continue
        val = torch.sum(grad.detach() ** 2)
        total = val if total is None else total + val
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach())


def _shape_coupling_diagnostics(model: ComposedField, batch, config: Config) -> dict[str, float]:
    exp = config.experiment
    s = config.sampling
    l_eta = eta_loss(model, exp.r0, exp.V0)
    l_quad = central_quadrupole_loss(
        model,
        exp.r0,
        exp.V0,
        radius=s.quadrupole_loss_radius,
        grid=s.quadrupole_loss_grid,
    )
    z = batch.interior.detach()
    with torch.no_grad():
        mapped = model.shape_network(z)
        disp = torch.linalg.vector_norm(mapped - z, dim=1)
    return {
        "grad_phi_eta": _shape_grad_norm(l_eta, model),
        "grad_phi_quad": _shape_grad_norm(l_quad, model),
        "disp_mean": float(torch.mean(disp).detach()),
        "disp_max": float(torch.max(disp).detach()),
    }


def train_joint(config: Config, steps: int | None = None, results_root: Path = RESULTS_ROOT) -> dict:
    torch.manual_seed(config.experiment.seed)
    rng = np.random.default_rng(config.experiment.seed)

    trap = build_trap(config)
    shape_network = build_shape_network(config, trap)
    backend = build_backend(config)
    model = ComposedField(shape_network, backend).float()

    _identity_pretrain(model, config)
    optimizer = _build_optimizer(model, config)
    total_steps = steps if steps is not None else config.optimizer.steps
    scheduler = build_scheduler(optimizer, config.optimizer.scheduler, total_steps)

    s = config.sampling
    exp = config.experiment
    center_grid = trap.center_probe_grid(s.center_probe_radius, s.center_probe_grid)

    backend_name = config.physics_backend.type
    tag = f"{backend_name}_seed{exp.seed}"
    run_dir = results_root / "joint" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    history = []
    skipped = 0
    for step in range(total_steps):
        if step % s.resample_every == 0:
            batch = sample_batch(
                trap, rng, s.interior_points,
                s.electrode_boundary_points_per_electrode, s.outer_boundary_points,
            )
        optimizer.zero_grad()
        total, terms = compute_total_loss(model, batch, trap, config, include_geometry=True)
        if not np.isfinite(terms["total"]):
            skipped += 1
            continue
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        do_fem = step % config.validation.fem_checkpoint_every == 0 or step + 1 == total_steps
        if step % 200 == 0 or do_fem:
            eta_pinn = eta_mod.eta_from_backend_on_grid(
                backend, center_grid, exp.r0, exp.V0, s.center_probe_radius
            )["eta_fit"]
            detj = min_det_jacobian(model.shape_network, batch.interior)
            coupling = _shape_coupling_diagnostics(model, batch, config)
            row = {
                "backend": backend_name, "seed": exp.seed, "step": step,
                "eta_pinn": eta_pinn,
                "loss_pde": terms["pde"], "loss_bc": terms["bc"], "loss_eta": terms["eta_obj"],
                "loss_quad": terms.get("quadrupole", 0.0),
                "loss_J": terms.get("jacobian", 0.0), "loss_gap": terms.get("gap", 0.0),
                "loss_disp": terms.get("displacement", 0.0),
                "loss_curv": terms.get("curvature", 0.0),
                "grad_phi_eta": coupling["grad_phi_eta"],
                "grad_phi_quad": coupling["grad_phi_quad"],
                "disp_mean": coupling["disp_mean"],
                "disp_max": coupling["disp_max"],
                "min_detJ": detj,
                "min_gap": min_gap(model.shape_network, trap),
                "max_curv": max_curvature(model.shape_network, trap),
                "wall_time_sec": time.time() - start,
            }
            if do_fem:
                fem, geom_path = _fem_validate(model, trap, config, run_dir, step)
                row["eta_fem"] = fem["eta_fem"]
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
                save_checkpoint(model, checkpoint_path, {"step": step, "eta_fem": fem["eta_fem"], "min_detJ": detj})
                valid = detj > config.validation.valid_min_detJ
                append_csv(results_root / "joint" / "checkpoints.csv", {
                    "backend": backend_name, "seed": exp.seed, "step": step,
                    "checkpoint_path": str(checkpoint_path),
                    "geometry_path": str(geom_path),
                    "fem_result_path": str(run_dir / "result.json"),
                    "valid_geometry": valid, "notes": "",
                }, CHECKPOINT_FIELDS)
                print(f"[joint:{backend_name}] step {step:6d} eta_pinn={eta_pinn:.4f} "
                      f"eta_fem={fem['eta_fem']:.4f} detJ={detj:.3f}", flush=True)
            append_csv(results_root / "joint" / "metrics.csv", row, METRIC_FIELDS)
            history.append(row)

    save_checkpoint(model, run_dir / "model.pt")
    final_eta = eta_mod.eta_from_backend_on_grid(
        backend, center_grid, exp.r0, exp.V0, s.center_probe_radius
    )
    fem, _ = _fem_validate(model, trap, config, run_dir, total_steps)
    fem.pop("_solved", None)
    result = {
        "backend": backend_name, "seed": exp.seed,
        "final_eta_pinn": final_eta, "final_eta_fem": fem["eta_fem"],
        "skipped_nonfinite_steps": skipped,
        "history": history, "config": config.to_dict(),
    }
    write_json(run_dir / "result.json", result)
    print(f"[joint:{backend_name}] DONE eta_pinn={final_eta['eta_fit']:.4f} eta_fem={fem['eta_fem']:.4f}")
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
    train_joint(config, steps=args.steps)


if __name__ == "__main__":
    main()
