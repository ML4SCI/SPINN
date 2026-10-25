"""Jointly optimize the field backend and coordinate-based trap geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..geometry.export_fem import deformed_electrode_polygons
from ..geometry.four_rod import FourRodTrap, build_trap
from ..losses.geometry_constraints import max_curvature, min_det_jacobian, min_gap
from ..losses.objective import (
    central_quadrupole_loss,
    compute_total_loss,
    differentiable_eta,
)
from ..models.backends import build_backend
from ..models.composed import ComposedField
from ..models.shape_network import build_shape_network
from ..physics.eta import eta_from_backend_on_grid
from ..physics.fem_reference import fem_eta
from .runtime import (
    RESULTS_ROOT,
    RunContext,
    append_csv,
    apply_gradient_step,
    build_scheduler,
    parse_training_config,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    seed_everything,
    training_batches,
    write_json,
)

METRIC_FIELDS = [
    "backend",
    "seed",
    "step",
    "eta_pinn",
    "eta_fem",
    "loss_pde",
    "loss_bc",
    "loss_eta",
    "loss_quad",
    "loss_J",
    "loss_disp",
    "loss_gap",
    "loss_curv",
    "grad_phi_eta",
    "grad_phi_quad",
    "disp_mean",
    "disp_max",
    "min_detJ",
    "min_gap",
    "max_curv",
    "wall_time_sec",
]
CHECKPOINT_FIELDS = [
    "backend",
    "seed",
    "step",
    "checkpoint_path",
    "geometry_path",
    "fem_result_path",
    "valid_geometry",
    "notes",
]
# CSV column -> compute_total_loss term
_LOSS_COLUMNS = {
    "loss_pde": "pde",
    "loss_bc": "bc",
    "loss_eta": "eta_obj",
    "loss_quad": "quadrupole",
    "loss_J": "jacobian",
    "loss_gap": "gap",
    "loss_disp": "displacement",
    "loss_curv": "curvature",
}
# Identity pretraining is capped: the zero-initialized head already starts exact.
_MAX_IDENTITY_PRETRAIN_STEPS = 2000


def _build_optimizer(model: ComposedField, config: Config) -> torch.optim.Optimizer:
    opt = config.optimizer
    groups = model.backend.param_groups(
        opt.lr_theta,
        lr_centers=opt.lr_pig_centers,
        lr_covariances=opt.lr_pig_covariances,
    )
    groups.append({"params": list(model.shape_network.parameters()), "lr": opt.lr_phi})
    return torch.optim.Adam(groups)


def _identity_pretrain(
    model: ComposedField,
    trap: FourRodTrap,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Warm-start the shape network toward identity (plan stage 0).

    With the zero-initialized final layer the residual map already starts at
    identity, so this only keeps it there; it matters when the final layer is
    not zero (e.g. sine activation experiments).
    """
    steps = min(
        config.shape_network.identity_pretrain_steps, _MAX_IDENTITY_PRETRAIN_STEPS
    )
    rng = np.random.default_rng(config.experiment.seed + 11)
    optimizer = torch.optim.Adam(model.shape_network.parameters(), lr=1e-3)
    for _ in range(steps):
        z = torch.as_tensor(trap.sample_interior(1024, rng), device=device, dtype=dtype)
        optimizer.zero_grad()
        loss = torch.mean((model.shape_network(z) - z) ** 2)
        loss.backward()
        optimizer.step()
        if float(loss.detach()) < 1e-10:
            break


def _fem_validate(
    model: ComposedField, trap: FourRodTrap, config: Config, run_dir: Path, step: int
) -> tuple[dict, Path]:
    """FD-validate the current deformed geometry and export its electrode polygons."""
    geom = deformed_electrode_polygons(model.shape_network, trap)
    fem = fem_eta(
        geom,
        config.experiment.r0,
        config.experiment.V0,
        grid=config.validation.fem_grid,
        center_radius=config.sampling.center_probe_radius,
    )
    geom_path = run_dir / "geometry" / f"checkpoint_{step}.npz"
    geom_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        geom_path,
        **{f"electrode_{i}": poly for i, (poly, _) in enumerate(geom.electrodes)},
        voltages=np.array([v for _, v in geom.electrodes]),
    )
    return fem, geom_path


def shape_grad_norm(loss: torch.Tensor, model: ComposedField) -> float:
    """``||d loss / d phi||_2`` over the shape-network parameters (0 if disconnected)."""
    params = [p for p in model.shape_network.parameters() if p.requires_grad]
    grads = [
        g for g in torch.autograd.grad(loss, params, allow_unused=True) if g is not None
    ]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads)))


def _shape_coupling_diagnostics(model: ComposedField, batch, config: Config) -> dict:
    exp, s = config.experiment, config.sampling
    l_eta = -differentiable_eta(model, exp.r0, exp.V0)
    l_quad = central_quadrupole_loss(
        model,
        exp.r0,
        exp.V0,
        radius=s.quadrupole_loss_radius,
        grid=s.quadrupole_loss_grid,
    )
    z = batch.interior.detach()
    with torch.no_grad():
        disp = torch.linalg.vector_norm(model.shape_network(z) - z, dim=1)
    return {
        "grad_phi_eta": shape_grad_norm(l_eta, model),
        "grad_phi_quad": shape_grad_norm(l_quad, model),
        "disp_mean": float(torch.mean(disp)),
        "disp_max": float(torch.max(disp)),
    }


def train_joint(
    config: Config,
    steps: int | None = None,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    """Run joint field/shape optimization and periodic FD validation."""
    config.validate()
    rng = seed_everything(config)
    device = resolve_device(config)
    dtype = resolve_dtype(config)
    s, exp = config.sampling, config.experiment
    name = config.physics_backend.type
    label = f"joint:{name}"

    trap = build_trap(config)
    shape_network = build_shape_network(config, trap)
    backend = build_backend(config)
    model = ComposedField(shape_network, backend).to(device=device, dtype=dtype)

    _identity_pretrain(model, trap, config, device=device, dtype=dtype)
    optimizer = _build_optimizer(model, config)
    total_steps = steps if steps is not None else config.optimizer.steps
    scheduler = build_scheduler(optimizer, config.optimizer.scheduler, total_steps)
    center_grid = trap.center_probe_grid(s.center_probe_radius, s.center_probe_grid)
    run = RunContext.create("joint", config, results_root)

    def pinn_eta() -> dict:
        return eta_from_backend_on_grid(
            backend,
            center_grid,
            exp.r0,
            exp.V0,
            s.center_probe_radius,
            device=device,
            dtype=dtype,
        )

    history = []
    skipped = 0
    for step, batch in training_batches(
        config, trap, rng, total_steps, device=device, dtype=dtype
    ):
        optimizer.zero_grad()
        total, terms = compute_total_loss(model, batch, trap, config)
        if not apply_gradient_step(
            total,
            model,
            optimizer,
            scheduler,
            config,
            label=label,
            step=step,
            terms=terms,
        ):
            skipped += 1
            continue

        do_fem = (
            step % config.validation.fem_checkpoint_every == 0
            or step + 1 == total_steps
        )
        if not (step % config.runtime.log_every == 0 or do_fem):
            continue

        eta_pinn = pinn_eta()["eta_fit"]
        detj = min_det_jacobian(shape_network, batch.interior)
        row = {
            "backend": name,
            "seed": exp.seed,
            "step": step,
            "eta_pinn": eta_pinn,
            **{column: terms[term] for column, term in _LOSS_COLUMNS.items()},
            **_shape_coupling_diagnostics(model, batch, config),
            "min_detJ": detj,
            "min_gap": min_gap(shape_network, trap),
            "max_curv": max_curvature(shape_network, trap),
            "wall_time_sec": run.elapsed_seconds,
        }
        if do_fem:
            fem, geom_path = _fem_validate(model, trap, config, run.directory, step)
            row["eta_fem"] = fem["eta_fem"]
            checkpoint_path = run.directory / "checkpoints" / f"model_step_{step}.pt"
            save_checkpoint(
                model,
                checkpoint_path,
                {"step": step, "eta_fem": fem["eta_fem"], "min_detJ": detj},
            )
            append_csv(
                results_root / "joint" / "checkpoints.csv",
                {
                    "backend": name,
                    "seed": exp.seed,
                    "step": step,
                    "checkpoint_path": str(checkpoint_path),
                    "geometry_path": str(geom_path),
                    "fem_result_path": str(run.directory / "result.json"),
                    "valid_geometry": detj > config.validation.valid_min_detJ,
                    "notes": "",
                },
                CHECKPOINT_FIELDS,
            )
            print(
                f"[{label}] step {step:6d} eta_pinn={eta_pinn:.4f} "
                f"eta_fem={fem['eta_fem']:.4f} detJ={detj:.3f}",
                flush=True,
            )
        append_csv(results_root / "joint" / "metrics.csv", row, METRIC_FIELDS)
        history.append(row)

    save_checkpoint(model, run.directory / "model.pt")
    final_eta = pinn_eta()
    fem, _ = _fem_validate(model, trap, config, run.directory, total_steps)
    result = {
        "backend": name,
        "seed": exp.seed,
        "final_eta_pinn": final_eta,
        "final_eta_fem": fem["eta_fem"],
        "skipped_nonfinite_steps": skipped,
        "steps": total_steps,
        "history": history,
        "config": config.to_dict(),
    }
    write_json(run.directory / "result.json", result)
    run.finish(
        eta_pinn=final_eta["eta_fit"],
        eta_fem=fem["eta_fem"],
        skipped_nonfinite_steps=skipped,
    )
    print(
        f"[{label}] DONE eta_pinn={final_eta['eta_fit']:.4f} eta_fem={fem['eta_fem']:.4f}"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    config, steps = parse_training_config(__doc__, argv)
    train_joint(config, steps=steps)


if __name__ == "__main__":
    main()
