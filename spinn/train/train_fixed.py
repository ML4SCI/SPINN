"""Train and validate one physics backend on the fixed trap geometry."""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config
from ..geometry.export_fem import reference_geometry
from ..geometry.four_rod import build_trap
from ..losses.boundary import boundary_loss
from ..losses.pde import pde_loss
from ..models.backends import build_backend
from ..models.composed import ComposedField, IdentityShapeNetwork
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

FIXED_FIELDS = [
    "backend",
    "seed",
    "step",
    "loss_pde",
    "loss_bc",
    "eta_pinn",
    "eta_fem",
    "eta_abs_error",
    "wall_time_sec",
]


def _fixed_learning_rates(config: Config) -> dict:
    opt = config.optimizer
    kind = config.physics_backend.type
    if kind == "mlp":
        return {"lr_main": opt.lr_fixed_mlp}
    if kind == "pixel":
        return {"lr_main": opt.lr_fixed_pixel}
    return {
        "lr_main": opt.lr_fixed_pig_decoder,
        "lr_centers": opt.lr_fixed_pig_centers,
        "lr_covariances": opt.lr_fixed_pig_centers,
    }


def train_fixed(
    config: Config,
    steps: int | None = None,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    """Train a backend with an identity shape map and compare it with FD."""
    config.validate()
    rng = seed_everything(config)
    device = resolve_device(config)
    dtype = resolve_dtype(config)
    s, exp, w = config.sampling, config.experiment, config.loss_weights
    name = config.physics_backend.type
    label = f"fixed:{name}"

    trap = build_trap(config)
    backend = build_backend(config)
    model = ComposedField(IdentityShapeNetwork(), backend).to(
        device=device, dtype=dtype
    )
    optimizer = torch.optim.Adam(backend.param_groups(**_fixed_learning_rates(config)))
    total_steps = steps if steps is not None else config.optimizer.steps
    scheduler = build_scheduler(optimizer, config.optimizer.scheduler, total_steps)
    run = RunContext.create("fixed", config, results_root)

    history = []
    for step, batch in training_batches(
        config, trap, rng, total_steps, device=device, dtype=dtype
    ):
        optimizer.zero_grad()
        potential, mapped = model(batch.interior)
        l_pde = pde_loss(
            potential, batch.interior, model.shape_network, mapped_points=mapped
        )
        l_bc = boundary_loss(model, batch)
        loss = w.pde * l_pde + w.bc * l_bc
        if not apply_gradient_step(
            loss, model, optimizer, scheduler, config, label=label, step=step
        ):
            continue

        if step % config.runtime.log_every == 0 or step + 1 == total_steps:
            row = {
                "backend": name,
                "seed": exp.seed,
                "step": step,
                "loss_pde": float(l_pde.detach()),
                "loss_bc": float(l_bc.detach()),
                "wall_time_sec": run.elapsed_seconds,
            }
            history.append(row)
            print(
                f"[{label}] step {step:6d} pde={row['loss_pde']:.3e} bc={row['loss_bc']:.3e}",
                flush=True,
            )

    eta_pinn = eta_from_backend_on_grid(
        backend,
        trap.center_probe_grid(s.center_probe_radius, s.center_probe_grid),
        exp.r0,
        exp.V0,
        s.center_probe_radius,
        device=device,
        dtype=dtype,
    )
    fem = fem_eta(
        reference_geometry(trap),
        exp.r0,
        exp.V0,
        grid=config.validation.fem_grid,
        center_radius=s.center_probe_radius,
    )
    eta_abs_error = abs(eta_pinn["eta_fit"] - fem["eta_fem"])
    last = history[-1] if history else {}

    append_csv(
        results_root / "fixed" / "metrics.csv",
        {
            "backend": name,
            "seed": exp.seed,
            "step": total_steps,
            "loss_pde": last.get("loss_pde", float("nan")),
            "loss_bc": last.get("loss_bc", float("nan")),
            "eta_pinn": eta_pinn["eta_fit"],
            "eta_fem": fem["eta_fem"],
            "eta_abs_error": eta_abs_error,
            "wall_time_sec": run.elapsed_seconds,
        },
        FIXED_FIELDS,
    )
    save_checkpoint(model, run.directory / "model.pt")
    result = {
        "backend": name,
        "seed": exp.seed,
        "eta_pinn": eta_pinn,
        "eta_fem": fem["eta_fem"],
        "eta_abs_error": eta_abs_error,
        "history": history,
        "steps": total_steps,
        "config": config.to_dict(),
        "fem": fem,
    }
    write_json(run.directory / "result.json", result)
    run.finish(
        eta_pinn=eta_pinn["eta_fit"],
        eta_fem=fem["eta_fem"],
        eta_abs_error=eta_abs_error,
    )
    print(
        f"[{label}] eta_pinn={eta_pinn['eta_fit']:.4f} eta_fem={fem['eta_fem']:.4f} |err|={eta_abs_error:.4f}"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    config, steps = parse_training_config(__doc__, argv)
    train_fixed(config, steps=steps)


if __name__ == "__main__":
    main()
