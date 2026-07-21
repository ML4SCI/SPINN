#!/usr/bin/env python3
"""Run one shape-optimization variation and save figures and metrics.

    python -m experiments.cpn_pinn_shape_optimization.run --variation C --steps 4000

Variations A (PINN-first) and C (joint) are fully wired; B, D, and E are
selectable scaffolds that run but are not tuned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPINN_ROOT = Path(__file__).resolve().parents[2]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

import numpy as np
import torch

from PINN.common.metrics import fit_local_quadratic, quadrupole_metrics

from experiments.cpn_pinn_shape_optimization import plotting
from experiments.cpn_pinn_shape_optimization.config import make_preset
from experiments.cpn_pinn_shape_optimization.geometry import build_geometry, build_sampler
from experiments.cpn_pinn_shape_optimization.train import train

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def center_metrics(result: dict, config) -> dict:
    """Quadrupole metrics from the model's physical field near the trap center."""

    model = result["model"]
    sampler = build_sampler(build_geometry(config.geometry), config.seed + 7)
    reference = torch.tensor(sampler.interior(4096), dtype=torch.float32, requires_grad=True)
    potential, physical = model(reference)
    physical_np = physical.detach().numpy()
    potential_np = potential.detach().numpy().ravel()

    near = np.linalg.norm(physical_np, axis=1) <= config.loss.fit_radius
    if np.count_nonzero(near) < 12:
        return {"note": "too few near-center points for a quadratic fit"}
    fit = fit_local_quadratic(physical_np[near], potential_np[near], radius=config.loss.fit_radius)
    metrics = quadrupole_metrics(
        np.asarray(fit["hessian"]),
        characteristic_radius=config.geometry.east_distance - config.geometry.radius,
        voltage_span=config.physics.rf_voltage,
        ion_mass_amu=config.physics.ion_mass_amu,
        rf_frequency_hz=config.physics.rf_frequency_hz,
    )
    return {"fit_rmse_V": fit["fit_rmse"], **metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variation", default="C")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    config = make_preset(args.variation)
    config.seed = args.seed
    if args.steps is not None:
        config.training.steps = args.steps

    result = train(config)

    run_dir = args.results_dir / config.variation_flags.name
    run_dir.mkdir(parents=True, exist_ok=True)
    plotting.save_all(result, config, run_dir)

    metrics = {
        "variation": config.variation,
        "k_quadratic_eV_per_m2": result["k_quadratic_eV_per_m2"],
        "skipped_nonfinite_steps": result["skipped_steps"],
        "final_losses": result["history"][-1] if result["history"] else {},
        "center_metrics": center_metrics(result, config),
        "config": config.to_dict(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (run_dir / "gradient_diagnostics.json").write_text(
        json.dumps(result["gradient_flow"], indent=2)
    )

    print(f"[run] variation {config.variation}: gradient flow {result['gradient_flow']}")
    print(f"[run] wrote results to {run_dir}")


if __name__ == "__main__":
    main()
