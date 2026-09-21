#!/usr/bin/env python3
"""Run the constrained joint-shape ablation grid.

The grid varies geometry freedom while keeping the central quadrupole-shape
objective enabled. Each run writes into ``spinn/results/ablations/<run>/<backend>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spinn.config import apply_overrides, load_config
from spinn.train.train_joint import train_joint


ABLATIONS = {
    "A_safe": {
        "loss_weights.jacobian": "500.0",
        "loss_weights.displacement": "10.0",
        "optimizer.lr_phi": "2.0e-5",
    },
    "B_more_displacement": {
        "loss_weights.jacobian": "500.0",
        "loss_weights.displacement": "1.0",
        "optimizer.lr_phi": "2.0e-5",
    },
    "C_weaker_jacobian": {
        "loss_weights.jacobian": "100.0",
        "loss_weights.displacement": "10.0",
        "optimizer.lr_phi": "2.0e-5",
    },
    "D_moderate": {
        "loss_weights.jacobian": "100.0",
        "loss_weights.displacement": "1.0",
        "optimizer.lr_phi": "2.0e-5",
    },
    "E_more_shape_lr": {
        "loss_weights.jacobian": "500.0",
        "loss_weights.displacement": "1.0",
        "optimizer.lr_phi": "5.0e-5",
    },
}

PRACTICAL_OVERRIDES = {
    "sampling.interior_points": "2048",
    "sampling.electrode_boundary_points_per_electrode": "256",
    "sampling.outer_boundary_points": "512",
    "sampling.center_probe_grid": "21",
    "sampling.quadrupole_loss_grid": "17",
    "sampling.resample_every": "250",
    "validation.fem_checkpoint_every": "250",
    "validation.fem_grid": "120",
    "validation.valid_min_detJ": "0.2",
}


def _override_list(mapping: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in mapping.items()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backends", nargs="+", default=["mlp", "pixel", "pig"])
    parser.add_argument("--runs", nargs="+", default=list(ABLATIONS))
    parser.add_argument("--results-root", default="spinn/results/ablations")
    args = parser.parse_args(argv)

    root = Path(args.results_root)
    for run_name in args.runs:
        if run_name not in ABLATIONS:
            raise ValueError(f"unknown ablation {run_name!r}; choose from {sorted(ABLATIONS)}")
        run_root = root / run_name
        overrides = _override_list(PRACTICAL_OVERRIDES | ABLATIONS[run_name])
        for backend in args.backends:
            cfg = load_config(f"joint_{backend}.yaml")
            cfg = apply_overrides(cfg, overrides)
            cfg.experiment.seed = args.seed
            cfg.experiment.name = f"{run_name}_{backend}_seed{args.seed}"
            print(f"[ablation:{run_name}:{backend}] steps={args.steps}", flush=True)
            train_joint(cfg, steps=args.steps, results_root=run_root)


if __name__ == "__main__":
    main()
