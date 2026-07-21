#!/usr/bin/env python3
"""Derivative-aware benchmark for an existing circular-trap PINN run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

SPINN_ROOT = Path(__file__).resolve().parents[2]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

from PINN.common.backends import ScatteredFieldBackend, TorchFieldBackend
from PINN.common.metrics import (
    error_statistics,
    fit_local_quadratic,
    minimax_escape_barrier,
    quadrupole_metrics,
)
from PINN.common.tecplot import read_devsim_tecplot


DEFAULT_DEVSIM = SPINN_ROOT / "DEVSIM" / "Circular_trap" / "paul_trap_basis_RF.dat"


def load_run_model(run_dir: Path):
    script_path = run_dir / "Paul_Trap_PINN.py"
    if not script_path.exists():
        script_path = SPINN_ROOT / "PINN" / "Vanilla_Paul_Trap_PINN" / "Paul_Trap_PINN.py"
    spec = importlib.util.spec_from_file_location("benchmark_run_model", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load model definition from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((run_dir / "config.json").read_text())
    module.set_global_config(config)
    model = module.MLP(
        width=config["model"]["width"],
        depth=config["model"]["depth"],
        activation=config["model"].get("activation", "tanh"),
    )
    state = torch.load(run_dir / "model_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return module, config, model


def electrode_mask(x: np.ndarray, y: np.ndarray, geometry: dict) -> np.ndarray:
    mask = np.zeros_like(x, dtype=bool)
    centers = [
        (0.0, geometry["N_distance"]),
        (0.0, -geometry["S_distance"]),
        (-geometry["W_distance"], 0.0),
        (+geometry["E_distance"], 0.0),
    ]
    for center_x, center_y in centers:
        mask |= (x - center_x) ** 2 + (y - center_y) ** 2 <= geometry["radius"] ** 2
    return mask


def pseudopotential_eV(
    gradient_per_cm: np.ndarray,
    config: dict,
) -> np.ndarray:
    physics = config["physics"]
    charge = physics["e_charge"]
    mass = physics["ion_mass_amu"] * physics["amu"]
    omega = 2.0 * np.pi * physics["rf_frequency_hz"]
    coefficient = charge**2 / (4.0 * mass * omega**2) / charge
    gradient_per_m = gradient_per_cm / physics["geom_unit_to_m"]
    return coefficient * np.sum(gradient_per_m**2, axis=1)


def escape_metric(
    backend,
    config: dict,
    *,
    nx: int,
    ny: int,
) -> dict[str, object]:
    geometry = config["geometry"]
    x = np.linspace(-geometry["world_size_x"] / 2.0, geometry["world_size_x"] / 2.0, nx)
    y = np.linspace(-geometry["world_size_y"] / 2.0, geometry["world_size_y"] / 2.0, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    gradient = backend.gradient(points)
    values = pseudopotential_eV(gradient, config).reshape(ny, nx)
    valid = ~electrode_mask(grid_x, grid_y, geometry)
    start = (int(np.argmin(np.abs(y))), int(np.argmin(np.abs(x))))
    result = minimax_escape_barrier(values, valid, start)
    return {
        "status": "grid_diagnostic_not_certified",
        "barrier_eV": result.barrier,
        "saddle_value_eV": result.saddle_value,
        "start_value_eV": result.start_value,
        "target_index": list(result.target_index),
        "grid_shape": [ny, nx],
        "grid_spacing_cm": [
            float(geometry["world_size_y"] / (ny - 1)),
            float(geometry["world_size_x"] / (nx - 1)),
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--devsim-rf", type=Path, default=DEFAULT_DEVSIM)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--fit-radius", type=float, default=0.035)
    parser.add_argument("--escape-nx", type=int, default=161)
    parser.add_argument("--escape-ny", type=int, default=129)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    module, config, model = load_run_model(run_dir)
    pinn = TorchFieldBackend(model)
    data = read_devsim_tecplot(args.devsim_rf)
    points = np.column_stack([data["x"], data["y"]])
    gradient = None
    if "Potential_gradx" in data and "Potential_grady" in data:
        gradient = np.column_stack([data["Potential_gradx"], data["Potential_grady"]])
    devsim = ScatteredFieldBackend(points, data["Potential"], gradient)

    rng = np.random.default_rng(0)
    if len(points) > args.max_points:
        sample_indices = rng.choice(len(points), args.max_points, replace=False)
        sample_points = points[sample_indices]
    else:
        sample_points = points
    potential_pinn = pinn.potential(sample_points)
    potential_devsim = devsim.potential(sample_points)
    field_pinn = -pinn.gradient(sample_points)
    field_devsim = -devsim.gradient(sample_points)

    pinn_center_hessian = pinn.hessian(np.zeros((1, 2)))[0]
    devsim_fit = fit_local_quadratic(
        points,
        data["Potential"],
        radius=args.fit_radius,
    )
    characteristic_radius = min(
        config["geometry"]["N_distance"],
        config["geometry"]["S_distance"],
        config["geometry"]["W_distance"],
        config["geometry"]["E_distance"],
    ) - config["geometry"]["radius"]
    voltage_span = max(config["boundary_conditions"]["rf_voltages"].values()) - min(
        config["boundary_conditions"]["rf_voltages"].values()
    )
    metrics = {
        "run_dir": str(run_dir),
        "devsim_rf": str(args.devsim_rf.resolve()),
        "sample_count": int(len(sample_points)),
        "potential_V": error_statistics(potential_pinn, potential_devsim),
        "electric_field_V_per_cm": error_statistics(field_pinn, field_devsim),
        "pinn_center": quadrupole_metrics(
            pinn_center_hessian,
            characteristic_radius,
            voltage_span,
            geometry_unit_to_m=config["physics"]["geom_unit_to_m"],
            ion_mass_amu=config["physics"]["ion_mass_amu"],
            rf_frequency_hz=config["physics"]["rf_frequency_hz"],
        ),
        "devsim_center": {
            **quadrupole_metrics(
                np.asarray(devsim_fit["hessian"]),
                characteristic_radius,
                voltage_span,
                geometry_unit_to_m=config["physics"]["geom_unit_to_m"],
                ion_mass_amu=config["physics"]["ion_mass_amu"],
                rf_frequency_hz=config["physics"]["rf_frequency_hz"],
            ),
            "quadratic_fit_rmse_V": devsim_fit["fit_rmse"],
            "quadratic_fit_points": devsim_fit["n_points"],
        },
        "pinn_escape": escape_metric(
            pinn,
            config,
            nx=args.escape_nx,
            ny=args.escape_ny,
        ),
        "devsim_escape": escape_metric(
            devsim,
            config,
            nx=args.escape_nx,
            ny=args.escape_ny,
        ),
    }
    pinn_efficiency = metrics["pinn_center"]["quadrupole_efficiency"]
    devsim_efficiency = metrics["devsim_center"]["quadrupole_efficiency"]
    metrics["center_efficiency_relative_error"] = abs(
        pinn_efficiency - devsim_efficiency
    ) / max(abs(devsim_efficiency), 1e-15)
    pinn_hessian = np.asarray(metrics["pinn_center"]["potential_hessian"])
    devsim_hessian = np.asarray(metrics["devsim_center"]["potential_hessian"])
    metrics["center_hessian_relative_frobenius_error"] = float(
        np.linalg.norm(pinn_hessian - devsim_hessian)
        / max(np.linalg.norm(devsim_hessian), 1e-15)
    )
    pinn_frequencies = np.asarray(metrics["pinn_center"]["secular_frequencies_Hz"])
    devsim_frequencies = np.asarray(metrics["devsim_center"]["secular_frequencies_Hz"])
    metrics["secular_frequency_relative_l2"] = float(
        np.linalg.norm(pinn_frequencies - devsim_frequencies)
        / max(np.linalg.norm(devsim_frequencies), 1e-15)
    )
    voltage_span = max(config["boundary_conditions"]["rf_voltages"].values()) - min(
        config["boundary_conditions"]["rf_voltages"].values()
    )
    metrics["single_seed_derivative_gate"] = {
        "potential_mae_percent_of_voltage": float(
            100.0 * metrics["potential_V"]["mae"] / voltage_span
        ),
        "potential_mae_below_0_5_percent": bool(
            100.0 * metrics["potential_V"]["mae"] / voltage_span < 0.5
        ),
        "field_relative_l2_below_5_percent": bool(
            metrics["electric_field_V_per_cm"]["relative_l2"] < 0.05
        ),
        "hessian_relative_error_below_5_percent": bool(
            metrics["center_hessian_relative_frobenius_error"] < 0.05
        ),
        "efficiency_relative_error_below_5_percent": bool(
            metrics["center_efficiency_relative_error"] < 0.05
        ),
        "frequency_relative_error_below_5_percent": bool(
            metrics["secular_frequency_relative_l2"] < 0.05
        ),
    }
    gate_values = [
        value
        for key, value in metrics["single_seed_derivative_gate"].items()
        if key != "potential_mae_percent_of_voltage"
    ]
    metrics["single_seed_derivative_gate"]["passed"] = bool(all(gate_values))
    metrics["validation_notes"] = [
        "The fixed-geometry gate is single-seed evidence; multi-seed confirmation remains required.",
        "Escape barrier is a structured-grid diagnostic and is excluded from the acceptance gate.",
    ]

    output_path = run_dir / "fixed_geometry_benchmark.json"
    output_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
