#!/usr/bin/env python3
"""Build a reproducible scientific-audit summary for the presentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SPINN_ROOT = Path(__file__).resolve().parents[2]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

from PINN.Coordinate_Projection.constraints import jacobian_constraints
from PINN.Coordinate_Projection.maps import ElectrodeTranslationMap
from PINN.Coordinate_Projection.validate_gradient import finite_difference_derivatives
from PINN.common.metrics import fit_local_quadratic, quadrupole_metrics
from PINN.common.tecplot import read_devsim_tecplot


DEFAULT_SUMMARY = SPINN_ROOT / "Hyperparameter" / "results" / "summary"
DEFAULT_BEST_RUN = (
    SPINN_ROOT
    / "Hyperparameter"
    / "results"
    / "runs"
    / "bc_weight_pinn_20260602_024202_bc_weight_1000"
)
DEFAULT_DEVSIM = SPINN_ROOT / "DEVSIM" / "Circular_trap" / "paul_trap_basis_RF.dat"


def map_validity() -> list[dict[str, float]]:
    coordinate_map = ElectrodeTranslationMap().double()
    axis = torch.linspace(-0.35, 0.35, 301, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    base = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)
    records = []
    for distance in np.linspace(0.11, 0.145, 8):
        points = base.clone().requires_grad_(True)
        parameter = torch.full((len(points), 1), distance, dtype=torch.float64)
        diagnostics = jacobian_constraints(coordinate_map, points, parameter)
        records.append({
            "distance_cm": float(distance),
            "minimum_determinant": float(diagnostics["determinant"].min().detach()),
            "maximum_condition": float(diagnostics["condition"].max().detach()),
        })
    return records


def fit_radius_sensitivity(devsim_path: Path) -> list[dict[str, float]]:
    data = read_devsim_tecplot(devsim_path)
    points = np.column_stack([data["x"], data["y"]])
    records = []
    for radius in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06]:
        fit = fit_local_quadratic(points, data["Potential"], radius=radius)
        metrics = quadrupole_metrics(
            np.asarray(fit["hessian"]),
            characteristic_radius=0.075,
            voltage_span=300.0,
        )
        records.append({
            "fit_radius_cm": radius,
            "n_points": int(fit["n_points"]),
            "fit_rmse_V": float(fit["fit_rmse"]),
            "quadrupole_efficiency": float(metrics["quadrupole_efficiency"]),
        })
    return records


def gradient_record(label: str, sweep_path: Path) -> dict[str, object]:
    records = json.loads((sweep_path / "sweep_metrics.json").read_text())
    by_distance = {round(record["distance_cm"], 8): record for record in records}
    distances = sorted(by_distance)
    result: dict[str, object] = {
        "label": label,
        "distances_cm": distances,
        "efficiencies": [by_distance[value]["quadrupole_efficiency"] for value in distances],
    }
    if len(distances) == 2:
        lower, upper = distances
        result["three_point_derivative_per_cm"] = (
            by_distance[upper]["quadrupole_efficiency"]
            - by_distance[lower]["quadrupole_efficiency"]
        ) / (upper - lower)
    elif len(distances) >= 4:
        stencil_distances = [distances[0], distances[1], distances[-2], distances[-1]]
        center = 0.5 * (stencil_distances[0] + stencil_distances[-1])
        epsilon = stencil_distances[1] - stencil_distances[0]
        expected = [
            round(center - 2.0 * epsilon, 8),
            round(center - epsilon, 8),
            round(center + epsilon, 8),
            round(center + 2.0 * epsilon, 8),
        ]
        values = [by_distance[value]["quadrupole_efficiency"] for value in expected]
        three_point, five_point = finite_difference_derivatives(values, epsilon)
        result["three_point_derivative_per_cm"] = three_point
        result["five_point_derivative_per_cm"] = five_point
        result["stencil_relative_spread"] = abs(five_point - three_point) / abs(five_point)
    else:
        result["note"] = "sweep does not match a centered two- or four-offset stencil"
    return result


def escape_convergence(paths: list[Path]) -> list[dict[str, object]]:
    records = []
    for path in paths:
        data = json.loads(path.read_text())
        records.append({
            "grid_shape": data["pinn_escape"]["grid_shape"],
            "pinn_barrier_eV": data["pinn_escape"]["barrier_eV"],
            "devsim_barrier_eV": data["devsim_escape"]["barrier_eV"],
            "pinn_target_index": data["pinn_escape"]["target_index"],
            "devsim_target_index": data["devsim_escape"]["target_index"],
        })
    return records


def plot_audit(audit: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for gradient in audit["gradient_convergence"]:
        axes[0].plot(
            gradient["distances_cm"],
            gradient["efficiencies"],
            marker="o",
            label=gradient["label"],
        )
    axes[0].set_xlabel("electrode-center distance [cm]")
    axes[0].set_ylabel("quadrupole efficiency")
    axes[0].set_title("Independent DEVSIM shape sweep")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    validity = audit["coordinate_map_validity"]
    distance = [row["distance_cm"] for row in validity]
    determinant = [row["minimum_determinant"] for row in validity]
    condition = [row["maximum_condition"] for row in validity]
    axes[1].plot(distance, determinant, marker="o", label=r"min det$(J)$")
    axes[1].plot(distance, condition, marker="s", label=r"max cond$(J)$")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("electrode-center distance [cm]")
    axes[1].set_title("Coordinate-map validity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-run", type=Path, default=DEFAULT_BEST_RUN)
    parser.add_argument("--devsim-rf", type=Path, default=DEFAULT_DEVSIM)
    parser.add_argument(
        "--gradient-sweep",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIRECTORY"),
        default=[],
    )
    parser.add_argument("--escape-benchmark", type=Path, action="append", default=[])
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)

    benchmark_path = args.best_run / "fixed_geometry_benchmark.json"
    if not benchmark_path.exists():
        raise SystemExit(f"run fixed_geometry_benchmark.py first: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text())
    gradients = [
        gradient_record(label, Path(directory))
        for label, directory in args.gradient_sweep
    ]
    fit_records = fit_radius_sensitivity(args.devsim_rf)
    audit = {
        "status": {
            "fixed_geometry_single_seed": "passes derivative-aware gate",
            "coordinate_projection_operators": "manufactured tests pass",
            "coordinate_projection_training": "smoke-tested only; no converged result",
            "escape_depth": (
                "2D cross-sectional grid diagnostic stabilizes after refinement; "
                "not equivalent to a physical 3D trap depth"
            ),
        },
        "best_fixed_geometry": benchmark,
        "coordinate_map_validity": map_validity(),
        "devsim_fit_radius_sensitivity": fit_records,
        "fit_radius_efficiency_relative_range": (
            max(row["quadrupole_efficiency"] for row in fit_records)
            - min(row["quadrupole_efficiency"] for row in fit_records)
        ) / np.mean([row["quadrupole_efficiency"] for row in fit_records]),
        "gradient_convergence": gradients,
        "escape_grid_convergence": escape_convergence(args.escape_benchmark),
    }
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.summary_dir / "experiment_audit.json"
    output_plot = args.summary_dir / "coordinate_projection_audit.png"
    output_json.write_text(json.dumps(audit, indent=2))
    plot_audit(audit, output_plot)
    print(f"Saved audit to {output_json}")
    print(f"Saved audit plot to {output_plot}")


if __name__ == "__main__":
    main()
