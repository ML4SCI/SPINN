"""Compare the PINN geometry derivative with an independent DEVSIM difference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

SPINN_ROOT = Path(__file__).resolve().parents[2]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

from PINN.Coordinate_Projection.train import (
    build_components,
    efficiency_and_gradient,
)


def finite_difference_derivatives(
    values: list[float],
    epsilon: float,
) -> tuple[float, float]:
    """Return centered three- and five-point first derivatives.

    ``values`` must correspond to ``[-2h, -h, +h, +2h]`` around the target.
    """

    if len(values) != 4:
        raise ValueError("five-point validation requires four offset values")
    minus_two, minus_one, plus_one, plus_two = values
    three_point = (plus_one - minus_one) / (2.0 * epsilon)
    five_point = (
        minus_two - 8.0 * minus_one + 8.0 * plus_one - plus_two
    ) / (12.0 * epsilon)
    return three_point, five_point


def compare_gradient_with_devsim(
    *,
    model,
    coordinate_map,
    config: dict,
    run_dir: Path,
    distance: float,
    epsilon: float,
) -> dict[str, float | list[float]]:
    reference_dir = run_dir / "devsim_gradient_check"
    distances = [
        distance - 2.0 * epsilon,
        distance - epsilon,
        distance + epsilon,
        distance + 2.0 * epsilon,
    ]
    geometry = config["geometry"]
    validation = config["validation"]
    if distances[0] < geometry["distance_min"] or distances[-1] > geometry["distance_max"]:
        raise ValueError(
            "finite-difference stencil leaves the trained geometry interval; "
            "reduce epsilon or move the validation distance"
        )
    command = [
        sys.executable,
        "-m",
        "PINN.Coordinate_Projection.devsim_reference",
        "--distances",
        *(str(value) for value in distances),
        "--output-dir",
        str(reference_dir),
        "--radius",
        str(geometry["radius"]),
        "--world-size-x",
        str(geometry["world_size_x"]),
        "--world-size-y",
        str(geometry["world_size_y"]),
        "--world-mesh-size",
        str(validation["devsim_world_mesh_size"]),
        "--electrode-mesh-size",
        str(validation["devsim_electrode_mesh_size"]),
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])
    records = json.loads((reference_dir / "sweep_metrics.json").read_text())
    devsim_values = [record["quadrupole_efficiency"] for record in records]
    three_point, five_point = finite_difference_derivatives(devsim_values, epsilon)

    efficiency, pinn_derivative, _ = efficiency_and_gradient(
        model,
        coordinate_map,
        distance,
        geometry,
    )
    relative_error = abs(float(pinn_derivative) - five_point) / max(
        abs(five_point),
        1e-12,
    )
    stencil_spread = abs(five_point - three_point) / max(abs(five_point), 1e-12)
    return {
        "distances_cm": distances,
        "devsim_efficiencies": devsim_values,
        "devsim_three_point_derivative_per_cm": three_point,
        "devsim_five_point_derivative_per_cm": five_point,
        "finite_difference_stencil_relative_spread": stencil_spread,
        "devsim_world_mesh_size_cm": validation["devsim_world_mesh_size"],
        "devsim_electrode_mesh_size_cm": validation["devsim_electrode_mesh_size"],
        "pinn_efficiency": float(efficiency.detach()),
        "pinn_autograd_derivative_per_cm": float(pinn_derivative.detach()),
        "relative_derivative_error": relative_error,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--epsilon", type=float)
    args = parser.parse_args(argv)

    bundle = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=False)
    config = bundle["config"]
    model, coordinate_map = build_components(config)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    distance = args.distance or config["optimization"]["start_distance"]
    epsilon = args.epsilon or config["validation"]["gradient_epsilon"]
    result = compare_gradient_with_devsim(
        model=model,
        coordinate_map=coordinate_map,
        config=config,
        run_dir=args.run_dir,
        distance=distance,
        epsilon=epsilon,
    )
    output_path = args.run_dir / "devsim_gradient_comparison.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
