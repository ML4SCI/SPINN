#!/usr/bin/env python3
"""Train and optimize a parametric coordinate-projection Paul-trap PINN."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SPINN_ROOT = Path(__file__).resolve().parents[2]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

from PINN.Coordinate_Projection.constraints import jacobian_constraints
from PINN.Coordinate_Projection.maps import ElectrodeTranslationMap
from PINN.Coordinate_Projection.model import ParametricPotentialMLP
from PINN.Coordinate_Projection.operators import physical_hessian, pullback_laplacian


DEFAULT_CONFIG = {
    "seed": 0,
    "output_root": "PINN/Coordinate_Projection/results",
    "run_name": "coordinate_projection",
    "device": "cpu",
    "dtype": "float32",
    "geometry": {
        "world_size_x": 2.5,
        "world_size_y": 2.0,
        "reference_distance": 0.125,
        "distance_min": 0.11,
        "distance_max": 0.145,
        "radius": 0.05,
        "plateau_margin": 0.005,
        "support_radius": 0.12,
        "minimum_gap": 0.045,
    },
    "model": {
        "width": 128,
        "depth": 4,
        "activation": "tanh",
    },
    "training": {
        "steps": 10000,
        "lr": 0.001,
        "n_interior": 2048,
        "n_map": 512,
        "n_world": 512,
        "n_electrode_per_electrode": 256,
        "bc_weight": 1000.0,
        "jacobian_weight": 10.0,
        "gap_weight": 100.0,
        "minimum_determinant": 0.25,
        "maximum_condition": 5.0,
        "grad_clip": 1.0,
        "print_every": 100,
    },
    "optimization": {
        "start_distance": 0.125,
        "steps": 200,
        "lr": 0.001,
    },
    "validation": {
        "n_interior": 4096,
        "gradient_epsilon": 0.0025,
        "run_devsim": False,
        "devsim_world_mesh_size": 0.025,
        "devsim_electrode_mesh_size": 0.0025,
    },
}


def deep_update(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config = deep_update(config, json.loads(path.read_text()))
    return config


def torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def sample_distances(
    count: int,
    geometry: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        geometry["distance_min"]
        + (geometry["distance_max"] - geometry["distance_min"])
        * torch.rand(count, 1, device=device, dtype=dtype)
    )


def inside_reference_electrodes(points: torch.Tensor, geometry: dict) -> torch.Tensor:
    distance = geometry["reference_distance"]
    centers = points.new_tensor([
        [distance, 0.0],
        [-distance, 0.0],
        [0.0, distance],
        [0.0, -distance],
    ])
    squared_distance = torch.sum((points[:, None, :] - centers[None, :, :]) ** 2, dim=2)
    tolerance = 32.0 * torch.finfo(points.dtype).eps
    return torch.any(
        squared_distance <= geometry["radius"] ** 2 * (1.0 + tolerance),
        dim=1,
    )


def sample_interior(
    count: int,
    geometry: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    half_x = geometry["world_size_x"] / 2.0
    half_y = geometry["world_size_y"] / 2.0
    batches = []
    found = 0
    while found < count:
        candidates = torch.empty(max(2 * (count - found), 64), 2, device=device, dtype=dtype)
        candidates[:, 0].uniform_(-half_x, half_x)
        candidates[:, 1].uniform_(-half_y, half_y)
        accepted = candidates[~inside_reference_electrodes(candidates, geometry)]
        batches.append(accepted)
        found += len(accepted)
    return torch.cat(batches, dim=0)[:count].requires_grad_(True)


def sample_world_boundary(
    count: int,
    geometry: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    half_x = geometry["world_size_x"] / 2.0
    half_y = geometry["world_size_y"] / 2.0
    sides = torch.randint(0, 4, (count,), device=device)
    points = torch.empty(count, 2, device=device, dtype=dtype)
    points[:, 0].uniform_(-half_x, half_x)
    points[:, 1].uniform_(-half_y, half_y)
    points[sides == 0, 1] = -half_y
    points[sides == 1, 1] = +half_y
    points[sides == 2, 0] = -half_x
    points[sides == 3, 0] = +half_x
    return points


def sample_map_transition(
    count: int,
    geometry: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Concentrate validity checks where the coordinate map changes fastest."""

    distance = geometry["reference_distance"]
    centers = torch.tensor(
        [
            (+distance, 0.0),
            (-distance, 0.0),
            (0.0, +distance),
            (0.0, -distance),
        ],
        device=device,
        dtype=dtype,
    )
    center_indices = torch.randint(0, len(centers), (count,), device=device)
    angle = 2.0 * torch.pi * torch.rand(count, device=device, dtype=dtype)
    lower = geometry["radius"] + geometry["plateau_margin"]
    radial = lower + (geometry["support_radius"] - lower) * torch.rand(
        count,
        device=device,
        dtype=dtype,
    )
    offset = torch.stack([radial * torch.cos(angle), radial * torch.sin(angle)], dim=1)
    return (centers[center_indices] + offset).requires_grad_(True)


def sample_electrode_boundaries(
    count_per_electrode: int,
    geometry: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = geometry["reference_distance"]
    centers = [
        (+distance, 0.0),
        (-distance, 0.0),
        (0.0, +distance),
        (0.0, -distance),
    ]
    values = [1.0, 1.0, 0.0, 0.0]
    point_batches = []
    value_batches = []
    for center, value in zip(centers, values):
        angle = 2.0 * torch.pi * torch.rand(count_per_electrode, device=device, dtype=dtype)
        points = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1) * geometry["radius"]
        points = points + points.new_tensor(center)
        point_batches.append(points)
        value_batches.append(torch.full((count_per_electrode, 1), value, device=device, dtype=dtype))
    return torch.cat(point_batches), torch.cat(value_batches)


def build_components(config: dict) -> tuple[ParametricPotentialMLP, ElectrodeTranslationMap]:
    geometry = config["geometry"]
    model = ParametricPotentialMLP(
        width=config["model"]["width"],
        depth=config["model"]["depth"],
        activation=config["model"]["activation"],
        distance_min=geometry["distance_min"],
        distance_max=geometry["distance_max"],
    )
    coordinate_map = ElectrodeTranslationMap(
        reference_distance=geometry["reference_distance"],
        radius=geometry["radius"],
        plateau_margin=geometry["plateau_margin"],
        support_radius=geometry["support_radius"],
    )
    return model, coordinate_map


def loss_terms(
    model: ParametricPotentialMLP,
    coordinate_map: ElectrodeTranslationMap,
    config: dict,
) -> dict[str, torch.Tensor]:
    geometry = config["geometry"]
    training = config["training"]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    interior = sample_interior(
        training["n_interior"],
        geometry,
        device=device,
        dtype=dtype,
    )
    distance_interior = sample_distances(
        len(interior),
        geometry,
        device=device,
        dtype=dtype,
    )
    potential = model(interior, distance_interior)
    residual = pullback_laplacian(
        potential,
        interior,
        coordinate_map,
        distance_interior,
    )
    residual_scale = geometry["reference_distance"] ** 2
    pde_loss = torch.mean((residual_scale * residual) ** 2)

    world = sample_world_boundary(
        training["n_world"],
        geometry,
        device=device,
        dtype=dtype,
    )
    distance_world = sample_distances(
        len(world),
        geometry,
        device=device,
        dtype=dtype,
    )
    world_loss = torch.mean(model(world, distance_world) ** 2)

    electrode, electrode_value = sample_electrode_boundaries(
        training["n_electrode_per_electrode"],
        geometry,
        device=device,
        dtype=dtype,
    )
    distance_electrode = sample_distances(
        len(electrode),
        geometry,
        device=device,
        dtype=dtype,
    )
    electrode_loss = torch.mean(
        (model(electrode, distance_electrode) - electrode_value) ** 2
    )
    bc_loss = world_loss + electrode_loss

    map_points = sample_map_transition(
        training["n_map"],
        geometry,
        device=device,
        dtype=dtype,
    )
    distance_map = sample_distances(
        len(map_points),
        geometry,
        device=device,
        dtype=dtype,
    )
    jacobian = jacobian_constraints(
        coordinate_map,
        map_points,
        distance_map,
        minimum_determinant=training["minimum_determinant"],
        maximum_condition=training["maximum_condition"],
    )
    gap = coordinate_map.minimum_electrode_gap(distance_map)
    gap_loss = torch.mean(torch.relu(geometry["minimum_gap"] - gap) ** 2)
    total = (
        pde_loss
        + training["bc_weight"] * bc_loss
        + training["jacobian_weight"] * (
            jacobian["determinant_loss"] + jacobian["distortion_loss"]
        )
        + training["gap_weight"] * gap_loss
    )
    return {
        "total": total,
        "pde": pde_loss,
        "bc": bc_loss,
        "determinant": jacobian["determinant_loss"],
        "distortion": jacobian["distortion_loss"],
        "gap": gap_loss,
        "minimum_det": torch.min(jacobian["determinant"]),
        "maximum_condition": torch.max(jacobian["condition"]),
    }


def train_model(
    model: ParametricPotentialMLP,
    coordinate_map: ElectrodeTranslationMap,
    config: dict,
    run_dir: Path,
) -> None:
    training = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    history = []
    for step in range(training["steps"]):
        optimizer.zero_grad()
        terms = loss_terms(model, coordinate_map, config)
        terms["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            training["grad_clip"],
        )
        optimizer.step()
        if step % training["print_every"] == 0 or step + 1 == training["steps"]:
            row = {
                "step": step,
                **{name: float(value.detach()) for name, value in terms.items()},
                "gradient_norm": float(gradient_norm),
            }
            history.append(row)
            print(
                f"{step:6d} total={row['total']:.3e} pde={row['pde']:.3e} "
                f"bc={row['bc']:.3e} det_min={row['minimum_det']:.3f}"
            )
    with (run_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def efficiency_and_gradient(
    model: ParametricPotentialMLP,
    coordinate_map: ElectrodeTranslationMap,
    distance_value: float,
    geometry: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    center = torch.zeros(1, 2, device=device, dtype=dtype, requires_grad=True)
    distance = torch.tensor(
        [[distance_value]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    normalized_potential = model(center, distance)
    hessian = physical_hessian(
        normalized_potential,
        center,
        coordinate_map,
        distance,
    )[0]
    eigenvalues = torch.linalg.eigvalsh(hessian)
    strength = 0.5 * (eigenvalues[-1] - eigenvalues[0])
    characteristic_radius = distance.reshape(()) - geometry["radius"]
    efficiency = characteristic_radius**2 * strength
    gradient = torch.autograd.grad(efficiency, distance, create_graph=False)[0].reshape(())
    return efficiency, gradient, hessian


def optimize_distance(
    model: ParametricPotentialMLP,
    coordinate_map: ElectrodeTranslationMap,
    config: dict,
) -> list[dict[str, float]]:
    geometry = config["geometry"]
    settings = config["optimization"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    raw = torch.tensor(
        0.0,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
        requires_grad=True,
    )
    start_fraction = (
        settings["start_distance"] - geometry["distance_min"]
    ) / (geometry["distance_max"] - geometry["distance_min"])
    start_fraction = float(np.clip(start_fraction, 1e-5, 1.0 - 1e-5))
    raw.data.fill_(float(np.log(start_fraction / (1.0 - start_fraction))))
    optimizer = torch.optim.Adam([raw], lr=settings["lr"])
    history = []
    for step in range(settings["steps"]):
        optimizer.zero_grad()
        fraction = torch.sigmoid(raw)
        distance = geometry["distance_min"] + (
            geometry["distance_max"] - geometry["distance_min"]
        ) * fraction
        center = torch.zeros(
            1,
            2,
            device=raw.device,
            dtype=raw.dtype,
            requires_grad=True,
        )
        distance_batch = distance.reshape(1, 1)
        potential = model(center, distance_batch)
        hessian = physical_hessian(
            potential,
            center,
            coordinate_map,
            distance_batch,
        )[0]
        eigenvalues = torch.linalg.eigvalsh(hessian)
        strength = 0.5 * (eigenvalues[-1] - eigenvalues[0])
        efficiency = (distance - geometry["radius"]) ** 2 * strength
        loss = -efficiency
        loss.backward()
        optimizer.step()
        history.append({
            "step": step,
            "distance_cm": float(distance.detach()),
            "quadrupole_efficiency": float(efficiency.detach()),
        })
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return history


def validate(
    model: ParametricPotentialMLP,
    coordinate_map: ElectrodeTranslationMap,
    config: dict,
    run_dir: Path,
) -> dict:
    geometry = config["geometry"]
    validation = config["validation"]
    model.eval()
    distance = config["optimization"]["start_distance"]
    efficiency, gradient, hessian = efficiency_and_gradient(
        model,
        coordinate_map,
        distance,
        geometry,
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    heldout = sample_interior(
        validation["n_interior"],
        geometry,
        device=device,
        dtype=dtype,
    )
    heldout_distance = torch.full(
        (len(heldout), 1),
        distance,
        device=device,
        dtype=dtype,
    )
    residual = pullback_laplacian(
        model(heldout, heldout_distance),
        heldout,
        coordinate_map,
        heldout_distance,
    )
    map_points = sample_map_transition(
        validation["n_interior"],
        geometry,
        device=device,
        dtype=dtype,
    )
    map_distance = torch.cat([
        torch.full(
            (len(map_points) // 2, 1),
            geometry["distance_min"],
            device=device,
            dtype=dtype,
        ),
        torch.full(
            (len(map_points) - len(map_points) // 2, 1),
            geometry["distance_max"],
            device=device,
            dtype=dtype,
        ),
    ])
    jacobian = jacobian_constraints(coordinate_map, map_points, map_distance)
    metrics = {
        "distance_cm": distance,
        "quadrupole_efficiency": float(efficiency.detach()),
        "pinn_autograd_efficiency_derivative_per_cm": float(gradient.detach()),
        "center_normalized_potential_hessian_per_cm2": hessian.detach().cpu().tolist(),
        "heldout_scaled_residual_rmse": float(
            geometry["reference_distance"] ** 2 * torch.sqrt(torch.mean(residual**2)).detach()
        ),
        "minimum_map_determinant": float(torch.min(jacobian["determinant"]).detach()),
        "maximum_map_condition": float(torch.max(jacobian["condition"]).detach()),
        "minimum_electrode_gap_cm": float(
            coordinate_map.minimum_electrode_gap(torch.tensor(distance)).detach()
        ),
    }
    if validation.get("run_devsim", False):
        from PINN.Coordinate_Projection.validate_gradient import compare_gradient_with_devsim

        metrics["devsim_gradient_check"] = compare_gradient_with_devsim(
            model=model,
            coordinate_map=coordinate_map,
            config=config,
            run_dir=run_dir,
            distance=distance,
            epsilon=validation["gradient_epsilon"],
        )
    return metrics


def run(config: dict) -> Path:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    dtype = torch_dtype(config["dtype"])
    device = torch.device(config["device"])
    torch.set_default_dtype(dtype)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["output_root"]) / f"{config['run_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    model, coordinate_map = build_components(config)
    model = model.to(device=device, dtype=dtype)
    coordinate_map = coordinate_map.to(device=device, dtype=dtype)
    train_model(model, coordinate_map, config, run_dir)

    optimization_history = optimize_distance(model, coordinate_map, config)
    with (run_dir / "geometry_optimization.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(optimization_history[0]))
        writer.writeheader()
        writer.writerows(optimization_history)

    metrics = validate(model, coordinate_map, config, run_dir)
    metrics["optimized_distance_cm"] = optimization_history[-1]["distance_cm"]
    metrics["optimized_quadrupole_efficiency"] = optimization_history[-1][
        "quadrupole_efficiency"
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        run_dir / "model.pt",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved run to: {run_dir.resolve()}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
