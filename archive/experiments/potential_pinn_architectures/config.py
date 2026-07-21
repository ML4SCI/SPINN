"""Configuration for the potential PINN architecture comparison."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GeometryConfig:
    world_size_x: float = 2.5
    world_size_y: float = 2.0
    radius: float = 0.5e-1
    north_distance: float = 1.25e-1
    south_distance: float = 1.25e-1
    west_distance: float = 1.25e-1
    east_distance: float = 1.25e-1


@dataclass
class BoundaryConfig:
    world_voltage: float = 0.0
    north_voltage: float = 0.0
    south_voltage: float = 0.0
    west_voltage: float = 300.0
    east_voltage: float = 300.0


@dataclass
class TrainingConfig:
    steps: int = 5000
    lr: float = 1e-3
    optimizer: str = "adam"
    n_interior: int = 2048
    n_world: int = 512
    n_electrode: int = 1024
    bc_weight: float = 100.0
    print_every: int = 200
    eval_every: int = 200
    grad_clip: float = 1.0


@dataclass
class ModelConfig:
    kind: str = "vanilla"
    activation: str = "tanh"
    vanilla_width: int = 128
    vanilla_depth: int = 4
    pixel_grid_resolution: int = 32
    pixel_feature_dim: int = 4
    pixel_hidden_width: int = 16
    pixel_num_grids: int = 4
    pig_num_gaussians: int = 512
    pig_feature_dim: int = 4
    pig_hidden_width: int = 16
    pig_init_sigma: float = 0.1


@dataclass
class EvaluationConfig:
    devsim_rf: str = "DEVSIM/Circular_trap/paul_trap_basis_RF.dat"
    max_points_plot: int = 20000
    chunk_size: int = 8192


@dataclass
class ExperimentConfig:
    seed: int = 0
    output_root: str = "experiments/potential_pinn_architectures/results"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def domain_bounds(self) -> tuple[float, float, float, float]:
        g = self.geometry
        return (
            -g.world_size_x / 2.0,
            g.world_size_x / 2.0,
            -g.world_size_y / 2.0,
            g.world_size_y / 2.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _apply_override(data: dict[str, Any], override: str) -> None:
    key, value = override.split("=", 1)
    target = data
    parts = key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = _parse_value(value)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        seed=data["seed"],
        output_root=data["output_root"],
        geometry=GeometryConfig(**data["geometry"]),
        boundary=BoundaryConfig(**data["boundary"]),
        training=TrainingConfig(**data["training"]),
        model=ModelConfig(**data["model"]),
        evaluation=EvaluationConfig(**data["evaluation"]),
    )


def make_config(kind: str = "vanilla", overrides: list[str] | None = None) -> ExperimentConfig:
    data = ExperimentConfig().to_dict()
    data["model"]["kind"] = kind
    for override in overrides or []:
        _apply_override(data, override)
    return config_from_dict(data)


def write_config(path: str | Path, config: ExperimentConfig) -> None:
    import json

    Path(path).write_text(json.dumps(config.to_dict(), indent=2))

