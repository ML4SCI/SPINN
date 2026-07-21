"""Shared experiment configuration.

A single ``ExperimentConfig`` defines every hyperparameter that must stay
identical across the three architectures for the comparison to be fair
(seed, optimizer, learning rate, steps, batch size, domain, normalization).
Per-model YAML files only override the ``model`` block; everything else is
inherited so the training and evaluation pipeline is shared.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GeometryConfig:
    """Reference domain and electrode layout, mirrored from CPN.py."""

    world_size_x: float = 2.5
    world_size_y: float = 2.0
    radius: float = 0.5e-1
    north_distance: float = 1.25e-1
    south_distance: float = 1.25e-1
    west_distance: float = 1.25e-1
    east_distance: float = 1.25e-1


@dataclass
class TrainingConfig:
    steps: int = 5000
    lr: float = 1e-3
    batch_interior: int = 2048
    batch_world: int = 512
    batch_electrode: int = 1024
    print_every: int = 200
    eval_every: int = 200
    val_batch: int = 4096


@dataclass
class EvaluationConfig:
    nx: int = 250
    ny: int = 200
    quiver_stride: int = 12


@dataclass
class ModelConfig:
    """Architecture selector plus per-architecture hyperparameters.

    ``kind`` picks the registered model. The remaining fields are read only
    by the architecture that needs them, so a single flat block keeps the
    YAML files short and diffable.
    """

    kind: str = "vanilla"
    mlp_width: int = 50
    mlp_depth: int = 6
    zero_last_layer: bool = True

    pixel_grid_resolution: int = 32
    pixel_feature_dim: int = 4
    pixel_hidden_width: int = 16

    pig_num_gaussians: int = 256
    pig_feature_dim: int = 4
    pig_hidden_width: int = 16
    pig_init_sigma: float = 0.1


@dataclass
class ExperimentConfig:
    seed: int = 0
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def domain_bounds(self) -> tuple[float, float, float, float]:
        g = self.geometry
        return (
            -g.world_size_x / 2,
            g.world_size_x / 2,
            -g.world_size_y / 2,
            g.world_size_y / 2,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from a YAML file layered on the defaults."""

    with open(path) as handle:
        override = yaml.safe_load(handle) or {}

    defaults = ExperimentConfig().to_dict()
    merged = _merge(defaults, override)

    return ExperimentConfig(
        seed=merged["seed"],
        geometry=GeometryConfig(**merged["geometry"]),
        training=TrainingConfig(**merged["training"]),
        evaluation=EvaluationConfig(**merged["evaluation"]),
        model=ModelConfig(**merged["model"]),
    )
