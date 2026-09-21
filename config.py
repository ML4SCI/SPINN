"""Configuration dataclasses and YAML loading for the SPINN Paul-trap experiment.

The structure follows ``SPINN_Experiment_Plan.pdf`` Appendix A: a single nested
config tree (``experiment``, ``shape_network``, ``physics_backend``, ``sampling``,
``loss_weights``, ``optimizer``, ``validation``) is loaded from a YAML file in
``spinn/configs/`` and overlaid with command-line overrides.

Units are normalized per the plan: ``r0 = 1``, ``V0 = 1``, rod radius
``1.145 r0``, outer radius ``5 r0``. Positive RF electrodes lie on the x-axis,
negative RF electrodes on the y-axis, outer boundary grounded.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class ExperimentSection:
    name: str = "joint_pig_four_rod"
    seed: int = 0
    geometry: str = "circular_four_rod"  # or "hyperbolic", "perturbed_four_rod"
    r0: float = 1.0
    rod_radius: float = 1.145
    outer_radius: float = 5.0
    V0: float = 1.0


@dataclass
class ShapeNetworkSection:
    type: str = "residual_mlp"
    hidden_layers: int = 5
    width: int = 64
    activation: str = "tanh"
    alpha: float = 0.10  # residual scale on the masked deformation
    identity_pretrain_steps: int = 8000
    jacobian_eps: float = 0.05
    symmetric: bool = True  # enforce x/y mirror symmetry of the deformation
    # Optional replacement for the legacy Boolean above. None preserves the
    # old behavior; otherwise choose none/x_mirror/y_mirror/xy_mirror/D4.
    symmetry: str | None = None


@dataclass
class PhysicsBackendSection:
    type: str = "pig"  # "mlp" | "pixel" | "pig"
    activation: str = "tanh"
    # MLP
    mlp_hidden_layers: int = 6
    mlp_width: int = 128
    # PIXEL
    pixel_grid_resolution: int = 16
    pixel_feature_dim: int = 4
    pixel_num_grids: int = 16
    pixel_decoder_layers: int = 2
    pixel_decoder_width: int = 16
    # PIG
    K: int = 800
    feature_dim: int = 4
    decoder_layers: int = 2
    decoder_width: int = 16
    init_sigma: float = 0.10
    # PIG scalar-output projection. "x_mirror" means x -> -x (a mirror
    # about the y axis); "y_mirror" means y -> -y. The default preserves
    # the previous unrestricted backend exactly.
    symmetry: str = "none"  # none | x_mirror | y_mirror | xy_mirror | d4_quadrupole


@dataclass
class SamplingSection:
    interior_points: int = 30000
    electrode_boundary_points_per_electrode: int = 1000
    outer_boundary_points: int = 3000
    center_probe_radius: float = 0.25
    center_probe_grid: int = 31
    quadrupole_loss_radius: float = 0.25
    quadrupole_loss_grid: int = 17
    resample_every: int = 500


@dataclass
class LossWeightsSection:
    pde: float = 1.0
    bc: float = 100.0
    eta: float = 0.05
    quadrupole: float = 0.0
    jacobian: float = 10.0
    displacement: float = 10.0
    gap: float = 100.0
    curvature: float = 1.0
    anchor: float = 10.0


@dataclass
class OptimizerSection:
    type: str = "adam"
    steps: int = 60000
    lr_theta: float = 1.0e-3
    lr_phi: float = 2.0e-4
    lr_pig_centers: float = 5.0e-5
    lr_pig_covariances: float = 5.0e-5
    scheduler: str = "cosine_decay"  # "cosine_decay" | "none"
    # fixed-geometry physics-pretrain learning rates (plan training table)
    lr_fixed_mlp: float = 1.0e-3
    lr_fixed_pixel: float = 2.0e-3
    lr_fixed_pig_decoder: float = 1.0e-3
    lr_fixed_pig_centers: float = 1.0e-4


@dataclass
class ValidationSection:
    fem_checkpoint_every: int = 5000
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    primary_metric: str = "eta_fem"
    valid_min_detJ: float = 0.2
    fem_grid: int = 256  # finite-difference grid resolution per axis


@dataclass
class Config:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    shape_network: ShapeNetworkSection = field(default_factory=ShapeNetworkSection)
    physics_backend: PhysicsBackendSection = field(default_factory=PhysicsBackendSection)
    sampling: SamplingSection = field(default_factory=SamplingSection)
    loss_weights: LossWeightsSection = field(default_factory=LossWeightsSection)
    optimizer: OptimizerSection = field(default_factory=OptimizerSection)
    validation: ValidationSection = field(default_factory=ValidationSection)

    # --- derived geometry -------------------------------------------------
    @property
    def rod_radius_abs(self) -> float:
        return self.experiment.rod_radius * self.experiment.r0

    @property
    def rod_center_distance(self) -> float:
        # plan: centers at (r0 + R_rod) from the origin
        return self.experiment.r0 + self.rod_radius_abs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTION_TYPES = {
    "experiment": ExperimentSection,
    "shape_network": ShapeNetworkSection,
    "physics_backend": PhysicsBackendSection,
    "sampling": SamplingSection,
    "loss_weights": LossWeightsSection,
    "optimizer": OptimizerSection,
    "validation": ValidationSection,
}


def _build_section(name: str, data: dict[str, Any]):
    cls = _SECTION_TYPES[name]
    valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown key(s) in section {name!r}: {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: dict[str, Any]) -> Config:
    sections = {}
    for name in _SECTION_TYPES:
        sections[name] = _build_section(name, dict(data.get(name, {})))
    return Config(**sections)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        candidate = CONFIG_DIR / path.name
        if candidate.exists():
            path = candidate
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    return config_from_dict(data)


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


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Apply ``section.key=value`` dotted overrides, returning a new Config."""

    data = config.to_dict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, raw = override.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"override key must be section.field, got {key!r}")
        section, field_name = parts
        if section not in data or field_name not in data[section]:
            raise ValueError(f"unknown override target {key!r}")
        data[section][field_name] = _parse_value(raw)
    return config_from_dict(data)


def merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result
