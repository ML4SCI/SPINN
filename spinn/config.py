"""Typed configuration and YAML inheritance for SPINN experiments.

The structure follows ``SPINN_Experiment_Plan.pdf`` Appendix A: a single nested
config tree is loaded from a YAML file in ``spinn/configs/`` (optionally
``extends:``-ing a parent file) and overlaid with ``section.key=value`` overrides.

Units are normalized per the plan: ``r0 = 1``, ``V0 = 1``, rod radius
``1.145 r0``, outer radius ``5 r0``. Positive RF electrodes lie on the x-axis,
negative RF electrodes on the y-axis, outer boundary grounded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BACKENDS = ("mlp", "pixel", "pig")


@dataclass
class ExperimentSection:
    name: str = "joint_pig_four_rod"
    seed: int = 0
    geometry: str = "circular_four_rod"  # or "perturbed_four_rod"
    r0: float = 1.0
    rod_radius: float = 1.145
    outer_radius: float = 5.0
    V0: float = 1.0


@dataclass
class ShapeNetworkSection:
    hidden_layers: int = 5
    width: int = 64
    activation: str = "tanh"  # tanh | sine
    alpha: float = 0.10  # residual scale on the masked deformation
    identity_pretrain_steps: int = 8000
    jacobian_eps: float = 0.05
    symmetry: str = "xy_mirror"  # none | x_mirror | y_mirror | xy_mirror | d4


@dataclass
class PhysicsBackendSection:
    type: str = "pig"  # mlp | pixel | pig
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
    symmetry: str = "none"  # none | x_mirror | y_mirror | xy_mirror | d4


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
    """Adam learning rates; joint training uses ``lr_theta``/``lr_phi``/``lr_pig_*``."""

    steps: int = 60000
    lr_theta: float = 1.0e-3
    lr_phi: float = 2.0e-4
    lr_pig_centers: float = 5.0e-5
    lr_pig_covariances: float = 5.0e-5
    scheduler: str = "cosine_decay"  # cosine_decay | exponential | none
    # fixed-geometry physics-pretrain learning rates (plan training table)
    lr_fixed_mlp: float = 1.0e-3
    lr_fixed_pixel: float = 2.0e-3
    lr_fixed_pig_decoder: float = 1.0e-3
    lr_fixed_pig_centers: float = 1.0e-4


@dataclass
class ValidationSection:
    fem_checkpoint_every: int = 5000
    valid_min_detJ: float = 0.2
    fem_grid: int = 256  # finite-difference grid resolution per axis


@dataclass
class RuntimeSection:
    device: str = "cpu"  # cpu | cuda | mps | auto
    dtype: str = "float32"
    deterministic: bool = True
    log_every: int = 200
    gradient_clip_norm: float = 1.0
    fail_on_nonfinite: bool = True


@dataclass
class Config:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    shape_network: ShapeNetworkSection = field(default_factory=ShapeNetworkSection)
    physics_backend: PhysicsBackendSection = field(
        default_factory=PhysicsBackendSection
    )
    sampling: SamplingSection = field(default_factory=SamplingSection)
    loss_weights: LossWeightsSection = field(default_factory=LossWeightsSection)
    optimizer: OptimizerSection = field(default_factory=OptimizerSection)
    validation: ValidationSection = field(default_factory=ValidationSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """Reject invalid settings before an expensive training run starts."""
        s = self.sampling
        checks = [
            (
                self.physics_backend.type in BACKENDS,
                f"physics_backend.type must be one of {BACKENDS}",
            ),
            (self.optimizer.steps > 0, "optimizer.steps must be positive"),
            (s.resample_every > 0, "sampling.resample_every must be positive"),
            (
                min(
                    s.interior_points,
                    s.electrode_boundary_points_per_electrode,
                    s.outer_boundary_points,
                )
                > 0,
                "all sampling counts must be positive",
            ),
            (
                self.validation.fem_checkpoint_every > 0,
                "validation.fem_checkpoint_every must be positive",
            ),
            (self.validation.fem_grid >= 16, "validation.fem_grid must be at least 16"),
            (
                self.runtime.dtype in {"float32", "float64"},
                "runtime.dtype must be float32 or float64",
            ),
            (self.runtime.log_every > 0, "runtime.log_every must be positive"),
            (
                self.runtime.gradient_clip_norm > 0,
                "runtime.gradient_clip_norm must be positive",
            ),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


_SECTION_TYPES = {f.name: f.default_factory for f in fields(Config)}

# Keys written by older versions into saved result.json / manifest.json configs.
_RETIRED_KEYS = {
    "shape_network": {"type", "symmetric"},
    "physics_backend": {"activation"},
    "sampling": {
        "interior_batch_size",
        "electrode_boundary_batch_size",
        "outer_boundary_batch_size",
    },
    "optimizer": {"type"},
    "validation": {"seeds", "primary_metric"},
}


def _upgrade_section(name: str, data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    if name == "shape_network" and data.get("symmetry") is None:
        data["symmetry"] = "xy_mirror" if data.get("symmetric", True) else "none"
    for key in _RETIRED_KEYS.get(name, ()):
        data.pop(key, None)
    return data


def config_from_dict(data: dict[str, Any]) -> Config:
    unknown = set(data) - set(_SECTION_TYPES)
    if unknown:
        raise ValueError(f"unknown configuration section(s): {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTION_TYPES.items():
        values = _upgrade_section(name, data.get(name, {}))
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown key(s) in section {name!r}: {sorted(unknown)}")
        sections[name] = cls(**values)
    config = Config(**sections)
    config.validate()
    return config


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration mappings without mutating either input."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _load_mapping(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"cyclic configuration inheritance: {chain}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge_dict(_load_mapping(path.parent / parent, seen | {path}), data)


def load_config(path: str | Path) -> Config:
    """Load a YAML config; bare file names also resolve inside ``spinn/configs``."""
    path = Path(path)
    if (
        not path.is_absolute()
        and not path.exists()
        and (CONFIG_DIR / path.name).exists()
    ):
        path = CONFIG_DIR / path.name
    return config_from_dict(_load_mapping(path))


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Apply ``section.key=value`` dotted overrides, returning a new Config."""
    data = config.to_dict()
    for override in overrides:
        key, sep, raw = override.partition("=")
        section, _, field_name = key.partition(".")
        if not sep or not field_name or "." in field_name:
            raise ValueError(f"override must be section.field=value, got {override!r}")
        if field_name not in data.get(section, {}):
            raise ValueError(f"unknown override target {key!r}")
        data[section][field_name] = _parse_value(raw)
    return config_from_dict(data)
