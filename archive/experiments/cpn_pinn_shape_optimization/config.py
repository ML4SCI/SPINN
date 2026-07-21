"""Configuration dataclasses and the five training-variation presets.

Geometry and physics defaults are mirrored from existing code so this
experiment operates on the same circular trap and the same pseudopotential
convention as the rest of the project (cited in the fields below).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GeometryConfig:
    """Circular trap, mirrored from CPN.py / TrapGeometry in the instructor module."""

    world_size_x: float = 2.5
    world_size_y: float = 2.0
    radius: float = 0.5e-1
    north_distance: float = 1.25e-1
    south_distance: float = 1.25e-1
    west_distance: float = 1.25e-1
    east_distance: float = 1.25e-1


@dataclass
class PhysicsConfig:
    """Pseudopotential constants, from Paul_Trap_PINN.py:24-28,207-213."""

    e_charge: float = 1.602e-19
    amu: float = 1.66054e-27
    ion_mass_amu: float = 40.0
    rf_frequency_hz: float = 10.2e6
    geom_unit_to_m: float = 1e-2
    rf_voltage: float = 300.0
    target_secular_freq_hz: float | None = 2.0e6


@dataclass
class ModelConfig:
    cpn_width: int = 50
    cpn_depth: int = 6
    cpn_zero_last_layer: bool = True
    pinn_width: int = 128
    pinn_depth: int = 4
    pinn_activation: str = "tanh"


@dataclass
class LossConfig:
    lambda_pde: float = 1.0
    lambda_pseudo: float = 1.0
    lambda_bc: float = 10.0
    lambda_deform: float = 0.0
    lambda_smooth: float = 0.0
    lambda_jac: float = 0.0
    fit_radius: float = 0.3


@dataclass
class TrainingConfig:
    steps: int = 4000
    lr: float = 1e-3
    n_interior: int = 2048
    n_world: int = 512
    n_electrode: int = 512
    grad_clip: float = 1.0
    print_every: int = 100
    eval_every: int = 500
    pinn_pretrain_steps: int = 3000
    cpn_pretrain_steps: int = 2000


@dataclass
class VariationConfig:
    """Which networks are pretrained, frozen, or jointly optimized.

    ``alternate`` is a placeholder schedule flag used only by the scaffolded
    variation E; it is not tuned in this pass.
    """

    name: str
    pinn_pretrained: bool
    pinn_frozen: bool
    cpn_pretrained_identity: bool
    cpn_frozen: bool
    use_jacobian_reg: bool = False
    use_deformation_reg: bool = False
    alternate: bool = False


@dataclass
class ExperimentConfig:
    variation: str = "C"
    seed: int = 0
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    variation_flags: VariationConfig | None = None

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


_VARIATION_FLAGS: dict[str, VariationConfig] = {
    "A": VariationConfig(
        name="A_pinn_first",
        pinn_pretrained=True,
        pinn_frozen=True,
        cpn_pretrained_identity=True,
        cpn_frozen=False,
    ),
    "B": VariationConfig(
        name="B_joint_jacobian_reg",
        pinn_pretrained=False,
        pinn_frozen=False,
        cpn_pretrained_identity=True,
        cpn_frozen=False,
        use_jacobian_reg=True,
    ),
    "C": VariationConfig(
        name="C_joint",
        pinn_pretrained=False,
        pinn_frozen=False,
        cpn_pretrained_identity=True,
        cpn_frozen=False,
    ),
    "D": VariationConfig(
        name="D_pinn_first_deformation_reg",
        pinn_pretrained=True,
        pinn_frozen=True,
        cpn_pretrained_identity=True,
        cpn_frozen=False,
        use_deformation_reg=True,
    ),
    "E": VariationConfig(
        name="E_both_pretrained_alternating",
        pinn_pretrained=True,
        pinn_frozen=False,
        cpn_pretrained_identity=True,
        cpn_frozen=False,
        alternate=True,
    ),
}


def make_preset(variation: str) -> ExperimentConfig:
    key = variation.upper()
    if key not in _VARIATION_FLAGS:
        raise ValueError(f"unknown variation {variation!r}; choose from {sorted(_VARIATION_FLAGS)}")
    flags = _VARIATION_FLAGS[key]
    config = ExperimentConfig(variation=key, variation_flags=flags)
    if flags.use_jacobian_reg:
        config.loss.lambda_jac = 1.0
    if flags.use_deformation_reg:
        config.loss.lambda_deform = 1.0
        config.loss.lambda_smooth = 1.0
    return config
