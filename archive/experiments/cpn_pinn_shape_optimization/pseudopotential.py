"""Model and ideal RF pseudopotential.

``psi_model`` evaluates the physical pseudopotential of the composed model using
the pullback gradient ``grad_x phi = J^{-T} grad_z u`` so the value depends on
both networks. ``psi_ideal`` is the analytic harmonic target ``k (x^2 + y^2)``.

The pseudopotential formula and constants follow the existing project
convention (Ca-40, 10.2 MHz RF, geometry in cm):

    Psi = (e^2 / (4 M Omega^2)) |grad_x phi|^2,  reported in eV.

Source: ``Paul_Trap_PINN.py:207-213,533-537`` and ``PINN/common/metrics.py:119-121``.
"""

from __future__ import annotations

import math

import torch

from PINN.Coordinate_Projection.operators import physical_gradient

from .config import PhysicsConfig


def pseudopotential_scale_eV(physics: PhysicsConfig) -> float:
    """The prefactor ``e / (4 M Omega^2)`` so that ``Psi[eV] = scale |grad phi|^2``."""

    mass = physics.ion_mass_amu * physics.amu
    omega_rf = 2.0 * math.pi * physics.rf_frequency_hz
    return physics.e_charge / (4.0 * mass * omega_rf**2)


def psi_model(
    potential: torch.Tensor,
    reference_points: torch.Tensor,
    coordinate_map,
    physics: PhysicsConfig,
) -> torch.Tensor:
    """Physical pseudopotential of the composed model in eV, shaped ``(N, 1)``."""

    grad_physical = physical_gradient(potential, reference_points, coordinate_map)
    grad_per_metre = grad_physical / physics.geom_unit_to_m
    grad_squared = torch.sum(grad_per_metre**2, dim=1, keepdim=True)
    return pseudopotential_scale_eV(physics) * grad_squared


def psi_ideal(physical_points: torch.Tensor, k: float, physics: PhysicsConfig) -> torch.Tensor:
    """Harmonic target ``k (x^2 + y^2)`` in eV, positions converted to metres."""

    positions_m = physical_points * physics.geom_unit_to_m
    radius_squared = torch.sum(positions_m**2, dim=1, keepdim=True)
    return k * radius_squared


def derive_k_from_secular_freq(physics: PhysicsConfig) -> float:
    """Curvature ``k`` [eV/m^2] giving the target secular frequency.

    A harmonic well ``U = e Psi = e k r^2`` matched to ``U = (1/2) M omega^2 r^2``
    gives ``k = M omega^2 / (2 e)``.
    """

    if physics.target_secular_freq_hz is None:
        raise ValueError("target_secular_freq_hz must be set to derive k")
    mass = physics.ion_mass_amu * physics.amu
    omega = 2.0 * math.pi * physics.target_secular_freq_hz
    return mass * omega**2 / (2.0 * physics.e_charge)
