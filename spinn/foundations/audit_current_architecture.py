"""Executable mathematical audit of the coordinate-shape objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ..config import Config
from ..geometry.four_rod import build_trap
from ..geometry.masks import design_mask
from ..losses.objective import differentiable_eta
from ..models.composed import ComposedField
from ..physics.eta import eta_autograd


class ExactQuadrupole(nn.Module):
    """Exact physical field with a parameter for model API compatibility."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return 0.5 * (points[:, 0:1] ** 2 - points[:, 1:2] ** 2) + 0.0 * self.dummy


class CenterOnlyCoordinateWarp(nn.Module):
    """Rescale coordinates at the origin while remaining fixed outside r=0.5."""

    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(float(epsilon), dtype=torch.float64))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        radius_squared = torch.sum(points**2, dim=1, keepdim=True)
        bump = torch.clamp(1.0 - radius_squared / 0.5**2, min=0.0) ** 3
        return points * (1.0 + self.epsilon * bump)


def run_audit() -> dict:
    """Run deterministic mask and coordinate-gauge checks."""
    trap = build_trap(Config())
    probes = torch.tensor(
        [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [5.0, 0.0]],
        dtype=torch.float64,
    )
    masks = design_mask(probes, trap).detach().numpy().reshape(-1)

    backend = ExactQuadrupole().double()
    physical_eta = eta_autograd(backend, 1.0, 1.0, dtype=torch.float64)
    gauge_rows = []
    for epsilon in (0.0, 0.05, 0.10, 0.20):
        shape = CenterOnlyCoordinateWarp(epsilon).double()
        model = ComposedField(shape, backend)
        composed_eta = float(differentiable_eta(model, 1.0, 1.0).detach())
        boundary_probe = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        boundary_motion = float(
            torch.max(torch.abs(shape(boundary_probe) - boundary_probe)).detach()
        )
        gauge_rows.append(
            {
                "epsilon": epsilon,
                "physical_eta": physical_eta,
                "composed_reference_eta": composed_eta,
                "analytic_composed_eta": (1.0 + epsilon) ** 2,
                "boundary_motion_at_r1": boundary_motion,
            }
        )

    return {
        "design_mask": {
            "origin": float(masks[0]),
            "r_0p25": float(masks[1]),
            "r_1": float(masks[2]),
            "outer_x": float(masks[3]),
        },
        "coordinate_gauge_test": gauge_rows,
        "interpretation": (
            "The electrode boundary and exact physical field are unchanged, but "
            "reference-coordinate eta changes with the coordinate warp."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Write the audit result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("spinn/results/foundations/current_architecture_audit.json"),
    )
    args = parser.parse_args(argv)
    result = run_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(result["interpretation"])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
