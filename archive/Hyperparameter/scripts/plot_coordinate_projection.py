#!/usr/bin/env python3
"""Diagnostic plot for the residual coordinate-projection architecture.

This visualizes the Equation-13 residual shape network

    NN_phi(z) = z + NN_hat(z)

from Zhang et al. as implemented by ``ResidualCoordinateMap``. It is an
*architecture* diagnostic, not an optimization result:

* With no checkpoint, the residual output layer is zero-initialized, so the
  map is exactly the identity. Expected diagnostics: displacement near 0,
  ``det J`` near 1, condition number near 1. This is the honest, valid result
  for the architecture check.
* With ``--checkpoint``, a *trained* residual map is loaded and its actual
  learned deformation is shown.
* With ``--demo-deformation``, a synthetic deformation is injected purely to
  illustrate what the diagnostics look like under a non-trivial map. The plot
  is then clearly labeled SYNTHETIC/DEMO and claims no optimized geometry.
"""

from __future__ import annotations

import argparse
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
from PINN.Coordinate_Projection.maps import ResidualCoordinateMap


DEFAULT_SUMMARY = SPINN_ROOT / "Hyperparameter" / "results" / "summary"
DEFAULT_OUTPUT = DEFAULT_SUMMARY / "residual_coordinate_projection_diagnostic.png"
DOMAIN = (-0.35, 0.35)
ELECTRODE_DISTANCE = 0.125
ELECTRODE_RADIUS = 0.05


def build_grid(resolution: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.linspace(DOMAIN[0], DOMAIN[1], resolution, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)
    return points, grid_x, grid_y


def apply_demo_deformation(coordinate_map: ResidualCoordinateMap) -> None:
    """Inject a smooth, illustrative-only deformation into the residual net.

    This does NOT represent a trained or optimized geometry. It simply biases
    the final layer so the diagnostics show non-trivial structure for display.
    """

    output = coordinate_map.network[-1]
    with torch.no_grad():
        output.weight.normal_(0.0, 0.02)
        output.bias.normal_(0.0, 0.01)
    coordinate_map.max_displacement = 0.05


def compute_diagnostics(
    coordinate_map: ResidualCoordinateMap,
    points: torch.Tensor,
) -> dict[str, np.ndarray]:
    reference = points.clone().requires_grad_(True)
    diagnostics = jacobian_constraints(coordinate_map, reference)
    projected = coordinate_map(reference)
    displacement = projected - reference
    displacement_magnitude = torch.linalg.vector_norm(displacement, dim=1)
    return {
        "reference": reference.detach().numpy(),
        "projected": projected.detach().numpy(),
        "displacement_magnitude": displacement_magnitude.detach().numpy(),
        "determinant": diagnostics["determinant"].detach().numpy(),
        "condition": diagnostics["condition"].detach().numpy(),
    }


def electrode_circles(ax) -> None:
    centers = [
        (ELECTRODE_DISTANCE, 0.0),
        (-ELECTRODE_DISTANCE, 0.0),
        (0.0, ELECTRODE_DISTANCE),
        (0.0, -ELECTRODE_DISTANCE),
    ]
    for cx, cy in centers:
        ax.add_patch(
            plt.Circle((cx, cy), ELECTRODE_RADIUS, fill=False,
                       edgecolor="black", linewidth=0.8, linestyle="--")
        )


def plot_diagnostic(
    diag: dict[str, np.ndarray],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    mode_label: str,
    output_path: Path,
    stride: int = 8,
) -> None:
    shape = grid_x.shape
    ref = diag["reference"]
    proj = diag["projected"]
    disp = diag["displacement_magnitude"].reshape(shape)
    det = diag["determinant"].reshape(shape)
    cond = diag["condition"].reshape(shape)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))

    # Panel 1: reference grid
    ax = axes[0, 0]
    rx = ref[:, 0].reshape(shape)
    ry = ref[:, 1].reshape(shape)
    for i in range(0, shape[0], stride):
        ax.plot(rx[i, :], ry[i, :], color="0.6", linewidth=0.5)
    for j in range(0, shape[1], stride):
        ax.plot(rx[:, j], ry[:, j], color="0.6", linewidth=0.5)
    electrode_circles(ax)
    ax.set_title(r"Reference grid $z$")
    ax.set_aspect("equal")

    # Panel 2: projected grid
    ax = axes[0, 1]
    px = proj[:, 0].reshape(shape)
    py = proj[:, 1].reshape(shape)
    for i in range(0, shape[0], stride):
        ax.plot(px[i, :], py[i, :], color="C0", linewidth=0.5)
    for j in range(0, shape[1], stride):
        ax.plot(px[:, j], py[:, j], color="C0", linewidth=0.5)
    electrode_circles(ax)
    ax.set_title(r"Projected grid $x = NN_\phi(z)$")
    ax.set_aspect("equal")

    # Panel 3: displacement magnitude
    ax = axes[0, 2]
    im = ax.pcolormesh(grid_x.numpy(), grid_y.numpy(), disp, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(r"Displacement $\|NN_\phi(z) - z\|$")
    ax.set_aspect("equal")

    # Panel 4: Jacobian determinant
    ax = axes[1, 0]
    im = ax.pcolormesh(grid_x.numpy(), grid_y.numpy(), det, shading="auto", cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(r"Jacobian determinant $\det J$")
    ax.set_aspect("equal")

    # Panel 5: Jacobian condition number
    ax = axes[1, 1]
    im = ax.pcolormesh(grid_x.numpy(), grid_y.numpy(), cond, shading="auto", cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(r"Jacobian condition number $\kappa(J)$")
    ax.set_aspect("equal")

    # Panel 6: numeric summary
    ax = axes[1, 2]
    ax.axis("off")
    summary = "\n".join([
        f"max displacement  : {disp.max():.3e}",
        f"det J  min / max  : {det.min():.4f} / {det.max():.4f}",
        f"cond  min / max   : {cond.min():.4f} / {cond.max():.4f}",
        "",
        "det J > 0 everywhere: " + ("yes" if det.min() > 0 else "NO"),
    ])
    ax.text(0.02, 0.95, summary, va="top", ha="left", family="monospace", fontsize=11)

    fig.suptitle(
        f"Residual coordinate projection $NN_\\phi(z)=z+\\widehat{{NN}}(z)$  --  {mode_label}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=121)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="State dict for a trained ResidualCoordinateMap. If omitted, the "
             "identity-initialized architecture is plotted.",
    )
    parser.add_argument(
        "--demo-deformation",
        action="store_true",
        help="Inject a SYNTHETIC, illustrative-only deformation (no optimized "
             "geometry is claimed).",
    )
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.float64)

    coordinate_map = ResidualCoordinateMap(width=args.width, depth=args.depth).double()

    if args.checkpoint is not None and args.demo_deformation:
        raise SystemExit("--checkpoint and --demo-deformation are mutually exclusive")

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        coordinate_map.load_state_dict(state)
        mode_label = f"trained checkpoint: {args.checkpoint.name}"
    elif args.demo_deformation:
        apply_demo_deformation(coordinate_map)
        mode_label = "SYNTHETIC / DEMO deformation (not an optimized geometry)"
    else:
        mode_label = "identity initialization (architecture check; no training)"

    coordinate_map.eval()
    points, grid_x, grid_y = build_grid(args.resolution)
    diag = compute_diagnostics(coordinate_map, points)
    plot_diagnostic(diag, grid_x, grid_y, mode_label, args.output)

    print(f"Mode: {mode_label}")
    print(
        "Diagnostics: "
        f"max|disp|={diag['displacement_magnitude'].max():.3e}, "
        f"det J in [{diag['determinant'].min():.4f}, {diag['determinant'].max():.4f}], "
        f"cond in [{diag['condition'].min():.4f}, {diag['condition'].max():.4f}]"
    )
    print(f"Saved diagnostic plot to {args.output}")


if __name__ == "__main__":
    main()
