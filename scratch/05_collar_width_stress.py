from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from spinn.config import Config
from spinn.geometry.four_rod import build_trap
from spinn.geometry.masks import design_mask


WIDTHS = [0.20, 0.35, 0.50]
DISPLACEMENTS = np.linspace(-0.15, 0.15, 13)
GRID = 141
OUT = Path("scratch/results/05_collar_width_stress")


trap = build_trap(Config())
centers = torch.tensor([[e.cx, e.cy] for e in trap.electrodes()], dtype=torch.float64)
axis_directions = centers / torch.linalg.vector_norm(centers, dim=1, keepdim=True)


def raw_radial_translation(points, width):
    diff = points[:, None, :] - centers[None, :, :]
    distance2 = torch.sum(diff**2, dim=2)
    surface = (distance2 - trap.rod_radius_abs**2) / (2.0 * trap.rod_radius_abs)
    weights = torch.exp(-(surface / width) ** 2)
    return (weights[:, :, None] * axis_directions[None, :, :]).sum(dim=1) / (
        weights.sum(dim=1, keepdim=True) + 1e-12
    )


def coordinate_map(points, width, displacement):
    proposal = raw_radial_translation(points, width)
    mask = design_mask(points, trap, band_width=width)
    return points + displacement * mask * proposal


def jacobian_fields(points, width, displacement):
    z = points.detach().clone().requires_grad_(True)
    mapped = coordinate_map(z, width, displacement)
    row0 = torch.autograd.grad(mapped[:, 0].sum(), z, create_graph=True)[0]
    row1 = torch.autograd.grad(mapped[:, 1].sum(), z)[0]
    jac = torch.stack([row0, row1], dim=1)
    det = torch.linalg.det(jac)
    singular = torch.linalg.svdvals(jac)
    condition = singular[:, 0] / singular[:, 1]
    return mapped.detach(), det.detach(), singular.detach(), condition.detach()


def electrode_contours(width, displacement, n=500):
    theta = torch.linspace(0.0, 2.0 * torch.pi, n, dtype=torch.float64)
    contours = []
    achieved = []
    for index, electrode in enumerate(trap.electrodes()):
        ring = torch.stack([
            electrode.cx + electrode.radius * torch.cos(theta),
            electrode.cy + electrode.radius * torch.sin(theta),
        ], dim=1)
        mapped = coordinate_map(ring, width, displacement).detach()
        desired = displacement * axis_directions[index]
        error = torch.linalg.vector_norm((mapped - ring) - desired, dim=1)
        contours.append(mapped.numpy())
        achieved.append((float(torch.mean(error)), float(torch.max(error))))
    return contours, achieved


axis = torch.linspace(-trap.outer_radius, trap.outer_radius, GRID, dtype=torch.float64)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")
grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
vacuum = torch.tensor(trap.in_vacuum(grid_points.numpy()), dtype=torch.bool)
world = torch.linalg.vector_norm(grid_points, dim=1) <= trap.outer_radius

OUT.mkdir(parents=True, exist_ok=True)
rows = []

for width in WIDTHS:
    for displacement in DISPLACEMENTS:
        mapped, det, singular, condition = jacobian_fields(grid_points, width, float(displacement))
        active = vacuum
        contours, achieved = electrode_contours(width, float(displacement))
        center_move = torch.linalg.vector_norm(
            coordinate_map(torch.zeros(1, 2, dtype=torch.float64), width, float(displacement)), dim=1
        )[0]
        outer_probe = torch.tensor([[trap.outer_radius, 0.0]], dtype=torch.float64)
        outer_move = torch.linalg.vector_norm(
            coordinate_map(outer_probe, width, float(displacement)) - outer_probe, dim=1
        )[0]
        rows.append({
            "width": width,
            "displacement": float(displacement),
            "min_det": float(det[active].min()),
            "max_det": float(det[active].max()),
            "min_singular": float(singular[active, 1].min()),
            "max_condition": float(condition[active].max()),
            "center_move": float(center_move),
            "outer_move": float(outer_move),
            "surface_mean_error": float(np.mean([value[0] for value in achieved])),
            "surface_max_error": float(np.max([value[1] for value in achieved])),
        })

fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
for column, width in enumerate(WIDTHS):
    displacement = -0.15
    mapped, det, singular, condition = jacobian_fields(grid_points, width, displacement)
    contours, _ = electrode_contours(width, displacement)
    fields = [det, singular[:, 1], condition]
    titles = [r"$\det J$", r"$\sigma_{\min}(J)$", r"condition number $\kappa(J)$"]
    cmaps = ["coolwarm", "viridis", "magma"]
    for row, (field, title, cmap) in enumerate(zip(fields, titles, cmaps)):
        plot_field = torch.where(world, field, torch.nan).reshape(GRID, GRID).numpy()
        if row == 0:
            image = axes[row, column].imshow(
                plot_field, origin="lower", extent=[-5, 5, -5, 5], cmap=cmap,
                vmin=0.5, vmax=1.5,
            )
        elif row == 1:
            image = axes[row, column].imshow(
                plot_field, origin="lower", extent=[-5, 5, -5, 5], cmap=cmap,
                vmin=0.5, vmax=1.0,
            )
        else:
            image = axes[row, column].imshow(
                plot_field, origin="lower", extent=[-5, 5, -5, 5], cmap=cmap,
                vmin=1.0, vmax=2.0,
            )
        for electrode, contour in zip(trap.electrodes(), contours):
            axes[row, column].add_patch(Circle(
                (electrode.cx, electrode.cy), electrode.radius,
                fill=False, color="white", linestyle="--", linewidth=0.8,
            ))
            axes[row, column].plot(contour[:, 0], contour[:, 1], color="cyan", linewidth=1.3)
        axes[row, column].add_patch(Circle((0, 0), 0.40, fill=False, color="magenta", linewidth=1.0))
        axes[row, column].set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5))
        axes[row, column].set_title(f"{title}, width={width:.2f}, inward motion=0.15")
        fig.colorbar(image, ax=axes[row, column], shrink=0.75)

fig.suptitle("Manufactured electrode translation: geometry and Jacobian behavior", fontsize=16)
fig.savefig(OUT / "inward_015_jacobian_maps.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for width in WIDTHS:
    selected = [row for row in rows if row["width"] == width]
    d = np.array([row["displacement"] for row in selected])
    axes[0].plot(d, [row["min_det"] for row in selected], marker="o", label=f"w={width:.2f}")
    axes[1].plot(d, [row["min_singular"] for row in selected], marker="o", label=f"w={width:.2f}")
    axes[2].plot(d, [row["max_condition"] for row in selected], marker="o", label=f"w={width:.2f}")

axes[0].axhline(0.2, color="red", linestyle="--", label="current validity threshold")
axes[0].set(xlabel=r"radial electrode translation / $r_0$", ylabel=r"minimum $\det J$", title="Orientation and local area")
axes[1].set(xlabel=r"radial electrode translation / $r_0$", ylabel=r"minimum $\sigma_{min}$", title="Worst local compression")
axes[2].set(xlabel=r"radial electrode translation / $r_0$", ylabel=r"maximum $\kappa(J)$", title="Worst anisotropic distortion")
for ax in axes:
    ax.axvline(-0.15, color="gray", linewidth=0.8)
    ax.axvline(0.15, color="gray", linewidth=0.8)
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle("Collar-width sweep: negative motion moves all rods toward the trap center")
fig.savefig(OUT / "collar_width_summary.png", dpi=180)
plt.close(fig)

print("manufactured radial translation; cyan is deformed boundary, dashed white is original")
for width in WIDTHS:
    for displacement in [-0.15, -0.10, 0.10, 0.15]:
        row = min(
            (candidate for candidate in rows if candidate["width"] == width),
            key=lambda candidate: abs(candidate["displacement"] - displacement),
        )
        print(
            f"width={width:.2f} displacement={row['displacement']:+.2f} "
            f"min_det={row['min_det']:.3f} min_sigma={row['min_singular']:.3f} "
            f"max_condition={row['max_condition']:.2f} "
            f"surface_max_error={row['surface_max_error']:.3e}"
        )
print("center and outer motion maxima:", max(row["center_move"] for row in rows), max(row["outer_move"] for row in rows))
print("saved", OUT)
