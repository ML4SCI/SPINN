import numpy as np
import torch
import matplotlib.pyplot as plt

from spinn.config import Config
from spinn.geometry.four_rod import build_trap
from spinn.geometry.masks import design_mask


GRID = 241
LOCAL_WIDTH = 0.35
CENTER_FIXED_RADIUS = 0.40
CENTER_RELEASE_RADIUS = 0.85
OUTER_FIXED_WIDTH = 0.50
OUTPUT = "scratch/01_deformation_mask.png"


trap = build_trap(Config())
axis = torch.linspace(-trap.outer_radius, trap.outer_radius, GRID)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")
points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
r = torch.sqrt(torch.sum(points**2, dim=1))
centers = torch.tensor([[e.cx, e.cy] for e in trap.electrodes()], dtype=points.dtype)

def old_design_mask(points):
    diff = points.unsqueeze(1) - centers.unsqueeze(0)
    dist2 = torch.sum(diff**2, dim=-1)
    surface = torch.sqrt(torch.clamp(dist2, min=1e-12)) - trap.rod_radius_abs
    bumps = torch.exp(-(surface**2) / trap.rod_radius_abs**2)
    return torch.clamp(torch.sum(bumps, dim=1), max=1.0).unsqueeze(1)


old = old_design_mask(points).reshape(GRID, GRID).detach()

distance2_to_center = torch.sum((points[:, None, :] - centers[None, :, :])**2, dim=2)
smooth_surface_coordinate = (distance2_to_center - trap.rod_radius_abs**2) / (2.0 * trap.rod_radius_abs)
bumps = torch.exp(-(smooth_surface_coordinate / LOCAL_WIDTH)**2)
local_union = 1.0 - torch.prod(1.0 - bumps, dim=1)


def smootherstep(t):
    t = torch.clamp(t, 0.0, 1.0)
    return t**3 * (10.0 - 15.0*t + 6.0*t**2)


center_anchor = smootherstep(
    (r**2 - CENTER_FIXED_RADIUS**2) / (CENTER_RELEASE_RADIUS**2 - CENTER_FIXED_RADIUS**2)
)
outer_anchor = smootherstep(
    (trap.outer_radius**2 - r**2)
    / (trap.outer_radius**2 - (trap.outer_radius - OUTER_FIXED_WIDTH)**2)
)
new = design_mask(points, trap).reshape(GRID, GRID).detach()

inside_world = (xx**2 + yy**2) <= trap.outer_radius**2
old_plot = torch.where(inside_world, old, torch.nan)
new_plot = torch.where(inside_world, new, torch.nan)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
levels = np.linspace(0, 1, 21)

for ax, field, title in [
    (axes[0, 0], old_plot, "current mask"),
    (axes[0, 1], new_plot, "candidate: local surface band + exact anchors"),
]:
    image = ax.contourf(xx.numpy(), yy.numpy(), field.numpy(), levels=levels, cmap="viridis")
    for electrode in trap.electrodes():
        ax.add_patch(plt.Circle((electrode.cx, electrode.cy), electrode.radius, fill=False, color="white"))
    ax.add_patch(plt.Circle((0, 0), CENTER_FIXED_RADIUS, fill=False, color="magenta", linewidth=2))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x / r0")
    ax.set_ylabel("y / r0")

center_row = GRID // 2
axes[1, 0].plot(axis.numpy(), old[center_row].numpy(), label="current")
axes[1, 0].plot(axis.numpy(), new[center_row].numpy(), label="candidate")
axes[1, 0].axvspan(-CENTER_FIXED_RADIUS, CENTER_FIXED_RADIUS, color="magenta", alpha=0.15)
axes[1, 0].set(xlabel="x / r0 along y=0", ylabel="mask value", ylim=(-0.03, 1.03), title="centerline slice")
axes[1, 0].legend()

difference = torch.where(inside_world, new-old, torch.nan)
diff_image = axes[1, 1].contourf(xx.numpy(), yy.numpy(), difference.numpy(), levels=np.linspace(-1, 1, 21), cmap="coolwarm")
axes[1, 1].set_aspect("equal")
axes[1, 1].set_title("candidate minus current")
axes[1, 1].set_xlabel("x / r0")
axes[1, 1].set_ylabel("y / r0")

fig.colorbar(image, ax=axes[0, 1], label="deformation freedom", pad=0.03)
fig.colorbar(diff_image, ax=axes[1, 1], label="change")
fig.suptitle("Stress test 01: where can the coordinate network move points?")
fig.savefig(OUTPUT, dpi=180)

sample_points = torch.tensor([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [5.0, 0.0]])
sample_r = torch.sqrt(torch.sum(sample_points**2, dim=1))
sample_d2 = torch.sum((sample_points[:, None, :] - centers[None, :, :])**2, dim=2)
sample_surface = (sample_d2 - trap.rod_radius_abs**2) / (2.0 * trap.rod_radius_abs)
sample_bumps = torch.exp(-(sample_surface/LOCAL_WIDTH)**2)
sample_local = 1.0 - torch.prod(1.0-sample_bumps, dim=1)
sample_new = sample_local * smootherstep(
    (sample_r**2-CENTER_FIXED_RADIUS**2)/(CENTER_RELEASE_RADIUS**2-CENTER_FIXED_RADIUS**2)
) * smootherstep(
    (trap.outer_radius**2-sample_r**2)/(trap.outer_radius**2-(trap.outer_radius-OUTER_FIXED_WIDTH)**2)
)

print("picked: compact electrode-surface bands multiplied by exact center and outer anchors")
print("points: origin, center probe, nearest electrode surface, outer boundary")
print("old:", old_design_mask(sample_points).reshape(-1).detach().numpy())
print("production fixed:", design_mask(sample_points, trap).reshape(-1).detach().numpy())
print("saved", OUTPUT)
