import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.config import Config
from spinn.geometry.four_rod import build_trap
from spinn.geometry.masks import design_mask


CENTER_FIXED_RADIUS = 0.40
CENTER_RELEASE_RADIUS = 0.85
OUTER_FIXED_WIDTH = 0.50
LOCAL_WIDTH = 0.35
OUTPUT = "scratch/03_mask_eta_gauge_test.png"


trap = build_trap(Config())
centers = torch.tensor([[e.cx, e.cy] for e in trap.electrodes()], dtype=torch.float64)


def smootherstep(t):
    t = torch.clamp(t, 0.0, 1.0)
    return t**3 * (10.0 - 15.0*t + 6.0*t**2)


def old_mask(points):
    center_diff = points[:, None, :] - centers[None, :, :]
    distance2_to_center = torch.sum(center_diff**2, dim=2)
    surface = torch.sqrt(torch.clamp(distance2_to_center, min=1e-12)) - trap.rod_radius_abs
    bumps = torch.exp(-(surface**2) / trap.rod_radius_abs**2)
    return torch.clamp(torch.sum(bumps, dim=1), max=1.0).unsqueeze(1)


def fixed_mask(points):
    return design_mask(points, trap)


def candidate_formula_kept_here_for_plot_history(points):
    r2 = torch.sum(points**2, dim=1)
    center_diff = points[:, None, :] - centers[None, :, :]
    distance2_to_center = torch.sum(center_diff**2, dim=2)
    smooth_surface_coordinate = (
        distance2_to_center - trap.rod_radius_abs**2
    ) / (2.0 * trap.rod_radius_abs)
    bumps = torch.exp(-(smooth_surface_coordinate / LOCAL_WIDTH)**2)
    local_union = 1.0 - torch.prod(1.0 - bumps, dim=1)
    center_anchor = smootherstep(
        (r2 - CENTER_FIXED_RADIUS**2)
        / (CENTER_RELEASE_RADIUS**2 - CENTER_FIXED_RADIUS**2)
    )
    outer_anchor = smootherstep(
        (trap.outer_radius**2 - r2)
        / (trap.outer_radius**2 - (trap.outer_radius - OUTER_FIXED_WIDTH)**2)
    )
    return (local_union * center_anchor * outer_anchor).unsqueeze(1)


def measured_eta(scale, mask_function):
    z = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    x = z + scale * mask_function(z) * z
    potential = 0.5 * (x[:, 0:1] ** 2 - x[:, 1:2] ** 2)
    gradient = torch.autograd.grad(potential.sum(), z, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient[:, 0].sum(), z)[0][:, 0]
    return float(curvature)


scales = np.linspace(-0.35, 0.35, 101)
eta_old = np.array([measured_eta(float(scale), old_mask) for scale in scales])
eta_fixed = np.array([measured_eta(float(scale), fixed_mask) for scale in scales])

axis = torch.linspace(-1.0, 1.0, 301, dtype=torch.float64)
zero = torch.zeros_like(axis)
line = torch.column_stack([axis, zero])
old_line = old_mask(line).detach().numpy().ravel()
fixed_line = fixed_mask(line).detach().numpy().ravel()

probe_axis = np.linspace(-0.36, 0.36, 13)
px, py = np.meshgrid(probe_axis, probe_axis)
probe = np.column_stack([px.ravel(), py.ravel()])
probe = probe[np.hypot(probe[:, 0], probe[:, 1]) <= 0.35]
probe_t = torch.tensor(probe, dtype=torch.float64)
chosen_scale = 0.25
old_disp = (chosen_scale * old_mask(probe_t) * probe_t).detach().numpy()
fixed_disp = (chosen_scale * fixed_mask(probe_t) * probe_t).detach().numpy()

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

axes[0, 0].plot(axis, old_line, label="current mask", linewidth=2)
axes[0, 0].plot(axis, fixed_line, label="exact-center-anchor mask", linewidth=2)
axes[0, 0].axvspan(-0.25, 0.25, color="magenta", alpha=0.12, label="eta measurement disc")
axes[0, 0].set(xlabel=r"$x/r_0$ along $y=0$", ylabel="mask value", title="Mask inside the measurement region")
axes[0, 0].legend()

axes[0, 1].plot(scales, eta_old, label="current mask", linewidth=2)
axes[0, 1].plot(scales, eta_fixed, label="exact-center-anchor mask", linewidth=2)
axes[0, 1].axhline(1.0, color="black", linestyle="--", label="unchanged physical field: true eta = 1")
axes[0, 1].set(
    xlabel=r"coordinate scaling parameter $a$ in $x=z+a\,m(z)z$",
    ylabel="curvature reported to training",
    title="Gauge stress test: field unchanged, coordinates changed",
)
axes[0, 1].legend()

for ax, displacement, title in [
    (axes[1, 0], old_disp, "Current mask: center coordinates move"),
    (axes[1, 1], fixed_disp, "Fixed mask: measurement disc is exactly pinned"),
]:
    ax.quiver(
        probe[:, 0], probe[:, 1], displacement[:, 0], displacement[:, 1],
        angles="xy", scale_units="xy", scale=1, width=0.006,
    )
    ax.add_patch(plt.Circle((0, 0), 0.25, fill=False, color="magenta", linewidth=2))
    ax.set(xlim=(-0.42, 0.42), ylim=(-0.42, 0.42), aspect="equal", xlabel=r"$x/r_0$", ylabel=r"$y/r_0$", title=title)

fig.suptitle("Does the deformation mask let coordinate scaling fake an eta improvement?", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("physical field held fixed: V(x,y)=0.5*(x^2-y^2), true eta=1")
print("old mask at origin:", float(old_mask(torch.zeros(1, 2, dtype=torch.float64))))
print("fixed mask at origin:", float(fixed_mask(torch.zeros(1, 2, dtype=torch.float64))))
print("at coordinate expansion a=+0.25")
print("old-mask training eta:", measured_eta(0.25, old_mask))
print("fixed-mask training eta:", measured_eta(0.25, fixed_mask))
print("saved", OUTPUT)
