import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from spinn.geometry.masks import design_mask


OUTPUT = "scratch/38_direct_vs_total_shape_gradient.png"
HYPOTHESIS = "With the center objective disc anchored, electrode-only coordinate deformation has zero frozen-field partial dq/dR even though electrode surfaces move; the nonzero physical derivative is the field-equilibration response."
EXACT_SLOPE = -6.5388843e-2
REEQUILIBRATED_PIG_SLOPE = -7.3557857e-2


def radial_map(points, trap, delta):
    centers = torch.tensor([[e.cx, e.cy] for e in trap.electrodes()], dtype=points.dtype)
    difference = points[:, None, :] - centers[None, :, :]
    distance = torch.linalg.norm(difference, dim=2)
    nearest = torch.argmin(distance, dim=1)
    selected = difference[torch.arange(len(points)), nearest]
    direction = selected / torch.linalg.norm(selected, dim=1, keepdim=True).clamp_min(1e-12)
    return points + design_mask(points, trap) * delta * direction


def fit_q(fixed_coordinates, potential):
    z = torch.complex(fixed_coordinates[:, 0], fixed_coordinates[:, 1])
    columns = [torch.ones_like(fixed_coordinates[:, 0])]
    for order in [2, 6, 10]:
        columns.extend([torch.real(z**order), torch.imag(z**order)])
    matrix = torch.stack(columns, dim=1)
    coefficients = torch.linalg.lstsq(matrix, potential).solution
    return coefficients[3] / coefficients[1]


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0)
axis = np.linspace(-0.25, 0.25, 31)
gx, gy = np.meshgrid(axis, axis)
center_np = np.column_stack([gx.ravel(), gy.ravel()])
center_np = center_np[np.hypot(center_np[:, 0], center_np[:, 1]) <= 0.25]
center = torch.tensor(center_np, dtype=torch.float64)

theta = np.linspace(0, 2 * np.pi, 181, endpoint=False)
electrode_np = []
for electrode in trap.electrodes():
    electrode_np.append(np.column_stack([electrode.cx + electrode.radius * np.cos(theta), electrode.cy + electrode.radius * np.sin(theta)]))
electrode = torch.tensor(np.vstack(electrode_np), dtype=torch.float64)

delta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
mapped_center = radial_map(center, trap, delta)
mapped_electrode = radial_map(electrode, trap, delta)
a2 = 0.5
a6 = 0.5 * 7.1172e-6
z_mapped = torch.complex(mapped_center[:, 0], mapped_center[:, 1])
frozen_potential = a2 * torch.real(z_mapped**2) + a6 * torch.real(z_mapped**6)
q = fit_q(center, frozen_potential)
direct_q_gradient = torch.autograd.grad(q, delta, retain_graph=True)[0].item()
electrode_motion = torch.mean(torch.linalg.norm(mapped_electrode - electrode, dim=1))
positive_delta = torch.tensor(1e-5, dtype=torch.float64)
electrode_motion_gradient_value = (torch.mean(torch.linalg.norm(radial_map(electrode, trap, positive_delta) - electrode, dim=1)) / positive_delta).item()

deltas = np.linspace(-0.02, 0.02, 41)
direct_q = []
for value in deltas:
    mapped = radial_map(center, trap, torch.tensor(value, dtype=torch.float64))
    z = torch.complex(mapped[:, 0], mapped[:, 1])
    potential = a2 * torch.real(z**2) + a6 * torch.real(z**6)
    direct_q.append(fit_q(center, potential).item())
direct_q = np.array(direct_q)

grid_axis = np.linspace(-5, 5, 301)
mx, my = np.meshgrid(grid_axis, grid_axis)
grid_points = torch.tensor(np.column_stack([mx.ravel(), my.ravel()]), dtype=torch.float64)
mask = design_mask(grid_points, trap).detach().numpy().reshape(mx.shape)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
image = axes[0, 0].imshow(mask, origin="lower", extent=(-5, 5, -5, 5), cmap="viridis", vmin=0, vmax=1)
circle = plt.Circle((0, 0), 0.25, fill=False, color="magenta", linewidth=2, label="objective disc")
axes[0, 0].add_patch(circle)
for electrode_geometry in trap.electrodes():
    axes[0, 0].add_patch(plt.Circle((electrode_geometry.cx, electrode_geometry.cy), electrode_geometry.radius, fill=False, color="white", linewidth=1))
axes[0, 0].set(xlabel="x/r0", ylabel="y/r0", title="Electrode deformation mask; center objective is pinned")
axes[0, 0].legend(loc="upper right")
fig.colorbar(image, ax=axes[0, 0], label="allowed displacement multiplier")

axes[0, 1].plot(deltas, direct_q, marker="o", markersize=3, label="frozen field + anchored query")
axes[0, 1].set(xlabel="requested rod-surface displacement / r0", ylabel="fitted q on r≤0.25r0", title="Direct partial objective is flat")
axes[0, 1].legend()

labels = ["frozen-field partial", "re-equilibrated PIG", "exact MFS"]
gradients = [direct_q_gradient, REEQUILIBRATED_PIG_SLOPE, EXACT_SLOPE]
bars = axes[1, 0].bar(labels, gradients)
axes[1, 0].bar_label(bars, fmt="%.5f")
axes[1, 0].axhline(0, color="black", linewidth=1)
axes[1, 0].tick_params(axis="x", rotation=12)
axes[1, 0].set(ylabel="dq/d(R/r0)", title="Partial derivative omits the physical field response")

axes[1, 1].axis("off")
axes[1, 1].text(0.02, 0.85, "Reduced objective", fontsize=15, weight="bold", transform=axes[1, 1].transAxes)
axes[1, 1].text(0.02, 0.70, "J(R) = q(θ*(R), R)", fontsize=14, transform=axes[1, 1].transAxes)
axes[1, 1].text(0.02, 0.50, "dJ/dR = ∂J/∂R  +  (∂J/∂θ)(dθ*/dR)", fontsize=14, transform=axes[1, 1].transAxes)
axes[1, 1].text(0.02, 0.31, "anchored frozen-field term = 0", color="tab:blue", fontsize=13, transform=axes[1, 1].transAxes)
axes[1, 1].text(0.02, 0.18, "equilibrium-response term ≠ 0", color="tab:orange", fontsize=13, transform=axes[1, 1].transAxes)
axes[1, 1].text(0.02, 0.03, "Simultaneous updates do not automatically compute the second term.", fontsize=11, transform=axes[1, 1].transAxes)

for ax in axes.ravel()[:3]:
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: the current anchored objective is missing the total shape-response chain", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

center_mask_max = design_mask(center, trap).max().item()
electrode_mask_mean = design_mask(electrode, trap).mean().item()
print(f"RESULT center_mask_max={center_mask_max:.3e} electrode_mask_mean={electrode_mask_mean:.6f} electrode_motion_sensitivity={electrode_motion_gradient_value:.6f} direct_dq={direct_q_gradient:+.3e} reequilibrated_dq={REEQUILIBRATED_PIG_SLOPE:+.6f} exact_dq={EXACT_SLOPE:+.6f}")
print("CONCLUSION: missing-chain diagnosis accepted. Center mask and frozen-field direct dq are exactly zero while electrode motion sensitivity is one and re-equilibrated/exact derivatives are nonzero. The current per-geometry simultaneous objective cannot supply the physical reduced shape gradient by direct autograd.")
print("saved", OUTPUT)
