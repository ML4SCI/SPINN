import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from scratch.pig_gate_common import exact_field


OUTPUT = "exp/harmonic_source_shape_search/results/02_differentiable_radius_gradient.png"
COUNTS = np.array([8, 12, 16, 24, 32])
FRACTIONS = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
RADIUS = 1.0
MEASUREMENT_RADIUS = 0.5


def circle(center_x, center_y, radius, count, phase, device=None):
    theta = 2.0 * torch.pi * (torch.arange(count, dtype=torch.float64, device=device) / count) + phase
    return torch.stack([center_x + radius * torch.cos(theta), center_y + radius * torch.sin(theta)], dim=1)


def differentiable_q(rod_radius, count, fraction, return_condition=False):
    zero = rod_radius * 0.0
    distance = 1.0 + rod_radius
    centers = [(distance, zero), (-distance, zero), (zero, distance), (zero, -distance)]
    signs = [0.5, 0.5, -0.5, -0.5]
    sources, boundaries, values = [], [], []
    for (cx, cy), sign in zip(centers, signs):
        sources.append(circle(cx, cy, fraction * rod_radius, count, 0.013))
        boundaries.append(circle(cx, cy, rod_radius, 96, 0.0))
        values.append(torch.full((96,), sign, dtype=torch.float64))
    sources.append(circle(zero, zero, 6.5 + zero, 4 * count, 0.007))
    boundaries.append(circle(zero, zero, 5.0 + zero, 384, 0.0))
    values.append(torch.zeros(384, dtype=torch.float64))
    sources = torch.cat(sources)
    boundary = torch.cat(boundaries)
    target = torch.cat(values)
    matrix = torch.cat([
        torch.log(torch.linalg.vector_norm(boundary[:, None, :] - sources[None, :, :], dim=2)),
        torch.ones((len(boundary), 1), dtype=torch.float64),
    ], dim=1)
    coefficients = torch.linalg.lstsq(matrix, target, rcond=1e-12).solution

    theta = 2.0 * torch.pi * torch.arange(720, dtype=torch.float64) / 720
    probe = MEASUREMENT_RADIUS * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    potential = torch.cat([
        torch.log(torch.linalg.vector_norm(probe[:, None, :] - sources[None, :, :], dim=2)),
        torch.ones((len(probe), 1), dtype=torch.float64),
    ], dim=1) @ coefficients
    a2 = 2.0 * torch.mean(potential * torch.cos(2.0 * theta)) / MEASUREMENT_RADIUS**2
    a6 = 2.0 * torch.mean(potential * torch.cos(6.0 * theta)) / MEASUREMENT_RADIUS**6
    q6 = (a6 / a2) * MEASUREMENT_RADIUS**4
    if return_condition:
        condition = torch.linalg.cond(matrix).detach().item()
        return q6, condition
    return q6


def reference_q(radius):
    trap = FourRodTrap(rod_radius=float(radius))
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    points = MEASUREMENT_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])
    potential = exact_field(trap, points)[0]
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / MEASUREMENT_RADIUS**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / MEASUREMENT_RADIUS**6
    return (a6 / a2) * MEASUREMENT_RADIUS**4


print("HYPOTHESIS: differentiating through the exact linear amplitude solve gives the physical dq6/dR only in a source-placement band where the boundary matrix is accurate enough but not numerically singular.")
step = 1.0e-4
reference_value = reference_q(RADIUS)
reference_gradient = (reference_q(RADIUS + step) - reference_q(RADIUS - step)) / (2.0 * step)
print(f"reference q6={reference_value:+.9e}, finite-difference dq6/dR={reference_gradient:+.9e}")

shape = (len(COUNTS), len(FRACTIONS))
condition = np.zeros(shape)
q_error = np.zeros(shape)
gradient_error = np.zeros(shape)
autograd_fd_error = np.zeros(shape)
autograd_gradient = np.zeros(shape)

for row, count in enumerate(COUNTS):
    for column, fraction in enumerate(FRACTIONS):
        radius = torch.tensor(RADIUS, dtype=torch.float64, requires_grad=True)
        q6, condition[row, column] = differentiable_q(radius, int(count), float(fraction), True)
        gradient = torch.autograd.grad(q6, radius)[0].item()
        plus = differentiable_q(torch.tensor(RADIUS + step, dtype=torch.float64), int(count), float(fraction)).item()
        minus = differentiable_q(torch.tensor(RADIUS - step, dtype=torch.float64), int(count), float(fraction)).item()
        solver_fd = (plus - minus) / (2.0 * step)
        autograd_gradient[row, column] = gradient
        q_error[row, column] = abs(q6.item() - reference_value)
        gradient_error[row, column] = abs(gradient - reference_gradient)
        autograd_fd_error[row, column] = abs(gradient - solver_fd)

best = np.unravel_index(np.argmin(gradient_error + autograd_fd_error), shape)
print(f"best gradient point count={COUNTS[best[0]]}, fraction={FRACTIONS[best[1]]:.2f}, cond={condition[best]:.3e}, q error={q_error[best]:.3e}, autograd dq/dR={autograd_gradient[best]:+.8e}, reference-gradient error={gradient_error[best]:.3e}, same-solver AD/FD error={autograd_fd_error[best]:.3e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [FRACTIONS[0], FRACTIONS[-1], COUNTS[0], COUNTS[-1]]
panels = [
    (np.log10(condition), "viridis", "log10 matrix condition number"),
    (np.log10(q_error), "magma_r", "log10 physical q6 error"),
    (np.log10(gradient_error), "magma_r", "log10 error versus physical dq6/dR"),
    (np.log10(autograd_fd_error), "magma_r", "log10 AD versus same-solver finite difference"),
]
for ax, (values, cmap, title) in zip(axes.ravel()[:4], panels):
    image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.scatter([FRACTIONS[best[1]]], [COUNTS[best[0]]], marker="x", color="cyan", s=75)
    ax.set(xlabel="source radius / electrode radius", ylabel="sources per electrode", title=title)
    fig.colorbar(image, ax=ax, shrink=0.82)

for row, count in enumerate(COUNTS):
    axes[1, 1].plot(FRACTIONS, autograd_gradient[row], marker="o", label=f"{count} / electrode")
axes[1, 1].axhline(reference_gradient, color="black", linestyle="--", label="independent MFS finite difference")
axes[1, 1].set(xlabel="source radius / electrode radius", ylabel="dq6/dR", title="A field-accurate solve can still have a bad shape gradient")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.25)

valid = (q_error < 1e-6) & (gradient_error < 1e-5) & (autograd_fd_error < 1e-7)
axes[1, 2].imshow(valid.astype(float), origin="lower", aspect="auto", extent=extent, cmap="Blues", vmin=0, vmax=1)
axes[1, 2].set(xlabel="source radius / electrode radius", ylabel="sources per electrode", title="White-blue gate: field and gradient both trustworthy")
fig.suptitle("Plot 02 — differentiable nested electrostatics must pass a shape-gradient gate", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("jointly valid source settings:", int(np.sum(valid)), "of", valid.size)
print("CONCLUSION: select source placement using gradient agreement, not the smallest field residual; only jointly valid settings should enter electrode optimization.")
print("saved", OUTPUT)
