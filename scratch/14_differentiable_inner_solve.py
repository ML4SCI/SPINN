import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float64)
CHECK_RATIOS = [0.95, 1.00, 1.10, 1.20, 1.30]
STARTS = [0.90, 1.38]
STEPS = 16
LEARNING_RATE = 35.0
FD_STEP = 0.005
OUTER_RADIUS = 5.0
ROI = 0.60
OUTPUT = "scratch/14_differentiable_inner_solve.png"


def circle_points(center_x, center_y, radius, n, phase=0.0):
    angle = torch.arange(n) * (2.0 * torch.pi / n) + phase
    return torch.column_stack([center_x + radius * torch.cos(angle), center_y + radius * torch.sin(angle)])


def kernel(points, sources):
    return torch.log(torch.linalg.vector_norm(points[:, None, :] - sources[None, :, :], dim=2))


axis = torch.linspace(-ROI, ROI, 61)
gx, gy = torch.meshgrid(axis, axis, indexing="xy")
probe = torch.column_stack([gx.reshape(-1), gy.reshape(-1)])
probe = probe[torch.linalg.vector_norm(probe, dim=1) <= ROI]
z = torch.complex(probe[:, 0], probe[:, 1])
orders = [2, 6, 10, 14, 18]
fit_matrix = torch.column_stack(
    [torch.ones(len(probe))]
    + [part for order in orders for part in (torch.real(z**order), torch.imag(z**order))]
)


def harmonic_loss(radius):
    distance = 1.0 + radius
    centers = [(distance, 0.0), (-distance, 0.0), (0.0, distance), (0.0, -distance)]
    voltages = [1.0, 1.0, -1.0, -1.0]
    sources, boundaries, values = [], [], []
    for (cx, cy), voltage in zip(centers, voltages):
        sources.append(circle_points(cx, cy, 0.62 * radius, 32, 0.017))
        boundaries.append(circle_points(cx, cy, radius, 72))
        values.append(torch.full((72,), voltage))
    sources.append(circle_points(0.0, 0.0, 1.30 * OUTER_RADIUS, 128, 0.011))
    boundaries.append(circle_points(0.0, 0.0, OUTER_RADIUS, 256))
    values.append(torch.zeros(256))
    sources = torch.vstack(sources)
    boundary = torch.vstack(boundaries)
    matrix = torch.column_stack([kernel(boundary, sources), torch.ones(len(boundary))])
    right_hand_side = torch.cat(values)
    normal = matrix.T @ matrix
    ridge = 1.0e-10 * torch.trace(normal).detach() / len(normal)
    coefficients = torch.linalg.solve(
        normal + ridge * torch.eye(len(normal)), matrix.T @ right_hand_side
    )
    potential = kernel(probe, sources) @ coefficients[:-1] + coefficients[-1]
    fit_normal = fit_matrix.T @ fit_matrix
    fit = torch.linalg.solve(fit_normal, fit_matrix.T @ potential)
    q = fit[3] / fit[1]
    held_points, held_values = [], []
    for (cx, cy), voltage in zip(centers, voltages):
        held_points.append(circle_points(cx, cy, radius, 181, 0.0043))
        held_values.append(torch.full((181,), voltage))
    held_points.append(circle_points(0.0, 0.0, OUTER_RADIUS, 361, 0.0021))
    held_values.append(torch.zeros(361))
    held_points = torch.vstack(held_points)
    held_error = torch.max(torch.abs(kernel(held_points, sources) @ coefficients[:-1] + coefficients[-1] - torch.cat(held_values)))
    return q * q, q, held_error


autograd_gradient = []
finite_difference_gradient = []
boundary_errors = []
for value in CHECK_RATIOS:
    radius = torch.tensor(value, requires_grad=True)
    loss, q, boundary_error = harmonic_loss(radius)
    gradient = torch.autograd.grad(loss, radius)[0]
    plus = harmonic_loss(torch.tensor(value + FD_STEP))[0]
    minus = harmonic_loss(torch.tensor(value - FD_STEP))[0]
    fd = (plus - minus) / (2.0 * FD_STEP)
    autograd_gradient.append(gradient.detach().item())
    finite_difference_gradient.append(fd.detach().item())
    boundary_errors.append(boundary_error.detach().item())
    print(f"R/r0={value:.2f} q={q.detach().item():+.5e} autograd={gradient.detach().item():+.6e} independent_FD={fd.detach().item():+.6e} held_boundary={boundary_error.detach().item():.3e}")


histories = []
for start in STARTS:
    radius = torch.tensor(start, requires_grad=True)
    history = []
    for step in range(STEPS):
        loss, q, _ = harmonic_loss(radius)
        gradient = torch.autograd.grad(loss, radius)[0]
        history.append((radius.detach().item(), loss.detach().item(), q.detach().item(), gradient.detach().item()))
        radius = (radius - LEARNING_RATE * gradient).detach().requires_grad_()
    loss, q, _ = harmonic_loss(radius)
    history.append((radius.detach().item(), loss.detach().item(), q.detach().item(), np.nan))
    histories.append(np.array(history))
    print(f"start={start:.2f} final R/r0={radius.detach().item():.6f} q={q.detach().item():+.5e} J={loss.detach().item():.5e}")


fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
low = min(autograd_gradient + finite_difference_gradient)
high = max(autograd_gradient + finite_difference_gradient)
axes[0, 0].scatter(finite_difference_gradient, autograd_gradient, s=60)
axes[0, 0].plot([low, high], [low, high], color="black", linestyle="--", label="exact agreement")
for ratio, x, y in zip(CHECK_RATIOS, finite_difference_gradient, autograd_gradient):
    axes[0, 0].annotate(f"R={ratio:.2f}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
axes[0, 0].set(xlabel="independent central finite-difference gradient", ylabel="gradient through inner solve", title="Shape-gradient sign and magnitude gate")
axes[0, 0].legend()

relative_error = np.abs(np.array(autograd_gradient) - np.array(finite_difference_gradient)) / np.maximum(np.abs(finite_difference_gradient), 1e-12)
axes[0, 1].semilogy(CHECK_RATIOS, relative_error, marker="o")
axes[0, 1].set(xlabel=r"electrode radius $R/r_0$", ylabel="relative gradient disagreement", title="Numerical agreement")
boundary_axis = axes[0, 1].twinx()
boundary_axis.semilogy(CHECK_RATIOS, boundary_errors, marker="s", color="tab:orange")
boundary_axis.set_ylabel("held-out boundary voltage error", color="tab:orange")

for start, history in zip(STARTS, histories):
    axes[1, 0].plot(history[:, 0], marker="o", label=f"start {start:.2f}")
    axes[1, 1].semilogy(history[:, 1], marker="o", label=f"start {start:.2f}")
axes[1, 0].axhline(1.145, color="red", linestyle="--", label="known 1.145")
axes[1, 0].set(xlabel="optimization step", ylabel=r"$R/r_0$", title="Differentiable solve recovers electrode radius")
axes[1, 1].set(xlabel="optimization step", ylabel=r"$J=(a_6/a_2)^2$", title="Physical objective decreases")
axes[1, 0].legend()
axes[1, 1].legend()

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Bilevel prototype: differentiate through the physics solve", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("maximum relative gradient error:", relative_error.max())
print("maximum held-out boundary error:", max(boundary_errors))
print("saved", OUTPUT)
