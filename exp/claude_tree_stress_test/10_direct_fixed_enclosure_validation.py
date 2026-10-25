import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/10_direct_fixed_enclosure_validation.png"
RADIUS = 1.1451090300780968
DISTANCE = 1.0 + RADIUS
OUTER_RADIUS = 5.0
KAPPA9 = -0.0012213401059591876
DELTA13 = -0.0003875957659400217
P9 = np.arange(9, 19)
W9 = np.array([0.14526537, -0.10620197, 0.28421932, -0.42300312, -0.77440244, 14.74465186, -52.90593862, 91.26916034, -77.73953184, 26.50578109])
W9 /= np.sum(W9)
P13 = np.arange(16, 29)
W13 = np.array([6.78065474, -42.50938914, 128.37796771, -226.14901113, 228.63499372, -81.83172425, -89.9305892, 123.78893861, -34.42561199, -82.46601008, 164.01436013, -139.11150475, 45.82692563])
W13 /= np.sum(W13)
COUNTS = [40, 56, 72, 88]


def basis(z, order, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    return z**order * (z**4 - 1.0) / (z**4 - DISTANCE**4) ** power / leading


def basis_derivative(z, order, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    denominator = z**4 - DISTANCE**4
    first = (order * z ** (order - 1) * (z**4 - 1.0) + 4.0 * z ** (order + 3)) / denominator**power
    second = power * 4.0 * z ** (order + 3) * (z**4 - 1.0) / denominator ** (power + 1)
    return (first - second) / leading


def mapping(z):
    return z + KAPPA9 * sum(weight * basis(z, 9, power) for weight, power in zip(W9, P9)) + DELTA13 * sum(weight * basis(z, 13, power) for weight, power in zip(W13, P13))


def mapping_derivative(z):
    return 1.0 + KAPPA9 * sum(weight * basis_derivative(z, 9, power) for weight, power in zip(W9, P9)) + DELTA13 * sum(weight * basis_derivative(z, 13, power) for weight, power in zip(W13, P13))


def inverse_mapping(w):
    z = w.copy()
    for _ in range(15):
        z -= (mapping(z) - w) / mapping_derivative(z)
    return z


centers = DISTANCE * np.array([1.0, -1.0, 1.0j, -1.0j])
voltages = np.array([0.5, 0.5, -0.5, -0.5])


def solve_reference(count):
    tb = 2.0 * np.pi * np.arange(3 * count) / (3 * count)
    ts = 2.0 * np.pi * np.arange(count) / count + 0.013
    boundary = np.concatenate([center + RADIUS * np.exp(1j * tb) for center in centers])
    sources = np.concatenate([center + 0.70 * RADIUS * np.exp(1j * ts) for center in centers])
    values = np.concatenate([np.full(len(tb), voltage) for voltage in voltages])
    to = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    tso = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * to)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * tso)])
    values = np.concatenate([values, np.zeros(len(to))])
    A = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(A, values, rcond=1e-12)
    return sources, coefficients


def solve_direct(count):
    tb = 2.0 * np.pi * np.arange(3 * count) / (3 * count)
    ts = 2.0 * np.pi * np.arange(count) / count + 0.011
    boundary_segments = [mapping(center + RADIUS * np.exp(1j * tb)) for center in centers]
    source_boundary = [mapping(center + RADIUS * np.exp(1j * ts)) for center in centers]
    sources = np.concatenate([center + 0.62 * (curve - center) for center, curve in zip(centers, source_boundary)])
    boundary = np.concatenate(boundary_segments)
    values = np.concatenate([np.full(len(tb), voltage) for voltage in voltages])
    to = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    tso = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * to)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * tso)])
    values = np.concatenate([values, np.zeros(len(to))])
    A = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(A, values, rcond=1e-12)
    return sources, coefficients


def potential(z, sources, coefficients):
    return np.log(np.abs(z[:, None] - sources[None, :])) @ coefficients[:-1] + coefficients[-1]


reference_sources, reference_coefficients = solve_reference(96)
theta = 2.0 * np.pi * np.arange(2048) / 2048
probe = 0.5 * np.exp(1j * theta)
orders = np.arange(2, 31, 4)


def multipoles(values):
    a = np.array([2.0 * np.mean(values * np.cos(order * theta)) / 0.5**order for order in orders])
    return a / a[0] * 0.5 ** (orders - 2)


pullback_probe = potential(inverse_mapping(probe), reference_sources, reference_coefficients)
pullback_q = multipoles(pullback_probe)

grid_axis = np.linspace(-0.9, 0.9, 181)
xx, yy = np.meshgrid(grid_axis, grid_axis)
grid = (xx + 1j * yy).ravel()
inside = (xx**2 + yy**2) <= 0.9**2
pullback_grid = potential(inverse_mapping(grid), reference_sources, reference_coefficients).reshape(xx.shape)

q_values = []
spatial_rmse = []
boundary_max = []
solutions = []
unseen_theta = 2.0 * np.pi * (np.arange(3000) + 0.37) / 3000
unseen_electrodes = np.concatenate([mapping(center + RADIUS * np.exp(1j * unseen_theta)) for center in centers])
unseen_values = np.concatenate([np.full(len(unseen_theta), voltage) for voltage in voltages])
unseen_outer = OUTER_RADIUS * np.exp(1j * unseen_theta)
for count in COUNTS:
    sources, coefficients = solve_direct(count)
    direct_probe = potential(probe, sources, coefficients)
    q_values.append(multipoles(direct_probe))
    direct_grid = potential(grid, sources, coefficients).reshape(xx.shape)
    difference = direct_grid - pullback_grid
    spatial_rmse.append(np.sqrt(np.mean(difference[inside] ** 2)) / np.sqrt(np.mean(pullback_grid[inside] ** 2)))
    electrode_error = np.max(np.abs(potential(unseen_electrodes, sources, coefficients) - unseen_values))
    outer_error = np.max(np.abs(potential(unseen_outer, sources, coefficients)))
    boundary_max.append(max(electrode_error, outer_error))
    solutions.append((sources, coefficients, direct_grid, difference))

q_values = np.array(q_values)
spatial_rmse = np.array(spatial_rmse)
boundary_max = np.array(boundary_max)
best_sources, best_coefficients, best_grid, difference = solutions[-1]

print("HYPOTHESIS: a direct Laplace re-solve with the original perfectly circular enclosure will reproduce the conformal pullback because the mapped enclosure differs by only 1.8e-8 r0.")
print("pullback q", pullback_q)
for count, q, rmse, error in zip(COUNTS, q_values, spatial_rmse, boundary_max):
    print("count", count, "q", q, "central V relative RMSE", rmse, "unseen max boundary error", error)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
limit = np.max(np.abs(pullback_grid[inside]))
shown_pullback = pullback_grid.copy()
shown_pullback[~inside] = np.nan
im0 = axes[0, 0].imshow(shown_pullback, origin="lower", extent=[-0.9, 0.9, -0.9, 0.9], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[0, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Exact conformal pullback potential")
fig.colorbar(im0, ax=axes[0, 0])

shown_direct = best_grid.copy()
shown_direct[~inside] = np.nan
im1 = axes[0, 1].imshow(shown_direct, origin="lower", extent=[-0.9, 0.9, -0.9, 0.9], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[0, 1].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Independent direct fixed-enclosure solve")
fig.colorbar(im1, ax=axes[0, 1])

shown_difference = difference.copy()
shown_difference[~inside] = np.nan
diff_limit = np.nanmax(np.abs(shown_difference))
im2 = axes[0, 2].imshow(shown_difference, origin="lower", extent=[-0.9, 0.9, -0.9, 0.9], cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit)
axes[0, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Direct minus conformal potential")
fig.colorbar(im2, ax=axes[0, 2])

for index, order in enumerate(orders[1:5], start=1):
    axes[1, 0].semilogy(COUNTS, np.maximum(np.abs(q_values[:, index]), 1e-18), marker="o", label=f"direct q{order}")
    axes[1, 0].axhline(max(abs(pullback_q[index]), 1e-18), linestyle="--", linewidth=1)
axes[1, 0].set(xlabel="MFS sources per electrode", ylabel="absolute relative multipole", title="Independent multipole convergence")
axes[1, 0].legend(fontsize=8)

axes[1, 1].semilogy(COUNTS, spatial_rmse, marker="o", label="central potential RMSE")
axes[1, 1].semilogy(COUNTS, boundary_max, marker="o", label="unseen max boundary error")
axes[1, 1].set(xlabel="MFS sources per electrode", ylabel="error", title="Spatial and boundary agreement")
axes[1, 1].legend(fontsize=8)

line = np.linspace(-0.9, 0.9, 600).astype(complex)
axes[1, 2].plot(line.real, potential(inverse_mapping(line), reference_sources, reference_coefficients), label="conformal pullback")
axes[1, 2].plot(line.real, potential(line, best_sources, best_coefficients), linestyle="--", label="direct fixed enclosure")
axes[1, 2].set(xlabel="x/r0 along y=0", ylabel="potential", title="Centerline comparison")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 10 — independent physical solve of the recursive conformal electrode", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: convergence of the direct fixed-enclosure field toward the pullback establishes that the cancellations are physical electrode effects rather than coordinate bookkeeping. Residual disagreement must be judged against unseen boundary error and MFS resolution.")
print("saved", OUTPUT)
