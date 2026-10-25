import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/16_direct_lifetime_candidate_validation.png"
RADIUS = 0.8402181882818678
DISTANCE = 1.0 + RADIUS
OUTER_RADIUS = 5.0
KAPPA9 = -0.0005219378587971479
DELTA13 = 5.065762215517316e-05
P9 = np.arange(9, 19)
W9 = np.array([0.14526537, -0.10620197, 0.28421932, -0.42300312, -0.77440244, 14.74465186, -52.90593862, 91.26916034, -77.73953184, 26.50578109])
W9 /= np.sum(W9)
P13 = np.arange(16, 29)
W13 = np.array([6.78065474, -42.50938914, 128.37796771, -226.14901113, 228.63499372, -81.83172425, -89.9305892, 123.78893861, -34.42561199, -82.46601008, 164.01436013, -139.11150475, 45.82692563])
W13 /= np.sum(W13)
COUNTS = [40, 56, 72, 88]
ORDERS = np.arange(2, 31, 4)


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
    ninth = sum(weight * basis(z, 9, power) for weight, power in zip(W9, P9))
    thirteenth = sum(weight * basis(z, 13, power) for weight, power in zip(W13, P13))
    return z + KAPPA9 * ninth + DELTA13 * thirteenth


def mapping_derivative(z):
    ninth = sum(weight * basis_derivative(z, 9, power) for weight, power in zip(W9, P9))
    thirteenth = sum(weight * basis_derivative(z, 13, power) for weight, power in zip(W13, P13))
    return 1.0 + KAPPA9 * ninth + DELTA13 * thirteenth


def inverse_mapping(w):
    z = w.copy()
    for _ in range(18):
        z -= (mapping(z) - w) / mapping_derivative(z)
    return z


centers = DISTANCE * np.array([1.0, -1.0, 1.0j, -1.0j])
voltages = np.array([0.5, 0.5, -0.5, -0.5])


def solve_circle(count):
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
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def solve_direct(count):
    tb = 2.0 * np.pi * np.arange(3 * count) / (3 * count)
    ts = 2.0 * np.pi * np.arange(count) / count + 0.011
    boundary_curves = [mapping(center + RADIUS * np.exp(1j * tb)) for center in centers]
    source_curves = [mapping(center + RADIUS * np.exp(1j * ts)) for center in centers]
    boundary = np.concatenate(boundary_curves)
    sources = np.concatenate([center + 0.62 * (curve - center) for center, curve in zip(centers, source_curves)])
    values = np.concatenate([np.full(len(tb), voltage) for voltage in voltages])
    to = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    tso = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * to)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * tso)])
    values = np.concatenate([values, np.zeros(len(to))])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def potential(z, sources, coefficients):
    return np.log(np.abs(z[:, None] - sources[None, :])) @ coefficients[:-1] + coefficients[-1]


theta = 2.0 * np.pi * np.arange(2048) / 2048
probe_radius = 0.5
probe = probe_radius * np.exp(1j * theta)


def amplitudes(values):
    return np.array([2.0 * np.mean(values * np.cos(order * theta)) / probe_radius**order for order in ORDERS])


def relative(amplitude):
    return amplitude / amplitude[0] * probe_radius ** (ORDERS - 2)


circle_sources, circle_coefficients = solve_circle(96)
pullback_values = potential(inverse_mapping(probe), circle_sources, circle_coefficients)
pullback_a = amplitudes(pullback_values)
pullback_q = relative(pullback_a)

grid_axis = np.linspace(-0.9, 0.9, 181)
xx, yy = np.meshgrid(grid_axis, grid_axis)
grid = (xx + 1j * yy).ravel()
inside = xx**2 + yy**2 <= 0.9**2
pullback_grid = potential(inverse_mapping(grid), circle_sources, circle_coefficients).reshape(xx.shape)

unseen_theta = 2.0 * np.pi * (np.arange(3000) + 0.37) / 3000
unseen_rods = np.concatenate([mapping(center + RADIUS * np.exp(1j * unseen_theta)) for center in centers])
unseen_values = np.concatenate([np.full(len(unseen_theta), voltage) for voltage in voltages])
unseen_outer = OUTER_RADIUS * np.exp(1j * unseen_theta)

direct_a = []
direct_q = []
spatial_rmse = []
boundary_error = []
solutions = []
for count in COUNTS:
    sources, coefficients = solve_direct(count)
    values = potential(probe, sources, coefficients)
    a = amplitudes(values)
    direct_a.append(a)
    direct_q.append(relative(a))
    field = potential(grid, sources, coefficients).reshape(xx.shape)
    difference = field - pullback_grid
    spatial_rmse.append(np.sqrt(np.mean(difference[inside] ** 2)) / np.sqrt(np.mean(pullback_grid[inside] ** 2)))
    rod_error = np.max(np.abs(potential(unseen_rods, sources, coefficients) - unseen_values))
    outer_error = np.max(np.abs(potential(unseen_outer, sources, coefficients)))
    boundary_error.append(max(rod_error, outer_error))
    solutions.append((sources, coefficients, field, difference))

direct_a = np.array(direct_a)
direct_q = np.array(direct_q)
spatial_rmse = np.array(spatial_rmse)
boundary_error = np.array(boundary_error)
best_sources, best_coefficients, best_grid, best_difference = solutions[-1]

print("HYPOTHESIS: an independent fixed-enclosure solve will retain beta=0.025 and the q10/q14 cancellations of the physical lifetime-band electrode.")
print("pullback a2 eta", pullback_a[0], 2.0 * pullback_a[0], "q", pullback_q[:6])
for count, a, q, rmse, error in zip(COUNTS, direct_a, direct_q, spatial_rmse, boundary_error):
    print("count", count, "eta", 2.0 * a[0], "beta", a[1] / a[0], "q", q[:6], "central relative RMSE", rmse, "unseen boundary max", error)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
limit = np.max(np.abs(pullback_grid[inside]))
shown_pullback = pullback_grid.copy()
shown_pullback[~inside] = np.nan
image0 = axes[0, 0].imshow(shown_pullback, origin="lower", extent=[-0.9, 0.9, -0.9, 0.9], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[0, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Conformal construction")
fig.colorbar(image0, ax=axes[0, 0])

shown_direct = best_grid.copy()
shown_direct[~inside] = np.nan
image1 = axes[0, 1].imshow(shown_direct, origin="lower", extent=[-0.9, 0.9, -0.9, 0.9], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[0, 1].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Independent fixed-enclosure solve")
fig.colorbar(image1, ax=axes[0, 1])

shown_difference = best_difference.copy()
shown_difference[~inside] = np.nan
difference_limit = np.nanmax(np.abs(shown_difference))
image2 = axes[0, 2].imshow(shown_difference, origin="lower", extent=[-0.9, 0.9, -0.9, 0.9], cmap="coolwarm", vmin=-difference_limit, vmax=difference_limit)
axes[0, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Independent minus construction")
fig.colorbar(image2, ax=axes[0, 2])

for index, order in enumerate(ORDERS[1:6], start=1):
    axes[1, 0].semilogy(COUNTS, np.maximum(np.abs(direct_q[:, index]), 1e-18), marker="o", label=f"q{order}")
    axes[1, 0].axhline(max(abs(pullback_q[index]), 1e-18), linestyle="--", linewidth=1)
axes[1, 0].set(xlabel="MFS sources per electrode", ylabel="absolute relative coefficient", title="Physical multipole convergence")
axes[1, 0].legend(fontsize=8)

axes[1, 1].semilogy(COUNTS, spatial_rmse, marker="o", label="central potential RMSE")
axes[1, 1].semilogy(COUNTS, boundary_error, marker="o", label="unseen boundary max")
axes[1, 1].set(xlabel="MFS sources per electrode", ylabel="error", title="Independent-solve resolution")
axes[1, 1].legend(fontsize=8)

eta_relative_error = np.abs(2.0 * direct_a[:, 0] / (2.0 * pullback_a[0]) - 1.0)
beta_absolute_error = np.abs(direct_a[:, 1] / direct_a[:, 0] - 0.025)
axes[1, 2].semilogy(COUNTS, eta_relative_error, marker="o", label="relative eta error")
axes[1, 2].semilogy(COUNTS, beta_absolute_error, marker="s", label="absolute beta error")
axes[1, 2].set(xlabel="MFS sources per electrode", ylabel="error", title="Efficiency and intended a6 converge")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 16 — independent solve of the lifetime-band electrode", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: agreement or disagreement with beta=0.025 must be read together with the independent solver's boundary and spatial convergence; this distinguishes a physical electrode effect from conformal bookkeeping.")
print("saved", OUTPUT)
