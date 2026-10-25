import matplotlib.pyplot as plt
import numpy as np
import runpy


BASE = runpy.run_path("exp/claude_tree_stress_test/51_implicit_mfs_derivative.py")
OUTPUT = "exp/claude_tree_stress_test/results/52_analytic_mfs_derivative.png"
POINT = BASE["POINT"]
COUNT = BASE["COUNT"]
matrix, values, evaluation, projection, inverse, coefficients, singular, keep = BASE["state"]
residual = BASE["residual"]
null_projector = np.eye(matrix.shape[1]) - inverse @ matrix
finite_harmonic = BASE["finite_harmonic_derivatives"]
finite_potential = BASE["finite_potential_derivatives"]
finite_beta = BASE["finite_beta"]


def local_and_derivatives(radius, c4, c8, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    cosine = np.cos(theta)
    sine = np.sin(theta)
    cos4 = np.cos(4.0 * theta)
    sin4 = np.sin(4.0 * theta)
    cos8 = np.cos(8.0 * theta)
    sin8 = np.sin(8.0 * theta)
    h = radius * (1.0 + c4 * cos4 + c8 * cos8)
    hp = radius * (-4.0 * c4 * sin4 - 8.0 * c8 * sin8)

    def curve(h_value, hp_value):
        return (h_value * cosine - hp_value * sine) + 1j * (h_value * sine + hp_value * cosine)

    point = curve(h, hp)
    derivatives = np.array([
        curve(1.0 + c4 * cos4 + c8 * cos8, -4.0 * c4 * sin4 - 8.0 * c8 * sin8),
        curve(radius * cos4, -4.0 * radius * sin4),
        curve(radius * cos8, -8.0 * radius * sin8),
    ])
    return point, derivatives


def geometry_and_derivatives(point):
    radius, c4, c8 = point
    distance = 1.0 + radius * (1.0 + c4 + c8)
    distance_derivatives = np.array([1.0 + c4 + c8, radius, radius])
    directions = np.array([1.0, -1.0, 1.0j, -1.0j])
    centers = distance * directions
    center_derivatives = distance_derivatives[:, None] * directions[None, :]
    shape, shape_derivatives = local_and_derivatives(radius, c4, c8, 3 * COUNT)
    source_shape, source_shape_derivatives = local_and_derivatives(radius, c4, c8, COUNT, 0.013)
    boundary = np.concatenate([center + rotation * shape for center, rotation in zip(centers, directions)])
    sources = np.concatenate([center + rotation * 0.70 * source_shape for center, rotation in zip(centers, directions)])
    boundary_derivatives = np.array([
        np.concatenate([center_derivatives[axis, rod] + directions[rod] * shape_derivatives[axis] for rod in range(4)])
        for axis in range(3)
    ])
    source_derivatives = np.array([
        np.concatenate([center_derivatives[axis, rod] + directions[rod] * 0.70 * source_shape_derivatives[axis] for rod in range(4)])
        for axis in range(3)
    ])
    outer_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([boundary, 5.0 * np.exp(1j * outer_theta)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * source_theta)])
    boundary_derivatives = np.column_stack([boundary_derivatives, np.zeros((3, 12 * COUNT), dtype=complex)])
    source_derivatives = np.column_stack([source_derivatives, np.zeros((3, 4 * COUNT), dtype=complex)])
    return boundary, sources, boundary_derivatives, source_derivatives


boundary, sources, boundary_derivatives, source_derivatives = geometry_and_derivatives(POINT)
delta = boundary[:, None] - sources[None, :]
theta = 2.0 * np.pi * np.arange(2048) / 2048
ring = 0.80 * np.exp(1j * theta)
ring_delta = ring[:, None] - sources[None, :]
analytic_harmonic = []
analytic_potential = []
matrix_comparisons = []
for axis in range(3):
    matrix_derivative = np.column_stack([
        np.real((boundary_derivatives[axis, :, None] - source_derivatives[axis, None, :]) / delta),
        np.zeros(len(boundary)),
    ])
    evaluation_derivative = np.column_stack([
        np.real(-source_derivatives[axis, None, :] / ring_delta),
        np.zeros(len(ring)),
    ])
    coefficient_derivative = (
        -inverse @ (matrix_derivative @ coefficients)
        + inverse @ inverse.T @ (matrix_derivative.T @ residual)
        + null_projector @ matrix_derivative.T @ inverse.T @ coefficients
    )
    potential_derivative = evaluation_derivative @ coefficients + evaluation @ coefficient_derivative
    analytic_potential.append(potential_derivative)
    analytic_harmonic.append(projection @ potential_derivative)
    step = BASE["COORDINATE_SCALES"][axis] * 1e-4
    plus = POINT.copy()
    minus = POINT.copy()
    plus[axis] += step
    minus[axis] -= step
    numeric_matrix = (BASE["matrices"](plus)[0] - BASE["matrices"](minus)[0]) / (2.0 * step)
    matrix_comparisons.append(np.linalg.norm(matrix_derivative - numeric_matrix) / np.linalg.norm(numeric_matrix))
analytic_harmonic = np.array(analytic_harmonic)
analytic_potential = np.array(analytic_potential)
analytic_beta = np.array([BASE["beta_derivative"](BASE["base_harmonics"], row) for row in analytic_harmonic])
harmonic_errors = np.linalg.norm(analytic_harmonic - finite_harmonic, axis=1) / np.linalg.norm(finite_harmonic, axis=1)
potential_errors = np.linalg.norm(analytic_potential - finite_potential, axis=1) / np.linalg.norm(finite_potential, axis=1)
beta_errors = np.abs((analytic_beta - finite_beta) / finite_beta)

print("HYPOTHESIS: analytic boundary/source velocities and d log|b-s| will match both matrix differences and fresh neighboring-geometry solves.")
print("analytic versus matrix-difference dA", matrix_comparisons)
print("analytic versus fresh solve harmonic ringV beta", harmonic_errors, potential_errors, beta_errors)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
labels = ["R", "c4", "c8"]
axes[0, 0].bar(labels, matrix_comparisons)
axes[0, 0].set(yscale="log", ylabel="relative error", title="Analytic dA vs matrix difference")
axes[0, 1].bar(np.arange(3) - 0.2, harmonic_errors, 0.4, label="all harmonics")
axes[0, 1].bar(np.arange(3) + 0.2, potential_errors, 0.4, label="ring potential")
axes[0, 1].set_xticks(np.arange(3), labels)
axes[0, 1].set(yscale="log", ylabel="relative error", title="Analytic derivative vs fresh solves")
axes[0, 1].legend(fontsize=8)
axes[0, 2].bar(labels, beta_errors)
axes[0, 2].set(yscale="log", ylabel="relative beta-gradient error", title="Optimizer-facing gate")
positions = np.arange(3)
axes[1, 0].bar(positions - 0.2, finite_beta, 0.4, label="fresh geometry solves")
axes[1, 0].bar(positions + 0.2, analytic_beta, 0.4, label="analytic implicit")
axes[1, 0].set_xticks(positions, labels)
axes[1, 0].set(ylabel="d beta/d geometry", title="Same physical shape derivative")
axes[1, 0].legend(fontsize=8)
axes[1, 1].plot(theta, finite_potential[2], color="black", label="fresh geometry solves")
axes[1, 1].plot(theta, analytic_potential[2], linestyle="--", label="analytic implicit")
axes[1, 1].set(xlabel="angle on 0.8 r0 ring", ylabel="dV/dc8", title="Full spatial derivative")
axes[1, 1].legend(fontsize=8)
rod_points = boundary[:12 * COUNT:6]
rod_velocity = boundary_derivatives[2, :12 * COUNT:6]
axes[1, 2].scatter(rod_points.real, rod_points.imag, s=8, color="black")
axes[1, 2].quiver(rod_points.real, rod_points.imag, rod_velocity.real, rod_velocity.imag, angles="xy", scale_units="xy", scale=25.0, color="tab:red")
axes[1, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Analytic c8 boundary velocity")
fig.suptitle("Claude-tree stress test 52 — fully analytic exact-layer shape derivative", fontsize=14)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: one retained-rank factorization plus analytic geometry velocities supplies every shape direction; no neighboring matrix assembly or field solve is required.")
print("saved", OUTPUT)
