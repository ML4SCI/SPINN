import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/51_implicit_mfs_derivative.png"
COUNT = 36
RING_RADIUS = 0.80
POINT = np.array([0.70, 0.005, 0.0025])
PHYSICAL_STEPS = np.array([0.001, 0.0002, 0.0001])
COORDINATE_SCALES = np.array([0.05, 0.02, 0.01])
MATRIX_STEPS = [1e-2, 1e-3, 1e-4]
ORDERS = np.arange(2, 39, 4)


def local_support(radius, c4, c8, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    support = radius * (1.0 + c4 * np.cos(4.0 * theta) + c8 * np.cos(8.0 * theta))
    derivative = radius * (-4.0 * c4 * np.sin(4.0 * theta) - 8.0 * c8 * np.sin(8.0 * theta))
    return (support * np.cos(theta) - derivative * np.sin(theta)) + 1j * (support * np.sin(theta) + derivative * np.cos(theta))


def matrices(point):
    radius, c4, c8 = point
    distance = 1.0 + radius * (1.0 + c4 + c8)
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    rotations = np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    shape = local_support(radius, c4, c8, 3 * COUNT)
    source_shape = 0.70 * local_support(radius, c4, c8, COUNT, 0.013)
    boundary = np.concatenate([center + rotation * shape for center, rotation in zip(centers, rotations)])
    sources = np.concatenate([center + rotation * source_shape for center, rotation in zip(centers, rotations)])
    values = np.concatenate([np.full(3 * COUNT, voltage) for voltage in voltages])
    outer_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([boundary, 5.0 * np.exp(1j * outer_theta)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * source_theta)])
    values = np.concatenate([values, np.zeros(12 * COUNT)])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    theta = 2.0 * np.pi * np.arange(2048) / 2048
    ring = RING_RADIUS * np.exp(1j * theta)
    evaluation = np.column_stack([np.log(np.abs(ring[:, None] - sources[None, :])), np.ones(len(ring))])
    harmonic_projection = np.array([2.0 * np.cos(order * theta) / (len(theta) * RING_RADIUS**order) for order in ORDERS])
    return matrix, values, evaluation, harmonic_projection


def truncated_pseudoinverse(matrix):
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > 1e-12 * singular[0]
    inverse = (vt[keep].T / singular[keep]) @ u[:, keep].T
    return inverse, singular, keep


def solve(point):
    matrix, values, evaluation, projection = matrices(point)
    inverse, singular, keep = truncated_pseudoinverse(matrix)
    coefficients = inverse @ values
    harmonics = projection @ (evaluation @ coefficients)
    return harmonics, evaluation @ coefficients, (matrix, values, evaluation, projection, inverse, coefficients, singular, keep)


base_harmonics, base_potential, state = solve(POINT)
matrix, values, evaluation, projection, inverse, coefficients, singular, keep = state
residual = values - matrix @ coefficients
null_projector = np.eye(matrix.shape[1]) - inverse @ matrix

finite_harmonic_derivatives = []
finite_potential_derivatives = []
for axis in range(3):
    plus = POINT.copy()
    minus = POINT.copy()
    plus[axis] += PHYSICAL_STEPS[axis]
    minus[axis] -= PHYSICAL_STEPS[axis]
    plus_h, plus_v, _ = solve(plus)
    minus_h, minus_v, _ = solve(minus)
    finite_harmonic_derivatives.append((plus_h - minus_h) / (2.0 * PHYSICAL_STEPS[axis]))
    finite_potential_derivatives.append((plus_v - minus_v) / (2.0 * PHYSICAL_STEPS[axis]))
finite_harmonic_derivatives = np.array(finite_harmonic_derivatives)
finite_potential_derivatives = np.array(finite_potential_derivatives)

implicit_by_step = []
potential_by_step = []
for normalized_step in MATRIX_STEPS:
    harmonic_rows = []
    potential_rows = []
    for axis in range(3):
        plus = POINT.copy()
        minus = POINT.copy()
        physical_delta = COORDINATE_SCALES[axis] * normalized_step
        plus[axis] += physical_delta
        minus[axis] -= physical_delta
        plus_matrix, _, plus_evaluation, _ = matrices(plus)
        minus_matrix, _, minus_evaluation, _ = matrices(minus)
        matrix_derivative = (plus_matrix - minus_matrix) / (2.0 * physical_delta)
        evaluation_derivative = (plus_evaluation - minus_evaluation) / (2.0 * physical_delta)
        coefficient_derivative = (
            -inverse @ (matrix_derivative @ coefficients)
            + inverse @ inverse.T @ (matrix_derivative.T @ residual)
            + null_projector @ matrix_derivative.T @ inverse.T @ coefficients
        )
        potential_derivative = evaluation_derivative @ coefficients + evaluation @ coefficient_derivative
        harmonic_rows.append(projection @ potential_derivative)
        potential_rows.append(potential_derivative)
    implicit_by_step.append(np.array(harmonic_rows))
    potential_by_step.append(np.array(potential_rows))
implicit_by_step = np.array(implicit_by_step)
potential_by_step = np.array(potential_by_step)


def beta_derivative(harmonics, derivative):
    return (derivative[1] * harmonics[0] - harmonics[1] * derivative[0]) / harmonics[0] ** 2


finite_beta = np.array([beta_derivative(base_harmonics, row) for row in finite_harmonic_derivatives])
implicit_beta = np.array([[beta_derivative(base_harmonics, row) for row in block] for block in implicit_by_step])
harmonic_errors = np.linalg.norm(implicit_by_step - finite_harmonic_derivatives[None, :, :], axis=2) / np.linalg.norm(finite_harmonic_derivatives, axis=1)[None, :]
potential_errors = np.linalg.norm(potential_by_step - finite_potential_derivatives[None, :, :], axis=2) / np.linalg.norm(finite_potential_derivatives, axis=1)[None, :]
beta_errors = np.abs((implicit_beta - finite_beta[None, :]) / finite_beta[None, :])

print("HYPOTHESIS: differentiating the truncated least-squares MFS system will reproduce geometry finite differences without two new field solves per derivative label.")
print("matrix shape rank condition retained", matrix.shape, np.count_nonzero(keep), singular[0] / singular[keep][-1])
print("boundary residual relative", np.linalg.norm(residual) / np.linalg.norm(values))
print("finite beta derivatives", finite_beta)
for step, h_error, v_error, b_error in zip(MATRIX_STEPS, harmonic_errors, potential_errors, beta_errors):
    print("matrix derivative step", step, "harmonic", h_error, "ring V", v_error, "beta", b_error)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].semilogy(singular / singular[0], marker=".")
axes[0, 0].axhline(1e-12, color="red", linestyle="--", label="truncation")
axes[0, 0].set(xlabel="singular-value index", ylabel="normalized singular value", title="MFS least-squares spectrum")
axes[0, 0].legend(fontsize=8)
labels = ["R", "c4", "c8"]
for axis in range(3):
    axes[0, 1].plot(MATRIX_STEPS, harmonic_errors[:, axis], marker="o", label=labels[axis])
    axes[0, 2].plot(MATRIX_STEPS, potential_errors[:, axis], marker="o", label=labels[axis])
axes[0, 1].set(xscale="log", yscale="log", xlabel="normalized matrix-difference step", ylabel="relative error", title="All harmonic-coefficient derivatives")
axes[0, 2].set(xscale="log", yscale="log", xlabel="normalized matrix-difference step", ylabel="relative error", title="Full ring-potential derivatives")
axes[0, 1].legend(fontsize=8)
axes[0, 2].legend(fontsize=8)
positions = np.arange(3)
width = 0.18
axes[1, 0].bar(positions - width, finite_beta, width, label="fresh geometry finite difference")
for index, step in enumerate(MATRIX_STEPS):
    axes[1, 0].bar(positions + index * width, implicit_beta[index], width, label=f"implicit h={step:g}")
axes[1, 0].set_xticks(positions, labels)
axes[1, 0].set(ylabel="d beta/d geometry", title="Optimizer-facing derivative")
axes[1, 0].legend(fontsize=7)
for axis in range(3):
    axes[1, 1].plot(MATRIX_STEPS, beta_errors[:, axis], marker="o", label=labels[axis])
axes[1, 1].set(xscale="log", yscale="log", xlabel="normalized matrix-difference step", ylabel="relative beta-gradient error", title="Does implicit differentiation pass?")
axes[1, 1].legend(fontsize=8)
theta_plot = np.linspace(0.0, 2.0 * np.pi, len(base_potential), endpoint=False)
axes[1, 2].plot(theta_plot, finite_potential_derivatives[2], color="black", label="fresh finite difference")
axes[1, 2].plot(theta_plot, potential_by_step[-1, 2], linestyle="--", label="implicit")
axes[1, 2].set(xlabel="angle on 0.8 r0 ring", ylabel="dV/dc8", title="Spatial derivative, not only beta")
axes[1, 2].legend(fontsize=8)
fig.suptitle("Claude-tree stress test 51 — implicit differentiation of the exact MFS layer", fontsize=14)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: implicit differentiation is acceptable only if the full spatial and beta derivatives converge to fresh geometry solves; the singular spectrum shows why the retained-rank convention is part of the model definition.")
print("saved", OUTPUT)
