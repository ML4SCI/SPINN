import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import lstsq


REFINE_C4 = os.environ.get("REFINE_C4", "0") == "1"
OUTPUT = "exp/claude_tree_stress_test/results/33_adaptive_c4_refinement.png" if REFINE_C4 else "exp/claude_tree_stress_test/results/32_two_coordinate_harmonic_head.png"
TRAIN_R = np.array([0.65, 0.70, 0.75])
TRAIN_C4 = np.linspace(-0.02, 0.02, 5) if REFINE_C4 else np.array([-0.02, 0.0, 0.02])
TEST_C4 = [-0.015, -0.005, 0.005, 0.015] if REFINE_C4 else [-0.01, 0.01]
TEST = [(radius, c4) for radius in [0.675, 0.725] for c4 in TEST_C4]
ORDERS = np.arange(2, 39, 4)
COUNT = 48
OUTER_RADIUS = 5.0
RING_RADIUS = 0.90


def local_support(radius, c4, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    support = radius * (1.0 + c4 * np.cos(4.0 * theta))
    derivative = -4.0 * radius * c4 * np.sin(4.0 * theta)
    return (support * np.cos(theta) - derivative * np.sin(theta)) + 1j * (support * np.sin(theta) + derivative * np.cos(theta))


def solve(radius, c4):
    distance = 1.0 + radius * (1.0 + c4)
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    rotations = np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    local_boundary = local_support(radius, c4, 3 * COUNT)
    local_sources = 0.70 * local_support(radius, c4, COUNT, 0.013)
    boundary = np.concatenate([center + rotation * local_boundary for center, rotation in zip(centers, rotations)])
    sources = np.concatenate([center + rotation * local_sources for center, rotation in zip(centers, rotations)])
    values = np.concatenate([np.full(3 * COUNT, voltage) for voltage in voltages])
    theta_outer = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    theta_sources = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * theta_outer)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * theta_sources)])
    values = np.concatenate([values, np.zeros(12 * COUNT)])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = lstsq(matrix, values, cond=1e-12, lapack_driver="gelsy")
    return sources, coefficients


def reference(radius, c4, z):
    sources, coefficients = solve(radius, c4)
    delta = z[:, None] - sources[None, :]
    potential = np.log(np.abs(delta)) @ coefficients[:-1] + coefficients[-1]
    complex_gradient = np.sum(coefficients[None, :-1] / np.conj(delta), axis=1)
    return potential, np.column_stack([-complex_gradient.real, -complex_gradient.imag])


theta = 2.0 * np.pi * np.arange(2048) / 2048
ring = RING_RADIUS * np.exp(1j * theta)


def exact_coefficients(radius, c4):
    values = reference(radius, c4, ring)[0]
    return np.array([2.0 * np.mean(values * np.cos(order * theta)) / RING_RADIUS**order for order in ORDERS])


training_pairs = [(radius, c4) for radius in TRAIN_R for c4 in TRAIN_C4]
training_coefficients = np.array([exact_coefficients(*pair) for pair in training_pairs])
training_coordinates = np.array([[(radius - 0.70) / 0.05, c4 / 0.02] for radius, c4 in training_pairs])
DEGREES = [2, 4 if REFINE_C4 else 2]
design = np.polynomial.chebyshev.chebvander2d(training_coordinates[:, 0], training_coordinates[:, 1], DEGREES).reshape(len(training_pairs), -1)
response = np.linalg.solve(design, training_coefficients)


def predicted_coefficients(radius, c4):
    coordinate = np.array([(radius - 0.70) / 0.05, c4 / 0.02])
    row = np.polynomial.chebyshev.chebvander2d(coordinate[0], coordinate[1], DEGREES).reshape(-1)
    return row @ response


def harmonic_field(z, coefficients):
    potential = np.zeros(len(z))
    complex_field = np.zeros(len(z), dtype=complex)
    for order, coefficient in zip(ORDERS, coefficients):
        potential += coefficient * np.real(z**order)
        complex_field -= order * coefficient * np.conj(z ** (order - 1))
    return potential, np.column_stack([complex_field.real, complex_field.imag])


rng = np.random.default_rng(3201)
angle = rng.uniform(0.0, 2.0 * np.pi, 3000)
radial = 0.80 * np.sqrt(rng.uniform(0.0, 1.0, 3000))
probe = radial * np.exp(1j * angle)
records = {}
for radius, c4 in TEST:
    exact_v, exact_e = reference(radius, c4, probe)
    coefficients = predicted_coefficients(radius, c4)
    predicted_v, predicted_e = harmonic_field(probe, coefficients)
    exact_a = exact_coefficients(radius, c4)
    h_r = 0.001
    h_c = 0.0002
    exact_dr = ((exact_coefficients(radius + h_r, c4)[1] / exact_coefficients(radius + h_r, c4)[0]) - (exact_coefficients(radius - h_r, c4)[1] / exact_coefficients(radius - h_r, c4)[0])) / (2.0 * h_r)
    exact_dc = ((exact_coefficients(radius, c4 + h_c)[1] / exact_coefficients(radius, c4 + h_c)[0]) - (exact_coefficients(radius, c4 - h_c)[1] / exact_coefficients(radius, c4 - h_c)[0])) / (2.0 * h_c)
    surrogate_h = 1e-5
    pred_dr = ((predicted_coefficients(radius + surrogate_h, c4)[1] / predicted_coefficients(radius + surrogate_h, c4)[0]) - (predicted_coefficients(radius - surrogate_h, c4)[1] / predicted_coefficients(radius - surrogate_h, c4)[0])) / (2.0 * surrogate_h)
    pred_dc = ((predicted_coefficients(radius, c4 + surrogate_h)[1] / predicted_coefficients(radius, c4 + surrogate_h)[0]) - (predicted_coefficients(radius, c4 - surrogate_h)[1] / predicted_coefficients(radius, c4 - surrogate_h)[0])) / (2.0 * surrogate_h)
    records[(radius, c4)] = {
        "v": np.sqrt(np.mean(((predicted_v - exact_v) / np.max(np.abs(exact_v))) ** 2)),
        "e": np.sqrt(np.mean((np.linalg.norm(predicted_e - exact_e, axis=1) / np.max(np.linalg.norm(exact_e, axis=1))) ** 2)),
        "beta_error": abs(coefficients[1] / coefficients[0] - exact_a[1] / exact_a[0]),
        "exact_dr": exact_dr,
        "pred_dr": pred_dr,
        "exact_dc": exact_dc,
        "pred_dc": pred_dc,
    }
    print("unseen", radius, c4, records[(radius, c4)])

surface_r = np.linspace(0.65, 0.75, 13)
surface_c = np.linspace(-0.02, 0.02, 13)
exact_surface = np.zeros((len(surface_c), len(surface_r)))
predicted_surface = np.zeros_like(exact_surface)
for i, c4 in enumerate(surface_c):
    for j, radius in enumerate(surface_r):
        exact_a = exact_coefficients(radius, c4)
        predicted_a = predicted_coefficients(radius, c4)
        exact_surface[i, j] = exact_a[1] / exact_a[0]
        predicted_surface[i, j] = predicted_a[1] / predicted_a[0]

shown = TEST[2]
exact_v, _ = reference(*shown, probe)
predicted_v, _ = harmonic_field(probe, predicted_coefficients(*shown))
grid_axis = np.linspace(-0.80, 0.80, 81)
xx, yy = np.meshgrid(grid_axis, grid_axis)
grid = (xx + 1j * yy).ravel()
inside = np.abs(grid) <= 0.80
grid_exact = reference(*shown, grid[inside])[0]
grid_prediction = harmonic_field(grid[inside], predicted_coefficients(*shown))[0]
difference = np.full(xx.shape, np.nan)
difference.ravel()[inside] = grid_prediction - grid_exact

print("HYPOTHESIS:", "refining only the c4 coordinate identified by the striped error map will reduce beta error without a full 5x5 tensor grid." if REFINE_C4 else "the exact-harmonic head can extend from one geometry coordinate to a two-coordinate noncircular family using a small tensor-product snapshot grid.")
print("worst unseen V E beta", max(row["v"] for row in records.values()), max(row["e"] for row in records.values()), max(row["beta_error"] for row in records.values()))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
image0 = axes[0, 0].imshow(exact_surface, origin="lower", aspect="auto", extent=[surface_r[0], surface_r[-1], surface_c[0], surface_c[-1]], cmap="viridis")
axes[0, 0].scatter([pair[0] for pair in training_pairs], [pair[1] for pair in training_pairs], color="white", marker="o", label="training")
axes[0, 0].scatter([pair[0] for pair in TEST], [pair[1] for pair in TEST], color="red", marker="x", label="unseen")
axes[0, 0].set(xlabel="radius/r0", ylabel="c4 support coefficient", title="Exact beta over a noncircular family")
axes[0, 0].legend(fontsize=8)
fig.colorbar(image0, ax=axes[0, 0])

error_surface = np.abs(predicted_surface - exact_surface)
image1 = axes[0, 1].imshow(error_surface, origin="lower", aspect="auto", extent=[surface_r[0], surface_r[-1], surface_c[0], surface_c[-1]], cmap="magma")
axes[0, 1].set(xlabel="radius/r0", ylabel="c4", title="Absolute beta interpolation error")
fig.colorbar(image1, ax=axes[0, 1])

positions = np.arange(len(TEST))
axes[0, 2].bar(positions - 0.2, [records[pair]["v"] for pair in TEST], 0.4, label="V RMSE")
axes[0, 2].bar(positions + 0.2, [records[pair]["e"] for pair in TEST], 0.4, label="E RMSE")
axes[0, 2].set_xticks(positions, [f"{r:.3f}\n{c:+.3f}" for r, c in TEST], rotation=25, fontsize=6)
axes[0, 2].set(yscale="log", ylabel="normalized RMSE", title="Held-out full central fields")
axes[0, 2].legend(fontsize=8)

axes[1, 0].bar(positions - 0.2, [records[pair]["exact_dr"] for pair in TEST], 0.4, label="exact")
axes[1, 0].bar(positions + 0.2, [records[pair]["pred_dr"] for pair in TEST], 0.4, label="harmonic head")
axes[1, 0].set_xticks(positions, [f"{r:.3f}\n{c:+.3f}" for r, c in TEST], rotation=25, fontsize=6)
axes[1, 0].set(ylabel="d beta / d radius", title="Unseen radius derivative")
axes[1, 0].legend(fontsize=8)

axes[1, 1].bar(positions - 0.2, [records[pair]["exact_dc"] for pair in TEST], 0.4, label="exact")
axes[1, 1].bar(positions + 0.2, [records[pair]["pred_dc"] for pair in TEST], 0.4, label="harmonic head")
axes[1, 1].set_xticks(positions, [f"{r:.3f}\n{c:+.3f}" for r, c in TEST], rotation=25, fontsize=6)
axes[1, 1].set(ylabel="d beta / d c4", title="Unseen noncircular-shape derivative")
axes[1, 1].legend(fontsize=8)

limit = np.nanmax(np.abs(difference))
image2 = axes[1, 2].imshow(difference, origin="lower", extent=[-0.8, 0.8, -0.8, 0.8], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title=f"Unseen R={shown[0]:.3f}, c4={shown[1]:+.3f}: error")
fig.colorbar(image2, ax=axes[1, 2])

fig.suptitle("Claude-tree stress test 33 — plot-directed sparse refinement of the c4 shape coordinate" if REFINE_C4 else "Claude-tree stress test 32 — exact-harmonic conditioning on two shape coordinates", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: passing this test establishes local two-coordinate interpolation, not free-form generality. Tensor grids scale exponentially, so the next architecture needs adaptive sparse geometry sampling or a low-rank coefficient map.")
print("saved", OUTPUT)
