import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/30_geometry_conditioned_harmonic_head.png"
TRAIN_RADII = np.array([0.60, 0.64, 0.68, 0.72, 0.76])
TEST_RADII = np.array([0.62, 0.66, 0.70, 0.74])
ORDERS = np.arange(2, 39, 4)
OUTER_RADIUS = 5.0
COUNT = 72
RING_RADIUS = 0.90


def circle_points(center, radius, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    return center + radius * np.exp(1j * theta)


def solve_circle(radius):
    distance = 1.0 + radius
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    boundary = np.concatenate([circle_points(center, radius, 3 * COUNT) for center in centers])
    sources = np.concatenate([circle_points(center, 0.70 * radius, COUNT, 0.013) for center in centers])
    values = np.concatenate([np.full(3 * COUNT, voltage) for voltage in voltages])
    boundary = np.concatenate([boundary, circle_points(0.0, OUTER_RADIUS, 12 * COUNT)])
    sources = np.concatenate([sources, circle_points(0.0, 6.5, 4 * COUNT, 0.007)])
    values = np.concatenate([values, np.zeros(12 * COUNT)])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def reference(radius, z):
    sources, coefficients = solve_circle(radius)
    delta = z[:, None] - sources[None, :]
    potential = np.log(np.abs(delta)) @ coefficients[:-1] + coefficients[-1]
    complex_gradient = np.sum(coefficients[None, :-1] / np.conj(delta), axis=1)
    field = -np.column_stack([complex_gradient.real, complex_gradient.imag])
    return potential, field


theta = 2.0 * np.pi * np.arange(4096) / 4096
ring = RING_RADIUS * np.exp(1j * theta)


def exact_coefficients(radius):
    values = reference(radius, ring)[0]
    return np.array([2.0 * np.mean(values * np.cos(order * theta)) / RING_RADIUS**order for order in ORDERS])


training_coefficients = np.array([exact_coefficients(radius) for radius in TRAIN_RADII])
t_train = (TRAIN_RADII - 0.68) / 0.08
response_coefficients = np.column_stack([
    np.polynomial.chebyshev.chebfit(t_train, training_coefficients[:, index], 4)
    for index in range(len(ORDERS))
])


def predicted_coefficients(radius):
    t = (radius - 0.68) / 0.08
    return np.array([np.polynomial.chebyshev.chebval(t, response_coefficients[:, index]) for index in range(len(ORDERS))])


def predicted_derivative(radius):
    t = (radius - 0.68) / 0.08
    derivative = []
    for index in range(len(ORDERS)):
        polynomial = np.polynomial.chebyshev.chebder(response_coefficients[:, index])
        derivative.append(np.polynomial.chebyshev.chebval(t, polynomial) / 0.08)
    return np.array(derivative)


def harmonic_field(z, coefficients, count=None):
    count = len(ORDERS) if count is None else count
    potential = np.zeros(len(z))
    complex_field = np.zeros(len(z), dtype=complex)
    for order, coefficient in zip(ORDERS[:count], coefficients[:count]):
        potential += coefficient * np.real(z**order)
        complex_field -= order * coefficient * np.conj(z ** (order - 1))
    return potential, np.column_stack([complex_field.real, complex_field.imag])


axis = np.linspace(-0.80, 0.80, 81)
xx, yy = np.meshgrid(axis, axis)
all_z = (xx + 1j * yy).ravel()
inside = np.abs(all_z) <= 0.80
probe = all_z[inside]
records = {}
for radius in TEST_RADII:
    exact_values, exact_field = reference(radius, probe)
    coefficients = predicted_coefficients(radius)
    values, field = harmonic_field(probe, coefficients)
    exact_a = exact_coefficients(radius)
    derivative = predicted_derivative(radius)
    predicted_beta_derivative = (derivative[1] * coefficients[0] - coefficients[1] * derivative[0]) / coefficients[0] ** 2
    h = 0.002
    minus = exact_coefficients(radius - h)
    plus = exact_coefficients(radius + h)
    exact_beta_derivative = ((plus[1] / plus[0]) - (minus[1] / minus[0])) / (2.0 * h)
    records[radius] = {
        "v": np.sqrt(np.mean(((values - exact_values) / np.max(np.abs(exact_values))) ** 2)),
        "e": np.sqrt(np.mean((np.linalg.norm(field - exact_field, axis=1) / np.max(np.linalg.norm(exact_field, axis=1))) ** 2)),
        "exact_beta": exact_a[1] / exact_a[0],
        "predicted_beta": coefficients[1] / coefficients[0],
        "exact_eta": 2.0 * exact_a[0],
        "predicted_eta": 2.0 * coefficients[0],
        "exact_derivative": exact_beta_derivative,
        "predicted_derivative": predicted_beta_derivative,
    }
    print("unseen radius", radius, records[radius])

shown_radius = 0.70
shown_exact, _ = reference(shown_radius, probe)
shown_prediction, _ = harmonic_field(probe, predicted_coefficients(shown_radius))
difference = np.full(xx.shape, np.nan)
difference.ravel()[inside] = shown_prediction - shown_exact

truncation_counts = np.arange(1, len(ORDERS) + 1)
truncation_v = []
truncation_e = []
exact_values, exact_field = reference(shown_radius, probe)
for count in truncation_counts:
    values, field = harmonic_field(probe, predicted_coefficients(shown_radius), count)
    truncation_v.append(np.sqrt(np.mean(((values - exact_values) / np.max(np.abs(exact_values))) ** 2)))
    truncation_e.append(np.sqrt(np.mean((np.linalg.norm(field - exact_field, axis=1) / np.max(np.linalg.norm(exact_field, axis=1))) ** 2)))

print("HYPOTHESIS: an exact D4 harmonic head conditioned only on geometry coefficients will remove PIG/PIXEL representation artifacts in the source-free objective disc.")
print("worst unseen V E beta derivative relative", max(row["v"] for row in records.values()), max(row["e"] for row in records.values()), max(abs(row["predicted_beta"] - row["exact_beta"]) for row in records.values()), max(abs(row["predicted_derivative"] - row["exact_derivative"]) / abs(row["exact_derivative"]) for row in records.values()))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
fine_radii = np.linspace(0.60, 0.76, 81)
exact_beta = np.array([exact_coefficients(radius)[1] / exact_coefficients(radius)[0] for radius in fine_radii])
predicted_beta = np.array([predicted_coefficients(radius)[1] / predicted_coefficients(radius)[0] for radius in fine_radii])
axes[0, 0].plot(fine_radii, exact_beta, color="black", label="exact solve")
axes[0, 0].plot(fine_radii, predicted_beta, linestyle="--", color="tab:red", label="harmonic response head")
axes[0, 0].scatter(TRAIN_RADII, training_coefficients[:, 1] / training_coefficients[:, 0], label="five training geometries")
axes[0, 0].set(xlabel="rod radius/r0", ylabel="beta=a6/a2", title="Low-order geometry response, exact spatial physics")
axes[0, 0].legend(fontsize=8)

positions = np.arange(len(TEST_RADII))
axes[0, 1].bar(positions - 0.2, [records[r]["v"] for r in TEST_RADII], 0.4, label="V RMSE")
axes[0, 1].bar(positions + 0.2, [records[r]["e"] for r in TEST_RADII], 0.4, label="E RMSE")
axes[0, 1].set_xticks(positions, [f"{r:.2f}" for r in TEST_RADII])
axes[0, 1].set(yscale="log", xlabel="unseen rod radius/r0", ylabel="normalized spatial RMSE", title="Full central-field interpolation")
axes[0, 1].legend(fontsize=8)

axes[0, 2].bar(positions - 0.2, [records[r]["exact_derivative"] for r in TEST_RADII], 0.4, label="exact finite difference")
axes[0, 2].bar(positions + 0.2, [records[r]["predicted_derivative"] for r in TEST_RADII], 0.4, label="harmonic-head derivative")
axes[0, 2].set_xticks(positions, [f"{r:.2f}" for r in TEST_RADII])
axes[0, 2].set(xlabel="unseen rod radius/r0", ylabel="d beta / d radius", title="Analytic geometry derivative")
axes[0, 2].legend(fontsize=8)

limit = np.nanmax(np.abs(difference))
image = axes[1, 0].imshow(difference, origin="lower", extent=[-0.8, 0.8, -0.8, 0.8], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Unseen R=.70: harmonic head minus exact")
fig.colorbar(image, ax=axes[1, 0])

axes[1, 1].semilogy(ORDERS, truncation_v, marker="o", label="V RMSE")
axes[1, 1].semilogy(ORDERS, truncation_e, marker="s", label="E RMSE")
axes[1, 1].set(xlabel="highest retained harmonic order", ylabel="normalized RMSE", title="How many exact modes does the disc need?")
axes[1, 1].legend(fontsize=8)

names = ["moment PIG", "global PIXEL", "central PIXEL + E", "harmonic head"]
v_errors = [0.00112836, 0.003502, 0.002421, max(row["v"] for row in records.values())]
e_errors = [0.00586167, 0.017782, 0.038803, max(row["e"] for row in records.values())]
x = np.arange(len(names))
axes[1, 2].bar(x - 0.2, v_errors, 0.4, label="worst unseen V")
axes[1, 2].bar(x + 0.2, e_errors, 0.4, label="worst unseen E")
axes[1, 2].set_xticks(x, [name.replace(" ", "\n") for name in names], fontsize=8)
axes[1, 2].set(yscale="log", ylabel="normalized RMSE", title="Representation comparison on the same family")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 30 — a geometry-conditioned exact-harmonic objective head", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: the harmonic head is valid only inside a declared source-free disc. It should protect center multipoles and trajectory fields, while PIG/PIXEL or the exact boundary solver handles the global domain and electrode conditions.")
print("saved", OUTPUT)
