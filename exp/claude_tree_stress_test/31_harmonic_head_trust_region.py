import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/31_harmonic_head_trust_region.png"
EVALUATION_RADII = np.linspace(0.40, 1.20, 81)
ORDERS = np.arange(2, 39, 4)
OUTER_RADIUS = 5.0
COUNT = 56
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


theta = 2.0 * np.pi * np.arange(2048) / 2048
ring = RING_RADIUS * np.exp(1j * theta)


def reference(radius, z):
    sources, coefficients = solve_circle(radius)
    delta = z[:, None] - sources[None, :]
    potential = np.log(np.abs(delta)) @ coefficients[:-1] + coefficients[-1]
    complex_gradient = np.sum(coefficients[None, :-1] / np.conj(delta), axis=1)
    return potential, np.column_stack([-complex_gradient.real, -complex_gradient.imag])


def exact_coefficients(radius):
    values = reference(radius, ring)[0]
    return np.array([2.0 * np.mean(values * np.cos(order * theta)) / RING_RADIUS**order for order in ORDERS])


exact = np.array([exact_coefficients(radius) for radius in EVALUATION_RADII])


def lobatto_nodes(count, lower, upper):
    unit = np.cos(np.pi * np.arange(count) / (count - 1))[::-1]
    return 0.5 * (lower + upper) + 0.5 * (upper - lower) * unit


def make_response(training_radii, degree, lower, upper):
    training = np.array([exact_coefficients(radius) for radius in training_radii])
    t = 2.0 * (training_radii - lower) / (upper - lower) - 1.0
    fits = np.column_stack([np.polynomial.chebyshev.chebfit(t, training[:, index], degree) for index in range(len(ORDERS))])

    def evaluate(radius):
        coordinate = 2.0 * (radius - lower) / (upper - lower) - 1.0
        return np.array([np.polynomial.chebyshev.chebval(coordinate, fits[:, index]) for index in range(len(ORDERS))])

    return evaluate


model_specs = {
    "local 5 snapshots": (np.linspace(0.60, 0.76, 5), 4, 0.60, 0.76),
    "global 5 snapshots": (lobatto_nodes(5, 0.40, 1.20), 4, 0.40, 1.20),
    "global 9 snapshots": (lobatto_nodes(9, 0.40, 1.20), 8, 0.40, 1.20),
}
models = {name: make_response(*spec) for name, spec in model_specs.items()}
predictions = {name: np.array([model(radius) for radius in EVALUATION_RADII]) for name, model in models.items()}

exact_beta = exact[:, 1] / exact[:, 0]
exact_eta = 2.0 * exact[:, 0]
beta_errors = {name: np.abs(values[:, 1] / values[:, 0] - exact_beta) for name, values in predictions.items()}
eta_errors = {name: np.abs(2.0 * values[:, 0] - exact_eta) for name, values in predictions.items()}
exact_derivative = np.gradient(exact_beta, EVALUATION_RADII)
derivative_errors = {name: np.abs(np.gradient(values[:, 1] / values[:, 0], EVALUATION_RADII) - exact_derivative) for name, values in predictions.items()}


def harmonic_field(z, coefficients):
    potential = np.zeros(len(z))
    complex_field = np.zeros(len(z), dtype=complex)
    for order, coefficient in zip(ORDERS, coefficients):
        potential += coefficient * np.real(z**order)
        complex_field -= order * coefficient * np.conj(z ** (order - 1))
    return potential, np.column_stack([complex_field.real, complex_field.imag])


rng = np.random.default_rng(3101)
angles = rng.uniform(0.0, 2.0 * np.pi, 2500)
radii = 0.80 * np.sqrt(rng.uniform(0.0, 1.0, 2500))
probe = radii * np.exp(1j * angles)
field_radii = [0.45, 0.55, 0.68, 0.85, 1.00, 1.15]
field_errors = {name: [] for name in models}
for radius in field_radii:
    reference_v, reference_e = reference(radius, probe)
    for name, model in models.items():
        predicted_v, predicted_e = harmonic_field(probe, model(radius))
        field_errors[name].append((
            np.sqrt(np.mean(((predicted_v - reference_v) / np.max(np.abs(reference_v))) ** 2)),
            np.sqrt(np.mean((np.linalg.norm(predicted_e - reference_e, axis=1) / np.max(np.linalg.norm(reference_e, axis=1))) ** 2)),
        ))

print("HYPOTHESIS: the five-snapshot harmonic head is a local interpolant, not a global surrogate; Chebyshev-spaced exact solves should extend its trusted geometry interval predictably.")
for name in models:
    local = (EVALUATION_RADII >= 0.60) & (EVALUATION_RADII <= 0.76)
    print(name, "max beta error local/global", np.max(beta_errors[name][local]), np.max(beta_errors[name]), "max derivative error local/global", np.max(derivative_errors[name][local]), np.max(derivative_errors[name]))
    print(name, "field errors", field_errors[name])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
colors = {"local 5 snapshots": "tab:red", "global 5 snapshots": "tab:orange", "global 9 snapshots": "tab:blue"}
axes[0, 0].plot(EVALUATION_RADII, exact_beta, color="black", label="exact")
for name, values in predictions.items():
    axes[0, 0].plot(EVALUATION_RADII, values[:, 1] / values[:, 0], color=colors[name], linestyle="--", label=name)
axes[0, 0].axvspan(0.60, 0.76, color="grey", alpha=0.15, label="local training hull")
axes[0, 0].set(xlabel="rod radius/r0", ylabel="beta=a6/a2", title="Interpolation and extrapolation")
axes[0, 0].legend(fontsize=7)

for name, error in beta_errors.items():
    axes[0, 1].semilogy(EVALUATION_RADII, np.maximum(error, 1e-16), color=colors[name], label=name)
axes[0, 1].axvspan(0.60, 0.76, color="grey", alpha=0.15)
axes[0, 1].set(xlabel="rod radius/r0", ylabel="absolute beta error", title="A local head must not extrapolate silently")
axes[0, 1].legend(fontsize=7)

for name, error in derivative_errors.items():
    axes[0, 2].semilogy(EVALUATION_RADII, np.maximum(error, 1e-16), color=colors[name], label=name)
axes[0, 2].axvspan(0.60, 0.76, color="grey", alpha=0.15)
axes[0, 2].set(xlabel="rod radius/r0", ylabel="absolute derivative error", title="Geometry-gradient trust region")
axes[0, 2].legend(fontsize=7)

for offset, (name, errors) in zip([-0.25, 0.0, 0.25], field_errors.items()):
    axes[1, 0].bar(np.arange(len(field_radii)) + offset, [value[0] for value in errors], 0.25, color=colors[name], label=name)
axes[1, 0].set_xticks(np.arange(len(field_radii)), [f"{radius:.2f}" for radius in field_radii])
axes[1, 0].set(yscale="log", xlabel="rod radius/r0", ylabel="V RMSE", title="Central potential across the broad family")
axes[1, 0].legend(fontsize=7)

for offset, (name, errors) in zip([-0.25, 0.0, 0.25], field_errors.items()):
    axes[1, 1].bar(np.arange(len(field_radii)) + offset, [value[1] for value in errors], 0.25, color=colors[name], label=name)
axes[1, 1].set_xticks(np.arange(len(field_radii)), [f"{radius:.2f}" for radius in field_radii])
axes[1, 1].set(yscale="log", xlabel="rod radius/r0", ylabel="E RMSE", title="Central electric field across the broad family")
axes[1, 1].legend(fontsize=7)

for name, spec in model_specs.items():
    axes[1, 2].scatter(spec[0], np.full(len(spec[0]), list(model_specs).index(name)), color=colors[name], label=name)
axes[1, 2].set_yticks(np.arange(len(model_specs)), list(model_specs))
axes[1, 2].set(xlim=[0.38, 1.22], xlabel="radius of exact inner solve", title="Snapshot placement defines the trusted hull")
axes[1, 2].legend(fontsize=7)

fig.suptitle("Claude-tree stress test 31 — the exact-harmonic head needs an explicit geometry trust region", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: use the harmonic head only inside the convex hull of validated geometry snapshots. Outside it, request an exact solve, add that geometry, and refit; increasing polynomial degree without expanding evidence is not a safe substitute.")
print("saved", OUTPUT)
