import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/11_manufacturing_tolerance.png"
SYMMETRY_OUTPUT = "exp/claude_tree_stress_test/results/12_symmetry_manufacturing_tolerance.png"
RADIUS = 1.1451090300780968
DISTANCE = 1.0 + RADIUS
OUTER_RADIUS = 5.0
COUNT = 58
PROBE_RADIUS = 0.5
KAPPA9 = -0.0012213401059591876
DELTA13 = -0.0003875957659400217
P9 = np.arange(9, 19)
W9 = np.array([0.14526537, -0.10620197, 0.28421932, -0.42300312, -0.77440244, 14.74465186, -52.90593862, 91.26916034, -77.73953184, 26.50578109])
W9 /= np.sum(W9)
P13 = np.arange(16, 29)
W13 = np.array([6.78065474, -42.50938914, 128.37796771, -226.14901113, 228.63499372, -81.83172425, -89.9305892, 123.78893861, -34.42561199, -82.46601008, 164.01436013, -139.11150475, 45.82692563])
W13 /= np.sum(W13)


def basis(z, order, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    return z**order * (z**4 - 1.0) / (z**4 - DISTANCE**4) ** power / leading


def mapping(z):
    return z + KAPPA9 * sum(weight * basis(z, 9, power) for weight, power in zip(W9, P9)) + DELTA13 * sum(weight * basis(z, 13, power) for weight, power in zip(W13, P13))


centers = DISTANCE * np.array([1.0, -1.0, 1.0j, -1.0j])
voltages = np.array([0.5, 0.5, -0.5, -0.5])
local_theta = 2.0 * np.pi * np.arange(3 * COUNT) / (3 * COUNT)


def build_system(use_candidate):
    boundary_segments = []
    source_segments = []
    normal_segments = []
    for center in centers:
        reference_boundary = center + RADIUS * np.exp(1j * local_theta)
        boundary = mapping(reference_boundary) if use_candidate else reference_boundary
        tangent = (np.roll(boundary, -1) - np.roll(boundary, 1)) / 2.0
        normal = -1.0j * tangent / np.abs(tangent)
        source_theta = 2.0 * np.pi * np.arange(COUNT) / COUNT + 0.011
        reference_source_boundary = center + RADIUS * np.exp(1j * source_theta)
        physical_source_boundary = mapping(reference_source_boundary) if use_candidate else reference_source_boundary
        sources = center + 0.62 * (physical_source_boundary - center)
        boundary_segments.append(boundary)
        source_segments.append(sources)
        normal_segments.append(normal)
    electrode_boundary = np.concatenate(boundary_segments)
    electrode_normals = np.concatenate(normal_segments)
    sources = np.concatenate(source_segments)
    values = np.concatenate([np.full(len(local_theta), voltage) for voltage in voltages])
    outer_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    outer_source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([electrode_boundary, OUTER_RADIUS * np.exp(1j * outer_theta)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * outer_source_theta)])
    values = np.concatenate([values, np.zeros(len(outer_theta))])
    A = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(A, values, rcond=1e-12)
    return A, sources, coefficients, boundary, electrode_boundary, electrode_normals


def field_on_boundary(boundary, sources, coefficients):
    delta = boundary[:, None] - sources[None, :]
    complex_gradient = np.sum(coefficients[:-1][None, :] / delta, axis=1)
    return complex_gradient


def response_model(use_candidate):
    A, sources, coefficients, boundary, electrode_boundary, electrode_normals = build_system(use_candidate)
    electrode_count = len(electrode_boundary)
    gradient = field_on_boundary(electrode_boundary, sources, coefficients)
    normal_derivative = np.real(gradient * electrode_normals)
    modes = []
    labels = []
    for rod in range(4):
        segment = slice(rod * len(local_theta), (rod + 1) * len(local_theta))
        for order in range(13):
            h = np.zeros(electrode_count)
            h[segment] = np.cos(order * local_theta)
            modes.append(h)
            labels.append(f"rod {rod + 1} cos {order}")
        for order in range(1, 13):
            h = np.zeros(electrode_count)
            h[segment] = np.sin(order * local_theta)
            modes.append(h)
            labels.append(f"rod {rod + 1} sin {order}")
    H = np.column_stack(modes)
    induced = np.zeros((len(boundary), H.shape[1]))
    induced[:electrode_count] = H * normal_derivative[:, None]
    response_coefficients, *_ = np.linalg.lstsq(A, induced, rcond=1e-12)
    theta = 2.0 * np.pi * np.arange(1440) / 1440
    probe = PROBE_RADIUS * np.exp(1j * theta)
    P = np.column_stack([np.log(np.abs(probe[:, None] - sources[None, :])), np.ones(len(probe))])
    base_values = P @ coefficients
    response_values = P @ response_coefficients
    base_a2 = 2.0 * np.mean(base_values * np.cos(2.0 * theta)) / PROBE_RADIUS**2
    orders = np.array([1, 6, 10, 14])
    complex_response = np.array([
        2.0 * np.mean(response_values * np.exp(-1j * order * theta)[:, None], axis=0) / PROBE_RADIUS**order
        for order in orders
    ])
    return H, complex_response, base_a2, labels, sources, response_coefficients


models = {"circle": response_model(False), "conformal candidate": response_model(True)}
rng = np.random.default_rng(18)
draws = rng.normal(size=(1600, models["circle"][0].shape[1]))
tolerances = np.logspace(-6, -2, 9)
statistics = {}
unit_samples = {}
for name, (H, complex_response, base_a2, labels, sources, response_coefficients) in models.items():
    h = H @ draws.T
    rms = np.sqrt(np.mean(h**2, axis=0))
    normalized_draws = draws / rms[:, None]
    response = complex_response @ normalized_draws.T
    unit = np.vstack([
        np.abs(response[0]) / (2.0 * abs(base_a2)),
        np.abs(response[1] / base_a2) * PROBE_RADIUS**4,
        np.abs(response[2] / base_a2) * PROBE_RADIUS**8,
        np.abs(response[3] / base_a2) * PROBE_RADIUS**12,
    ])
    unit_samples[name] = unit
    statistics[name] = np.array([[np.percentile(metric * tolerance, [5, 50, 95]) for tolerance in tolerances] for metric in unit])

print("HYPOTHESIS: exact q10=q14 cancellation is useful only if realistic independent rod errors do not immediately regenerate those terms or move the saddle by an unacceptable fraction of r0.")
circle_q10 = 9.54171958e-6
for name, unit in unit_samples.items():
    threshold = circle_q10 / unit[2]
    print(name, "tolerance where induced q10 equals circular baseline, 5/50/95 percent", np.percentile(threshold, [5, 50, 95]))
    print(name, "at 1e-3 r0 tolerance median saddle shift,q6,q10,q14", np.median(unit, axis=1) * 1e-3)

metric_labels = ["saddle shift/r0", "induced |q6|", "induced |q10|", "induced |q14|"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for metric_index, ax in enumerate(axes.ravel()[:4]):
    for name, color in [("circle", "black"), ("conformal candidate", "tab:blue")]:
        stat = statistics[name][metric_index]
        ax.loglog(tolerances, stat[:, 1], color=color, marker="o", label=name)
        ax.fill_between(tolerances, stat[:, 0], stat[:, 2], color=color, alpha=0.15)
    if metric_index == 2:
        ax.axhline(circle_q10, color="tab:red", linestyle="--", label="circle's original |q10|")
    ax.set(xlabel="RMS normal manufacturing error/r0", ylabel=metric_labels[metric_index], title=metric_labels[metric_index])
    ax.legend(fontsize=8)

candidate_H, candidate_response, candidate_a2, labels, sources, response_coefficients = models["conformal candidate"]
q10_sensitivity = np.abs(candidate_response[2] / candidate_a2) * PROBE_RADIUS**8
top = np.argsort(q10_sensitivity)[-12:]
axes[1, 1].barh(np.arange(len(top)), q10_sensitivity[top])
axes[1, 1].set_yticks(np.arange(len(top)), [labels[index] for index in top], fontsize=7)
axes[1, 1].set(xlabel="induced |q10| per unit mode amplitude", title="Most dangerous fabrication modes")

example_coefficients = draws[0]
example_h = candidate_H @ example_coefficients
example_coefficients *= 1e-3 / np.sqrt(np.mean(example_h**2))
example_response_coefficients = response_coefficients @ example_coefficients
grid_axis = np.linspace(-0.75, 0.75, 151)
xx, yy = np.meshgrid(grid_axis, grid_axis)
grid = (xx + 1j * yy).ravel()
Pgrid = np.column_stack([np.log(np.abs(grid[:, None] - sources[None, :])), np.ones(len(grid))])
response_grid = (Pgrid @ example_response_coefficients).reshape(xx.shape)
inside = xx**2 + yy**2 <= 0.7**2
response_grid[~inside] = np.nan
limit = np.nanmax(np.abs(response_grid))
image = axes[1, 2].imshow(response_grid, origin="lower", extent=[-0.75, 0.75, -0.75, 0.75], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Field error from one 0.001 r0 fabrication draw")
fig.colorbar(image, ax=axes[1, 2])

fig.suptitle("Claude-tree stress test 11 — manufacturing robustness of exact multipole cancellation", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: the percentile curves turn exact mathematical zeros into fabrication tolerances. Compare the candidate with the circle at equal physical boundary RMS; the most dangerous Fourier modes identify what should be penalized or measured during manufacturing.")
print("saved", OUTPUT)

print("HYPOTHESIS: cloning one imperfect rod profile by exact 90-degree rotations will remove saddle displacement and all harmonics forbidden by quadrupole symmetry, leaving only the allowed q6,q10,q14 sequence.")
shared_draws = rng.normal(size=(1600, 25))
correlated_blocks = []
rod_phases = [0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi]
for phase in rod_phases:
    block = np.zeros_like(shared_draws)
    block[:, 0] = shared_draws[:, 0]
    for order in range(1, 13):
        cosine = shared_draws[:, order]
        sine = shared_draws[:, 12 + order]
        block[:, order] = cosine * np.cos(order * phase) - sine * np.sin(order * phase)
        block[:, 12 + order] = cosine * np.sin(order * phase) + sine * np.cos(order * phase)
    correlated_blocks.append(block)
correlated_draws = np.hstack(correlated_blocks)
candidate_H, candidate_response, candidate_a2, labels, sources, response_coefficients = models["conformal candidate"]


def normalized_metrics(coefficient_draws):
    h = candidate_H @ coefficient_draws.T
    rms = np.sqrt(np.mean(h**2, axis=0))
    normalized = coefficient_draws / rms[:, None]
    response = candidate_response @ normalized.T
    return np.vstack([
        np.abs(response[0]) / (2.0 * abs(candidate_a2)),
        np.abs(response[1] / candidate_a2) * PROBE_RADIUS**4,
        np.abs(response[2] / candidate_a2) * PROBE_RADIUS**8,
        np.abs(response[3] / candidate_a2) * PROBE_RADIUS**12,
    ])


independent_unit = normalized_metrics(draws)
correlated_unit = normalized_metrics(correlated_draws)

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for metric_index, ax in enumerate(axes2.ravel()[:4]):
    for values, label, color in [(independent_unit, "independent rods", "tab:red"), (correlated_unit, "one profile cloned by rotations", "tab:blue")]:
        median = np.median(values[metric_index]) * tolerances
        low = np.percentile(values[metric_index], 5) * tolerances
        high = np.percentile(values[metric_index], 95) * tolerances
        ax.loglog(tolerances, np.maximum(median, 1e-20), marker="o", color=color, label=label)
        ax.fill_between(tolerances, np.maximum(low, 1e-20), np.maximum(high, 1e-20), color=color, alpha=0.15)
    ax.set(xlabel="RMS normal profile error/r0", ylabel=metric_labels[metric_index], title=metric_labels[metric_index])
    ax.legend(fontsize=8)

ratios = np.median(correlated_unit, axis=1) / np.maximum(np.median(independent_unit, axis=1), 1e-30)
axes2[1, 1].bar(np.arange(4), ratios)
axes2[1, 1].set_yscale("log")
axes2[1, 1].set_xticks(np.arange(4), ["shift", "q6", "q10", "q14"])
axes2[1, 1].set(ylabel="correlated / independent median", title="What exact rotational cloning suppresses")

order_response_independent = np.median(independent_unit, axis=1) * 1e-3
order_response_correlated = np.median(correlated_unit, axis=1) * 1e-3
xbar = np.arange(4)
axes2[1, 2].bar(xbar - 0.18, order_response_independent, 0.36, label="independent")
axes2[1, 2].bar(xbar + 0.18, order_response_correlated, 0.36, label="cloned")
axes2[1, 2].set_yscale("log")
axes2[1, 2].set_xticks(xbar, ["shift", "q6", "q10", "q14"])
axes2[1, 2].set(ylabel="median at 0.001 r0 error", title="Same profile accuracy, different assembly symmetry")
axes2[1, 2].legend(fontsize=8)

fig2.suptitle("Claude-tree stress test 12 — symmetry as a fabrication-robustness mechanism", fontsize=14)
fig2.savefig(SYMMETRY_OUTPUT, dpi=180)
print("independent median at 1e-3", order_response_independent)
print("rotationally cloned median at 1e-3", order_response_correlated)
print("CONCLUSION: exact rotational cloning suppresses forbidden asymmetric responses but does not remove allowed D4 multipoles. This supports a symmetry toggle in PIG/shape search and, physically, matched-profile manufacturing or metrology-aware assembly.")
print("saved", SYMMETRY_OUTPUT)
