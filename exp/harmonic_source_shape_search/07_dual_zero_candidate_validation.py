import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, root
from scipy.spatial.distance import cdist


OUTPUT = "exp/harmonic_source_shape_search/results/07_dual_zero_candidate_validation.png"
COUNTS = np.array([16, 24, 32, 48])
FRACTIONS = np.array([0.45, 0.58, 0.70])
MEASUREMENT_RADIUS = 0.5


def local_support(radius, c4, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    p = radius * (1.0 + c4 * np.cos(4.0 * theta))
    dp = -4.0 * radius * c4 * np.sin(4.0 * theta)
    return np.column_stack([p * np.cos(theta) - dp * np.sin(theta), p * np.sin(theta) + dp * np.cos(theta)])


def solve(radius, c4, count, fraction):
    boundary_count = 3 * count
    local_boundary = local_support(radius, c4, boundary_count)
    local_sources = fraction * local_support(radius, c4, count, 0.013)
    distance = 1.0 + radius * (1.0 + c4)
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    sources, boundary, values = [], [], []
    for radial, tangential, voltage in frames:
        rotation = np.column_stack([radial, tangential])
        center = distance * radial
        sources.append(center + local_sources @ rotation.T)
        boundary.append(center + local_boundary @ rotation.T)
        values.append(np.full(boundary_count, voltage))
    source_theta = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    boundary_theta = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    sources.append(6.5 * np.column_stack([np.cos(source_theta), np.sin(source_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(boundary_theta), np.sin(boundary_theta)]))
    values.append(np.zeros(12 * count))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    matrix = np.column_stack([np.log(np.linalg.norm(boundary[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients, distance


def evaluate(points, sources, coefficients):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = 0.5 * np.log(radius2) @ coefficients[:-1] + coefficients[-1]
    field = -np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, field


theta = 2.0 * np.pi * np.arange(720) / 720
probe = MEASUREMENT_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])


def multipoles(radius, c4, count, fraction):
    sources, coefficients, _ = solve(radius, c4, count, fraction)
    potential = evaluate(probe, sources, coefficients)[0]
    values = []
    for order in [2, 6, 10, 14]:
        values.append(2.0 * np.mean(potential * np.cos(order * theta)) / MEASUREMENT_RADIUS**order)
    a2, a6, a10, a14 = values
    return 2.0 * a2, (a6 / a2) * MEASUREMENT_RADIUS**4, (a10 / a2) * MEASUREMENT_RADIUS**8, (a14 / a2) * MEASUREMENT_RADIUS**12


def unseen_boundary_error(radius, c4, count, fraction):
    sources, coefficients, distance = solve(radius, c4, count, fraction)
    local = local_support(radius, c4, 211, 0.0041)
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    points, values = [], []
    for radial, tangential, voltage in frames:
        points.append(distance * radial + local @ np.column_stack([radial, tangential]).T)
        values.append(np.full(211, voltage))
    outer_theta = 2.0 * np.pi * np.arange(509) / 509 + 0.0023
    points.append(5.0 * np.column_stack([np.cos(outer_theta), np.sin(outer_theta)]))
    values.append(np.zeros(509))
    points = np.vstack(points)
    values = np.concatenate(values)
    prediction = evaluate(points, sources, coefficients)[0]
    return np.sqrt(np.mean((prediction - values) ** 2))


def field_nonlinearity(radius, c4, count, fraction, probe_radius):
    sources, coefficients, _ = solve(radius, c4, count, fraction)
    eta, _, _, _ = multipoles(radius, c4, count, fraction)
    points = probe_radius * np.column_stack([np.cos(theta), np.sin(theta)])
    _, field = evaluate(points, sources, coefficients)
    a2 = 0.5 * eta
    ideal = np.column_stack([-2.0 * a2 * points[:, 0], 2.0 * a2 * points[:, 1]])
    return np.sqrt(np.mean(np.sum((field - ideal) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(ideal**2, axis=1)))


print("HYPOTHESIS: the support-mode candidate genuinely removes q6 and q10 across converged source discretizations, reducing finite-amplitude field nonlinearity by much more than its roughly 0.36% loss of eta.")
shape = (len(COUNTS), len(FRACTIONS))
candidate_radius = np.zeros(shape)
candidate_c4 = np.zeros(shape)
eta_ratio = np.zeros(shape)
q14 = np.zeros(shape)
field_ratio = np.zeros(shape)
boundary_error = np.zeros(shape)
circle_root = np.zeros(shape)

for row, count in enumerate(COUNTS):
    for column, fraction in enumerate(FRACTIONS):
        solution = root(
            lambda variables: np.array(multipoles(variables[0], variables[1], int(count), float(fraction))[1:3]),
            np.array([1.693, 0.0293]),
            method="hybr",
            options={"xtol": 1e-9},
        ).x
        radius, c4 = solution
        candidate_radius[row, column] = radius
        candidate_c4[row, column] = c4
        candidate_eta, candidate_q6, candidate_q10, q14[row, column] = multipoles(radius, c4, int(count), float(fraction))
        circle_radius = brentq(lambda value: multipoles(value, 0.0, int(count), float(fraction))[1], 1.0, 1.3, xtol=1e-10)
        circle_root[row, column] = circle_radius
        circle_eta = multipoles(circle_radius, 0.0, int(count), float(fraction))[0]
        eta_ratio[row, column] = candidate_eta / circle_eta
        field_ratio[row, column] = field_nonlinearity(radius, c4, int(count), float(fraction), 0.9) / field_nonlinearity(circle_radius, 0.0, int(count), float(fraction), 0.9)
        boundary_error[row, column] = unseen_boundary_error(radius, c4, int(count), float(fraction))

trusted = (np.abs(circle_root - 1.14521) < 2e-3) & (boundary_error < 1e-5)
print("trusted settings:", int(np.sum(trusted)), "of", trusted.size)
print(f"trusted R range={np.min(candidate_radius[trusted]):.8f} to {np.max(candidate_radius[trusted]):.8f}")
print(f"trusted c4 range={np.min(candidate_c4[trusted]):.8f} to {np.max(candidate_c4[trusted]):.8f}")
print(f"trusted eta ratio={np.min(eta_ratio[trusted]):.8f} to {np.max(eta_ratio[trusted]):.8f}")
print(f"trusted field-nonlinearity ratio at 0.9 r0={np.min(field_ratio[trusted]):.4f} to {np.max(field_ratio[trusted]):.4f}")
print(f"trusted physical q14 range={np.min(q14[trusted]):+.3e} to {np.max(q14[trusted]):+.3e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [FRACTIONS[0], FRACTIONS[-1], COUNTS[0], COUNTS[-1]]
panels = [
    (candidate_radius, "viridis", "dual-zero support radius R/r0"),
    (candidate_c4, "viridis", "dual-zero support coefficient c4"),
    (eta_ratio, "RdYlGn", "eta: dual-zero / circular"),
    (field_ratio, "magma_r", "E nonlinearity at 0.9r0: dual / circle"),
    (np.log10(boundary_error), "magma_r", "log10 unseen boundary RMSE"),
]
for ax, (values, cmap, title) in zip(axes.ravel()[:5], panels):
    image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.set(xlabel="source radius / electrode radius", ylabel="sources per electrode", title=title)
    fig.colorbar(image, ax=ax, shrink=0.82)

representative_count = 48
representative_fraction = 0.58
row = np.where(COUNTS == representative_count)[0][0]
column = np.where(FRACTIONS == representative_fraction)[0][0]
dual_r = candidate_radius[row, column]
dual_c = candidate_c4[row, column]
circle_r = circle_root[row, column]
amplitudes = np.linspace(0.1, 0.95, 35)
circle_curve = np.array([field_nonlinearity(circle_r, 0.0, representative_count, representative_fraction, value) for value in amplitudes])
dual_curve = np.array([field_nonlinearity(dual_r, dual_c, representative_count, representative_fraction, value) for value in amplitudes])
axes[1, 2].semilogy(amplitudes, circle_curve, color="black", label="circular q6=0")
axes[1, 2].semilogy(amplitudes, dual_curve, color="tab:blue", label="support q6=q10=0")
axes[1, 2].set(xlabel="ion amplitude / r0", ylabel="relative E nonlinearity", title="Physical payoff after two multipoles are removed")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.25)
fig.suptitle("Plot 07 — resolution and physical validation of the dual-zero electrode", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

local = local_support(dual_r, dual_c, 720)
distance = 1.0 + dual_r * (1.0 + dual_c)
east = np.array([distance, 0.0]) + local
north = np.array([0.0, distance]) + local @ np.array([[0.0, 1.0], [-1.0, 0.0]]).T
minimum_gap = np.min(cdist(east, north))
outer_clearance = 5.0 - np.max(np.linalg.norm(east, axis=1))
minimum_curvature = dual_r * (1.0 - 15.0 * abs(dual_c))
print(f"representative geometry: R={dual_r:.8f}, c4={dual_c:.8f}, minimum adjacent gap={minimum_gap:.6f} r0, outer clearance={outer_clearance:.6f} r0, minimum curvature radius={minimum_curvature:.6f} r0")
print("CONCLUSION: inspect the resolution plateau and amplitude curve; accept the candidate only as a Pareto improvement in field linearity, not as a stronger quadrupole.")
print("saved", OUTPUT)
