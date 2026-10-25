import matplotlib.pyplot as plt
import numpy as np

from spinn.geometry.four_rod import FourRodTrap


OUTPUT = "exp/harmonic_source_shape_search/results/01_source_conditioning.png"
ROD_RADIUS = 1.0
SOURCE_COUNTS = np.array([8, 16, 32, 48])
SOURCE_FRACTIONS = np.linspace(0.25, 0.90, 18)
MEASUREMENT_RADIUS = 0.5


def circle(cx, cy, radius, count, phase=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


def assemble(trap, source_count, source_fraction, boundary_count=96):
    sources, boundary, values = [], [], []
    for electrode in trap.electrodes():
        sources.append(circle(electrode.cx, electrode.cy, source_fraction * electrode.radius, source_count, 0.013))
        boundary.append(circle(electrode.cx, electrode.cy, electrode.radius, boundary_count))
        values.append(np.full(boundary_count, trap.electrode_voltage(electrode)))
    sources.append(circle(0.0, 0.0, 1.30 * trap.outer_radius, 4 * source_count, 0.007))
    boundary.append(circle(0.0, 0.0, trap.outer_radius, 4 * boundary_count))
    values.append(np.zeros(4 * boundary_count))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
    target = np.concatenate(values)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=1e-12)
    return sources, coefficients, np.linalg.cond(matrix), np.linalg.norm(coefficients[:-1])


def evaluate(points, sources, coefficients):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = 0.5 * np.log(radius2) @ coefficients[:-1] + coefficients[-1]
    gradient = np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, -gradient


def multipoles(sources, coefficients):
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    points = MEASUREMENT_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])
    potential = evaluate(points, sources, coefficients)[0]
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / MEASUREMENT_RADIUS**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / MEASUREMENT_RADIUS**6
    return a2, (a6 / a2) * MEASUREMENT_RADIUS**4


def held_boundary(trap):
    points, values = [], []
    for electrode in trap.electrodes():
        points.append(circle(electrode.cx, electrode.cy, electrode.radius, 211, 0.0037))
        values.append(np.full(211, trap.electrode_voltage(electrode)))
    points.append(circle(0.0, 0.0, trap.outer_radius, 509, 0.0021))
    values.append(np.zeros(509))
    return np.vstack(points), np.concatenate(values)


print("HYPOTHESIS: source placement has an accuracy-conditioning window; moving logarithmic sources deeper into metal improves conditioning but eventually loses the boundary modes needed for q6, while moving them toward the surface improves approximation at the cost of unstable coefficients.")
trap = FourRodTrap(rod_radius=ROD_RADIUS)
held_points, held_values = held_boundary(trap)

axis = np.linspace(-0.75, 0.75, 51)
gx, gy = np.meshgrid(axis, axis)
probe_all = np.column_stack([gx.ravel(), gy.ravel()])
probe = probe_all[np.hypot(probe_all[:, 0], probe_all[:, 1]) <= 0.75]

reference_sources, reference_coefficients, _, _ = assemble(trap, 80, 0.62, 180)
reference_potential, reference_field = evaluate(probe, reference_sources, reference_coefficients)
reference_a2, reference_q6 = multipoles(reference_sources, reference_coefficients)
potential_scale = np.max(np.abs(reference_potential))
field_scale = np.max(np.linalg.norm(reference_field, axis=1))

shape = (len(SOURCE_COUNTS), len(SOURCE_FRACTIONS))
condition = np.zeros(shape)
coefficient_norm = np.zeros(shape)
boundary_rmse = np.zeros(shape)
potential_rmse = np.zeros(shape)
field_rmse = np.zeros(shape)
q6_error = np.zeros(shape)
q6_value = np.zeros(shape)

for row, count in enumerate(SOURCE_COUNTS):
    for column, fraction in enumerate(SOURCE_FRACTIONS):
        sources, coefficients, condition[row, column], coefficient_norm[row, column] = assemble(trap, int(count), float(fraction))
        held_prediction = evaluate(held_points, sources, coefficients)[0]
        potential, field = evaluate(probe, sources, coefficients)
        _, q6_value[row, column] = multipoles(sources, coefficients)
        boundary_rmse[row, column] = np.sqrt(np.mean((held_prediction - held_values) ** 2))
        potential_rmse[row, column] = np.sqrt(np.mean((potential - reference_potential) ** 2)) / potential_scale
        field_rmse[row, column] = np.sqrt(np.mean(np.sum((field - reference_field) ** 2, axis=1))) / field_scale
        q6_error[row, column] = abs(q6_value[row, column] - reference_q6)

best = np.unravel_index(np.argmin(q6_error + field_rmse), shape)
print(f"reference a2={reference_a2:+.8f}, physical q6={reference_q6:+.8e}")
print(f"best joint q6+field point: count={SOURCE_COUNTS[best[0]]}, fraction={SOURCE_FRACTIONS[best[1]]:.3f}, cond={condition[best]:.3e}, boundary RMSE={boundary_rmse[best]:.3e}, field RMSE={field_rmse[best]:.3e}, q6 error={q6_error[best]:.3e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [SOURCE_FRACTIONS[0], SOURCE_FRACTIONS[-1], SOURCE_COUNTS[0], SOURCE_COUNTS[-1]]
panels = [
    (np.log10(condition), "viridis", "log10 boundary matrix condition number"),
    (np.log10(boundary_rmse), "magma_r", "log10 unseen boundary RMSE"),
    (np.log10(field_rmse), "magma_r", "log10 independent E-field RMSE"),
    (np.log10(q6_error), "magma_r", "log10 absolute physical q6 error"),
    (np.log10(coefficient_norm), "viridis", "log10 source-charge coefficient norm"),
]
for ax, (values, cmap, title) in zip(axes.ravel()[:5], panels):
    image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.scatter([SOURCE_FRACTIONS[best[1]]], [SOURCE_COUNTS[best[0]]], marker="x", color="cyan", s=70)
    ax.set(xlabel="electrode source radius / electrode radius", ylabel="sources per electrode", title=title)
    fig.colorbar(image, ax=ax, shrink=0.83)

for row, count in enumerate(SOURCE_COUNTS):
    axes[1, 2].plot(SOURCE_FRACTIONS, q6_value[row], marker="o", markersize=3, label=f"{count} / electrode")
axes[1, 2].axhline(reference_q6, color="black", linestyle="--", label="high-resolution reference")
axes[1, 2].set(xlabel="electrode source radius / electrode radius", ylabel="physical q6 at 0.5 r0", title="The geometry score is more sensitive than boundary RMSE")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.25)
fig.suptitle("Plot 01 — logarithmic-source placement before making the solver differentiable", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

good = (field_rmse < 1e-5) & (q6_error < 1e-6)
print("points with E RMSE < 1e-5 and q6 error < 1e-6:", int(np.sum(good)), "of", good.size)
print("CONCLUSION: inspect the heatmaps before selecting the source parameterization; the acceptable region must satisfy field and q6 accuracy without pushing the linear system beyond stable precision.")
print("saved", OUTPUT)
