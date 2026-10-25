import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/harmonic_source_shape_search/results/13_boundary_shape_derivative.png"
COUNT = 56
FRACTION = 0.70
PROBE_RADIUS = 0.5


def local_support(radius, c4, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    p = radius * (1.0 + c4 * np.cos(4.0 * theta))
    dp = -4.0 * radius * c4 * np.sin(4.0 * theta)
    points = np.column_stack([p * np.cos(theta) - dp * np.sin(theta), p * np.sin(theta) + dp * np.cos(theta)])
    normals = np.column_stack([np.cos(theta), np.sin(theta)])
    return theta, points, normals


def geometry(radius, c4):
    theta, local_boundary, local_normals = local_support(radius, c4, 3 * COUNT)
    _, local_sources, _ = local_support(radius, c4, COUNT, 0.013)
    local_sources *= FRACTION
    distance = 1.0 + radius * (1.0 + c4)
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    sources, boundary, values, normals = [], [], [], []
    radius_motion, c4_motion = [], []
    n = np.column_stack([np.cos(theta), np.sin(theta)])
    t = np.column_stack([-np.sin(theta), np.cos(theta)])
    local_dradius = local_boundary / radius
    dp_dc4 = radius * np.cos(4.0 * theta)
    ddp_dc4 = -4.0 * radius * np.sin(4.0 * theta)
    local_dc4 = dp_dc4[:, None] * n + ddp_dc4[:, None] * t
    for radial, tangential, voltage in frames:
        rotation = np.column_stack([radial, tangential])
        center = distance * radial
        sources.append(center + local_sources @ rotation.T)
        boundary.append(center + local_boundary @ rotation.T)
        values.append(np.full(len(local_boundary), voltage))
        normal = local_normals @ rotation.T
        normals.append(normal)
        dradius = (1.0 + c4) * radial + local_dradius @ rotation.T
        dc4 = radius * radial + local_dc4 @ rotation.T
        radius_motion.append(np.sum(dradius * normal, axis=1))
        c4_motion.append(np.sum(dc4 * normal, axis=1))
    outer_source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    outer_boundary_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    sources.append(6.5 * np.column_stack([np.cos(outer_source_theta), np.sin(outer_source_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(outer_boundary_theta), np.sin(outer_boundary_theta)]))
    values.append(np.zeros(12 * COUNT))
    normals.append(np.column_stack([np.cos(outer_boundary_theta), np.sin(outer_boundary_theta)]))
    radius_motion.append(np.zeros(12 * COUNT))
    c4_motion.append(np.zeros(12 * COUNT))
    return (
        np.vstack(sources), np.vstack(boundary), np.concatenate(values), np.vstack(normals),
        np.concatenate(radius_motion), np.concatenate(c4_motion), distance,
    )


def matrix(points, sources):
    return np.column_stack([np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(points))])


def evaluate(points, sources, coefficients):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = 0.5 * np.log(radius2) @ coefficients[:-1] + coefficients[-1]
    field = -np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, field


def solve(radius, c4, boundary_values=None, fixed_sources=None):
    sources, boundary, values, normals, radius_motion, c4_motion, distance = geometry(radius, c4)
    if fixed_sources is not None:
        sources = fixed_sources
    if boundary_values is not None:
        values = boundary_values
    coefficients, *_ = np.linalg.lstsq(matrix(boundary, sources), values, rcond=1e-12)
    return sources, coefficients, boundary, normals, radius_motion, c4_motion, distance


probe_theta = 2.0 * np.pi * np.arange(1440) / 1440
probe = PROBE_RADIUS * np.column_stack([np.cos(probe_theta), np.sin(probe_theta)])


def metric_from_values(values):
    amplitudes = np.array([
        2.0 * np.mean(values * np.cos(order * probe_theta)) / PROBE_RADIUS**order
        for order in [2, 6, 10]
    ])
    a2, a6, a10 = amplitudes
    return np.array([2.0 * a2, (a6 / a2) * PROBE_RADIUS**4, (a10 / a2) * PROBE_RADIUS**8])


def metrics(radius, c4):
    sources, coefficients, *_ = solve(radius, c4)
    return metric_from_values(evaluate(probe, sources, coefficients)[0])


def shape_derivative(radius, c4, parameter):
    sources, coefficients, boundary, normals, radius_motion, c4_motion, _ = solve(radius, c4)
    field = evaluate(boundary, sources, coefficients)[1]
    motion = radius_motion if parameter == "radius" else c4_motion
    induced_boundary = motion * np.sum(field * normals, axis=1)
    _, response_coefficients, *_ = solve(radius, c4, boundary_values=induced_boundary, fixed_sources=sources)
    response_values = evaluate(probe, sources, response_coefficients)[0]
    base_values = evaluate(probe, sources, coefficients)[0]
    base_metric = metric_from_values(base_values)
    response_amplitudes = np.array([
        2.0 * np.mean(response_values * np.cos(order * probe_theta)) / PROBE_RADIUS**order
        for order in [2, 6, 10]
    ])
    a2 = 0.5 * base_metric[0]
    a6 = base_metric[1] * a2 / PROBE_RADIUS**4
    a10 = base_metric[2] * a2 / PROBE_RADIUS**8
    da2, da6, da10 = response_amplitudes
    derivative = np.array([
        2.0 * da2,
        PROBE_RADIUS**4 * (da6 * a2 - a6 * da2) / a2**2,
        PROBE_RADIUS**8 * (da10 * a2 - a10 * da2) / a2**2,
    ])
    return derivative, sources, coefficients, response_coefficients, boundary, normals, motion, induced_boundary


circle_radius = brentq(lambda radius: metrics(radius, 0.0)[1], 1.0, 1.3, xtol=1e-12)
dual_radius = 1.630529854
dual_c4 = 0.026949792

print("HYPOTHESIS: the boundary condition deltaV=-h*dV/dn predicts physical geometry derivatives without a coordinate map, and tangential boundary motion contributes exactly zero because only h=motion dot normal enters.")

cases = [
    ("circle radius", circle_radius, 0.0, "radius", 2.0e-4),
    ("dual radius", dual_radius, dual_c4, "radius", 2.0e-4),
    ("dual c4", dual_radius, dual_c4, "c4", 2.0e-5),
]
comparisons = []
for name, radius, c4, parameter, step in cases:
    derivative, *_ = shape_derivative(radius, c4, parameter)
    if parameter == "radius":
        finite = (metrics(radius + step, c4) - metrics(radius - step, c4)) / (2.0 * step)
    else:
        finite = (metrics(radius, c4 + step) - metrics(radius, c4 - step)) / (2.0 * step)
    comparisons.append((name, derivative, finite))
    print(name, "boundary derivative", derivative, "finite difference", finite, "relative norm error", np.linalg.norm(derivative - finite) / np.linalg.norm(finite))

derivative, sources, coefficients, response_coefficients, boundary, normals, motion, induced_boundary = shape_derivative(circle_radius, 0.0, "radius")
step = 0.01
plus = solve(circle_radius + step, 0.0)
minus = solve(circle_radius - step, 0.0)

line_radii = circle_radius + np.linspace(-0.035, 0.035, 29)
exact_q6 = np.array([metrics(radius, 0.0)[1] for radius in line_radii])
linear_q6 = metrics(circle_radius, 0.0)[1] + derivative[1] * (line_radii - circle_radius)

grid_axis = np.linspace(-1.0, 1.0, 181)
xx, yy = np.meshgrid(grid_axis, grid_axis)
points = np.column_stack([xx.ravel(), yy.ravel()])
predicted_response = evaluate(points, sources, response_coefficients)[0].reshape(xx.shape)
plus_values = evaluate(points, plus[0], plus[1])[0]
minus_values = evaluate(points, minus[0], minus[1])[0]
finite_response = ((plus_values - minus_values) / (2.0 * step)).reshape(xx.shape)
difference = predicted_response - finite_response

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
electrode_count = 4 * 3 * COUNT
sample = np.arange(0, electrode_count, 8)
axes[0, 0].plot(boundary[:electrode_count, 0], boundary[:electrode_count, 1], color="black", linewidth=1)
axes[0, 0].quiver(
    boundary[sample, 0], boundary[sample, 1],
    motion[sample] * normals[sample, 0], motion[sample] * normals[sample, 1],
    motion[sample], cmap="viridis", scale=28,
)
axes[0, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Physical normal motion h(s) for radius change")

theta_one = np.linspace(0.0, 2.0 * np.pi, 3 * COUNT, endpoint=False)
axes[0, 1].plot(theta_one, motion[:3 * COUNT], label="normal motion h")
axes[0, 1].plot(theta_one, induced_boundary[:3 * COUNT], label="induced deltaV")
axes[0, 1].set(xlabel="right-electrode boundary angle", title="The shape derivative boundary data")
axes[0, 1].legend()

limit = np.max(np.abs(predicted_response))
im0 = axes[0, 2].imshow(predicted_response, origin="lower", extent=[-1, 1, -1, 1], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[0, 2].set(xlabel="x/r0", ylabel="y/r0", title="Predicted physical field response deltaV")
fig.colorbar(im0, ax=axes[0, 2])

axes[1, 0].plot(line_radii, exact_q6, color="black", label="full re-solve")
axes[1, 0].plot(line_radii, linear_q6, linestyle="--", color="tab:blue", label="one boundary derivative")
axes[1, 0].axvline(circle_radius, color="grey", linewidth=1)
axes[1, 0].set(xlabel="circular R/r0", ylabel="physical q6 at 0.5 r0", title="Local prediction away from the base shape")
axes[1, 0].legend()

metric_labels = ["eta", "q6", "q10"]
x = np.arange(3)
width = 0.12
for case_index, (name, boundary_derivative, finite_derivative) in enumerate(comparisons):
    relative = np.abs(boundary_derivative - finite_derivative) / np.maximum(np.abs(finite_derivative), 1e-12)
    axes[1, 1].bar(x + (case_index - 1) * width, relative, width, label=name)
axes[1, 1].set_yscale("log")
axes[1, 1].set_xticks(x, metric_labels)
axes[1, 1].set(ylabel="relative derivative disagreement", title="Boundary derivative versus full geometry re-solve")
axes[1, 1].legend(fontsize=8)

diff_limit = np.max(np.abs(difference))
im1 = axes[1, 2].imshow(difference, origin="lower", extent=[-1, 1, -1, 1], cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit)
axes[1, 2].set(xlabel="x/r0", ylabel="y/r0", title=f"Prediction minus finite difference, step={step}")
fig.colorbar(im1, ax=axes[1, 2])
for ax in axes.ravel():
    ax.grid(alpha=0.15)
fig.suptitle("Plot 13 — a boundary-only shape gradient removes coordinate-warp gauge freedom", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: inspect the derivative agreement and response-difference panels. Tangential reparameterization is absent by construction because the update consumes only normal motion h(s).")
print("saved", OUTPUT)
