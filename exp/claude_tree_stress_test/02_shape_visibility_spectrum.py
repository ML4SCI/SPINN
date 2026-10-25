import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/claude_tree_stress_test/results/02_shape_visibility_spectrum.png"
COUNT = 52
FRACTION = 0.70
MODES = np.arange(17)


def local_circle(radius, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    points = radius * np.column_stack([np.cos(theta), np.sin(theta)])
    normals = np.column_stack([np.cos(theta), np.sin(theta)])
    return theta, points, normals


def geometry(radius):
    theta, local_boundary, local_normals = local_circle(radius, 3 * COUNT)
    _, local_sources, _ = local_circle(FRACTION * radius, COUNT, 0.013)
    distance = 1.0 + radius
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    sources, boundary, values, normals = [], [], [], []
    mode_motion = []
    for radial, tangential, voltage in frames:
        rotation = np.column_stack([radial, tangential])
        center = distance * radial
        sources.append(center + local_sources @ rotation.T)
        boundary.append(center + local_boundary @ rotation.T)
        values.append(np.full(len(local_boundary), voltage))
        normals.append(local_normals @ rotation.T)
        mode_motion.append(np.column_stack([np.cos(k * theta) for k in MODES]))
    outer_theta, outer_boundary, outer_normals = local_circle(5.0, 12 * COUNT)
    _, outer_sources, _ = local_circle(6.5, 4 * COUNT, 0.007)
    sources.append(outer_sources)
    boundary.append(outer_boundary)
    values.append(np.zeros(len(outer_boundary)))
    normals.append(outer_normals)
    mode_motion.append(np.zeros((len(outer_boundary), len(MODES))))
    return np.vstack(sources), np.vstack(boundary), np.concatenate(values), np.vstack(normals), np.vstack(mode_motion)


def matrix(points, sources):
    return np.column_stack([np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(points))])


def evaluate_many(points, sources, coefficients):
    distances = np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)
    return np.column_stack([np.log(distances), np.ones(len(points))]) @ coefficients


def base_q6(radius):
    sources, boundary, values, _, _ = geometry(radius)
    coefficients, *_ = np.linalg.lstsq(matrix(boundary, sources), values, rcond=1e-12)
    angle = 2.0 * np.pi * np.arange(720) / 720
    probe = 0.5 * np.column_stack([np.cos(angle), np.sin(angle)])
    values_probe = evaluate_many(probe, sources, coefficients[:, None])[:, 0]
    a2 = 2.0 * np.mean(values_probe * np.cos(2.0 * angle)) / 0.5**2
    a6 = 2.0 * np.mean(values_probe * np.cos(6.0 * angle)) / 0.5**6
    return (a6 / a2) * 0.5**4


radius = brentq(base_q6, 1.05, 1.25)
sources, boundary, boundary_values, normals, motion = geometry(radius)
A = matrix(boundary, sources)
base_coefficients, *_ = np.linalg.lstsq(A, boundary_values, rcond=1e-12)

delta = boundary[:, None, :] - sources[None, :, :]
radius2 = np.sum(delta**2, axis=2)
field = -np.sum(delta / radius2[:, :, None] * base_coefficients[None, :-1, None], axis=1)
normal_field = np.sum(field * normals, axis=1)
induced = motion * normal_field[:, None]
response_coefficients, *_ = np.linalg.lstsq(A, induced, rcond=1e-12)

print("HYPOTHESIS: the map from physical electrode normal motion to a central field is strongly smoothing, so only a few combinations of many boundary modes are observable on a small measurement disc.")
print("q6-zero circular radius", radius)

rng = np.random.default_rng(7)
sample_angle = rng.uniform(0.0, 2.0 * np.pi, 3500)
sample_unit_radius = np.sqrt(rng.uniform(0.0, 1.0, 3500))
radii = [0.25, 0.50, 0.75]
responses = {}
singular_values = {}
right_vectors = {}
for probe_radius in radii:
    points = probe_radius * sample_unit_radius[:, None] * np.column_stack([np.cos(sample_angle), np.sin(sample_angle)])
    response = evaluate_many(points, sources, response_coefficients)
    response -= np.mean(response, axis=0, keepdims=True)
    response /= np.sqrt(len(points))
    responses[probe_radius] = response
    _, singular, vt = np.linalg.svd(response, full_matrices=False)
    singular_values[probe_radius] = singular
    right_vectors[probe_radius] = vt
    print("probe", probe_radius, "normalized singular values", singular / singular[0])

probe_theta = 2.0 * np.pi * np.arange(1440) / 1440
probe = 0.5 * np.column_stack([np.cos(probe_theta), np.sin(probe_theta)])
probe_response = evaluate_many(probe, sources, response_coefficients)
orders = np.array([2, 6, 10, 14, 18, 22])
coefficient_response = np.array([
    [2.0 * np.mean(probe_response[:, k] * np.cos(order * probe_theta)) / 0.5**order for k in MODES]
    for order in orders
])
physical_visibility = np.abs(coefficient_response * 0.5**orders[:, None])

grid_axis = np.linspace(-0.85, 0.85, 181)
xx, yy = np.meshgrid(grid_axis, grid_axis)
grid = np.column_stack([xx.ravel(), yy.ravel()])
grid_response = evaluate_many(grid, sources, response_coefficients)
inside = xx**2 + yy**2 <= 0.75**2

most_visible = right_vectors[0.5][0]
least_visible = right_vectors[0.5][-1]
local_theta = 2.0 * np.pi * np.arange(3 * COUNT) / (3 * COUNT)
most_motion = np.column_stack([np.cos(k * local_theta) for k in MODES]) @ most_visible
least_motion = np.column_stack([np.cos(k * local_theta) for k in MODES]) @ least_visible

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
electrode_count = 4 * 3 * COUNT
axes[0, 0].plot(boundary[:electrode_count, 0], boundary[:electrode_count, 1], color="black", linewidth=0.8)
sample = np.arange(0, electrode_count, 12)
shown_mode = 12
axes[0, 0].quiver(
    boundary[sample, 0], boundary[sample, 1],
    motion[sample, shown_mode] * normals[sample, 0], motion[sample, shown_mode] * normals[sample, 1],
    motion[sample, shown_mode], cmap="coolwarm", scale=28,
)
circle = plt.Circle((0, 0), 0.5, fill=False, color="magenta", linewidth=1.5)
axes[0, 0].add_patch(circle)
axes[0, 0].set(aspect="equal", xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="x/r0", ylabel="y/r0", title="A physical normal mode, not a coordinate warp")

for probe_radius in radii:
    axes[0, 1].semilogy(MODES, singular_values[probe_radius] / singular_values[probe_radius][0], marker="o", label=f"disc r={probe_radius}")
axes[0, 1].set(xlabel="singular direction index", ylabel="relative field singular value", title="Shape-to-central-field information spectrum")
axes[0, 1].legend()

im0 = axes[0, 2].imshow(
    np.log10(np.maximum(physical_visibility, 1e-16)), aspect="auto", origin="lower", cmap="viridis",
    extent=[MODES[0] - 0.5, MODES[-1] + 0.5, orders[0] - 2, orders[-1] + 2],
)
axes[0, 2].set(xlabel="rod normal-motion mode k", ylabel="central harmonic order", yticks=orders, title="log10 physical harmonic response at r=0.5")
fig.colorbar(im0, ax=axes[0, 2])

for column, title, ax in [(0, "uniform radius response", axes[1, 0]), (12, "fine boundary ripple response", axes[1, 1])]:
    field_image = grid_response[:, column].reshape(xx.shape)
    field_image[~inside] = np.nan
    limit = np.nanmax(np.abs(field_image))
    image = ax.imshow(field_image, origin="lower", extent=[-0.85, 0.85, -0.85, 0.85], cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title=title)
    fig.colorbar(image, ax=ax)

axes[1, 2].plot(local_theta / np.pi, most_motion / np.max(np.abs(most_motion)), label="most visible combination")
axes[1, 2].plot(local_theta / np.pi, least_motion / np.max(np.abs(least_motion)), label="least visible combination")
axes[1, 2].set(xlabel="local rod angle / pi", ylabel="normalized normal motion", title="What the central disc sees and nearly misses")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 02 — observability of physical electrode motion", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

ratio = singular_values[0.5][-1] / singular_values[0.5][0]
print("CONCLUSION: the r=0.5 central-field map spans a wide visibility spectrum; least/most singular ratio", ratio, ". This is a physical inverse-problem conditioning issue even after coordinate gauge freedom is removed.")
print("saved", OUTPUT)
