import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import brentq


OUTPUT = "exp/harmonic_source_shape_search/results/05_candidate_physical_tradeoffs.png"
COUNT = 48
FRACTION = 0.58
ELLIPSE_EPSILON = -0.15


def ellipse_points(center, radial, tangential, a, b, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    return center + a * np.cos(theta)[:, None] * radial + b * np.sin(theta)[:, None] * tangential


def solve(radius, epsilon):
    a = radius * (1.0 + epsilon)
    b = radius * (1.0 - epsilon)
    distance = 1.0 + a
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    sources, boundary, values = [], [], []
    for radial, tangential, voltage in frames:
        center = distance * radial
        sources.append(ellipse_points(center, radial, tangential, FRACTION * a, FRACTION * b, COUNT, 0.013))
        boundary.append(ellipse_points(center, radial, tangential, a, b, 3 * COUNT))
        values.append(np.full(3 * COUNT, voltage))
    source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    sources.append(6.5 * np.column_stack([np.cos(source_theta), np.sin(source_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(boundary_theta), np.sin(boundary_theta)]))
    values.append(np.zeros(12 * COUNT))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    matrix = np.column_stack([np.log(np.linalg.norm(boundary[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients, a, b, distance


def evaluate(points, sources, coefficients):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = 0.5 * np.log(radius2) @ coefficients[:-1] + coefficients[-1]
    field = -np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, field


theta = 2.0 * np.pi * np.arange(720) / 720


def coefficients(radius, epsilon):
    sources, charges, a, b, distance = solve(radius, epsilon)
    points = 0.5 * np.column_stack([np.cos(theta), np.sin(theta)])
    potential = evaluate(points, sources, charges)[0]
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / 0.5**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / 0.5**6
    a10 = 2.0 * np.mean(potential * np.cos(10.0 * theta)) / 0.5**10
    return a2, a6, a10


def q6(radius, epsilon):
    a2, a6, _ = coefficients(radius, epsilon)
    return (a6 / a2) * 0.5**4


print("HYPOTHESIS: the replicated ellipse is not a uniformly better trap; its 0.114% quadrupole gain will be accompanied by worse finite-amplitude linearity, sharper curvature, or larger surface field because q10 already increased by 45.6%.")
circle_radius = brentq(lambda radius: q6(radius, 0.0), 1.0, 1.3, xtol=1e-12)
ellipse_radius = brentq(lambda radius: q6(radius, ELLIPSE_EPSILON), 0.65, 0.90, xtol=1e-12)
designs = {
    "circular q6=0": (circle_radius, 0.0, "black"),
    "ellipse q6=0": (ellipse_radius, ELLIPSE_EPSILON, "tab:green"),
}

radii = np.linspace(0.05, 0.95, 46)
results = {}
for name, (radius, epsilon, color) in designs.items():
    sources, charges, a, b, distance = solve(radius, epsilon)
    a2, a6, a10 = coefficients(radius, epsilon)
    potential_error, field_error, pseudo_spread = [], [], []
    for probe_radius in radii:
        points = probe_radius * np.column_stack([np.cos(theta), np.sin(theta)])
        potential, field = evaluate(points, sources, charges)
        ideal_potential = a2 * probe_radius**2 * np.cos(2.0 * theta)
        ideal_field = np.column_stack([-2.0 * a2 * points[:, 0], 2.0 * a2 * points[:, 1]])
        potential_error.append(np.sqrt(np.mean((potential - ideal_potential) ** 2)) / np.sqrt(np.mean(ideal_potential**2)))
        field_error.append(np.sqrt(np.mean(np.sum((field - ideal_field) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(ideal_field**2, axis=1))))
        pseudo = np.sum(field**2, axis=1)
        pseudo_spread.append((np.max(pseudo) - np.min(pseudo)) / np.mean(pseudo))

    surface_theta = 2.0 * np.pi * np.arange(720) / 720
    east_surface = np.array([distance, 0.0]) + np.column_stack([a * np.cos(surface_theta), b * np.sin(surface_theta)])
    surface_field = np.linalg.norm(evaluate(east_surface, sources, charges)[1], axis=1)
    minimum_curvature_radius = min(a**2 / b, b**2 / a)
    results[name] = {
        "radius": radius,
        "epsilon": epsilon,
        "color": color,
        "sources": sources,
        "charges": charges,
        "a": a,
        "b": b,
        "distance": distance,
        "eta": 2.0 * a2,
        "q10": (a10 / a2) * 0.5**8,
        "potential_error": np.array(potential_error),
        "field_error": np.array(field_error),
        "pseudo_spread": np.array(pseudo_spread),
        "surface_field": surface_field,
        "minimum_curvature": minimum_curvature_radius,
        "area": np.pi * a * b,
    }

circle = results["circular q6=0"]
candidate = results["ellipse q6=0"]
print(f"circle R={circle_radius:.9f}, eta={circle['eta']:.9f}, q10={circle['q10']:+.4e}, max surface E={np.max(circle['surface_field']):.6f}, minimum curvature radius={circle['minimum_curvature']:.6f}, area={circle['area']:.6f}")
print(f"ellipse R={ellipse_radius:.9f}, eta={candidate['eta']:.9f}, q10={candidate['q10']:+.4e}, max surface E={np.max(candidate['surface_field']):.6f}, minimum curvature radius={candidate['minimum_curvature']:.6f}, area={candidate['area']:.6f}")
print(f"at 0.9 r0: potential nonlinearity ratio ellipse/circle={candidate['potential_error'][42]/circle['potential_error'][42]:.3f}, field nonlinearity ratio={candidate['field_error'][42]/circle['field_error'][42]:.3f}, pseudopotential angular-spread ratio={candidate['pseudo_spread'][42]/circle['pseudo_spread'][42]:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
for name, result in results.items():
    axes[0, 0].semilogy(radii, result["potential_error"], color=result["color"], label=name)
    axes[0, 0].semilogy(radii, result["field_error"], color=result["color"], linestyle="--")
    axes[0, 1].semilogy(radii, result["pseudo_spread"], color=result["color"], label=name)
    axes[0, 2].plot(surface_theta / np.pi, result["surface_field"], color=result["color"], label=name)
axes[0, 0].set(xlabel="measurement radius / r0", ylabel="relative error", title="Solid: V nonlinearity; dashed: E nonlinearity")
axes[0, 0].legend(fontsize=8)
axes[0, 1].set(xlabel="measurement radius / r0", ylabel="(max-min)/mean of |E|^2", title="Pseudopotential angular nonuniformity")
axes[0, 1].legend(fontsize=8)
axes[0, 2].set(xlabel="east-electrode boundary angle / pi", ylabel="surface |E|", title="Surface-field concentration")
axes[0, 2].legend(fontsize=8)

axis = np.linspace(-0.95, 0.95, 151)
gx, gy = np.meshgrid(axis, axis)
grid = np.column_stack([gx.ravel(), gy.ravel()])
error_maps = []
for column, (name, result) in enumerate(results.items()):
    potential = evaluate(grid, result["sources"], result["charges"])[0]
    ideal = 0.5 * result["eta"] * (grid[:, 0] ** 2 - grid[:, 1] ** 2)
    error_maps.append((potential - ideal).reshape(gx.shape))
limit = max(np.max(np.abs(values)) for values in error_maps)
for column, ((name, result), values) in enumerate(zip(results.items(), error_maps)):
    image = axes[1, column].imshow(values, origin="lower", extent=(-0.95, 0.95, -0.95, 0.95), cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1, column].set(xlabel="x/r0", ylabel="y/r0", title=f"{name}: V - fitted quadrupole")
    fig.colorbar(image, ax=axes[1, column], shrink=0.78)

labels = ["eta", "|q10|", "E surface max", "min curvature", "area"]
ratios = [
    candidate["eta"] / circle["eta"],
    abs(candidate["q10"] / circle["q10"]),
    np.max(candidate["surface_field"]) / np.max(circle["surface_field"]),
    candidate["minimum_curvature"] / circle["minimum_curvature"],
    candidate["area"] / circle["area"],
]
bars = axes[1, 2].bar(labels, ratios, color=["tab:green", "tab:red", "tab:orange", "tab:blue", "tab:purple"])
axes[1, 2].axhline(1.0, color="black", linewidth=1)
axes[1, 2].bar_label(bars, fmt="%.3f")
axes[1, 2].tick_params(axis="x", rotation=25)
axes[1, 2].set(ylabel="ellipse / circle", title="The candidate is a tradeoff, not a dominant design")

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 05 — physical review of the first replicated noncircular candidate", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: retain the ellipse as a Pareto example only if its higher eta is worth its finite-amplitude, surface-field, and curvature costs; do not call it an unqualified improvement.")
print("saved", OUTPUT)
