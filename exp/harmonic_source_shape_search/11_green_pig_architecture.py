import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/harmonic_source_shape_search/results/14_green_pig_architecture.png"
RNG = np.random.default_rng(7)
PROBE_RADIUS = 0.5


def circle_geometry(radius, count, fraction):
    theta_b = 2.0 * np.pi * np.arange(3 * count) / (3 * count)
    theta_s = 2.0 * np.pi * np.arange(count) / count + 0.013
    unit_b = np.column_stack([np.cos(theta_b), np.sin(theta_b)])
    unit_s = np.column_stack([np.cos(theta_s), np.sin(theta_s)])
    distance = 1.0 + radius
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    sources, boundary, values, normals, motion = [], [], [], [], []
    for radial, tangential, voltage in frames:
        rotation = np.column_stack([radial, tangential])
        center = distance * radial
        sources.append(center + fraction * radius * unit_s @ rotation.T)
        boundary.append(center + radius * unit_b @ rotation.T)
        values.append(np.full(len(unit_b), voltage))
        normal = unit_b @ rotation.T
        normals.append(normal)
        displacement = radial + unit_b @ rotation.T
        motion.append(np.sum(displacement * normal, axis=1))
    theta_os = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    theta_ob = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    sources.append(6.5 * np.column_stack([np.cos(theta_os), np.sin(theta_os)]))
    boundary.append(5.0 * np.column_stack([np.cos(theta_ob), np.sin(theta_ob)]))
    values.append(np.zeros(len(theta_ob)))
    normals.append(np.column_stack([np.cos(theta_ob), np.sin(theta_ob)]))
    motion.append(np.zeros(len(theta_ob)))
    return np.vstack(sources), np.vstack(boundary), np.concatenate(values), np.vstack(normals), np.concatenate(motion)


def green_basis(points, sources):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    value = 0.5 * np.log(radius2)
    gradient = delta / radius2[:, :, None]
    return value, gradient


def green_solve(radius, count=56, fraction=0.70, values=None):
    sources, boundary, target, normals, motion = circle_geometry(radius, count, fraction)
    if values is not None:
        target = values
    basis, _ = green_basis(boundary, sources)
    coefficients, *_ = np.linalg.lstsq(np.column_stack([basis, np.ones(len(boundary))]), target, rcond=1e-12)
    return sources, coefficients, boundary, target, normals, motion


def green_evaluate(points, sources, coefficients):
    basis, gradient_basis = green_basis(points, sources)
    value = basis @ coefficients[:-1] + coefficients[-1]
    gradient = np.einsum("nka,k->na", gradient_basis, coefficients[:-1])
    return value, gradient


theta = 2.0 * np.pi * np.arange(1440) / 1440
probe = PROBE_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])


def green_q6(radius):
    sources, coefficients, *_ = green_solve(radius)
    values = green_evaluate(probe, sources, coefficients)[0]
    a2 = 2.0 * np.mean(values * np.cos(2.0 * theta)) / PROBE_RADIUS**2
    a6 = 2.0 * np.mean(values * np.cos(6.0 * theta)) / PROBE_RADIUS**6
    return (a6 / a2) * PROBE_RADIUS**4


radius = brentq(green_q6, 1.0, 1.3, xtol=1e-12)
ref_sources, ref_coefficients, boundary, targets, normals, motion = green_solve(radius, 64, 0.70)

grid = np.linspace(-5.0, 5.0, 19)
gx, gy = np.meshgrid(grid, grid)
gaussian_centers = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.zeros(len(gaussian_centers), dtype=bool)
for center in [[1 + radius, 0], [-1 - radius, 0], [0, 1 + radius], [0, -1 - radius]]:
    inside |= np.sum((gaussian_centers - center) ** 2, axis=1) < (0.8 * radius) ** 2
gaussian_centers = gaussian_centers[~inside]

interior = RNG.uniform(-5.0, 5.0, size=(3500, 2))
inside = np.zeros(len(interior), dtype=bool)
for center in [[1 + radius, 0], [-1 - radius, 0], [0, 1 + radius], [0, -1 - radius]]:
    inside |= np.sum((interior - center) ** 2, axis=1) < radius**2
interior = interior[~inside][:2400]


def gaussian_basis(points, sigma):
    delta = points[:, None, :] - gaussian_centers[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    value = np.exp(-0.5 * radius2 / sigma**2)
    gradient = -delta / sigma**2 * value[:, :, None]
    laplace = (radius2 / sigma**4 - 2.0 / sigma**2) * value
    return value, gradient, laplace


def gaussian_solve(sigma, pde_weight, boundary_values=targets):
    boundary_basis = gaussian_basis(boundary, sigma)[0]
    laplace_basis = gaussian_basis(interior, sigma)[2]
    design = np.vstack([
        np.column_stack([boundary_basis, np.ones(len(boundary))]),
        np.sqrt(pde_weight) * np.column_stack([laplace_basis, np.zeros(len(interior))]),
    ])
    right = np.concatenate([boundary_values, np.zeros(len(interior))])
    coefficients, *_ = np.linalg.lstsq(design, right, rcond=1e-10)
    return coefficients


def gaussian_evaluate(points, sigma, coefficients):
    value_basis, gradient_basis, laplace_basis = gaussian_basis(points, sigma)
    value = value_basis @ coefficients[:-1] + coefficients[-1]
    gradient = np.einsum("nka,k->na", gradient_basis, coefficients[:-1])
    laplace = laplace_basis @ coefficients[:-1]
    return value, gradient, laplace


unseen_theta = 2.0 * np.pi * (np.arange(317) + 0.37) / 317
unit = np.column_stack([np.cos(unseen_theta), np.sin(unseen_theta)])
right_boundary = np.array([1.0 + radius, 0.0]) + radius * unit
right_normal = unit
ref_flux = np.sum(green_evaluate(right_boundary, ref_sources, ref_coefficients)[1] * right_normal, axis=1)
central_axis = np.linspace(-0.95, 0.95, 300)
central_points = np.column_stack([central_axis, np.zeros_like(central_axis)])
ref_values, ref_gradient = green_evaluate(central_points, ref_sources, ref_coefficients)

sigmas = [0.35, 0.60, 1.0, 1.6]
pde_weights = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
records = []
models = {}
print("HYPOTHESIS: ordinary Gaussian PIG atoms face a PDE-versus-boundary-flux tradeoff, while Green-function PIG atoms make the PDE exact and preserve the electrode flux needed by the physical shape derivative.")
for sigma in sigmas:
    for weight in pde_weights:
        coefficients = gaussian_solve(sigma, weight)
        boundary_value, _, _ = gaussian_evaluate(boundary, sigma, coefficients)
        _, right_gradient, _ = gaussian_evaluate(right_boundary, sigma, coefficients)
        _, _, laplace = gaussian_evaluate(interior, sigma, coefficients)
        values, gradient, _ = gaussian_evaluate(central_points, sigma, coefficients)
        bc_error = np.sqrt(np.mean((boundary_value - targets) ** 2))
        laplace_error = np.sqrt(np.mean(laplace**2))
        flux_error = np.sqrt(np.mean((np.sum(right_gradient * right_normal, axis=1) - ref_flux) ** 2)) / np.sqrt(np.mean(ref_flux**2))
        field_error = np.sqrt(np.mean(np.sum((gradient - ref_gradient) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(ref_gradient**2, axis=1)))
        records.append((sigma, weight, bc_error, laplace_error, flux_error, field_error))
        models[(sigma, weight)] = coefficients
records = np.array(records)
best_index = np.argmin(records[:, 4] + records[:, 5])
best_sigma, best_weight = records[best_index, :2]
best_coefficients = models[(best_sigma, best_weight)]
print(f"best Gaussian diagnostic setting sigma={best_sigma:.2f}, PDE weight={best_weight:g}: boundary={records[best_index,2]:.3e}, laplace={records[best_index,3]:.3e}, flux relative={records[best_index,4]:.3e}, center field relative={records[best_index,5]:.3e}")

green_sources, green_coefficients, green_boundary, _, _, _ = green_solve(radius, 24, 0.70)
green_boundary_value = green_evaluate(boundary, green_sources, green_coefficients)[0]
green_flux = np.sum(green_evaluate(right_boundary, green_sources, green_coefficients)[1] * right_normal, axis=1)
green_gradient = green_evaluate(central_points, green_sources, green_coefficients)[1]
green_bc = np.sqrt(np.mean((green_boundary_value - targets) ** 2))
green_flux_error = np.sqrt(np.mean((green_flux - ref_flux) ** 2)) / np.sqrt(np.mean(ref_flux**2))
green_field_error = np.sqrt(np.mean(np.sum((green_gradient - ref_gradient) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(ref_gradient**2, axis=1)))
print(f"Green PIG with 24 atoms/electrode: boundary={green_bc:.3e}, Laplace=exact, flux relative={green_flux_error:.3e}, center field relative={green_field_error:.3e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for sigma in sigmas:
    subset = records[records[:, 0] == sigma]
    axes[0, 0].loglog(subset[:, 3], subset[:, 2], marker="o", label=f"sigma={sigma}")
axes[0, 0].set(xlabel="interior Laplace RMSE", ylabel="boundary potential RMSE", title="Gaussian atoms: PDE versus boundary Pareto front")
axes[0, 0].legend(fontsize=8)

scatter = axes[0, 1].scatter(records[:, 5], records[:, 4], c=np.log10(records[:, 1]), cmap="viridis", s=55)
axes[0, 1].scatter([green_field_error], [green_flux_error], marker="*", s=220, color="red", label="Green PIG")
axes[0, 1].set(xscale="log", yscale="log", xlabel="central field relative error", ylabel="electrode flux relative error", title="What the shape optimizer actually needs")
axes[0, 1].legend()
fig.colorbar(scatter, ax=axes[0, 1], label="log10 PDE weight")

best_flux = np.sum(gaussian_evaluate(right_boundary, best_sigma, best_coefficients)[1] * right_normal, axis=1)
axes[0, 2].plot(unseen_theta, ref_flux, color="black", label="reference")
axes[0, 2].plot(unseen_theta, best_flux, label="Gaussian PIG")
axes[0, 2].plot(unseen_theta, green_flux, linestyle="--", label="Green PIG")
axes[0, 2].set(xlabel="right-electrode angle", ylabel="normal potential gradient", title="Unseen electrode flux")
axes[0, 2].legend(fontsize=8)

best_values, best_gradient, best_laplace = gaussian_evaluate(central_points, best_sigma, best_coefficients)
green_values, green_gradient = green_evaluate(central_points, green_sources, green_coefficients)
axes[1, 0].plot(central_axis, ref_values, color="black", label="reference")
axes[1, 0].plot(central_axis, best_values, label="Gaussian PIG")
axes[1, 0].plot(central_axis, green_values, linestyle="--", label="Green PIG")
axes[1, 0].set(xlabel="x/r0", ylabel="potential", title="Central potential slice")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(central_axis, best_laplace, label="Gaussian PIG")
axes[1, 1].axhline(0.0, color="black", linewidth=1, label="Green PIG: exact zero")
axes[1, 1].set(xlabel="x/r0", ylabel="Laplacian", title="PDE residual along the measurement axis")
axes[1, 1].legend(fontsize=8)

metric_names = ["boundary V", "center E", "electrode flux"]
gaussian_errors = [records[best_index, 2], records[best_index, 5], records[best_index, 4]]
green_errors = [green_bc, green_field_error, green_flux_error]
x = np.arange(3)
axes[1, 2].bar(x - 0.18, gaussian_errors, 0.36, label="Gaussian PIG")
axes[1, 2].bar(x + 0.18, green_errors, 0.36, label="Green PIG")
axes[1, 2].set_yscale("log")
axes[1, 2].set_xticks(x, metric_names, rotation=15)
axes[1, 2].set(ylabel="error", title="Architecture-level comparison")
axes[1, 2].legend(fontsize=8)
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 14 — modifying PIG atoms to obey electrostatics before training", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: decide from the flux and Pareto panels whether Gaussian potential atoms are an efficient field representation, or whether PIG should parameterize harmonic Green atoms / boundary charge instead.")
print("saved", OUTPUT)
