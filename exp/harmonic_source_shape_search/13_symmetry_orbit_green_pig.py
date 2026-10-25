import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/harmonic_source_shape_search/results/16_symmetry_orbit_green_pig.png"
RADIUS = 1.14511
FRACTION = 0.70


def system(count):
    theta_b = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.0017
    theta_s = 2.0 * np.pi * np.arange(count) / count + 0.013
    local_b = RADIUS * np.column_stack([np.cos(theta_b), np.sin(theta_b)])
    local_s = FRACTION * RADIUS * np.column_stack([np.cos(theta_s), np.sin(theta_s)])
    distance = 1.0 + RADIUS
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
        sources.append(center + local_s @ rotation.T)
        boundary.append(center + local_b @ rotation.T)
        values.append(np.full(len(local_b), voltage))
    outer_s_theta = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    outer_b_theta = 2.0 * np.pi * np.arange(16 * count) / (16 * count) + 0.0021
    sources.append(6.5 * np.column_stack([np.cos(outer_s_theta), np.sin(outer_s_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(outer_b_theta), np.sin(outer_b_theta)]))
    values.append(np.zeros(len(outer_b_theta)))
    return np.vstack(sources), np.vstack(boundary), np.concatenate(values)


def basis(points, sources):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    return 0.5 * np.log(radius2), delta / radius2[:, :, None]


def fit(count, symmetric):
    sources, boundary, target = system(count)
    full = basis(boundary, sources)[0]
    if symmetric:
        transform = np.zeros((8 * count, 2 * count))
        for j in range(count):
            transform[j, j] = +1.0
            transform[count + j, j] = +1.0
            transform[2 * count + j, j] = -1.0
            transform[3 * count + j, j] = -1.0
            transform[4 * count + j, count + j] = +1.0
            transform[5 * count + j, count + j] = -1.0
            transform[6 * count + j, count + j] = +1.0
            transform[7 * count + j, count + j] = -1.0
        reduced = full @ transform
        orbit_coefficients, *_ = np.linalg.lstsq(reduced, target, rcond=1e-12)
        coefficients = transform @ orbit_coefficients
        condition = np.linalg.cond(reduced)
        parameter_count = len(orbit_coefficients)
    else:
        design = np.column_stack([full, np.ones(len(boundary))])
        raw, *_ = np.linalg.lstsq(design, target, rcond=1e-12)
        coefficients = raw[:-1]
        condition = np.linalg.cond(design)
        parameter_count = len(raw)
    return sources, coefficients, condition, parameter_count


def evaluate(points, sources, coefficients):
    values, gradients = basis(points, sources)
    return values @ coefficients, np.einsum("nka,k->na", gradients, coefficients)


probe_theta = 2.0 * np.pi * np.arange(1440) / 1440
probe = 0.5 * np.column_stack([np.cos(probe_theta), np.sin(probe_theta)])
unseen_theta = 2.0 * np.pi * (np.arange(401) + 0.37) / 401
unit = np.column_stack([np.cos(unseen_theta), np.sin(unseen_theta)])
distance = 1.0 + RADIUS
right_boundary = np.array([distance, 0.0]) + RADIUS * unit
right_normal = unit

reference_sources, reference_coefficients, *_ = fit(64, True)
reference_values, reference_gradient = evaluate(probe, reference_sources, reference_coefficients)
reference_flux = np.sum(evaluate(right_boundary, reference_sources, reference_coefficients)[1] * right_normal, axis=1)

counts = [6, 8, 12, 16, 24, 32, 40]
records = {False: [], True: []}
models = {}
print("HYPOTHESIS: D4 character tying (+,+,-,-) removes redundant Green-PIG parameters and forbidden asymmetric modes without reducing the physical quadrupole solution space.")
for symmetric in [False, True]:
    for count in counts:
        sources, coefficients, condition, parameters = fit(count, symmetric)
        values, gradient = evaluate(probe, sources, coefficients)
        flux = np.sum(evaluate(right_boundary, sources, coefficients)[1] * right_normal, axis=1)
        field_error = np.sqrt(np.mean(np.sum((gradient - reference_gradient) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(reference_gradient**2, axis=1)))
        flux_error = np.sqrt(np.mean((flux - reference_flux) ** 2)) / np.sqrt(np.mean(reference_flux**2))
        rotated_points = np.column_stack([-probe[:, 1], probe[:, 0]])
        rotated_values = evaluate(rotated_points, sources, coefficients)[0]
        symmetry_error = np.sqrt(np.mean((rotated_values + values) ** 2))
        forbidden = []
        for order in [0, 4, 8, 12]:
            if order == 0:
                forbidden.append(abs(np.mean(values)))
            else:
                forbidden.append(abs(2.0 * np.mean(values * np.cos(order * probe_theta))))
        records[symmetric].append([count, parameters, condition, field_error, flux_error, symmetry_error, max(forbidden)])
        models[(symmetric, count)] = (sources, coefficients)
    records[symmetric] = np.array(records[symmetric])
    final = records[symmetric][-1]
    print(f"{'orbit-tied' if symmetric else 'untied'} count40: parameters={int(final[1])}, condition={final[2]:.3e}, field={final[3]:.3e}, flux={final[4]:.3e}, rotation symmetry={final[5]:.3e}, forbidden={final[6]:.3e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
styles = [(False, "untied sources", "tab:orange"), (True, "D4 orbit-tied", "tab:blue")]
for symmetric, label, color in styles:
    values = records[symmetric]
    axes[0, 0].semilogy(values[:, 1], values[:, 3], marker="o", color=color, label=label)
    axes[0, 1].semilogy(values[:, 1], values[:, 4], marker="o", color=color, label=label)
    axes[0, 2].semilogy(values[:, 1], values[:, 5], marker="o", color=color, label=label)
    axes[1, 0].semilogy(values[:, 1], values[:, 6], marker="o", color=color, label=label)
    axes[1, 1].semilogy(values[:, 1], values[:, 2], marker="o", color=color, label=label)
axes[0, 0].set(xlabel="independent parameters", ylabel="central field relative error", title="Accuracy per independent parameter")
axes[0, 1].set(xlabel="independent parameters", ylabel="electrode flux relative error", title="Shape-gradient information")
axes[0, 2].set(xlabel="independent parameters", ylabel="RMS of V(R90 x)+V(x)", title="Exact quadrupole rotation character")
axes[1, 0].set(xlabel="independent parameters", ylabel="largest forbidden mode", title="Modes symmetry says cannot exist")
axes[1, 1].set(xlabel="independent parameters", ylabel="matrix condition number", title="Does tying improve conditioning?")
for ax in axes.ravel()[:5]:
    ax.legend(fontsize=8)

for symmetric, label, color in styles:
    sources, coefficients = models[(symmetric, 16)]
    flux = np.sum(evaluate(right_boundary, sources, coefficients)[1] * right_normal, axis=1)
    axes[1, 2].plot(unseen_theta, flux, color=color, label=label)
axes[1, 2].plot(unseen_theta, reference_flux, color="black", linestyle="--", label="high-resolution reference")
axes[1, 2].set(xlabel="right-electrode angle", ylabel="normal potential gradient", title="One atom, cloned with symmetry signs")
axes[1, 2].legend(fontsize=8)
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 16 — toggleable symmetry as a group projection, not data augmentation", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: orbit tying is useful only if it reaches the same field and flux accuracy with fewer independent parameters while driving forbidden modes to numerical zero.")
print("saved", OUTPUT)
