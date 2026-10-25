import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/20_orthogonal_boundary_coordinates.png"
RADIUS = 1.1451090300780968
DISTANCE = 1.0 + RADIUS
OUTER_RADIUS = 5.0
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


def basis_derivative(z, order, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    denominator = z**4 - DISTANCE**4
    first = (order * z ** (order - 1) * (z**4 - 1.0) + 4.0 * z ** (order + 3)) / denominator**power
    second = power * 4.0 * z ** (order + 3) * (z**4 - 1.0) / denominator ** (power + 1)
    return (first - second) / leading


theta_fit = 2.0 * np.pi * np.arange(1024) / 1024
right_fit = DISTANCE + RADIUS * np.exp(1j * theta_fit)


def boundary_matrix(order, powers):
    curves = np.column_stack([basis(right_fit, order, power) for power in powers])
    return np.vstack([curves.real, curves.imag]) / np.sqrt(len(right_fit))


def svd_family(order, powers, target_weights):
    matrix = boundary_matrix(order, powers)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    target_coordinate = singular * (vh @ target_weights)
    reconstructions = []
    for rank in range(1, len(powers) + 1):
        coordinate = target_coordinate[:rank].copy()
        jet_functional = (vh[:rank] @ np.ones(len(powers))) / singular[:rank]
        coordinate += jet_functional * (1.0 - jet_functional @ coordinate) / (jet_functional @ jet_functional)
        weights = vh[:rank].T @ (coordinate / singular[:rank])
        reconstructions.append(weights)
    return singular, target_coordinate, reconstructions


s9, c9, reconstructions9 = svd_family(9, P9, W9)
s13, c13, reconstructions13 = svd_family(13, P13, W13)


def mapping(z, weights9, weights13):
    ninth = sum(weight * basis(z, 9, power) for weight, power in zip(weights9, P9))
    thirteenth = sum(weight * basis(z, 13, power) for weight, power in zip(weights13, P13))
    return z + KAPPA9 * ninth + DELTA13 * thirteenth


def mapping_derivative(z, weights9, weights13):
    ninth = sum(weight * basis_derivative(z, 9, power) for weight, power in zip(weights9, P9))
    thirteenth = sum(weight * basis_derivative(z, 13, power) for weight, power in zip(weights13, P13))
    return 1.0 + KAPPA9 * ninth + DELTA13 * thirteenth


def curvature(curve):
    step = 2.0 * np.pi / len(curve)
    first = (np.roll(curve, -1) - np.roll(curve, 1)) / (2.0 * step)
    second = (np.roll(curve, -1) - 2.0 * curve + np.roll(curve, 1)) / step**2
    return np.imag(np.conj(first) * second) / np.maximum(np.abs(first) ** 3, 1e-15)


theta = 2.0 * np.pi * np.arange(4096) / 4096
right = DISTANCE + RADIUS * np.exp(1j * theta)
target_curve = mapping(right, W9, W13)
outer = OUTER_RADIUS * np.exp(1j * theta)
boundary_error = np.zeros((len(P9), len(P13)))
minimum_curvature = np.zeros_like(boundary_error)
enclosure_motion = np.zeros_like(boundary_error)
raw_norm = np.zeros_like(boundary_error)
for i, weights9 in enumerate(reconstructions9):
    for j, weights13 in enumerate(reconstructions13):
        curve = mapping(right, weights9, weights13)
        boundary_error[i, j] = np.max(np.abs(curve - target_curve))
        minimum_curvature[i, j] = np.min(curvature(curve))
        enclosure_motion[i, j] = np.max(np.abs(mapping(outer, weights9, weights13) - outer))
        raw_norm[i, j] = np.sqrt(np.sum(weights9**2) + np.sum(weights13**2))

feasible = (minimum_curvature > 0.0) & (boundary_error < 1e-3) & (enclosure_motion < 1e-5)
if np.any(feasible):
    scores = np.add.outer(np.arange(1, len(P9) + 1), np.arange(1, len(P13) + 1)).astype(float)
    scores[~feasible] = np.inf
    chosen_i, chosen_j = np.unravel_index(np.argmin(scores), scores.shape)
else:
    chosen_i, chosen_j = len(P9) - 1, len(P13) - 1

chosen9 = reconstructions9[chosen_i]
chosen13 = reconstructions13[chosen_j]
chosen_curve = mapping(right, chosen9, chosen13)

axis = np.linspace(-5.0, 5.0, 401)
xx, yy = np.meshgrid(axis, axis)
sample = (xx + 1j * yy).ravel()
vacuum = np.abs(sample) < OUTER_RADIUS
centers = DISTANCE * np.array([1.0, -1.0, 1.0j, -1.0j])
for center in centers:
    vacuum &= np.abs(sample - center) > RADIUS
sample = sample[vacuum][::10]
minimum_derivative = np.min(np.abs(mapping_derivative(sample, chosen9, chosen13)))

print("HYPOTHESIS: SVD coordinates of physical boundary motion can remove redundant pole-power directions while retaining exact center jets, convexity, and enclosure locality.")
print("condition numbers B9 B13", s9[0] / s9[-1], s13[0] / s13[-1])
print("current raw norms", np.linalg.norm(W9), np.linalg.norm(W13), "orthogonal coordinate norms", np.linalg.norm(c9), np.linalg.norm(c13))
print("chosen ranks", chosen_i + 1, chosen_j + 1, "of", len(P9), len(P13))
print("chosen exact jet sums", np.sum(chosen9), np.sum(chosen13))
print("chosen boundary max error", boundary_error[chosen_i, chosen_j], "minimum curvature", minimum_curvature[chosen_i, chosen_j])
print("chosen enclosure motion", enclosure_motion[chosen_i, chosen_j], "minimum sampled |f prime|", minimum_derivative)
print("chosen raw norms", np.linalg.norm(chosen9), np.linalg.norm(chosen13))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].semilogy(np.arange(1, len(s9) + 1), s9 / s9[0], marker="o", label="ninth-order family")
axes[0, 0].semilogy(np.arange(1, len(s13) + 1), s13 / s13[0], marker="s", label="thirteenth-order family")
axes[0, 0].set(xlabel="SVD mode", ylabel="relative boundary singular value", title="Raw pole functions are nearly dependent")
axes[0, 0].legend(fontsize=8)

axes[0, 1].semilogy(np.arange(1, len(c9) + 1), np.maximum(np.abs(c9), 1e-18), marker="o", label="ninth-order target")
axes[0, 1].semilogy(np.arange(1, len(c13) + 1), np.maximum(np.abs(c13), 1e-18), marker="s", label="thirteenth-order target")
axes[0, 1].set(xlabel="orthogonal boundary mode", ylabel="absolute physical coordinate", title="The realized shape in orthogonal coordinates")
axes[0, 1].legend(fontsize=8)

image0 = axes[0, 2].imshow(np.log10(np.maximum(boundary_error, 1e-16)).T, origin="lower", aspect="auto", extent=[0.5, len(P9) + 0.5, 0.5, len(P13) + 0.5], cmap="viridis_r")
axes[0, 2].scatter([chosen_i + 1], [chosen_j + 1], color="red", marker="x", s=70)
axes[0, 2].set(xlabel="retained ninth-order modes", ylabel="retained thirteenth-order modes", title="log10 max boundary error")
fig.colorbar(image0, ax=axes[0, 2])

image1 = axes[1, 0].imshow(minimum_curvature.T, origin="lower", aspect="auto", extent=[0.5, len(P9) + 0.5, 0.5, len(P13) + 0.5], cmap="coolwarm", vmin=-np.max(np.abs(minimum_curvature)), vmax=np.max(np.abs(minimum_curvature)))
axes[1, 0].contour(np.arange(1, len(P9) + 1), np.arange(1, len(P13) + 1), minimum_curvature.T, levels=[0.0], colors="black")
axes[1, 0].scatter([chosen_i + 1], [chosen_j + 1], color="yellow", marker="x", s=70)
axes[1, 0].set(xlabel="retained ninth-order modes", ylabel="retained thirteenth-order modes", title="Minimum signed curvature")
fig.colorbar(image1, ax=axes[1, 0])

axes[1, 1].plot(target_curve.real, target_curve.imag, color="black", label="full raw basis")
axes[1, 1].plot(chosen_curve.real, chosen_curve.imag, linestyle="--", color="tab:red", label=f"SVD ranks {chosen_i + 1}, {chosen_j + 1}")
axes[1, 1].scatter([1.0], [0.0], color="magenta", s=20, label="fixed inner tip")
axes[1, 1].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Does compression change the electrode?")
axes[1, 1].legend(fontsize=8)

for i in range(len(P9)):
    for j in range(len(P13)):
        axes[1, 2].scatter(boundary_error[i, j], raw_norm[i, j], s=12, color="tab:blue" if minimum_curvature[i, j] > 0 else "tab:grey", alpha=0.6)
axes[1, 2].scatter([boundary_error[chosen_i, chosen_j]], [raw_norm[chosen_i, chosen_j]], color="red", marker="x", s=80, label="chosen truncation")
axes[1, 2].axhline(np.sqrt(np.sum(W9**2) + np.sum(W13**2)), color="black", linestyle="--", label="full raw norm")
axes[1, 2].set(xscale="log", yscale="log", xlabel="max boundary error/r0", ylabel="combined raw weight norm", title="Blue is convex: conditioning tradeoff")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 20 — orthogonal coordinates for localized conformal shape modes", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: truncation succeeds only if boundary fidelity, convexity, and locality survive together. Orthogonal coordinates remove redundant optimizer directions even when the exact localized shape still requires many singular modes.")
print("saved", OUTPUT)
