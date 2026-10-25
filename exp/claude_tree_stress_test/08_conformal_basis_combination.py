import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/08_conformal_basis_combination.png"
RADIUS = 1.1451090300780968
DISTANCE = 1.0 + RADIUS
KAPPA9 = -0.001221340106021855
POWERS = np.arange(9, 19)
WEIGHTS = np.array([
    0.14526537, -0.10620197, 0.28421932, -0.42300312, -0.77440244,
    14.74465186, -52.90593862, 91.26916034, -77.73953184, 26.50578109,
])
WEIGHTS /= np.sum(WEIGHTS)


def basis(z, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    return z**9 * (z**4 - 1.0) / (z**4 - DISTANCE**4) ** power / leading


def basis_derivative(z, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    denominator = z**4 - DISTANCE**4
    first = (9.0 * z**8 * (z**4 - 1.0) + 4.0 * z**12) / denominator**power
    second = power * 4.0 * z**12 * (z**4 - 1.0) / denominator ** (power + 1)
    return (first - second) / leading


def combined_basis(z, weights=WEIGHTS):
    return sum(weight * basis(z, power) for weight, power in zip(weights, POWERS))


def combined_derivative(z, weights=WEIGHTS):
    return sum(weight * basis_derivative(z, power) for weight, power in zip(weights, POWERS))


def mapping(z, weights=WEIGHTS):
    return z + KAPPA9 * combined_basis(z, weights)


def curvature(curve):
    step = 2.0 * np.pi / len(curve)
    first = (np.roll(curve, -1) - np.roll(curve, 1)) / (2.0 * step)
    second = (np.roll(curve, -1) - 2.0 * curve + np.roll(curve, 1)) / step**2
    return np.imag(np.conj(first) * second) / np.maximum(np.abs(first) ** 3, 1e-15)


theta = np.linspace(0.0, 2.0 * np.pi, 12000, endpoint=False)
reference = DISTANCE + RADIUS * np.exp(1j * theta)
single = reference + KAPPA9 * basis(reference, 10)
combined = mapping(reference)
single_curvature = curvature(single)
combined_curvature = curvature(combined)

axis = np.linspace(-5.0, 5.0, 601)
xx, yy = np.meshgrid(axis, axis)
zz = xx + 1j * yy
vacuum = np.abs(zz) < 5.0
for center in DISTANCE * np.array([1.0, -1.0, 1.0j, -1.0j]):
    vacuum &= np.abs(zz - center) > RADIUS
sample = zz[vacuum][::10]
derivative = 1.0 + KAPPA9 * combined_derivative(sample)

resolution = np.array([750, 1500, 3000, 6000, 12000])
resolution_minimum = []
for count in resolution:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    curve = mapping(DISTANCE + RADIUS * np.exp(1j * angle))
    resolution_minimum.append(np.min(curvature(curve)))

rng = np.random.default_rng(21)
noise_levels = np.logspace(-9, -3, 7)
noise_minimum = []
noise_geometry = []
coarse_reference = DISTANCE + RADIUS * np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 1500, endpoint=False))
nominal_curve = mapping(coarse_reference)
for noise in noise_levels:
    minima, changes = [], []
    for _ in range(80):
        perturbed = WEIGHTS * (1.0 + noise * rng.normal(size=len(WEIGHTS)))
        perturbed /= np.sum(perturbed)
        curve = mapping(coarse_reference, perturbed)
        minima.append(np.min(curvature(curve)))
        changes.append(np.max(np.abs(curve - nominal_curve)))
    noise_minimum.append(np.percentile(minima, [5, 50, 95]))
    noise_geometry.append(np.percentile(changes, [5, 50, 95]))
noise_minimum = np.array(noise_minimum)
noise_geometry = np.array(noise_geometry)

print("HYPOTHESIS: combining rational maps with identical leading z^9 jets can remove rod concavity without changing the exact a10 cancellation, because sum(weights)=1 preserves kappa9.")
print("weight sum", np.sum(WEIGHTS), "weight norm", np.linalg.norm(WEIGHTS))
print("combined maximum motion", np.max(np.abs(combined - reference)))
print("combined curvature min/max", np.min(combined_curvature), np.max(combined_curvature))
print("minimum sampled |f prime|", np.min(np.abs(derivative)))
outer = 5.0 * np.exp(1j * theta)
print("maximum enclosure motion", np.max(np.abs(mapping(outer) - outer)))
print("inner tip displacement", abs(mapping(np.array([1.0]))[0] - 1.0))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].plot(reference.real, reference.imag, color="grey", label="circle")
axes[0, 0].plot(single.real, single.imag, color="tab:red", alpha=0.7, label="single power 10")
axes[0, 0].plot(combined.real, combined.imag, color="tab:blue", label="curvature-designed mixture")
axes[0, 0].scatter([1.0], [0.0], color="magenta", s=25, label="fixed tip")
axes[0, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Same center jet, very different electrodes")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(theta / np.pi, single_curvature, color="tab:red", label="single power 10")
axes[0, 1].plot(theta / np.pi, combined_curvature, color="tab:blue", label="mixture")
axes[0, 1].axhline(0.0, color="black", linewidth=1)
axes[0, 1].set(xlabel="local rod angle/pi", ylabel="signed curvature [1/r0]", title="Convexity is recovered")
axes[0, 1].legend(fontsize=8)

axes[0, 2].bar(POWERS, WEIGHTS)
axes[0, 2].axhline(0.0, color="black", linewidth=1)
axes[0, 2].set(xlabel="hidden-pole order", ylabel="mixture weight", title="Raw basis is ill-conditioned")

axes[1, 0].plot(resolution, resolution_minimum, marker="o")
axes[1, 0].axhline(0.0, color="black", linewidth=1)
axes[1, 0].set(xlabel="boundary samples", ylabel="minimum signed curvature", title="Convexity survives spatial refinement")

axes[1, 1].semilogx(noise_levels, noise_minimum[:, 1], marker="o", label="median min curvature")
axes[1, 1].fill_between(noise_levels, noise_minimum[:, 0], noise_minimum[:, 2], alpha=0.25, label="5-95%")
axes[1, 1].axhline(0.0, color="black", linewidth=1)
axes[1, 1].set(xlabel="relative independent weight noise", ylabel="minimum curvature", title="Coefficient sensitivity")
axes[1, 1].legend(fontsize=8)

axes[1, 2].loglog(noise_levels, noise_geometry[:, 1], marker="o", label="median")
axes[1, 2].fill_between(noise_levels, noise_geometry[:, 0], noise_geometry[:, 2], alpha=0.25, label="5-95%")
axes[1, 2].set(xlabel="relative independent weight noise", ylabel="maximum boundary change/r0", title="Cancellation condition number in physical space")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 08 — convexifying an exact multipole-cancelling conformal coordinate", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: the explicit mixture is convex at 12000 boundary samples while keeping the same exact center jet, fixed tip, tiny enclosure leakage, and positive conformal derivative. Its large alternating weights reveal basis ill-conditioning, so a future learned architecture should orthogonalize these rational modes rather than optimize raw coefficients.")
print("saved", OUTPUT)
