import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/07_conformal_manufacturability.png"
RADIUS = 1.1451090300780968
DISTANCE = 1.0 + RADIUS
KAPPA9 = -0.001221340106021855
OUTER_RADIUS = 5.0
POWERS = np.arange(6, 17)
THETA = np.linspace(0.0, 2.0 * np.pi, 6000, endpoint=False)


def basis(z, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    return z**9 * (z**4 - 1.0) / (z**4 - DISTANCE**4) ** power / leading


def basis_derivative(z, power):
    leading = (-1.0) / ((-DISTANCE**4) ** power)
    denominator = z**4 - DISTANCE**4
    first = (9.0 * z**8 * (z**4 - 1.0) + 4.0 * z**12) / denominator**power
    second = power * 4.0 * z**12 * (z**4 - 1.0) / denominator ** (power + 1)
    return (first - second) / leading


def mapping(z, power):
    return z + KAPPA9 * basis(z, power)


def periodic_curvature(curve):
    step = 2.0 * np.pi / len(curve)
    first = (np.roll(curve, -1) - np.roll(curve, 1)) / (2.0 * step)
    second = (np.roll(curve, -1) - 2.0 * curve + np.roll(curve, 1)) / step**2
    return np.imag(np.conj(first) * second) / np.maximum(np.abs(first) ** 3, 1e-15)


right_reference = DISTANCE + RADIUS * np.exp(1j * THETA)
outer_reference = OUTER_RADIUS * np.exp(1j * THETA)
axis = np.linspace(-5.0, 5.0, 501)
xx, yy = np.meshgrid(axis, axis)
sample = (xx + 1j * yy).ravel()
vacuum = np.abs(sample) < OUTER_RADIUS
for center in DISTANCE * np.array([1.0, -1.0, 1.0j, -1.0j]):
    vacuum &= np.abs(sample - center) > RADIUS
sample = sample[vacuum][::10]

max_motion = []
outer_motion = []
minimum_derivative = []
minimum_curvature = []
maximum_curvature = []
negative_fraction = []
curves = {}
curvatures = {}
for power in POWERS:
    curve = mapping(right_reference, power)
    curvature = periodic_curvature(curve)
    curves[power] = curve
    curvatures[power] = curvature
    max_motion.append(np.max(np.abs(curve - right_reference)))
    outer_motion.append(np.max(np.abs(mapping(outer_reference, power) - outer_reference)))
    minimum_derivative.append(np.min(np.abs(1.0 + KAPPA9 * basis_derivative(sample, power))))
    minimum_curvature.append(np.min(curvature))
    maximum_curvature.append(np.max(curvature))
    negative_fraction.append(np.mean(curvature < 0.0))

max_motion = np.array(max_motion)
outer_motion = np.array(outer_motion)
minimum_derivative = np.array(minimum_derivative)
minimum_curvature = np.array(minimum_curvature)
maximum_curvature = np.array(maximum_curvature)
negative_fraction = np.array(negative_fraction)

print("HYPOTHESIS: increasing rational pole order will improve enclosure locality and reduce displacement, but may concentrate boundary curvature; a useful conformal mode must remain injective and have manufacturable curvature.")
for i, power in enumerate(POWERS):
    print(power, "max motion", max_motion[i], "outer", outer_motion[i], "min |f'|", minimum_derivative[i], "min/max curvature", minimum_curvature[i], maximum_curvature[i], "negative fraction", negative_fraction[i])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].plot(POWERS, max_motion, marker="o", label="rod displacement")
axes[0, 0].semilogy(POWERS, outer_motion, marker="o", label="enclosure displacement")
axes[0, 0].axhline(0.15, color="black", linestyle="--", label="0.15 r0 motion budget")
axes[0, 0].set(xlabel="hidden-pole order", ylabel="maximum displacement/r0", title="Locality improves rapidly with pole order")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(POWERS, minimum_derivative, marker="o")
axes[0, 1].axhline(0.0, color="black")
axes[0, 1].set(xlabel="hidden-pole order", ylabel="minimum sampled |f'|", title="Conformal-map injectivity margin")

axes[0, 2].plot(POWERS, minimum_curvature, marker="o", label="minimum signed curvature")
axes[0, 2].plot(POWERS, maximum_curvature, marker="o", label="maximum signed curvature")
axes[0, 2].axhline(0.0, color="black", linewidth=1)
axes[0, 2].set(xlabel="hidden-pole order", ylabel="curvature [1/r0]", title="The price is boundary curvature")
axes[0, 2].legend(fontsize=8)

colors = {8: "tab:orange", 10: "tab:blue", 14: "tab:green"}
axes[1, 0].plot(right_reference.real, right_reference.imag, color="grey", label="circle")
for power, color in colors.items():
    axes[1, 0].plot(curves[power].real, curves[power].imag, color=color, label=f"power {power}")
axes[1, 0].scatter([1.0], [0.0], color="magenta", s=25, label="fixed inner tip")
axes[1, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Right-electrode shapes")
axes[1, 0].legend(fontsize=8)

for power, color in colors.items():
    axes[1, 1].plot(THETA / np.pi, curvatures[power], color=color, label=f"power {power}")
axes[1, 1].axhline(0.0, color="black", linewidth=1)
axes[1, 1].set(xlabel="local rod angle/pi", ylabel="signed curvature [1/r0]", title="Localized ripples create concave segments")
axes[1, 1].legend(fontsize=8)

scatter = axes[1, 2].scatter(max_motion, minimum_derivative, c=maximum_curvature, cmap="viridis", s=70)
for i, power in enumerate(POWERS):
    axes[1, 2].annotate(str(power), (max_motion[i], minimum_derivative[i]), xytext=(3, 3), textcoords="offset points", fontsize=8)
axes[1, 2].set(xlabel="maximum rod displacement/r0", ylabel="minimum sampled |f'|", title="No free choice: one jet, many boundary realizations")
fig.colorbar(scatter, ax=axes[1, 2], label="maximum curvature")

fig.suptitle("Claude-tree stress test 07 — can the exact conformal electrode actually be manufactured?", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

feasible = (max_motion <= 0.15) & (minimum_derivative > 0.25) & (minimum_curvature >= 0.0)
print("strict convex candidates", POWERS[feasible])
print("CONCLUSION: multipole cancellation is identical for every power because the center jet is identical. The plot determines whether any realization also passes displacement, injectivity, and curvature gates; if none is convex, smooth combinations of pole families rather than higher pole order are the next move.")
print("saved", OUTPUT)
