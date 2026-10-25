import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/claude_tree_stress_test/results/15_physical_lifetime_band_candidate.png"
COUNT = 60
OUTER_RADIUS = 5.0
PROBE_RADIUS = 0.5
TARGET_BETA = 0.025
P9 = np.arange(9, 19)
W9 = np.array([0.14526537, -0.10620197, 0.28421932, -0.42300312, -0.77440244, 14.74465186, -52.90593862, 91.26916034, -77.73953184, 26.50578109])
W9 /= np.sum(W9)
P13 = np.arange(16, 29)
W13 = np.array([6.78065474, -42.50938914, 128.37796771, -226.14901113, 228.63499372, -81.83172425, -89.9305892, 123.78893861, -34.42561199, -82.46601008, 164.01436013, -139.11150475, 45.82692563])
W13 /= np.sum(W13)
THETA = 2.0 * np.pi * np.arange(2048) / 2048
ORDERS = np.arange(2, 35, 4)


def solve_circle(radius):
    tb = 2.0 * np.pi * np.arange(3 * COUNT) / (3 * COUNT)
    ts = 2.0 * np.pi * np.arange(COUNT) / COUNT + 0.013
    distance = 1.0 + radius
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    boundary = np.concatenate([center + radius * np.exp(1j * tb) for center in centers])
    sources = np.concatenate([center + 0.70 * radius * np.exp(1j * ts) for center in centers])
    values = np.concatenate([np.full(len(tb), voltage) for voltage in voltages])
    to = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    tso = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * to)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * tso)])
    values = np.concatenate([values, np.zeros(len(to))])
    A = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(A, values, rcond=1e-12)
    return sources, coefficients, boundary, centers


def potential(z, sources, coefficients):
    return np.log(np.abs(z[:, None] - sources[None, :])) @ coefficients[:-1] + coefficients[-1]


def amplitudes(values, radius=PROBE_RADIUS):
    return np.array([2.0 * np.mean(values * np.cos(order * THETA)) / radius**order for order in ORDERS])


def circle_amplitudes(radius):
    sources, coefficients, *_ = solve_circle(radius)
    return amplitudes(potential(PROBE_RADIUS * np.exp(1j * THETA), sources, coefficients))


def circle_beta(radius):
    a = circle_amplitudes(radius)
    return a[1] / a[0]


radius = brentq(lambda value: circle_beta(value) - TARGET_BETA, 0.70, 0.95, xtol=1e-12)
sources, source_coefficients, reference_boundary, centers = solve_circle(radius)
distance = 1.0 + radius
reference_a = circle_amplitudes(radius)
kappa9 = reference_a[2] / (2.0 * reference_a[0])


def basis(z, order, power):
    leading = (-1.0) / ((-distance**4) ** power)
    return z**order * (z**4 - 1.0) / (z**4 - distance**4) ** power / leading


def basis_derivative(z, order, power):
    leading = (-1.0) / ((-distance**4) ** power)
    denominator = z**4 - distance**4
    first = (order * z ** (order - 1) * (z**4 - 1.0) + 4.0 * z ** (order + 3)) / denominator**power
    second = power * 4.0 * z ** (order + 3) * (z**4 - 1.0) / denominator ** (power + 1)
    return (first - second) / leading


def g(z, order, powers, weights):
    return sum(weight * basis(z, order, power) for weight, power in zip(weights, powers))


def dg(z, order, powers, weights):
    return sum(weight * basis_derivative(z, order, power) for weight, power in zip(weights, powers))


def mapping(z, delta13):
    return z + kappa9 * g(z, 9, P9, W9) + delta13 * g(z, 13, P13, W13)


def mapping_derivative(z, delta13):
    return 1.0 + kappa9 * dg(z, 9, P9, W9) + delta13 * dg(z, 13, P13, W13)


def invert(w, delta13):
    z = w.copy()
    for _ in range(15):
        z -= (mapping(z, delta13) - w) / mapping_derivative(z, delta13)
    return z


def mapped_amplitudes(delta13):
    physical = PROBE_RADIUS * np.exp(1j * THETA)
    return amplitudes(potential(invert(physical, delta13), sources, source_coefficients))


delta13 = brentq(lambda value: mapped_amplitudes(value)[3], -0.005, 0.005, xtol=1e-14)
corrected_a = mapped_amplitudes(delta13)


def relative(a):
    return a / a[0] * PROBE_RADIUS ** (ORDERS - 2)


reference_q = relative(reference_a)
corrected_q = relative(corrected_a)


def curvature(curve):
    step = 2.0 * np.pi / len(curve)
    first = (np.roll(curve, -1) - np.roll(curve, 1)) / (2.0 * step)
    second = (np.roll(curve, -1) - 2.0 * curve + np.roll(curve, 1)) / step**2
    return np.imag(np.conj(first) * second) / np.maximum(np.abs(first) ** 3, 1e-15)


shape_theta = np.linspace(0.0, 2.0 * np.pi, 12000, endpoint=False)
right_reference = distance + radius * np.exp(1j * shape_theta)
right_corrected = mapping(right_reference, delta13)
shape_curvature = curvature(right_corrected)

axis = np.linspace(-5.0, 5.0, 501)
xx, yy = np.meshgrid(axis, axis)
sample = (xx + 1j * yy).ravel()
vacuum = np.abs(sample) < OUTER_RADIUS
for center in centers:
    vacuum &= np.abs(sample - center) > radius
sample = sample[vacuum][::8]

radius_scan = np.linspace(0.65, 1.22, 90)
scan_a = np.array([circle_amplitudes(value) for value in radius_scan])
scan_eta = 2.0 * scan_a[:, 0]
scan_beta = scan_a[:, 1] / scan_a[:, 0]
q6_zero_radius = brentq(circle_beta, 1.05, 1.25)
q6_zero_eta = 2.0 * circle_amplitudes(q6_zero_radius)[0]
target_eta = 2.0 * reference_a[0]

print("HYPOTHESIS: use circular radius to realize beta=a6/a2=0.025, then small ninth- and thirteenth-order conformal jets can cancel a10 and a14 without changing the lifetime-driving a6.")
print("target radius", radius, "eta", target_eta, "eta ratio versus q6-zero circle", target_eta / q6_zero_eta)
print("kappa9", kappa9, "delta13", delta13)
print("reference q", reference_q[:6])
print("corrected q", corrected_q[:6])
print("corrected beta", corrected_a[1] / corrected_a[0])
print("max rod motion", np.max(np.abs(right_corrected - right_reference)))
print("curvature min/max", np.min(shape_curvature), np.max(shape_curvature))
print("minimum sampled |f prime|", np.min(np.abs(mapping_derivative(sample, delta13))))
outer = OUTER_RADIUS * np.exp(1j * shape_theta)
print("maximum enclosure motion", np.max(np.abs(mapping(outer, delta13) - outer)))

radii = np.linspace(0.05, 0.95, 300)


def tail(a):
    return np.sqrt(np.sum((a[1:, None] * radii[None, :] ** ORDERS[1:, None]) ** 2, axis=0)) / (abs(a[0]) * radii**2)


fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].plot(scan_beta, scan_eta, color="black")
axes[0, 0].scatter([0.0], [q6_zero_eta], color="tab:blue", label="static q6-zero circle")
axes[0, 0].scatter([TARGET_BETA], [target_eta], color="tab:red", label="lifetime-band circle")
axes[0, 0].set(xlabel="circle a6/a2", ylabel="eta", title="The dynamical target has an efficiency cost")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(right_reference.real, right_reference.imag, color="grey", label="target circle")
axes[0, 1].plot(right_corrected.real, right_corrected.imag, color="tab:red", label="higher-order corrected")
axes[0, 1].scatter([1.0], [0.0], color="magenta", s=25, label="fixed inner tip")
axes[0, 1].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Physical lifetime-band electrode")
axes[0, 1].legend(fontsize=8)

for q, label, color in [(reference_q, "target circle", "grey"), (corrected_q, "corrected", "tab:red")]:
    axes[0, 2].semilogy(ORDERS[1:7], np.maximum(np.abs(q[1:7]), 1e-18), marker="o", color=color, label=label)
axes[0, 2].set(xlabel="harmonic order", ylabel="absolute relative coefficient at r=0.5", title="Keep q6; cancel q10 and q14")
axes[0, 2].legend(fontsize=8)

axes[1, 0].semilogy(radii, tail(reference_a), color="grey", label="target circle")
axes[1, 0].semilogy(radii, tail(corrected_a), color="tab:red", label="corrected")
axes[1, 0].set(xlabel="radius/r0", ylabel="RMS tail / quadrupole", title="The intended q6 remains dominant")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(shape_theta / np.pi, shape_curvature)
axes[1, 1].axhline(0.0, color="black", linewidth=1)
axes[1, 1].set(xlabel="local rod angle/pi", ylabel="signed curvature [1/r0]", title="Convexity after higher-order correction")

axes[1, 2].plot(shape_theta / np.pi, np.abs(right_corrected - right_reference))
axes[1, 2].set(xlabel="local rod angle/pi", ylabel="boundary displacement/r0", title="Correction is small relative to the radius change")

fig.suptitle("Claude-tree stress test 15 — a physical electrode for the nonzero-lifetime band", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: the target nonzero a6 can be realized by a smaller circular rod and its a10/a14 removed by small convex conformal corrections. The cost is lower eta, so equal-q dynamics require proportionally higher RF voltage; this is a Pareto candidate, not a universal replacement.")
print("saved", OUTPUT)
