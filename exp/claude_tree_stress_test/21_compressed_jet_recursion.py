import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/claude_tree_stress_test/results/21_compressed_jet_recursion.png"
RADIUS = 1.1451090300780968
DISTANCE = 1.0 + RADIUS
OUTER_RADIUS = 5.0
KAPPA9 = -0.0012213401059591876
FULL_DELTA13 = -0.0003875957659400217
P9 = np.arange(9, 19)
W9 = np.array([0.14526537, -0.10620197, 0.28421932, -0.42300312, -0.77440244, 14.74465186, -52.90593862, 91.26916034, -77.73953184, 26.50578109])
W9 /= np.sum(W9)
P13 = np.arange(16, 29)
W13 = np.array([6.78065474, -42.50938914, 128.37796771, -226.14901113, 228.63499372, -81.83172425, -89.9305892, 123.78893861, -34.42561199, -82.46601008, 164.01436013, -139.11150475, 45.82692563])
W13 /= np.sum(W13)
ORDERS = np.arange(2, 31, 4)
PROBE_RADIUS = 0.5


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


def compressed_weights(order, powers, target, rank):
    curves = np.column_stack([basis(right_fit, order, power) for power in powers])
    matrix = np.vstack([curves.real, curves.imag]) / np.sqrt(len(right_fit))
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    coordinate = (singular * (vh @ target))[:rank]
    jet_functional = (vh[:rank] @ np.ones(len(powers))) / singular[:rank]
    coordinate += jet_functional * (1.0 - jet_functional @ coordinate) / (jet_functional @ jet_functional)
    return vh[:rank].T @ (coordinate / singular[:rank])


C9 = compressed_weights(9, P9, W9, 9)
C13 = compressed_weights(13, P13, W13, 9)


def mapping(z, weights9, weights13, delta13):
    ninth = sum(weight * basis(z, 9, power) for weight, power in zip(weights9, P9))
    thirteenth = sum(weight * basis(z, 13, power) for weight, power in zip(weights13, P13))
    return z + KAPPA9 * ninth + delta13 * thirteenth


def mapping_derivative(z, weights9, weights13, delta13):
    ninth = sum(weight * basis_derivative(z, 9, power) for weight, power in zip(weights9, P9))
    thirteenth = sum(weight * basis_derivative(z, 13, power) for weight, power in zip(weights13, P13))
    return 1.0 + KAPPA9 * ninth + delta13 * thirteenth


def inverse_mapping(w, weights9, weights13, delta13):
    z = w.copy()
    for _ in range(18):
        z -= (mapping(z, weights9, weights13, delta13) - w) / mapping_derivative(z, weights9, weights13, delta13)
    return z


centers = DISTANCE * np.array([1.0, -1.0, 1.0j, -1.0j])
voltages = np.array([0.5, 0.5, -0.5, -0.5])


def solve_circle(count=88):
    tb = 2.0 * np.pi * np.arange(3 * count) / (3 * count)
    ts = 2.0 * np.pi * np.arange(count) / count + 0.013
    boundary = np.concatenate([center + RADIUS * np.exp(1j * tb) for center in centers])
    sources = np.concatenate([center + 0.70 * RADIUS * np.exp(1j * ts) for center in centers])
    values = np.concatenate([np.full(len(tb), voltage) for voltage in voltages])
    to = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    tso = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * to)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * tso)])
    values = np.concatenate([values, np.zeros(len(to))])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def solve_direct(weights9, weights13, delta13, count=88):
    tb = 2.0 * np.pi * np.arange(3 * count) / (3 * count)
    ts = 2.0 * np.pi * np.arange(count) / count + 0.011
    curves = [mapping(center + RADIUS * np.exp(1j * tb), weights9, weights13, delta13) for center in centers]
    source_curves = [mapping(center + RADIUS * np.exp(1j * ts), weights9, weights13, delta13) for center in centers]
    boundary = np.concatenate(curves)
    sources = np.concatenate([center + 0.62 * (curve - center) for center, curve in zip(centers, source_curves)])
    values = np.concatenate([np.full(len(tb), voltage) for voltage in voltages])
    to = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    tso = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * to)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * tso)])
    values = np.concatenate([values, np.zeros(len(to))])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def potential(z, sources, coefficients):
    return np.log(np.abs(z[:, None] - sources[None, :])) @ coefficients[:-1] + coefficients[-1]


sources, coefficients = solve_circle()
theta = 2.0 * np.pi * np.arange(2048) / 2048
probe = PROBE_RADIUS * np.exp(1j * theta)


def amplitudes(values):
    return np.array([2.0 * np.mean(values * np.cos(order * theta)) / PROBE_RADIUS**order for order in ORDERS])


def pullback_amplitudes(weights9, weights13, delta13):
    reference = inverse_mapping(probe, weights9, weights13, delta13)
    return amplitudes(potential(reference, sources, coefficients))


def relative(amplitude):
    return amplitude / amplitude[0] * PROBE_RADIUS ** (ORDERS - 2)


full_a = pullback_amplitudes(W9, W13, FULL_DELTA13)
compressed_old_a = pullback_amplitudes(C9, C13, FULL_DELTA13)
retuned_delta13 = brentq(lambda delta: pullback_amplitudes(C9, C13, delta)[3], -0.005, 0.005, xtol=1e-14)
compressed_new_a = pullback_amplitudes(C9, C13, retuned_delta13)
direct_sources, direct_coefficients = solve_direct(C9, C13, retuned_delta13)
direct_a = amplitudes(potential(probe, direct_sources, direct_coefficients))

full_q = relative(full_a)
compressed_old_q = relative(compressed_old_a)
compressed_new_q = relative(compressed_new_a)
direct_q = relative(direct_a)

shape_theta = 2.0 * np.pi * np.arange(8192) / 8192
right = DISTANCE + RADIUS * np.exp(1j * shape_theta)
full_curve = mapping(right, W9, W13, FULL_DELTA13)
retuned_curve = mapping(right, C9, C13, retuned_delta13)


def curvature(curve):
    step = 2.0 * np.pi / len(curve)
    first = (np.roll(curve, -1) - np.roll(curve, 1)) / (2.0 * step)
    second = (np.roll(curve, -1) - 2.0 * curve + np.roll(curve, 1)) / step**2
    return np.imag(np.conj(first) * second) / np.maximum(np.abs(first) ** 3, 1e-15)


delta_scan = np.linspace(retuned_delta13 - 0.001, retuned_delta13 + 0.001, 81)
q14_scan = np.array([relative(pullback_amplitudes(C9, C13, delta))[3] for delta in delta_scan])

print("HYPOTHESIS: SVD compression preserves the leading jet but changes downstream Taylor coefficients, so q14 must be re-solved recursively after compression.")
print("full q", full_q[:6])
print("compressed with old delta q", compressed_old_q[:6])
print("old delta", FULL_DELTA13, "retuned delta", retuned_delta13)
print("compressed retuned q", compressed_new_q[:6])
print("independent direct q", direct_q[:6])
print("retuned minimum curvature", np.min(curvature(retuned_curve)), "max boundary difference from full", np.max(np.abs(retuned_curve - full_curve)))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for q, label, color in [(full_q, "full basis", "black"), (compressed_old_q, "compressed, old delta13", "tab:orange"), (compressed_new_q, "compressed, retuned", "tab:red")]:
    axes[0, 0].semilogy(ORDERS[1:6], np.maximum(np.abs(q[1:6]), 1e-18), marker="o", color=color, label=label)
axes[0, 0].set(xlabel="harmonic order", ylabel="absolute relative coefficient", title="Compression couples downstream jets")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(delta_scan, q14_scan)
axes[0, 1].axhline(0.0, color="black", linewidth=1)
axes[0, 1].axvline(FULL_DELTA13, color="tab:orange", linestyle="--", label="old delta13")
axes[0, 1].axvline(retuned_delta13, color="tab:red", linestyle="--", label="re-solved delta13")
axes[0, 1].set(xlabel="delta13", ylabel="relative q14", title="One scalar recursive correction restores q14")
axes[0, 1].legend(fontsize=8)

axes[0, 2].plot(full_curve.real, full_curve.imag, color="black", label="full basis")
axes[0, 2].plot(retuned_curve.real, retuned_curve.imag, linestyle="--", color="tab:red", label="compressed + retuned")
axes[0, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Physical boundary remains close")
axes[0, 2].legend(fontsize=8)

axes[1, 0].plot(shape_theta / np.pi, curvature(full_curve), color="black", label="full basis")
axes[1, 0].plot(shape_theta / np.pi, curvature(retuned_curve), linestyle="--", color="tab:red", label="compressed + retuned")
axes[1, 0].axhline(0.0, color="grey", linewidth=1)
axes[1, 0].set(xlabel="local rod angle/pi", ylabel="signed curvature", title="Convexity after recursive correction")
axes[1, 0].legend(fontsize=8)

for q, label, color in [(compressed_new_q, "conformal pullback", "tab:red"), (direct_q, "independent fixed enclosure", "tab:blue")]:
    axes[1, 1].semilogy(ORDERS[1:6], np.maximum(np.abs(q[1:6]), 1e-18), marker="o", color=color, label=label)
axes[1, 1].set(xlabel="harmonic order", ylabel="absolute relative coefficient", title="Independent physical field check")
axes[1, 1].legend(fontsize=8)

axes[1, 2].plot(shape_theta / np.pi, np.abs(retuned_curve - full_curve))
axes[1, 2].set(xlabel="local rod angle/pi", ylabel="boundary difference/r0", title="Cost of dropping five SVD directions")

fig.suptitle("Claude-tree stress test 21 — compressed modes need triangular jet recursion", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: orthogonal boundary coordinates are viable, but the architecture must solve jet coefficients in ascending order. Independent per-order normalization is insufficient because lower-order shape modes carry higher Taylor jets.")
print("saved", OUTPUT)
