import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/claude_tree_stress_test/results/09_recursive_conformal_cancellation.png"
RADIUS_GUESS = (1.05, 1.25)
COUNT = 56
OUTER_RADIUS = 5.0
PROBE_RADIUS = 0.5
P9 = np.arange(9, 19)
W9 = np.array([0.14526537, -0.10620197, 0.28421932, -0.42300312, -0.77440244, 14.74465186, -52.90593862, 91.26916034, -77.73953184, 26.50578109])
W9 /= np.sum(W9)
P13 = np.arange(16, 29)
W13 = np.array([6.78065474, -42.50938914, 128.37796771, -226.14901113, 228.63499372, -81.83172425, -89.9305892, 123.78893861, -34.42561199, -82.46601008, 164.01436013, -139.11150475, 45.82692563])
W13 /= np.sum(W13)


def solve_circle(radius):
    tb = 2.0 * np.pi * np.arange(3 * COUNT) / (3 * COUNT)
    ts = 2.0 * np.pi * np.arange(COUNT) / COUNT + 0.013
    distance = 1.0 + radius
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    volts = np.array([0.5, 0.5, -0.5, -0.5])
    boundary = np.concatenate([center + radius * np.exp(1j * tb) for center in centers])
    sources = np.concatenate([center + 0.70 * radius * np.exp(1j * ts) for center in centers])
    values = np.concatenate([np.full(len(tb), voltage) for voltage in volts])
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


theta_probe = 2.0 * np.pi * np.arange(2048) / 2048
orders = np.arange(2, 39, 4)


def amplitudes(values, radius=PROBE_RADIUS):
    return np.array([2.0 * np.mean(values * np.cos(order * theta_probe)) / radius**order for order in orders])


def reference_amplitudes(radius):
    sources, coefficients, *_ = solve_circle(radius)
    z = PROBE_RADIUS * np.exp(1j * theta_probe)
    return amplitudes(potential(z, sources, coefficients))


radius = brentq(lambda value: reference_amplitudes(value)[1], *RADIUS_GUESS, xtol=1e-12)
sources, source_coefficients, boundary, centers = solve_circle(radius)
distance = 1.0 + radius
reference_a = reference_amplitudes(radius)
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


b9_z13 = np.sum(W9 * (P9 / distance**4 - 1.0))
target_kappa13 = reference_a[3] / (2.0 * reference_a[0])
delta13 = target_kappa13 - kappa9 * b9_z13


def g9(z):
    return sum(weight * basis(z, 9, power) for weight, power in zip(W9, P9))


def dg9(z):
    return sum(weight * basis_derivative(z, 9, power) for weight, power in zip(W9, P9))


def g13(z):
    return sum(weight * basis(z, 13, power) for weight, power in zip(W13, P13))


def dg13(z):
    return sum(weight * basis_derivative(z, 13, power) for weight, power in zip(W13, P13))


def map_q10(z):
    return z + kappa9 * g9(z)


def derivative_q10(z):
    return 1.0 + kappa9 * dg9(z)


def map_q10_q14(z):
    return z + kappa9 * g9(z) + delta13 * g13(z)


def derivative_q10_q14(z):
    return 1.0 + kappa9 * dg9(z) + delta13 * dg13(z)


def invert(w, mapping, derivative):
    z = w.copy()
    for _ in range(15):
        z -= (mapping(z) - w) / derivative(z)
    return z


physical_circle = PROBE_RADIUS * np.exp(1j * theta_probe)
preimage_10 = invert(physical_circle, map_q10, derivative_q10)
preimage_14 = invert(physical_circle, map_q10_q14, derivative_q10_q14)
a10_shape = amplitudes(potential(preimage_10, sources, source_coefficients))
a14_shape = amplitudes(potential(preimage_14, sources, source_coefficients))


def relative(a):
    return a / a[0] * PROBE_RADIUS ** (orders - 2)


q_reference = relative(reference_a)
q10_shape = relative(a10_shape)
q14_shape = relative(a14_shape)


def curvature(curve):
    step = 2.0 * np.pi / len(curve)
    first = (np.roll(curve, -1) - np.roll(curve, 1)) / (2.0 * step)
    second = (np.roll(curve, -1) - 2.0 * curve + np.roll(curve, 1)) / step**2
    return np.imag(np.conj(first) * second) / np.maximum(np.abs(first) ** 3, 1e-15)


theta = np.linspace(0.0, 2.0 * np.pi, 12000, endpoint=False)
right_reference = distance + radius * np.exp(1j * theta)
right_10 = map_q10(right_reference)
right_14 = map_q10_q14(right_reference)
curvature_10 = curvature(right_10)
curvature_14 = curvature(right_14)

axis = np.linspace(-5.0, 5.0, 501)
xx, yy = np.meshgrid(axis, axis)
sample = (xx + 1j * yy).ravel()
vacuum = np.abs(sample) < OUTER_RADIUS
for center in centers:
    vacuum &= np.abs(sample - center) > radius
sample = sample[vacuum][::8]

print("HYPOTHESIS: center jets form a triangular control system: a z9 map jet cancels a10 and an independent z13 jet then cancels a14 without changing a2, a6, or a10.")
print("kappa9", kappa9, "effective inherited kappa13", kappa9 * b9_z13, "added delta13", delta13, "target total", target_kappa13)
print("reference q", q_reference[:6])
print("q10-cancelled q", q10_shape[:6])
print("q10-q14-cancelled q", q14_shape[:6])
print("q10-q14 max rod motion", np.max(np.abs(right_14 - right_reference)))
print("q10-q14 curvature min/max", np.min(curvature_14), np.max(curvature_14))
print("q10-q14 minimum sampled |f prime|", np.min(np.abs(derivative_q10_q14(sample))))
outer = OUTER_RADIUS * np.exp(1j * theta)
print("q10-q14 maximum enclosure motion", np.max(np.abs(map_q10_q14(outer) - outer)))

radii = np.linspace(0.05, 0.95, 300)


def tail(a):
    return np.sqrt(np.sum((a[1:, None] * radii[None, :] ** orders[1:, None]) ** 2, axis=0)) / (abs(a[0]) * radii**2)


fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].plot(right_reference.real, right_reference.imag, color="grey", label="circle")
axes[0, 0].plot(right_10.real, right_10.imag, label="cancel q10")
axes[0, 0].plot(right_14.real, right_14.imag, label="cancel q10 and q14")
axes[0, 0].scatter([1.0], [0.0], color="magenta", s=25, label="fixed inner tip")
axes[0, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Recursive center-jet electrode family")
axes[0, 0].legend(fontsize=8)

for q, label in [(q_reference, "circle"), (q10_shape, "q10 cancelled"), (q14_shape, "q10 and q14 cancelled")]:
    axes[0, 1].semilogy(orders[1:8], np.maximum(np.abs(q[1:8]), 1e-18), marker="o", label=label)
axes[0, 1].set(xlabel="harmonic order", ylabel="absolute relative coefficient at r=0.5", title="Full-field multipole verification")
axes[0, 1].legend(fontsize=8)

axes[0, 2].semilogy(radii, tail(reference_a), label="circle")
axes[0, 2].semilogy(radii, tail(a10_shape), label="q10 cancelled")
axes[0, 2].semilogy(radii, tail(a14_shape), label="q10 and q14 cancelled")
axes[0, 2].set(xlabel="radius/r0", ylabel="RMS tail / quadrupole", title="Finite-amplitude field hierarchy")
axes[0, 2].legend(fontsize=8)

axes[1, 0].plot(theta / np.pi, curvature_10, label="q10 cancelled")
axes[1, 0].plot(theta / np.pi, curvature_14, label="q10 and q14 cancelled")
axes[1, 0].axhline(0.0, color="black", linewidth=1)
axes[1, 0].set(xlabel="local rod angle/pi", ylabel="signed curvature [1/r0]", title="Convexity after the second jet")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(theta / np.pi, np.abs(right_10 - right_reference), label="q10 cancelled")
axes[1, 1].plot(theta / np.pi, np.abs(right_14 - right_reference), label="q10 and q14 cancelled")
axes[1, 1].set(xlabel="local rod angle/pi", ylabel="boundary displacement/r0", title="Small physical deformation, large raw basis weights")
axes[1, 1].legend(fontsize=8)

axes[1, 2].bar(["W9", "W13"], [np.linalg.norm(W9), np.linalg.norm(W13)])
axes[1, 2].set_yscale("log")
axes[1, 2].set(ylabel="raw coefficient norm", title="The remaining problem is basis conditioning")

fig.suptitle("Claude-tree stress test 09 — recursive exact cancellation through conformal center jets", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: the full pulled-back MFS field tests whether the triangular center-jet derivation survives the rational realization. If q10 and q14 vanish while convexity and |f'| remain positive, this is a concrete exact-physics shape architecture; orthogonalization and independent fixed-enclosure validation are the next gates.")
print("saved", OUTPUT)
