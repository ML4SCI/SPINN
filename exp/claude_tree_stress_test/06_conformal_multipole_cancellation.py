import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/claude_tree_stress_test/results/06_conformal_multipole_cancellation.png"
COUNT = 64
SOURCE_FRACTION = 0.70
OUTER_RADIUS = 5.0
PROBE_RADIUS = 0.5
POLE_POWER = 10


def circle_geometry(radius):
    theta_boundary = 2.0 * np.pi * np.arange(3 * COUNT) / (3 * COUNT)
    theta_source = 2.0 * np.pi * np.arange(COUNT) / COUNT + 0.013
    local_boundary = radius * np.exp(1j * theta_boundary)
    local_sources = SOURCE_FRACTION * radius * np.exp(1j * theta_source)
    distance = 1.0 + radius
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    boundary = np.concatenate([center + local_boundary for center in centers])
    sources = np.concatenate([center + local_sources for center in centers])
    values = np.concatenate([np.full(len(local_boundary), voltage) for voltage in voltages])
    outer_boundary_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    outer_source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * outer_boundary_theta)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * outer_source_theta)])
    values = np.concatenate([values, np.zeros(len(outer_boundary_theta))])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients, boundary, values, centers


def potential(z, sources, coefficients):
    return np.log(np.abs(z[:, None] - sources[None, :])) @ coefficients[:-1] + coefficients[-1]


probe_theta = 2.0 * np.pi * np.arange(2048) / 2048


def coefficients_on_circle(values, radius, orders):
    return np.array([2.0 * np.mean(values * np.cos(order * probe_theta)) / radius**order for order in orders])


def circle_multipoles(radius):
    sources, coefficients, *_ = circle_geometry(radius)
    z = PROBE_RADIUS * np.exp(1j * probe_theta)
    amplitudes = coefficients_on_circle(potential(z, sources, coefficients), PROBE_RADIUS, np.array([2, 6, 10]))
    return amplitudes


radius = brentq(lambda value: circle_multipoles(value)[1], 1.05, 1.25, xtol=1e-12)
sources, source_coefficients, reference_boundary, boundary_values, centers = circle_geometry(radius)
orders = np.arange(2, 35, 4)
reference_probe = PROBE_RADIUS * np.exp(1j * probe_theta)
reference_amplitudes = coefficients_on_circle(potential(reference_probe, sources, source_coefficients), PROBE_RADIUS, orders)
a2 = reference_amplitudes[0]
a10 = reference_amplitudes[2]
kappa9 = a10 / (2.0 * a2)
distance = 1.0 + radius


def raw_map(z):
    return z**9 * (z**4 - 1.0) / (z**4 - distance**4) ** POLE_POWER


leading = (-1.0) / ((-distance**4) ** POLE_POWER)


def basis(z):
    return raw_map(z) / leading


def basis_derivative(z):
    denominator = z**4 - distance**4
    first = (9.0 * z**8 * (z**4 - 1.0) + 4.0 * z**12) / denominator**POLE_POWER
    second = POLE_POWER * 4.0 * z**12 * (z**4 - 1.0) / denominator ** (POLE_POWER + 1)
    return (first - second) / leading


def mapping(z):
    return z + kappa9 * basis(z)


def mapping_derivative(z):
    return 1.0 + kappa9 * basis_derivative(z)


def inverse_mapping(w):
    z = w.copy()
    for _ in range(12):
        z -= (mapping(z) - w) / mapping_derivative(z)
    return z


physical_probe = PROBE_RADIUS * np.exp(1j * probe_theta)
preimage_probe = inverse_mapping(physical_probe)
physical_values = potential(preimage_probe, sources, source_coefficients)
physical_amplitudes = coefficients_on_circle(physical_values, PROBE_RADIUS, orders)

reference_q = reference_amplitudes / a2 * PROBE_RADIUS ** (orders - 2)
physical_q = physical_amplitudes / physical_amplitudes[0] * PROBE_RADIUS ** (orders - 2)

rod_count = 4 * 3 * COUNT
mapped_boundary = mapping(reference_boundary)
rod_motion = np.abs(mapped_boundary[:rod_count] - reference_boundary[:rod_count])
outer_motion = np.abs(mapped_boundary[rod_count:] - reference_boundary[rod_count:])

axis = np.linspace(-5.0, 5.0, 501)
xx, yy = np.meshgrid(axis, axis)
zz = xx + 1j * yy
vacuum = np.abs(zz) < OUTER_RADIUS
for center in centers:
    vacuum &= np.abs(zz - center) > radius
sample = zz[vacuum][::20]
minimum_derivative = np.min(np.abs(mapping_derivative(sample)))

print("HYPOTHESIS: a tip-anchored D4 conformal map with leading term kappa9*z^9 can cancel a10 exactly while preserving r0, a2, and a6; the pulled-back full field should confirm the local series prediction.")
print("reference radius", radius)
print("reference a2,a6,a10", reference_amplitudes[:3])
print("predicted kappa9", kappa9)
print("physical a2,a6,a10", physical_amplitudes[:3])
print("maximum rod displacement", np.max(rod_motion))
print("maximum enclosure displacement", np.max(outer_motion))
print("minimum sampled |f prime|", minimum_derivative)
print("four inner-tip displacements", np.abs(mapping(np.array([1.0, -1.0, 1.0j, -1.0j])) - np.array([1.0, -1.0, 1.0j, -1.0j])))

radii = np.linspace(0.05, 0.95, 300)
reference_tail = np.sqrt(np.sum((reference_amplitudes[1:, None] * radii[None, :] ** orders[1:, None]) ** 2, axis=0)) / (abs(reference_amplitudes[0]) * radii**2)
physical_tail = np.sqrt(np.sum((physical_amplitudes[1:, None] * radii[None, :] ** orders[1:, None]) ** 2, axis=0)) / (abs(physical_amplitudes[0]) * radii**2)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for index in range(4):
    segment = slice(index * 3 * COUNT, (index + 1) * 3 * COUNT)
    axes[0, 0].plot(reference_boundary[segment].real, reference_boundary[segment].imag, color="grey", linewidth=1, label="circular reference" if index == 0 else None)
    axes[0, 0].plot(mapped_boundary[segment].real, mapped_boundary[segment].imag, color="tab:blue", label="conformal a10 cancellation" if index == 0 else None)
axes[0, 0].scatter([1, -1, 0, 0], [0, 0, 1, -1], color="magenta", s=22, label="fixed inner tips")
axes[0, 0].set(aspect="equal", xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="x/r0", ylabel="y/r0", title="A new exact-harmonic electrode family")
axes[0, 0].legend(fontsize=8)

shown = np.arange(1, 7)
axes[0, 1].semilogy(orders[shown], np.maximum(np.abs(reference_q[shown]), 1e-18), marker="o", label="circle")
axes[0, 1].semilogy(orders[shown], np.maximum(np.abs(physical_q[shown]), 1e-18), marker="o", label="conformal shape")
axes[0, 1].set_xticks(orders[shown])
axes[0, 1].set(xlabel="harmonic order", ylabel="absolute coefficient relative to quadrupole at r=0.5", title="Full pulled-back field, not only the series")
axes[0, 1].legend()

axes[0, 2].semilogy(radii, reference_tail, label="circular q6=0")
axes[0, 2].semilogy(radii, physical_tail, label="conformal q6=q10=0")
axes[0, 2].set(xlabel="radius/r0", ylabel="RMS multipole tail / quadrupole", title="Finite-amplitude field nonlinearity")
axes[0, 2].legend()

local_theta = 2.0 * np.pi * np.arange(3 * COUNT) / (3 * COUNT)
right = slice(0, 3 * COUNT)
normal_motion = np.real((mapped_boundary[right] - reference_boundary[right]) * np.exp(-1j * local_theta))
tangential_motion = np.imag((mapped_boundary[right] - reference_boundary[right]) * np.exp(-1j * local_theta))
axes[1, 0].plot(local_theta / np.pi, normal_motion, label="normal")
axes[1, 0].plot(local_theta / np.pi, tangential_motion, label="tangential")
axes[1, 0].set(xlabel="right-rod local angle/pi", ylabel="boundary displacement/r0", title="What physical rod deformation is required")
axes[1, 0].legend()

outer_reference = reference_boundary[rod_count:]
axes[1, 1].plot(np.angle(outer_reference), outer_motion)
axes[1, 1].set(xlabel="enclosure angle", ylabel="enclosure displacement/r0", title="Outer boundary is nearly, not exactly, fixed")

scales = np.linspace(0.0, 1.35, 100)
minimum_derivatives = []
maximum_motions = []
for scale in scales:
    minimum_derivatives.append(np.min(np.abs(1.0 + scale * kappa9 * basis_derivative(sample))))
    maximum_motions.append(np.max(np.abs(scale * kappa9 * basis(reference_boundary[:rod_count]))))
axes[1, 2].plot(maximum_motions, minimum_derivatives)
axes[1, 2].scatter([np.max(rod_motion)], [minimum_derivative], color="red", label="a10 cancellation")
axes[1, 2].axhline(0.0, color="black", linewidth=1)
axes[1, 2].set(xlabel="maximum rod displacement/r0", ylabel="minimum sampled |f'|", title="Injectivity margin along the conformal mode")
axes[1, 2].legend()

fig.suptitle("Claude-tree stress test 06 — exact conformal cancellation of the circular trap's next multipole", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: if the pulled-back a10 is zero with unchanged a2/a6 and positive |f'|, this is a viable exact-harmonic shape coordinate. It improves on a generic coordinate network by encoding Laplace physics, D4 symmetry, fixed r0, and chosen center jets, but its small enclosure motion and global injectivity still need independent physical comparison.")
print("saved", OUTPUT)
