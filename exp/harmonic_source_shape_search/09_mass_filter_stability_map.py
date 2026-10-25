import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/harmonic_source_shape_search/results/12_mass_filter_stability_map.png"
COUNT = 64
SOURCE_FRACTION = 0.70
INITIAL_RADIUS = 0.50
RF_PERIODS = 30
DT = 0.04


def local_support(radius, c4, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    p = radius * (1.0 + c4 * np.cos(4.0 * theta))
    dp = -4.0 * radius * c4 * np.sin(4.0 * theta)
    return np.column_stack([p * np.cos(theta) - dp * np.sin(theta), p * np.sin(theta) + dp * np.cos(theta)])


def solve(radius, c4):
    local_boundary = local_support(radius, c4, 3 * COUNT)
    local_sources = SOURCE_FRACTION * local_support(radius, c4, COUNT, 0.013)
    distance = 1.0 + radius * (1.0 + c4)
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
        sources.append(center + local_sources @ rotation.T)
        boundary.append(center + local_boundary @ rotation.T)
        values.append(np.full(3 * COUNT, voltage))
    source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    sources.append(6.5 * np.column_stack([np.cos(source_theta), np.sin(source_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(boundary_theta), np.sin(boundary_theta)]))
    values.append(np.zeros(12 * COUNT))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    matrix = np.column_stack([
        np.log(np.linalg.norm(boundary[:, None, :] - sources[None, :, :], axis=2)),
        np.ones(len(boundary)),
    ])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def potential(points, sources, coefficients):
    distance = np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)
    return np.log(distance) @ coefficients[:-1] + coefficients[-1]


def harmonic_coefficients(radius, c4):
    sources, coefficients = solve(radius, c4)
    theta = 2.0 * np.pi * np.arange(1440) / 1440
    probe_radius = 0.5
    points = probe_radius * np.column_stack([np.cos(theta), np.sin(theta)])
    values = potential(points, sources, coefficients)
    orders = np.arange(2, 31, 4)
    amplitudes = np.array([2.0 * np.mean(values * np.cos(order * theta)) / probe_radius**order for order in orders])
    return orders, amplitudes


def q6(radius):
    orders, amplitudes = harmonic_coefficients(radius, 0.0)
    return amplitudes[orders == 6][0] / amplitudes[orders == 2][0] * 0.5**4


def normalized_field(z, orders, amplitudes):
    result = np.zeros_like(z, dtype=complex)
    for order, amplitude in zip(orders, amplitudes):
        result += order * amplitude * np.conj(z ** (order - 1))
    return -result / (2.0 * amplitudes[0])


def stability_map(orders, amplitudes, a_values, q_values):
    aa, qq = np.meshgrid(a_values, q_values, indexing="ij")
    base_a = aa.ravel()
    base_q = qq.ravel()
    angles = np.array([0.0, np.pi / 8.0, np.pi / 4.0])
    phases = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    scenario_angles, scenario_phases = np.meshgrid(angles, phases, indexing="ij")
    scenario_angles = scenario_angles.ravel()
    scenario_phases = scenario_phases.ravel()
    a = np.repeat(base_a, len(scenario_angles))
    q = np.repeat(base_q, len(scenario_angles))
    angle = np.tile(scenario_angles, len(base_a))
    phase = np.tile(scenario_phases, len(base_a))
    z = INITIAL_RADIUS * np.exp(1j * angle)
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    steps = int(RF_PERIODS * np.pi / DT)
    time = 0.0
    clipped = np.minimum(np.abs(z), 1.1) * np.exp(1j * np.angle(z))
    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(clipped, orders, amplitudes)
    for _ in range(steps):
        z[alive] += DT * velocity[alive] + 0.5 * DT**2 * acceleration[alive]
        time += DT
        alive &= np.abs(z) < 1.0
        clipped = np.minimum(np.abs(z), 1.1) * np.exp(1j * np.angle(z))
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(clipped, orders, amplitudes)
        velocity[alive] += 0.5 * DT * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    return alive.reshape(len(base_a), len(scenario_angles)).mean(axis=1).reshape(len(a_values), len(q_values))


print("HYPOTHESIS: at finite amplitude, the dual-zero shape will track the ideal Mathieu stability region more closely than either circular geometry, but the literature-compensated R/r0=1.13 circle may still be competitive near the mass-filter stability tip.")

q6_zero_radius = brentq(q6, 1.0, 1.3, xtol=1e-11)
geometries = {
    "ideal quadrupole": (np.array([2]), np.array([0.5])),
    "circle R/r0=1.13": harmonic_coefficients(1.13, 0.0),
    "circle q6=0": harmonic_coefficients(q6_zero_radius, 0.0),
    "support q6=q10=0": harmonic_coefficients(1.630529854, 0.026949792),
}

a_values = np.linspace(0.17, 0.29, 23)
q_values = np.linspace(0.56, 0.82, 25)
maps = {}
for name, (orders, amplitudes) in geometries.items():
    maps[name] = stability_map(orders, amplitudes, a_values, q_values)
    print(f"{name}: mean acceptance={maps[name].mean():.4f}, fully accepted cells={np.mean(maps[name] == 1.0):.4f}")

ideal = maps["ideal quadrupole"]
for name in list(geometries)[1:]:
    print(f"{name}: mean absolute map disagreement from ideal={np.mean(np.abs(maps[name] - ideal)):.5f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [q_values[0], q_values[-1], a_values[0], a_values[-1]]
for ax, name in zip(axes.ravel()[:4], geometries):
    image = ax.imshow(maps[name], origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.plot(q_values, (0.237 / 0.706) * q_values, color="white", linestyle="--", linewidth=1.2, label="tip scan a/q")
    ax.set(xlabel="Mathieu q", ylabel="Mathieu a", title=name)
    ax.legend(fontsize=8)

for ax, name in zip(axes.ravel()[4:], ["circle R/r0=1.13", "support q6=q10=0"]):
    difference = maps[name] - ideal
    difference_image = ax.imshow(difference, origin="lower", aspect="auto", extent=extent, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.plot(q_values, (0.237 / 0.706) * q_values, color="black", linestyle="--", linewidth=1.2)
    ax.set(xlabel="Mathieu q", ylabel="Mathieu a", title=f"{name} minus ideal")

fig.colorbar(image, ax=axes[:2, :2], label="surviving fraction")
fig.colorbar(difference_image, ax=axes[:, 2], label="acceptance difference")
fig.suptitle(f"Plot 12 — finite-amplitude mass-filter stability stress test ({RF_PERIODS} RF periods, initial radius {INITIAL_RADIUS:.2f} r0)", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

best_name = min(list(geometries)[1:], key=lambda name: np.mean(np.abs(maps[name] - ideal)))
print(f"CONCLUSION: {best_name} is closest to the ideal map in this finite-amplitude, finite-duration test. This is a stability-map experiment, not yet a resolution or transmission calculation with an injected beam.")
print("saved", OUTPUT)
