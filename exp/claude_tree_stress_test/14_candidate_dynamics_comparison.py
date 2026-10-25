import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/14_candidate_dynamics_comparison.png"
ORDERS = np.arange(2, 31, 4)
RHO = 0.5
Q_VALUES = np.linspace(0.50, 0.84, 69)
A_OVER_Q = 0.237 / 0.706
INITIAL_RADII = [0.50, 0.72]
PERIODS = 180
DT = 0.04


relative_multipoles = {
    "ideal": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "circle q6=0": np.array([1.0, 0.0, -9.54171958e-6, -6.89746563e-8, -3.34722203e-10, -1.21227111e-12, -3.29814122e-15, 0.0]),
    "support q6=q10=0": np.array([1.0, 1.85007411e-11, 1.96277608e-13, 3.84009287e-8, 8.93658962e-10, 1.77261570e-11, 3.37627025e-13, 7.47240068e-15]),
    "conformal q6=q10=q14=0": np.array([1.0, 0.0, 0.0, 0.0, -1.33795580e-8, -8.48346458e-10, -3.44416576e-11, -1.09485347e-12]),
    "synthetic beta=0.025": np.array([1.0, 0.025 * RHO**4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}


def amplitude_ratios(q):
    return q / RHO ** (ORDERS - 2)


def normalized_field(z, q):
    ratios = amplitude_ratios(q)
    result = np.zeros_like(z, dtype=complex)
    for order, ratio in zip(ORDERS, ratios):
        result += 0.5 * order * ratio * np.conj(z ** (order - 1))
    return -result


def escape_times(q_coefficients, initial_radius):
    angles = np.array([0.0, np.pi / 8.0, np.pi / 4.0])
    phases = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    angle, phase = np.meshgrid(angles, phases, indexing="ij")
    angle = angle.ravel()
    phase = phase.ravel()
    q = np.repeat(Q_VALUES, len(angle))
    a = A_OVER_Q * q
    phase = np.tile(phase, len(Q_VALUES))
    z = initial_radius * np.exp(1j * np.tile(angle, len(Q_VALUES)))
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    escaped = np.full(len(z), PERIODS, dtype=float)
    steps = int(PERIODS * np.pi / DT)
    time = 0.0
    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(z, q_coefficients)
    for step in range(steps):
        z[alive] += DT * velocity[alive] + 0.5 * DT**2 * acceleration[alive]
        time += DT
        newly_dead = alive & (np.abs(z) >= 1.0)
        escaped[newly_dead] = step * DT / np.pi
        alive[newly_dead] = False
        clipped = np.minimum(np.abs(z), 1.0) * np.exp(1j * np.angle(z))
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(clipped, q_coefficients)
        velocity[alive] += 0.5 * DT * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    return escaped.reshape(len(Q_VALUES), len(angle))


print("HYPOTHESIS: the support and conformal multipole-cancelled shapes should track ideal dynamics closely across q, while a deliberately nonzero dodecapole may win only in a narrow operating band.")
results = {}
for initial_radius in INITIAL_RADII:
    results[initial_radius] = {}
    for name, coefficients in relative_multipoles.items():
        times = escape_times(coefficients, initial_radius)
        results[initial_radius][name] = times
        print("radius", initial_radius, name, "mean escape periods", times.mean(), "full-horizon fraction", np.mean(times >= PERIODS))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
colors = {
    "ideal": "black",
    "circle q6=0": "tab:grey",
    "support q6=q10=0": "tab:green",
    "conformal q6=q10=q14=0": "tab:blue",
    "synthetic beta=0.025": "tab:red",
}
for row, initial_radius in enumerate(INITIAL_RADII):
    ideal_mean = results[initial_radius]["ideal"].mean(axis=1)
    for name, times in results[initial_radius].items():
        mean = times.mean(axis=1)
        axes[row, 0].plot(Q_VALUES, mean, color=colors[name], label=name)
        if name != "ideal":
            axes[row, 1].plot(Q_VALUES, mean - ideal_mean, color=colors[name], label=name)
    axes[row, 0].set(xlabel="Mathieu q along a/q=0.336", ylabel="mean escape periods", title=f"initial radius {initial_radius} r0")
    axes[row, 1].axhline(0.0, color="black", linewidth=1)
    axes[row, 1].set(xlabel="Mathieu q", ylabel="mean lifetime minus ideal", title="Where each field helps or hurts")
    axes[row, 0].legend(fontsize=7)
    axes[row, 1].legend(fontsize=7)

q_index = int(np.argmin(np.abs(Q_VALUES - 0.69)))
names = list(relative_multipoles)
positions = np.arange(len(names))
for offset, initial_radius in zip([-0.17, 0.17], INITIAL_RADII):
    values = [results[initial_radius][name][q_index].mean() for name in names]
    axes[0, 2].bar(positions + offset, values, 0.34, label=f"initial r={initial_radius}")
axes[0, 2].set_xticks(positions, [name.replace(" ", "\n") for name in names], rotation=20, fontsize=7)
axes[0, 2].set(ylabel="mean escape periods", title=f"Operating-point comparison at q={Q_VALUES[q_index]:.3f}")
axes[0, 2].legend(fontsize=8)

for name, q in relative_multipoles.items():
    tail = np.sqrt(np.sum((amplitude_ratios(q)[1:, None] * np.linspace(0.05, 0.95, 250)[None, :] ** ORDERS[1:, None]) ** 2, axis=0)) / np.linspace(0.05, 0.95, 250) ** 2
    axes[1, 2].semilogy(np.linspace(0.05, 0.95, 250), np.maximum(tail, 1e-18), color=colors[name], label=name)
axes[1, 2].set(xlabel="radius/r0", ylabel="RMS static tail / quadrupole", title="Static ranking does not fully predict lifetime")
axes[1, 2].legend(fontsize=7)

fig.suptitle("Claude-tree stress test 14 — existing and new electrode candidates under one dynamical objective", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

for initial_radius in INITIAL_RADII:
    ideal = results[initial_radius]["ideal"].mean(axis=1)
    for name in ["support q6=q10=0", "conformal q6=q10=q14=0"]:
        disagreement = np.mean(np.abs(results[initial_radius][name].mean(axis=1) - ideal))
        print("radius", initial_radius, name, "mean absolute lifetime disagreement from ideal", disagreement)
print("CONCLUSION: compare the full q curves rather than one best point. Near-identical support/conformal/ideal curves mean their remaining static tails are dynamically negligible in this model; a nonzero dodecapole can improve one band while degrading others, so the operating distribution must define the optimization target.")
print("saved", OUTPUT)
