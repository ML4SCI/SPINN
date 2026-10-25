import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/17_voltage_aware_dynamics.png"
ORDERS = np.arange(2, 31, 4)
RHO = 0.5
NOMINAL_Q = np.linspace(0.50, 0.84, 69)
A_OVER_Q = 0.237 / 0.706
INITIAL_RADII = [0.50, 0.72]
PERIODS = 180
DT = 0.04
ETA_RATIO = 0.9744662440131631
VOLTAGE_MULTIPLIER = 1.0 / ETA_RATIO

BASELINE = np.array([1.0, 0.0, -9.54171958e-6, -6.89746563e-8, -3.34722203e-10, -1.21227111e-12, -3.29814122e-15, 0.0])
PHYSICAL = np.array([1.0, 1.56250015e-3, -3.90109192e-12, -2.36420786e-12, -6.22013422e-9, -5.48756817e-10, 0.0, 0.0])
SYNTHETIC = np.array([1.0, 0.025 * RHO**4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def amplitude_ratios(relative_coefficients):
    return relative_coefficients / RHO ** (ORDERS - 2)


def normalized_field(z, relative_coefficients):
    result = np.zeros_like(z, dtype=complex)
    for order, ratio in zip(ORDERS, amplitude_ratios(relative_coefficients)):
        result += 0.5 * order * ratio * np.conj(z ** (order - 1))
    return -result


def escape_times(relative_coefficients, initial_radius, q_scale):
    angles = np.array([0.0, np.pi / 8.0, np.pi / 4.0])
    phases = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    angle, phase = np.meshgrid(angles, phases, indexing="ij")
    angle = angle.ravel()
    phase = phase.ravel()
    effective_q = np.repeat(q_scale * NOMINAL_Q, len(angle))
    effective_a = A_OVER_Q * effective_q
    phase = np.tile(phase, len(NOMINAL_Q))
    z = initial_radius * np.exp(1j * np.tile(angle, len(NOMINAL_Q)))
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    escaped = np.full(len(z), PERIODS, dtype=float)
    steps = int(PERIODS * np.pi / DT)
    time = 0.0
    acceleration = (effective_a - 2.0 * effective_q * np.cos(2.0 * time + phase)) * normalized_field(z, relative_coefficients)
    for step in range(steps):
        z[alive] += DT * velocity[alive] + 0.5 * DT**2 * acceleration[alive]
        time += DT
        newly_dead = alive & (np.abs(z) >= 1.0)
        escaped[newly_dead] = step * DT / np.pi
        alive[newly_dead] = False
        clipped = np.minimum(np.abs(z), 1.0) * np.exp(1j * np.angle(z))
        next_acceleration = (effective_a - 2.0 * effective_q * np.cos(2.0 * time + phase)) * normalized_field(clipped, relative_coefficients)
        velocity[alive] += 0.5 * DT * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    return escaped.reshape(len(NOMINAL_Q), len(angle))


designs = {
    "q6-zero circle, same voltage": (BASELINE, 1.0),
    "physical beta=.025, same voltage": (PHYSICAL, ETA_RATIO),
    "physical beta=.025, retuned": (PHYSICAL, 1.0),
    "synthetic beta=.025, retuned": (SYNTHETIC, 1.0),
}

print("HYPOTHESIS: some of the beta=.025 lifetime advantage may be a normalization artifact because equal Mathieu q requires 1/eta_ratio times the voltage of the q6-zero circle.")
print("eta ratio", ETA_RATIO, "required voltage multiplier", VOLTAGE_MULTIPLIER)
results = {}
for radius in INITIAL_RADII:
    results[radius] = {}
    for name, (coefficients, scale) in designs.items():
        values = escape_times(coefficients, radius, scale)
        results[radius][name] = values
        print("radius", radius, name, "mean", values.mean(), "survival fraction", np.mean(values >= PERIODS))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
colors = {
    "q6-zero circle, same voltage": "black",
    "physical beta=.025, same voltage": "tab:orange",
    "physical beta=.025, retuned": "tab:red",
    "synthetic beta=.025, retuned": "tab:blue",
}
styles = {
    "q6-zero circle, same voltage": "-",
    "physical beta=.025, same voltage": "-",
    "physical beta=.025, retuned": "-",
    "synthetic beta=.025, retuned": "--",
}

for row, radius in enumerate(INITIAL_RADII):
    baseline_mean = results[radius]["q6-zero circle, same voltage"].mean(axis=1)
    for name, values in results[radius].items():
        mean = values.mean(axis=1)
        axes[row, 0].plot(NOMINAL_Q, mean, color=colors[name], linestyle=styles[name], label=name)
        if name != "q6-zero circle, same voltage":
            axes[row, 1].plot(NOMINAL_Q, mean - baseline_mean, color=colors[name], linestyle=styles[name], label=name)
    axes[row, 0].set(xlabel="nominal q proportional to applied voltage", ylabel="mean escape periods", title=f"initial radius {radius} r0")
    axes[row, 1].axhline(0.0, color="black", linewidth=1)
    axes[row, 1].set(xlabel="nominal q", ylabel="lifetime minus q6-zero circle", title="Benefit at an equal voltage coordinate")
    axes[row, 0].legend(fontsize=7)
    axes[row, 1].legend(fontsize=7)

physical_retuned = results[0.72]["physical beta=.025, retuned"].mean(axis=1)
synthetic_retuned = results[0.72]["synthetic beta=.025, retuned"].mean(axis=1)
axes[0, 2].plot(NOMINAL_Q, physical_retuned - synthetic_retuned, color="tab:purple")
axes[0, 2].axhline(0.0, color="black", linewidth=1)
axes[0, 2].set(xlabel="Mathieu q", ylabel="physical minus synthetic lifetime", title="Do residual physical multipoles matter?")

operating_q = 0.688
index = int(np.argmin(np.abs(NOMINAL_Q - operating_q)))
labels = list(designs)
means = [results[0.72][name][index].mean() for name in labels]
survival = [np.mean(results[0.72][name][index] >= PERIODS) for name in labels]
positions = np.arange(len(labels))
axes[1, 2].bar(positions - 0.18, means, 0.36, label="mean lifetime")
axes[1, 2].bar(positions + 0.18, np.array(survival) * PERIODS, 0.36, label="survival fraction x horizon")
axes[1, 2].set_xticks(positions, [label.replace(", ", "\n") for label in labels], rotation=15, fontsize=7)
axes[1, 2].set(ylabel="periods", title=f"Nominal operating point q={NOMINAL_Q[index]:.3f}, initial r=.72")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 17 — voltage-aware dynamics of the physical beta=.025 electrode", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

for name in labels:
    print("operating point", NOMINAL_Q[index], name, "mean", results[0.72][name][index].mean(), "individual", results[0.72][name][index])
print("CONCLUSION: the same-voltage curve tests a realizable comparison; the retuned curve isolates field-shape benefit at a 2.62% voltage cost. Agreement of physical and synthetic curves tests whether the uncancelled higher multipoles alter the reduced dynamics.")
print("saved", OUTPUT)
