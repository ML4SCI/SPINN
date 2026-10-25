import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


OUTPUT = "exp/claude_tree_stress_test/results/18_robust_operating_distribution.png"
ORDERS = np.arange(2, 31, 4)
RHO = 0.5
NOMINAL_Q = np.linspace(0.62, 0.73, 45)
A_OVER_Q = 0.237 / 0.706
INITIAL_RADIUS = 0.72
PERIODS = 180
DT = 0.04
ETA_RATIO = 0.9744662440131631
SCENARIOS = 96

BASELINE = np.array([1.0, 0.0, -9.54171958e-6, -6.89746563e-8, -3.34722203e-10, -1.21227111e-12, -3.29814122e-15, 0.0])
PHYSICAL = np.array([1.0, 1.56250015e-3, -3.90109192e-12, -2.36420786e-12, -6.22013422e-9, -5.48756817e-10, 0.0, 0.0])


def amplitude_ratios(relative_coefficients):
    return relative_coefficients / RHO ** (ORDERS - 2)


def normalized_field(z, relative_coefficients):
    result = np.zeros_like(z, dtype=complex)
    for order, ratio in zip(ORDERS, amplitude_ratios(relative_coefficients)):
        result += 0.5 * order * ratio * np.conj(z ** (order - 1))
    return -result


samples = qmc.Halton(2, scramble=False).random(SCENARIOS + 1)[1:]
scenario_angles = 0.5 * np.pi * samples[:, 0]
scenario_phases = 2.0 * np.pi * samples[:, 1]


def escape_times(relative_coefficients, q_scale):
    q = np.repeat(q_scale * NOMINAL_Q, SCENARIOS)
    a = A_OVER_Q * q
    phase = np.tile(scenario_phases, len(NOMINAL_Q))
    z = INITIAL_RADIUS * np.exp(1j * np.tile(scenario_angles, len(NOMINAL_Q)))
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    escaped = np.full(len(z), PERIODS, dtype=float)
    steps = int(PERIODS * np.pi / DT)
    time = 0.0
    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(z, relative_coefficients)
    for step in range(steps):
        z[alive] += DT * velocity[alive] + 0.5 * DT**2 * acceleration[alive]
        time += DT
        newly_dead = alive & (np.abs(z) >= 1.0)
        escaped[newly_dead] = step * DT / np.pi
        alive[newly_dead] = False
        clipped = np.minimum(np.abs(z), 1.0) * np.exp(1j * np.angle(z))
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(clipped, relative_coefficients)
        velocity[alive] += 0.5 * DT * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    return escaped.reshape(len(NOMINAL_Q), SCENARIOS)


designs = {
    "q6-zero circle": (BASELINE, 1.0),
    "beta=.025, same voltage": (PHYSICAL, ETA_RATIO),
    "beta=.025, retuned voltage": (PHYSICAL, 1.0),
}

print("HYPOTHESIS: a genuine shape benefit should survive averaging over injection angle, RF phase, and modest uncertainty in nominal q; isolated resonance spikes should not.")
results = {}
for name, (coefficients, scale) in designs.items():
    results[name] = escape_times(coefficients, scale)
    print(name, "mean over full scan", results[name].mean(), "full-horizon fraction", np.mean(results[name] >= PERIODS))


def gaussian_average(mean_curve, center, sigma):
    if sigma == 0.0:
        return mean_curve[np.argmin(np.abs(NOMINAL_Q - center))]
    weights = np.exp(-0.5 * ((NOMINAL_Q - center) / sigma) ** 2)
    weights /= np.sum(weights)
    return np.sum(weights * mean_curve)


center_values = np.linspace(0.64, 0.71, 57)
sigmas = [0.0, 0.003, 0.006, 0.010]
robust = {
    sigma: {
        name: np.array([gaussian_average(values.mean(axis=1), center, sigma) for center in center_values])
        for name, values in results.items()
    }
    for sigma in sigmas
}

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
colors = {"q6-zero circle": "black", "beta=.025, same voltage": "tab:orange", "beta=.025, retuned voltage": "tab:red"}

for name, values in results.items():
    axes[0, 0].plot(NOMINAL_Q, values.mean(axis=1), color=colors[name], label=name)
axes[0, 0].set(xlabel="nominal q", ylabel="mean escape periods", title=f"Raw mean over {SCENARIOS} angle-phase samples")
axes[0, 0].legend(fontsize=8)

for sigma in sigmas:
    axes[0, 1].plot(center_values, robust[sigma]["beta=.025, same voltage"] - robust[sigma]["q6-zero circle"], label=f"sigma q={sigma:.3f}")
axes[0, 1].axhline(0.0, color="black", linewidth=1)
axes[0, 1].set(xlabel="mean nominal q", ylabel="same-voltage lifetime gain", title="Does voltage uncertainty erase the gain?")
axes[0, 1].legend(fontsize=8)

for sigma in sigmas:
    axes[0, 2].plot(center_values, robust[sigma]["beta=.025, retuned voltage"] - robust[sigma]["q6-zero circle"], label=f"sigma q={sigma:.3f}")
axes[0, 2].axhline(0.0, color="black", linewidth=1)
axes[0, 2].set(xlabel="mean Mathieu q", ylabel="retuned lifetime gain", title="Field-shape benefit after voltage retuning")
axes[0, 2].legend(fontsize=8)

subset_sizes = [12, 24, 48, 96]
for name in designs:
    curve = [gaussian_average(results[name][:, :size].mean(axis=1), 0.69, 0.006) for size in subset_sizes]
    axes[1, 0].plot(subset_sizes, curve, marker="o", color=colors[name], label=name)
axes[1, 0].set(xlabel="nested angle-phase samples", ylabel="expected escape periods", title="Ensemble convergence at q ~ N(.69,.006)")
axes[1, 0].legend(fontsize=8)

q_index = int(np.argmin(np.abs(NOMINAL_Q - 0.69)))
for name, values in results.items():
    ordered = np.sort(values[q_index])
    axes[1, 1].step(ordered, np.arange(1, SCENARIOS + 1) / SCENARIOS, where="post", color=colors[name], label=name)
axes[1, 1].set(xlabel="escape periods", ylabel="empirical CDF", title=f"Trajectory distribution at q={NOMINAL_Q[q_index]:.4f}")
axes[1, 1].legend(fontsize=8)

sigma_axis = np.linspace(0.0, 0.015, 61)
for name in ["beta=.025, same voltage", "beta=.025, retuned voltage"]:
    gain = []
    for sigma in sigma_axis:
        candidate = gaussian_average(results[name].mean(axis=1), 0.69, sigma)
        baseline = gaussian_average(results["q6-zero circle"].mean(axis=1), 0.69, sigma)
        gain.append(candidate - baseline)
    axes[1, 2].plot(sigma_axis, gain, color=colors[name], label=name)
axes[1, 2].axhline(0.0, color="black", linewidth=1)
axes[1, 2].set(xlabel="standard deviation of q", ylabel="expected lifetime gain", title="Robustness at mean nominal q=.69")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 18 — optimize an operating distribution, not one trajectory", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

for sigma in sigmas:
    values = {name: gaussian_average(result.mean(axis=1), 0.69, sigma) for name, result in results.items()}
    print("q mean 0.69 sigma", sigma, values)
print("CONCLUSION: broad gains that persist as q uncertainty and ensemble size increase are candidates for optimization. Sign changes or large sample drift identify fragile resonance tuning that should not be advertised as a general electrode improvement.")
print("saved", OUTPUT)
