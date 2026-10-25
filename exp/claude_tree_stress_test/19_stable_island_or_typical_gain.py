import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, qmc


OUTPUT = "exp/claude_tree_stress_test/results/19_stable_island_or_typical_gain.png"
ORDERS = np.arange(2, 31, 4)
RHO = 0.5
A_OVER_Q = 0.237 / 0.706
INITIAL_RADIUS = 0.72
PERIODS = 180
DT = 0.04
ETA_RATIO = 0.9744662440131631
SAMPLES = 2048
SIGMAS = np.array([0.003, 0.006, 0.010])

BASELINE = np.array([1.0, 0.0, -9.54171958e-6, -6.89746563e-8, -3.34722203e-10, -1.21227111e-12, -3.29814122e-15, 0.0])
PHYSICAL = np.array([1.0, 1.56250015e-3, -3.90109192e-12, -2.36420786e-12, -6.22013422e-9, -5.48756817e-10, 0.0, 0.0])


def normalized_field(z, coefficients):
    result = np.zeros_like(z, dtype=complex)
    for order, relative in zip(ORDERS, coefficients):
        ratio = relative / RHO ** (order - 2)
        result += 0.5 * order * ratio * np.conj(z ** (order - 1))
    return -result


unit = qmc.Halton(3, scramble=True, seed=1701).random(SAMPLES)
normal_coordinate = norm.ppf(np.clip(unit[:, 0], 1e-10, 1.0 - 1e-10))
angles = 0.5 * np.pi * unit[:, 1]
phases = 2.0 * np.pi * unit[:, 2]


def escape_times(coefficients, q_scale):
    nominal_q = np.concatenate([0.69 + sigma * normal_coordinate for sigma in SIGMAS])
    angle = np.tile(angles, len(SIGMAS))
    phase = np.tile(phases, len(SIGMAS))
    q = q_scale * nominal_q
    a = A_OVER_Q * q
    z = INITIAL_RADIUS * np.exp(1j * angle)
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    escaped = np.full(len(z), PERIODS, dtype=float)
    steps = int(PERIODS * np.pi / DT)
    time = 0.0
    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(z, coefficients)
    for step in range(steps):
        z[alive] += DT * velocity[alive] + 0.5 * DT**2 * acceleration[alive]
        time += DT
        newly_dead = alive & (np.abs(z) >= 1.0)
        escaped[newly_dead] = step * DT / np.pi
        alive[newly_dead] = False
        clipped = np.minimum(np.abs(z), 1.0) * np.exp(1j * np.angle(z))
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(clipped, coefficients)
        velocity[alive] += 0.5 * DT * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    return escaped.reshape(len(SIGMAS), SAMPLES)


designs = {
    "q6-zero circle": (BASELINE, 1.0),
    "beta=.025, same voltage": (PHYSICAL, ETA_RATIO),
    "beta=.025, retuned voltage": (PHYSICAL, 1.0),
}
colors = {"q6-zero circle": "black", "beta=.025, same voltage": "tab:orange", "beta=.025, retuned voltage": "tab:red"}

print("HYPOTHESIS: if beta=.025 creates only a small stable island, its mean can improve while its median and low quantiles remain unchanged; a broad improvement should shift the whole lifetime distribution.")
results = {}
for name, (coefficients, scale) in designs.items():
    results[name] = escape_times(coefficients, scale)
    for sigma, values in zip(SIGMAS, results[name]):
        print(name, "sigma", sigma, "mean", np.mean(values), "median", np.median(values), "q90", np.quantile(values, 0.9), "P(T>10)", np.mean(values > 10.0), "P(T=180)", np.mean(values >= PERIODS))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
sigma_index = 1
for name, values in results.items():
    ordered = np.sort(values[sigma_index])
    axes[0, 0].step(ordered, np.arange(1, SAMPLES + 1) / SAMPLES, where="post", color=colors[name], label=name)
axes[0, 0].set(xlim=[0, 20], xlabel="escape periods", ylabel="empirical CDF", title="Typical trajectories: q sigma=.006")
axes[0, 0].legend(fontsize=8)

for name, values in results.items():
    ordered = np.sort(values[sigma_index])
    survival = 1.0 - np.arange(SAMPLES) / SAMPLES
    axes[0, 1].semilogy(ordered, survival, color=colors[name], label=name)
axes[0, 1].set(xlim=[0, PERIODS], ylim=[1 / SAMPLES, 1], xlabel="escape periods", ylabel="surviving fraction", title="Rare long-lived tail")
axes[0, 1].legend(fontsize=8)

thresholds = np.linspace(0.0, 60.0, 121)
for name, values in results.items():
    survival = [np.mean(values[sigma_index] > threshold) for threshold in thresholds]
    axes[0, 2].plot(thresholds, survival, color=colors[name], label=name)
axes[0, 2].set(xlabel="required lifetime [periods]", ylabel="success probability", title="Specification-dependent ranking")
axes[0, 2].legend(fontsize=8)

prefixes = np.array([32, 64, 128, 256, 512, 1024, 2048])
for name, values in results.items():
    means = [np.mean(values[sigma_index, :prefix]) for prefix in prefixes]
    axes[1, 0].semilogx(prefixes, means, marker="o", color=colors[name], label=name)
axes[1, 0].set(xlabel="nested joint samples", ylabel="mean escape periods", title="Does the rare-tail mean converge?")
axes[1, 0].legend(fontsize=8)

metrics = ["median", "90th percentile", "mean", "P(T>10) x 10"]
positions = np.arange(len(metrics))
for offset, name in zip([-0.25, 0.0, 0.25], designs):
    values = results[name][sigma_index]
    metric_values = [np.median(values), np.quantile(values, 0.9), np.mean(values), 10.0 * np.mean(values > 10.0)]
    axes[1, 1].bar(positions + offset, metric_values, 0.25, color=colors[name], label=name)
axes[1, 1].set_xticks(positions, metrics, rotation=15, fontsize=8)
axes[1, 1].set(ylabel="metric value", title="One design can win one objective and lose another")
axes[1, 1].legend(fontsize=7)

for name, values in results.items():
    axes[1, 2].plot(SIGMAS, np.mean(values, axis=1), marker="o", color=colors[name], label=f"{name}: mean")
    axes[1, 2].plot(SIGMAS, np.quantile(values, 0.9, axis=1), marker="s", linestyle="--", color=colors[name], alpha=0.7)
axes[1, 2].set(xlabel="q standard deviation", ylabel="escape periods", title="Solid mean; dashed 90th percentile")
axes[1, 2].legend(fontsize=7)

fig.suptitle("Claude-tree stress test 19 — stable island or population-wide improvement?", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: compare the CDF and survival curve, not only the mean. A mean-only gain caused by a tiny long-lived tail is a real stable-island result, but it is a different claim from improving typical ion confinement.")
print("saved", OUTPUT)
