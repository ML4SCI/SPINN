import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, qmc


OUTPUT = "exp/claude_tree_stress_test/results/22_objective_changes_optimal_shape.png"
RADII = np.linspace(0.72, 1.20, 49)
OUTER_RADIUS = 5.0
COUNT = 48
PROBE_RADIUS = 0.5
Q_MEAN = 0.69
Q_SIGMA = 0.006
A_OVER_Q = 0.237 / 0.706
INITIAL_RADIUS = 0.72
PERIODS = 180
DT = 0.04
SAMPLES = 384


def solve_circle(radius):
    distance = 1.0 + radius
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    tb = 2.0 * np.pi * np.arange(3 * COUNT) / (3 * COUNT)
    ts = 2.0 * np.pi * np.arange(COUNT) / COUNT + 0.013
    boundary = np.concatenate([center + radius * np.exp(1j * tb) for center in centers])
    sources = np.concatenate([center + 0.70 * radius * np.exp(1j * ts) for center in centers])
    values = np.concatenate([np.full(len(tb), voltage) for voltage in voltages])
    to = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    tso = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([boundary, OUTER_RADIUS * np.exp(1j * to)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * tso)])
    values = np.concatenate([values, np.zeros(len(to))])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def potential(z, sources, coefficients):
    return np.log(np.abs(z[:, None] - sources[None, :])) @ coefficients[:-1] + coefficients[-1]


theta = 2.0 * np.pi * np.arange(1024) / 1024
probe = PROBE_RADIUS * np.exp(1j * theta)
eta = []
beta = []
for radius in RADII:
    sources, coefficients = solve_circle(radius)
    values = potential(probe, sources, coefficients)
    a2 = 2.0 * np.mean(values * np.cos(2.0 * theta)) / PROBE_RADIUS**2
    a6 = 2.0 * np.mean(values * np.cos(6.0 * theta)) / PROBE_RADIUS**6
    eta.append(2.0 * a2)
    beta.append(a6 / a2)
eta = np.array(eta)
beta = np.array(beta)
zero_index = np.argmin(np.abs(beta))
eta_reference = eta[zero_index]
eta_ratio = eta / eta_reference

unit = qmc.Halton(3, scramble=True, seed=2201).random(SAMPLES)
nominal_q_sample = Q_MEAN + Q_SIGMA * norm.ppf(np.clip(unit[:, 0], 1e-10, 1.0 - 1e-10))
angles = 0.5 * np.pi * unit[:, 1]
phases = 2.0 * np.pi * unit[:, 2]


def escape_times(q_scale):
    local_beta = np.repeat(beta, SAMPLES)
    q = np.repeat(q_scale, SAMPLES) * np.tile(nominal_q_sample, len(RADII))
    a = A_OVER_Q * q
    phase = np.tile(phases, len(RADII))
    z = INITIAL_RADIUS * np.exp(1j * np.tile(angles, len(RADII)))
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    escaped = np.full(len(z), PERIODS, dtype=float)
    steps = int(PERIODS * np.pi / DT)
    time = 0.0

    def field(position):
        return -np.conj(position) - 3.0 * local_beta * np.conj(position**5)

    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(z)
    for step in range(steps):
        z[alive] += DT * velocity[alive] + 0.5 * DT**2 * acceleration[alive]
        time += DT
        newly_dead = alive & (np.abs(z) >= 1.0)
        escaped[newly_dead] = step * DT / np.pi
        alive[newly_dead] = False
        clipped = np.minimum(np.abs(z), 1.0) * np.exp(1j * np.angle(z))
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(clipped)
        velocity[alive] += 0.5 * DT * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    return escaped.reshape(len(RADII), SAMPLES)


print("HYPOTHESIS: the optimal physical rod radius depends on whether the outer loss rewards typical lifetime, rare stable islands, or voltage efficiency; no single beta should optimize all of them.")
same_voltage = escape_times(eta_ratio)
retuned_voltage = escape_times(np.ones_like(eta_ratio))


def metrics(values):
    return {
        "median": np.median(values, axis=1),
        "q90": np.quantile(values, 0.9, axis=1),
        "mean": np.mean(values, axis=1),
        "p10": np.mean(values > 10.0, axis=1),
        "p180": np.mean(values >= PERIODS, axis=1),
    }


same_metrics = metrics(same_voltage)
retuned_metrics = metrics(retuned_voltage)
for comparison, values in [("same voltage", same_metrics), ("retuned", retuned_metrics)]:
    print(comparison)
    for name, curve in values.items():
        index = np.argmax(curve)
        print(name, "best radius", RADII[index], "beta", beta[index], "eta ratio", eta_ratio[index], "value", curve[index])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].plot(beta, eta_ratio, color="black")
axes[0, 0].scatter([beta[zero_index]], [eta_ratio[zero_index]], color="tab:blue", label="q6-zero reference")
axes[0, 0].set(xlabel="beta = a6/a2", ylabel="eta / reference eta", title="The physical circle family couples beta and efficiency")
axes[0, 0].legend(fontsize=8)

for name, label, color in [("median", "median", "tab:blue"), ("q90", "90th percentile", "tab:green"), ("mean", "mean", "tab:red")]:
    axes[0, 1].plot(beta, same_metrics[name], color=color, label=label)
axes[0, 1].set(xlabel="beta", ylabel="escape periods", title="Same voltage: lifetime summaries disagree")
axes[0, 1].legend(fontsize=8)

axes[0, 2].plot(beta, same_metrics["p10"], color="tab:orange", label="P(T>10)")
axes[0, 2].plot(beta, same_metrics["p180"], color="tab:purple", label="P(T=180)")
axes[0, 2].set(xlabel="beta", ylabel="probability", title="Same voltage: explicit success specifications")
axes[0, 2].legend(fontsize=8)

for name, label, color in [("median", "median", "tab:blue"), ("q90", "90th percentile", "tab:green"), ("mean", "mean", "tab:red")]:
    axes[1, 0].plot(beta, retuned_metrics[name], color=color, label=label)
axes[1, 0].set(xlabel="beta", ylabel="escape periods", title="Retuned voltage: isolate field-shape effect")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(beta, retuned_metrics["p10"], color="tab:orange", label="P(T>10)")
axes[1, 1].plot(beta, retuned_metrics["p180"], color="tab:purple", label="P(T=180)")
axes[1, 1].set(xlabel="beta", ylabel="probability", title="Retuned voltage: stable-island objectives")
axes[1, 1].legend(fontsize=8)

objectives = ["median", "q90", "mean", "p10", "p180"]
same_best = [beta[np.argmax(same_metrics[name])] for name in objectives]
retuned_best = [beta[np.argmax(retuned_metrics[name])] for name in objectives]
positions = np.arange(len(objectives))
axes[1, 2].bar(positions - 0.18, same_best, 0.36, label="same voltage")
axes[1, 2].bar(positions + 0.18, retuned_best, 0.36, label="retuned voltage")
axes[1, 2].axhline(0.0, color="black", linewidth=1)
axes[1, 2].set_xticks(positions, objectives)
axes[1, 2].set(ylabel="beta of best sampled radius", title="The loss function chooses the electrode")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 22 — different physical objectives select different shapes", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: the geometry optimizer cannot be scientifically defined until the trajectory statistic and voltage constraint are explicit. A differentiable mean, quantile surrogate, or success-probability loss encodes different devices even on this one-parameter family.")
print("saved", OUTPUT)
