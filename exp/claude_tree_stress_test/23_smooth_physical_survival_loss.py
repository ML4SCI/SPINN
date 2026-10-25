import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, qmc


OUTPUT = "exp/claude_tree_stress_test/results/23_smooth_physical_survival_loss.png"
DATA_OUTPUT = "exp/claude_tree_stress_test/results/23_smooth_physical_survival_loss.npz"
RADII = np.linspace(0.40, 1.20, 81)
OUTER_RADIUS = 5.0
COUNT = 40
PROBE_RADIUS = 0.5
Q_MEAN = 0.69
Q_SIGMA = 0.006
A_OVER_Q = 0.237 / 0.706
INITIAL_RADIUS = 0.72
TARGET_PERIODS = 10
DT = 0.02
SAMPLES = 512
SMOOTH_MAX_TEMPERATURE = 0.001
BOUNDARY_TEMPERATURES = [0.002, 0.010, 0.030]


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
eta_ratio = eta / eta[np.argmin(np.abs(beta))]

unit = qmc.Halton(3, scramble=True, seed=2301).random(SAMPLES)
nominal_q = Q_MEAN + Q_SIGMA * norm.ppf(np.clip(unit[:, 0], 1e-10, 1.0 - 1e-10))
angles = 0.5 * np.pi * unit[:, 1]
phases = 2.0 * np.pi * unit[:, 2]

local_beta = np.repeat(beta, SAMPLES)
q = np.repeat(eta_ratio, SAMPLES) * np.tile(nominal_q, len(RADII))
a = A_OVER_Q * q
phase = np.tile(phases, len(RADII))
z = INITIAL_RADIUS * np.exp(1j * np.tile(angles, len(RADII)))
velocity = np.zeros_like(z)
maximum_radius = np.abs(z).copy()
logsumexp_radius = np.abs(z) / SMOOTH_MAX_TEMPERATURE
active = np.ones(len(z), dtype=bool)
steps = int(TARGET_PERIODS * np.pi / DT)
time = 0.0


def field(position):
    return -np.conj(position) - 3.0 * local_beta * np.conj(position**5)


acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(z)
for _ in range(steps):
    z[active] += DT * velocity[active] + 0.5 * DT**2 * acceleration[active]
    time += DT
    radius_now = np.abs(z)
    maximum_radius = np.maximum(maximum_radius, radius_now)
    logsumexp_radius = np.logaddexp(logsumexp_radius, np.minimum(radius_now, 1.25) / SMOOTH_MAX_TEMPERATURE)
    newly_far = active & (radius_now > 1.25)
    active[newly_far] = False
    clipped = np.minimum(radius_now, 1.25) * np.exp(1j * np.angle(z))
    next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(clipped)
    velocity[active] += 0.5 * DT * (acceleration[active] + next_acceleration[active])
    acceleration = next_acceleration

maximum_radius = maximum_radius.reshape(len(RADII), SAMPLES)
smooth_maximum = (SMOOTH_MAX_TEMPERATURE * logsumexp_radius).reshape(len(RADII), SAMPLES)
exact_survival = np.mean(maximum_radius < 1.0, axis=1)
smooth_survival = {}
for temperature in BOUNDARY_TEMPERATURES:
    score = 1.0 / (1.0 + np.exp(np.clip((smooth_maximum - 1.0) / temperature, -60.0, 60.0)))
    smooth_survival[temperature] = np.mean(score, axis=1)

print("HYPOTHESIS: a smooth maximum-radius chance surrogate can give a usable geometry gradient while retaining the same broad optimum as exact P(T>10).")
print("exact best beta radius probability", beta[np.argmax(exact_survival)], RADII[np.argmax(exact_survival)], np.max(exact_survival))
for temperature, curve in smooth_survival.items():
    index = np.argmax(curve)
    print("boundary temperature", temperature, "best beta radius score", beta[index], RADII[index], curve[index], "correlation with exact", np.corrcoef(curve, exact_survival)[0, 1])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].step(beta, exact_survival, where="mid", color="black", label="exact P(T>10)")
for temperature, curve in smooth_survival.items():
    axes[0, 0].plot(beta, curve, label=f"smooth eps={temperature:.3f}")
axes[0, 0].set(xlabel="beta", ylabel="survival score", title="Smooth loss versus physical event")
axes[0, 0].legend(fontsize=8)

exact_derivative = np.gradient(exact_survival, beta)
axes[0, 1].plot(beta, exact_derivative, color="black", label="exact finite difference")
for temperature, curve in smooth_survival.items():
    axes[0, 1].plot(beta, np.gradient(curve, beta), label=f"smooth eps={temperature:.3f}")
axes[0, 1].set(xlabel="beta", ylabel="d objective / d beta", title="Does smoothing create a usable direction?")
axes[0, 1].legend(fontsize=8)

for temperature, curve in smooth_survival.items():
    axes[0, 2].scatter(exact_survival, curve, s=20, label=f"eps={temperature:.3f}")
axes[0, 2].plot([exact_survival.min(), exact_survival.max()], [exact_survival.min(), exact_survival.max()], color="black", linestyle="--")
axes[0, 2].set(xlabel="exact probability", ylabel="smooth score", title="Bias versus rank preservation")
axes[0, 2].legend(fontsize=8)

selected = [0, len(RADII) // 2, len(RADII) - 1]
for index in selected:
    axes[1, 0].hist(1.0 - maximum_radius[index], bins=np.linspace(-0.3, 0.1, 80), histtype="step", density=True, label=f"beta={beta[index]:.3f}")
axes[1, 0].axvline(0.0, color="black", linewidth=1)
axes[1, 0].set(xlabel="hard minimum boundary margin", ylabel="density", title="Why the chance loss is statistically noisy")
axes[1, 0].legend(fontsize=8)

prefixes = [32, 64, 128, 256, 512]
for index in selected:
    exact_prefix = [np.mean(maximum_radius[index, :prefix] < 1.0) for prefix in prefixes]
    score = 1.0 / (1.0 + np.exp(np.clip((smooth_maximum[index] - 1.0) / 0.01, -60.0, 60.0)))
    smooth_prefix = [np.mean(score[:prefix]) for prefix in prefixes]
    axes[1, 1].plot(prefixes, exact_prefix, marker="o", label=f"exact beta={beta[index]:.3f}")
    axes[1, 1].plot(prefixes, smooth_prefix, marker="s", linestyle="--", label=f"smooth beta={beta[index]:.3f}")
axes[1, 1].set(xscale="log", xlabel="nested joint samples", ylabel="objective", title="Sampling convergence")
axes[1, 1].legend(fontsize=7)

rank_exact = np.argsort(np.argsort(exact_survival))
for temperature, curve in smooth_survival.items():
    rank_smooth = np.argsort(np.argsort(curve))
    axes[1, 2].plot(beta, np.abs(rank_smooth - rank_exact), label=f"eps={temperature:.3f}")
axes[1, 2].set(xlabel="beta", ylabel="absolute rank change", title="Too much smoothing can choose a different shape")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 23 — a differentiable physical survival objective", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

np.savez(DATA_OUTPUT, radii=RADII, beta=beta, eta_ratio=eta_ratio, maximum_radius=maximum_radius, smooth_maximum=smooth_maximum, exact_survival=exact_survival)

print("CONCLUSION: boundary smoothing preserves the event value but does not by itself remove jagged shape derivatives. Larger temperatures trade derivative smoothness for ranking bias. The loss belongs outside the exact field solve and needs sampling or parameter-space robustification before gradient optimization.")
print("saved", OUTPUT)
