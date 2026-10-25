import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/04_lifetime_peak_resolution.png"
BETAS = np.linspace(-0.04, 0.06, 101)
Q_CENTER = 0.6882142857142857
Q_VALUES = [Q_CENTER - 0.008, Q_CENTER, Q_CENTER + 0.008]
A_OVER_Q = 0.237 / 0.706
INITIAL_RADIUS = 0.72
PERIODS = 250


def field(z, beta):
    return -np.conj(z) - 3.0 * beta * np.conj(z**5)


def escape_periods(q, dt, betas=BETAS):
    angles = np.array([0.0, np.pi / 8.0, np.pi / 4.0])
    phases = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    scenario_angle, scenario_phase = np.meshgrid(angles, phases, indexing="ij")
    scenario_angle = scenario_angle.ravel()
    scenario_phase = scenario_phase.ravel()
    beta = np.repeat(betas, len(scenario_angle))
    phase = np.tile(scenario_phase, len(betas))
    z = INITIAL_RADIUS * np.exp(1j * np.tile(scenario_angle, len(betas)))
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    steps = int(PERIODS * np.pi / dt)
    escaped = np.full(len(z), PERIODS, dtype=float)
    time = 0.0
    a = A_OVER_Q * q
    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(z, beta)
    for step in range(steps):
        z[alive] += dt * velocity[alive] + 0.5 * dt**2 * acceleration[alive]
        time += dt
        newly_dead = alive & (np.abs(z) >= 1.0)
        escaped[newly_dead] = step * dt / np.pi
        alive[newly_dead] = False
        clipped = np.minimum(np.abs(z), 1.0) * np.exp(1j * np.angle(z))
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(clipped, beta)
        velocity[alive] += 0.5 * dt * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    return escaped.reshape(len(betas), len(scenario_angle))


def trace(q, beta, angle, phase, dt=0.01, periods=80):
    z = INITIAL_RADIUS * np.exp(1j * angle)
    velocity = 0.0j
    a = A_OVER_Q * q
    times, points = [0.0], [z]
    time = 0.0
    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(z, beta)
    steps = int(periods * np.pi / dt)
    for step in range(steps):
        z += dt * velocity + 0.5 * dt**2 * acceleration
        time += dt
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * field(z, beta)
        velocity += 0.5 * dt * (acceleration + next_acceleration)
        acceleration = next_acceleration
        if step % max(1, int(np.pi / dt)) == 0:
            times.append(time / np.pi)
            points.append(z)
        if abs(z) >= 1.0:
            break
    return np.array(times), np.array(points)


print("HYPOTHESIS: the small positive dodecapole lifetime peak is physical only if its location survives time-step refinement, a longer horizon, and nearby Mathieu q values.")
dt_results = {dt: escape_periods(Q_CENTER, dt) for dt in [0.08, 0.04, 0.02]}
q_results = {q: escape_periods(q, 0.04) for q in Q_VALUES}

finest_mean = dt_results[0.02].mean(axis=1)
best_index = int(np.argmax(finest_mean))
best_beta = BETAS[best_index]
zero_index = int(np.argmin(np.abs(BETAS)))
scenario_gain = dt_results[0.02][best_index] - dt_results[0.02][zero_index]
chosen_scenario = int(np.argmax(scenario_gain))
angles = np.repeat(np.array([0.0, np.pi / 8.0, np.pi / 4.0]), 4)
phases = np.tile(np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]), 3)

print("best beta at dt=0.02", best_beta)
for dt, values in dt_results.items():
    index = int(np.argmax(values.mean(axis=1)))
    print("dt", dt, "best beta", BETAS[index], "mean escape periods", values[index].mean(), "zero mean", values[zero_index].mean())

zero_time, zero_trace = trace(Q_CENTER, 0.0, angles[chosen_scenario], phases[chosen_scenario])
best_time, best_trace = trace(Q_CENTER, best_beta, angles[chosen_scenario], phases[chosen_scenario])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for dt, values in dt_results.items():
    axes[0, 0].plot(BETAS, values.mean(axis=1), label=f"dt={dt}")
axes[0, 0].axvline(0.0, color="black", linestyle="--")
axes[0, 0].set(xlabel="beta=a6/a2", ylabel="mean escape time [RF periods]", title="Time-step refinement at fixed q")
axes[0, 0].legend()

for q, values in q_results.items():
    axes[0, 1].plot(BETAS, values.mean(axis=1), label=f"q={q:.3f}")
axes[0, 1].axvline(0.0, color="black", linestyle="--")
axes[0, 1].set(xlabel="beta=a6/a2", ylabel="mean escape time [RF periods]", title="The feature moves with operating point")
axes[0, 1].legend()

for horizon in [25, 70, 150, 250]:
    capped = np.minimum(dt_results[0.02], horizon).mean(axis=1)
    axes[0, 2].plot(BETAS, capped, label=f"horizon={horizon}")
axes[0, 2].axvline(0.0, color="black", linestyle="--")
axes[0, 2].set(xlabel="beta=a6/a2", ylabel="mean capped escape time", title="Finite observation horizon")
axes[0, 2].legend(fontsize=8)

axes[1, 0].scatter(dt_results[0.02][zero_index], dt_results[0.02][best_index], c=np.arange(12), cmap="viridis", s=50)
axes[1, 0].plot([0, PERIODS], [0, PERIODS], color="black", linestyle="--")
axes[1, 0].set(xlabel="escape periods at beta=0", ylabel=f"escape periods at beta={best_beta:.3f}", title="The same 12 angle/phase scenarios")

axes[1, 1].plot(zero_time, np.abs(zero_trace), label="beta=0")
axes[1, 1].plot(best_time, np.abs(best_trace), label=f"beta={best_beta:.3f}")
axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
axes[1, 1].set(xlabel="RF periods", ylabel="radius/r0", title="Most improved individual scenario")
axes[1, 1].legend()

axes[1, 2].plot(zero_trace.real, zero_trace.imag, marker="o", markersize=2, label="beta=0")
axes[1, 2].plot(best_trace.real, best_trace.imag, marker="o", markersize=2, label=f"beta={best_beta:.3f}")
circle = plt.Circle((0, 0), 1.0, fill=False, color="black", linestyle="--")
axes[1, 2].add_patch(circle)
axes[1, 2].set(aspect="equal", xlabel="stroboscopic x/r0", ylabel="stroboscopic y/r0", title="One point per RF period")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 04 — numerical audit of the nonzero-lifetime optimum", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: compare the peak across dt, horizon, and q. Persistence under dt refinement supports a dynamical effect; movement with q is expected for a nonlinear resonance and means no geometry is universally optimal without an operating distribution.")
print("saved", OUTPUT)
