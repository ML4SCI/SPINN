import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/03_dodecapole_lifetime_scan.png"
BETAS = np.linspace(-0.45, 0.45, 61)
Q_VALUES = np.linspace(0.50, 0.84, 57)
A_OVER_Q = 0.237 / 0.706
INITIAL_RADII = [0.50, 0.72]
RF_PERIODS = 70
DT = 0.04
ESCAPE_RADIUS = 1.0


def normalized_field(z, beta):
    return -np.conj(z) - 3.0 * beta * np.conj(z**5)


def lifetime_map(initial_radius):
    beta_grid, q_grid = np.meshgrid(BETAS, Q_VALUES, indexing="ij")
    base_beta = beta_grid.ravel()
    base_q = q_grid.ravel()
    angles = np.array([0.0, np.pi / 8.0, np.pi / 4.0])
    phases = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    scenario_angle, scenario_phase = np.meshgrid(angles, phases, indexing="ij")
    scenario_angle = scenario_angle.ravel()
    scenario_phase = scenario_phase.ravel()
    beta = np.repeat(base_beta, len(scenario_angle))
    q = np.repeat(base_q, len(scenario_angle))
    a = A_OVER_Q * q
    phase = np.tile(scenario_phase, len(base_beta))
    z = initial_radius * np.exp(1j * np.tile(scenario_angle, len(base_beta)))
    velocity = np.zeros_like(z)
    alive = np.ones(len(z), dtype=bool)
    escape_step = np.full(len(z), int(RF_PERIODS * np.pi / DT), dtype=int)
    time = 0.0
    acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(z, beta)
    steps = int(RF_PERIODS * np.pi / DT)
    for step in range(steps):
        z[alive] += DT * velocity[alive] + 0.5 * DT**2 * acceleration[alive]
        time += DT
        newly_dead = alive & (np.abs(z) >= ESCAPE_RADIUS)
        escape_step[newly_dead] = step
        alive[newly_dead] = False
        clipped = np.minimum(np.abs(z), ESCAPE_RADIUS) * np.exp(1j * np.angle(z))
        next_acceleration = (a - 2.0 * q * np.cos(2.0 * time + phase)) * normalized_field(clipped, beta)
        velocity[alive] += 0.5 * DT * (acceleration[alive] + next_acceleration[alive])
        acceleration = next_acceleration
    survival = alive.reshape(len(base_beta), len(scenario_angle)).mean(axis=1)
    mean_lifetime = escape_step.reshape(len(base_beta), len(scenario_angle)).mean(axis=1) / steps
    shape = (len(BETAS), len(Q_VALUES))
    return survival.reshape(shape), mean_lifetime.reshape(shape)


print("HYPOTHESIS: if |a6| is a sufficient trap-quality score, lifetime should peak at beta=a6/a2=0 and fall approximately symmetrically. Nonlinear resonance physics instead predicts sign-dependent bands and nonzero local optima.")
maps = {}
for initial_radius in INITIAL_RADII:
    survival, lifetime = lifetime_map(initial_radius)
    maps[initial_radius] = (survival, lifetime)
    zero_index = np.argmin(np.abs(BETAS))
    best_index = np.argmax(lifetime, axis=0)
    gain = lifetime[best_index, np.arange(len(Q_VALUES))] - lifetime[zero_index]
    print("initial radius", initial_radius, "max nonzero-beta lifetime gain", np.max(gain), "fraction q cells improved", np.mean(gain > 1e-6))

large_lifetime = maps[INITIAL_RADII[1]][1]
zero_index = np.argmin(np.abs(BETAS))
best_index = np.argmax(large_lifetime, axis=0)
best_beta = BETAS[best_index]
gain = large_lifetime[best_index, np.arange(len(Q_VALUES))] - large_lifetime[zero_index]
slice_index = int(np.argmax(gain))
slice_q = Q_VALUES[slice_index]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [Q_VALUES[0], Q_VALUES[-1], BETAS[0], BETAS[-1]]
for column, initial_radius in enumerate(INITIAL_RADII):
    survival, lifetime = maps[initial_radius]
    im0 = axes[0, column].imshow(lifetime, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, column].axhline(0.0, color="white", linestyle="--", linewidth=1)
    axes[0, column].set(xlabel="Mathieu q along a/q=0.336", ylabel="beta = a6/a2", title=f"mean normalized lifetime, initial r={initial_radius}")
    im1 = axes[1, column].imshow(survival, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="magma")
    axes[1, column].axhline(0.0, color="white", linestyle="--", linewidth=1)
    axes[1, column].set(xlabel="Mathieu q along a/q=0.336", ylabel="beta = a6/a2", title=f"surviving fraction after {RF_PERIODS} periods")

axes[0, 2].plot(Q_VALUES, best_beta, color="tab:blue", label="best beta")
axes[0, 2].axhline(0.0, color="black", linestyle="--", label="multipole cancellation")
axes[0, 2].fill_between(Q_VALUES, 0.0, gain, color="tab:green", alpha=0.25, label="lifetime gain (same vertical scale)")
axes[0, 2].set(xlabel="Mathieu q", ylabel="best beta or lifetime gain", title=f"Optimal dodecapole at initial r={INITIAL_RADII[1]}")
axes[0, 2].legend(fontsize=8)

axes[1, 2].plot(BETAS, large_lifetime[:, slice_index], color="tab:blue", label=f"q={slice_q:.3f}")
axes[1, 2].axvline(0.0, color="black", linestyle="--")
axes[1, 2].scatter([best_beta[slice_index]], [large_lifetime[best_index[slice_index], slice_index]], color="red", zorder=3, label="best sampled beta")
axes[1, 2].set(xlabel="beta = a6/a2", ylabel="mean normalized lifetime", title="A fixed-q slice tests monotonicity directly")
axes[1, 2].legend()

fig.colorbar(im0, ax=axes[0, :2], label="mean lifetime / requested duration")
fig.colorbar(im1, ax=axes[1, :2], label="surviving scenario fraction")
fig.suptitle("Claude-tree stress test 03 — does cancelling a6 maximize finite-amplitude ion lifetime?", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("largest-gain slice q", slice_q, "best beta", best_beta[slice_index], "gain", gain[slice_index])
print("CONCLUSION: inspect the maps for sign asymmetry and isolated bands. This reduced dodecapole model can falsify |a6|-only ranking, but it does not yet establish a new physical electrode because injection, damping, axial coupling, and the remaining multipoles are omitted.")
print("saved", OUTPUT)
