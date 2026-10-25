import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from spinn.geometry.four_rod import FourRodTrap
from pig_gate_common import exact_field


OUTPUT = "scratch/60_exact_nested_geometry_baseline.png"
HYPOTHESIS = "When the field is solved fully before evaluating geometry, the circular four-rod family has a clear physical a6 zero near the known 1.145 r0 radius, while quadrupole efficiency varies smoothly."
MEASUREMENT_RADIUS = 0.50


def coefficients(trap, radius=MEASUREMENT_RADIUS, samples=720):
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    potential = exact_field(trap, points)[0]
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / radius**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / radius**6
    a10 = 2.0 * np.mean(potential * np.cos(10.0 * theta)) / radius**10
    return a2, a6, a10


def draw_electrodes(ax, trap, title):
    for electrode in trap.electrodes():
        color = "tab:red" if electrode.voltage_sign > 0 else "tab:blue"
        ax.add_patch(Circle((electrode.cx, electrode.cy), electrode.radius, facecolor=color, edgecolor="black", alpha=0.75))
    ax.add_patch(Circle((0, 0), MEASUREMENT_RADIUS, fill=False, edgecolor="magenta", linewidth=2, linestyle="--"))
    ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", xlabel="x/r0", ylabel="y/r0", title=title)
    ax.grid(alpha=0.2)


print("HYPOTHESIS:", HYPOTHESIS)
rod_radii = np.linspace(0.65, 1.55, 37)
rows = []
for rod_radius in rod_radii:
    a2, a6, a10 = coefficients(FourRodTrap(r0=1.0, rod_radius=float(rod_radius), outer_radius=5.0, V0=1.0))
    eta = 2.0 * a2
    q6 = (a6 / a2) * MEASUREMENT_RADIUS**4
    q10 = (a10 / a2) * MEASUREMENT_RADIUS**8
    rows.append((rod_radius, eta, a6, q6, q10))
rows = np.array(rows)

sign_changes = np.where(np.sign(rows[:-1, 2]) != np.sign(rows[1:, 2]))[0]
index = sign_changes[np.argmin(np.abs(rows[sign_changes, 0] - 1.145))]
x0, x1 = rows[index, 0], rows[index + 1, 0]
y0, y1 = rows[index, 2], rows[index + 1, 2]
zero_radius = x0 - y0 * (x1 - x0) / (y1 - y0)
zero_a2, zero_a6, zero_a10 = coefficients(FourRodTrap(rod_radius=float(zero_radius)))
zero_eta = 2.0 * zero_a2
zero_q10 = abs(zero_a10 / zero_a2) * MEASUREMENT_RADIUS**8
print(f"interpolated a6-zero rod radius={zero_radius:.6f} r0 eta={zero_eta:.6f} residual_a6={zero_a6:.3e} physical_q10={zero_q10:.3e}")

bad_trap = FourRodTrap(rod_radius=0.70)
optimal_trap = FourRodTrap(rod_radius=float(zero_radius))
fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
draw_electrodes(axes[0, 0], bad_trap, "Physically anharmonic circular trap\nrod radius = 0.70 r0")
draw_electrodes(axes[0, 1], optimal_trap, f"Nested-solve a6 zero\nrod radius = {zero_radius:.4f} r0")

axes[1, 0].plot(rows[:, 0], rows[:, 1], marker="o", markersize=3, color="tab:green", label="eta = 2 a2")
axes[1, 0].axvline(zero_radius, color="black", linestyle="--", label="a6 zero")
axes[1, 0].set(xlabel="circular rod radius / r0", ylabel="quadrupole efficiency eta", title="Efficiency after a complete inner field solve")

axes[1, 1].plot(rows[:, 0], rows[:, 3], marker="o", markersize=3, label="signed hexapole contribution q6")
axes[1, 1].plot(rows[:, 0], rows[:, 4], marker="o", markersize=3, label="signed decapole contribution q10")
axes[1, 1].axhline(0.0, color="black", linewidth=1)
axes[1, 1].axvline(zero_radius, color="black", linestyle="--", label=f"a6 zero = {zero_radius:.4f}")
axes[1, 1].set(xlabel="circular rod radius / r0", ylabel="multipole voltage / quadrupole voltage at 0.5 r0", title="Outer geometry-quality objective")

for ax in axes[1]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Correct nesting: solve the electrostatic field first, score the electrodes second", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

known_agreement = abs(zero_radius - 1.145) < 0.01
print("PLOT READOUT recovered known circular-trap radius within 0.01 r0:", known_agreement)
if known_agreement:
    print("CONCLUSION: the exact nested baseline recovers the known circular-electrode optimum. A learned inner solver must reproduce this outer curve before flexible shape discovery is credible.")
else:
    print("CONCLUSION: the exact nested baseline did not recover the expected circular optimum; audit the MFS reference or multipole convention before training another network.")
print("saved", OUTPUT)
