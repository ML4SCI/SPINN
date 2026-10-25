import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from spinn.geometry.four_rod import FourRodTrap


RATIOS = np.linspace(0.80, 1.50, 29)
OUTER_RADIUS = 5.0
ROI = 0.60
OUTPUT = "scratch/10_mfs_known_optimum.png"


def circle_points(cx, cy, radius, n, phase=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])


def kernel(points, sources):
    distance = np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)
    return np.log(distance)


def solve_mfs(trap):
    sources = []
    boundary = []
    values = []
    for electrode in trap.electrodes():
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, phase=0.017))
        boundary.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0.0, 0.0, 1.30 * trap.outer_radius, 128, phase=0.011))
    boundary.append(circle_points(0.0, 0.0, trap.outer_radius, 256))
    values.append(np.zeros(256))

    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def evaluate(points, sources, coefficients):
    return kernel(points, sources) @ coefficients[:-1] + coefficients[-1]


def multipoles(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    columns = [np.ones(len(points))]
    for order in orders:
        columns.extend([np.real(z**order), np.imag(z**order)])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, potential, rcond=None)
    signed_a2 = coefficients[1]
    signed_a6 = coefficients[3]
    a10 = np.hypot(coefficients[5], coefficients[6])
    return signed_a2, signed_a6 / signed_a2, a10 / abs(signed_a2)


probe_axis = np.linspace(-ROI, ROI, 61)
gx, gy = np.meshgrid(probe_axis, probe_axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
probe = probe[np.hypot(probe[:, 0], probe[:, 1]) <= ROI]

rows = []
for ratio in RATIOS:
    trap = FourRodTrap(r0=1.0, rod_radius=float(ratio), outer_radius=OUTER_RADIUS, V0=1.0)
    sources, coefficients = solve_mfs(trap)
    potential = evaluate(probe, sources, coefficients)
    a2, a6_a2, a10_a2 = multipoles(probe, potential)

    held_points = []
    held_values = []
    for electrode in trap.electrodes():
        held_points.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 361, phase=0.0043))
        held_values.append(np.full(361, trap.electrode_voltage(electrode)))
    held_points.append(circle_points(0.0, 0.0, trap.outer_radius, 721, phase=0.0021))
    held_values.append(np.zeros(721))
    held_points = np.vstack(held_points)
    held_values = np.concatenate(held_values)
    boundary_error = np.max(np.abs(evaluate(held_points, sources, coefficients) - held_values))

    rows.append((ratio, 2.0 * a2, a6_a2, a10_a2, boundary_error))
    print(
        f"R/r0={ratio:.3f} eta={2*a2:.6f} signed_a6/a2={a6_a2:+.4e} "
        f"a10/a2={a10_a2:.4e} held_boundary_max={boundary_error:.3e}"
    )

rows = np.array(rows)
ratio = rows[:, 0]
eta = rows[:, 1]
a6 = rows[:, 2]
a10 = rows[:, 3]
boundary_error = rows[:, 4]

crossings = []
for index in range(len(ratio) - 1):
    if a6[index] * a6[index + 1] <= 0:
        fraction = -a6[index] / (a6[index + 1] - a6[index])
        crossings.append(ratio[index] + fraction * (ratio[index + 1] - ratio[index]))

fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
axes[0, 0].plot(ratio, a6, marker="o")
axes[0, 0].axhline(0.0, color="black", linewidth=1)
axes[0, 0].set(ylabel=r"signed $a_6/a_2$", title="Leading non-quadrupole harmonic")

axes[0, 1].plot(ratio, a10, marker="o", color="tab:green")
axes[0, 1].set(ylabel=r"$|a_{10}|/|a_2|$", title="Next symmetry-allowed harmonic")

axes[1, 0].plot(ratio, eta, marker="o", color="tab:orange")
axes[1, 0].set(xlabel=r"circular rod radius $R/r_0$", ylabel=r"central $\eta=2a_2$", title="Quadrupole amplitude is a different objective")

axes[1, 1].semilogy(ratio, boundary_error, marker="o", color="tab:red")
axes[1, 1].set(xlabel=r"circular rod radius $R/r_0$", ylabel="held-out maximum voltage error", title="Boundary-collocation reference check")

for ax in axes.ravel():
    ax.axvline(1.145, color="red", linestyle="--", label="classical ratio 1.145")
    ax.grid(alpha=0.2)
    ax.legend()
axes[0, 0].set_xlabel(r"circular rod radius $R/r_0$")
axes[0, 1].set_xlabel(r"circular rod radius $R/r_0$")

fig.suptitle("Exact-circle boundary solve: which objective recovers the known rod ratio?", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("a6 zero crossings:", crossings)
print("largest held-out boundary error:", boundary_error.max())
print("saved", OUTPUT)
