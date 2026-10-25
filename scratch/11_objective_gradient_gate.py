import matplotlib.pyplot as plt
import numpy as np

from spinn.geometry.four_rod import FourRodTrap


RATIOS = np.linspace(0.90, 1.38, 49)
FD_STEPS = [0.04, 0.02, 0.01, 0.005]
OUTER_RADIUS = 5.0
ROI = 0.60
OUTPUT = "scratch/11_objective_gradient_gate.png"


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
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
        boundary.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0.0, 0.0, 1.30 * trap.outer_radius, 128, 0.011))
    boundary.append(circle_points(0.0, 0.0, trap.outer_radius, 256))
    values.append(np.zeros(256))
    sources = np.vstack(sources)
    matrix = np.column_stack([kernel(np.vstack(boundary), sources), np.ones(sum(map(len, boundary)))])
    coefficients, *_ = np.linalg.lstsq(matrix, np.concatenate(values), rcond=1e-12)
    return sources, coefficients


probe_axis = np.linspace(-ROI, ROI, 61)
gx, gy = np.meshgrid(probe_axis, probe_axis)
PROBE = np.column_stack([gx.ravel(), gy.ravel()])
PROBE = PROBE[np.hypot(PROBE[:, 0], PROBE[:, 1]) <= ROI]
Z = PROBE[:, 0] + 1j * PROBE[:, 1]
ORDERS = [2, 6, 10, 14, 18]
DESIGN = np.column_stack(
    [np.ones(len(PROBE))]
    + [component for order in ORDERS for component in (np.real(Z**order), np.imag(Z**order))]
)


def harmonic_ratio(ratio):
    trap = FourRodTrap(r0=1.0, rod_radius=float(ratio), outer_radius=OUTER_RADIUS, V0=1.0)
    sources, coefficients = solve_mfs(trap)
    potential = kernel(PROBE, sources) @ coefficients[:-1] + coefficients[-1]
    fit, *_ = np.linalg.lstsq(DESIGN, potential, rcond=None)
    return fit[3] / fit[1]


def objective(ratio):
    q = harmonic_ratio(ratio)
    return q * q


q = np.array([harmonic_ratio(ratio) for ratio in RATIOS])
loss = q**2
curve_gradient = np.gradient(loss, RATIOS, edge_order=2)

check_ratios = np.array([1.00, 1.10, 1.145, 1.20, 1.30])
fd_gradients = np.array(
    [[(objective(r + h) - objective(r - h)) / (2.0 * h) for h in FD_STEPS] for r in check_ratios]
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
axes[0, 0].plot(RATIOS, q, marker=".")
axes[0, 0].axhline(0.0, color="black", linewidth=1)
axes[0, 0].set(ylabel=r"signed $q=a_6/a_2$", title="Physical field error changes sign")

axes[0, 1].semilogy(RATIOS, loss, marker=".")
axes[0, 1].set(ylabel=r"design loss $J=q^2$", title="Simplest non-negative optimization target")

axes[1, 0].plot(RATIOS, curve_gradient, label="derivative of dense sweep")
axes[1, 0].scatter(check_ratios, fd_gradients[:, -1], color="tab:red", zorder=3, label=f"independent central FD, h={FD_STEPS[-1]}")
axes[1, 0].axhline(0.0, color="black", linewidth=1)
axes[1, 0].set(xlabel=r"rod radius $R/r_0$", ylabel=r"$dJ/d(R/r_0)$", title="Gradient points toward the optimum")
axes[1, 0].legend()

for index, ratio in enumerate(check_ratios):
    axes[1, 1].plot(FD_STEPS, fd_gradients[index], marker="o", label=f"R/r0={ratio:.3f}")
axes[1, 1].set_xscale("log")
axes[1, 1].invert_xaxis()
axes[1, 1].set(xlabel="central-difference step h", ylabel=r"finite-difference $dJ/dR$", title="Gradient convergence under step refinement")
axes[1, 1].legend(ncol=2, fontsize=8)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.axvline(1.145, color="red", linestyle="--", alpha=0.7)
axes[0, 0].set_xlabel(r"rod radius $R/r_0$")
axes[0, 1].set_xlabel(r"rod radius $R/r_0$")
fig.suptitle("Objective and shape-gradient gate on exactly circular electrodes", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

for ratio, gradients in zip(check_ratios, fd_gradients):
    print(f"R/r0={ratio:.3f} gradients " + " ".join(f"h={h:.3f}:{g:+.6e}" for h, g in zip(FD_STEPS, gradients)))
print("minimum sampled J at R/r0=", RATIOS[np.argmin(loss)], "J=", loss.min())
print("saved", OUTPUT)
