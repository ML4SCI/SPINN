import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from spinn.geometry.four_rod import FourRodTrap


STARTS = [0.90, 1.38]
STEPS = 16
LEARNING_RATE = 35.0
FD_STEP = 0.005
OUTER_RADIUS = 5.0
ROI = 0.60
OUTPUT = "scratch/12_one_parameter_electrode_optimization.png"


def circle_points(cx, cy, radius, n, phase=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


def solve(ratio):
    trap = FourRodTrap(r0=1.0, rod_radius=float(ratio), outer_radius=OUTER_RADIUS, V0=1.0)
    sources, boundary, values = [], [], []
    for electrode in trap.electrodes():
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
        boundary.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0.0, 0.0, 1.30 * trap.outer_radius, 128, 0.011))
    boundary.append(circle_points(0.0, 0.0, trap.outer_radius, 256))
    values.append(np.zeros(256))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, np.concatenate(values), rcond=1e-12)
    return trap, sources, coefficients


axis = np.linspace(-ROI, ROI, 61)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(probe[:, 0], probe[:, 1]) <= ROI
fit_points = probe[inside]
z = fit_points[:, 0] + 1j * fit_points[:, 1]
orders = [2, 6, 10, 14, 18]
design = np.column_stack(
    [np.ones(len(fit_points))]
    + [part for order in orders for part in (np.real(z**order), np.imag(z**order))]
)


def evaluate(points, sources, coefficients):
    return kernel(points, sources) @ coefficients[:-1] + coefficients[-1]


def state(ratio):
    trap, sources, coefficients = solve(ratio)
    potential = evaluate(fit_points, sources, coefficients)
    fit, *_ = np.linalg.lstsq(design, potential, rcond=None)
    q = fit[3] / fit[1]
    return q * q, q, fit[1], trap, sources, coefficients


histories = []
for start in STARTS:
    ratio = start
    history = []
    for step in range(STEPS):
        loss, q, *_ = state(ratio)
        gradient = (state(ratio + FD_STEP)[0] - state(ratio - FD_STEP)[0]) / (2.0 * FD_STEP)
        history.append((ratio, loss, q, gradient))
        print(f"start={start:.2f} step={step:02d} R/r0={ratio:.6f} q={q:+.5e} J={loss:.5e} dJ/dR={gradient:+.5e}")
        ratio -= LEARNING_RATE * gradient
    loss, q, *_ = state(ratio)
    history.append((ratio, loss, q, np.nan))
    histories.append(np.array(history))
    print(f"start={start:.2f} final R/r0={ratio:.6f} q={q:+.5e} J={loss:.5e}")


fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
colors = ["tab:blue", "tab:orange"]
for start, history, color in zip(STARTS, histories, colors):
    axes[0, 0].plot(history[:, 0], marker="o", color=color, label=f"start {start:.2f}")
    axes[0, 1].semilogy(history[:, 1], marker="o", color=color, label=f"start {start:.2f}")
axes[0, 0].axhline(1.145, color="red", linestyle="--", label="known 1.145")
axes[0, 0].set(xlabel="optimization step", ylabel=r"electrode radius $R/r_0$", title="Both initial electrode sizes converge")
axes[0, 1].set(xlabel="optimization step", ylabel=r"$J=(a_6/a_2)^2$", title="Physical harmonic loss decreases")
axes[0, 0].legend()
axes[0, 1].legend()

for start, history, color in zip(STARTS, histories, colors):
    final = history[-1, 0]
    for ratio, linestyle, alpha in ((start, ":", 0.45), (final, "-", 1.0)):
        trap = FourRodTrap(r0=1.0, rod_radius=float(ratio), outer_radius=OUTER_RADIUS, V0=1.0)
        for electrode in trap.electrodes():
            axes[1, 0].add_patch(Circle((electrode.cx, electrode.cy), electrode.radius, fill=False, color=color, linestyle=linestyle, alpha=alpha))
axes[1, 0].add_patch(Circle((0, 0), ROI, fill=False, color="magenta", linewidth=1, label="multipole-fit disc"))
axes[1, 0].set(xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), aspect="equal", xlabel=r"$x/r_0$", ylabel=r"$y/r_0$", title="Electrodes: dotted initial, solid optimized")

error_maps = []
labels = []
for start, history in zip(STARTS, histories):
    for ratio, label in ((start, f"start {start:.2f}"), (history[-1, 0], f"final {history[-1, 0]:.4f}")):
        _, _, a2, _, sources, coefficients = state(ratio)
        potential = evaluate(fit_points, sources, coefficients)
        ideal = a2 * np.real(z**2)
        scale = np.max(np.abs(ideal))
        field_error = np.full(gx.shape, np.nan)
        field_error.ravel()[inside] = (potential - ideal) / scale
        error_maps.append(field_error)
        labels.append(label)

initial_error = np.maximum(np.abs(error_maps[0]), np.abs(error_maps[2]))
final_error = np.maximum(np.abs(error_maps[1]), np.abs(error_maps[3]))
display = np.column_stack([initial_error, final_error])
limit = np.nanmax(np.abs(display))
image = axes[1, 1].imshow(display, origin="lower", extent=(-ROI, ROI, -ROI, ROI), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
axes[1, 1].axvline(0.0, color="black", linewidth=2)
axes[1, 1].text(-0.30, ROI * 0.92, "worst initial", ha="center", va="top")
axes[1, 1].text(+0.30, ROI * 0.92, "worst optimized", ha="center", va="top")
axes[1, 1].set(xlabel="two ROI maps placed side-by-side", ylabel=r"$y/r_0$", title="Non-quadrupole potential error / ideal peak")
fig.colorbar(image, ax=axes[1, 1], label="normalized potential error")

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("First optimization gate: recover the known circular-electrode design", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("saved", OUTPUT)
