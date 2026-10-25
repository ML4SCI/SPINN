import matplotlib.pyplot as plt
import numpy as np

from spinn.geometry.four_rod import FourRodTrap


RATIOS = np.linspace(0.94, 1.34, 41)
OUTER_RADIUS = 5.0
ROI = 0.60
OUTPUT = "scratch/13_joint_vs_bilevel_gradient.png"


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


axis = np.linspace(-ROI, ROI, 61)
gx, gy = np.meshgrid(axis, axis)
points = np.column_stack([gx.ravel(), gy.ravel()])
points = points[np.hypot(points[:, 0], points[:, 1]) <= ROI]
z = points[:, 0] + 1j * points[:, 1]
orders = [2, 6, 10, 14, 18]
fit_matrix = np.column_stack(
    [np.ones(len(points))]
    + [part for order in orders for part in (np.real(z**order), np.imag(z**order))]
)


def physical_loss(ratio):
    trap = FourRodTrap(r0=1.0, rod_radius=float(ratio), outer_radius=OUTER_RADIUS, V0=1.0)
    sources, boundary, values = [], [], []
    for electrode in trap.electrodes():
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
        boundary.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0, 0, 1.30 * OUTER_RADIUS, 128, 0.011))
    boundary.append(circle_points(0, 0, OUTER_RADIUS, 256))
    values.append(np.zeros(256))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, np.concatenate(values), rcond=1e-12)
    potential = kernel(points, sources) @ coefficients[:-1] + coefficients[-1]
    fit, *_ = np.linalg.lstsq(fit_matrix, potential, rcond=None)
    q = fit[3] / fit[1]
    return q * q


resolved_loss = np.array([physical_loss(ratio) for ratio in RATIOS])
total_gradient = np.gradient(resolved_loss, RATIOS, edge_order=2)
frozen_loss = np.full_like(RATIOS, physical_loss(1.00))
partial_gradient = np.zeros_like(RATIOS)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

axes[0].axis("off")
axes[0].text(0.5, 0.90, "Corrected center anchor", ha="center", fontsize=13, weight="bold")
axes[0].text(0.5, 0.72, r"$x=z$ throughout the query disc", ha="center", fontsize=12, bbox=dict(boxstyle="round", fc="#e8f3ff"))
axes[0].annotate("", xy=(0.5, 0.57), xytext=(0.5, 0.68), arrowprops=dict(arrowstyle="->"))
axes[0].text(0.5, 0.48, r"At fixed field weights:  $\partial J/\partial\phi=0$", ha="center", fontsize=12, bbox=dict(boxstyle="round", fc="#ffe8e8"))
axes[0].annotate("", xy=(0.5, 0.32), xytext=(0.5, 0.44), arrowprops=dict(arrowstyle="->"))
axes[0].text(0.5, 0.20, r"Physical sensitivity needs $d\theta^*/d\phi$" "\n(re-solve or differentiate through the PDE solve)", ha="center", fontsize=12, bbox=dict(boxstyle="round", fc="#e9f8e9"))
axes[0].set_title("Why the hard mask changes optimization")

axes[1].semilogy(RATIOS, resolved_loss, linewidth=2, label=r"re-solved field $J(\theta^*(R),R)$")
axes[1].semilogy(RATIOS, frozen_loss, linestyle="--", linewidth=2, label=r"field frozen $J(\theta,R)$")
axes[1].axvline(1.145, color="red", linestyle=":", label="known optimum")
axes[1].set(xlabel=r"electrode radius $R/r_0$", ylabel=r"$J=(a_6/a_2)^2$", title="The objective moves only after physics responds")
axes[1].legend(fontsize=8)

axes[2].plot(RATIOS, total_gradient, linewidth=2, label=r"total $dJ/dR$")
axes[2].plot(RATIOS, partial_gradient, linestyle="--", linewidth=2, label=r"direct partial $\partial J/\partial R$")
axes[2].axhline(0.0, color="black", linewidth=1)
axes[2].axvline(1.145, color="red", linestyle=":")
axes[2].set(xlabel=r"electrode radius $R/r_0$", ylabel="shape gradient", title="A simultaneous scalar loss misses the useful term")
axes[2].legend(fontsize=8)

for ax in axes[1:]:
    ax.grid(alpha=0.2)
fig.suptitle("Shape sensitivity after removing the coordinate-gauge shortcut", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("maximum direct partial gradient with frozen field:", np.max(np.abs(partial_gradient)))
print("total gradient at R/r0=1.00:", total_gradient[np.argmin(np.abs(RATIOS - 1.00))])
print("total gradient at R/r0=1.20:", total_gradient[np.argmin(np.abs(RATIOS - 1.20))])
print("saved", OUTPUT)
