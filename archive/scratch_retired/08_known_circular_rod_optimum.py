import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from spinn.geometry.export_fem import reference_geometry
from spinn.geometry.four_rod import FourRodTrap
from spinn.physics.eta import eta_fit, harmonic_purity
from spinn.physics.fem_reference import solve_laplace_fd


RATIOS = np.linspace(0.75, 1.55, 17)
GRID = 181
ROI = 0.50
OUTPUT = "scratch/08_known_circular_rod_optimum.png"


rows = []
for ratio in RATIOS:
    trap = FourRodTrap(r0=1.0, rod_radius=float(ratio), outer_radius=5.0, V0=1.0)
    solved = solve_laplace_fd(reference_geometry(trap), grid=GRID)
    xs = solved["xs"]
    ys = solved["ys"]
    h = solved["h"]
    phi = solved["field"]
    gx, gy = np.meshgrid(xs, ys)
    radius = np.hypot(gx, gy)
    roi = radius <= ROI

    ideal_phi = 0.5 * (gx**2 - gy**2)
    dphi_dy, dphi_dx = np.gradient(phi, h, h)
    ex = -dphi_dx
    ey = -dphi_dy
    ideal_ex = -gx
    ideal_ey = gy

    phi_error = np.sqrt(np.mean((phi[roi] - ideal_phi[roi]) ** 2)) / np.sqrt(np.mean(ideal_phi[roi] ** 2))
    field_error = np.sqrt(np.mean((ex[roi] - ideal_ex[roi]) ** 2 + (ey[roi] - ideal_ey[roi]) ** 2)) / np.sqrt(
        np.mean(ideal_ex[roi] ** 2 + ideal_ey[roi] ** 2)
    )

    points = np.column_stack([gx[roi], gy[roi]])
    values = phi[roi]
    eta = eta_fit(points, values, 1.0, 1.0, radius=ROI)
    purity = harmonic_purity(points, values, radius=ROI)
    rows.append({
        "ratio": float(ratio),
        "phi_error": float(phi_error),
        "field_error": float(field_error),
        "eta": eta["eta_fit"],
        "purity": purity["purity_c2"],
    })
    print(
        f"R/r0={ratio:.3f} field_error={field_error:.4f} "
        f"phi_error={phi_error:.4f} eta={eta['eta_fit']:.4f} purity={purity['purity_c2']:.4f}"
    )

ratios = np.array([row["ratio"] for row in rows])
field_errors = np.array([row["field_error"] for row in rows])
phi_errors = np.array([row["phi_error"] for row in rows])
etas = np.array([row["eta"] for row in rows])
purities = np.array([row["purity"] for row in rows])

best_field = int(np.argmin(field_errors))
best_phi = int(np.argmin(phi_errors))

fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
axes[0, 0].plot(ratios, field_errors, marker="o", label="electric-field NRMSE")
axes[0, 0].plot(ratios, phi_errors, marker="o", label="potential NRMSE")
axes[0, 0].scatter(ratios[best_field], field_errors[best_field], s=100, facecolors="none", edgecolors="red", linewidths=2)
axes[0, 0].set(ylabel="error over fixed central disc", title="Finite-region ideal-quadrupole objective")
axes[0, 0].legend()

axes[0, 1].plot(ratios, etas, marker="o")
axes[0, 1].axhline(1.0, color="black", linestyle="--", label="ideal eta")
axes[0, 1].set(ylabel=r"finite-difference $\eta$", title="Quadrupole strength alone")
axes[0, 1].legend()

axes[1, 0].plot(ratios, purities, marker="o", color="tab:green")
axes[1, 0].set(xlabel=r"circular rod radius $R/r_0$", ylabel="quadrupole harmonic purity", title="Suppression of higher harmonics")

for ratio, color in [(0.75, "tab:blue"), (1.15, "tab:orange"), (1.55, "tab:green")]:
    trap = FourRodTrap(r0=1.0, rod_radius=ratio, outer_radius=5.0, V0=1.0)
    for electrode in trap.electrodes():
        axes[1, 1].add_patch(Circle(
            (electrode.cx, electrode.cy), electrode.radius,
            fill=False, color=color, linewidth=1.5,
        ))
    axes[1, 1].plot([], [], color=color, label=f"R/r0={ratio:.2f}")
axes[1, 1].add_patch(Circle((0, 0), 1.0, fill=False, color="black", linestyle="--", linewidth=1.0))
axes[1, 1].set(
    xlim=(-4.0, 4.0), ylim=(-4.0, 4.0), aspect="equal",
    xlabel=r"$x/r_0$", ylabel=r"$y/r_0$",
    title="Electrode behavior: aperture held fixed at r0",
)
axes[1, 1].legend()

for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.axvline(1.145, color="red", linestyle="--", label="classical circular-rod ratio 1.145")
    ax.grid(alpha=0.2)
axes[1, 0].legend()
axes[0, 0].set_xlabel(r"circular rod radius $R/r_0$")
axes[0, 1].set_xlabel(r"circular rod radius $R/r_0$")

fig.suptitle("Known-optimum test before neural electrode-shape optimization", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print("best field-error ratio:", ratios[best_field], field_errors[best_field])
print("best potential-error ratio:", ratios[best_phi], phi_errors[best_phi])
print("classical expected neighborhood: R/r0 about 1.145")
print("saved", OUTPUT)
