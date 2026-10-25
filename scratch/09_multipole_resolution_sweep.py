import matplotlib.pyplot as plt
import numpy as np

from spinn.geometry.export_fem import reference_geometry
from spinn.geometry.four_rod import FourRodTrap
from spinn.physics.fem_reference import solve_laplace_fd


RATIOS = np.linspace(0.90, 1.40, 11)
GRIDS = [161, 241, 321]
ROI = 0.50
OUTPUT = "scratch/09_multipole_resolution_sweep.png"


def symmetry_multipoles(x, y, potential):
    complex_coordinate = x + 1j * y
    orders = [2, 6, 10, 14, 18]
    columns = [np.ones_like(x)]
    for order in orders:
        columns.extend([np.real(complex_coordinate**order), np.imag(complex_coordinate**order)])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, potential, rcond=None)
    a2 = np.hypot(coefficients[1], coefficients[2])
    a6 = np.hypot(coefficients[3], coefficients[4])
    a10 = np.hypot(coefficients[5], coefficients[6])
    signed_a6 = coefficients[3] * np.sign(coefficients[1])
    return signed_a6 / a2, a10 / a2


curves = {}
for grid in GRIDS:
    rows = []
    for ratio in RATIOS:
        trap = FourRodTrap(r0=1.0, rod_radius=float(ratio), outer_radius=5.0, V0=1.0)
        solved = solve_laplace_fd(reference_geometry(trap), grid=grid)
        gx, gy = np.meshgrid(solved["xs"], solved["ys"])
        keep = np.hypot(gx, gy) <= ROI
        a6_a2, a10_a2 = symmetry_multipoles(gx[keep], gy[keep], solved["field"][keep])
        rows.append((a6_a2, a10_a2))
        print(f"grid={grid} R/r0={ratio:.3f} signed_a6/a2={a6_a2:+.5e} a10/a2={a10_a2:.5e}")
    curves[grid] = np.array(rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for grid in GRIDS:
    axes[0].plot(RATIOS, curves[grid][:, 0], marker="o", label=f"FD grid {grid}")
    axes[1].plot(RATIOS, curves[grid][:, 1], marker="o", label=f"FD grid {grid}")

for ax in axes:
    ax.axvline(1.145, color="red", linestyle="--", label="classical ratio 1.145")
    ax.grid(alpha=0.2)
    ax.set_xlabel(r"circular rod radius $R/r_0$")
    ax.legend()
axes[0].axhline(0.0, color="black", linewidth=1)
axes[0].set(ylabel=r"signed $a_6/a_2$", title=r"Leading allowed error harmonic: $r^6\cos(6\theta)$")
axes[1].set(ylabel=r"$|a_{10}|/|a_2|$", title=r"Next allowed error harmonic: $r^{10}\cos(10\theta)$")
fig.suptitle("Does the classical circular-rod optimum survive FD grid refinement?", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

for grid in GRIDS:
    best = np.argmin(np.abs(curves[grid][:, 0]))
    print(f"grid {grid}: smallest |a6/a2| at R/r0={RATIOS[best]:.3f}")
print("saved", OUTPUT)
