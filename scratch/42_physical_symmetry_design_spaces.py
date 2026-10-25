import matplotlib.pyplot as plt
import numpy as np

from spinn.geometry.four_rod import FourRodTrap
from pig_gate_common import exact_field


OUTPUT = "scratch/42_physical_symmetry_design_spaces.png"
HYPOTHESIS = "One mirror confines the electrical null to a line but permits a physical shift along it; two mirrors and D4 pin the null to the origin, so one-axis optimization is a strictly richer physical design space."


def polynomial_null(points, potential):
    x, y = points[:, 0], points[:, 1]
    terms = [(i, total-i) for total in range(6) for i in range(total+1)]
    matrix = np.column_stack([x**i*y**j for i, j in terms])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    point = np.zeros(2)
    for _ in range(12):
        px, py = point
        gradient = np.zeros(2)
        hessian = np.zeros((2, 2))
        for (i, j), c in zip(terms, coefficients):
            if i:
                gradient[0] += c*i*px**(i-1)*py**j
            if j:
                gradient[1] += c*j*px**i*py**(j-1)
            if i >= 2:
                hessian[0, 0] += c*i*(i-1)*px**(i-2)*py**j
            if i and j:
                hessian[0, 1] += c*i*j*px**(i-1)*py**(j-1)
            if j >= 2:
                hessian[1, 1] += c*j*(j-1)*px**i*py**(j-2)
        hessian[1, 0] = hessian[0, 1]
        step = np.linalg.solve(hessian, gradient)
        point -= step
        if np.linalg.norm(step) < 1e-12:
            break
    return point


print("HYPOTHESIS:", HYPOTHESIS)
designs = [
    ("x -> -x only", 6, FourRodTrap(center_perturb=(.05, .05, .09, -.04), radius_perturb=(.04, .04, .07, -.02))),
    ("y -> -y only", 6, FourRodTrap(center_perturb=(.08, -.04, .05, .05), radius_perturb=(.05, -.03, .04, .04))),
    ("both axis mirrors", 4, FourRodTrap(center_perturb=(.06, .06, -.04, -.04), radius_perturb=(.05, .05, -.03, -.03))),
    ("D4 quadrupole", 2, FourRodTrap(center_perturb=(.04, .04, .04, .04), radius_perturb=(.03, .03, .03, .03))),
]
axis = np.linspace(-3.5, 3.5, 141)
gx, gy = np.meshgrid(axis, axis)
points = np.column_stack([gx.ravel(), gy.ravel()])
probe_axis = np.linspace(-.55, .55, 61)
px, py = np.meshgrid(probe_axis, probe_axis)
probe = np.column_stack([px.ravel(), py.ravel()])
keep_probe = np.hypot(probe[:, 0], probe[:, 1]) <= .55
probe = probe[keep_probe]

fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
all_results = []
for ax, (label, degrees, trap) in zip(axes.ravel(), designs):
    valid = trap.in_vacuum(points)
    potential = np.full(len(points), np.nan)
    potential[valid] = exact_field(trap, points[valid])[0]
    probe_potential = exact_field(trap, probe)[0]
    null = polynomial_null(probe, probe_potential)
    mirrored_x = exact_field(trap, probe*np.array([-1., 1.]))[0]
    mirrored_y = exact_field(trap, probe*np.array([1., -1.]))[0]
    scale = np.max(np.abs(probe_potential))
    rx = np.sqrt(np.mean((probe_potential-mirrored_x)**2))/scale
    ry = np.sqrt(np.mean((probe_potential-mirrored_y)**2))/scale
    all_results.append((label, null, rx, ry))
    levels = np.linspace(-.5, .5, 21)
    contour = ax.contourf(gx, gy, potential.reshape(gx.shape), levels=levels, cmap="coolwarm", extend="both")
    for electrode in trap.electrodes():
        circle = plt.Circle((electrode.cx, electrode.cy), electrode.radius, facecolor="white", edgecolor="black", linewidth=1.8)
        ax.add_patch(circle)
        ax.text(electrode.cx, electrode.cy, "+" if electrode.voltage_sign > 0 else "-", ha="center", va="center", fontsize=16)
    ax.scatter(*null, marker="x", s=90, color="lime", linewidth=2.5)
    ax.axhline(0, color="black", alpha=.2, linewidth=.8)
    ax.axvline(0, color="black", alpha=.2, linewidth=.8)
    ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal", xlabel="x/r0", ylabel="y/r0", title=f"{label}: {degrees} circular-shape DOF\nnull=({null[0]:+.3f}, {null[1]:+.3f}) r0")

fig.colorbar(contour, ax=axes, label="independent electrostatic potential V/V0", shrink=.8)
fig.suptitle("Hypothesis: symmetry choice controls which physical electrode designs and null shifts are possible", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

for label, null, rx, ry in all_results:
    print(f"{label:18s} null=({null[0]:+.6f},{null[1]:+.6f}) x_reflection={rx:.3e} y_reflection={ry:.3e}")
one_axis_on_line = abs(all_results[0][1][0]) < 1e-5 and abs(all_results[0][1][1]) > 1e-2 and abs(all_results[1][1][1]) < 1e-5 and abs(all_results[1][1][0]) > 1e-2
two_axis_at_origin = np.linalg.norm(all_results[2][1]) < 1e-5 and np.linalg.norm(all_results[3][1]) < 1e-5
print("GATES one_axis_moves_along_line=", one_axis_on_line, "two_axis_pins_origin=", two_axis_at_origin)
print("CONCLUSION: accept only if the independent fields show the predicted null geometry; symmetry labels alone do not prove the electrodes obey the group.")
print("saved", OUTPUT)
