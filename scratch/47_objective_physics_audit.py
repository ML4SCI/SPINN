import matplotlib.pyplot as plt
import numpy as np

from spinn.geometry.four_rod import FourRodTrap
from pig_gate_common import exact_field


OUTPUT = "scratch/47_objective_physics_audit.png"
HYPOTHESIS = "Origin quadrupole strength changes weakly under north-south asymmetry while null displacement and odd harmonics grow strongly, so eta alone cannot define a usable optimal trap."


def trap_at(delta):
    return FourRodTrap(r0=1., rod_radius=1.145, outer_radius=5., V0=1., center_perturb=(0., 0., delta, -delta))


def harmonic_fit(points, potential, maximum_order=8):
    z = points[:, 0]+1j*points[:, 1]
    matrix = [np.ones(len(points))]
    for order in range(1, maximum_order+1):
        matrix.extend([np.real(z**order), np.imag(z**order)])
    coefficients, *_ = np.linalg.lstsq(np.column_stack(matrix), potential, rcond=None)
    complex_coefficients = {}
    for order in range(1, maximum_order+1):
        complex_coefficients[order] = coefficients[2*order-1]-1j*coefficients[2*order]
    return complex_coefficients


def null_from_harmonics(coefficients):
    z = 0j
    for _ in range(15):
        derivative = sum(order*coefficients[order]*z**(order-1) for order in coefficients)
        second = sum(order*(order-1)*coefficients[order]*z**(order-2) for order in coefficients if order >= 2)
        step = derivative/second
        z -= step
        if abs(step) < 1e-13:
            break
    return np.array([z.real, z.imag])


print("HYPOTHESIS:", HYPOTHESIS)
axis = np.linspace(-.35, .35, 71)
gx, gy = np.meshgrid(axis, axis)
points = np.column_stack([gx.ravel(), gy.ravel()])
points = points[np.hypot(points[:, 0], points[:, 1]) <= .35]
deltas = np.linspace(-.08, .08, 17)
radius = .25
rows = []
for delta in deltas:
    potential = exact_field(trap_at(float(delta)), points)[0]
    coefficients = harmonic_fit(points, potential)
    amplitudes = {order: abs(value) for order, value in coefficients.items()}
    null = null_from_harmonics(coefficients)
    quadrupole = amplitudes[2]
    odd_rms = np.sqrt(sum((amplitudes[order]*radius**order)**2 for order in [1, 3, 5, 7]))
    quadrupole_rms = quadrupole*radius**2
    high_even_rms = np.sqrt(sum((amplitudes[order]*radius**order)**2 for order in [4, 6, 8]))
    rows.append((delta, quadrupole, np.linalg.norm(null), odd_rms/quadrupole_rms, high_even_rms/quadrupole_rms))
rows = np.array(rows)
center_index = np.argmin(np.abs(rows[:, 0]))
eta_relative = rows[:, 1]/rows[center_index, 1]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
axes[0, 0].plot(rows[:, 0], eta_relative, marker="o")
axes[0, 0].axhline(1, color="black", linestyle="--")
axes[0, 0].set(xlabel="north/south asymmetry delta", ylabel="quadrupole strength / symmetric value", title="What eta-like curvature reports")
axes[0, 1].plot(rows[:, 0], rows[:, 2], marker="o", color="tab:red")
axes[0, 1].axhline(.01, color="black", linestyle="--", label="example 0.01 r0 centering gate")
axes[0, 1].set(xlabel="north/south asymmetry delta", ylabel="RF-null displacement / r0", title="Physical centering failure")
axes[0, 1].legend(fontsize=8)
axes[1, 0].plot(rows[:, 0], rows[:, 3], marker="o", color="tab:purple")
axes[1, 0].axhline(.01, color="black", linestyle="--", label="example 1% odd-harmonic gate")
axes[1, 0].set(yscale="log", xlabel="north/south asymmetry delta", ylabel="odd / quadrupole amplitude at 0.25 r0", title="Dipole/hexapole contamination")
axes[1, 0].legend(fontsize=8)
axes[1, 1].plot(rows[:, 0], rows[:, 4], marker="o", color="tab:green")
axes[1, 1].set(yscale="log", xlabel="north/south asymmetry delta", ylabel="higher-even / quadrupole amplitude", title="Even anharmonic contamination")
for ax in axes.ravel():
    ax.grid(alpha=.25)
fig.suptitle("Hypothesis: eta is a performance quantity, not a complete trap-design objective", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

eta_span = np.max(eta_relative)-np.min(eta_relative)
edge_null = max(rows[0, 2], rows[-1, 2])
edge_odd = max(rows[0, 3], rows[-1, 3])
print(f"eta_relative_range={eta_relative.min():.6f}..{eta_relative.max():.6f} span={eta_span:.3%}")
print(f"edge_null={edge_null:.5f} r0 edge_odd_ratio={edge_odd:.3f}")
print("GATES eta_weak_change=", eta_span < .10, "null_large=", edge_null > .05, "odd_large=", edge_odd > .10)
print("CONCLUSION: eta must be optimized subject to independent null, odd-harmonic, depth/stability, geometry-safety and fabrication constraints; do not collapse these into unvalidated alpha weights yet.")
print("saved", OUTPUT)
