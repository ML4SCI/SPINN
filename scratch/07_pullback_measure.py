import matplotlib.pyplot as plt
import numpy as np


N = 200001
SCALES = np.linspace(-0.30, 0.30, 61)
OUTPUT = "scratch/07_pullback_measure.png"


z = np.linspace(-1.0, 1.0, N)
true_physical_mean_x2 = 1.0 / 3.0
unweighted = []
weighted = []
minimum_jacobian = []

for a in SCALES:
    x = z + a * z * (1.0 - z**2)
    jacobian = 1.0 + a * (1.0 - 3.0 * z**2)
    field = x**2
    unweighted.append(np.trapezoid(field, z) / 2.0)
    weighted.append(np.trapezoid(field * jacobian, z) / np.trapezoid(jacobian, z))
    minimum_jacobian.append(np.min(jacobian))

unweighted = np.array(unweighted)
weighted = np.array(weighted)
minimum_jacobian = np.array(minimum_jacobian)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
axes[0].plot(SCALES, unweighted, linewidth=2, label=r"uniform average in reference $z$")
axes[0].plot(SCALES, weighted, linewidth=2, label=r"physical measure: weight by $\det J$")
axes[0].axhline(true_physical_mean_x2, color="black", linestyle="--", label=r"exact physical mean of $x^2$: $1/3$")
axes[0].set(
    xlabel=r"warp amplitude $a$ in $x=z+a z(1-z^2)$",
    ylabel=r"estimated physical mean of $x^2$",
    title="Same physical domain, different coordinate parameterization",
)
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(SCALES, np.abs(unweighted - true_physical_mean_x2), linewidth=2, label="unweighted error")
axes[1].plot(SCALES, np.abs(weighted - true_physical_mean_x2), linewidth=2, label="Jacobian-weighted error")
axes[1].set_yscale("log")
axes[1].set(
    xlabel="warp amplitude",
    ylabel="absolute integration error",
    title="Coordinate-gauge error in a mapped integral",
)
axes[1].legend()
axes[1].grid(alpha=0.2)

fig.suptitle("Why mapped collocation integrals require the Jacobian measure", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("map fixes x=-1 and x=+1 for every warp amplitude")
print("minimum Jacobian over sweep:", minimum_jacobian.min())
print("maximum unweighted error:", np.max(np.abs(unweighted - true_physical_mean_x2)))
print("maximum Jacobian-weighted error:", np.max(np.abs(weighted - true_physical_mean_x2)))
print("saved", OUTPUT)
