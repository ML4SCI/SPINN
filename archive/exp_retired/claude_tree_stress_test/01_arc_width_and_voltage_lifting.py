import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


OUTPUT = "exp/claude_tree_stress_test/results/01_arc_width_and_voltage_lifting.png"
N_THETA = 720
VOLTAGE = 0.5


theta = 2.0 * np.pi * (np.arange(N_THETA) + 0.5) / N_THETA


def four_arc_boundary(full_width):
    wrapped = (theta[:, None] - np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])[None, :] + np.pi) % (2.0 * np.pi) - np.pi
    inside = np.abs(wrapped) <= 0.5 * full_width
    signed = inside.astype(float)
    return VOLTAGE * (signed[:, 0] - signed[:, 1] + signed[:, 2] - signed[:, 3])


def cosine_coefficient(boundary, order):
    return 2.0 * np.mean(boundary * np.cos(order * theta))


def exact_coefficient(full_width, order):
    if order % 4 != 2:
        return 0.0
    return 8.0 * VOLTAGE * np.sin(0.5 * order * full_width) / (np.pi * order)


widths = np.linspace(0.03, 0.49 * np.pi, 500)
orders = [2, 6, 10]
exact = {order: np.array([exact_coefficient(width, order) for width in widths]) for order in orders}
sample_widths = np.linspace(0.04, 0.48 * np.pi, 120)
numeric = {
    order: np.array([cosine_coefficient(four_arc_boundary(width), order) for width in sample_widths])
    for order in orders
}

full_width_zero = np.pi / 3.0
half_width_zero = np.pi / 6.0

print("HYPOTHESIS: a6 vanishes at full arc width pi/3, equivalently half-width pi/6. Calling pi/3 a half-width is a factor-of-two error.")
print("exact a6 at full width pi/3", exact_coefficient(full_width_zero, 6))
print("exact a6 at full width 2*pi/3 (half-width pi/3)", exact_coefficient(2.0 * np.pi / 3.0, 6))

# D4-antisymmetric boundary control on one eighth of the outer circle.
# This is a voltage-control relaxation on a fixed circle, not a geometry optimization.
n_wedge = N_THETA // 8
B = np.zeros((N_THETA, n_wedge))
for i, angle in enumerate(theta):
    quadrant = int(angle / (0.5 * np.pi))
    local = angle - quadrant * 0.5 * np.pi
    folded = min(local, 0.5 * np.pi - local)
    j = min(int(folded / (0.25 * np.pi) * n_wedge), n_wedge - 1)
    diagonal_sign = 1.0 if local <= 0.25 * np.pi else -1.0
    B[i, j] = diagonal_sign * (1.0 if quadrant % 2 == 0 else -1.0)

fourier_rows = {order: 2.0 * np.mean(B * np.cos(order * theta)[:, None], axis=0) for order in orders + [14, 18, 22]}
result = linprog(
    -fourier_rows[2],
    A_eq=np.vstack([fourier_rows[6], fourier_rows[10]]),
    b_eq=np.zeros(2),
    bounds=[(-VOLTAGE, VOLTAGE)] * n_wedge,
    method="highs",
)
lp_boundary = B @ result.x
arc_boundary = four_arc_boundary(full_width_zero)
lp_coefficients = np.array([cosine_coefficient(lp_boundary, order) for order in orders + [14, 18, 22]])
arc_coefficients = np.array([cosine_coefficient(arc_boundary, order) for order in orders + [14, 18, 22]])

print("LP success", result.success)
print("LP coefficients a2,a6,a10,a14,a18,a22", lp_coefficients)
print("pi/3 full-width arc coefficients", arc_coefficients)
print("LP distinct rounded voltage values", np.unique(np.round(lp_boundary, 6)))

field_axis = np.linspace(-0.98, 0.98, 241)
x, y = np.meshgrid(field_axis, field_axis)
rr = np.sqrt(x**2 + y**2)
tt = np.arctan2(y, x)


def harmonic_extension(boundary, max_order=70):
    values = np.full(rr.shape, np.mean(boundary))
    for order in range(1, max_order + 1):
        cosine = 2.0 * np.mean(boundary * np.cos(order * theta))
        sine = 2.0 * np.mean(boundary * np.sin(order * theta))
        values += rr**order * (cosine * np.cos(order * tt) + sine * np.sin(order * tt))
    values[rr > 0.98] = np.nan
    return values


lp_field = harmonic_extension(lp_boundary)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for order, color in zip(orders, ["tab:blue", "tab:red", "tab:green"]):
    axes[0, 0].plot(widths / np.pi, exact[order], color=color, label=f"exact a{order}")
    axes[0, 0].scatter(sample_widths / np.pi, numeric[order], color=color, s=5, alpha=0.35)
axes[0, 0].axvline(full_width_zero / np.pi, color="black", linestyle="--", label="full width = pi/3")
axes[0, 0].axvline((2.0 * half_width_zero) / np.pi, color="grey", linestyle=":")
axes[0, 0].set(xlabel="full angular width / pi", ylabel="interior harmonic coefficient", title="Exact disk Fourier coefficients")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(theta / np.pi, arc_boundary, label="four arcs, full width pi/3")
axes[0, 1].plot(theta / np.pi, lp_boundary, label="fixed-circle voltage LP", alpha=0.8)
axes[0, 1].set(xlabel="boundary angle / pi", ylabel="boundary voltage", title="These are boundary voltages, not electrode shapes")
axes[0, 1].legend(fontsize=8)

shown_orders = np.array(orders + [14, 18, 22])
bar_x = np.arange(len(shown_orders))
axes[0, 2].bar(bar_x - 0.18, arc_coefficients, 0.36, label="four arcs")
axes[0, 2].bar(bar_x + 0.18, lp_coefficients, 0.36, label="voltage LP")
axes[0, 2].set_xticks(bar_x, [f"a{o}" for o in shown_orders])
axes[0, 2].set(ylabel="coefficient", title="Two cancelled terms do not cancel the tail")
axes[0, 2].legend()

im = axes[1, 0].imshow(lp_field, origin="lower", extent=[-0.98, 0.98, -0.98, 0.98], cmap="coolwarm", vmin=-VOLTAGE, vmax=VOLTAGE)
axes[1, 0].set(aspect="equal", xlabel="x/R", ylabel="y/R", title="Exact harmonic extension of LP voltage")
fig.colorbar(im, ax=axes[1, 0])

axes[1, 1].step(np.arange(n_wedge), result.x, where="mid")
axes[1, 1].set(xlabel="independent patch in 0 to pi/4", ylabel="optimized voltage", title="LP solution is bang-bang patch control")

probe_radii = np.linspace(0.05, 0.95, 300)
for boundary, label in [(arc_boundary, "four arcs"), (lp_boundary, "voltage LP")]:
    a2 = cosine_coefficient(boundary, 2)
    tail = np.sqrt(sum((cosine_coefficient(boundary, order) * probe_radii**order)**2 for order in range(6, 71, 4)))
    axes[1, 2].semilogy(probe_radii, tail / (np.abs(a2) * probe_radii**2), label=label)
axes[1, 2].set(xlabel="measurement radius / outer radius", ylabel="RMS allowed tail / quadrupole", title="Central quality depends on measurement radius")
axes[1, 2].legend()

fig.suptitle("Claude-tree stress test 01 — arc-width notation and the honest voltage-control relaxation", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: pi/3 is the first non-overlapping zero when it is the full arc width, not half-width. The LP enforces a6=a10=0 on a fixed circular actuator boundary but reduces a2 versus the simple arcs, produces a mostly bang-bang voltage pattern, and is not an upper bound over arbitrary electrode geometry.")
print("saved", OUTPUT)
