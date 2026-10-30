import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/13_claim_level_falsifiers.png"
N = 1440
theta = 2.0 * np.pi * (np.arange(N) + 0.5) / N


def harmonic_extension(boundary, max_order=40):
    axis = np.linspace(-0.98, 0.98, 241)
    xx, yy = np.meshgrid(axis, axis)
    z = xx + 1j * yy
    values = np.full(xx.shape, np.mean(boundary))
    coefficients = []
    for order in range(1, max_order + 1):
        cosine = 2.0 * np.mean(boundary * np.cos(order * theta))
        sine = 2.0 * np.mean(boundary * np.sin(order * theta))
        coefficients.append(cosine - 1j * sine)
        values += np.real((cosine - 1j * sine) * z**order)
    values[np.abs(z) > 1.0] = np.nan
    return xx, yy, values, np.array(coefficients)


quadrupole_boundary = 0.5 * np.sign(np.cos(2.0 * theta))
dodecapole_boundary = 0.5 * np.sign(np.cos(6.0 * theta))
qx, qy, qfield, qcoeff = harmonic_extension(quadrupole_boundary)
dx, dy, dfield, dcoeff = harmonic_extension(dodecapole_boundary)

# Exact exterior correction for an off-center circular grounded obstacle in incident Re(z^2).
# About the obstacle center z=d+zeta, the correction has only zeta^-1 and zeta^-2.
# Re-expanding about the trap center produces all positive Taylor orders.
ratios = np.linspace(0.08, 0.48, 300)
orders = np.arange(1, 13)
d = 1.0
scattering = np.array([
    [2.0 * ratio**2 / d**order - (order + 1.0) * ratio**4 / d**order for order in orders]
    for ratio in ratios
])
chosen_ratio = 0.30
chosen_coefficients = np.array([2.0 * chosen_ratio**2 - (order + 1.0) * chosen_ratio**4 for order in orders])

rho = 0.5
probe = rho * np.exp(1j * theta)
zeta = probe - d
exact_scattered = np.real(-2.0 * d * chosen_ratio**2 / zeta - chosen_ratio**4 / zeta**2)
series_scattered = np.real(sum(chosen_coefficients[order - 1] * probe**order for order in orders))

print("HYPOTHESIS: equal positive/negative hitting probability at the center fixes only V(0), not the multipoles; and translating a circular obstacle away from the expansion center generates a nonzero a6.")
print("quadrupole and dodecapole positive boundary fractions", np.mean(quadrupole_boundary > 0), np.mean(dodecapole_boundary > 0))
print("their center potentials", np.mean(quadrupole_boundary), np.mean(dodecapole_boundary))
print("quadrupole pattern a2,a6", qcoeff[1], qcoeff[5])
print("dodecapole pattern a2,a6", dcoeff[1], dcoeff[5])
print("off-center single-circle scattered a6 at R/d=0.3", chosen_coefficients[5])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].plot(theta / np.pi, quadrupole_boundary, label="sign cos(2 theta)")
axes[0, 0].plot(theta / np.pi, dodecapole_boundary, label="sign cos(6 theta)", alpha=0.75)
axes[0, 0].set(xlabel="boundary angle/pi", ylabel="boundary voltage", title="Same center hitting totals, different boundary labels")
axes[0, 0].legend(fontsize=8)

im0 = axes[0, 1].imshow(qfield, origin="lower", extent=[-0.98, 0.98, -0.98, 0.98], cmap="coolwarm", vmin=-0.5, vmax=0.5)
axes[0, 1].set(aspect="equal", xlabel="x/R", ylabel="y/R", title="Harmonic field: quadrupole boundary pattern")
fig.colorbar(im0, ax=axes[0, 1])

im1 = axes[0, 2].imshow(dfield, origin="lower", extent=[-0.98, 0.98, -0.98, 0.98], cmap="coolwarm", vmin=-0.5, vmax=0.5)
axes[0, 2].set(aspect="equal", xlabel="x/R", ylabel="y/R", title="Harmonic field: dodecapole boundary pattern")
fig.colorbar(im1, ax=axes[0, 2])

shown = np.arange(1, 15)
axes[1, 0].stem(shown - 0.08, np.abs(qcoeff[:14]), linefmt="C0-", markerfmt="C0o", basefmt=" ", label="quadrupole pattern")
axes[1, 0].stem(shown + 0.08, np.abs(dcoeff[:14]), linefmt="C1-", markerfmt="C1s", basefmt=" ", label="dodecapole pattern")
axes[1, 0].set_yscale("log")
axes[1, 0].set(xlabel="interior harmonic order", ylabel="absolute coefficient", title="Center value agrees; derivatives do not")
axes[1, 0].legend(fontsize=8)

for order in [2, 4, 6, 8, 10]:
    axes[1, 1].plot(ratios, scattering[:, order - 1], label=f"central a{order}")
axes[1, 1].axhline(0.0, color="black", linewidth=1)
axes[1, 1].axvline(chosen_ratio, color="grey", linestyle="--")
axes[1, 1].set(xlabel="off-center circle R/d", ylabel="scattered Taylor coefficient", title="One translated circle generates mode 6")
axes[1, 1].legend(fontsize=8)

axes[1, 2].plot(theta / np.pi, exact_scattered, label="exact Kelvin correction")
axes[1, 2].plot(theta / np.pi, series_scattered, linestyle="--", label="12-term expansion about trap center")
axes[1, 2].set(xlabel="probe angle/pi", ylabel="scattered potential at rho=0.5d", title="Translation mixes local obstacle modes globally")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 13 — exact counterexamples to two overstrong claims", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: harmonic measure at one starting point is not a sufficient multipole statistic; its spatial derivatives are required. A circular obstacle is mode-diagonal only about its own center, while an off-center rod has nonzero central a6. Branches 05 and 06 need these qualifications before use.")
print("saved", OUTPUT)
