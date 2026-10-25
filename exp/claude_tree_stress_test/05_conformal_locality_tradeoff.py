import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/claude_tree_stress_test/results/05_conformal_locality_tradeoff.png"
ROD_RADIUS = 1.1451
ROD_CENTER = 1.0 + ROD_RADIUS
OUTER_RADIUS = 5.0
TARGET_MOTION = 0.15
COLLAR_WIDTH = 0.35
CENTER_FIXED = 0.40
CENTER_RELEASE = 0.85
OUTER_FIXED_WIDTH = 0.50


def smootherstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


centers_complex = ROD_CENTER * np.array([1.0, -1.0, 1.0j, -1.0j])
centers = np.column_stack([centers_complex.real, centers_complex.imag])


def design_mask(points):
    diff = points[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    surface = (dist2 - ROD_RADIUS**2) / (2.0 * ROD_RADIUS)
    bumps = np.exp(-(surface / COLLAR_WIDTH) ** 2)
    collars = 1.0 - np.prod(1.0 - bumps, axis=1)
    radius2 = np.sum(points**2, axis=1)
    center = smootherstep((radius2 - CENTER_FIXED**2) / (CENTER_RELEASE**2 - CENTER_FIXED**2))
    release = OUTER_RADIUS - OUTER_FIXED_WIDTH
    outer = smootherstep((OUTER_RADIUS**2 - radius2) / (OUTER_RADIUS**2 - release**2))
    return collars * center * outer


def collar_displacement(points):
    diff = points[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(diff, axis=2)
    nearest = np.argmin(np.abs(distance - ROD_RADIUS), axis=1)
    selected = diff[np.arange(len(points)), nearest]
    normal = selected / np.maximum(np.linalg.norm(selected, axis=1, keepdims=True), 1e-12)
    return TARGET_MOTION * design_mask(points)[:, None] * normal


def rational_shape(z):
    return z**5 / (z**4 - ROD_CENTER**4) ** 2


def rational_shape_derivative(z):
    denominator = z**4 - ROD_CENTER**4
    return 5.0 * z**4 / denominator**2 - 8.0 * z**8 / denominator**3


boundary_theta = np.linspace(0.0, 2.0 * np.pi, 1200, endpoint=False)
rod_boundaries = np.concatenate([center + ROD_RADIUS * np.exp(1j * boundary_theta) for center in centers_complex])
unit_boundary_motion = np.abs(rational_shape(rod_boundaries))
epsilon = TARGET_MOTION / np.max(unit_boundary_motion)

axis = np.linspace(-5.0, 5.0, 401)
xx, yy = np.meshgrid(axis, axis)
zz = xx + 1j * yy
points = np.column_stack([xx.ravel(), yy.ravel()])
inside_outer = np.abs(zz) < OUTER_RADIUS
inside_rods = np.zeros_like(inside_outer)
for center in centers_complex:
    inside_rods |= np.abs(zz - center) < ROD_RADIUS
vacuum = inside_outer & ~inside_rods

rational = epsilon * rational_shape(zz)
rational_derivative = epsilon * rational_shape_derivative(zz)
collar = collar_displacement(points).reshape(xx.shape + (2,))

# Cauchy-Riemann defect for u+iv: (u_x-v_y, u_y+v_x).
du_dy, du_dx = np.gradient(collar[:, :, 0], axis, axis)
dv_dy, dv_dx = np.gradient(collar[:, :, 1], axis, axis)
collar_cr = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2)
rational_cr = np.zeros_like(xx)

rational_magnitude = np.abs(rational)
collar_magnitude = np.linalg.norm(collar, axis=2)
for array in [rational_magnitude, collar_magnitude, collar_cr, rational_cr]:
    array[~vacuum] = np.nan

scales = np.linspace(0.0, 0.22, 100)
outer_z = OUTER_RADIUS * np.exp(1j * boundary_theta)
center_z = 0.4 * np.exp(1j * boundary_theta)
unit_scale = 1.0 / np.max(unit_boundary_motion)
outer_leak = np.array([np.max(np.abs(scale * unit_scale * rational_shape(outer_z))) for scale in scales])
center_leak = np.array([np.max(np.abs(scale * unit_scale * rational_shape(center_z))) for scale in scales])
min_conformal_scale = []
sample_vacuum = zz[vacuum][::12]
for scale in scales:
    derivative = 1.0 + scale * unit_scale * rational_shape_derivative(sample_vacuum)
    min_conformal_scale.append(np.min(np.abs(derivative)))
min_conformal_scale = np.array(min_conformal_scale)

print("HYPOTHESIS: an exactly conformal deformation can delete the Laplace residual, but it cannot also be exactly identity on an open center anchor. Rational poles hidden inside electrodes may give approximate locality while preserving f(0)=0 and f'(0)=1.")
print("rational epsilon", epsilon)
print("max rational center-disc displacement", np.nanmax(np.abs(epsilon * rational_shape(center_z))))
print("max rational outer-boundary displacement", np.nanmax(np.abs(epsilon * rational_shape(outer_z))))
print("minimum |f prime| on sampled vacuum", np.nanmin(np.abs(1.0 + rational_derivative[vacuum])))
print("collar center and outer maximum displacement", np.nanmax(collar_magnitude[np.abs(zz) <= CENTER_FIXED]), np.nanmax(collar_magnitude[np.abs(zz) >= OUTER_RADIUS - 0.01]))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [-5, 5, -5, 5]
im0 = axes[0, 0].imshow(rational_magnitude, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=TARGET_MOTION)
axes[0, 0].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Conformal rational displacement magnitude")
fig.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(collar_magnitude, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=TARGET_MOTION)
axes[0, 1].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Current C2 collar displacement magnitude")
fig.colorbar(im1, ax=axes[0, 1])

finite_cr = collar_cr[np.isfinite(collar_cr)]
im2 = axes[0, 2].imshow(np.log10(np.maximum(collar_cr, 1e-12)), origin="lower", extent=extent, cmap="magma", vmin=-6, vmax=np.log10(np.percentile(finite_cr, 99)))
axes[0, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Collar log10 Cauchy-Riemann defect")
fig.colorbar(im2, ax=axes[0, 2])

mapped_rods = rod_boundaries + epsilon * rational_shape(rod_boundaries)
for center in centers_complex:
    original = center + ROD_RADIUS * np.exp(1j * boundary_theta)
    axes[1, 0].plot(original.real, original.imag, color="grey", linewidth=0.8)
for index in range(4):
    segment = mapped_rods[index * len(boundary_theta):(index + 1) * len(boundary_theta)]
    axes[1, 0].plot(segment.real, segment.imag, color="tab:blue")
axes[1, 0].set(aspect="equal", xlim=(-3.6, 3.6), ylim=(-3.6, 3.6), xlabel="x/r0", ylabel="y/r0", title="Physical boundaries from f(z)=z+g(z)")

line = np.linspace(0.0, OUTER_RADIUS, 600)
line_z = line.astype(complex)
line_points = np.column_stack([line, np.zeros_like(line)])
axes[1, 1].semilogy(line, np.abs(epsilon * rational_shape(line_z)), label="conformal rational")
axes[1, 1].semilogy(line, np.linalg.norm(collar_displacement(line_points), axis=1), label="C2 collar")
axes[1, 1].axvspan(0.0, CENTER_FIXED, color="magenta", alpha=0.12, label="exact center anchor")
axes[1, 1].axvline(OUTER_RADIUS, color="black", linestyle="--")
axes[1, 1].set(xlabel="x/r0", ylabel="displacement magnitude", ylim=(1e-12, 1), title="Exact anchors versus analytic tails")
axes[1, 1].legend(fontsize=8)

axes[1, 2].plot(scales, outer_leak, label="outer-boundary leakage")
axes[1, 2].plot(scales, center_leak, label="center-disc leakage")
axes[1, 2].plot(scales, min_conformal_scale, label="minimum |f'| in vacuum")
axes[1, 2].axvline(TARGET_MOTION, color="black", linestyle="--", label="tested electrode motion")
axes[1, 2].set(xlabel="maximum electrode displacement", ylabel="magnitude", title="Locality and injectivity tradeoff")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 05 — conformal invariance versus the required anchors", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: strict conformality and an exact open center anchor cannot coexist nontrivially by the identity theorem. The rational D4 map is a promising separate low-dimensional geometry family because its poles lie inside electrodes and its center jet is fixed, but it only approximates the outer anchor and must pass injectivity and boundary-voltage tests.")
print("saved", OUTPUT)
