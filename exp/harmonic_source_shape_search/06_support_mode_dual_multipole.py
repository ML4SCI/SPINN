import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/harmonic_source_shape_search/results/06_support_mode_dual_multipole.png"
RADII = np.linspace(0.90, 1.90, 51)
SUPPORT_C4 = np.linspace(-0.055, 0.055, 23)
COUNT = 24
BOUNDARY_COUNT = 72
SOURCE_FRACTION = 0.58
MEASUREMENT_RADIUS = 0.5


def support_boundary(radius, c4, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    p = radius * (1.0 + c4 * np.cos(4.0 * theta))
    dp = -4.0 * radius * c4 * np.sin(4.0 * theta)
    x = p * np.cos(theta) - dp * np.sin(theta)
    y = p * np.sin(theta) + dp * np.cos(theta)
    return np.column_stack([x, y])


def solve(radius, c4):
    local_boundary = support_boundary(radius, c4, BOUNDARY_COUNT)
    local_sources = SOURCE_FRACTION * support_boundary(radius, c4, COUNT, 0.013)
    inner_support = radius * (1.0 + c4)
    distance = 1.0 + inner_support
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    sources, boundary, values = [], [], []
    for radial, tangential, voltage in frames:
        rotation = np.column_stack([radial, tangential])
        center = distance * radial
        sources.append(center + local_sources @ rotation.T)
        boundary.append(center + local_boundary @ rotation.T)
        values.append(np.full(BOUNDARY_COUNT, voltage))
    source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary_theta = 2.0 * np.pi * np.arange(4 * BOUNDARY_COUNT) / (4 * BOUNDARY_COUNT)
    sources.append(6.5 * np.column_stack([np.cos(source_theta), np.sin(source_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(boundary_theta), np.sin(boundary_theta)]))
    values.append(np.zeros(4 * BOUNDARY_COUNT))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    matrix = np.column_stack([np.log(np.linalg.norm(boundary[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients, distance, local_boundary


theta = 2.0 * np.pi * np.arange(720) / 720
probe = MEASUREMENT_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])


def score(radius, c4):
    sources, coefficients, _, _ = solve(radius, c4)
    matrix = np.column_stack([np.log(np.linalg.norm(probe[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(probe))])
    potential = matrix @ coefficients
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / MEASUREMENT_RADIUS**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / MEASUREMENT_RADIUS**6
    a10 = 2.0 * np.mean(potential * np.cos(10.0 * theta)) / MEASUREMENT_RADIUS**10
    return 2.0 * a2, (a6 / a2) * MEASUREMENT_RADIUS**4, (a10 / a2) * MEASUREMENT_RADIUS**8


print("HYPOTHESIS: a convex local support-function mode p(theta)=R[1+c4 cos(4 theta)] can control q10 independently enough that radius cancels q6 and c4 cancels q10, producing a smoother finite-amplitude field than the ellipse candidate.")
eta = np.zeros((len(SUPPORT_C4), len(RADII)))
q6 = np.zeros_like(eta)
q10 = np.zeros_like(eta)
for row, c4 in enumerate(SUPPORT_C4):
    for column, radius in enumerate(RADII):
        eta[row, column], q6[row, column], q10[row, column] = score(float(radius), float(c4))

root_radius = np.full(len(SUPPORT_C4), np.nan)
root_eta = np.full(len(SUPPORT_C4), np.nan)
root_q10 = np.full(len(SUPPORT_C4), np.nan)
for row, c4 in enumerate(SUPPORT_C4):
    changes = np.where(np.sign(q6[row, :-1]) != np.sign(q6[row, 1:]))[0]
    if len(changes) == 0:
        continue
    index = changes[np.argmin(np.abs(RADII[changes] - 1.145))]
    weight = -q6[row, index] / (q6[row, index + 1] - q6[row, index])
    root_radius[row] = RADII[index] + weight * (RADII[index + 1] - RADII[index])
    root_eta[row] = eta[row, index] + weight * (eta[row, index + 1] - eta[row, index])
    root_q10[row] = q10[row, index] + weight * (q10[row, index + 1] - q10[row, index])

finite_pairs = np.isfinite(root_q10[:-1]) & np.isfinite(root_q10[1:])
dual_changes = np.where(finite_pairs & (np.sign(root_q10[:-1]) != np.sign(root_q10[1:])))[0]
dual_exists = len(dual_changes) > 0
if dual_exists:
    index = dual_changes[np.argmin(np.abs(SUPPORT_C4[dual_changes]))]
    weight = -root_q10[index] / (root_q10[index + 1] - root_q10[index])
    dual_c4 = SUPPORT_C4[index] + weight * (SUPPORT_C4[index + 1] - SUPPORT_C4[index])
    dual_radius = root_radius[index] + weight * (root_radius[index + 1] - root_radius[index])
    dual_eta = root_eta[index] + weight * (root_eta[index + 1] - root_eta[index])
else:
    index = np.nanargmin(np.abs(root_q10))
    dual_c4 = SUPPORT_C4[index]
    dual_radius = root_radius[index]
    dual_eta = root_eta[index]

circle_index = np.argmin(np.abs(SUPPORT_C4))
circle_eta = root_eta[circle_index]
minimum_curvature = dual_radius * (1.0 - 15.0 * abs(dual_c4))
print(f"circle: R={root_radius[circle_index]:.7f}, eta={circle_eta:.9f}, q10={root_q10[circle_index]:+.3e}")
print(f"dual-zero crossing exists={dual_exists}; candidate c4={dual_c4:+.6f}, R={dual_radius:.7f}, eta={dual_eta:.9f}, relative eta={(dual_eta/circle_eta-1.0):+.3%}, minimum curvature radius={minimum_curvature:.6f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
extent = [RADII[0], RADII[-1], SUPPORT_C4[0], SUPPORT_C4[-1]]
image = axes[0, 0].imshow(q6, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=-np.max(np.abs(q6)), vmax=np.max(np.abs(q6)))
axes[0, 0].contour(RADII, SUPPORT_C4, q6, levels=[0.0], colors="black", linewidths=2)
axes[0, 0].set(xlabel="support radius R/r0", ylabel="support coefficient c4", title="q6; black curve is q6=0")
fig.colorbar(image, ax=axes[0, 0], shrink=0.82)

image = axes[0, 1].imshow(q10, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=-np.max(np.abs(q10)), vmax=np.max(np.abs(q10)))
axes[0, 1].contour(RADII, SUPPORT_C4, q6, levels=[0.0], colors="black", linewidths=2)
axes[0, 1].contour(RADII, SUPPORT_C4, q10, levels=[0.0], colors="cyan", linewidths=2)
axes[0, 1].set(xlabel="support radius R/r0", ylabel="support coefficient c4", title="q10; cyan curve is q10=0")
fig.colorbar(image, ax=axes[0, 1], shrink=0.82)

image = axes[0, 2].imshow(eta, origin="lower", aspect="auto", extent=extent, cmap="viridis")
axes[0, 2].contour(RADII, SUPPORT_C4, q6, levels=[0.0], colors="white", linewidths=2)
axes[0, 2].set(xlabel="support radius R/r0", ylabel="support coefficient c4", title="eta with q6=0 overlay")
fig.colorbar(image, ax=axes[0, 2], shrink=0.82)

axes[1, 0].plot(SUPPORT_C4, root_radius, marker="o")
axes[1, 0].scatter([dual_c4], [dual_radius], marker="*", s=160, color="gold", edgecolor="black", zorder=4)
axes[1, 0].set(xlabel="c4", ylabel="R/r0 on q6=0", title="Radius continues to remove q6")

axes[1, 1].plot(SUPPORT_C4, root_q10, marker="o", color="tab:red", label="q10 along q6=0")
axes[1, 1].axhline(0.0, color="black", linewidth=1)
axes[1, 1].scatter([dual_c4], [0.0 if dual_exists else root_q10[index]], marker="*", s=160, color="gold", edgecolor="black", zorder=4)
axes[1, 1].set(xlabel="c4", ylabel="q10", title="Does the new mode remove the next multipole?")

circle_sources, circle_coefficients, circle_distance, circle_local = solve(root_radius[circle_index], 0.0)
dual_sources, dual_coefficients, dual_distance, dual_local = solve(dual_radius, dual_c4)
for local, distance, color, label in [
    (circle_local, circle_distance, "black", "circular q6=0"),
    (dual_local, dual_distance, "tab:blue", "support-mode candidate"),
]:
    for angle, radial, tangential in [
        (0, np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (90, np.array([0.0, 1.0]), np.array([-1.0, 0.0])),
        (180, np.array([-1.0, 0.0]), np.array([0.0, -1.0])),
        (270, np.array([0.0, -1.0]), np.array([1.0, 0.0])),
    ]:
        rotation = np.column_stack([radial, tangential])
        points = distance * radial + local @ rotation.T
        axes[1, 2].plot(points[:, 0], points[:, 1], color=color, linewidth=2, label=label if angle == 0 else None)
axes[1, 2].set(xlim=(-3.6, 3.6), ylim=(-3.6, 3.6), aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Support-function geometry")
axes[1, 2].legend(fontsize=8)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 06 — use a shape mode to target the next allowed multipole", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

if dual_exists and minimum_curvature > 0.0:
    print("CONCLUSION: the support mode found a convex q6=q10=0 candidate; it must now pass resolution and finite-amplitude validation before being called an improvement.")
else:
    print("CONCLUSION: this single convex support mode cannot remove q10 in the tested range; add a different mode rather than widening c4 beyond the convexity limit.")
print("saved", OUTPUT)
