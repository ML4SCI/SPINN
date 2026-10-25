import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


OUTPUT = "exp/harmonic_source_shape_search/results/03_elliptical_shape_map.png"
RADII = np.linspace(0.75, 1.55, 29)
ELLIPTICITIES = np.linspace(-0.24, 0.24, 17)
SOURCE_COUNT = 24
BOUNDARY_COUNT = 64
SOURCE_FRACTION = 0.58
MEASUREMENT_RADIUS = 0.5


def ellipse_points(center, radial, tangential, radial_radius, tangential_radius, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    return center + radial_radius * np.cos(theta)[:, None] * radial + tangential_radius * np.sin(theta)[:, None] * tangential


def solve(rod_radius, ellipticity):
    radial_radius = rod_radius * (1.0 + ellipticity)
    tangential_radius = rod_radius * (1.0 - ellipticity)
    center_distance = 1.0 + radial_radius
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    sources, boundary, values = [], [], []
    for radial, tangential, voltage in frames:
        center = center_distance * radial
        sources.append(ellipse_points(center, radial, tangential, SOURCE_FRACTION * radial_radius, SOURCE_FRACTION * tangential_radius, SOURCE_COUNT, 0.013))
        boundary.append(ellipse_points(center, radial, tangential, radial_radius, tangential_radius, BOUNDARY_COUNT))
        values.append(np.full(BOUNDARY_COUNT, voltage))

    outer_source_theta = 2.0 * np.pi * np.arange(4 * SOURCE_COUNT) / (4 * SOURCE_COUNT) + 0.007
    outer_boundary_theta = 2.0 * np.pi * np.arange(4 * BOUNDARY_COUNT) / (4 * BOUNDARY_COUNT)
    sources.append(6.5 * np.column_stack([np.cos(outer_source_theta), np.sin(outer_source_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(outer_boundary_theta), np.sin(outer_boundary_theta)]))
    values.append(np.zeros(4 * BOUNDARY_COUNT))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    matrix = np.column_stack([
        np.log(np.linalg.norm(boundary[:, None, :] - sources[None, :, :], axis=2)),
        np.ones(len(boundary)),
    ])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients, radial_radius, tangential_radius, center_distance


def evaluate(points, sources, coefficients):
    return np.column_stack([
        np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)),
        np.ones(len(points)),
    ]) @ coefficients


theta = 2.0 * np.pi * np.arange(720) / 720
probe = MEASUREMENT_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])


def score(rod_radius, ellipticity):
    sources, coefficients, radial_radius, tangential_radius, center_distance = solve(rod_radius, ellipticity)
    potential = evaluate(probe, sources, coefficients)
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / MEASUREMENT_RADIUS**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / MEASUREMENT_RADIUS**6
    a10 = 2.0 * np.mean(potential * np.cos(10.0 * theta)) / MEASUREMENT_RADIUS**10
    return 2.0 * a2, (a6 / a2) * MEASUREMENT_RADIUS**4, (a10 / a2) * MEASUREMENT_RADIUS**8


print("HYPOTHESIS: allowing a single smooth radial-versus-tangential ellipticity creates a curve of q6-free electrode designs; some may increase quadrupole efficiency at fixed closest ion-electrode distance, but the gain may be paid for by larger q10.")
eta = np.zeros((len(ELLIPTICITIES), len(RADII)))
q6 = np.zeros_like(eta)
q10 = np.zeros_like(eta)
for row, ellipticity in enumerate(ELLIPTICITIES):
    for column, radius in enumerate(RADII):
        eta[row, column], q6[row, column], q10[row, column] = score(float(radius), float(ellipticity))

root_radius = np.full(len(ELLIPTICITIES), np.nan)
root_eta = np.full(len(ELLIPTICITIES), np.nan)
root_q10 = np.full(len(ELLIPTICITIES), np.nan)
for row, ellipticity in enumerate(ELLIPTICITIES):
    changes = np.where(np.sign(q6[row, :-1]) != np.sign(q6[row, 1:]))[0]
    if len(changes) == 0:
        continue
    index = changes[np.argmin(np.abs(RADII[changes] - 1.145))]
    fraction = -q6[row, index] / (q6[row, index + 1] - q6[row, index])
    root_radius[row] = RADII[index] + fraction * (RADII[index + 1] - RADII[index])
    root_eta[row] = eta[row, index] + fraction * (eta[row, index + 1] - eta[row, index])
    root_q10[row] = q10[row, index] + fraction * (q10[row, index + 1] - q10[row, index])

circle_index = np.argmin(np.abs(ELLIPTICITIES))
best_index = np.nanargmax(root_eta)
circle_radius = root_radius[circle_index]
circle_eta = root_eta[circle_index]
best_ellipticity = ELLIPTICITIES[best_index]
best_radius = root_radius[best_index]
best_eta = root_eta[best_index]
best_q10 = root_q10[best_index]
print(f"circular q6-zero: R={circle_radius:.6f}, eta={circle_eta:.8f}, q10={root_q10[circle_index]:+.3e}")
print(f"best tested ellipse: epsilon={best_ellipticity:+.3f}, R={best_radius:.6f}, eta={best_eta:.8f}, relative eta gain={(best_eta/circle_eta-1.0):+.3%}, q10={best_q10:+.3e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
extent = [RADII[0], RADII[-1], ELLIPTICITIES[0], ELLIPTICITIES[-1]]
image = axes[0, 0].imshow(q6, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=-np.max(np.abs(q6)), vmax=np.max(np.abs(q6)))
axes[0, 0].contour(RADII, ELLIPTICITIES, q6, levels=[0.0], colors="black", linewidths=2)
axes[0, 0].set(xlabel="base rod radius R/r0", ylabel="ellipticity epsilon", title="Physical q6; black is q6=0")
fig.colorbar(image, ax=axes[0, 0], shrink=0.82)

image = axes[0, 1].imshow(eta, origin="lower", aspect="auto", extent=extent, cmap="viridis")
axes[0, 1].contour(RADII, ELLIPTICITIES, q6, levels=[0.0], colors="white", linewidths=2)
axes[0, 1].set(xlabel="base rod radius R/r0", ylabel="ellipticity epsilon", title="Quadrupole efficiency eta")
fig.colorbar(image, ax=axes[0, 1], shrink=0.82)

image = axes[0, 2].imshow(np.abs(q10), origin="lower", aspect="auto", extent=extent, cmap="magma")
axes[0, 2].contour(RADII, ELLIPTICITIES, q6, levels=[0.0], colors="cyan", linewidths=2)
axes[0, 2].set(xlabel="base rod radius R/r0", ylabel="ellipticity epsilon", title="Absolute q10; cyan is q6=0")
fig.colorbar(image, ax=axes[0, 2], shrink=0.82)

axes[1, 0].plot(ELLIPTICITIES, root_radius, marker="o")
axes[1, 0].axhline(circle_radius, color="black", linestyle="--", label="circular optimum")
axes[1, 0].set(xlabel="ellipticity epsilon", ylabel="R/r0 required for q6=0", title="A continuum of hexapole-free shapes")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(ELLIPTICITIES, root_eta / circle_eta - 1.0, marker="o", color="tab:green", label="relative eta")
axes[1, 1].axhline(0.0, color="black", linewidth=1)
axes[1, 1].scatter([best_ellipticity], [best_eta / circle_eta - 1.0], marker="*", s=150, color="gold", edgecolor="black", zorder=4, label="best tested")
axes[1, 1].set(xlabel="ellipticity epsilon", ylabel="eta / circular eta - 1", title="Efficiency along the physical q6=0 curve")
second = axes[1, 1].twinx()
second.plot(ELLIPTICITIES, np.abs(root_q10), marker="s", color="tab:red", alpha=0.7, label="absolute q10")
second.set_ylabel("absolute q10", color="tab:red")
axes[1, 1].legend(loc="upper left", fontsize=8)

def draw_shape(ax, radius, ellipticity, color, label):
    radial_radius = radius * (1.0 + ellipticity)
    tangential_radius = radius * (1.0 - ellipticity)
    distance = 1.0 + radial_radius
    for angle in [0.0, 90.0, 180.0, 270.0]:
        radians = np.deg2rad(angle)
        center = distance * np.array([np.cos(radians), np.sin(radians)])
        ax.add_patch(Ellipse(center, 2.0 * radial_radius, 2.0 * tangential_radius, angle=angle, fill=False, edgecolor=color, linewidth=2, label=label if angle == 0.0 else None))

draw_shape(axes[1, 2], circle_radius, 0.0, "black", "circular q6=0")
draw_shape(axes[1, 2], best_radius, best_ellipticity, "tab:green", "best tested ellipse")
axes[1, 2].add_patch(plt.Circle((0.0, 0.0), MEASUREMENT_RADIUS, fill=False, edgecolor="magenta", linestyle="--"))
axes[1, 2].set(xlim=(-3.6, 3.6), ylim=(-3.6, 3.6), aspect="equal", xlabel="x/r0", ylabel="y/r0", title="The actual electrode change")
axes[1, 2].legend(fontsize=8)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 03 — first nested search beyond circular electrodes", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: the q6-free condition is a curve, not a unique circle; judge any eta gain together with q10 and the visible electrode deformation before calling it an improved trap.")
print("saved", OUTPUT)
