import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


OUTPUT = "exp/harmonic_source_shape_search/results/04_ellipse_candidate_replication.png"
COUNTS = np.array([12, 16, 24, 32, 48])
FRACTIONS = np.array([0.45, 0.58, 0.70])
ELLIPSE_EPSILON = -0.15
MEASUREMENT_RADIUS = 0.5


def ellipse_points(center, radial, tangential, a, b, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    return center + a * np.cos(theta)[:, None] * radial + b * np.sin(theta)[:, None] * tangential


def solve(radius, epsilon, count, fraction):
    a = radius * (1.0 + epsilon)
    b = radius * (1.0 - epsilon)
    distance = 1.0 + a
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    boundary_count = 3 * count
    sources, boundary, values = [], [], []
    for radial, tangential, voltage in frames:
        center = distance * radial
        sources.append(ellipse_points(center, radial, tangential, fraction * a, fraction * b, count, 0.013))
        boundary.append(ellipse_points(center, radial, tangential, a, b, boundary_count))
        values.append(np.full(boundary_count, voltage))
    source_theta = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    boundary_theta = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    sources.append(6.5 * np.column_stack([np.cos(source_theta), np.sin(source_theta)]))
    boundary.append(5.0 * np.column_stack([np.cos(boundary_theta), np.sin(boundary_theta)]))
    values.append(np.zeros(12 * count))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    matrix = np.column_stack([np.log(np.linalg.norm(boundary[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients, a, b, distance


theta = 2.0 * np.pi * np.arange(720) / 720
probe = MEASUREMENT_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])


def score(radius, epsilon, count, fraction):
    sources, coefficients, a, b, distance = solve(radius, epsilon, count, fraction)
    matrix = np.column_stack([np.log(np.linalg.norm(probe[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(probe))])
    potential = matrix @ coefficients
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / MEASUREMENT_RADIUS**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / MEASUREMENT_RADIUS**6
    a10 = 2.0 * np.mean(potential * np.cos(10.0 * theta)) / MEASUREMENT_RADIUS**10
    return 2.0 * a2, (a6 / a2) * MEASUREMENT_RADIUS**4, (a10 / a2) * MEASUREMENT_RADIUS**8


def root_score(epsilon, count, fraction, bracket):
    root = brentq(lambda radius: score(radius, epsilon, count, fraction)[1], bracket[0], bracket[1], xtol=1e-11)
    eta, q6, q10 = score(root, epsilon, count, fraction)
    return root, eta, q10


def unseen_boundary_error(radius, epsilon, count, fraction):
    sources, coefficients, a, b, distance = solve(radius, epsilon, count, fraction)
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    points, values = [], []
    for radial, tangential, voltage in frames:
        points.append(ellipse_points(distance * radial, radial, tangential, a, b, 211, 0.0041))
        values.append(np.full(211, voltage))
    outer_theta = 2.0 * np.pi * np.arange(509) / 509 + 0.0023
    points.append(5.0 * np.column_stack([np.cos(outer_theta), np.sin(outer_theta)]))
    values.append(np.zeros(509))
    points = np.vstack(points)
    values = np.concatenate(values)
    prediction = np.column_stack([np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)), np.ones(len(points))]) @ coefficients
    return np.sqrt(np.mean((prediction - values) ** 2))


print("HYPOTHESIS: the 0.114% eta gain of the epsilon=-0.15 q6-free ellipse is physical only if it remains positive and nearly constant across source counts and source offsets, with small unseen boundary error.")
shape = (len(COUNTS), len(FRACTIONS))
circle_root = np.zeros(shape)
ellipse_root = np.zeros(shape)
eta_gain = np.zeros(shape)
q10_ratio = np.zeros(shape)
boundary_error = np.zeros(shape)

for row, count in enumerate(COUNTS):
    for column, fraction in enumerate(FRACTIONS):
        circle_radius, circle_eta, circle_q10 = root_score(0.0, int(count), float(fraction), (1.0, 1.3))
        candidate_radius, candidate_eta, candidate_q10 = root_score(ELLIPSE_EPSILON, int(count), float(fraction), (0.65, 0.90))
        circle_root[row, column] = circle_radius
        ellipse_root[row, column] = candidate_radius
        eta_gain[row, column] = candidate_eta / circle_eta - 1.0
        q10_ratio[row, column] = abs(candidate_q10) / abs(circle_q10)
        boundary_error[row, column] = unseen_boundary_error(candidate_radius, ELLIPSE_EPSILON, int(count), float(fraction))

print(f"eta gain range: {np.min(eta_gain):+.6%} to {np.max(eta_gain):+.6%}; mean={np.mean(eta_gain):+.6%}; standard deviation={np.std(eta_gain):.3e}")
print(f"circle root range: {np.min(circle_root):.8f} to {np.max(circle_root):.8f}")
print(f"ellipse root range: {np.min(ellipse_root):.8f} to {np.max(ellipse_root):.8f}")
print(f"q10 penalty ratio range: {np.min(q10_ratio):.3f} to {np.max(q10_ratio):.3f}")
print(f"unseen boundary RMSE range: {np.min(boundary_error):.3e} to {np.max(boundary_error):.3e}")
trusted = (np.abs(circle_root - 1.14521) < 2.0e-3) & (boundary_error < 1.0e-5)
trusted_gain = eta_gain[trusted]
trusted_q10_ratio = q10_ratio[trusted]
print("trusted settings after known-circle and unseen-boundary gates:", int(np.sum(trusted)), "of", trusted.size)
print(f"trusted eta gain range: {np.min(trusted_gain):+.6%} to {np.max(trusted_gain):+.6%}; mean={np.mean(trusted_gain):+.6%}")
print(f"trusted q10 penalty ratio range: {np.min(trusted_q10_ratio):.3f} to {np.max(trusted_q10_ratio):.3f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [FRACTIONS[0], FRACTIONS[-1], COUNTS[0], COUNTS[-1]]
panels = [
    (100.0 * eta_gain, "RdYlGn", "ellipse eta gain [%]"),
    (circle_root, "viridis", "recovered circular q6=0 radius"),
    (ellipse_root, "viridis", "recovered ellipse q6=0 base radius"),
    (q10_ratio, "magma", "absolute q10: ellipse / circle"),
    (np.log10(boundary_error), "magma_r", "log10 unseen ellipse boundary RMSE"),
]
for ax, (values, cmap, title) in zip(axes.ravel()[:5], panels):
    image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.set(xlabel="source radius / electrode radius", ylabel="sources per electrode", title=title)
    fig.colorbar(image, ax=ax, shrink=0.82)

for column, fraction in enumerate(FRACTIONS):
    axes[1, 2].plot(COUNTS, 100.0 * eta_gain[:, column], marker="o", label=f"source fraction {fraction:.2f}")
axes[1, 2].axhline(0.0, color="black", linewidth=1)
axes[1, 2].set(xlabel="sources per electrode", ylabel="eta gain [%]", title="A physical gain should form a resolution plateau")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.25)
fig.suptitle("Plot 04 — replicate the first noncircular candidate before believing it", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

spread = np.max(trusted_gain) - np.min(trusted_gain)
if np.min(trusted_gain) > 0.0 and spread < 0.0002:
    print("CONCLUSION: the small eta gain survives all tested discretizations and is larger than its numerical spread, but it is a tradeoff rather than dominance because q10 increases.")
else:
    print("CONCLUSION: the apparent eta gain is not separated from solver variation; reject the candidate and improve the inner solve before further shape search.")
print("saved", OUTPUT)
