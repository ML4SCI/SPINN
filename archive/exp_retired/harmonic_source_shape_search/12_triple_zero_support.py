import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root


OUTPUT = "exp/harmonic_source_shape_search/results/15_triple_zero_support.png"
FRACTION = 0.70
PROBE_RADIUS = 0.5


def local_support(radius, c4, c8, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    p = radius * (1.0 + c4 * np.cos(4.0 * theta) + c8 * np.cos(8.0 * theta))
    dp = radius * (-4.0 * c4 * np.sin(4.0 * theta) - 8.0 * c8 * np.sin(8.0 * theta))
    points = np.column_stack([p * np.cos(theta) - dp * np.sin(theta), p * np.sin(theta) + dp * np.cos(theta)])
    return theta, points


def solve(radius, c4, c8, count):
    _, local_boundary = local_support(radius, c4, c8, 3 * count)
    _, local_sources = local_support(radius, c4, c8, count, 0.013)
    local_sources *= FRACTION
    inward_support = radius * (1.0 + c4 + c8)
    distance = 1.0 + inward_support
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
        values.append(np.full(len(local_boundary), voltage))
    theta_s = 2.0 * np.pi * np.arange(4 * count) / (4 * count) + 0.007
    theta_b = 2.0 * np.pi * np.arange(12 * count) / (12 * count)
    sources.append(6.5 * np.column_stack([np.cos(theta_s), np.sin(theta_s)]))
    boundary.append(5.0 * np.column_stack([np.cos(theta_b), np.sin(theta_b)]))
    values.append(np.zeros(len(theta_b)))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    values = np.concatenate(values)
    delta = boundary[:, None, :] - sources[None, :, :]
    matrix = np.column_stack([0.5 * np.log(np.sum(delta**2, axis=2)), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients, distance, boundary, values


def evaluate(points, sources, coefficients):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = 0.5 * np.log(radius2) @ coefficients[:-1] + coefficients[-1]
    field = -np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, field


theta = 2.0 * np.pi * np.arange(1440) / 1440
probe = PROBE_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])


def multipoles(variables, count):
    radius, c4, c8 = variables
    sources, coefficients, *_ = solve(radius, c4, c8, count)
    values = evaluate(probe, sources, coefficients)[0]
    amplitudes = np.array([
        2.0 * np.mean(values * np.cos(order * theta)) / PROBE_RADIUS**order
        for order in [2, 6, 10, 14, 18]
    ])
    a2 = amplitudes[0]
    physical = np.array([(amplitudes[i] / a2) * PROBE_RADIUS ** (4 * i) for i in range(1, 5)])
    return 2.0 * a2, physical


def convexity(radius, c4, c8):
    angle = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)
    curvature_radius = radius * (1.0 - 15.0 * c4 * np.cos(4.0 * angle) - 63.0 * c8 * np.cos(8.0 * angle))
    return curvature_radius.min(), curvature_radius


print("HYPOTHESIS: a third convex support mode can cancel q14 as well as q6 and q10, but the 63x curvature leverage of c8 may make the solution non-convex or impractical.")
counts = [32, 40, 48, 56, 64]
initial = np.array([1.63053, 0.02695, 0.0])
records = []
solutions = []
for count in counts:
    result = root(lambda variables: multipoles(variables, count)[1][:3], initial, options={"xtol": 1e-10})
    initial = result.x
    radius, c4, c8 = result.x
    eta, qs = multipoles(result.x, count)
    minimum_curvature, _ = convexity(radius, c4, c8)
    sources, coefficients, distance, boundary, target = solve(radius, c4, c8, count)
    unseen_theta = 2.0 * np.pi * (np.arange(503) + 0.31) / 503
    _, unseen_local = local_support(radius, c4, c8, len(unseen_theta), 2.0 * np.pi * 0.31 / 503)
    unseen = np.array([distance, 0.0]) + unseen_local
    unseen_error = np.sqrt(np.mean((evaluate(unseen, sources, coefficients)[0] - 0.5) ** 2))
    outer_clearance = 5.0 - (distance + radius * (1.0 + c4 + c8))
    records.append([count, radius, c4, c8, eta, *qs, minimum_curvature, outer_clearance, unseen_error])
    solutions.append(result.success)
    print(f"count={count}: success={result.success}, R={radius:.9f}, c4={c4:+.9f}, c8={c8:+.9f}, eta={eta:.9f}, q={qs}, min curvature={minimum_curvature:.6f}, outer clearance={outer_clearance:.6f}, unseen={unseen_error:.3e}")
records = np.array(records)

final = records[-1]
radius, c4, c8 = final[1:4]
eta, qs = multipoles([radius, c4, c8], 64)
dual_eta, dual_qs = multipoles([1.630529854, 0.026949792, 0.0], 64)
amplitudes = np.linspace(0.1, 0.95, 50)


def multipole_error_curve(q_values):
    orders = np.array([6, 10, 14, 18])
    return np.sqrt(np.sum((0.5 * orders[:, None] * q_values[:, None] * (amplitudes[None, :] / PROBE_RADIUS) ** (orders[:, None] - 2)) ** 2, axis=0))


fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
angle, local = local_support(radius, c4, c8, 1000)
dual_angle, dual_local = local_support(1.630529854, 0.026949792, 0.0, 1000)
distance = 1.0 + radius * (1.0 + c4 + c8)
dual_distance = 1.0 + 1.630529854 * (1.0 + 0.026949792)
for points, center, label, color in [(dual_local, dual_distance, "dual-zero", "tab:blue"), (local, distance, "triple-zero", "tab:orange")]:
    axes[0, 0].plot(center + points[:, 0], points[:, 1], color=color, label=label)
axes[0, 0].set(aspect="equal", xlim=(0.8, 5.0), ylim=(-2.2, 2.2), xlabel="x/r0", ylabel="y/r0", title="Right-electrode geometry")
axes[0, 0].legend()

for column, label in zip([1, 2, 3], ["R", "c4", "c8"]):
    axes[0, 1].plot(records[:, 0], records[:, column], marker="o", label=label)
axes[0, 1].set(xlabel="sources per electrode", title="Does the geometry converge?")
axes[0, 1].legend()

axes[0, 2].semilogy(records[:, 0], np.abs(records[:, 5]), marker="o", label="q6")
axes[0, 2].semilogy(records[:, 0], np.abs(records[:, 6]), marker="o", label="q10")
axes[0, 2].semilogy(records[:, 0], np.abs(records[:, 7]), marker="o", label="q14")
axes[0, 2].semilogy(records[:, 0], np.abs(records[:, 8]), marker="o", label="q18")
axes[0, 2].set(xlabel="sources per electrode", ylabel="absolute physical multipole", title="Cancelled and remaining modes")
axes[0, 2].legend(fontsize=8)

minimum_curvature, curvature_curve = convexity(radius, c4, c8)
dual_curvature, dual_curve = convexity(1.630529854, 0.026949792, 0.0)
curvature_angle = np.linspace(0.0, 2.0 * np.pi, len(curvature_curve), endpoint=False)
axes[1, 0].plot(curvature_angle, dual_curve, label="dual-zero")
axes[1, 0].plot(curvature_angle, curvature_curve, label="triple-zero attempt")
axes[1, 0].axhline(0.0, color="black", linewidth=1)
axes[1, 0].set(xlabel="local boundary angle", ylabel="radius of curvature / r0", title="Convexity cost of the extra mode")
axes[1, 0].legend()

axes[1, 1].semilogy(amplitudes, multipole_error_curve(dual_qs), label="dual-zero")
axes[1, 1].semilogy(amplitudes, multipole_error_curve(qs), label="triple-zero")
axes[1, 1].set(xlabel="amplitude / r0", ylabel="multipole E-error proxy", title="Predicted usable-field extension")
axes[1, 1].legend()

labels = ["eta", "min curvature", "outer clearance"]
dual_values = [1.0, dual_curvature, 5.0 - 2.0 * dual_distance + 1.0]
triple_values = [eta / dual_eta, minimum_curvature / dual_curvature, final[10] / (5.0 - 2.0 * dual_distance + 1.0)]
axes[1, 2].bar(np.arange(3), triple_values, color=["tab:green", "tab:purple", "tab:red"])
axes[1, 2].axhline(1.0, color="black", linewidth=1)
axes[1, 2].set_xticks(np.arange(3), labels, rotation=15)
axes[1, 2].set(ylabel="triple-zero / dual-zero", title="Benefit versus geometric cost")
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 15 — failed triple-zero attempt: cancellation runs into enclosure clearance", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: accept the third mode only if the root converges across resolutions and remains convex with useful clearance; otherwise the dual-zero design is the honest stopping point.")
print("saved", OUTPUT)
