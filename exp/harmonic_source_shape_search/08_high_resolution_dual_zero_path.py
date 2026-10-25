import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, root


OUTPUT = "exp/harmonic_source_shape_search/results/08_high_resolution_dual_zero_path.png"
SUMMARY_OUTPUT = "exp/harmonic_source_shape_search/results/09_final_dual_zero_summary.png"
DYNAMICS_OUTPUT = "exp/harmonic_source_shape_search/results/10_rf_trajectory_stress_test.png"
ROBUSTNESS_OUTPUT = "exp/harmonic_source_shape_search/results/11_manufacturing_robustness.png"
COUNTS = np.array([24, 32, 40, 48, 56, 64])
FRACTIONS = [0.58, 0.70]
MEASUREMENT_RADIUS = 0.5


def local_support(radius, c4, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    p = radius * (1.0 + c4 * np.cos(4.0 * theta))
    dp = -4.0 * radius * c4 * np.sin(4.0 * theta)
    return np.column_stack([p * np.cos(theta) - dp * np.sin(theta), p * np.sin(theta) + dp * np.cos(theta)])


def solve(radius, c4, count, fraction):
    local_boundary = local_support(radius, c4, 3 * count)
    local_sources = fraction * local_support(radius, c4, count, 0.013)
    distance = 1.0 + radius * (1.0 + c4)
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
        values.append(np.full(3 * count, voltage))
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
    return sources, coefficients, distance


def evaluate(points, sources, coefficients):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = 0.5 * np.log(radius2) @ coefficients[:-1] + coefficients[-1]
    field = -np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, field


theta = 2.0 * np.pi * np.arange(720) / 720
probe = MEASUREMENT_RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])


def multipoles(radius, c4, count, fraction):
    sources, coefficients, _ = solve(radius, c4, count, fraction)
    potential = evaluate(probe, sources, coefficients)[0]
    harmonics = [2.0 * np.mean(potential * np.cos(order * theta)) / MEASUREMENT_RADIUS**order for order in [2, 6, 10, 14]]
    a2, a6, a10, a14 = harmonics
    return 2.0 * a2, (a6 / a2) * MEASUREMENT_RADIUS**4, (a10 / a2) * MEASUREMENT_RADIUS**8, (a14 / a2) * MEASUREMENT_RADIUS**12


def boundary_error(radius, c4, count, fraction):
    sources, coefficients, distance = solve(radius, c4, count, fraction)
    local = local_support(radius, c4, 307, 0.0031)
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), +0.5),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0]), +0.5),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0]), -0.5),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0]), -0.5),
    ]
    points, values = [], []
    for radial, tangential, voltage in frames:
        points.append(distance * radial + local @ np.column_stack([radial, tangential]).T)
        values.append(np.full(len(local), voltage))
    outer_theta = 2.0 * np.pi * np.arange(613) / 613 + 0.0023
    points.append(5.0 * np.column_stack([np.cos(outer_theta), np.sin(outer_theta)]))
    values.append(np.zeros(613))
    return np.sqrt(np.mean((evaluate(np.vstack(points), sources, coefficients)[0] - np.concatenate(values)) ** 2))


def field_error(radius, c4, count, fraction, amplitude=0.9):
    sources, coefficients, _ = solve(radius, c4, count, fraction)
    eta = multipoles(radius, c4, count, fraction)[0]
    points = amplitude * np.column_stack([np.cos(theta), np.sin(theta)])
    field = evaluate(points, sources, coefficients)[1]
    a2 = 0.5 * eta
    ideal = np.column_stack([-2.0 * a2 * points[:, 0], 2.0 * a2 * points[:, 1]])
    return np.sqrt(np.mean(np.sum((field - ideal) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(ideal**2, axis=1)))


print("HYPOTHESIS: increasing source resolution produces a plateau in the dual-zero geometry and its physical payoff; if R, c4, eta ratio, and 0.9r0 field-error ratio keep drifting, the candidate is not resolved.")
records = {fraction: [] for fraction in FRACTIONS}
for fraction in FRACTIONS:
    initial = np.array([1.63, 0.027])
    for count in COUNTS:
        solution = root(lambda variables: np.array(multipoles(variables[0], variables[1], int(count), fraction)[1:3]), initial, options={"xtol": 1e-10}).x
        initial = solution
        radius, c4 = solution
        eta, q6, q10, q14 = multipoles(radius, c4, int(count), fraction)
        circle_radius = brentq(lambda value: multipoles(value, 0.0, int(count), fraction)[1], 1.0, 1.3, xtol=1e-11)
        circle_eta = multipoles(circle_radius, 0.0, int(count), fraction)[0]
        ratio = field_error(radius, c4, int(count), fraction) / field_error(circle_radius, 0.0, int(count), fraction)
        records[fraction].append((count, radius, c4, eta / circle_eta, q14, boundary_error(radius, c4, int(count), fraction), ratio, q6, q10))
    records[fraction] = np.array(records[fraction])
    final = records[fraction][-1]
    print(f"fraction={fraction:.2f} count=64: R={final[1]:.9f}, c4={final[2]:.9f}, eta ratio={final[3]:.9f}, q14={final[4]:+.3e}, boundary RMSE={final[5]:.3e}, field ratio={final[6]:.5f}, q6={final[7]:+.2e}, q10={final[8]:+.2e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for fraction, values in records.items():
    label = f"source fraction {fraction:.2f}"
    axes[0, 0].plot(values[:, 0], values[:, 1], marker="o", label=label)
    axes[0, 1].plot(values[:, 0], values[:, 2], marker="o", label=label)
    axes[0, 2].plot(values[:, 0], values[:, 3], marker="o", label=label)
    axes[1, 0].semilogy(values[:, 0], np.abs(values[:, 4]), marker="o", label=label)
    axes[1, 1].semilogy(values[:, 0], values[:, 5], marker="o", label=label)
    axes[1, 2].plot(values[:, 0], values[:, 6], marker="o", label=label)

axes[0, 0].set(ylabel="R/r0", title="Dual-zero support radius")
axes[0, 1].set(ylabel="c4", title="Dual-zero support coefficient")
axes[0, 2].axhline(1.0, color="black", linewidth=1)
axes[0, 2].set(ylabel="eta dual / eta circle", title="Confinement tradeoff")
axes[1, 0].set(ylabel="absolute physical q14", title="First remaining allowed multipole")
axes[1, 1].set(ylabel="unseen boundary RMSE", title="Independent boundary gate")
axes[1, 2].axhline(1.0, color="black", linewidth=1)
axes[1, 2].set(ylabel="E nonlinearity dual / circle", title="Finite-amplitude payoff at 0.9 r0")
for ax in axes.ravel():
    ax.set_xlabel("sources per electrode")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Plot 08 — high-resolution path for the q6=q10=0 support electrode", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

last_a = records[FRACTIONS[0]][-1]
last_b = records[FRACTIONS[1]][-1]
print(f"cross-source-placement disagreement at count64: dR={abs(last_a[1]-last_b[1]):.3e}, dc4={abs(last_a[2]-last_b[2]):.3e}, d eta ratio={abs(last_a[3]-last_b[3]):.3e}, d field ratio={abs(last_a[6]-last_b[6]):.3e}")
print("CONCLUSION: use the final two curves to decide whether the dual-zero shape is numerically resolved or still source-placement dependent.")
print("saved", OUTPUT)


count = 64
fraction = 0.70
dual_radius = last_b[1]
dual_c4 = last_b[2]
circle_radius = brentq(lambda value: multipoles(value, 0.0, count, fraction)[1], 1.0, 1.3, xtol=1e-12)
circle_eta, circle_q6, circle_q10, circle_q14 = multipoles(circle_radius, 0.0, count, fraction)
dual_eta, dual_q6, dual_q10, dual_q14 = multipoles(dual_radius, dual_c4, count, fraction)
circle_sources, circle_coefficients, circle_distance = solve(circle_radius, 0.0, count, fraction)
dual_sources, dual_coefficients, dual_distance = solve(dual_radius, dual_c4, count, fraction)

amplitudes = np.linspace(0.1, 0.95, 40)
circle_curve = np.array([field_error(circle_radius, 0.0, count, fraction, value) for value in amplitudes])
dual_curve = np.array([field_error(dual_radius, dual_c4, count, fraction, value) for value in amplitudes])

surface_theta = 2.0 * np.pi * np.arange(720) / 720
circle_surface = np.array([circle_distance, 0.0]) + local_support(circle_radius, 0.0, 720)
dual_surface = np.array([dual_distance, 0.0]) + local_support(dual_radius, dual_c4, 720)
circle_surface_max = np.max(np.linalg.norm(evaluate(circle_surface, circle_sources, circle_coefficients)[1], axis=1))
dual_surface_max = np.max(np.linalg.norm(evaluate(dual_surface, dual_sources, dual_coefficients)[1], axis=1))
circle_curvature = circle_radius
dual_curvature = dual_radius * (1.0 - 15.0 * abs(dual_c4))
circle_area = np.pi * circle_radius**2
dual_area = np.pi * dual_radius**2 * (1.0 - 7.5 * dual_c4**2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
for radius, c4, distance, color, label in [
    (circle_radius, 0.0, circle_distance, "black", "circular q6=0"),
    (dual_radius, dual_c4, dual_distance, "tab:blue", "support q6=q10=0"),
]:
    local = local_support(radius, c4, 720)
    frames = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([-1.0, 0.0]), np.array([0.0, -1.0])),
        (np.array([0.0, 1.0]), np.array([-1.0, 0.0])),
        (np.array([0.0, -1.0]), np.array([1.0, 0.0])),
    ]
    for index, (radial, tangential) in enumerate(frames):
        points = distance * radial + local @ np.column_stack([radial, tangential]).T
        axes[0, 0].plot(points[:, 0], points[:, 1], color=color, linewidth=2, label=label if index == 0 else None)
axes[0, 0].add_patch(plt.Circle((0.0, 0.0), 0.9, fill=False, edgecolor="magenta", linestyle="--", label="0.9 r0 amplitude"))
axes[0, 0].set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Resolved physical geometries")
axes[0, 0].legend(fontsize=8)

orders = [6, 10, 14]
circle_modes = np.maximum(np.abs([circle_q6, circle_q10, circle_q14]) * np.array([(0.9 / 0.5) ** 4, (0.9 / 0.5) ** 8, (0.9 / 0.5) ** 12]), 1e-14)
dual_modes = np.maximum(np.abs([dual_q6, dual_q10, dual_q14]) * np.array([(0.9 / 0.5) ** 4, (0.9 / 0.5) ** 8, (0.9 / 0.5) ** 12]), 1e-14)
x = np.arange(3)
axes[0, 1].bar(x - 0.18, circle_modes, 0.36, color="black", label="circle")
axes[0, 1].bar(x + 0.18, dual_modes, 0.36, color="tab:blue", label="dual-zero")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xticks(x, ["q6", "q10", "q14"])
axes[0, 1].set(ylabel="mode / quadrupole at 0.9 r0", title="First allowed field contaminants")
axes[0, 1].legend(fontsize=8)

axes[0, 2].semilogy(amplitudes, circle_curve, color="black", label="circle")
axes[0, 2].semilogy(amplitudes, dual_curve, color="tab:blue", label="dual-zero")
axes[0, 2].set(xlabel="ion amplitude / r0", ylabel="relative E nonlinearity", title="Usable-field improvement")
axes[0, 2].legend(fontsize=8)

axis = np.linspace(-0.95, 0.95, 151)
gx, gy = np.meshgrid(axis, axis)
grid = np.column_stack([gx.ravel(), gy.ravel()])
maps = []
for eta_value, sources, coefficients in [
    (circle_eta, circle_sources, circle_coefficients),
    (dual_eta, dual_sources, dual_coefficients),
]:
    potential = evaluate(grid, sources, coefficients)[0]
    ideal = 0.5 * eta_value * (grid[:, 0] ** 2 - grid[:, 1] ** 2)
    maps.append((potential - ideal).reshape(gx.shape))
limit = max(np.max(np.abs(values)) for values in maps)
for column, (values, title) in enumerate(zip(maps, ["circle: V - quadrupole", "dual-zero: V - quadrupole"])):
    image = axes[1, column].imshow(values, origin="lower", extent=(-0.95, 0.95, -0.95, 0.95), cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1, column].set(xlabel="x/r0", ylabel="y/r0", title=title)
    fig.colorbar(image, ax=axes[1, column], shrink=0.78)

labels = ["eta", "E nonlinear\n@0.9", "surface E\nmax", "min\ncurvature", "electrode\narea"]
ratios = [
    dual_eta / circle_eta,
    field_error(dual_radius, dual_c4, count, fraction, 0.9) / field_error(circle_radius, 0.0, count, fraction, 0.9),
    dual_surface_max / circle_surface_max,
    dual_curvature / circle_curvature,
    dual_area / circle_area,
]
bars = axes[1, 2].bar(labels, ratios, color=["tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:red"])
axes[1, 2].axhline(1.0, color="black", linewidth=1)
axes[1, 2].bar_label(bars, fmt="%.3f")
axes[1, 2].set(ylabel="dual-zero / circle", title="Benefit and engineering costs")
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 09 — final high-resolution summary of the new support-function electrode", fontsize=15)
fig.savefig(SUMMARY_OUTPUT, dpi=180)
print(f"FINAL CANDIDATE: R={dual_radius:.9f}, c4={dual_c4:.9f}, eta ratio={dual_eta/circle_eta:.9f}, field nonlinearity ratio at 0.9r0={ratios[1]:.5f}, surface-field ratio={ratios[2]:.5f}, curvature ratio={ratios[3]:.5f}, area ratio={ratios[4]:.5f}")
print("saved", SUMMARY_OUTPUT)


def normalized_harmonics(sources, coefficients, radius=0.9):
    series_theta = 2.0 * np.pi * np.arange(1440) / 1440
    points = radius * np.column_stack([np.cos(series_theta), np.sin(series_theta)])
    potential = evaluate(points, sources, coefficients)[0]
    orders = np.array([2, 6, 10, 14, 18, 22, 26, 30])
    values = np.array([2.0 * np.mean(potential * np.cos(order * series_theta)) / radius**order for order in orders])
    return orders, values / values[0]


def simulate(coefficients_by_order, orders, initial_positions, mathieu_q=0.4, periods=200, steps_per_period=100):
    positions = initial_positions.copy()
    velocities = np.zeros_like(positions)
    maximum_radius = np.linalg.norm(positions, axis=1)
    period_steps = steps_per_period
    dt = np.pi / steps_per_period
    strobe_positions = []
    strobe_velocities = []

    def acceleration(current, time):
        z = current[:, 0] + 1j * current[:, 1]
        magnitude = np.abs(z)
        z = np.where(magnitude > 1.2, z * 1.2 / np.maximum(magnitude, 1e-30), z)
        field_x = np.zeros(len(current))
        field_y = np.zeros(len(current))
        for order, coefficient in zip(orders, coefficients_by_order):
            derivative = order * coefficient * z ** (order - 1)
            field_x -= np.real(derivative)
            field_y += np.imag(derivative)
        return mathieu_q * np.column_stack([field_x, field_y]) * np.cos(2.0 * time)

    time = 0.0
    accel = acceleration(positions, time)
    for step_index in range(periods * period_steps):
        half_velocity = velocities + 0.5 * dt * accel
        positions = positions + dt * half_velocity
        time += dt
        accel = acceleration(positions, time)
        velocities = half_velocity + 0.5 * dt * accel
        maximum_radius = np.maximum(maximum_radius, np.linalg.norm(positions, axis=1))
        if (step_index + 1) % period_steps == 0:
            strobe_positions.append(positions.copy())
            strobe_velocities.append(velocities.copy())
    return maximum_radius, np.array(strobe_positions), np.array(strobe_velocities)


orders, circle_harmonics = normalized_harmonics(circle_sources, circle_coefficients)
_, dual_harmonics = normalized_harmonics(dual_sources, dual_coefficients)
ideal_harmonics = np.zeros_like(circle_harmonics)
ideal_harmonics[0] = 1.0
initial_amplitudes = np.linspace(0.1, 0.9, 9)
initial_angles = 2.0 * np.pi * np.arange(12) / 12 + np.pi / 24
initial_positions = np.vstack([
    amplitude * np.column_stack([np.cos(initial_angles), np.sin(initial_angles)])
    for amplitude in initial_amplitudes
])

dynamics = {}
for name, harmonics in [("ideal", ideal_harmonics), ("circular", circle_harmonics), ("dual-zero", dual_harmonics)]:
    dynamics[name] = simulate(harmonics, orders, initial_positions)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
colors = {"ideal": "tab:gray", "circular": "black", "dual-zero": "tab:blue"}
ideal_strobe = dynamics["ideal"][1]
for name, (maximum_radius, strobe_position, strobe_velocity) in dynamics.items():
    grouped_max = maximum_radius.reshape(len(initial_amplitudes), len(initial_angles))
    survival = np.mean(grouped_max < 1.0, axis=1)
    axes[0, 0].plot(initial_amplitudes, survival, marker="o", color=colors[name], label=name)
    axes[0, 1].plot(initial_amplitudes, np.mean(grouped_max, axis=1), marker="o", color=colors[name], label=name)
    displacement = np.sqrt(np.mean(np.sum((strobe_position - ideal_strobe) ** 2, axis=2), axis=0)).reshape(len(initial_amplitudes), len(initial_angles)).mean(axis=1)
    if name != "ideal":
        axes[0, 2].semilogy(initial_amplitudes, displacement, marker="o", color=colors[name], label=name)

axes[0, 0].set(xlabel="initial amplitude / r0", ylabel="fraction never exceeding r0", title="Bounded-aperture survival over 200 RF periods")
axes[0, 1].plot(initial_amplitudes, initial_amplitudes, color="tab:gray", linestyle="--", linewidth=1)
axes[0, 1].set(xlabel="initial amplitude / r0", ylabel="mean maximum radius / r0", title="Micromotion and nonlinear radial growth")
axes[0, 2].set(xlabel="initial amplitude / r0", ylabel="RMS stroboscopic position error", title="Departure from the ideal quadrupole trajectory")
for ax in axes[0]:
    ax.legend(fontsize=8)

amplitude_index = np.argmin(np.abs(initial_amplitudes - 0.8))
angle_index = np.argmin(np.abs(initial_angles - np.pi / 8))
trajectory_index = amplitude_index * len(initial_angles) + angle_index
for column, name in enumerate(["ideal", "circular", "dual-zero"]):
    strobe_position = dynamics[name][1][:, trajectory_index]
    strobe_velocity = dynamics[name][2][:, trajectory_index]
    axes[1, column].scatter(strobe_position[:, 0], strobe_velocity[:, 0], s=8, alpha=0.65, color=colors[name])
    axes[1, column].set(xlabel="x / r0", ylabel="dx/dtau", title=f"{name}: x phase strobe\ninitial radius 0.8 r0")

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 10 — time-dependent RF trajectory check at equal normalized Mathieu q=0.4", fontsize=15)
fig.savefig(DYNAMICS_OUTPUT, dpi=180)
circle_survival = np.mean(dynamics["circular"][0].reshape(len(initial_amplitudes), len(initial_angles)) < 1.0, axis=1)
dual_survival = np.mean(dynamics["dual-zero"][0].reshape(len(initial_amplitudes), len(initial_angles)) < 1.0, axis=1)
print("TRAJECTORY CHECK survival at initial amplitudes:")
for amplitude, circle_value, dual_value in zip(initial_amplitudes, circle_survival, dual_survival):
    print(f"  amplitude={amplitude:.1f}: circle={circle_value:.3f}, dual-zero={dual_value:.3f}")
print("CONCLUSION: this trajectory test compares nonlinear RF motion at equal central Mathieu q; inspect survival and phase strobes before translating field linearity into a device-performance claim.")
print("saved", DYNAMICS_OUTPUT)


def q_vector(radius, c4):
    return np.array(multipoles(radius, c4, count, fraction)[1:4])


def q_jacobian(radius, c4):
    radius_step = 1.0e-4
    c4_step = 1.0e-5
    derivative_radius = (q_vector(radius + radius_step, c4) - q_vector(radius - radius_step, c4)) / (2.0 * radius_step)
    derivative_c4 = (q_vector(radius, c4 + c4_step) - q_vector(radius, c4 - c4_step)) / (2.0 * c4_step)
    derivative_support_displacement = derivative_c4 / radius
    return np.column_stack([derivative_radius, derivative_support_displacement])


def multipole_field_proxy(q_values, amplitude=0.9):
    mode_orders = np.array([6, 10, 14])
    radial_scale = (amplitude / MEASUREMENT_RADIUS) ** (mode_orders - 2)
    field_scale = 0.5 * mode_orders
    return np.sqrt(np.sum((q_values * radial_scale * field_scale) ** 2, axis=-1))


circle_q = q_vector(circle_radius, 0.0)
dual_q = q_vector(dual_radius, dual_c4)
circle_jacobian = q_jacobian(circle_radius, 0.0)
dual_jacobian = q_jacobian(dual_radius, dual_c4)
tolerances = np.logspace(-5, np.log10(0.02), 24)
rng = np.random.default_rng(20260825)
standard_errors = rng.normal(size=(12000, 2))
robustness = {}
for name, base, jacobian in [
    ("circular", circle_q, circle_jacobian),
    ("dual-zero", dual_q, dual_jacobian),
]:
    median, upper = [], []
    for tolerance in tolerances:
        perturbed_q = base + (standard_errors * tolerance) @ jacobian.T
        proxy = multipole_field_proxy(perturbed_q)
        median.append(np.median(proxy))
        upper.append(np.quantile(proxy, 0.95))
    robustness[name] = (np.array(median), np.array(upper))

selected_tolerance = 0.005
circle_samples = circle_q + (standard_errors * selected_tolerance) @ circle_jacobian.T
dual_samples = dual_q + (standard_errors * selected_tolerance) @ dual_jacobian.T
circle_proxy = multipole_field_proxy(circle_samples)
dual_proxy = multipole_field_proxy(dual_samples)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
labels = ["q6", "q10", "q14"]
x = np.arange(3)
axes[0, 0].bar(x - 0.18, np.linalg.norm(circle_jacobian, axis=1), 0.36, color="black", label="circle")
axes[0, 0].bar(x + 0.18, np.linalg.norm(dual_jacobian, axis=1), 0.36, color="tab:blue", label="dual-zero")
axes[0, 0].set_yscale("log")
axes[0, 0].set_xticks(x, labels)
axes[0, 0].set(ylabel="gradient norm per r0 boundary error", title="First-order manufacturing sensitivity")
axes[0, 0].legend(fontsize=8)

for name, color in [("circular", "black"), ("dual-zero", "tab:blue")]:
    median, upper = robustness[name]
    axes[0, 1].loglog(tolerances, median, color=color, label=f"{name} median")
    axes[0, 1].loglog(tolerances, upper, color=color, linestyle="--", label=f"{name} 95th percentile")
axes[0, 1].set(xlabel="support-mode coefficient scale sigma / r0", ylabel="multipole E-error proxy at 0.9 r0", title="When does cancellation lose its advantage?")
axes[0, 1].legend(fontsize=8)

bins = np.logspace(np.log10(min(circle_proxy.min(), dual_proxy.min())), np.log10(max(circle_proxy.max(), dual_proxy.max())), 60)
axes[1, 0].hist(circle_proxy, bins=bins, histtype="step", linewidth=2, color="black", label="circle")
axes[1, 0].hist(dual_proxy, bins=bins, histtype="step", linewidth=2, color="tab:blue", label="dual-zero")
axes[1, 0].set_xscale("log")
axes[1, 0].set(xlabel="multipole E-error proxy at 0.9 r0", ylabel="samples", title=f"Distribution at sigma = {selected_tolerance:.3f} r0")
axes[1, 0].legend(fontsize=8)

axes[1, 1].scatter(circle_samples[:, 0], circle_samples[:, 1], s=2, alpha=0.08, color="black", label="circle")
axes[1, 1].scatter(dual_samples[:, 0], dual_samples[:, 1], s=2, alpha=0.08, color="tab:blue", label="dual-zero")
axes[1, 1].scatter([circle_q[0], dual_q[0]], [circle_q[1], dual_q[1]], marker="x", s=80, color=["black", "tab:blue"])
axes[1, 1].set(xlabel="physical q6 at 0.5 r0", ylabel="physical q10 at 0.5 r0", title="How fabrication errors repopulate cancelled modes")
axes[1, 1].legend(fontsize=8)
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 11 — first-order two-mode robustness of the support-function electrode", fontsize=15)
fig.savefig(ROBUSTNESS_OUTPUT, dpi=180)

advantage = robustness["dual-zero"][1] / robustness["circular"][1]
print(f"ROBUSTNESS at modal coefficient sigma={selected_tolerance:.3f} r0: circle median proxy={np.median(circle_proxy):.4e}, dual median={np.median(dual_proxy):.4e}, ratio={np.median(dual_proxy)/np.median(circle_proxy):.3f}")
print(f"dual/circle 95th-percentile proxy ratio over tolerance sweep: {np.min(advantage):.3f} to {np.max(advantage):.3f}")
print("CONCLUSION: the robust value of exact multipole cancellation depends on the boundary tolerance; use the percentile curves, not the nominal zero, when setting manufacturing requirements.")
print("saved", ROBUSTNESS_OUTPUT)
