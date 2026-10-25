import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import lstsq, qr


PAIRWISE = os.environ.get("PAIRWISE", "0") == "1"
TRIPLE = os.environ.get("TRIPLE", "0") == "1"
REFINE_C8 = os.environ.get("REFINE_C8", "0") == "1"
IDENTIFIABLE_C8 = os.environ.get("IDENTIFIABLE_C8", "0") == "1"
TOTAL_DEGREE = os.environ.get("TOTAL_DEGREE", "0") == "1"
PIVOTED_TOTAL = os.environ.get("PIVOTED_TOTAL", "0") == "1"
DERIVATIVE_INFORMED = os.environ.get("DERIVATIVE_INFORMED", "0") == "1"
ALL_DERIVATIVES = os.environ.get("ALL_DERIVATIVES", "0") == "1"
DEGREE5 = os.environ.get("DEGREE5", "0") == "1"
DERIVATIVE_LOBATTO = os.environ.get("DERIVATIVE_LOBATTO", "0") == "1"
HYBRID_DERIVATIVES = os.environ.get("HYBRID_DERIVATIVES", "0") == "1"
DERIVATIVE_LOBATTO = DERIVATIVE_LOBATTO or HYBRID_DERIVATIVES
DEGREE5 = DEGREE5 or DERIVATIVE_LOBATTO
ALL_DERIVATIVES = ALL_DERIVATIVES or DEGREE5
DERIVATIVE_INFORMED = DERIVATIVE_INFORMED or ALL_DERIVATIVES
PIVOTED_TOTAL = PIVOTED_TOTAL or DERIVATIVE_INFORMED
TOTAL_DEGREE = TOTAL_DEGREE or PIVOTED_TOTAL
IDENTIFIABLE_C8 = IDENTIFIABLE_C8 or TOTAL_DEGREE
REFINE_C8 = REFINE_C8 or IDENTIFIABLE_C8
TRIPLE = TRIPLE or REFINE_C8
PAIRWISE = PAIRWISE or TRIPLE
OUTPUT = "exp/claude_tree_stress_test/results/50_hybrid_derivative_design.png" if HYBRID_DERIVATIVES else ("exp/claude_tree_stress_test/results/49_lobatto_derivative_design.png" if DERIVATIVE_LOBATTO else ("exp/claude_tree_stress_test/results/48_degree5_derivative_head.png" if DEGREE5 else ("exp/claude_tree_stress_test/results/47_all_derivative_three_coordinate_head.png" if ALL_DERIVATIVES else ("exp/claude_tree_stress_test/results/46_derivative_informed_three_coordinate_head.png" if DERIVATIVE_INFORMED else ("exp/claude_tree_stress_test/results/45_pivoted_total_degree_head.png" if PIVOTED_TOTAL else ("exp/claude_tree_stress_test/results/44_total_degree_three_coordinate_head.png" if TOTAL_DEGREE else ("exp/claude_tree_stress_test/results/43_identifiable_c8_triple_head.png" if IDENTIFIABLE_C8 else ("exp/claude_tree_stress_test/results/42_c8_refined_triple_head.png" if REFINE_C8 else ("exp/claude_tree_stress_test/results/41_three_coordinate_triple_head.png" if TRIPLE else ("exp/claude_tree_stress_test/results/40_three_coordinate_pairwise_head.png" if PAIRWISE else "exp/claude_tree_stress_test/results/39_three_coordinate_additive_head.png"))))))))))
ORDERS = np.arange(2, 39, 4)
COUNT = 44
RING_RADIUS = 0.90
R_NODES = np.linspace(0.65, 0.75, 5)
C4_NODES = np.linspace(-0.02, 0.02, 5)
C8_NODES = np.linspace(-0.01, 0.01, 5)
TEST_NORMALIZED = np.array([
    [-0.75, -0.75, -0.25], [-0.75, -0.25, 0.75], [-0.75, 0.25, -0.75], [-0.75, 0.75, 0.25],
    [-0.25, -0.75, 0.75], [-0.25, -0.25, -0.25], [-0.25, 0.25, 0.25], [-0.25, 0.75, -0.75],
    [0.25, -0.75, -0.75], [0.25, -0.25, 0.25], [0.25, 0.25, -0.25], [0.25, 0.75, 0.75],
    [0.75, -0.75, 0.25], [0.75, -0.25, -0.75], [0.75, 0.25, 0.75], [0.75, 0.75, -0.25],
])
TEST = [(0.70 + 0.05 * x, 0.02 * y, 0.01 * z) for x, y, z in TEST_NORMALIZED]


def local_support(radius, c4, c8, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    support = radius * (1.0 + c4 * np.cos(4.0 * theta) + c8 * np.cos(8.0 * theta))
    derivative = radius * (-4.0 * c4 * np.sin(4.0 * theta) - 8.0 * c8 * np.sin(8.0 * theta))
    return (support * np.cos(theta) - derivative * np.sin(theta)) + 1j * (support * np.sin(theta) + derivative * np.cos(theta))


def solve(radius, c4, c8):
    distance = 1.0 + radius * (1.0 + c4 + c8)
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    rotations = np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    shape = local_support(radius, c4, c8, 3 * COUNT)
    source_shape = 0.70 * local_support(radius, c4, c8, COUNT, 0.013)
    boundary = np.concatenate([center + rotation * shape for center, rotation in zip(centers, rotations)])
    sources = np.concatenate([center + rotation * source_shape for center, rotation in zip(centers, rotations)])
    values = np.concatenate([np.full(3 * COUNT, voltage) for voltage in voltages])
    outer_theta = 2.0 * np.pi * np.arange(12 * COUNT) / (12 * COUNT)
    source_theta = 2.0 * np.pi * np.arange(4 * COUNT) / (4 * COUNT) + 0.007
    boundary = np.concatenate([boundary, 5.0 * np.exp(1j * outer_theta)])
    sources = np.concatenate([sources, 6.5 * np.exp(1j * source_theta)])
    values = np.concatenate([values, np.zeros(12 * COUNT)])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = lstsq(matrix, values, cond=1e-12, lapack_driver="gelsy")
    return sources, coefficients


def reference(radius, c4, c8, points):
    sources, coefficients = solve(radius, c4, c8)
    delta = points[:, None] - sources[None, :]
    potential = np.log(np.abs(delta)) @ coefficients[:-1] + coefficients[-1]
    gradient = np.sum(coefficients[None, :-1] / np.conj(delta), axis=1)
    return potential, np.column_stack([-gradient.real, -gradient.imag])


theta = 2.0 * np.pi * np.arange(2048) / 2048
ring = RING_RADIUS * np.exp(1j * theta)
cache = {}


def exact_coefficients(radius, c4, c8):
    key = tuple(round(float(value), 8) for value in (radius, c4, c8))
    if key not in cache:
        values = reference(radius, c4, c8, ring)[0]
        cache[key] = np.array([2.0 * np.mean(values * np.cos(order * theta)) / RING_RADIUS**order for order in ORDERS])
    return cache[key]


def normalized(radius, c4, c8):
    return (radius - 0.70) / 0.05, c4 / 0.02, c8 / 0.01


def cheb(coordinate, degree):
    return np.polynomial.chebyshev.chebval(coordinate, [0.0] * degree + [1.0])


def anchored_cheb(coordinate, degree):
    return cheb(coordinate, degree) - cheb(0.0, degree)


def features(radius, c4, c8):
    x, y, z = normalized(radius, c4, c8)
    if TOTAL_DEGREE:
        degree = 5 if DEGREE5 else 4
        return np.array([cheb(x, i) * cheb(y, j) * cheb(z, k) for i in range(degree + 1) for j in range(degree + 1) for k in range(degree + 1) if i + j + k <= degree])
    univariate = [cheb(x, degree) for degree in range(5)] + [cheb(y, degree) for degree in range(1, 5)] + [cheb(z, degree) for degree in range(1, 5)]
    if not PAIRWISE:
        return np.array(univariate)
    mixed = []
    for first, second in [(x, y), (x, z), (y, z)]:
        mixed += [anchored_cheb(first, 1) * anchored_cheb(second, 1), anchored_cheb(first, 2) * anchored_cheb(second, 1), anchored_cheb(first, 1) * anchored_cheb(second, 2), anchored_cheb(first, 2) * anchored_cheb(second, 2)]
    triple = [anchored_cheb(x, i) * anchored_cheb(y, j) * anchored_cheb(z, k) for i in [1, 2] for j in [1, 2] for k in ([1, 2, 3, 4] if REFINE_C8 else [1, 2])] if TRIPLE else []
    return np.array(univariate + mixed + triple)


training_pairs = [(radius, 0.0, 0.0) for radius in R_NODES]
training_pairs += [(0.70, c4, 0.0) for c4 in C4_NODES if c4 != 0.0]
training_pairs += [(0.70, 0.0, c8) for c8 in C8_NODES if c8 != 0.0]
if PAIRWISE:
    training_pairs += [(radius, c4, 0.0) for radius in [0.65, 0.75] for c4 in [-0.02, 0.02]]
    training_pairs += [(radius, 0.0, c8) for radius in [0.65, 0.75] for c8 in [-0.01, 0.01]]
    training_pairs += [(0.70, c4, c8) for c4 in [-0.02, 0.02] for c8 in [-0.01, 0.01]]
if TRIPLE:
    interior_c8 = 0.01 / np.sqrt(2.0) if IDENTIFIABLE_C8 else 0.005
    training_pairs += [(radius, c4, c8) for radius in [0.65, 0.75] for c4 in [-0.02, 0.02] for c8 in ([-0.01, -interior_c8, interior_c8, 0.01] if REFINE_C8 else [-0.01, 0.01])]
if PIVOTED_TOTAL:
    candidate_count = 6 if DEGREE5 else 5
    candidate_axis = np.sort(np.cos(np.arange(candidate_count) * np.pi / (candidate_count - 1.0)))
    candidates = [(0.70 + 0.05 * x, 0.02 * y, 0.01 * z) for x in candidate_axis for y in candidate_axis for z in candidate_axis]
    candidate_design = np.array([features(*point) for point in candidates])
    _, _, pivots = qr(candidate_design.T, pivoting=True, mode="economic")
    feature_count = 56 if DEGREE5 else 35
    training_pairs = [candidates[index] for index in pivots[:feature_count]]
design = np.array([features(*point) for point in training_pairs])
targets = np.array([exact_coefficients(*point) for point in training_pairs])
if DERIVATIVE_INFORMED:
    derivative_node = 1.0 / np.sqrt(2.0) if DERIVATIVE_LOBATTO else 0.5
    derivative_points = [(0.70 + 0.05 * x, 0.02 * y, 0.01 * z) for x in [-derivative_node, derivative_node] for y in [-derivative_node, derivative_node] for z in [-derivative_node, derivative_node]]
    normalized_step = 1e-5
    derivative_design = []
    derivative_targets = []
    axes_to_fit = range(3) if ALL_DERIVATIVES else [2]
    coordinate_scales = [0.05, 0.02, 0.01]
    physical_steps = [0.001, 0.0002, 0.0001]
    for axis in axes_to_fit:
        for point in derivative_points:
            plus = list(point)
            minus = list(point)
            plus[axis] += coordinate_scales[axis] * normalized_step
            minus[axis] -= coordinate_scales[axis] * normalized_step
            derivative_design.append((features(*plus) - features(*minus)) / (2.0 * normalized_step))
            plus = list(point)
            minus = list(point)
            plus[axis] += physical_steps[axis]
            minus[axis] -= physical_steps[axis]
            derivative_targets.append(coordinate_scales[axis] * (exact_coefficients(*plus) - exact_coefficients(*minus)) / (2.0 * physical_steps[axis]))
    if HYBRID_DERIVATIVES:
        central_points = [(0.70 + 0.05 * x, 0.02 * y, 0.01 * z) for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]]
        for point in central_points:
            plus = list(point)
            minus = list(point)
            plus[2] += coordinate_scales[2] * normalized_step
            minus[2] -= coordinate_scales[2] * normalized_step
            derivative_design.append((features(*plus) - features(*minus)) / (2.0 * normalized_step))
            plus = list(point)
            minus = list(point)
            plus[2] += physical_steps[2]
            minus[2] -= physical_steps[2]
            derivative_targets.append(coordinate_scales[2] * (exact_coefficients(*plus) - exact_coefficients(*minus)) / (2.0 * physical_steps[2]))
    derivative_design = np.array(derivative_design)
    derivative_targets = np.array(derivative_targets)
    design = np.vstack([design, derivative_design])
    targets = np.vstack([targets, derivative_targets])
response = lstsq(design, targets, cond=1e-12, lapack_driver="gelsy")[0] if TOTAL_DEGREE else np.linalg.solve(design, targets)


def predict(radius, c4, c8):
    return features(radius, c4, c8) @ response


def harmonic_field(points, coefficients):
    potential = np.zeros(len(points))
    field = np.zeros(len(points), dtype=complex)
    for order, coefficient in zip(ORDERS, coefficients):
        potential += coefficient * np.real(points**order)
        field -= order * coefficient * np.conj(points ** (order - 1))
    return potential, np.column_stack([field.real, field.imag])


rng = np.random.default_rng(3901)
angle = rng.uniform(0.0, 2.0 * np.pi, 1600)
radial = 0.80 * np.sqrt(rng.uniform(0.0, 1.0, 1600))
probe = radial * np.exp(1j * angle)
steps = np.array([0.001, 0.0002, 0.0001])
surrogate_steps = np.array([1e-5, 1e-5, 1e-5])
records = []
for point in TEST:
    exact_a = exact_coefficients(*point)
    predicted_a = predict(*point)
    exact_v, exact_e = reference(*point, probe)
    predicted_v, predicted_e = harmonic_field(probe, predicted_a)
    exact_gradients = []
    predicted_gradients = []
    for axis in range(3):
        plus = list(point)
        minus = list(point)
        plus[axis] += steps[axis]
        minus[axis] -= steps[axis]
        a_plus = exact_coefficients(*plus)
        a_minus = exact_coefficients(*minus)
        exact_gradients.append((a_plus[1] / a_plus[0] - a_minus[1] / a_minus[0]) / (2.0 * steps[axis]))
        plus = list(point)
        minus = list(point)
        plus[axis] += surrogate_steps[axis]
        minus[axis] -= surrogate_steps[axis]
        a_plus = predict(*plus)
        a_minus = predict(*minus)
        predicted_gradients.append((a_plus[1] / a_plus[0] - a_minus[1] / a_minus[0]) / (2.0 * surrogate_steps[axis]))
    records.append({
        "v": np.sqrt(np.mean(((predicted_v - exact_v) / np.max(np.abs(exact_v))) ** 2)),
        "e": np.sqrt(np.mean((np.linalg.norm(predicted_e - exact_e, axis=1) / np.max(np.linalg.norm(exact_e, axis=1))) ** 2)),
        "beta": abs(predicted_a[1] / predicted_a[0] - exact_a[1] / exact_a[0]),
        "exact_gradients": np.array(exact_gradients),
        "predicted_gradients": np.array(predicted_gradients),
    })

surface_c4 = np.linspace(-0.02, 0.02, 17)
surface_c8 = np.linspace(-0.01, 0.01, 17)
exact_surface = np.zeros((17, 17))
error_surface = np.zeros_like(exact_surface)
for i, c8 in enumerate(surface_c8):
    for j, c4 in enumerate(surface_c4):
        exact_a = exact_coefficients(0.725, c4, c8)
        predicted_a = predict(0.725, c4, c8)
        exact_surface[i, j] = exact_a[1] / exact_a[0]
        error_surface[i, j] = abs(predicted_a[1] / predicted_a[0] - exact_surface[i, j])

gradient_errors = np.array([np.abs((row["predicted_gradients"] - row["exact_gradients"]) / row["exact_gradients"]) for row in records])
print("HYPOTHESIS:", "adding central c8 derivatives to the edge-oriented design will cover the remaining interior gradient error within 120 exact solves." if HYBRID_DERIVATIVES else ("moving the same eight derivative points to Chebyshev locations will reduce edge-gradient error without increasing the solve count." if DERIVATIVE_LOBATTO else ("raising the pivoted total-degree model from degree 4 to 5 will push all three derivative errors below one percent while remaining cheaper than a 5x5x5 tensor grid." if DEGREE5 else ("symmetric derivative supervision at eight interior geometries will improve all optimizer-facing gradients without sacrificing held-out fields." if ALL_DERIVATIVES else ("eight interior c8 coefficient derivatives will improve the optimizer-facing c8 gradient beyond value-only interpolation." if DERIVATIVE_INFORMED else ("pivoted QR can choose 35 identifiable total-degree samples before any PDE solves, repairing Plot 44's rank-deficient experimental design." if PIVOTED_TOTAL else ("an overdetermined total-degree-4 polynomial will give better unseen gradients than high-degree corner interpolation using the same 41 exact solves." if TOTAL_DEGREE else ("using c8 nodes at +/-1/sqrt(2) separates T2 from T4 and removes Plot 42's exact sampling degeneracy." if IDENTIFIABLE_C8 else ("the failed c8 derivative is caused by sampling the triple interaction only at c8 endpoints; two interior c8 levels should repair it without filling the cube." if REFINE_C8 else ("the pairwise error vanishes on coordinate planes and grows off them, so eight cube corners and sparse triple-parity terms should repair all three gradients." if TRIPLE else ("three-coordinate response is dominated by univariate and pairwise interactions, so 25 sparse snapshots can replace a 125-point tensor grid." if PAIRWISE else "a 13-snapshot axis-only head is the simplest three-coordinate baseline; its error will reveal whether interactions are negligible.")))))))))))
print("value snapshots", len(training_pairs), "derivative rows", (32 if HYBRID_DERIVATIVES else (24 if ALL_DERIVATIVES else 8)) if DERIVATIVE_INFORMED else 0, "design condition", np.linalg.cond(design))
print("worst unseen V E beta", max(row["v"] for row in records), max(row["e"] for row in records), max(row["beta"] for row in records))
print("worst relative beta gradients R c4 c8", np.max(gradient_errors, axis=0))
for axis, label in enumerate(["R", "c4", "c8"]):
    worst_index = np.argmax(gradient_errors[:, axis])
    print("worst", label, "gradient at", TEST[worst_index], "exact predicted", records[worst_index]["exact_gradients"][axis], records[worst_index]["predicted_gradients"][axis])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
extent = [surface_c4[0], surface_c4[-1], surface_c8[0], surface_c8[-1]]
image0 = axes[0, 0].imshow(exact_surface, origin="lower", extent=extent, aspect="auto", cmap="viridis")
axes[0, 0].set(xlabel="c4", ylabel="c8", title="Exact beta at R=0.725 r0")
fig.colorbar(image0, ax=axes[0, 0])
image1 = axes[0, 1].imshow(error_surface, origin="lower", extent=extent, aspect="auto", cmap="magma")
axes[0, 1].set(xlabel="c4", ylabel="c8", title="Held-slice beta error")
fig.colorbar(image1, ax=axes[0, 1])
positions = np.arange(len(records))
axes[0, 2].plot(positions, [row["v"] for row in records], marker="o", label="V RMSE")
axes[0, 2].plot(positions, [row["e"] for row in records], marker="o", label="E RMSE")
axes[0, 2].plot(positions, [row["beta"] for row in records], marker="o", label="beta error")
axes[0, 2].set(yscale="log", xlabel="16 unseen 3D-coordinate points", ylabel="error", title="Independent value gates")
axes[0, 2].legend(fontsize=8)
for axis, (label, plot_axis) in enumerate(zip(["R", "c4", "c8"], axes[1])):
    exact_gradient = np.array([row["exact_gradients"][axis] for row in records])
    predicted_gradient = np.array([row["predicted_gradients"][axis] for row in records])
    plot_axis.scatter(exact_gradient, predicted_gradient)
    limits = [min(exact_gradient.min(), predicted_gradient.min()), max(exact_gradient.max(), predicted_gradient.max())]
    plot_axis.plot(limits, limits, color="black", linestyle="--")
    plot_axis.set(xlabel=f"exact d beta/d{label}", ylabel="surrogate derivative", title=f"{label} shape-gradient gate")
fig.suptitle("Claude-tree stress test 50 — hybrid derivative design" if HYBRID_DERIVATIVES else ("Claude-tree stress test 49 — Chebyshev derivative design" if DERIVATIVE_LOBATTO else ("Claude-tree stress test 48 — degree-5 derivative-informed head" if DEGREE5 else ("Claude-tree stress test 47 — all-derivative three-coordinate head" if ALL_DERIVATIVES else ("Claude-tree stress test 46 — derivative-informed three-coordinate head" if DERIVATIVE_INFORMED else ("Claude-tree stress test 45 — pivoted total-degree experimental design" if PIVOTED_TOTAL else ("Claude-tree stress test 44 — total-degree-4 three-coordinate head" if TOTAL_DEGREE else ("Claude-tree stress test 43 — identifiable c8-refined triple head" if IDENTIFIABLE_C8 else ("Claude-tree stress test 42 — c8-refined sparse triple head" if REFINE_C8 else ("Claude-tree stress test 41 — sparse triple-interaction harmonic head" if TRIPLE else ("Claude-tree stress test 40 — sparse pairwise three-coordinate harmonic head" if PAIRWISE else "Claude-tree stress test 39 — additive three-coordinate harmonic head")))))))))), fontsize=14)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: accept only if unseen fields and all three geometry derivatives pass; pairwise success would establish sparse three-coordinate scaling, not free-form generality.")
print("saved", OUTPUT)
