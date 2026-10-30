import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import lstsq


MIXED_TERM = os.environ.get("MIXED_TERM", "0") == "1"
PROFILE_TERM = os.environ.get("PROFILE_TERM", "0") == "1"
SPARSE_TERM = os.environ.get("SPARSE_TERM", "0") == "1"
OUTPUT = "exp/claude_tree_stress_test/results/37_sparse_mixed_geometry_head.png" if SPARSE_TERM else ("exp/claude_tree_stress_test/results/36_profiled_mixed_geometry_head.png" if PROFILE_TERM else ("exp/claude_tree_stress_test/results/35_low_rank_mixed_geometry_head.png" if MIXED_TERM else "exp/claude_tree_stress_test/results/34_additive_geometry_head.png"))
R_NODES = np.linspace(0.65, 0.75, 5)
C_NODES = np.linspace(-0.02, 0.02, 5)
TEST = [(r, c) for r in [0.6625, 0.6875, 0.7125, 0.7375] for c in [-0.015, -0.005, 0.005, 0.015]]
ORDERS = np.arange(2, 39, 4)
COUNT = 48
RING_RADIUS = 0.90


def local_support(radius, c4, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    support = radius * (1.0 + c4 * np.cos(4.0 * theta))
    derivative = -4.0 * radius * c4 * np.sin(4.0 * theta)
    return (support * np.cos(theta) - derivative * np.sin(theta)) + 1j * (support * np.sin(theta) + derivative * np.cos(theta))


def solve(radius, c4):
    distance = 1.0 + radius * (1.0 + c4)
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    rotations = np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    shape = local_support(radius, c4, 3 * COUNT)
    source_shape = 0.70 * local_support(radius, c4, COUNT, 0.013)
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


def reference(radius, c4, z):
    sources, coefficients = solve(radius, c4)
    delta = z[:, None] - sources[None, :]
    potential = np.log(np.abs(delta)) @ coefficients[:-1] + coefficients[-1]
    gradient = np.sum(coefficients[None, :-1] / np.conj(delta), axis=1)
    return potential, np.column_stack([-gradient.real, -gradient.imag])


theta = 2.0 * np.pi * np.arange(2048) / 2048
ring = RING_RADIUS * np.exp(1j * theta)
cache = {}


def exact_coefficients(radius, c4):
    key = (round(float(radius), 8), round(float(c4), 8))
    if key not in cache:
        values = reference(radius, c4, ring)[0]
        cache[key] = np.array([2.0 * np.mean(values * np.cos(order * theta)) / RING_RADIUS**order for order in ORDERS])
    return cache[key]


radius_cross = np.array([exact_coefficients(radius, 0.0) for radius in R_NODES])
c4_cross = np.array([exact_coefficients(0.70, c4) for c4 in C_NODES])
center = exact_coefficients(0.70, 0.0)
radius_fit = np.polynomial.chebyshev.chebfit((R_NODES - 0.70) / 0.05, radius_cross, 4)
c4_fit = np.polynomial.chebyshev.chebfit(C_NODES / 0.02, c4_cross, 4)
corner_residuals = []
for radius in [0.65, 0.75]:
    for c4 in [-0.02, 0.02]:
        additive = exact_coefficients(radius, 0.0) + exact_coefficients(0.70, c4) - center
        corner_residuals.append((exact_coefficients(radius, c4) - additive) / (((radius - 0.70) / 0.05) * (c4 / 0.02)))
mixed_coefficient = np.mean(corner_residuals, axis=0)
top_residual = np.array([exact_coefficients(radius, 0.02) - exact_coefficients(radius, 0.0) - exact_coefficients(0.70, 0.02) + center for radius in R_NODES])
top_residual_fit = np.polynomial.chebyshev.chebfit((R_NODES - 0.70) / 0.05, top_residual, 4)


def sparse_features(radius, c4):
    x = (radius - 0.70) / 0.05
    y = c4 / 0.02
    tx = [np.polynomial.chebyshev.chebval(x, [0] * degree + [1]) for degree in range(5)]
    ty = [np.polynomial.chebyshev.chebval(y, [0] * degree + [1]) for degree in range(5)]
    return np.array(tx + ty[1:] + [tx[1] * ty[1], tx[2] * ty[1], tx[1] * ty[2], tx[2] * ty[2]])


sparse_pairs = [(radius, 0.0) for radius in R_NODES] + [(0.70, c4) for c4 in C_NODES if c4 != 0.0] + [(radius, c4) for radius in [0.65, 0.75] for c4 in [-0.02, 0.02]]
sparse_design = np.array([sparse_features(*pair) for pair in sparse_pairs])
sparse_response = np.linalg.solve(sparse_design, np.array([exact_coefficients(*pair) for pair in sparse_pairs]))


def predict(radius, c4):
    radial = np.polynomial.chebyshev.chebval((radius - 0.70) / 0.05, radius_fit)
    angular = np.polynomial.chebyshev.chebval(c4 / 0.02, c4_fit)
    result = radial + angular - center
    if SPARSE_TERM:
        result = sparse_features(radius, c4) @ sparse_response
    elif PROFILE_TERM:
        result += (c4 / 0.02) * np.polynomial.chebyshev.chebval((radius - 0.70) / 0.05, top_residual_fit)
    elif MIXED_TERM:
        result += ((radius - 0.70) / 0.05) * (c4 / 0.02) * mixed_coefficient
    return result


def harmonic_field(z, coefficients):
    potential = np.zeros(len(z))
    field = np.zeros(len(z), dtype=complex)
    for order, coefficient in zip(ORDERS, coefficients):
        potential += coefficient * np.real(z**order)
        field -= order * coefficient * np.conj(z ** (order - 1))
    return potential, np.column_stack([field.real, field.imag])


rng = np.random.default_rng(3401)
angles = rng.uniform(0.0, 2.0 * np.pi, 2000)
radii = 0.80 * np.sqrt(rng.uniform(0.0, 1.0, 2000))
probe = radii * np.exp(1j * angles)
records = {}
for radius, c4 in TEST:
    exact_v, exact_e = reference(radius, c4, probe)
    exact_a = exact_coefficients(radius, c4)
    predicted_a = predict(radius, c4)
    predicted_v, predicted_e = harmonic_field(probe, predicted_a)
    hr = 0.001
    hc = 0.0002
    hs = 1e-5
    exact_dr = ((exact_coefficients(radius + hr, c4)[1] / exact_coefficients(radius + hr, c4)[0]) - (exact_coefficients(radius - hr, c4)[1] / exact_coefficients(radius - hr, c4)[0])) / (2.0 * hr)
    exact_dc = ((exact_coefficients(radius, c4 + hc)[1] / exact_coefficients(radius, c4 + hc)[0]) - (exact_coefficients(radius, c4 - hc)[1] / exact_coefficients(radius, c4 - hc)[0])) / (2.0 * hc)
    pred_dr = ((predict(radius + hs, c4)[1] / predict(radius + hs, c4)[0]) - (predict(radius - hs, c4)[1] / predict(radius - hs, c4)[0])) / (2.0 * hs)
    pred_dc = ((predict(radius, c4 + hs)[1] / predict(radius, c4 + hs)[0]) - (predict(radius, c4 - hs)[1] / predict(radius, c4 - hs)[0])) / (2.0 * hs)
    records[(radius, c4)] = {
        "v": np.sqrt(np.mean(((predicted_v - exact_v) / np.max(np.abs(exact_v))) ** 2)),
        "e": np.sqrt(np.mean((np.linalg.norm(predicted_e - exact_e, axis=1) / np.max(np.linalg.norm(exact_e, axis=1))) ** 2)),
        "beta": abs(predicted_a[1] / predicted_a[0] - exact_a[1] / exact_a[0]),
        "exact_dr": exact_dr,
        "exact_dc": exact_dc,
        "pred_dr": pred_dr,
        "pred_dc": pred_dc,
    }

surface_r = np.linspace(0.65, 0.75, 13)
surface_c = np.linspace(-0.02, 0.02, 13)
exact_beta = np.zeros((13, 13))
predicted_beta = np.zeros_like(exact_beta)
interaction = np.zeros_like(exact_beta)
for i, c4 in enumerate(surface_c):
    for j, radius in enumerate(surface_r):
        exact_a = exact_coefficients(radius, c4)
        predicted_a = predict(radius, c4)
        exact_beta[i, j] = exact_a[1] / exact_a[0]
        predicted_beta[i, j] = predicted_a[1] / predicted_a[0]
        additive_exact = exact_coefficients(radius, 0.0) + exact_coefficients(0.70, c4) - center
        interaction[i, j] = np.linalg.norm(exact_a - additive_exact) / np.linalg.norm(exact_a)

worst_v = max(row["v"] for row in records.values())
worst_e = max(row["e"] for row in records.values())
worst_beta = max(row["beta"] for row in records.values())
print("HYPOTHESIS:", "the same cross-plus-corners data support four independent mixed parity terms, which should beat collapsing the interaction to one rank-one product." if SPARSE_TERM else ("the remaining interaction is nonlinear in radius but nearly linear in c4, so one complete off-axis radius slice will outperform the four-corner correction at the same 13-snapshot cost." if PROFILE_TERM else ("the structured error in Plot 34 is dominated by one separable R*c4 interaction, so four corner solves will repair it without a tensor grid." if MIXED_TERM else "radius and c4 have weak interactions, so a nine-snapshot additive harmonic head can replace a tensor grid.")))
print(13 if (MIXED_TERM or PROFILE_TERM or SPARSE_TERM) else 9, "unique snapshots, worst unseen V E beta", worst_v, worst_e, worst_beta)
print("maximum relative coefficient interaction", np.max(interaction))
if SPARSE_TERM:
    max_dr = max(abs(row["pred_dr"] - row["exact_dr"]) / abs(row["exact_dr"]) for row in records.values())
    max_dc = max(abs(row["pred_dc"] - row["exact_dc"]) / abs(row["exact_dc"]) for row in records.values())
    print("worst relative radius and c4 beta-gradient errors", max_dr, max_dc)

fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
extent = [surface_r[0], surface_r[-1], surface_c[0], surface_c[-1]]
image0 = axes[0, 0].imshow(exact_beta, origin="lower", extent=extent, aspect="auto", cmap="viridis")
axes[0, 0].scatter(R_NODES, np.zeros_like(R_NODES), color="white", label="radius cross")
axes[0, 0].scatter(np.full_like(C_NODES, 0.70), C_NODES, color="red", marker="x", label="c4 cross")
if MIXED_TERM or SPARSE_TERM:
    axes[0, 0].scatter([0.65, 0.65, 0.75, 0.75], [-0.02, 0.02, -0.02, 0.02], color="cyan", marker="s", label="mixed corners")
axes[0, 0].set(xlabel="radius/r0", ylabel="c4", title="Exact beta and sparse snapshots" if SPARSE_TERM else "Exact beta and cross snapshots")
axes[0, 0].legend(fontsize=8)
fig.colorbar(image0, ax=axes[0, 0])
image1 = axes[0, 1].imshow(np.abs(predicted_beta - exact_beta), origin="lower", extent=extent, aspect="auto", cmap="magma")
axes[0, 1].set(xlabel="radius/r0", ylabel="c4", title="Sparse-head beta error" if SPARSE_TERM else "Additive-head beta error")
fig.colorbar(image1, ax=axes[0, 1])
image2 = axes[1, 0].imshow(interaction, origin="lower", extent=extent, aspect="auto", cmap="magma")
axes[1, 0].set(xlabel="radius/r0", ylabel="c4", title="Non-additive coefficient interaction")
fig.colorbar(image2, ax=axes[1, 0])
positions = np.arange(len(TEST))
axes[1, 1].plot(positions, [records[p]["v"] for p in TEST], marker="o", label="V RMSE")
axes[1, 1].plot(positions, [records[p]["e"] for p in TEST], marker="o", label="E RMSE")
axes[1, 1].plot(positions, [records[p]["beta"] for p in TEST], marker="o", label="beta error")
axes[1, 1].set(yscale="log", xlabel="16 unseen geometries", ylabel="error", title="Independent held-out gates")
axes[1, 1].legend(fontsize=8)
fig.suptitle("Claude-tree stress test 37 — sparse mixed-parity head" if SPARSE_TERM else ("Claude-tree stress test 36 — profiled low-rank interaction" if PROFILE_TERM else ("Claude-tree stress test 35 — one low-rank mixed interaction" if MIXED_TERM else "Claude-tree stress test 34 — can geometry dependence avoid a tensor grid?")), fontsize=14)
fig.savefig(OUTPUT, dpi=180)
if SPARSE_TERM:
    figure2, gradient_axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    exact_dr = np.array([records[p]["exact_dr"] for p in TEST])
    pred_dr = np.array([records[p]["pred_dr"] for p in TEST])
    exact_dc = np.array([records[p]["exact_dc"] for p in TEST])
    pred_dc = np.array([records[p]["pred_dc"] for p in TEST])
    gradient_axes[0].scatter(exact_dr, pred_dr, color="tab:blue")
    limits = [min(exact_dr.min(), pred_dr.min()), max(exact_dr.max(), pred_dr.max())]
    gradient_axes[0].plot(limits, limits, color="black", linestyle="--")
    gradient_axes[0].set(xlabel="exact d beta/dR", ylabel="sparse-head derivative", title="Radius-gradient gate")
    gradient_axes[1].scatter(exact_dc, pred_dc, color="tab:orange")
    limits = [min(exact_dc.min(), pred_dc.min()), max(exact_dc.max(), pred_dc.max())]
    gradient_axes[1].plot(limits, limits, color="black", linestyle="--")
    gradient_axes[1].set(xlabel="exact d beta/dc4", ylabel="sparse-head derivative", title="Noncircular-gradient gate")
    gradient_axes[2].plot(np.abs((pred_dr - exact_dr) / exact_dr), marker="o", label="radius derivative")
    gradient_axes[2].plot(np.abs((pred_dc - exact_dc) / exact_dc), marker="o", label="c4 derivative")
    gradient_axes[2].set(yscale="log", xlabel="16 unseen geometries", ylabel="relative error", title="Optimizer-facing error")
    gradient_axes[2].legend(fontsize=8)
    figure2.suptitle("Claude-tree stress test 38 — sparse harmonic-head shape gradients", fontsize=14)
    figure2.savefig("exp/claude_tree_stress_test/results/38_sparse_head_gradient_gate.png", dpi=180)
print("CONCLUSION:", "this is the fair 13-snapshot comparison against the rank-one and profiled interaction heads; any gain comes from representing interaction structure, not more exact solves." if SPARSE_TERM else ("compare with Plot 35 at equal solve count; success would support adaptive slices instead of isotropic tensor refinement." if PROFILE_TERM else ("compare directly with Plots 33 and 34: this tests whether plot structure can identify a cheaper interaction correction." if MIXED_TERM else "compare the nine-snapshot errors with Plot 33's fifteen-snapshot tensor head; low interaction supports additive sampling, while localized interaction identifies where cross terms are required.")))
print("saved", OUTPUT)
