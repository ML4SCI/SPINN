import matplotlib.pyplot as plt
import numpy as np
import runpy


BASE = runpy.run_path("exp/claude_tree_stress_test/51_implicit_mfs_derivative.py")
OUTPUT = "exp/claude_tree_stress_test/results/53_retained_rank_stability.png"
LEVELS = [-1.0, 0.0, 1.0]
POINTS = [(0.70 + 0.05 * x, 0.02 * y, 0.01 * z) for x in LEVELS for y in LEVELS for z in LEVELS]
CUTOFF = 1e-12

ranks = []
last_retained = []
first_discarded = []
condition = []
for point in POINTS:
    matrix = BASE["matrices"](np.array(point))[0]
    singular = np.linalg.svd(matrix, compute_uv=False)
    normalized = singular / singular[0]
    rank = np.count_nonzero(normalized > CUTOFF)
    ranks.append(rank)
    last_retained.append(normalized[rank - 1])
    first_discarded.append(normalized[rank] if rank < len(normalized) else np.nan)
    condition.append(1.0 / normalized[rank - 1])
ranks = np.array(ranks)
last_retained = np.array(last_retained)
first_discarded = np.array(first_discarded)
condition = np.array(condition)

print("HYPOTHESIS: the fixed-rank implicit derivative is valid only if the retained rank and a spectral separation persist across the geometry trust region.")
print("rank min max unique", ranks.min(), ranks.max(), np.unique(ranks, return_counts=True))
print("last retained normalized min max", last_retained.min(), last_retained.max())
print("first discarded normalized min max", np.nanmin(first_discarded), np.nanmax(first_discarded))
print("minimum cutoff separation factors retained/threshold threshold/discarded", np.min(last_retained / CUTOFF), np.nanmin(CUTOFF / first_discarded))

fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
positions = np.arange(len(POINTS))
axes[0, 0].plot(positions, ranks, marker="o")
axes[0, 0].set(xlabel="27 trust-region corners/planes", ylabel="retained rank", title="Does numerical rank switch?")
axes[0, 1].semilogy(positions, last_retained, marker="o", label="smallest retained")
axes[0, 1].semilogy(positions, first_discarded, marker="o", label="largest discarded")
axes[0, 1].axhline(CUTOFF, color="red", linestyle="--", label="cutoff")
axes[0, 1].set(xlabel="geometry point", ylabel="normalized singular value", title="Gap around the rank cutoff")
axes[0, 1].legend(fontsize=8)
axes[1, 0].semilogy(positions, condition, marker="o")
axes[1, 0].set(xlabel="geometry point", ylabel="retained condition number", title="Condition of the differentiable solve")
rank_cube = ranks.reshape(3, 3, 3)
image = axes[1, 1].imshow(rank_cube[:, :, 1], origin="lower", extent=[-0.02, 0.02, 0.65, 0.75], aspect="auto", cmap="viridis", vmin=ranks.min() - 0.5, vmax=ranks.max() + 0.5)
axes[1, 1].set(xlabel="c4", ylabel="R/r0", title="Retained rank on c8=0 slice")
fig.colorbar(image, ax=axes[1, 1])
fig.suptitle("Claude-tree stress test 53 — is the retained-rank derivative smooth over geometry?", fontsize=14)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: constant rank supports local differentiation; a rank switch or weak gap would require a fixed subspace or regularized solve rather than a hard adaptive cutoff.")
print("saved", OUTPUT)
