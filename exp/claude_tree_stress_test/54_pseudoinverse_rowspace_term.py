import matplotlib.pyplot as plt
import numpy as np
import runpy


OUTPUT = "exp/claude_tree_stress_test/results/54_pseudoinverse_rowspace_term.png"
BASE = runpy.run_path("exp/claude_tree_stress_test/52_analytic_mfs_derivative.py")
INNER = BASE["BASE"]

matrix = BASE["matrix"]
values = BASE["values"]
evaluation = BASE["evaluation"]
projection = BASE["projection"]
inverse = BASE["inverse"]
coefficients = BASE["coefficients"]
singular = BASE["singular"]
keep = BASE["keep"]
residual = BASE["residual"]
finite_potential = BASE["finite_potential"]
finite_beta = BASE["finite_beta"]
boundary = BASE["boundary"]
sources = BASE["sources"]
boundary_derivatives = BASE["boundary_derivatives"]
source_derivatives = BASE["source_derivatives"]
ring = BASE["ring"]
delta = boundary[:, None] - sources[None, :]
ring_delta = ring[:, None] - sources[None, :]

null_projector = np.eye(matrix.shape[1]) - inverse @ matrix
two_term_potential = []
full_potential = []
extra_norms = []
two_term_beta = []
full_beta = []

print("HYPOTHESIS: if the discarded singular direction is a moving coefficient gauge, the omitted Moore-Penrose row-space term will be nonzero and the full three-term derivative will agree better with fresh solves.")

for axis in range(3):
    matrix_derivative = np.column_stack([
        np.real((boundary_derivatives[axis, :, None] - source_derivatives[axis, None, :]) / delta),
        np.zeros(len(boundary)),
    ])
    evaluation_derivative = np.column_stack([
        np.real(-source_derivatives[axis, None, :] / ring_delta),
        np.zeros(len(ring)),
    ])
    first = -inverse @ (matrix_derivative @ coefficients)
    residual_term = inverse @ inverse.T @ (matrix_derivative.T @ residual)
    rowspace_term = null_projector @ matrix_derivative.T @ inverse.T @ coefficients
    two_term_dc = first + residual_term
    full_dc = two_term_dc + rowspace_term
    two_term_v = evaluation_derivative @ coefficients + evaluation @ two_term_dc
    full_v = evaluation_derivative @ coefficients + evaluation @ full_dc
    two_term_potential.append(two_term_v)
    full_potential.append(full_v)
    extra_norms.append(np.linalg.norm(rowspace_term) / np.linalg.norm(two_term_dc))
    two_term_beta.append(INNER["beta_derivative"](INNER["base_harmonics"], projection @ two_term_v))
    full_beta.append(INNER["beta_derivative"](INNER["base_harmonics"], projection @ full_v))

two_term_potential = np.array(two_term_potential)
full_potential = np.array(full_potential)
two_term_beta = np.array(two_term_beta)
full_beta = np.array(full_beta)
two_spatial_error = np.linalg.norm(two_term_potential - finite_potential, axis=1) / np.linalg.norm(finite_potential, axis=1)
full_spatial_error = np.linalg.norm(full_potential - finite_potential, axis=1) / np.linalg.norm(finite_potential, axis=1)
two_beta_error = np.abs((two_term_beta - finite_beta) / finite_beta)
full_beta_error = np.abs((full_beta - finite_beta) / finite_beta)

print("row-space term / two-term coefficient derivative", extra_norms)
print("two-term spatial errors", two_spatial_error)
print("full MP spatial errors", full_spatial_error)
print("two-term beta errors", two_beta_error)
print("full MP beta errors", full_beta_error)

labels = ["R", "c4", "c8"]
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
axes[0].bar(labels, extra_norms, color="tab:purple")
axes[0].set_yscale("log")
axes[0].set(ylabel="relative coefficient contribution", title="Omitted row-space term")

x = np.arange(3)
width = 0.36
axes[1].bar(x - width / 2, two_spatial_error, width, label="two-term", color="tab:blue")
axes[1].bar(x + width / 2, full_spatial_error, width, label="full Moore-Penrose", color="tab:orange")
axes[1].set_yscale("log")
axes[1].set_xticks(x, labels)
axes[1].set(ylabel="relative spatial derivative error", title="Against fresh field solves")
axes[1].legend(fontsize=9)

axes[2].bar(x - width / 2, two_beta_error, width, label="two-term", color="tab:blue")
axes[2].bar(x + width / 2, full_beta_error, width, label="full Moore-Penrose", color="tab:orange")
axes[2].set_yscale("log")
axes[2].set_xticks(x, labels)
axes[2].set(ylabel="relative beta-gradient error", title="Optimizer-facing comparison")
axes[2].legend(fontsize=9)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
fig.savefig(OUTPUT, dpi=180)

if np.max(full_spatial_error) < np.max(two_spatial_error):
    print("CONCLUSION: the row-space term is physically relevant and improves fresh-solve agreement; update the analytic layer and presentation formula.")
else:
    print("CONCLUSION: the row-space term does not improve the tested derivative; inspect whether the discarded direction is an invariant gauge before retaining the shorter formula.")
print("saved", OUTPUT)
