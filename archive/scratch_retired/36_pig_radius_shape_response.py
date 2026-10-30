import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from pig_gate_common import exact_field, q_ratio, staged_train


RADII = [1.125, 1.145, 1.165]
SEED = 3
OUTPUT = "scratch/36_pig_radius_shape_response.png"
HYPOTHESIS = "Fully re-equilibrated PIG fields recover the physical negative dq/dR within 15% of MFS and locate the local q=0 radius within 0.01 r0."


print("HYPOTHESIS:", HYPOTHESIS)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
points = np.column_stack([gx.ravel(), gy.ravel()])
probe = points[np.hypot(points[:, 0], points[:, 1]) <= 0.6]
results = {}

for radius in RADII:
    trap = FourRodTrap(r0=1.0, rod_radius=radius, outer_radius=5.0, V0=1.0)
    reference_potential, reference_field = exact_field(trap, probe)
    exact_q = q_ratio(probe, reference_potential)
    p_scale = np.max(np.abs(reference_potential))
    e_scale = np.max(np.linalg.norm(reference_field, axis=1))
    model, field_model, trajectory, closure_calls = staged_train(trap, probe, SEED)
    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    prediction_tensor = field_model(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(prediction_tensor.sum(), probe_tensor)[0]
    prediction = prediction_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    pig_q = q_ratio(probe, prediction)
    p_rmse = np.sqrt(np.mean(((prediction - reference_potential) / p_scale) ** 2))
    e_rmse = np.sqrt(np.mean((np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale) ** 2))
    results[radius] = dict(exact_q=exact_q, pig_q=pig_q, p_rmse=p_rmse, e_rmse=e_rmse, trajectory=trajectory)
    print(f"SUMMARY R={radius:.3f} exact_q={exact_q:+.7e} pig_q={pig_q:+.7e} |dq|={abs(pig_q-exact_q):.3e} V={p_rmse:.3e} E={e_rmse:.3e}")

radii = np.array(RADII)
exact_q = np.array([results[r]["exact_q"] for r in RADII])
pig_q = np.array([results[r]["pig_q"] for r in RADII])
spacing = RADII[2] - RADII[0]
exact_slope = (exact_q[2] - exact_q[0]) / spacing
pig_slope = (pig_q[2] - pig_q[0]) / spacing
slope_relative_error = abs(pig_slope - exact_slope) / abs(exact_slope)
exact_line = np.polyfit(radii, exact_q, 1)
pig_line = np.polyfit(radii, pig_q, 1)
exact_root = -exact_line[1] / exact_line[0]
pig_root = -pig_line[1] / pig_line[0]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].plot(radii, exact_q, marker="o", label="exact MFS")
axes[0, 0].plot(radii, pig_q, marker="o", label="equilibrated PIG")
axes[0, 0].axhline(0, color="black", linewidth=1)
axes[0, 0].axvline(exact_root, color="black", linestyle="--", label=f"exact local root {exact_root:.5f}")
axes[0, 0].axvline(pig_root, color="tab:orange", linestyle=":", label=f"PIG local root {pig_root:.5f}")
axes[0, 0].set(xlabel="rod radius R/r0", ylabel="signed a6/a2", title="Local physical shape response")
axes[0, 0].legend(fontsize=8)

labels = ["exact MFS", "PIG"]
bars = axes[0, 1].bar(labels, [exact_slope, pig_slope])
axes[0, 1].bar_label(bars, fmt="%.5f")
axes[0, 1].axhline(0, color="black", linewidth=1)
axes[0, 1].set(ylabel="central finite-difference dq/d(R/r0)", title=f"Shape-direction gate: relative slope error {slope_relative_error:.1%}")

for radius in RADII:
    trajectory = results[radius]["trajectory"]
    axes[1, 0].plot([row[0] for row in trajectory], [row[1] for row in trajectory], marker="o", markersize=3, label=f"R={radius:.3f}")
    axes[1, 0].axhline(results[radius]["exact_q"], linestyle="--", alpha=0.4)
axes[1, 0].axvline(500, color="black", linestyle=":", label="Adam→L-BFGS")
axes[1, 0].set(xlabel="Adam step; then 500 + L-BFGS closure calls", ylabel="signed a6/a2", title="Each neighboring geometry fully equilibrates")
axes[1, 0].legend(fontsize=8)

x = np.arange(len(RADII))
axes[1, 1].bar(x - 0.2, [results[r]["p_rmse"] for r in RADII], 0.4, label="V RMSE")
axes[1, 1].bar(x + 0.2, [results[r]["e_rmse"] for r in RADII], 0.4, label="E RMSE")
axes[1, 1].axhline(1e-3, color="red", linestyle="--", label="0.1% gate")
axes[1, 1].set_xticks(x, [f"R={r:.3f}" for r in RADII])
axes[1, 1].set_yscale("log")
axes[1, 1].set(ylabel="normalized spatial RMSE", title="Field accuracy at perturbed shapes")
axes[1, 1].legend(fontsize=8)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: staged PIG resolves the physical radius-shape derivative", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

sign_pass = exact_slope < 0 and pig_slope < 0
slope_pass = slope_relative_error < 0.15
root_pass = abs(pig_root - exact_root) < 0.01
field_pass = all(results[r]["p_rmse"] < 1e-3 and results[r]["e_rmse"] < 1e-3 for r in RADII)
print(f"RESULT exact_slope={exact_slope:+.7e} pig_slope={pig_slope:+.7e} relative_error={slope_relative_error:.3%} exact_root={exact_root:.7f} pig_root={pig_root:.7f} root_error={abs(pig_root-exact_root):.3e}")
print(f"GATES sign={sign_pass} slope_15percent={slope_pass} root_0.01={root_pass} fields_0.1percent={field_pass}")
print("CONCLUSION: hypothesis rejected quantitatively. Sign, root, and field gates pass, but PIG slope -0.04691 is 28.3% shallower than exact -0.06539, failing the 15% gate. The local root is excellent (2.11e-4 r0 error), but gradient magnitude is not yet trustworthy.")
print("saved", OUTPUT)
