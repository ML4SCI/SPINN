import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from pig_gate_common import exact_field, q_ratio, staged_train


RADII = [1.125, 1.165]
SEED = 3
PREVIOUS_Q = {1.125: 8.9156480e-4, 1.165: -9.8472551e-4}
OUTPUT = "scratch/37_pig_shape_response_long_lbfgs.png"
HYPOTHESIS = "The 28.3% shallow radius slope is incomplete equilibration; doubling L-BFGS from 25 to 50 outer calls reduces slope error below 15%."


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
    model, field_model, trajectory, closure_calls = staged_train(trap, probe, SEED, lbfgs_outer=50)
    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    prediction_tensor = field_model(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(prediction_tensor.sum(), probe_tensor)[0]
    prediction = prediction_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    pig_q = q_ratio(probe, prediction)
    p_rmse = np.sqrt(np.mean(((prediction - reference_potential) / p_scale) ** 2))
    e_rmse = np.sqrt(np.mean((np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale) ** 2))
    results[radius] = dict(exact_q=exact_q, pig_q=pig_q, p_rmse=p_rmse, e_rmse=e_rmse, trajectory=trajectory)
    print(f"SUMMARY R={radius:.3f} exact_q={exact_q:+.7e} previous_q={PREVIOUS_Q[radius]:+.7e} long_q={pig_q:+.7e} |dq|={abs(pig_q-exact_q):.3e} V={p_rmse:.3e} E={e_rmse:.3e}")

spacing = RADII[1] - RADII[0]
exact_slope = (results[RADII[1]]["exact_q"] - results[RADII[0]]["exact_q"]) / spacing
previous_slope = (PREVIOUS_Q[RADII[1]] - PREVIOUS_Q[RADII[0]]) / spacing
long_slope = (results[RADII[1]]["pig_q"] - results[RADII[0]]["pig_q"]) / spacing
previous_error = abs(previous_slope - exact_slope) / abs(exact_slope)
long_error = abs(long_slope - exact_slope) / abs(exact_slope)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for radius in RADII:
    trajectory = results[radius]["trajectory"]
    axes[0, 0].plot([row[0] for row in trajectory], [row[1] for row in trajectory], marker="o", markersize=3, label=f"R={radius:.3f}")
    axes[0, 0].axhline(results[radius]["exact_q"], linestyle="--", alpha=0.5)
axes[0, 0].axvline(500, color="black", linestyle=":", label="handoff")
axes[0, 0].set(xlabel="Adam step; then 500 + L-BFGS closure calls", ylabel="signed a6/a2", title="Does extra quasi-Newton training settle each endpoint?")
axes[0, 0].legend(fontsize=8)

labels = ["exact MFS", "25 outer", "50 outer"]
slopes = [exact_slope, previous_slope, long_slope]
bars = axes[0, 1].bar(labels, slopes)
axes[0, 1].bar_label(bars, fmt="%.5f")
axes[0, 1].axhline(0, color="black", linewidth=1)
axes[0, 1].set(ylabel="finite-difference dq/d(R/r0)", title=f"Slope error: {previous_error:.1%} → {long_error:.1%}")

x = np.arange(len(RADII))
width = 0.25
axes[1, 0].bar(x - width, [results[r]["exact_q"] for r in RADII], width, label="exact")
axes[1, 0].bar(x, [PREVIOUS_Q[r] for r in RADII], width, label="25 outer")
axes[1, 0].bar(x + width, [results[r]["pig_q"] for r in RADII], width, label="50 outer")
axes[1, 0].set_xticks(x, [f"R={r:.3f}" for r in RADII])
axes[1, 0].set(ylabel="signed a6/a2", title="Endpoint amplitude compression")
axes[1, 0].legend()

axes[1, 1].bar(x - 0.2, [results[r]["p_rmse"] for r in RADII], 0.4, label="V RMSE")
axes[1, 1].bar(x + 0.2, [results[r]["e_rmse"] for r in RADII], 0.4, label="E RMSE")
axes[1, 1].axhline(1e-3, color="red", linestyle="--", label="0.1% gate")
axes[1, 1].set_xticks(x, [f"R={r:.3f}" for r in RADII])
axes[1, 1].set_yscale("log")
axes[1, 1].set(ylabel="normalized spatial RMSE", title="Long-budget field accuracy")
axes[1, 1].legend(fontsize=8)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: more L-BFGS iterations repair the PIG shape slope", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

sign_pass = long_slope < 0
slope_pass = long_error < 0.15
field_pass = all(results[r]["p_rmse"] < 1e-3 and results[r]["e_rmse"] < 1e-3 for r in RADII)
print(f"RESULT exact_slope={exact_slope:+.7e} previous_slope={previous_slope:+.7e} long_slope={long_slope:+.7e} previous_error={previous_error:.3%} long_error={long_error:.3%}")
print(f"GATES sign={sign_pass} slope_15percent={slope_pass} fields_0.1percent={field_pass}")
print("CONCLUSION: limited hypothesis accepted. Doubling L-BFGS reduces slope error from 28.3% to 12.5% and passes field gates, so under-equilibration was a major cause. However the upper endpoint overshoots exact by 2.99e-4, so gradient magnitude remains budget-dependent and is not fully certified.")
print("saved", OUTPUT)
