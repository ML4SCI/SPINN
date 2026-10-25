import matplotlib.pyplot as plt
import numpy as np


INPUT = "exp/claude_tree_stress_test/results/23_smooth_physical_survival_loss.npz"
OUTPUT = "exp/claude_tree_stress_test/results/24_fabrication_smoothed_gradient.png"
SIGMAS = [0.005, 0.010, 0.020, 0.040]

data = np.load(INPUT)
radii = data["radii"]
beta = data["beta"]
eta_ratio = data["eta_ratio"]
objective = data["exact_survival"]


def robust_curve(sigma):
    values = []
    derivatives = []
    effective_samples = []
    for center in radii:
        weights = np.exp(-0.5 * ((radii - center) / sigma) ** 2)
        weights /= np.sum(weights)
        values.append(np.sum(weights * objective))
        score = (radii - center) / sigma**2
        score -= np.sum(weights * score)
        derivatives.append(np.sum(weights * objective * score))
        effective_samples.append(1.0 / np.sum(weights**2))
    return np.array(values), np.array(derivatives), np.array(effective_samples)


robust = {sigma: robust_curve(sigma) for sigma in SIGMAS}
interior = (radii >= 0.48) & (radii <= 1.12)

print("HYPOTHESIS: averaging exact survival over a physically stated radius-realization error will smooth the shape objective and provide a score-function gradient without differentiating trajectories.")
raw_index = np.argmax(objective)
print("raw optimum radius beta eta probability", radii[raw_index], beta[raw_index], eta_ratio[raw_index], objective[raw_index])
for sigma, (curve, derivative, _) in robust.items():
    allowed = np.where(interior)[0]
    index = allowed[np.argmax(curve[allowed])]
    numerical_derivative = np.gradient(curve, radii)
    print("sigma radius", sigma, "best radius beta eta robust probability", radii[index], beta[index], eta_ratio[index], curve[index], "gradient identity RMSE", np.sqrt(np.mean((derivative[interior] - numerical_derivative[interior]) ** 2)))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].step(radii, objective, where="mid", color="black", label="exact sampled P(T>10)")
for sigma, (curve, _, _) in robust.items():
    axes[0, 0].plot(radii, curve, label=f"radius sigma={sigma:.3f}")
axes[0, 0].set(xlabel="nominal rod radius/r0", ylabel="expected success probability", title="Fabrication uncertainty defines a smooth robust objective")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(radii, np.gradient(objective, radii), color="black", label="raw finite difference")
for sigma, (_, derivative, _) in robust.items():
    axes[0, 1].plot(radii, derivative, label=f"score gradient sigma={sigma:.3f}")
axes[0, 1].set(xlim=[0.48, 1.12], xlabel="nominal rod radius/r0", ylabel="d objective / d radius", title="A usable direction without trajectory backpropagation")
axes[0, 1].legend(fontsize=8)

sigma = 0.02
curve, score_derivative, _ = robust[sigma]
finite_derivative = np.gradient(curve, radii)
axes[0, 2].scatter(finite_derivative[interior], score_derivative[interior], s=18)
limit = np.max(np.abs(np.concatenate([finite_derivative[interior], score_derivative[interior]])))
axes[0, 2].plot([-limit, limit], [-limit, limit], color="black", linestyle="--")
axes[0, 2].set(xlabel="finite derivative of convolution", ylabel="Gaussian score identity", title="Independent gradient identity check")

best_radii = []
best_beta = []
best_eta = []
for sigma, (curve, _, _) in robust.items():
    allowed = np.where(interior)[0]
    index = allowed[np.argmax(curve[allowed])]
    best_radii.append(radii[index])
    best_beta.append(beta[index])
    best_eta.append(eta_ratio[index])
axes[1, 0].plot(SIGMAS, best_radii, marker="o", label="optimal radius")
ax_beta = axes[1, 0].twinx()
ax_beta.plot(SIGMAS, best_beta, marker="s", color="tab:red", label="optimal beta")
axes[1, 0].set(xlabel="radius uncertainty sigma/r0", ylabel="optimal nominal radius/r0", title="Robustification can move the chosen electrode")
ax_beta.set_ylabel("optimal beta", color="tab:red")

axes[1, 1].plot(beta, objective, color="black", label="raw")
for sigma, (curve, _, _) in robust.items():
    axes[1, 1].plot(beta, curve, label=f"sigma R={sigma:.3f}")
axes[1, 1].set(xlabel="beta=a6/a2", ylabel="expected success probability", title="Same result in field coordinates")
axes[1, 1].legend(fontsize=8)

axes[1, 2].plot(radii, eta_ratio, color="black")
for sigma, radius, efficiency in zip(SIGMAS, best_radii, best_eta):
    axes[1, 2].scatter([radius], [efficiency], label=f"sigma R={sigma:.3f}")
axes[1, 2].set(xlabel="nominal rod radius/r0", ylabel="eta / q6-zero eta", title="Robust survival still carries an efficiency cost")
axes[1, 2].legend(fontsize=8)

fig.suptitle("Claude-tree stress test 24 — manufacturing smoothing gives a physical zeroth-order gradient", fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: randomized parameter smoothing converts fabrication tolerance into the optimization distribution and yields a score-function gradient. It avoids differentiating sensitive trajectories, but its sigma is a physical tolerance assumption and must not be tuned only to make optimization easy.")
print("saved", OUTPUT)
