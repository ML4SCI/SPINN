import matplotlib.pyplot as plt
import numpy as np
import torch


OUTPUT = "scratch/49_pixel_kernel_derivative_continuity.png"
HYPOTHESIS = "PIXEL's cosine interpolation is C1 but not C2 across cell boundaries; a quintic smootherstep keeps the same local interpolation while removing the second-derivative jumps."
CONTROL = torch.tensor([0.10, 0.75, -0.35, 0.55, 0.20, -0.50], dtype=torch.float64)


def cosine(t):
    return 0.5 * (1.0 - torch.cos(torch.pi * t))


def quintic(t):
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def interpolate(x, transition):
    left = torch.floor(x).long().clamp(0, len(CONTROL) - 2)
    t = x - left
    return (1.0 - transition(t)) * CONTROL[left] + transition(t) * CONTROL[left + 1]


def value_derivatives(x_values, transition):
    x = torch.tensor(x_values, dtype=torch.float64, requires_grad=True)
    value = interpolate(x, transition)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    return value.detach().numpy(), first.detach().numpy(), second.detach().numpy()


print("HYPOTHESIS:", HYPOTHESIS)
x = np.linspace(0.0, len(CONTROL) - 1 - 1e-7, 2501)
curves = {
    "cosine (current PIXEL)": value_derivatives(x, cosine),
    "quintic C2 candidate": value_derivatives(x, quintic),
}

nodes = np.arange(1, len(CONTROL) - 1, dtype=float)
epsilon = 1e-7
jumps = {}
for name, transition in [("cosine", cosine), ("quintic", quintic)]:
    left_second = value_derivatives(nodes - epsilon, transition)[2]
    right_second = value_derivatives(nodes + epsilon, transition)[2]
    jumps[name] = np.abs(right_second - left_second)
    print(f"{name:8s} max one-sided second-derivative jump = {jumps[name].max():.6f}")

analytic_cosine_jump = 0.5 * np.pi**2 * np.abs(
    CONTROL[2:].numpy() - CONTROL[:-2].numpy()
)
print(f"cosine analytic/numeric jump mismatch = {np.max(np.abs(analytic_cosine_jump - jumps['cosine'])):.3e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
colors = ["tab:blue", "tab:orange"]
for (name, (value, first, second)), color in zip(curves.items(), colors):
    axes[0, 0].plot(x, value, color=color, label=name)
    axes[0, 1].plot(x, first, color=color, label=name)
    axes[1, 0].plot(x, second, color=color, label=name)
axes[0, 0].scatter(np.arange(len(CONTROL)), CONTROL, color="black", zorder=5, label="learned cell values")
axes[0, 0].set(title="Potential interpolation", xlabel="grid coordinate", ylabel="V")
axes[0, 1].set(title="First derivative (electric-field continuity)", xlabel="grid coordinate", ylabel="dV/dx")
axes[1, 0].set(title="Second derivative (PDE quantity)", xlabel="grid coordinate", ylabel="d2V/dx2")

width = 0.34
axes[1, 1].bar(nodes - width / 2, jumps["cosine"], width, label="cosine: numeric jump", color="tab:blue")
axes[1, 1].bar(nodes + width / 2, jumps["quintic"], width, label="quintic: numeric jump", color="tab:orange")
axes[1, 1].scatter(nodes, analytic_cosine_jump, marker="x", s=70, color="black", label="cosine: analytic jump")
axes[1, 1].set(title="One-sided jump at each cell boundary", xlabel="grid boundary", ylabel="|right d2V/dx2 - left d2V/dx2|")

for ax in axes.ravel():
    for node in nodes:
        ax.axvline(node, color="grey", alpha=0.17, linewidth=0.8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("Plot-first PIXEL diagnostic: does the interpolation have the smoothness Laplace's equation needs?", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

if jumps["cosine"].max() > 1.0 and jumps["quintic"].max() < 1e-4:
    print("CONCLUSION: the plot confirms the predicted C2 defect in the current cosine interpolant. The quintic transition removes that defect, so it is worth testing on the fixed Paul-trap PDE next.")
else:
    print("CONCLUSION: the predicted separation was not observed; do not change the physical PIXEL experiment.")
print("saved", OUTPUT)
