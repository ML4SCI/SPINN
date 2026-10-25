import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "exp/harmonic_source_shape_search/results/17_pixel_kernel_smoothness.png"
NODE_VALUES = np.array([0.2, 1.0, -0.4, 0.7, 0.1])


def kernel(frac, name):
    if name == "cosine":
        return (
            0.5 * (1.0 - np.cos(np.pi * frac)),
            0.5 * np.pi * np.sin(np.pi * frac),
            0.5 * np.pi**2 * np.cos(np.pi * frac),
        )
    if name == "quintic":
        return (
            6.0 * frac**5 - 15.0 * frac**4 + 10.0 * frac**3,
            30.0 * frac**2 * (frac - 1.0) ** 2,
            60.0 * frac * (2.0 * frac**2 - 3.0 * frac + 1.0),
        )
    return frac, np.ones_like(frac), np.zeros_like(frac)


def interpolate(x, name):
    cell = np.floor(x).astype(int).clip(0, len(NODE_VALUES) - 2)
    frac = x - cell
    w, dw, d2w = kernel(frac, name)
    delta = NODE_VALUES[cell + 1] - NODE_VALUES[cell]
    return NODE_VALUES[cell] + delta * w, delta * dw, delta * d2w


print("HYPOTHESIS: PIXEL's cosine kernel is C1 but not C2 across cell boundaries; a quintic smoothstep makes the potential, field, and Laplacian continuous without changing locality or parameter count.")
x = np.linspace(0.0, 4.0, 4001)
names = ["bilinear", "cosine", "quintic"]
colors = ["tab:red", "tab:blue", "tab:green"]
results = {name: interpolate(x, name) for name in names}

epsilon = 1.0e-7
jumps = {}
for name in names:
    boundary_jumps = []
    for edge in [1.0, 2.0, 3.0]:
        left = interpolate(np.array([edge - epsilon]), name)
        right = interpolate(np.array([edge + epsilon]), name)
        boundary_jumps.append([abs(right[i][0] - left[i][0]) for i in range(3)])
    jumps[name] = np.array(boundary_jumps)
    print(name, "maximum [value, first, second] boundary jump", jumps[name].max(axis=0))

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
labels = ["interpolated value", "first derivative (field)", "second derivative (Laplacian component)"]
for column in range(3):
    for name, color in zip(names, colors):
        axes[0, column].plot(x, results[name][column], color=color, label=name)
    for edge in [1, 2, 3]:
        axes[0, column].axvline(edge, color="grey", linewidth=0.7, alpha=0.5)
    axes[0, column].set(xlabel="position / pixel spacing", ylabel=labels[column], title=labels[column])
    axes[0, column].legend(fontsize=8)

for column, derivative_name in enumerate(["value", "first derivative", "second derivative"]):
    width = 0.24
    for index, (name, color) in enumerate(zip(names, colors)):
        axes[1, column].bar(np.arange(3) + (index - 1) * width, jumps[name][:, column], width, color=color, label=name)
    axes[1, column].set_yscale("log")
    axes[1, column].set_xticks(np.arange(3), ["edge 1", "edge 2", "edge 3"])
    axes[1, column].set(ylabel="left/right jump", title=f"Continuity of {derivative_name}")
    axes[1, column].legend(fontsize=8)
axes[1, 2].text(0.03, 0.95, "bilinear: distributional delta at edge\n(not represented by pointwise autograd)", transform=axes[1, 2].transAxes, va="top", fontsize=9, color="tab:red")
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Plot 17 — PIXEL kernel smoothness under the derivatives used by Laplace's equation", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("CONCLUSION: use quintic interpolation when differentiating PIXEL twice; cosine is only C1 across cells and bilinear is only C0. The option should remain toggleable for controlled comparison.")
print("saved", OUTPUT)
