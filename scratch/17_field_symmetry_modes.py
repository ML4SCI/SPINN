import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.models.backends.pig import PigBackend


OUTPUT = "scratch/17_field_symmetry_modes.png"
HYPOTHESIS = "Final-output projection enforces selected PIG symmetries to floating-point precision without changing the PDE domain."


class SymmetryProjection(nn.Module):
    def __init__(self, backend, mode):
        super().__init__()
        self.backend = backend
        self.mode = mode

    def raw(self, points):
        return self.backend(points)

    def mirror_x(self, points):
        return points * points.new_tensor([-1.0, 1.0])

    def mirror_y(self, points):
        return points * points.new_tensor([1.0, -1.0])

    def even_x(self, points):
        return 0.5 * (self.raw(points) + self.raw(self.mirror_x(points)))

    def even_y(self, points):
        return 0.5 * (self.raw(points) + self.raw(self.mirror_y(points)))

    def even_xy(self, points):
        return 0.25 * (
            self.raw(points)
            + self.raw(self.mirror_x(points))
            + self.raw(self.mirror_y(points))
            + self.raw(-points)
        )

    def forward(self, points):
        if self.mode == "none":
            return self.raw(points)
        if self.mode == "mirror_x":
            return self.even_x(points)
        if self.mode == "mirror_y":
            return self.even_y(points)
        if self.mode == "two_axes":
            return self.even_xy(points)
        if self.mode == "quadrupole":
            return 0.5 * (self.even_xy(points) - self.even_xy(points[:, [1, 0]]))


def parity_errors(model, points):
    value = model(points)
    scale = torch.max(torch.abs(value)).clamp_min(1.0e-12)
    error_x = torch.max(torch.abs(value - model(points * points.new_tensor([-1.0, 1.0])))) / scale
    error_y = torch.max(torch.abs(value - model(points * points.new_tensor([1.0, -1.0])))) / scale
    error_swap = torch.max(torch.abs(value + model(points[:, [1, 0]]))) / scale
    return np.array([error_x.item(), error_y.item(), error_swap.item()])


print("HYPOTHESIS:", HYPOTHESIS)
torch.manual_seed(4)
pig = PigBackend(
    outer_radius=5.0,
    num_gaussians=64,
    feature_dim=4,
    init_sigma=0.16,
    decoder_layers=2,
    decoder_width=16,
)

axis = torch.linspace(-5.0, 5.0, 121)
gx, gy = torch.meshgrid(axis, axis, indexing="xy")
points = torch.column_stack([gx.reshape(-1), gy.reshape(-1)])
modes = ["none", "mirror_x", "mirror_y", "two_axes", "quadrupole"]
fields = []
errors = []
with torch.no_grad():
    for mode in modes:
        model = SymmetryProjection(pig, mode)
        fields.append(model(points).reshape(121, 121).numpy())
        errors.append(parity_errors(model, points))
errors = np.array(errors)
limit = max(np.max(np.abs(field)) for field in fields)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for ax, mode, field in zip(axes.ravel()[:5], modes, fields):
    image = ax.imshow(field, origin="lower", extent=(-5, 5, -5, 5), cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set(xlabel="x/r0", ylabel="y/r0", title=mode)

labels = ["x reflection", "y reflection", "axis-swap antisymmetry"]
x = np.arange(len(modes))
width = 0.25
for index, label in enumerate(labels):
    axes[1, 2].semilogy(x + (index - 1) * width, np.maximum(errors[:, index], 1e-9), marker="o", linestyle="none", label=label)
axes[1, 2].set_xticks(x, modes, rotation=25)
axes[1, 2].set(ylabel="maximum normalized parity error", title="What each toggle actually enforces")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.2)
fig.colorbar(image, ax=axes.ravel()[:5], label="untrained PIG potential")
fig.suptitle("Hypothesis: output projection gives exact, toggleable field symmetry", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

for mode, row in zip(modes, errors):
    print(f"{mode:12s} normalized errors x={row[0]:.3e} y={row[1]:.3e} swap={row[2]:.3e}")
quad_max = errors[-1].max()
print("CONCLUSION: quadrupole mode enforces all requested identities to", f"{quad_max:.3e}")
print("The single-axis modes leave the other identities unrestricted, as intended.")
print("saved", OUTPUT)
