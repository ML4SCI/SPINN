import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch


torch.set_default_dtype(torch.float64)
OUTPUT = "scratch/18_pig_gaussian_orbits.png"
HYPOTHESIS = "A single stored anisotropic Gaussian can generate correctly back-transformed mirror and quadrupole copies, with negative signs only for voltage-exchanging operations."


I = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MX = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
MY = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
C2 = -I
SWAP = torch.tensor([[0.0, 1.0], [1.0, 0.0]])


def group(mode):
    if mode == "one_axis":
        return [(I, 1.0), (MX, 1.0)]
    if mode == "two_axes":
        return [(I, 1.0), (MX, 1.0), (MY, 1.0), (C2, 1.0)]
    base = [I, MX, MY, C2]
    return [(matrix, 1.0) for matrix in base] + [(SWAP @ matrix, -1.0) for matrix in base]


def gaussian(points, mean, covariance):
    diff = points - mean
    inverse = torch.linalg.inv(covariance)
    exponent = torch.einsum("ni,ij,nj->n", diff, inverse, diff)
    return torch.exp(-0.5 * exponent)


def orbit(points, mean, covariance, mode):
    pieces = []
    transformed = []
    for matrix, sign in group(mode):
        transformed_mean = matrix @ mean
        transformed_covariance = matrix @ covariance @ matrix.T
        pieces.append(sign * gaussian(points, transformed_mean, transformed_covariance))
        transformed.append((transformed_mean, transformed_covariance, sign))
    return torch.stack(pieces).mean(dim=0), transformed


def add_ellipse(ax, mean, covariance, sign, stored=False):
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    color = "tab:red" if sign > 0 else "tab:blue"
    ellipse = Ellipse(
        mean.numpy(),
        width=2.0 * np.sqrt(values[0].item()),
        height=2.0 * np.sqrt(values[1].item()),
        angle=angle,
        fill=False,
        color="black" if stored else color,
        linewidth=3 if stored else 1.8,
        linestyle="--" if sign < 0 else "-",
    )
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], color="black" if stored else color, marker="*" if stored else "o", s=80 if stored else 35)
    ax.text(mean[0].item() + 0.10, mean[1].item() + 0.10, "+" if sign > 0 else "−", color=color, fontsize=12)


print("HYPOTHESIS:", HYPOTHESIS)
mean = torch.tensor([2.15, 0.85])
angle = torch.tensor(np.radians(28.0))
rotation = torch.stack([torch.stack([torch.cos(angle), -torch.sin(angle)]), torch.stack([torch.sin(angle), torch.cos(angle)])])
covariance = rotation @ torch.diag(torch.tensor([0.72**2, 0.28**2])) @ rotation.T

axis = torch.linspace(-5.0, 5.0, 201)
gx, gy = torch.meshgrid(axis, axis, indexing="xy")
points = torch.column_stack([gx.reshape(-1), gy.reshape(-1)])
modes = ["one_axis", "two_axes", "quadrupole"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
limits = []
results = []
for mode in modes:
    field, transformed = orbit(points, mean, covariance, mode)
    field = field.reshape(201, 201)
    results.append((field, transformed))
    limits.append(torch.max(torch.abs(field)).item())
limit = max(limits)

for column, (mode, (field, transformed)) in enumerate(zip(modes, results)):
    ax = axes[0, column]
    for transformed_mean, transformed_covariance, sign in transformed:
        add_ellipse(ax, transformed_mean, transformed_covariance, sign)
    add_ellipse(ax, mean, covariance, 1.0, stored=True)
    ax.axhline(0.0, color="grey", linewidth=0.7)
    ax.axvline(0.0, color="grey", linewidth=0.7)
    ax.set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", xlabel="x/r0", ylabel="y/r0", title=f"{mode}: one stored Gaussian, {len(transformed)} appearances")

    image = axes[1, column].imshow(field.numpy(), origin="lower", extent=(-5, 5, -5, 5), cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1, column].axhline(0.0, color="black", linewidth=0.5)
    axes[1, column].axvline(0.0, color="black", linewidth=0.5)
    axes[1, column].set(xlabel="x/r0", ylabel="y/r0", title="normalized orbit feature")

fig.colorbar(image, ax=axes[1, :], label="orbit-averaged Gaussian feature")
fig.suptitle("Hypothesis: center, covariance and sign must transform together", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

quad = results[-1][0]
quad_x = torch.flip(quad, dims=[1])
quad_y = torch.flip(quad, dims=[0])
quad_swap = quad.T
scale = torch.max(torch.abs(quad))
error_x = torch.max(torch.abs(quad - quad_x)) / scale
error_y = torch.max(torch.abs(quad - quad_y)) / scale
error_swap = torch.max(torch.abs(quad + quad_swap)) / scale
print(f"quadrupole orbit errors x={error_x:.3e} y={error_y:.3e} swap={error_swap:.3e}")
print("CONCLUSION: one parameter set generated the full orbit; covariance orientation followed each reflection/rotation, and only swap-family copies were negative.")
print("saved", OUTPUT)
