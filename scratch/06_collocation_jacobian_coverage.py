import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from spinn.config import Config
from spinn.geometry.four_rod import build_trap
from spinn.geometry.masks import design_mask


SEED = 7
N = 4096
WIDTH = 0.35
DISPLACEMENT = -0.15
GRID = 181
OUTPUT = "scratch/06_collocation_jacobian_coverage.png"


trap = build_trap(Config())
rng = np.random.default_rng(SEED)
centers = torch.tensor([[e.cx, e.cy] for e in trap.electrodes()], dtype=torch.float64)
directions = centers / torch.linalg.vector_norm(centers, dim=1, keepdim=True)


def map_points(points):
    diff = points[:, None, :] - centers[None, :, :]
    distance2 = torch.sum(diff**2, dim=2)
    surface = (distance2 - trap.rod_radius_abs**2) / (2.0 * trap.rod_radius_abs)
    weights = torch.exp(-(surface / WIDTH) ** 2)
    proposal = (weights[:, :, None] * directions[None, :, :]).sum(dim=1) / (
        weights.sum(dim=1, keepdim=True) + 1e-12
    )
    return points + DISPLACEMENT * design_mask(points, trap, band_width=WIDTH) * proposal


def determinant(points):
    z = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    mapped = map_points(z)
    row0 = torch.autograd.grad(mapped[:, 0].sum(), z, create_graph=True)[0]
    row1 = torch.autograd.grad(mapped[:, 1].sum(), z)[0]
    jac = torch.stack([row0, row1], dim=1)
    return torch.linalg.det(jac).detach().numpy()


def sample_surface_vacuum(n):
    collected = []
    while sum(len(part) for part in collected) < n:
        count = max(2 * n, 1024)
        index = rng.integers(0, 4, count)
        theta = rng.uniform(0.0, 2.0 * np.pi, count)
        offset = rng.uniform(1e-3, 0.55, count)
        electrodes = trap.electrodes()
        cx = np.array([electrodes[i].cx for i in index])
        cy = np.array([electrodes[i].cy for i in index])
        radius = np.array([electrodes[i].radius for i in index]) + offset
        points = np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
        collected.append(points[trap.in_vacuum(points, margin=1e-3)])
    return np.vstack(collected)[:n]


def sample_annulus(n, lower, upper):
    radius = np.sqrt(rng.uniform(lower**2, upper**2, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


uniform = trap.sample_interior(N, rng)
stratified = np.vstack([
    trap.sample_interior(2048, rng),
    sample_surface_vacuum(1024),
    sample_annulus(512, 0.40, 0.90),
    sample_annulus(512, 4.50, 4.99),
])

uniform_det = determinant(uniform)
stratified_det = determinant(stratified)

axis = np.linspace(-trap.outer_radius, trap.outer_radius, GRID)
gx, gy = np.meshgrid(axis, axis)
grid_points = np.column_stack([gx.ravel(), gy.ravel()])
grid_det = determinant(grid_points)
world = np.hypot(grid_points[:, 0], grid_points[:, 1]) <= trap.outer_radius
grid_plot = np.where(world, grid_det, np.nan).reshape(GRID, GRID)

fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
for ax, points, title in [
    (axes[0, 0], uniform, "Current sampler: uniform reference-vacuum points"),
    (axes[0, 1], stratified, "Candidate sampler: global + collar + anchor transitions"),
]:
    image = ax.imshow(
        grid_plot, origin="lower", extent=[-5, 5, -5, 5],
        cmap="coolwarm", vmin=0.4, vmax=1.4,
    )
    ax.scatter(points[:, 0], points[:, 1], s=1.5, color="black", alpha=0.35)
    for electrode in trap.electrodes():
        ax.add_patch(Circle((electrode.cx, electrode.cy), electrode.radius, fill=False, color="white", linewidth=0.8))
    ax.add_patch(Circle((0, 0), 0.40, fill=False, color="magenta", linewidth=1.2))
    ax.set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5), title=title, xlabel=r"$x/r_0$", ylabel=r"$y/r_0$")
fig.colorbar(image, ax=axes[0, :], label=r"manufactured $\det J$", shrink=0.8)

bins = np.linspace(0.35, 1.45, 60)
axes[1, 0].hist(uniform_det, bins=bins, density=True, alpha=0.65, label="uniform")
axes[1, 0].hist(stratified_det, bins=bins, density=True, alpha=0.65, label="stratified")
axes[1, 0].axvline(0.6, color="red", linestyle="--")
axes[1, 0].set(xlabel=r"$\det J$ at sampled points", ylabel="sample density", title="Does training see the distorted region?")
axes[1, 0].legend()

thresholds = [0.9, 0.75, 0.6, 0.5]
u_fraction = [np.mean(uniform_det < threshold) for threshold in thresholds]
s_fraction = [np.mean(stratified_det < threshold) for threshold in thresholds]
positions = np.arange(len(thresholds))
axes[1, 1].bar(positions - 0.18, u_fraction, width=0.36, label="uniform")
axes[1, 1].bar(positions + 0.18, s_fraction, width=0.36, label="stratified")
axes[1, 1].set_xticks(positions, [f"detJ < {value}" for value in thresholds])
axes[1, 1].set(ylabel="fraction of collocation points", title="Low-Jacobian coverage")
axes[1, 1].legend()

fig.suptitle("Collocation stress test on the 0.15 r0 inward electrode deformation", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("uniform min det and fractions:", uniform_det.min(), u_fraction)
print("stratified min det and fractions:", stratified_det.min(), s_fraction)
print("both samplers use", len(uniform), "points")
print("saved", OUTPUT)
