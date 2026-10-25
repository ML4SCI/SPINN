import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from spinn.models.shape_network import ResidualShapeNetwork


OUTPUT = "scratch/43_toggleable_shape_symmetry_electrodes.png"
HYPOTHESIS = "Vector Reynolds projections make electrode motion obey exactly the selected mirror group while the C2 deformation mask keeps the center and grounded enclosure fixed and the map non-folded."


def boundary_points(trap, count=160):
    theta = torch.linspace(0, 2*torch.pi, count+1)[:-1]
    rows = []
    for electrode in trap.electrodes():
        rows.append(torch.stack([electrode.cx + electrode.radius*torch.cos(theta), electrode.cy + electrode.radius*torch.sin(theta)], dim=1))
    return rows


def jacobian_minimum(model, points):
    points = points.clone().requires_grad_(True)
    mapped = model(points)
    rows = []
    for component in range(2):
        rows.append(torch.autograd.grad(mapped[:, component].sum(), points, create_graph=False, retain_graph=True)[0])
    jacobian = torch.stack(rows, dim=1)
    determinant = jacobian[:, 0, 0]*jacobian[:, 1, 1] - jacobian[:, 0, 1]*jacobian[:, 1, 0]
    return determinant.min().item()


def equivariance_error(model, points, matrix):
    with torch.no_grad():
        u = model.displacement(points)
        transformed_u = model.displacement(points @ matrix.T)
        expected = u @ matrix.T
    return torch.max(torch.abs(transformed_u-expected)).item()


print("HYPOTHESIS:", HYPOTHESIS)
torch.manual_seed(17)
trap = FourRodTrap()
base = ResidualShapeNetwork(trap, width=24, depth=3, alpha=.18, symmetry="none")
torch.nn.init.normal_(base.network[-1].weight, std=.35)
torch.nn.init.normal_(base.network[-1].bias, std=.20)
modes = ["x_mirror", "y_mirror", "xy_mirror", "d4"]
matrix_x = torch.tensor([[-1., 0.], [0., 1.]])
matrix_y = torch.tensor([[1., 0.], [0., -1.]])
matrix_swap = torch.tensor([[0., 1.], [1., 0.]])
probe = torch.tensor(np.random.default_rng(4).uniform(-4.5, 4.5, (512, 2)), dtype=torch.float32)
grid_axis = torch.linspace(-4.7, 4.7, 45)
gx, gy = torch.meshgrid(grid_axis, grid_axis, indexing="xy")
grid = torch.stack([gx.ravel(), gy.ravel()], dim=1)
grid = grid[torch.tensor(trap.in_vacuum(grid.numpy(), margin=.02))]
boundaries = boundary_points(trap)

fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
results = []
colors = ["tab:red", "tab:red", "tab:blue", "tab:blue"]
for ax, mode in zip(axes.ravel(), modes):
    model = ResidualShapeNetwork(trap, width=24, depth=3, alpha=.18, symmetry=mode)
    model.network.load_state_dict(base.network.state_dict())
    rx = equivariance_error(model, probe, matrix_x)
    ry = equivariance_error(model, probe, matrix_y)
    rs = equivariance_error(model, probe, matrix_swap)
    det_min = jacobian_minimum(model, grid)
    center = torch.zeros(1, 2)
    outer = torch.stack([5*torch.cos(torch.linspace(0, 2*torch.pi, 200)), 5*torch.sin(torch.linspace(0, 2*torch.pi, 200))], dim=1)
    with torch.no_grad():
        center_motion = torch.linalg.vector_norm(model.displacement(center), dim=1).max().item()
        outer_motion = torch.linalg.vector_norm(model.displacement(outer), dim=1).max().item()
    results.append((mode, rx, ry, rs, det_min, center_motion, outer_motion))
    for index, boundary in enumerate(boundaries):
        with torch.no_grad():
            deformed = model(boundary).numpy()
            original = boundary.numpy()
        ax.plot(original[:, 0], original[:, 1], "--", color="grey", linewidth=1)
        ax.plot(np.r_[deformed[:, 0], deformed[0, 0]], np.r_[deformed[:, 1], deformed[0, 1]], color=colors[index], linewidth=2)
        every = np.arange(0, len(original), 12)
        movement = deformed[every]-original[every]
        ax.quiver(original[every, 0], original[every, 1], movement[:, 0], movement[:, 1], angles="xy", scale_units="xy", scale=1, width=.004, color="black")
    ax.add_patch(plt.Circle((0, 0), .4, fill=False, color="magenta", linewidth=1.5, label="exact center anchor"))
    ax.set(xlim=(-3.6, 3.6), ylim=(-3.6, 3.6), aspect="equal", xlabel="x/r0", ylabel="y/r0", title=f"{mode}: min detJ={det_min:.3f}\nerrors x={rx:.1e}, y={ry:.1e}, swap={rs:.1e}")
    ax.grid(alpha=.2)
axes[0, 0].legend(handles=[plt.Line2D([0], [0], color="grey", linestyle="--", label="initial electrode"), plt.Line2D([0], [0], color="tab:red", label="deformed electrode"), plt.Line2D([0], [0], color="magenta", label="fixed center")], fontsize=8)
fig.suptitle("Hypothesis: the shape-symmetry toggle controls physical electrode pairing", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

for row in results:
    print(f"{row[0]:9s} x={row[1]:.3e} y={row[2]:.3e} swap={row[3]:.3e} min_detJ={row[4]:.3f} center={row[5]:.3e} outer={row[6]:.3e}")
required = [results[0][1], results[1][2], max(results[2][1], results[2][2]), max(results[3][1], results[3][2], results[3][3])]
exact_symmetry = max(required) < 2e-7
anchors_exact = max(max(row[5], row[6]) for row in results) < 1e-10
nonfolded = min(row[4] for row in results) > .2
print("GATES exact_selected_symmetry=", exact_symmetry, "anchors_exact=", anchors_exact, "nonfolded=", nonfolded)
print("CONCLUSION: accept the production shape toggle only if selected pairings, hard anchors, and Jacobian safety all pass together.")
print("saved", OUTPUT)
