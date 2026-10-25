from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from spinn.config import config_from_dict
from spinn.geometry.export_fem import reference_geometry
from spinn.geometry.four_rod import build_trap
from spinn.models.backends import build_backend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from spinn.physics.fem_reference import solve_laplace_fd


GRID = 161
BATCH = 256
BACKENDS = ["mlp", "pixel", "pig"]
ROOT = Path("spinn/results/fixed")
OUT = Path("scratch/results/02_fixed_spatial_field_comparison")


def load_model(name):
    run = ROOT / f"{name}_seed0"
    result = json.loads((run / "result.json").read_text())
    config = config_from_dict(result["config"])
    backend = build_backend(config)
    model = ComposedField(IdentityShapeNetwork(), backend).float()
    state = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return config, model


def neural_fields(model, points):
    phi_parts = []
    ex_parts = []
    ey_parts = []
    for start in range(0, len(points), BATCH):
        xy = torch.tensor(points[start:start + BATCH], dtype=torch.float32, requires_grad=True)
        phi = model.backend(xy)
        grad = torch.autograd.grad(phi.sum(), xy)[0]
        phi_parts.append(phi.detach().numpy().ravel())
        ex_parts.append((-grad[:, 0]).detach().numpy())
        ey_parts.append((-grad[:, 1]).detach().numpy())
    phi = np.concatenate(phi_parts)
    ex = np.concatenate(ex_parts)
    ey = np.concatenate(ey_parts)
    return phi, ex, ey, np.hypot(ex, ey)


def boundary_error(model, trap):
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    errors = []
    for electrode in trap.electrodes():
        points = np.column_stack([
            electrode.cx + electrode.radius * np.cos(theta),
            electrode.cy + electrode.radius * np.sin(theta),
        ])
        with torch.no_grad():
            predicted = model.backend(torch.tensor(points, dtype=torch.float32)).numpy().ravel()
        errors.extend(np.abs(predicted - trap.electrode_voltage(electrode)))
    outer = np.column_stack([
        trap.outer_radius * np.cos(theta),
        trap.outer_radius * np.sin(theta),
    ])
    with torch.no_grad():
        predicted_outer = model.backend(torch.tensor(outer, dtype=torch.float32)).numpy().ravel()
    errors.extend(np.abs(predicted_outer))
    return float(np.mean(errors)), float(np.max(errors))


def metrics(reference, predicted, mask):
    a = reference[mask]
    b = predicted[mask]
    error = b - a
    rmse = np.sqrt(np.mean(error ** 2))
    rms_reference = np.sqrt(np.mean(a ** 2))
    return float(rmse), float(rmse / rms_reference), float(np.corrcoef(a, b)[0, 1])


def add_geometry(ax, trap):
    for electrode in trap.electrodes():
        ax.add_patch(Circle(
            (electrode.cx, electrode.cy), electrode.radius,
            facecolor="white", edgecolor="black", linewidth=0.5,
        ))
    ax.add_patch(Circle((0, 0), trap.outer_radius, fill=False, edgecolor="black", linewidth=0.5))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_backend(name, trap, extent, valid, reference, predicted):
    labels = [r"$\Phi$", r"$E_x$", r"$E_y$", r"$|E|$"]
    fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    for row, (label, ref, pred) in enumerate(zip(labels, reference, predicted)):
        ref_plot = np.where(valid, ref, np.nan)
        pred_plot = np.where(valid, pred, np.nan)
        diff_plot = np.where(valid, pred - ref, np.nan)

        if row < 3:
            limit = np.nanpercentile(np.abs(np.concatenate([ref_plot.ravel(), pred_plot.ravel()])), 99)
            field_kwargs = {"cmap": "coolwarm", "vmin": -limit, "vmax": limit}
        else:
            limit = np.nanpercentile(np.concatenate([ref_plot.ravel(), pred_plot.ravel()]), 99)
            field_kwargs = {"cmap": "viridis", "vmin": 0, "vmax": limit}
        diff_limit = np.nanpercentile(np.abs(diff_plot), 99)

        images = [
            axes[row, 0].imshow(ref_plot, origin="lower", extent=extent, **field_kwargs),
            axes[row, 1].imshow(pred_plot, origin="lower", extent=extent, **field_kwargs),
            axes[row, 2].imshow(diff_plot, origin="lower", extent=extent, cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit),
        ]
        axes[row, 0].set_ylabel(label, fontsize=13)
        for col in range(3):
            add_geometry(axes[row, col], trap)
            fig.colorbar(images[col], ax=axes[row, col], shrink=0.78)

    axes[0, 0].set_title("finite-difference reference")
    axes[0, 1].set_title(f"{name.upper()} fixed solver")
    axes[0, 2].set_title("model - reference")
    fig.suptitle(f"Fixed circular four-rod trap: full spatial field comparison ({name.upper()})", fontsize=15)
    fig.savefig(OUT / f"{name}_vs_fd_fields.png", dpi=180)
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)

config, first_model = load_model(BACKENDS[0])
trap = build_trap(config)
solved = solve_laplace_fd(reference_geometry(trap), grid=GRID)
xs = solved["xs"]
ys = solved["ys"]
h = solved["h"]
gx, gy = np.meshgrid(xs, ys)
points = np.column_stack([gx.ravel(), gy.ravel()])

phi_fd = solved["field"]
dphi_dy, dphi_dx = np.gradient(phi_fd, h, h)
ex_fd = -dphi_dx
ey_fd = -dphi_dy
emag_fd = np.hypot(ex_fd, ey_fd)

comparison_margin = 2 * h
valid = trap.in_vacuum(points, margin=comparison_margin).reshape(GRID, GRID)
extent = [xs[0], xs[-1], ys[0], ys[-1]]
reference = [phi_fd, ex_fd, ey_fd, emag_fd]

rows = []
for name in BACKENDS:
    print(f"evaluating {name.upper()} on {GRID}x{GRID} grid")
    config, model = load_model(name)
    phi, ex, ey, emag = neural_fields(model, points)
    predicted = [field.reshape(GRID, GRID) for field in [phi, ex, ey, emag]]
    mean_bc, max_bc = boundary_error(model, trap)

    for quantity, ref, pred in zip(["phi", "ex", "ey", "emag"], reference, predicted):
        rmse, nrmse, correlation = metrics(ref, pred, valid)
        rows.append({
            "backend": name,
            "quantity": quantity,
            "rmse": rmse,
            "nrmse": nrmse,
            "correlation": correlation,
            "mean_boundary_abs_error": mean_bc,
            "max_boundary_abs_error": max_bc,
        })
        print(f"{name:5s} {quantity:4s}: NRMSE={nrmse:.3f} correlation={correlation:.3f}")
    print(f"{name:5s} boundary: mean |error|={mean_bc:.3e}, max |error|={max_bc:.3e}")
    plot_backend(name, trap, extent, valid, reference, predicted)

with open(OUT / "spatial_metrics.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("picked: shared finite-difference reference matching your circular Dirichlet geometry")
print(f"comparison mask: vacuum points at least {comparison_margin:.4f} from every boundary")
print(f"saved plots and metrics to {OUT}")
