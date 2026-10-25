import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.config import Config
from spinn.geometry.four_rod import build_trap
from spinn.geometry.masks import design_mask


N = 4001
OUTPUT = "scratch/04_mask_derivatives.png"


trap = build_trap(Config())
x = torch.linspace(-trap.outer_radius, trap.outer_radius, N, dtype=torch.float64, requires_grad=True)
points = torch.stack([x, torch.zeros_like(x)], dim=1)

m = design_mask(points, trap).squeeze(1)
dm = torch.autograd.grad(m, x, torch.ones_like(m), create_graph=True)[0]
d2m = torch.autograd.grad(dm, x, torch.ones_like(dm))[0]

xn = x.detach().numpy()
mn = m.detach().numpy()
dmn = dm.detach().numpy()
d2mn = d2m.detach().numpy()

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
axes[0].plot(xn, mn, color="tab:blue", linewidth=2)
axes[1].plot(xn, dmn, color="tab:orange", linewidth=1.5)
axes[2].plot(xn, d2mn, color="tab:green", linewidth=1.5)

axes[0].set_ylabel(r"$m(x,0)$")
axes[1].set_ylabel(r"$\partial m/\partial x$")
axes[2].set_ylabel(r"$\partial^2m/\partial x^2$")
axes[2].set_xlabel(r"$x/r_0$ along $y=0$")

for ax in axes:
    ax.axvspan(-0.25, 0.25, color="magenta", alpha=0.10)
    for value in [-0.40, 0.40]:
        ax.axvline(value, color="magenta", linestyle="--", linewidth=1)
    for value in [-0.85, 0.85]:
        ax.axvline(value, color="magenta", linestyle=":", linewidth=1)
    for electrode in trap.electrodes():
        if abs(electrode.cy) < 1e-12:
            ax.axvline(electrode.cx - electrode.radius, color="black", linewidth=0.8)
            ax.axvline(electrode.cx + electrode.radius, color="black", linewidth=0.8)
            ax.axvline(electrode.cx, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(-4.5, color="purple", linestyle=":", linewidth=1)
    ax.axvline(4.5, color="purple", linestyle=":", linewidth=1)
    ax.grid(alpha=0.15)

axes[0].set_title("Production deformation mask and its autograd derivatives")
axes[0].text(0.01, 0.90, "magenta shading: eta measurement disc", transform=axes[0].transAxes)
axes[1].text(0.01, 0.88, "black: electrode surfaces; gray: rod centers", transform=axes[1].transAxes)
axes[2].text(0.01, 0.88, "purple dotted: start of outer anchor", transform=axes[2].transAxes)
fig.savefig(OUTPUT, dpi=180)

center = np.abs(xn) <= 0.40
measurement = np.abs(xn) <= 0.25
outer = np.abs(xn) >= trap.outer_radius - 1e-12

print("production mask C2 diagnostic")
print("finite m, dm, d2m:", np.isfinite(mn).all(), np.isfinite(dmn).all(), np.isfinite(d2mn).all())
print("measurement disc max |m|, |dm|, |d2m|:", np.max(np.abs(mn[measurement])), np.max(np.abs(dmn[measurement])), np.max(np.abs(d2mn[measurement])))
print("center anchor max |m|, |dm|, |d2m|:", np.max(np.abs(mn[center])), np.max(np.abs(dmn[center])), np.max(np.abs(d2mn[center])))
print("outer boundary m values:", mn[outer])
print("global max |dm|, |d2m|:", np.max(np.abs(dmn)), np.max(np.abs(d2mn)))
print("saved", OUTPUT)
