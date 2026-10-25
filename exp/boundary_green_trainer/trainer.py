import os

import matplotlib.pyplot as plt
import numpy as np


OUT = "exp/boundary_green_trainer/results"
ENCLOSURE = 5.0
N_ROD = 256
N_ENC = 1024
VOLTAGES = [+0.5, +0.5, -0.5, -0.5]
FRAMES = [
    (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
    (np.array([-1.0, 0.0]), np.array([0.0, -1.0])),
    (np.array([0.0, 1.0]), np.array([-1.0, 0.0])),
    (np.array([0.0, -1.0]), np.array([1.0, 0.0])),
]
os.makedirs(OUT, exist_ok=True)


# ---- 1. boundary layer: support-function rod, cloned by the four-rod orbit ----
# local support function p(phi) = R + c4 cos(4 phi) + c8 cos(8 phi), phi measured from the
# rod's radial axis; boundary x = p n + p' t, so the outward normal is exactly n(phi).

def rod_local(shape, n=N_ROD):
    R, c4, c8 = shape
    phi = 2.0 * np.pi * (np.arange(n) + 0.5) / n
    p = R + c4 * np.cos(4 * phi) + c8 * np.cos(8 * phi)
    dp = -4 * c4 * np.sin(4 * phi) - 8 * c8 * np.sin(8 * phi)
    nrm = np.column_stack([np.cos(phi), np.sin(phi)])
    tan = np.column_stack([-np.sin(phi), np.cos(phi)])
    return phi, p[:, None] * nrm + dp[:, None] * tan, nrm


def geometry(shape, n=N_ROD):
    # the nearest support point sits at p(0) = R + c4 + c8, so place the rod there to keep r0 = 1
    R, c4, c8 = shape
    phi, local, nrm = rod_local(shape, n)
    gap = 1.0 + R + c4 + c8
    pts, nrms, vals, rod_id = [], [], [], []
    for k, ((radial, tangential), v) in enumerate(zip(FRAMES, VOLTAGES)):
        rot = np.column_stack([radial, tangential])
        pts.append(gap * radial + local @ rot.T)
        nrms.append(nrm @ rot.T)
        vals.append(np.full(n, v))
        rod_id.append(np.full(n, k))
    th = 2.0 * np.pi * np.arange(N_ENC) / N_ENC
    enc = ENCLOSURE * np.column_stack([np.cos(th), np.sin(th)])
    return dict(
        phi=phi, gap=gap,
        rods=np.vstack(pts), rod_normals=np.vstack(nrms), rod_values=np.concatenate(vals),
        rod_id=np.concatenate(rod_id), enc=enc, enc_normals=enc / ENCLOSURE,
    )


def plot_geometry():
    shapes = {"circular R=1.145": (1.145, 0.0, 0.0), "c4=0.04 perturbed": (1.145, 0.04, 0.0)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    for ax, (name, shape) in zip(axes, shapes.items()):
        g = geometry(shape)
        for k in range(4):
            m = g["rod_id"] == k
            ax.fill(*g["rods"][m].T, color="tab:red" if VOLTAGES[k] > 0 else "tab:blue", alpha=0.5)
            q = g["rods"][m][::16]
            ax.quiver(*q.T, *g["rod_normals"][m][::16].T, scale=12, width=0.003)
        ax.plot(*np.vstack([g["enc"], g["enc"][:1]]).T, "k-", lw=1)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls="--", color="gray"))
        ax.set_aspect("equal"); ax.set_title(f"{name}  (r0 circle dashed)")
        ax.set_xlim(-5.3, 5.3); ax.set_ylim(-5.3, 5.3)
    fig.savefig(f"{OUT}/01_geometry.png", dpi=130)
    print(f"saved {OUT}/01_geometry.png")


if __name__ == "__main__":
    plot_geometry()
