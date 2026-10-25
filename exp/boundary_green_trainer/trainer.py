import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path


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

def rod_local(shape, n=N_ROD, offset=0.5):
    R, c4, c8 = shape
    phi = 2.0 * np.pi * (np.arange(n) + offset) / n
    p = R + c4 * np.cos(4 * phi) + c8 * np.cos(8 * phi)
    dp = -4 * c4 * np.sin(4 * phi) - 8 * c8 * np.sin(8 * phi)
    nrm = np.column_stack([np.cos(phi), np.sin(phi)])
    tan = np.column_stack([-np.sin(phi), np.cos(phi)])
    return phi, p[:, None] * nrm + dp[:, None] * tan, nrm


def geometry(shape, n=N_ROD, offset=0.5):
    # the nearest support point sits at p(0) = R + c4 + c8, so place the rod there to keep r0 = 1
    R, c4, c8 = shape
    phi, local, nrm = rod_local(shape, n, offset)
    gap = 1.0 + R + c4 + c8
    pts, nrms, vals, rod_id = [], [], [], []
    for k, ((radial, tangential), v) in enumerate(zip(FRAMES, VOLTAGES)):
        rot = np.column_stack([radial, tangential])
        pts.append(gap * radial + local @ rot.T)
        nrms.append(nrm @ rot.T)
        vals.append(np.full(n, v))
        rod_id.append(np.full(n, k))
    th = 2.0 * np.pi * (np.arange(N_ENC) + offset - 0.5) / N_ENC
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


# ---- 2. symmetry-orbit Green-PIG field ----
# log atoms on a shrunken copy of each rod (depth fraction f) plus a ring outside the enclosure.
# One rod's coefficients a are cloned with signs (+,+,-,-); ring blocks rotated by 90 deg flip sign.
# Coefficients come from a QR least-squares fit; f is trained with Adam on held-out boundary points.

N_SRC = 24
ENC_SRC_R = 6.5


def sources(shape, f):
    R, c4, c8 = shape
    _, local, _ = rod_local(shape, N_SRC, 0.25)
    gap = 1.0 + R + c4 + c8
    local = torch.as_tensor(local)
    rods = [torch.as_tensor(gap * radial) + f * local @ torch.as_tensor(np.column_stack([radial, tangential])).T
            for radial, tangential in FRAMES]
    th = 2.0 * np.pi * (np.arange(4 * N_SRC) + 0.3) / (4 * N_SRC)
    ring = torch.as_tensor(ENC_SRC_R * np.column_stack([np.cos(th), np.sin(th)]))
    return torch.cat(rods + [ring])


def tie():
    T = torch.zeros(8 * N_SRC, 2 * N_SRC, dtype=torch.float64)
    eye = torch.eye(N_SRC, dtype=torch.float64)
    for k, s in enumerate([+1, +1, -1, -1]):
        T[k * N_SRC:(k + 1) * N_SRC, :N_SRC] = s * eye
    for k, s in enumerate([+1, -1, +1, -1]):
        T[(4 + k) * N_SRC:(5 + k) * N_SRC, N_SRC:] = s * eye
    return T


def green(points, src):
    d = points[:, None, :] - src[None, :, :]
    r2 = (d**2).sum(-1)
    return 0.5 * torch.log(r2), d / r2[..., None]


def boundary_set(shape, offset=0.5):
    g = geometry(shape, offset=offset)
    pts = torch.as_tensor(np.vstack([g["rods"], g["enc"]]))
    vals = torch.as_tensor(np.concatenate([g["rod_values"], np.zeros(N_ENC)]))
    return g, pts, vals


def fit_field(shape, f):
    _, pts, vals = boundary_set(shape)
    src = sources(shape, f)
    T = tie()
    A = green(pts, src)[0] @ T
    Q, Rm = torch.linalg.qr(A)
    a = torch.linalg.solve_triangular(Rm, (Q.T @ vals)[:, None], upper=True)[:, 0]
    return src, T @ a


def potential(points, src, coef):
    return green(points, src)[0] @ coef


def train_depth(shape, steps=150):
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)  # f = 0.3 + 0.6 sigmoid(theta)
    opt = torch.optim.Adam([theta], lr=0.1)
    _, held, held_v = boundary_set(shape, offset=0.0)
    hist = []
    for _ in range(steps):
        f = 0.3 + 0.6 * torch.sigmoid(theta)
        src, coef = fit_field(shape, f)
        loss = ((potential(held, src, coef) - held_v) ** 2).mean().sqrt()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append((f.item(), loss.item()))
    return 0.3 + 0.6 * torch.sigmoid(theta.detach()), np.array(hist)


def plot_field(shape=(1.145, 0.04, 0.0)):
    f, hist = train_depth(shape)
    with torch.no_grad():
        src, coef = fit_field(shape, f)
        g, held, held_v = boundary_set(shape, offset=0.0)
        res = (potential(held, src, coef) - held_v).numpy()
        xs = np.linspace(-5, 5, 300)
        X, Y = np.meshgrid(xs, xs)
        P = np.column_stack([X.ravel(), Y.ravel()])
        V = potential(torch.as_tensor(P), src, coef).numpy().reshape(X.shape)
    inside = np.hypot(X, Y) > ENCLOSURE
    for k in range(4):
        inside |= Path(g["rods"][g["rod_id"] == k]).contains_points(P).reshape(X.shape)
    V[inside] = np.nan
    rod_res, enc_res = res[:4 * N_ROD], res[4 * N_ROD:]
    print(f"trained f={f.item():.3f}  held-out boundary residual: rods max {np.abs(rod_res).max():.2e}, enclosure max {np.abs(enc_res).max():.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    cs = ax[0].contourf(X, Y, V, 41, cmap="RdBu_r"); fig.colorbar(cs, ax=ax[0])
    ax[0].plot(*src.numpy().T, "k.", ms=2); ax[0].set_aspect("equal"); ax[0].set_xlim(-6.7, 6.7); ax[0].set_ylim(-6.7, 6.7)
    ax[0].set_title(f"Green-PIG potential, shape {shape}, f={f.item():.3f}")
    for k in range(4):
        ax[1].semilogy(g["phi"], np.abs(rod_res[k * N_ROD:(k + 1) * N_ROD]) + 1e-18, lw=1, label=f"rod {k} ({VOLTAGES[k]:+.1f} V)")
    ax[1].semilogy(np.linspace(0, 2 * np.pi, N_ENC, endpoint=False), np.abs(enc_res) + 1e-18, "k", lw=1, label="enclosure (0 V)")
    ax[1].set_xlabel("boundary angle"); ax[1].set_title("|V - V_prescribed| at held-out boundary points"); ax[1].legend(fontsize=8)
    ax[2].plot(hist[:, 0], hist[:, 1], ".-", ms=3); ax[2].set_yscale("log")
    ax[2].set_xlabel("source depth fraction f"); ax[2].set_ylabel("held-out boundary RMSE"); ax[2].set_title("Adam on source depth")
    fig.savefig(f"{OUT}/02_green_pig_field.png", dpi=130)
    print(f"saved {OUT}/02_green_pig_field.png")


if __name__ == "__main__":
    plot_geometry()
    plot_field()
