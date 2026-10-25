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


def fit_field(shape, f, vals=None):
    _, pts, bc = boundary_set(shape)
    vals = bc if vals is None else vals
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


# ---- 3. multipole readout ----
# V(r, theta) = sum_n A_n (r / r0)^n cos(n theta); read A_n on the 0.9 r0 circle, report q_n = A_n / A_2.
# Torch end to end, so gradients flow through the fitted coefficients; shape derivatives come from step 4.

R_READ = 0.9
ORDERS = [2, 6, 10, 14]
N_READ = 720
_th = torch.arange(N_READ, dtype=torch.float64) * 2 * np.pi / N_READ
READ_PTS = R_READ * torch.stack([torch.cos(_th), torch.sin(_th)], 1)


def raw_multipoles(src, coef, orders=ORDERS):
    v = potential(READ_PTS, src, coef)
    return torch.stack([2 * (v * torch.cos(n * _th)).mean() / R_READ**n for n in orders])


def multipoles(src, coef, orders=ORDERS):
    A = raw_multipoles(src, coef, orders)
    return A / A[0]


def rod_residual(shape, src, coef):
    _, held, held_v = boundary_set(shape, offset=0.0)
    return (potential(held, src, coef) - held_v)[:4 * N_ROD].abs().max().item()


def plot_multipoles(radii=(1.0, 1.145, 1.3)):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    width = 0.25
    for i, R in enumerate(radii):
        shape = (R, 0.0, 0.0)
        f, _ = train_depth(shape, steps=60)
        with torch.no_grad():
            src, coef = fit_field(shape, f)
            q = multipoles(src, coef).numpy()
        print(f"R={R}: q6={q[1]:+.3e} q10={q[2]:+.3e} q14={q[3]:+.3e}  rod BC max err {rod_residual(shape, src, coef):.1e}")
        x = np.arange(1, len(ORDERS)) + (i - 1) * width
        ax.bar(x, np.abs(q[1:]), width, label=f"R={R}", color=["tab:blue", "tab:green", "tab:red"][i])
        for xi, qi in zip(x, q[1:]):
            ax.text(xi, abs(qi) * 1.3, "+" if qi > 0 else "-", ha="center")
    ax.set_yscale("log"); ax.set_xticks(range(1, len(ORDERS)), [f"|q{n}|" for n in ORDERS[1:]])
    ax.set_title("circular rods: multipoles relative to quadrupole (sign above bar)"); ax.legend()
    fig.savefig(f"{OUT}/03_multipoles.png", dpi=130)
    print(f"saved {OUT}/03_multipoles.png")


# ---- 4. Hadamard shape gradient ----
# Normal electrode velocity for shape mode s: h = n . radial * d(gap)/ds + dp/ds, with d(gap)/ds = 1 and
# dp/ds = 1, cos 4phi, cos 8phi. Sensitivity V' is harmonic with V' = -h dV/dn on rods, 0 on the
# enclosure, solved in the same Green basis. dq_n follows from the quotient rule on A_n / A_2.

F_DEPTH = 0.73
MODES = ["R", "c4", "c8"]


def shape_gradient(shape, f=F_DEPTH):
    g, pts, _ = boundary_set(shape)
    src, coef = fit_field(shape, f)
    grad = green(pts[:4 * N_ROD], src)[1]
    dVdn = (torch.einsum("nka,k->na", grad, coef) * torch.as_tensor(g["rod_normals"])).sum(1)
    radial = np.array([FRAMES[k][0] for k in g["rod_id"]])
    n_dot_radial = torch.as_tensor((g["rod_normals"] * radial).sum(1))
    phi = torch.as_tensor(np.tile(g["phi"], 4))
    A = raw_multipoles(src, coef)
    dq = []
    for dp in [torch.ones_like(phi), torch.cos(4 * phi), torch.cos(8 * phi)]:
        h = n_dot_radial + dp
        vals = torch.cat([-h * dVdn, torch.zeros(N_ENC, dtype=torch.float64)])
        dA = raw_multipoles(*fit_field(shape, f, vals))
        dq.append((dA * A[0] - A * dA[0]) / A[0] ** 2)
    return torch.stack(dq, 1).numpy()  # rows: orders, cols: modes


def fd_gradient(shape, f=F_DEPTH, eps=1e-3):
    cols = []
    for j in range(3):
        up, dn = list(shape), list(shape)
        up[j] += eps; dn[j] -= eps
        cols.append((multipoles(*fit_field(tuple(up), f)) - multipoles(*fit_field(tuple(dn), f))).numpy() / (2 * eps))
    return np.stack(cols, 1)


def plot_shape_gradient():
    Rs = np.linspace(1.0, 1.3, 7)
    with torch.no_grad():
        had = np.array([shape_gradient((R, 0.02, 0.0)) for R in Rs])
        fd = np.array([fd_gradient((R, 0.02, 0.0)) for R in Rs])
        src, coef = fit_field((1.145, 0.02, 0.0), F_DEPTH)
        print(f"rod BC max err at (1.145, 0.02, 0): {rod_residual((1.145, 0.02, 0.0), src, coef):.1e}")
    rel = np.abs(had[:, 1:3] - fd[:, 1:3]).max() / np.abs(fd[:, 1:3]).max()
    print(f"max |Hadamard - FD| / max |FD| over q6, q10: {rel:.1e}")
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for i, order_idx in enumerate([1, 2]):
        for j, mode in enumerate(MODES):
            ax[i, j].plot(Rs, had[:, order_idx, j], "-", label="Hadamard")
            ax[i, j].plot(Rs, fd[:, order_idx, j], "o", mfc="none", label="central FD re-solve")
            ax[i, j].set_title(f"d q{ORDERS[order_idx]} / d {mode}   (c4=0.02)")
            ax[i, j].set_xlabel("R"); ax[i, j].axhline(0, color="gray", lw=0.5)
    ax[0, 0].legend()
    fig.savefig(f"{OUT}/04_hadamard_vs_fd.png", dpi=130)
    print(f"saved {OUT}/04_hadamard_vs_fd.png")


# ---- 5. outer loop on R alone: minimize q6^2 with the Hadamard gradient ----

def optimize_R(R0, lr=60.0, steps=30):
    R, hist = R0, []
    with torch.no_grad():
        for _ in range(steps):
            shape = (R, 0.0, 0.0)
            q6 = multipoles(*fit_field(shape, F_DEPTH))[1].item()
            dq6 = shape_gradient(shape)[1, 0]
            hist.append((R, q6))
            R -= lr * 2 * q6 * dq6
    return np.array(hist)


def plot_optimize_R(starts=(0.9, 1.3, 1.45)):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for R0 in starts:
        h = optimize_R(R0)
        with torch.no_grad():
            src, coef = fit_field((h[-1, 0], 0.0, 0.0), F_DEPTH)
            print(f"start R={R0}: final R={h[-1, 0]:.5f}, q6={h[-1, 1]:+.1e}, rod BC max err {rod_residual((h[-1, 0], 0.0, 0.0), src, coef):.1e}")
        ax[0].plot(h[:, 0], ".-", label=f"start {R0}")
        ax[1].semilogy(np.abs(h[:, 1]) + 1e-16, ".-", label=f"start {R0}")
    ax[0].axhline(1.145168, color="k", ls="--", label="known optimum 1.145168")
    ax[0].set_xlabel("outer step"); ax[0].set_ylabel("R / r0"); ax[0].legend(); ax[0].set_title("gradient descent on R, J = q6^2")
    ax[1].set_xlabel("outer step"); ax[1].set_ylabel("|q6|"); ax[1].set_title("q6 during descent")
    fig.savefig(f"{OUT}/05_optimize_R.png", dpi=130)
    print(f"saved {OUT}/05_optimize_R.png")


# ---- 6. outer loop on (R, c4): cancel q6 and q10 with a hard clearance floor ----
# damped Gauss-Newton on r = (q6, q10) with the Hadamard Jacobian; any step that breaks the clearance
# floor to the enclosure or rod convexity (p + p'' > 0) is halved until feasible.

CLEARANCE = 0.3


def feasible(shape):
    R, c4, c8 = shape
    g = geometry(shape)
    return ENCLOSURE - np.linalg.norm(g["rods"], axis=1).max() >= CLEARANCE and R - 15 * abs(c4) - 63 * abs(c8) > 0


def optimize_R_c4(x0, steps=20, damping=1e-8):
    x, hist = np.array(x0, float), []
    with torch.no_grad():
        for _ in range(steps):
            shape = (x[0], x[1], 0.0)
            q = multipoles(*fit_field(shape, F_DEPTH)).numpy()
            J = shape_gradient(shape)[1:3, :2]
            hist.append([*x, q[1], q[2]])
            dx = -np.linalg.solve(J.T @ J + damping * np.eye(2), J.T @ q[1:3])
            while not feasible((x[0] + dx[0], x[1] + dx[1], 0.0)):
                dx /= 2
            x = x + dx
    return np.array(hist)


def plot_optimize_R_c4(starts=((1.145, 0.0), (1.3, 0.03), (1.0, -0.02))):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for i, x0 in enumerate(starts):
        h = optimize_R_c4(x0)
        shape = (h[-1, 0], h[-1, 1], 0.0)
        with torch.no_grad():
            src, coef = fit_field(shape, F_DEPTH)
            q = multipoles(src, coef).numpy()
            err = rod_residual(shape, src, coef)
        g = geometry(shape)
        clear = ENCLOSURE - np.linalg.norm(g["rods"], axis=1).max()
        print(f"start {x0}: R={shape[0]:.5f} c4={shape[1]:+.5f} q6={q[1]:+.1e} q10={q[2]:+.1e} q14={q[3]:+.1e} clearance={clear:.3f} rod BC max err {err:.1e}")
        c = f"C{i}"
        ax[0].plot(h[:, 0], h[:, 1], ".-", color=c, label=f"start {x0}")
        ax[0].plot(*h[-1, :2], "*", color=c, ms=14)
        ax[2].semilogy(np.abs(h[:, 2]), "-", color=c, label=f"|q6| start {x0}")
        ax[2].semilogy(np.abs(h[:, 3]), "--", color=c, label=f"|q10| start {x0}")
        if i == 0:
            circ = geometry((1.145168, 0.0, 0.0))
            for k in range(4):
                m = g["rod_id"] == k
                ax[1].fill(*g["rods"][m].T, color="tab:red" if VOLTAGES[k] > 0 else "tab:blue", alpha=0.5)
                ax[1].plot(*circ["rods"][circ["rod_id"] == k].T, "k--", lw=0.8)
            ax[1].plot(*np.vstack([g["enc"], g["enc"][:1]]).T, "k-", lw=1)
            ax[1].add_patch(plt.Circle((0, 0), 1.0, fill=False, ls=":", color="gray"))
            ax[1].set_aspect("equal"); ax[1].set_title(f"dual-zero electrode R={shape[0]:.4f}, c4={shape[1]:+.4f} (dashed: circular q6=0)")
    ax[0].set_xlabel("R"); ax[0].set_ylabel("c4"); ax[0].set_title("shape trajectories (Gauss-Newton, Hadamard Jacobian)"); ax[0].legend()
    ax[2].set_xlabel("outer step"); ax[2].set_title("multipole history"); ax[2].legend(fontsize=7)
    fig.savefig(f"{OUT}/06_optimize_R_c4.png", dpi=130)
    print(f"saved {OUT}/06_optimize_R_c4.png")


if __name__ == "__main__":
    plot_geometry()
    plot_field()
    plot_multipoles()
    plot_shape_gradient()
    plot_optimize_R()
    plot_optimize_R_c4()
