"""Hard-constraint PINN + automated hyperparameter search for the four-rod Paul trap.

Problem definition (mirrors ``spinn/geometry/four_rod.py`` / SPINN_Experiment_Plan.pdf)
---------------------------------------------------------------------------------------
* Governing equation:  Laplace,  lap V = 0  in the vacuum region.
* Domain:              disk of radius R_out = 5 r0 minus four rods of radius
                       R_rod = 1.145 r0 centered at (+/-(r0+R_rod), 0), (0, +/-(r0+R_rod)).
* Boundary conditions: Dirichlet  V = +V0/2 on the x-axis rods, V = -V0/2 on the
                       y-axis rods, V = 0 on the outer circle.
* Symmetry:            D4 quadrupole symmetry with the sign character:
                           V(-x, y) = V(x, -y) = V(x, y)      (mirror-even)
                           V( y, x) = -V(x, y)                (swap-odd)

Hard constraints (NO soft penalty terms)
----------------------------------------
Both the Dirichlet data and the symmetry are built into the forward pass, so the
training loss is the *pure* PDE residual -- there is no `loss_weights.bc`, no
BC/PDE competition, and no symmetry penalty.

1. Exact Dirichlet BCs via the Lagaris / transfinite-interpolation ansatz:

       V(x) = G(x) + D_hat(x) * N_sym(x)

   * phi_i(x) are smooth "boundary functions" that vanish exactly on boundary
     component i and are positive inside the vacuum.  For circles we use the
     POLYNOMIAL normalized form  phi = (|x-c|^2 - R^2) / (2R)  instead of the
     true distance  |x-c| - R:  the polynomial is C-infinity everywhere (the
     true outer distance  R_out - |x|  has a singular Laplacian ~ 1/|x| at the
     origin -- exactly where eta is measured!) and agrees with the distance to
     first order at the boundary.
   * G(x) = sum_i V_i * w_i(x)  with leave-one-out product weights
     w_i = prod_{j != i} phi_j / sum_k prod_{j != k} phi_j.  On boundary i every
     product containing phi_i vanishes, so w_i = 1 and w_{j != i} = 0: G
     interpolates the electrode voltages EXACTLY.  The weights are a convex
     combination, so |G| <= V0/2 everywhere (no far-field blow-up).
   * D(x) = prod_i phi_i vanishes on every boundary; we use the bounded form
     D_hat = D / (1 + D)  which still vanishes exactly on the boundary, is
     smooth, and stays O(1) across the whole domain (raw D varies by ~5 orders
     of magnitude between the trap center and the far field).
   Because the boundaries are disjoint circles, every denominator is strictly
   positive on the closed domain, so the ansatz is smooth wherever the PDE
   residual is sampled.

2. Exact D4 symmetry.  G and D_hat are already exactly symmetric by
   construction (the four rods are processed by identical code paths and the
   swap x<->y exchanges +V rods with -V rods, flipping G's sign).  The network
   factor N_sym is symmetrized architecturally, with two interchangeable
   mechanisms exposed to the hyperparameter search:
   * "reynolds"  -- group-averaging (Reynolds operator) over the 8 elements of
     D4 with the sign character:  N_sym(x) = 1/8 sum_g chi(g) N(g^-1 x).
     Exact for ANY inner network; costs 8 forward evaluations (batched as one).
   * "prefactor" -- invariant-feature factorization:
     N_sym(x) = (x^2 - y^2) * F(r^2, x^2 y^2, (x^2-y^2)^2).  The prefactor
     carries the swap-odd sign; the features are complete D4 invariants, so the
     ansatz can only represent the allowed harmonics m = 2, 6, 10, ...
     One forward evaluation; a strong inductive bias for quadrupole fields.

Composability with the CPN shape optimizer: the ansatz lives in REFERENCE
coordinates z, where the rod circles are analytic.  For joint shape
optimization use  V(z) = G(z) + D_hat(z) * N_sym(NN_phi(z)):  D_hat(z) = 0 on
the reference boundary, whose image under NN_phi *is* the deformed electrode
surface, so the Dirichlet data remains exact on the moving geometry.

Auto-hyperparameterization
--------------------------
An Optuna TPE study (MedianPruner for early stopping of bad trials) searches:
symmetrization mode, depth, width, activation (smooth only -- the residual
needs 2 derivatives), Fourier-feature count and bandwidth, learning rate,
optimizer type, weight decay, cosine schedule on/off, collocation batch size,
and the fraction of collocation points boosted near the trap center (where eta
is measured).  The objective is the mean squared PDE residual on a FIXED
held-out collocation set (same points for every trial).  If Optuna is not
installed the script falls back to pure random search over the same space
(``pip install optuna`` to enable TPE + pruning).

Usage (from SPINN/):
    python -m spinn.scripts.tune_hard_bc --selftest          # exactness checks
    python -m spinn.scripts.tune_hard_bc --smoke             # 2 tiny trials
    python -m spinn.scripts.tune_hard_bc --trials 20         # real study
    pip install optuna                                       # enable TPE/pruning
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:  # Optuna is optional: the script degrades to random search without it.
    import optuna

    HAVE_OPTUNA = True
except ImportError:
    HAVE_OPTUNA = False

# ---------------------------------------------------------------------------
# Geometry constants -- keep in lockstep with spinn/geometry/four_rod.py.
# ---------------------------------------------------------------------------
R0 = 1.0  # characteristic trap radius (dimensionless units)
V0 = 1.0  # peak-to-peak RF voltage scale; rails are driven at +/- V0/2
ROD_RADIUS = 1.145 * R0  # circular-rod approximation to hyperbolic electrodes
ROD_CENTER = R0 + ROD_RADIUS  # rod centers sit at distance r0 + R_rod
OUTER_RADIUS = 5.0 * R0  # grounded outer disk

# Boundary components: (cx, cy, radius, voltage).  Order fixed; the code below
# treats all four rods identically so G/D inherit the exact D4 symmetry.
BOUNDARIES = [
    (+ROD_CENTER, 0.0, ROD_RADIUS, +0.5 * V0),  # East  (+V0/2)
    (-ROD_CENTER, 0.0, ROD_RADIUS, +0.5 * V0),  # West  (+V0/2)
    (0.0, +ROD_CENTER, ROD_RADIUS, -0.5 * V0),  # North (-V0/2)
    (0.0, -ROD_CENTER, ROD_RADIUS, -0.5 * V0),  # South (-V0/2)
]


# ---------------------------------------------------------------------------
# 1. Hard-constraint ansatz: boundary functions, G, D_hat.
# ---------------------------------------------------------------------------
class HardConstraintPotential(nn.Module):
    """V(x) = G(x) + D_hat(x) * core(x): exact Dirichlet BCs by construction.

    ``core`` is any scalar network; if it is D4-symmetrized (see below) the
    total potential carries the exact quadrupole symmetry as well, because G
    and D_hat are symmetric by construction.
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core
        # Rod parameters as buffers so .to(device)/.double() move them along.
        # Built in float64 so a later cast never loses more precision than the
        # target dtype itself (float32 construction would bake ~1e-7 error into
        # the rod centers and break float64 boundary exactness).
        f64 = torch.float64
        cx = torch.tensor([b[0] for b in BOUNDARIES], dtype=f64)
        cy = torch.tensor([b[1] for b in BOUNDARIES], dtype=f64)
        rr = torch.tensor([b[2] for b in BOUNDARIES], dtype=f64)
        vv = torch.tensor([b[3] for b in BOUNDARIES], dtype=f64)
        self.register_buffer("rod_cx", cx)
        self.register_buffer("rod_cy", cy)
        self.register_buffer("rod_r", rr)
        # Boundary values: 4 rods + outer circle (0 V).
        self.register_buffer("bvals", torch.cat([vv, torch.zeros(1, dtype=f64)]))
        # Each phi is rescaled so phi(origin) = 1: keeps D_hat ~ O(1) at the
        # trap center, where eta is measured, without affecting exactness
        # (any positive rescaling preserves phi = 0 on its boundary).
        rod_phi0 = (ROD_CENTER**2 - ROD_RADIUS**2) / (2.0 * ROD_RADIUS)
        out_phi0 = OUTER_RADIUS / 2.0
        self.register_buffer(
            "phi0", torch.tensor([rod_phi0] * 4 + [out_phi0], dtype=f64)
        )

    def phis(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized polynomial boundary functions, shape (N, 5).

        phi_rod  = (|x - c|^2 - R^2) / (2 R phi0)   -- zero on the rod surface
        phi_out  = (R_out^2 - |x|^2) / (2 R_out phi0) -- zero on the outer circle

        Polynomials in (x, y): C-infinity everywhere, so the autograd Laplacian
        of the full ansatz is well defined at every collocation point
        (including the origin, unlike R_out - |x|).
        """
        px, py = x[:, 0:1], x[:, 1:2]
        d2 = (px - self.rod_cx) ** 2 + (py - self.rod_cy) ** 2  # (N, 4)
        phi_rod = (d2 - self.rod_r**2) / (2.0 * self.rod_r)
        phi_out = (OUTER_RADIUS**2 - (px**2 + py**2)) / (2.0 * OUTER_RADIUS)
        return torch.cat([phi_rod, phi_out], dim=1) / self.phi0

    def G_and_Dhat(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transfinite boundary interpolant G and bounded vanishing factor D_hat."""
        phi = self.phis(x)  # (N, 5), all > 0 in the interior
        # Leave-one-out products prod_{j != i} phi_j, computed by explicit
        # masking (25 multiplies) rather than total_prod / phi_i, which would
        # be 0/0 on the boundary itself.
        n_b = phi.shape[1]
        loo = []
        for i in range(n_b):
            others = [phi[:, j : j + 1] for j in range(n_b) if j != i]
            prod = others[0]
            for o in others[1:]:
                prod = prod * o
            loo.append(prod)
        loo = torch.cat(loo, dim=1)  # (N, 5)
        # Convex weights: on boundary i, loo_i > 0 and loo_{j!=i} = 0 -> w_i = 1.
        w = loo / loo.sum(dim=1, keepdim=True)
        G = (w * self.bvals).sum(dim=1, keepdim=True)
        # D = prod_i phi_i vanishes on every boundary; D/(1+D) keeps it exact
        # (0 stays 0), smooth, and bounded in [0, 1) across the whole domain.
        D = loo[:, 0:1] * phi[:, 0:1]  # loo_0 * phi_0 = full product
        Dhat = D / (1.0 + D)
        return G, Dhat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G, Dhat = self.G_and_Dhat(x)
        return G + Dhat * self.core(x)


# ---------------------------------------------------------------------------
# 2. D4-symmetrized cores (both give the exact character symmetry).
# ---------------------------------------------------------------------------
class FourierFeatures(nn.Module):
    """Random Fourier features x -> [sin(Bx), cos(Bx)] to fight spectral bias.

    B is a FIXED Gaussian matrix (buffer, not a parameter) with bandwidth
    ``sigma``; both sigma and the feature count are tuned by the search.
    """

    def __init__(self, in_dim: int, n_features: int, sigma: float) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_features) * sigma)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


class _Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # smooth, SIREN-style
        return torch.sin(x)


def make_activation(name: str) -> nn.Module:
    # Only C-infinity activations: the PDE residual needs two derivatives, so
    # ReLU-family activations (piecewise-linear, zero Laplacian) are excluded.
    return {"tanh": nn.Tanh(), "sin": _Sine(), "gelu": nn.GELU()}[name]


def make_mlp(in_dim: int, width: int, depth: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, width), make_activation(activation)]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), make_activation(activation)]
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class ReynoldsD4(nn.Module):
    """Group-averaged core: N_sym(x) = 1/8 sum_g chi(g) N(g x).

    The 8 elements of D4 with the quadrupole sign character chi:
    +1 for identity, both axis mirrors, and the 180-degree rotation;
    -1 for both diagonal mirrors and the +/-90-degree rotations (they exchange
    the +V and -V rod pairs).  Exact symmetry for ANY inner network; the 8
    evaluations are batched into a single forward pass.
    """

    # (sign_x_from, take_y_first, sign_y_from, character) encoded explicitly:
    # each row builds input (a*u, b*v) where (u, v) is (x, y) or (y, x).
    TRANSFORMS = [
        # (swap?, sx, sy, chi)
        (False, +1.0, +1.0, +1.0),  # identity
        (False, -1.0, +1.0, +1.0),  # mirror across y-axis
        (False, +1.0, -1.0, +1.0),  # mirror across x-axis
        (False, -1.0, -1.0, +1.0),  # 180-degree rotation
        (True, +1.0, +1.0, -1.0),  # diagonal mirror (swap x<->y)
        (True, -1.0, +1.0, -1.0),  # 90-degree rotation
        (True, +1.0, -1.0, -1.0),  # 270-degree rotation
        (True, -1.0, -1.0, -1.0),  # anti-diagonal mirror
    ]

    def __init__(self, width: int, depth: int, activation: str, ff: FourierFeatures | None) -> None:
        super().__init__()
        self.ff = ff
        in_dim = ff.out_dim if ff is not None else 2
        self.net = make_mlp(in_dim, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px, py = x[:, 0:1], x[:, 1:2]
        inputs, chis = [], []
        for swap, sx, sy, chi in self.TRANSFORMS:
            u, v = (py, px) if swap else (px, py)
            inputs.append(torch.cat([sx * u / OUTER_RADIUS, sy * v / OUTER_RADIUS], dim=1))
            chis.append(chi)
        stacked = torch.cat(inputs, dim=0)  # (8N, 2), one batched forward
        if self.ff is not None:
            stacked = self.ff(stacked)
        out = self.net(stacked).reshape(8, -1, 1)
        chi_t = torch.tensor(chis, dtype=out.dtype, device=out.device).view(8, 1, 1)
        return (chi_t * out).mean(dim=0)


class PrefactorD4(nn.Module):
    """Invariant-feature core: N_sym(x) = (x^2 - y^2)/r0^2 * F(invariants).

    (x^2 - y^2) is the m = 2 harmonic: even under both axis mirrors, odd under
    the swap -- exactly the required character.  The features fed to F are
    complete D4 invariants (r^2, x^2 y^2, (x^2-y^2)^2; the third is dependent
    but harmless and helps conditioning), so F cannot break the symmetry.
    Cheaper than Reynolds (1 network evaluation vs 8) and restricts the ansatz
    to the physically allowed harmonics m = 2, 6, 10, ...
    """

    def __init__(self, width: int, depth: int, activation: str, ff: FourierFeatures | None) -> None:
        super().__init__()
        self.ff = ff
        in_dim = ff.out_dim if ff is not None else 3
        self.net = make_mlp(in_dim, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px, py = x[:, 0:1], x[:, 1:2]
        r2 = px**2 + py**2
        pref = (px**2 - py**2) / R0**2  # swap-odd prefactor
        # Invariants normalized to O(1) over the domain for conditioning.
        feats = torch.cat(
            [
                r2 / OUTER_RADIUS**2,
                (px**2 * py**2) / OUTER_RADIUS**4,
                (px**2 - py**2) ** 2 / OUTER_RADIUS**4,
            ],
            dim=1,
        )
        if self.ff is not None:
            feats = self.ff(feats)
        return pref * self.net(feats)


def build_model(cfg: dict, dtype: torch.dtype, device: str) -> HardConstraintPotential:
    """Assemble ansatz from a hyperparameter dict (shared by Optuna + fallback)."""
    ff = None
    if cfg["n_fourier"] > 0:
        in_dim = 2 if cfg["sym_mode"] == "reynolds" else 3
        ff = FourierFeatures(in_dim, cfg["n_fourier"], cfg["ff_sigma"])
    core_cls = ReynoldsD4 if cfg["sym_mode"] == "reynolds" else PrefactorD4
    core = core_cls(cfg["width"], cfg["depth"], cfg["activation"], ff)
    return HardConstraintPotential(core).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# 3. Physics: autograd Laplacian residual (the ONLY training loss).
# ---------------------------------------------------------------------------
def laplacian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """lap V at each collocation point via nested autograd (batch-independent)."""
    x = x.requires_grad_(True)
    v = model(x)
    (g,) = torch.autograd.grad(v.sum(), x, create_graph=True)
    (gxx,) = torch.autograd.grad(g[:, 0].sum(), x, create_graph=True)
    (gyy,) = torch.autograd.grad(g[:, 1].sum(), x, create_graph=True)
    return gxx[:, 0:1] + gyy[:, 1:2]


# ---------------------------------------------------------------------------
# 4. Collocation sampling (mirrors four_rod.py rejection sampling).
# ---------------------------------------------------------------------------
def in_vacuum(pts: np.ndarray, margin: float = 1e-3) -> np.ndarray:
    r = np.hypot(pts[:, 0], pts[:, 1])
    ok = r < OUTER_RADIUS - margin
    for cx, cy, rad, _ in BOUNDARIES:
        ok &= np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > rad + margin
    return ok


def sample_interior(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((0, 2))
    while len(out) < n:
        batch = rng.uniform(-OUTER_RADIUS, OUTER_RADIUS, size=(max(2 * n, 1024), 2))
        out = np.vstack([out, batch[in_vacuum(batch)]])
    return out[:n]


def sample_center_disk(n: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform points in the rod-free disk around the origin (eta region)."""
    r = radius * np.sqrt(rng.uniform(size=n))
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def sample_boundary(n_per: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Points on every boundary circle with their prescribed voltages (for tests)."""
    pts, vals = [], []
    for cx, cy, rad, volt in BOUNDARIES + [(0.0, 0.0, OUTER_RADIUS, 0.0)]:
        t = rng.uniform(0.0, 2.0 * np.pi, n_per)
        pts.append(np.column_stack([cx + rad * np.cos(t), cy + rad * np.sin(t)]))
        vals.append(np.full((n_per, 1), volt))
    return np.vstack(pts), np.vstack(vals)


# ---------------------------------------------------------------------------
# 5. eta estimator (same convention as spinn/physics/eta.py: eta = 2 a r0^2/V0
#    from the quadratic fit V ~ a(x^2-y^2) + b xy + cx + dy + e near center).
# ---------------------------------------------------------------------------
def eta_fit(points: np.ndarray, potential: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack([x**2 - y**2, x * y, x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, potential.reshape(-1), rcond=None)
    return float(2.0 * coef[0] * R0**2 / V0)


# ---------------------------------------------------------------------------
# 6. One training run (used by every trial).
# ---------------------------------------------------------------------------
def train_one(cfg: dict, trial, args, val_x: torch.Tensor) -> tuple[float, nn.Module]:
    """Train one hard-constraint model; returns (best validation residual MSE, model).

    ``trial`` supports report()/should_prune() -- a real Optuna trial or the
    random-search stand-in.  All trials share the same torch seed so parameter
    initialization noise does not confound the hyperparameter comparison.
    """
    device, dtype = args.device, torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_model(cfg, dtype, device)

    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        if cfg["scheduler"] == "cosine"
        else None
    )

    # Collocation pool, refreshed periodically (matches sampling.resample_every).
    pool = torch.as_tensor(sample_interior(args.pool, rng), dtype=dtype, device=device)
    n_center = int(cfg["center_frac"] * cfg["batch"])  # eta-region boost
    n_bulk = cfg["batch"] - n_center

    eval_every = max(args.steps // 10, 1)
    best_val = float("inf")
    for step in range(args.steps):
        if step > 0 and step % args.resample_every == 0:
            pool = torch.as_tensor(sample_interior(args.pool, rng), dtype=dtype, device=device)
        idx = torch.randint(0, pool.shape[0], (n_bulk,), device=device)
        batch = pool[idx]
        if n_center > 0:
            # Extra collocation density where eta is measured (disk r <= 0.5 is
            # entirely vacuum: the rod surfaces are at distance r0 = 1).
            center = sample_center_disk(n_center, 0.5, rng)
            batch = torch.cat(
                [batch, torch.as_tensor(center, dtype=dtype, device=device)], dim=0
            )

        res = laplacian(model, batch)
        loss = (res**2).mean()  # PURE physics loss: BCs and symmetry are exact
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        if (step + 1) % eval_every == 0 or step == args.steps - 1:
            val = float((laplacian(model, val_x.clone()) ** 2).mean().detach())
            best_val = min(best_val, val)
            trial.report(val, step)
            if trial.should_prune():  # MedianPruner kills clearly-bad trials
                raise optuna.TrialPruned()
    return best_val, model


# ---------------------------------------------------------------------------
# 7. Hyperparameter search space (shared by Optuna TPE and the fallback).
# ---------------------------------------------------------------------------
def suggest_config(trial) -> dict:
    cfg = {
        # Architectural choice: how the D4 character symmetry is hard-enforced.
        "sym_mode": trial.suggest_categorical("sym_mode", ["reynolds", "prefactor"]),
        "depth": trial.suggest_int("depth", 2, 5),
        "width": trial.suggest_int("width", 32, 192, log=True),
        "activation": trial.suggest_categorical("activation", ["tanh", "sin", "gelu"]),
        "n_fourier": trial.suggest_categorical("n_fourier", [0, 16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "none"]),
        "batch": trial.suggest_categorical("batch", [1024, 2048, 4096]),
        "center_frac": trial.suggest_float("center_frac", 0.0, 0.5),
    }
    # Conditional parameters (only meaningful for some branches).
    cfg["ff_sigma"] = (
        trial.suggest_float("ff_sigma", 0.5, 4.0, log=True) if cfg["n_fourier"] > 0 else 0.0
    )
    cfg["weight_decay"] = (
        trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        if cfg["optimizer"] == "adamw"
        else 0.0
    )
    return cfg


class RandomTrial:
    """Minimal stand-in so the script runs without Optuna (pure random search)."""

    def __init__(self, number: int, rng: np.random.Generator) -> None:
        self.number, self.rng, self.params, self.user_attrs = number, rng, {}, {}

    def suggest_float(self, name, lo, hi, log=False):
        v = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))) if log else self.rng.uniform(lo, hi))
        self.params[name] = v
        return v

    def suggest_int(self, name, lo, hi, log=False):
        v = int(round(self.suggest_float(name, lo, hi, log)))
        self.params[name] = v
        return v

    def suggest_categorical(self, name, choices):
        v = choices[int(self.rng.integers(len(choices)))]
        self.params[name] = v
        return v

    def report(self, value, step):  # no pruning without Optuna
        pass

    def should_prune(self) -> bool:
        return False

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


# ---------------------------------------------------------------------------
# 8. Study driver.
# ---------------------------------------------------------------------------
def run_study(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.float64 else torch.float32
    rng = np.random.default_rng(args.seed + 999)
    # FIXED held-out validation collocation set: identical for every trial so
    # the objective is comparable across hyperparameter configurations.
    val_x = torch.as_tensor(sample_interior(args.val_points, rng), dtype=dtype, device=args.device)
    center_grid = sample_center_disk(2000, 0.25, rng)  # eta probe (r <= 0.25 r0)

    best = {"value": float("inf")}

    def objective(trial) -> float:
        cfg = suggest_config(trial)
        t0 = time.time()
        val, model = train_one(cfg, trial, args, val_x)
        # Physics diagnostics logged per trial (not part of the objective).
        with torch.no_grad():
            grid_t = torch.as_tensor(center_grid, dtype=dtype, device=args.device)
            v = model(grid_t).cpu().numpy()
        eta = eta_fit(center_grid, v)
        trial.set_user_attr("eta_fit", eta)
        trial.set_user_attr("wall_time_sec", time.time() - t0)
        if val < best["value"]:  # persist the best model as we go
            best.update(value=val, params=dict(cfg), eta=eta)
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        return val

    records = []
    if HAVE_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=args.storage,
            study_name="hard_bc_paul_trap",
            load_if_exists=bool(args.storage),
        )
        study.optimize(objective, n_trials=args.trials)
        for t in study.trials:
            records.append(
                {
                    "number": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    "params": t.params,
                    "user_attrs": t.user_attrs,
                }
            )
    else:
        print("[warn] optuna not installed -> random search (pip install optuna for TPE+pruning)")
        search_rng = np.random.default_rng(args.seed)
        for i in range(args.trials):
            trial = RandomTrial(i, search_rng)
            value = objective(trial)
            records.append(
                {
                    "number": i,
                    "state": "COMPLETE",
                    "value": value,
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                }
            )
            print(
                f"[trial {i}] val_residual={value:.3e} eta={trial.user_attrs['eta_fit']:.4f} "
                f"params={trial.params}"
            )

    summary = {"best": best, "trials": records, "optuna": HAVE_OPTUNA}
    (out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nbest val residual MSE: {best['value']:.4e}")
    print(f"best eta_fit:          {best.get('eta', float('nan')):.5f}")
    print(f"best params:           {best.get('params')}")
    print(f"artifacts:             {out_dir}/study_summary.json, best_model.pt")


# ---------------------------------------------------------------------------
# 9. Self-tests: prove the constraints are architectural, not penalized.
# ---------------------------------------------------------------------------
def selftest() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    bpts, bvals = sample_boundary(400, rng)
    interior = sample_interior(1000, rng)

    for mode in ("reynolds", "prefactor"):
        cfg = dict(
            sym_mode=mode, depth=3, width=64, activation="tanh",
            n_fourier=16, ff_sigma=1.0, lr=1e-3, optimizer="adam",
            scheduler="none", batch=256, center_frac=0.2, weight_decay=0.0,
        )
        # float64: any residual boundary/symmetry error is pure rounding noise.
        model = build_model(cfg, torch.float64, "cpu")

        # (a) Dirichlet BCs exact on all five boundary circles, UNTRAINED model.
        xb = torch.as_tensor(bpts, dtype=torch.float64)
        err_bc = (model(xb).detach().numpy() - bvals).max()
        assert abs(err_bc) < 1e-9, f"BC not exact ({mode}): {err_bc:.2e}"

        # (b) D4 character symmetry exact at random interior points.
        xi = torch.as_tensor(interior, dtype=torch.float64)
        with torch.no_grad():
            v = model(xi)
            mirror = model(torch.stack([-xi[:, 0], xi[:, 1]], dim=1))
            swap = model(torch.stack([xi[:, 1], xi[:, 0]], dim=1))
        err_m = (mirror - v).abs().max().item()
        err_s = (swap + v).abs().max().item()
        assert err_m < 1e-9 and err_s < 1e-9, f"symmetry broken ({mode}): {err_m:.2e} {err_s:.2e}"

        # (c) Laplacian is finite everywhere sampled (incl. near the origin,
        # which is why the outer boundary function is polynomial).
        near0 = torch.as_tensor(sample_center_disk(200, 0.3, rng), dtype=torch.float64)
        lap = laplacian(model, torch.cat([xi[:200], near0], dim=0))
        assert torch.isfinite(lap).all(), f"non-finite Laplacian ({mode})"
        print(f"[selftest] {mode:9s}: BC err {abs(err_bc):.1e}, mirror {err_m:.1e}, "
              f"swap {err_s:.1e}, laplacian finite -- PASS")

    # (d) eta convention: perfect normalized quadrupole must give eta = 1.
    pts = sample_center_disk(500, 0.25, rng)
    v_quad = (V0 / (2.0 * R0**2)) * (pts[:, 0] ** 2 - pts[:, 1] ** 2)
    eta = eta_fit(pts, v_quad)
    assert abs(eta - 1.0) < 1e-9, f"eta_fit convention broken: {eta}"
    print(f"[selftest] eta_fit(hyperbolic quadrupole) = {eta:.9f} -- PASS")


# ---------------------------------------------------------------------------
# 10. CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trials", type=int, default=20, help="hyperparameter trials")
    p.add_argument("--steps", type=int, default=4000, help="training steps per trial")
    p.add_argument("--pool", type=int, default=20000, help="collocation pool size")
    p.add_argument("--val-points", type=int, default=4096, help="held-out residual points")
    p.add_argument("--resample-every", type=int, default=500, help="pool refresh cadence")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--float64", action="store_true", help="double precision run")
    p.add_argument("--storage", default=None, help="optuna storage URL (e.g. sqlite:///study.db)")
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "results" / "hard_bc_tuning"),
        help="output directory for study summary + best model",
    )
    p.add_argument("--selftest", action="store_true", help="run exactness checks and exit")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run (2 trials)")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if args.smoke:  # small enough for a laptop-CPU sanity pass
        args.trials, args.steps = 2, 200
        args.pool, args.val_points = 2000, 512
    run_study(args)


if __name__ == "__main__":
    main()
