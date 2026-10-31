import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# Same quadrupole problem, network, points and seeds as loss_only_experiment.py.
# Question: once loss weights follow gradient norms, does the Coulomb unit conversion still matter?
# Each converted term is a constant multiple of its dimensionless term, so a weight of 1/|grad L_i|
# should cancel the conversion exactly.

COULOMB_VOLT_METRE = 1.4399645478425669e-9
R0_METRES = 1.0e-3
V0_VOLTS = 100.0
TRAIN_STEPS = 400
REBALANCE_EVERY = 20
EMA = 0.9
OUTPUT = "exp/coulomb_natural_units_stress_test/results/07_gradnorm_weighting.png"


def exact_u(points):
    return 0.5 * (points[:, 0:1] ** 2 - points[:, 1:2] ** 2)


def exact_e_hat(points):
    return torch.cat([-points[:, 0:1], points[:, 1:2]], dim=1)


def square_boundary(count):
    c = torch.linspace(-1.0, 1.0, count, dtype=torch.float64)
    one = torch.ones_like(c)
    return torch.cat([torch.stack([c, one], 1), torch.stack([c, -one], 1), torch.stack([-one, c], 1), torch.stack([one, c], 1)])


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        ).double()

    def forward(self, points):
        return self.layers(points)


def derivatives(model, points):
    u = model(points)
    g = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    lap = sum(torch.autograd.grad(g[:, a].sum(), points, create_graph=True)[0][:, a:a + 1] for a in range(2))
    return u, -g, lap


def terms(model, units):
    interior = interior_base.clone().requires_grad_(True)
    field_points = field_base.clone().requires_grad_(True)
    _, _, lap = derivatives(model, interior)
    _, e_hat, _ = derivatives(model, field_points)
    dv = model(boundary) - exact_u(boundary)
    de = torch.linalg.vector_norm(e_hat - exact_e_hat(field_points), dim=1)
    dimensionless = torch.stack([torch.mean(lap**2), torch.mean(dv**2), torch.mean(de)])
    if units == "dimensionless":
        return dimensionless, dimensionless
    scale = torch.tensor([(V0_VOLTS / R0_METRES**2) ** 2, (V0_VOLTS / COULOMB_VOLT_METRE) ** 2,
                          V0_VOLTS / R0_METRES / COULOMB_VOLT_METRE], dtype=torch.float64)
    return scale * dimensionless, dimensionless


def grad_norms(ts, model):
    out = []
    for t in ts:
        gs = torch.autograd.grad(t, model.parameters(), retain_graph=True)
        out.append(torch.sqrt(sum((g**2).sum() for g in gs)))
    return torch.stack(out).detach()


torch.manual_seed(19)
generator = torch.Generator().manual_seed(23)
interior_base = 2.0 * torch.rand((512, 2), generator=generator, dtype=torch.float64) - 1.0
boundary = square_boundary(80)
field_base = 1.6 * torch.rand((192, 2), generator=generator, dtype=torch.float64) - 0.8
seed_state = {k: v.detach().clone() for k, v in SmallMLP().state_dict().items()}

# (units, weighting): "ones" = alpha 1, "fixed" = 1/|grad| at step 0, "adaptive" = 1/|grad| refreshed with EMA
schemes = {
    "dimensionless, alpha=1": ("dimensionless", "ones"),
    "Coulomb, alpha=1": ("converted", "ones"),
    "Coulomb, fixed grad-norm": ("converted", "fixed"),
    "dimensionless, adaptive grad-norm": ("dimensionless", "adaptive"),
    "Coulomb, adaptive grad-norm": ("converted", "adaptive"),
}

print("HYPOTHESIS: with 1/|grad L_i| weights, Coulomb-converted and dimensionless losses train identically; adaptive weights beat alpha=1.")
results = {}
for name, (units, weighting) in schemes.items():
    model = SmallMLP()
    model.load_state_dict(seed_state)
    opt = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    w = torch.ones(3, dtype=torch.float64)
    if weighting != "ones":
        w = 1.0 / grad_norms(terms(model, units)[0], model)
    hist, w_hist = [], []
    for step in range(TRAIN_STEPS):
        loss_terms, dimensionless = terms(model, units)
        if weighting == "adaptive" and step > 0 and step % REBALANCE_EVERY == 0:
            w = EMA * w + (1 - EMA) / grad_norms(loss_terms, model)
        opt.zero_grad()
        (w * loss_terms).sum().backward()
        opt.step()
        if step % 10 == 0 or step == TRAIN_STEPS - 1:
            hist.append([step, *dimensionless.detach().numpy()])
            w_hist.append([step, *(w / w[1]).numpy()])

    ax_ = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    gx, gy = torch.meshgrid(ax_, ax_, indexing="xy")
    pts = torch.stack([gx.ravel(), gy.ravel()], 1).requires_grad_(True)
    _, e_hat, _ = derivatives(model, pts)
    e_err = torch.linalg.vector_norm(e_hat - exact_e_hat(pts), dim=1).detach().reshape(101, 101).numpy()
    bc = torch.max(torch.abs(model(boundary) - exact_u(boundary))).item()
    results[name] = dict(hist=np.array(hist), w=np.array(w_hist), e_err=e_err, bc=bc)
    h = results[name]["hist"][-1]
    print(f"{name}: Laplace MSE={h[1]:.3e}  V MSE={h[2]:.3e}  E MAE={h[3]:.3e}  max boundary error={bc:.3e}")

a, b = results["dimensionless, adaptive grad-norm"]["hist"], results["Coulomb, adaptive grad-norm"]["hist"]
print(f"max relative gap between the two adaptive runs' diagnostics: {np.max(np.abs(a[:, 1:] - b[:, 1:]) / np.abs(a[:, 1:])):.2e}")

fig, axes = plt.subplots(2, 5, figsize=(22, 8.5), constrained_layout=True)
styles = {"dimensionless, alpha=1": ("tab:purple", "-"), "Coulomb, alpha=1": ("tab:blue", "-"),
          "Coulomb, fixed grad-norm": ("tab:orange", "-"), "dimensionless, adaptive grad-norm": ("tab:green", "-"),
          "Coulomb, adaptive grad-norm": ("black", ":")}
for name, r in results.items():
    c, ls = styles[name]
    for j, title in enumerate(["Laplace MSE", "boundary V MSE", "E MAE"]):
        axes[0, j].semilogy(r["hist"][:, 0], r["hist"][:, j + 1], color=c, ls=ls, lw=2 if ls == ":" else 1.3, label=name)
        axes[0, j].set(title=f"{title} (dimensionless)", xlabel="Adam step")
for k, (name, axis) in enumerate([("dimensionless, adaptive grad-norm", axes[0, 3]), ("Coulomb, adaptive grad-norm", axes[0, 4])]):
    w = results[name]["w"]
    for j, lab in enumerate(["Laplace", "V", "E"]):
        axis.semilogy(w[:, 0], w[:, j + 1], label=f"w_{lab} / w_V")
    axis.set(title=f"weights relative to V\n{name}", xlabel="Adam step")
    axis.legend(fontsize=8)
axes[0, 2].legend(fontsize=8)
vmax = max(r["e_err"].max() for r in results.values())
for axis, (name, r) in zip(axes[1], results.items()):
    im = axis.imshow(r["e_err"], origin="lower", extent=(-1, 1, -1, 1), cmap="magma", vmin=0, vmax=vmax)
    axis.set(title=f"{name}\nE MAE={r['hist'][-1, 3]:.4f}, max BC err={r['bc']:.1e}", xlabel="x/r0", ylabel="y/r0")
fig.colorbar(im, ax=axes[1], shrink=0.8, label="|E - E_exact| (shared scale)")
fig.suptitle("Gradient-norm weighting removes the effect of the Coulomb unit conversion", fontsize=15)
fig.savefig(OUTPUT, dpi=150)
print(f"saved {OUTPUT}")
