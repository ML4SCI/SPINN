import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn

from spinn.geometry.four_rod import FourRodTrap
from spinn.models.backends.decoders import DecoderMLP
from spinn.models.backends.pig import PigBackend
from pig_gate_common import exact_field, uniform_batch


DENSE_GEOMETRY = os.environ.get("DENSE_GEOMETRY", "0") == "1"
POLYNOMIAL_RESPONSE = os.environ.get("POLYNOMIAL_RESPONSE", "0") == "1"
SEED = int(os.environ.get("EXPERIMENT_SEED", "3"))
if POLYNOMIAL_RESPONSE:
    TRAIN_DELTA = np.linspace(-.05, .05, 7).tolist()
    TEST_DELTA = ((np.array(TRAIN_DELTA[:-1])+np.array(TRAIN_DELTA[1:]))/2).tolist()
    OUTPUT = "scratch/46_parity_polynomial_geometry_response.png" if SEED == 3 else f"scratch/48_parity_polynomial_seed{SEED}.png"
    HYPOTHESIS = "A quadratic geometry response with the exact delta/y parity removes unphysical interpolation modes and improves endpoint field and null accuracy over the free conditioned decoder." if SEED == 3 else "The accepted parity-polynomial response passes the same unseen field, null, and derivative gates from an independent initialization seed."
elif DENSE_GEOMETRY:
    TRAIN_DELTA = np.linspace(-.05, .05, 7).tolist()
    TEST_DELTA = ((np.array(TRAIN_DELTA[:-1])+np.array(TRAIN_DELTA[1:]))/2).tolist()
    OUTPUT = "scratch/45_dense_geometry_collocation.png"
    HYPOTHESIS = "Increasing only geometry collocation from three to seven physical slices reduces unseen field and null errors, proving that the Plot 44 failure was under-sampling along the design coordinate."
else:
    TRAIN_DELTA = [-.05, 0.0, .05]
    TEST_DELTA = [-.025, .025]
    OUTPUT = "scratch/44_conditioned_one_axis_null_gradient.png"
    HYPOTHESIS = "An x-mirror PIG conditioned on north-south electrode asymmetry interpolates unseen physical fields and recovers the derivative of the displaced electrical null with respect to that asymmetry."


def trap_at(delta):
    return FourRodTrap(r0=1., rod_radius=1.145, outer_radius=5., V0=1., center_perturb=(0., 0., delta, -delta))


class ConditionedMirrorPIG(nn.Module):
    def __init__(self):
        super().__init__()
        self.pig = PigBackend(outer_radius=5., num_gaussians=100, feature_dim=4, init_sigma=.1, decoder_layers=2, decoder_width=16)
        self.pig.decoder = nn.Identity()
        if POLYNOMIAL_RESPONSE:
            self.decoders = nn.ModuleList([DecoderMLP(4, layers=2, width=20) for _ in range(3)])
        else:
            self.decoder = DecoderMLP(5, layers=2, width=20)

    def head(self, index, points):
        return self.decoders[index](self.pig.features(points))

    def x_even_y_even(self, index, points):
        return .25*(self.head(index, points)+self.head(index, points*points.new_tensor([-1., 1.]))+self.head(index, points*points.new_tensor([1., -1.]))+self.head(index, -points))

    def x_even_y_odd(self, index, points):
        return .25*(self.head(index, points)+self.head(index, points*points.new_tensor([-1., 1.]))-self.head(index, points*points.new_tensor([1., -1.]))-self.head(index, -points))

    def d4_base(self, points):
        even = self.x_even_y_even(0, points)
        swapped = self.x_even_y_even(0, points[:, [1, 0]])
        return .5*(even-swapped)

    def polynomial_components(self, points):
        reflected_x = points*points.new_tensor([-1., 1.])
        reflected_y = points*points.new_tensor([1., -1.])
        swapped = points[:, [1, 0]]
        variants = [points, reflected_x, reflected_y, -points, swapped, swapped*points.new_tensor([-1., 1.]), swapped*points.new_tensor([1., -1.]), -swapped]
        # Gaussian features dominate the cost and are shared by all three
        # response heads. Compute each transformed feature block once.
        features = self.pig.features(torch.cat(variants, dim=0))
        head_values = [torch.chunk(decoder(features), 8, dim=0) for decoder in self.decoders]
        h0, h1, h2 = head_values
        base = .125*((h0[0]+h0[1]+h0[2]+h0[3])-(h0[4]+h0[5]+h0[6]+h0[7]))
        odd = .25*(h1[0]+h1[1]-h1[2]-h1[3])
        even2 = .25*(h2[0]+h2[1]+h2[2]+h2[3])
        return base, odd, even2

    def raw(self, points, delta):
        condition = (delta/.05).reshape(1, 1).expand(len(points), 1)
        return self.decoder(torch.cat([self.pig.features(points), condition], dim=1))

    def forward(self, points, delta):
        if POLYNOMIAL_RESPONSE:
            t = delta/.05
            base, odd, even2 = self.polynomial_components(points)
            return base+t*odd+t**2*even2
        reflected = points*points.new_tensor([-1., 1.])
        return .5*(self.raw(points, delta)+self.raw(reflected, delta))

    def groups(self):
        decoders = self.decoders.parameters() if POLYNOMIAL_RESPONSE else self.decoder.parameters()
        return [{"params": list(decoders)+[self.pig.weights], "lr": 1e-3}, {"params": [self.pig.means], "lr": 1e-4}, {"params": [self.pig.sigmas], "lr": 1e-4}]


def physics_loss(field, batch, delta):
    potential = field(batch.interior, delta)
    gradient = torch.autograd.grad(potential.sum(), batch.interior, create_graph=True)[0]
    laplacian = sum(torch.autograd.grad(gradient[:, i].sum(), batch.interior, create_graph=True)[0][:, i] for i in range(2))
    electrode = field(batch.electrode_points, delta)
    outer = field(batch.outer_points, delta)
    bc = torch.mean((electrode-batch.electrode_voltages)**2)+torch.mean((outer-batch.outer_voltages)**2)
    return torch.mean(laplacian**2), bc


line_y_np = np.linspace(-.30, .30, 61)
line_np = np.column_stack([np.zeros_like(line_y_np), line_y_np])
line = torch.tensor(line_np, dtype=torch.float32)


def null_torch(field, delta):
    potential = field(line, delta).squeeze(-1)
    matrix = torch.stack([line[:, 1]**order for order in range(8)], dim=1)
    coefficients = torch.linalg.lstsq(matrix, potential).solution
    root = delta*0.0
    for _ in range(10):
        gradient = sum(order*coefficients[order]*root**(order-1) for order in range(1, 8))
        curvature = sum(order*(order-1)*coefficients[order]*root**(order-2) for order in range(2, 8))
        root = root-gradient/curvature
    return root


def null_numpy(potential):
    coefficients = np.polynomial.polynomial.polyfit(line_y_np, potential, 7)
    first = np.polynomial.polynomial.polyder(coefficients)
    second = np.polynomial.polynomial.polyder(first)
    root = 0.
    for _ in range(10):
        root -= np.polynomial.polynomial.polyval(root, first)/np.polynomial.polynomial.polyval(root, second)
    return root


print("HYPOTHESIS:", HYPOTHESIS)
batches = {delta: uniform_batch(trap_at(delta), n_interior=768, n_electrode=64, n_outer=256, interior_seed=0) for delta in TRAIN_DELTA}
torch.manual_seed(SEED)
field = ConditionedMirrorPIG()


def total_loss():
    pde = 0.
    bc = 0.
    for value in TRAIN_DELTA:
        lpde, lbc = physics_loss(field, batches[value], torch.tensor(value, dtype=torch.float32))
        pde = pde+lpde
        bc = bc+lbc
    return pde/len(TRAIN_DELTA), bc/len(TRAIN_DELTA)


trajectory = {value: [] for value in TRAIN_DELTA}
def record(progress, stage):
    with torch.no_grad():
        values = []
        for value in TRAIN_DELTA:
            null = null_torch(field, torch.tensor(value)).item()
            trajectory[value].append((progress, null, stage))
            values.append(f"d={value:+.3f}:{null:+.4f}")
    print(stage, progress, " ".join(values))


optimizer = torch.optim.Adam(field.groups())
for step in range(500):
    optimizer.zero_grad()
    lpde, lbc = total_loss()
    (lpde+100*lbc).backward()
    optimizer.step()
    if step % 100 == 0 or step == 499:
        record(step, "Adam")

calls = 0
optimizer = torch.optim.LBFGS(field.parameters(), lr=.5, max_iter=20, max_eval=40, history_size=50, line_search_fn="strong_wolfe")
for _ in range(40):
    def closure():
        global calls
        optimizer.zero_grad()
        lpde, lbc = total_loss()
        loss = lpde+100*lbc
        loss.backward()
        calls += 1
        return loss
    optimizer.step(closure)
    record(500+calls, "LBFGS")

axis = np.linspace(-.6, .6, 51)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
probe_np = all_points[np.hypot(all_points[:, 0], all_points[:, 1]) <= .6]
probe = torch.tensor(probe_np, dtype=torch.float32)
fine_delta = np.linspace(-.05, .05, 17)
exact_curve = []
pig_curve = []
for value in fine_delta:
    exact_curve.append(null_numpy(exact_field(trap_at(float(value)), line_np)[0]))
    with torch.no_grad():
        pig_curve.append(null_torch(field, torch.tensor(float(value))).item())

results = {}
for value in sorted(TRAIN_DELTA+TEST_DELTA):
    trap = trap_at(value)
    reference_potential, reference_field = exact_field(trap, probe_np)
    exact_null = null_numpy(exact_field(trap, line_np)[0])
    delta = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    predicted_null = null_torch(field, delta)
    pig_slope = torch.autograd.grad(predicted_null, delta)[0].item()
    points = probe.clone().requires_grad_(True)
    prediction_tensor = field(points, delta.detach()).squeeze(-1)
    gradient = torch.autograd.grad(prediction_tensor.sum(), points)[0]
    prediction = prediction_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    p_scale = np.max(np.abs(reference_potential))
    e_scale = np.max(np.linalg.norm(reference_field, axis=1))
    v_rmse = np.sqrt(np.mean(((prediction-reference_potential)/p_scale)**2))
    e_rmse = np.sqrt(np.mean((np.linalg.norm(predicted_field-reference_field, axis=1)/e_scale)**2))
    h = .0025
    exact_minus = null_numpy(exact_field(trap_at(value-h), line_np)[0])
    exact_plus = null_numpy(exact_field(trap_at(value+h), line_np)[0])
    exact_slope = (exact_plus-exact_minus)/(2*h)
    results[value] = {"exact_null": exact_null, "pig_null": predicted_null.item(), "null_error": abs(predicted_null.item()-exact_null), "exact_slope": exact_slope, "pig_slope": pig_slope, "slope_error": abs(pig_slope-exact_slope)/abs(exact_slope), "v": v_rmse, "e": e_rmse}
    print(f"RESULT d={value:+.3f} {'UNSEEN' if value in TEST_DELTA else 'TRAIN'} null exact={exact_null:+.5f} pig={predicted_null.item():+.5f} error={results[value]['null_error']:.3e} slope exact={exact_slope:+.4f} pig={pig_slope:+.4f} rel={results[value]['slope_error']:.1%} V={v_rmse:.3e} E={e_rmse:.3e}")

ordered = sorted(results)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].plot(fine_delta, exact_curve, label="exact MFS")
axes[0, 0].plot(fine_delta, pig_curve, label="conditioned x-mirror PIG")
axes[0, 0].scatter(TEST_DELTA, [results[v]["pig_null"] for v in TEST_DELTA], marker="x", s=80, label="unseen")
axes[0, 0].set(xlabel="north/south asymmetry delta", ylabel="electrical-null y/r0", title="Physical null steering response")
axes[0, 0].legend()
for value in TRAIN_DELTA:
    rows = trajectory[value]
    axes[0, 1].plot([r[0] for r in rows], [r[1] for r in rows], marker="o", markersize=2, label=f"delta={value:+.2f}")
axes[0, 1].axvline(500, color="black", linestyle=":")
axes[0, 1].set(xlabel="optimizer evaluations", ylabel="predicted null y/r0", title="Joint family training")
axes[0, 1].legend(fontsize=8)
x = np.arange(len(ordered))
axes[1, 0].bar(x-.2, [results[v]["exact_slope"] for v in ordered], .4, label="exact finite difference")
axes[1, 0].bar(x+.2, [results[v]["pig_slope"] for v in ordered], .4, label="PIG autograd")
axes[1, 0].set_xticks(x, [f"{v:+.3f}"+("*" if v in TEST_DELTA else "") for v in ordered])
axes[1, 0].tick_params(axis="x", labelrotation=45, labelsize=7)
axes[1, 0].set(ylabel="d y_null / d delta", title="Shape gradient (* unseen)")
axes[1, 0].legend()
axes[1, 1].bar(x-.2, [results[v]["v"] for v in ordered], .4, label="V RMSE")
axes[1, 1].bar(x+.2, [results[v]["e"] for v in ordered], .4, label="E RMSE")
axes[1, 1].axhline(.005, color="red", linestyle="--", label="0.5% gate")
axes[1, 1].set_xticks(x, [f"{v:+.3f}"+("*" if v in TEST_DELTA else "") for v in ordered])
axes[1, 1].tick_params(axis="x", labelrotation=45, labelsize=7)
axes[1, 1].set(yscale="log", ylabel="normalized spatial RMSE", title="Independent field gate")
axes[1, 1].legend()
for ax in axes.ravel():
    ax.grid(alpha=.2)
plot_title = (f"parity-polynomial response replicates at seed {SEED}" if POLYNOMIAL_RESPONSE and SEED != 3 else "parity-aware polynomial response repairs geometry interpolation") if POLYNOMIAL_RESPONSE else ("dense geometry collocation repairs interpolation" if DENSE_GEOMETRY else "one-axis geometry conditioning supplies a physical null-steering gradient")
fig.suptitle("Hypothesis: " + plot_title, fontsize=15)
fig.savefig(OUTPUT, dpi=180)

null_pass = all(results[v]["null_error"] < .002 for v in TEST_DELTA)
field_pass = all(results[v]["v"] < .005 and results[v]["e"] < .005 for v in TEST_DELTA)
slope_pass = all(results[v]["slope_error"] < .25 for v in TEST_DELTA)
print("GATES unseen_null=", null_pass, "unseen_field=", field_pass, "unseen_slope_25percent=", slope_pass)
print("CONCLUSION: accept only if unseen field, null, and derivative gates pass; a smooth-looking response curve is not sufficient.")
print("saved", OUTPUT)
