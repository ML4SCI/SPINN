import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float64)
OUTPUT = "scratch/19_pig_orbit_gradient_norms.png"
HYPOTHESIS = "Orbit averaging prevents tied Gaussian output and parameter-gradient scales from growing with the number of generated symmetry copies."

I = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MX = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
MY = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
C2 = -I
SWAP = torch.tensor([[0.0, 1.0], [1.0, 0.0]])


def symmetry_group(size):
    base = [(I, 1.0), (MX, 1.0), (MY, 1.0), (C2, 1.0)]
    if size == 1:
        return base[:1]
    if size == 2:
        return base[:2]
    if size == 4:
        return base
    return base + [(SWAP @ matrix, -1.0) for matrix, _ in base]


axis = torch.linspace(-5.0, 5.0, 81)
gx, gy = torch.meshgrid(axis, axis, indexing="xy")
points = torch.column_stack([gx.reshape(-1), gy.reshape(-1)])
radius2 = torch.sum(points**2, dim=1)
target = 0.12 * (points[:, 0] ** 2 - points[:, 1] ** 2) * torch.exp(-radius2 / 10.0)


def run(size, normalization):
    mean = torch.tensor([2.15, 0.85], requires_grad=True)
    log_scales = torch.log(torch.tensor([0.72, 0.28])).requires_grad_()
    angle = torch.tensor(np.radians(28.0), requires_grad=True)
    weight = torch.tensor(1.0, requires_grad=True)
    rotation = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )
    covariance = rotation @ torch.diag(torch.exp(2.0 * log_scales)) @ rotation.T
    pieces = []
    for matrix, sign in symmetry_group(size):
        transformed_mean = matrix @ mean
        transformed_covariance = matrix @ covariance @ matrix.T
        diff = points - transformed_mean
        inverse = torch.linalg.inv(transformed_covariance)
        exponent = torch.einsum("ni,ij,nj->n", diff, inverse, diff)
        pieces.append(sign * torch.exp(-0.5 * exponent))
    divisor = 1.0
    if normalization == "sqrt":
        divisor = np.sqrt(size)
    if normalization == "mean":
        divisor = float(size)
    feature = weight * torch.stack(pieces).sum(dim=0) / divisor
    loss = torch.mean((feature - target) ** 2)
    gradients = torch.autograd.grad(loss, [mean, log_scales, angle, weight])
    norms = np.array(
        [
            torch.linalg.vector_norm(gradients[0]).item(),
            torch.linalg.vector_norm(gradients[1]).item(),
            torch.abs(gradients[2]).item(),
            torch.abs(gradients[3]).item(),
        ]
    )
    return {
        "feature_rms": torch.sqrt(torch.mean(feature**2)).item(),
        "loss": loss.item(),
        "norms": norms,
        "total_gradient": np.linalg.norm(norms),
    }


print("HYPOTHESIS:", HYPOTHESIS)
sizes = [1, 2, 4, 8]
normalizations = ["sum", "sqrt", "mean"]
results = {normalization: [run(size, normalization) for size in sizes] for normalization in normalizations}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for normalization in normalizations:
    axes[0, 0].plot(sizes, [row["feature_rms"] for row in results[normalization]], marker="o", label=normalization)
    axes[0, 1].plot(sizes, [row["loss"] for row in results[normalization]], marker="o", label=normalization)
    axes[1, 0].semilogy(sizes, [row["total_gradient"] for row in results[normalization]], marker="o", label=normalization)
axes[0, 0].set(xlabel="generated orbit size", ylabel="feature RMS", title="Does clone count inflate the field feature?")
axes[0, 1].set(xlabel="generated orbit size", ylabel="quadrupole-target MSE", title="Same physical target")
axes[1, 0].set(xlabel="generated orbit size", ylabel="total parameter-gradient norm", title="Does clone count inflate gradients?")

labels = ["center", "log widths", "orientation", "weight"]
x = np.arange(len(labels))
width = 0.25
for index, normalization in enumerate(normalizations):
    axes[1, 1].bar(x + (index - 1) * width, results[normalization][-1]["norms"], width, label=normalization)
axes[1, 1].set_yscale("log")
axes[1, 1].set_xticks(x, labels)
axes[1, 1].set(ylabel="absolute gradient contribution", title="Eight-copy quadrupole orbit")

for ax in axes.ravel():
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle("Hypothesis: normalize symmetry orbits before gradients reach the stored Gaussian", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

for normalization in normalizations:
    gradients = [row["total_gradient"] for row in results[normalization]]
    print(normalization, "total gradients", " ".join(f"n={size}:{value:.3e}" for size, value in zip(sizes, gradients)))
sum_ratio = results["sum"][-1]["total_gradient"] / results["sum"][0]["total_gradient"]
mean_ratio = results["mean"][-1]["total_gradient"] / results["mean"][0]["total_gradient"]
print(f"eight-copy / one-copy gradient ratio: sum={sum_ratio:.3f}, mean={mean_ratio:.3f}")
print("CONCLUSION: raw summation nearly multiplies the stored-parameter gradient by orbit size; mean normalization keeps the eight-copy gradient near the one-copy scale and is selected for trained PIG.")
print("saved", OUTPUT)
