#!/usr/bin/env python3
"""Check backend input differentiability and backend-to-shape gradient flow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from spinn.config import load_config
from spinn.geometry.four_rod import build_trap
from spinn.losses.objective import central_quadrupole_loss, eta_loss
from spinn.models.backends import build_backend
from spinn.models.composed import ComposedField
from spinn.models.shape_network import build_shape_network


def _norm_from_grads(grads) -> float:
    total = None
    for grad in grads:
        if grad is None:
            continue
        val = torch.sum(grad.detach() ** 2)
        total = val if total is None else total + val
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach())


def _shape_grad_norm(loss: torch.Tensor, model: ComposedField) -> float:
    params = [p for p in model.shape_network.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    return _norm_from_grads(grads)


def _backend_input_stats(backend, points: torch.Tensor) -> dict[str, float]:
    x = points.detach().clone().requires_grad_(True)
    v = backend(x)
    grad = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    d2x = torch.autograd.grad(
        grad[:, 0], x, torch.ones_like(grad[:, 0]), create_graph=True, retain_graph=True
    )[0][:, 0]
    d2y = torch.autograd.grad(
        grad[:, 1], x, torch.ones_like(grad[:, 1]), create_graph=True, retain_graph=True
    )[0][:, 1]
    grad_norm = torch.linalg.vector_norm(grad, dim=1)
    hess_diag_norm = torch.sqrt(d2x**2 + d2y**2)
    return {
        "dVdx_mean": float(grad_norm.mean().detach()),
        "dVdx_max": float(grad_norm.max().detach()),
        "d2V_diag_mean": float(hess_diag_norm.mean().detach()),
        "d2V_diag_max": float(hess_diag_norm.max().detach()),
        "nonzero_dVdx_fraction": float((grad_norm > 1.0e-10).float().mean().detach()),
        "nonzero_d2V_fraction": float((hess_diag_norm > 1.0e-10).float().mean().detach()),
    }


def diagnose_backend(name: str, n_points: int, seed: int) -> dict:
    cfg = load_config(f"joint_{name}.yaml")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    trap = build_trap(cfg)
    model = ComposedField(build_shape_network(cfg, trap), build_backend(cfg)).float()
    points = torch.tensor(
        trap.sample_interior(n_points, rng), dtype=torch.float32, requires_grad=True
    )
    stats = {"backend": name}
    stats.update(_backend_input_stats(model.backend, points))
    z = points.detach().clone().requires_grad_(True)
    v, x = model(z)
    shape_params = [p for p in model.shape_network.parameters() if p.requires_grad]
    stats["grad_phi_backend_meanV"] = _norm_from_grads(
        torch.autograd.grad(v.mean(), shape_params, allow_unused=True, retain_graph=True)
    )
    stats["mean_displacement_initial"] = float(
        torch.linalg.vector_norm((x - z).detach(), dim=1).mean()
    )
    stats["grad_phi_eta"] = _shape_grad_norm(eta_loss(model, cfg.experiment.r0, cfg.experiment.V0), model)
    stats["grad_phi_quad"] = _shape_grad_norm(
        central_quadrupole_loss(
            model,
            cfg.experiment.r0,
            cfg.experiment.V0,
            radius=cfg.sampling.quadrupole_loss_radius,
            grid=cfg.sampling.quadrupole_loss_grid,
        ),
        model,
    )
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["mlp", "pixel", "pig"])
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="spinn/results/backend_differentiability")
    args = parser.parse_args(argv)

    rows = [diagnose_backend(name, args.points, args.seed) for name in args.backends]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "backend_differentiability.json"
    csv_path = out / "backend_differentiability.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['backend']}: dVdx_mean={row['dVdx_mean']:.3e} "
            f"d2_mean={row['d2V_diag_mean']:.3e} "
            f"grad_phi_eta={row['grad_phi_eta']:.3e} "
            f"grad_phi_quad={row['grad_phi_quad']:.3e}"
        )
    print(f"[diagnose] wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
