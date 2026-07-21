#!/usr/bin/env python3
"""Compare a trained Circular Trap PINN run against DEVSIM RF ground truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PINN_SCRIPT = PROJECT_ROOT / "SPINN" / "PINN" / "Vanilla_Paul_Trap_PINN" / "Paul_Trap_PINN.py"
DEFAULT_DEVSIM_CANDIDATES = [
    PROJECT_ROOT / "SPINN" / "DEVSIM" / "Circular_trap" / "paul_trap_basis_RF.dat",
    PROJECT_ROOT / "paul_trap_basis_RF.dat",
]


def load_pinn_module():
    spec = importlib.util.spec_from_file_location("paul_trap_pinn", PINN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load PINN script: {PINN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these DEVSIM RF files exist: " + ", ".join(str(p) for p in paths))


def pick_air_block(dataset):
    best = None
    best_n = -1
    if hasattr(dataset, "n_blocks"):
        for i in range(dataset.n_blocks):
            block = dataset[i]
            if block is not None and getattr(block, "n_points", 0) > best_n:
                best = block
                best_n = block.n_points
        return best.cell_data_to_point_data()
    return dataset.cell_data_to_point_data()


def read_devsim_with_pyvista(path):
    import pyvista as pv

    dataset = pv.get_reader(str(path)).read()
    mesh = pick_air_block(dataset)
    coords = np.asarray(mesh.points[:, :2], dtype=np.float32)
    data = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Potential": np.asarray(mesh["Potential"], dtype=np.float32),
    }
    if "Potential_gradx" in mesh.array_names and "Potential_grady" in mesh.array_names:
        data["Potential_gradx"] = np.asarray(mesh["Potential_gradx"], dtype=np.float32)
        data["Potential_grady"] = np.asarray(mesh["Potential_grady"], dtype=np.float32)
    return data


def parse_cell_centered_indices(header):
    match = re.search(r"VARLOCATION=\(\[([^\]]+)\]=CELLCENTERED\)", header)
    if match is None:
        return set()
    result = set()
    for part in match.group(1).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    return result


def read_devsim_tecplot_text(path):
    text = path.read_text()
    variables_match = re.search(r"VARIABLES\s*=\s*(.+?)\nZONE", text, flags=re.S)
    if variables_match is None:
        raise ValueError(f"could not find VARIABLES header in {path}")
    variables = re.findall(r'"([^"]+)"', variables_match.group(1))

    zones = list(re.finditer(r"(?m)^ZONE\b.*$", text))
    if not zones:
        raise ValueError(f"could not find ZONE blocks in {path}")

    wanted = {"x", "y", "Potential", "Potential_gradx", "Potential_grady"}
    best = None
    for idx, zone in enumerate(zones):
        header = zone.group(0)
        nodes_match = re.search(r"NODES=(\d+)", header)
        elements_match = re.search(r"ELEMENTS=(\d+)", header)
        if nodes_match is None or elements_match is None:
            continue
        nodes = int(nodes_match.group(1))
        elements = int(elements_match.group(1))
        end = zones[idx + 1].start() if idx + 1 < len(zones) else len(text)
        tokens = np.fromstring(text[zone.end():end], sep=" ")
        cell_centered = parse_cell_centered_indices(header)

        offset = 0
        arrays = {}
        for var_idx, var_name in enumerate(variables, start=1):
            count = elements if var_idx in cell_centered else nodes
            if offset + count > len(tokens):
                break
            values = tokens[offset:offset + count]
            offset += count
            if var_name in wanted and count == nodes:
                arrays[var_name] = values.astype(np.float32, copy=False)

        if {"x", "y", "Potential"}.issubset(arrays) and (best is None or nodes > len(best["x"])):
            best = arrays

    if best is None:
        raise ValueError(f"could not parse an air-like DEVSIM zone from {path}")
    return best


def read_devsim(path):
    try:
        return read_devsim_with_pyvista(path)
    except Exception as exc:
        print(f"PyVista reader unavailable or failed ({exc}); using text Tecplot parser.")
        return read_devsim_tecplot_text(path)


def evaluate_pinn(run_dir, coords, chunk_size=8192):
    pinn = load_pinn_module()
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    pinn.set_global_config(cfg)

    model_cfg = cfg["model"]
    model = pinn.MLP(
        width=model_cfg["width"],
        depth=model_cfg["depth"],
        activation=model_cfg.get("activation", "tanh"),
    )
    state = torch.load(run_dir / "model_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    phi_parts = []
    grad_parts = []
    for start in range(0, len(coords), chunk_size):
        xy = torch.tensor(coords[start:start + chunk_size], dtype=torch.float32, requires_grad=True)
        phi = model(xy)
        grad = torch.autograd.grad(
            phi,
            xy,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
        )[0]
        phi_parts.append(phi.detach().cpu().numpy().reshape(-1))
        grad_parts.append(grad.detach().cpu().numpy())

    phi = np.concatenate(phi_parts)
    grad = np.vstack(grad_parts)
    grad_m = grad / pinn.geom_unit_to_m
    grad2 = np.sum(grad_m ** 2, axis=1)
    psi_eV = pinn.scale * grad2 / pinn.e_charge
    return phi, psi_eV


def error_stats(pred, truth, norm=None):
    err = pred - truth
    abs_err = np.abs(err)
    stats = {
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(abs_err)),
        "p50_abs": float(np.percentile(abs_err, 50)),
        "p95_abs": float(np.percentile(abs_err, 95)),
        "max_abs": float(np.max(abs_err)),
    }
    if norm is not None and norm > 0:
        stats["mean_abs_percent_of_norm"] = float(100.0 * np.mean(abs_err) / norm)
        stats["p95_abs_percent_of_norm"] = float(100.0 * np.percentile(abs_err, 95) / norm)
    return stats


def scatter_triptych(x, y, truth, pred, label, units, out_path, max_points=20000):
    rng = np.random.default_rng(0)
    if len(x) > max_points:
        idx = rng.choice(len(x), max_points, replace=False)
    else:
        idx = np.arange(len(x))

    x_s, y_s = x[idx], y[idx]
    truth_s, pred_s = truth[idx], pred[idx]
    err_s = pred_s - truth_s
    vmin = float(min(np.nanmin(truth_s), np.nanmin(pred_s)))
    vmax = float(max(np.nanmax(truth_s), np.nanmax(pred_s)))
    err_abs = float(max(abs(np.nanmin(err_s)), abs(np.nanmax(err_s))))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    s0 = axes[0].scatter(x_s, y_s, c=truth_s, cmap="jet", s=3, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"DEVSIM {label}")
    plt.colorbar(s0, ax=axes[0], label=units)

    s1 = axes[1].scatter(x_s, y_s, c=pred_s, cmap="jet", s=3, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"PINN {label}")
    plt.colorbar(s1, ax=axes[1], label=units)

    s2 = axes[2].scatter(x_s, y_s, c=err_s, cmap="bwr", s=3, vmin=-err_abs, vmax=err_abs)
    axes[2].set_title("PINN - DEVSIM")
    plt.colorbar(s2, ax=axes[2], label=units)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare a trained PINN run to Circular Trap DEVSIM output.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--devsim_rf", type=Path, default=None)
    parser.add_argument("--max_points", type=int, default=20000)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    devsim_path = args.devsim_rf.resolve() if args.devsim_rf else first_existing(DEFAULT_DEVSIM_CANDIDATES)
    data = read_devsim(devsim_path)
    coords = np.column_stack([data["x"], data["y"]]).astype(np.float32)
    phi_devsim = data["Potential"].astype(np.float32)
    phi_pinn, psi_pinn = evaluate_pinn(run_dir, coords)

    metrics = {
        "run_dir": str(run_dir),
        "devsim_rf": str(devsim_path),
        "n_points": int(len(coords)),
        "phi_rf": error_stats(phi_pinn, phi_devsim, norm=300.0),
    }

    if "Potential_gradx" in data and "Potential_grady" in data:
        with open(run_dir / "config.json") as f:
            cfg = json.load(f)
        pinn = load_pinn_module()
        pinn.set_global_config(cfg)
        grad2_vm2 = 1.0e4 * (data["Potential_gradx"] ** 2 + data["Potential_grady"] ** 2)
        psi_devsim = pinn.scale * grad2_vm2 / pinn.e_charge
        metrics["psi_rf_eV"] = error_stats(psi_pinn, psi_devsim)
        scatter_triptych(
            data["x"], data["y"], psi_devsim, psi_pinn,
            label="psi_RF", units="eV", out_path=run_dir / "comparison_psi_rf_eV.png",
            max_points=args.max_points,
        )

    scatter_triptych(
        data["x"], data["y"], phi_devsim, phi_pinn,
        label="Phi_RF", units="V", out_path=run_dir / "comparison_phi_rf.png",
        max_points=args.max_points,
    )

    out_json = run_dir / "comparison_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved comparison metrics to {out_json}")


if __name__ == "__main__":
    main()
