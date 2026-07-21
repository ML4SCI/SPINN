"""Independent Gmsh/DEVSIM reference solves for electrode-distance sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SPINN_ROOT = Path(__file__).resolve().parents[2]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

from PINN.common.metrics import fit_local_quadratic, quadrupole_metrics
from PINN.common.tecplot import read_devsim_tecplot


def generate_mesh(
    output_path: Path,
    *,
    distance: float,
    radius: float,
    world_size: tuple[float, float],
    world_mesh_size: float,
    electrode_mesh_size: float,
) -> None:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.model.add("coordinate_projection_reference")
        geometry = gmsh.model.geo
        half_x, half_y = world_size[0] / 2.0, world_size[1] / 2.0

        corners = [
            geometry.addPoint(-half_x, -half_y, 0.0, world_mesh_size),
            geometry.addPoint(+half_x, -half_y, 0.0, world_mesh_size),
            geometry.addPoint(+half_x, +half_y, 0.0, world_mesh_size),
            geometry.addPoint(-half_x, +half_y, 0.0, world_mesh_size),
        ]
        world_lines = [
            geometry.addLine(corners[index], corners[(index + 1) % 4])
            for index in range(4)
        ]
        world_loop = geometry.addCurveLoop(world_lines)

        electrode_specs = {
            "East": (+distance, 0.0),
            "West": (-distance, 0.0),
            "North": (0.0, +distance),
            "South": (0.0, -distance),
        }
        electrode_surfaces: dict[str, int] = {}
        electrode_arcs: dict[str, list[int]] = {}
        electrode_loops: list[int] = []
        for name, (center_x, center_y) in electrode_specs.items():
            center = geometry.addPoint(center_x, center_y, 0.0, electrode_mesh_size)
            boundary_points = [
                geometry.addPoint(center_x + radius, center_y, 0.0, electrode_mesh_size),
                geometry.addPoint(center_x, center_y + radius, 0.0, electrode_mesh_size),
                geometry.addPoint(center_x - radius, center_y, 0.0, electrode_mesh_size),
                geometry.addPoint(center_x, center_y - radius, 0.0, electrode_mesh_size),
            ]
            arcs = [
                geometry.addCircleArc(boundary_points[index], center, boundary_points[(index + 1) % 4])
                for index in range(4)
            ]
            loop = geometry.addCurveLoop(arcs)
            electrode_loops.append(loop)
            electrode_surfaces[name] = geometry.addPlaneSurface([loop])
            electrode_arcs[name] = arcs

        air_surface = geometry.addPlaneSurface([world_loop, *electrode_loops])
        geometry.synchronize()

        air_tag = gmsh.model.addPhysicalGroup(2, [air_surface])
        gmsh.model.setPhysicalName(2, air_tag, "air")
        for name, surface in electrode_surfaces.items():
            tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, tag, name)
            contact_tag = gmsh.model.addPhysicalGroup(1, electrode_arcs[name])
            gmsh.model.setPhysicalName(1, contact_tag, f"{name}_contact")
        world_tag = gmsh.model.addPhysicalGroup(1, world_lines)
        gmsh.model.setPhysicalName(1, world_tag, "world_boundary")

        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()


def solve_devsim(mesh_path: Path, output_path: Path, *, rf_voltage: float) -> None:
    import devsim

    device = "paul_trap"
    mesh = "paul_trap_mesh"
    region = "air"
    devsim.create_gmsh_mesh(file=str(mesh_path), mesh=mesh)
    for name, material in [
        ("air", "air"),
        ("North", "iron"),
        ("South", "iron"),
        ("East", "iron"),
        ("West", "iron"),
    ]:
        devsim.add_gmsh_region(mesh=mesh, gmsh_name=name, region=name, material=material)
    for name in ["world_boundary", "North", "South", "East", "West"]:
        gmsh_name = "world_boundary" if name == "world_boundary" else f"{name}_contact"
        contact_name = (
            "world_boundary_contact" if name == "world_boundary" else f"{name}_contact"
        )
        devsim.add_gmsh_contact(
            mesh=mesh,
            gmsh_name=gmsh_name,
            region=region,
            material="metal",
            name=contact_name,
        )
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=device)

    devsim.set_parameter(device=device, region=region, name="Permittivity", value=8.854e-14)
    devsim.node_solution(device=device, region=region, name="Potential")
    devsim.edge_from_node_model(device=device, region=region, node_model="Potential")
    devsim.edge_model(
        device=device,
        region=region,
        name="ElectricField",
        equation="(Potential@n0 - Potential@n1)*EdgeInverseLength",
        display_type="vector",
    )
    devsim.edge_model(
        device=device,
        region=region,
        name="ElectricField:Potential@n0",
        equation="EdgeInverseLength",
    )
    devsim.edge_model(
        device=device,
        region=region,
        name="ElectricField:Potential@n1",
        equation="-EdgeInverseLength",
    )
    devsim.edge_model(
        device=device,
        region=region,
        name="DField",
        equation="Permittivity*ElectricField",
        display_type="vector",
    )
    devsim.edge_model(
        device=device,
        region=region,
        name="DField:Potential@n0",
        equation="diff(Permittivity*ElectricField, Potential@n0)",
    )
    devsim.edge_model(
        device=device,
        region=region,
        name="DField:Potential@n1",
        equation="-DField:Potential@n0",
    )
    devsim.equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        edge_model="DField",
        variable_update="default",
    )

    biases = {
        "world_boundary_contact": 0.0,
        "North_contact": 0.0,
        "South_contact": 0.0,
        "East_contact": rf_voltage,
        "West_contact": rf_voltage,
    }
    for contact, bias in biases.items():
        devsim.set_parameter(device=device, name=f"{contact}_bias", value=bias)
        devsim.contact_node_model(
            device=device,
            contact=contact,
            name=f"{contact}_bc",
            equation=f"Potential - {contact}_bias",
        )
        devsim.contact_node_model(
            device=device,
            contact=contact,
            name=f"{contact}_bc:Potential",
            equation="1",
        )
        devsim.contact_equation(
            device=device,
            contact=contact,
            name="PotentialEquation",
            node_model=f"{contact}_bc",
            edge_charge_model="DField",
        )

    devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
    devsim.vector_gradient(device=device, region=region, node_model="Potential", calc_type="default")
    devsim.write_devices(file=str(output_path), type="tecplot")


def summarize_solution(
    field_path: Path,
    *,
    distance: float,
    radius: float,
    rf_voltage: float,
    fit_radius: float,
) -> dict[str, object]:
    data = read_devsim_tecplot(field_path)
    points = np.column_stack([data["x"], data["y"]])
    fit = fit_local_quadratic(points, data["Potential"], radius=fit_radius)
    metrics = quadrupole_metrics(
        np.asarray(fit["hessian"]),
        characteristic_radius=distance - radius,
        voltage_span=rf_voltage,
    )
    return {
        "distance_cm": distance,
        "radius_cm": radius,
        "fit_radius_cm": fit_radius,
        "fit_rmse_V": fit["fit_rmse"],
        **metrics,
    }


def run_single(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / "mesh.msh"
    field_path = args.output_dir / "field.dat"
    generate_mesh(
        mesh_path,
        distance=args.distance,
        radius=args.radius,
        world_size=(args.world_size_x, args.world_size_y),
        world_mesh_size=args.world_mesh_size,
        electrode_mesh_size=args.electrode_mesh_size,
    )
    solve_devsim(mesh_path, field_path, rf_voltage=args.rf_voltage)
    metrics = summarize_solution(
        field_path,
        distance=args.distance,
        radius=args.radius,
        rf_voltage=args.rf_voltage,
        fit_radius=args.fit_radius,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def run_sweep(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for distance in args.distances:
        run_dir = args.output_dir / f"distance_{distance:.6f}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--distance",
            str(distance),
            "--output-dir",
            str(run_dir),
            "--radius",
            str(args.radius),
            "--rf-voltage",
            str(args.rf_voltage),
            "--world-size-x",
            str(args.world_size_x),
            "--world-size-y",
            str(args.world_size_y),
            "--world-mesh-size",
            str(args.world_mesh_size),
            "--electrode-mesh-size",
            str(args.electrode_mesh_size),
            "--fit-radius",
            str(args.fit_radius),
        ]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])
        records.append(json.loads((run_dir / "metrics.json").read_text()))
    (args.output_dir / "sweep_metrics.json").write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records)} DEVSIM reference solves to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--distance", type=float)
    parser.add_argument("--distances", type=float, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--rf-voltage", type=float, default=300.0)
    parser.add_argument("--world-size-x", type=float, default=2.5)
    parser.add_argument("--world-size-y", type=float, default=2.0)
    parser.add_argument("--world-mesh-size", type=float, default=0.04)
    parser.add_argument("--electrode-mesh-size", type=float, default=0.004)
    parser.add_argument("--fit-radius", type=float, default=0.035)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.single:
        if args.distance is None:
            raise SystemExit("--single requires --distance")
        run_single(args)
    else:
        if not args.distances:
            raise SystemExit("provide --distances for a sweep")
        run_sweep(args)


if __name__ == "__main__":
    main()
