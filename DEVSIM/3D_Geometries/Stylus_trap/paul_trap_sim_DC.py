from devsim import *
import numpy as np

device = "paul_trap"
mesh = "paul_trap_mesh"
region = "air"

create_gmsh_mesh(file="paul_trap_stylus_mesh.msh", mesh=mesh)

ELECTRODES = ["Stylus_RF", "Ground_Ring", "DC_Comp_E", "DC_Comp_W"]

add_gmsh_region(mesh=mesh, gmsh_name="air", region="air", material="air")
for name in ELECTRODES:
    add_gmsh_region(mesh=mesh, gmsh_name=name, region=name, material="iron")

add_gmsh_contact(mesh=mesh, gmsh_name="world_boundary", region="air",
                 material="metal", name="world_boundary_contact")
for name in ELECTRODES:
    add_gmsh_contact(mesh=mesh, gmsh_name=f"{name}_contact", region="air",
                     material="metal", name=f"{name}_contact")

finalize_mesh(mesh=mesh)
create_device(mesh=mesh, device=device)

# ----------------------------
# Electrostatic PDE
# ----------------------------
eps0 = 8.854e-14   # F/cm
set_parameter(device=device, region=region, name="Permittivity", value=eps0)

node_solution(device=device, region=region, name="Potential")
edge_from_node_model(device=device, region=region, node_model="Potential")

edge_model(device=device, region=region, name="ElectricField",
           equation="(Potential@n0 - Potential@n1)*EdgeInverseLength",
           display_type="vector")
edge_model(device=device, region=region, name="ElectricField:Potential@n0",
           equation="EdgeInverseLength")
edge_model(device=device, region=region, name="ElectricField:Potential@n1",
           equation="-EdgeInverseLength")

edge_model(device=device, region=region, name="DField",
           equation="Permittivity*ElectricField", display_type="vector")
edge_model(device=device, region=region, name="DField:Potential@n0",
           equation="diff(Permittivity*ElectricField, Potential@n0)")
edge_model(device=device, region=region, name="DField:Potential@n1",
           equation="-DField:Potential@n0")

equation(device=device, region=region, name="PotentialEquation",
         variable_name="Potential", edge_model="DField", variable_update="default")

contacts = ["world_boundary_contact"] + [f"{n}_contact" for n in ELECTRODES]

for c in contacts:
    set_parameter(device=device, name=f"{c}_bias", value=0.0)
    contact_node_model(device=device, contact=c, name=f"{c}_bc",
                       equation=f"Potential - {c}_bias")
    contact_node_model(device=device, contact=c, name=f"{c}_bc:Potential", equation="1")
    contact_equation(device=device, contact=c, name="PotentialEquation",
                     node_model=f"{c}_bc", edge_charge_model="DField")

def solve_basis(active_contact, all_contacts):
    for c in all_contacts:
        if c == "world_boundary_contact":
            set_parameter(device=device, name=f"{c}_bias", value=0.0)
        else:
            set_parameter(device=device, name=f"{c}_bias",
                          value=1.0 if c == active_contact else 0.0)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
    vals = np.array(get_node_model_values(device=device, region=region, name="Potential"))
    print(f"Active contact: {active_contact}")
    write_devices(file=f"paul_trap_basis_DC_{active_contact}.dat", type="tecplot")
    return vals

# DC base functions for ring + comp pads + world. Stylus excluded (RF).
dc_active = [
    "Ground_Ring_contact",
    "DC_Comp_E_contact",
    "DC_Comp_W_contact",
    "world_boundary_contact",
]
basis = {c: solve_basis(c, contacts) for c in dc_active}

np.savez("basis_functions_DC.npz", **{f"phi_{c}": basis[c] for c in dc_active})
