from devsim import *
import numpy as np

device = "paul_trap"
mesh = "paul_trap_mesh"
region = "air"

create_gmsh_mesh(file="paul_trap_mesh.msh", mesh=mesh)

add_gmsh_region(mesh=mesh, gmsh_name="air", region="air", material="air")
add_gmsh_region(mesh=mesh, gmsh_name="North", region="North", material="iron")
add_gmsh_region(mesh=mesh, gmsh_name="South", region="South", material="iron")
add_gmsh_region(mesh=mesh, gmsh_name="East", region="East", material="iron")
add_gmsh_region(mesh=mesh, gmsh_name="West", region="West", material="iron")


add_gmsh_contact(mesh=mesh, gmsh_name="world_boundary", region="air", material="metal", name="world_boundary_contact")
add_gmsh_contact(mesh=mesh, gmsh_name="Northwest_contact",  region="air", material="metal", name="Northwest_contact")
add_gmsh_contact(mesh=mesh, gmsh_name="Southwest_contact", region="air", material="metal", name="Southwest_contact")
add_gmsh_contact(mesh=mesh, gmsh_name="Northeast_contact",  region="air", material="metal", name="Northeast_contact")
add_gmsh_contact(mesh=mesh, gmsh_name="Southeast_contact", region="air", material="metal", name="Southeast_contact")

finalize_mesh(mesh=mesh)
create_device(mesh=mesh, device=device)

# ----------------------------
# 2. Electrostatic PDE
# ----------------------------
eps0 = 8.854e-14   # F/cm if geometry is in cm
set_parameter(device=device, region=region, name="Permittivity", value=eps0)

node_solution(device=device, region=region, name="Potential")
edge_from_node_model(device=device, region=region, node_model="Potential")

edge_model(
    device=device, region=region, name="ElectricField",
    equation="(Potential@n0 - Potential@n1)*EdgeInverseLength",
    display_type="vector"
)
edge_model(device=device, region=region, name="ElectricField:Potential@n0", equation="EdgeInverseLength")
edge_model(device=device, region=region, name="ElectricField:Potential@n1", equation="-EdgeInverseLength")

edge_model(
    device=device, region=region, name="DField",
    equation="Permittivity*ElectricField",
    display_type="vector"
)
edge_model(device=device, region=region, name="DField:Potential@n0",
           equation="diff(Permittivity*ElectricField, Potential@n0)")
edge_model(device=device, region=region, name="DField:Potential@n1",
           equation="-DField:Potential@n0")

equation(
    device=device, region=region, name="PotentialEquation",
    variable_name="Potential", edge_model="DField", variable_update="default"
)

contacts = ["world_boundary_contact", "Northwest_contact", "Southwest_contact", "Northeast_contact", "Southeast_contact"]

for c in contacts:
    set_parameter(device=device, name=f"{c}_bias", value=0.0)

    contact_node_model(
        device=device, contact=c, name=f"{c}_bc",
        equation=f"Potential - {c}_bias"
    )
    contact_node_model(
        device=device, contact=c, name=f"{c}_bc:Potential",
        equation="1"
    )
    contact_equation(
        device=device, contact=c, name="PotentialEquation",
        node_model=f"{c}_bc", edge_charge_model="DField"
    )

# ----------------------------
# 3. Basis solve helper
# ----------------------------
def solve_basis(active_contact, all_contacts):
    for c in all_contacts:
        if c == "world_boundary_contact":
                set_parameter(device=device, name=f"{c}_bias", value=0.0)
        else:
                set_parameter(device=device, name=f"{c}_bias", value=1.0 if c == active_contact else 0.0)

    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)

    vals = np.array(get_node_model_values(device=device, region=region, name="Potential"))

    print(f"Active contact: {active_contact}")
    # print(f"vals: {vals}")

    write_devices(file=f"paul_trap_basis_DC_{active_contact}.dat", type="tecplot")

    return vals

# ----------------------------
# 4. Compute basis functions
# ----------------------------
phi_northeast_contact = solve_basis("Northeast_contact", contacts)
phi_southwest_contact = solve_basis("Southwest_contact", contacts)
phi_world_boundary_contact = solve_basis("world_boundary_contact", contacts)

# ----------------------------
# 5. Form total DC potential by superposition
# ----------------------------
north_potential = 1.0   # volts
south_potential = 1.0
world_boundary_potential = 0.0

node_models_list = get_node_model_list(device=device, region=region)
print(f"Node models list: {node_models_list}")

np.savez(
    "basis_functions_DC.npz",
    phi_northeast_contact=phi_northeast_contact,
    phi_southwest_contact=phi_southwest_contact,
    phi_world_boundary_contact=phi_world_boundary_contact,
)

