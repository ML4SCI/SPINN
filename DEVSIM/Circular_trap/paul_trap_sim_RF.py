from devsim import *
import numpy as np

# ============================================================
# Config
# ============================================================
device = "paul_trap"
mesh = "paul_trap_mesh"
region = "air"

world_bc_type = "neumann"   # choose "dirichlet" or "neumann"
world_bc_value = 0.0        # volts for Dirichlet. For Neumann, this must be 0.0

world_contact = "world_boundary_contact"

DC_contacts = [
    "North_contact",
    "South_contact",
]

RF_contacts = [
    "East_contact",
    "West_contact",
]

DC_voltage = 0.0
RF_voltage = 300.0

# ============================================================
# 1. Mesh
# ============================================================
create_gmsh_mesh(file="paul_trap_mesh.msh", mesh=mesh)

add_gmsh_region(mesh=mesh, gmsh_name="air", region="air", material="air")
add_gmsh_region(mesh=mesh, gmsh_name="North", region="North", material="iron")
add_gmsh_region(mesh=mesh, gmsh_name="South", region="South", material="iron")
add_gmsh_region(mesh=mesh, gmsh_name="East", region="East", material="iron")
add_gmsh_region(mesh=mesh, gmsh_name="West", region="West", material="iron")

add_gmsh_contact(
    mesh=mesh,
    gmsh_name="world_boundary",
    region="air",
    material="metal",
    name=world_contact,
)

add_gmsh_contact(
    mesh=mesh,
    gmsh_name="North_contact",
    region="air",
    material="metal",
    name="North_contact",
)

add_gmsh_contact(
    mesh=mesh,
    gmsh_name="South_contact",
    region="air",
    material="metal",
    name="South_contact",
)

add_gmsh_contact(
    mesh=mesh,
    gmsh_name="East_contact",
    region="air",
    material="metal",
    name="East_contact",
)

add_gmsh_contact(
    mesh=mesh,
    gmsh_name="West_contact",
    region="air",
    material="metal",
    name="West_contact",
)

finalize_mesh(mesh=mesh)
create_device(mesh=mesh, device=device)

# ============================================================
# 2. Electrostatic PDE
# ============================================================
eps0 = 8.854e-14   # F/cm if geometry is in cm

set_parameter(device=device, region=region, name="Permittivity", value=eps0)

node_solution(device=device, region=region, name="Potential")
edge_from_node_model(device=device, region=region, node_model="Potential")

edge_model(
    device=device,
    region=region,
    name="ElectricField",
    equation="(Potential@n0 - Potential@n1)*EdgeInverseLength",
    display_type="vector",
)

edge_model(
    device=device,
    region=region,
    name="ElectricField:Potential@n0",
    equation="EdgeInverseLength",
)

edge_model(
    device=device,
    region=region,
    name="ElectricField:Potential@n1",
    equation="-EdgeInverseLength",
)

edge_model(
    device=device,
    region=region,
    name="DField",
    equation="Permittivity*ElectricField",
    display_type="vector",
)

edge_model(
    device=device,
    region=region,
    name="DField:Potential@n0",
    equation="diff(Permittivity*ElectricField, Potential@n0)",
)

edge_model(
    device=device,
    region=region,
    name="DField:Potential@n1",
    equation="-DField:Potential@n0",
)

equation(
    device=device,
    region=region,
    name="PotentialEquation",
    variable_name="Potential",
    edge_model="DField",
    variable_update="default",
)

# ============================================================
# 3. Boundary conditions
# ============================================================
def add_dirichlet_contact(contact, bias_value):
    set_parameter(device=device, name=f"{contact}_bias", value=bias_value)

    contact_node_model(
        device=device,
        contact=contact,
        name=f"{contact}_bc",
        equation=f"Potential - {contact}_bias",
    )

    contact_node_model(
        device=device,
        contact=contact,
        name=f"{contact}_bc:Potential",
        equation="1",
    )

    contact_equation(
        device=device,
        contact=contact,
        name="PotentialEquation",
        node_model=f"{contact}_bc",
        edge_charge_model="DField",
    )


for c in DC_contacts:
    add_dirichlet_contact(c, DC_voltage)

for c in RF_contacts:
    add_dirichlet_contact(c, RF_voltage)


if world_bc_type.lower() == "dirichlet":
    add_dirichlet_contact(world_contact, world_bc_value)

elif world_bc_type.lower() == "neumann":
    if abs(world_bc_value) > 0.0:
        raise ValueError(
            "This script currently implements only homogeneous Neumann on the "
            "world boundary. Set world_bc_value = 0.0 for Neumann."
        )

    # Do not add a contact_equation for the world boundary.
    # For this electrostatic PDE, leaving this boundary unconstrained gives
    # the natural homogeneous Neumann condition:
    #
    #     normal(DField) = 0
    #
    # which behaves like an insulating/open boundary.
    pass

else:
    raise ValueError("world_bc_type must be either 'dirichlet' or 'neumann'.")

# ============================================================
# 4. Solve
# ============================================================
solve(
    type="dc",
    absolute_error=1.0,
    relative_error=1e-10,
    maximum_iterations=30,
)

vector_gradient(
    device=device,
    region=region,
    node_model="Potential",
    calc_type="default",
)

write_devices(file="paul_trap_basis_RF.dat", type="tecplot")

np.savez(
    "basis_functions_RF.npz",
    world_bc_type=np.array(world_bc_type),
    world_bc_value=np.array(world_bc_value),
    dc_voltage=np.array(DC_voltage),
    rf_voltage=np.array(RF_voltage),
    potential=np.array(
        get_node_model_values(
            device=device,
            region=region,
            name="Potential",
        )
    ),
)