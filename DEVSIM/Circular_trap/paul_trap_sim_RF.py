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
add_gmsh_contact(mesh=mesh, gmsh_name="North_contact",  region="air", material="metal", name="North_contact")
add_gmsh_contact(mesh=mesh, gmsh_name="South_contact", region="air", material="metal", name="South_contact")
add_gmsh_contact(mesh=mesh, gmsh_name="East_contact",  region="air", material="metal", name="East_contact")
add_gmsh_contact(mesh=mesh, gmsh_name="West_contact", region="air", material="metal", name="West_contact")

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

DC_contacts = ["world_boundary_contact", "North_contact", "South_contact"]
RF_contacts = ["East_contact", "West_contact"]

DC_voltage = 0.0
RF_votlage = 300.0

for c in DC_contacts:
    set_parameter(device=device, name=f"{c}_bias", value=DC_voltage)

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

for c in RF_contacts:
    set_parameter(device=device, name=f"{c}_bias", value=RF_votlage)

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

solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)

vector_gradient(device=device, region=region, node_model="Potential", calc_type="default")

write_devices(file=f"paul_trap_basis_RF.dat", type="tecplot")

