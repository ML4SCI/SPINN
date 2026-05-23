from devsim import *
import numpy as np

device = "paul_trap"
mesh = "paul_trap_mesh"
region = "air"

create_gmsh_mesh(file="paul_trap_perturbed_3D_mesh.msh", mesh=mesh)

ELECTRODES = ["Ring", "Endcap_Top", "Endcap_Bot"]

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

DC_contacts = ["world_boundary_contact", "Endcap_Top_contact", "Endcap_Bot_contact"]
RF_contacts = ["Ring_contact"]

DC_voltage = 0.0
RF_voltage = 300.0   # peak RF amplitude [V]

for c in DC_contacts:
    set_parameter(device=device, name=f"{c}_bias", value=DC_voltage)
    contact_node_model(device=device, contact=c, name=f"{c}_bc",
                       equation=f"Potential - {c}_bias")
    contact_node_model(device=device, contact=c, name=f"{c}_bc:Potential", equation="1")
    contact_equation(device=device, contact=c, name="PotentialEquation",
                     node_model=f"{c}_bc", edge_charge_model="DField")

for c in RF_contacts:
    set_parameter(device=device, name=f"{c}_bias", value=RF_voltage)
    contact_node_model(device=device, contact=c, name=f"{c}_bc",
                       equation=f"Potential - {c}_bias")
    contact_node_model(device=device, contact=c, name=f"{c}_bc:Potential", equation="1")
    contact_equation(device=device, contact=c, name="PotentialEquation",
                     node_model=f"{c}_bc", edge_charge_model="DField")

solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)

# Proper 3D DEVSIM recipe: project edge-based ElectricField onto element-based
# ElectricField_x/y/z. The written Tecplot then contains node-projected
# ElectricField_*_onNode variants for visualization.
# (vector_gradient is broken on 3D tet meshes; see devsim/devsim_3dmos for ref.)
element_from_edge_model(edge_model="ElectricField", device=device, region=region)

write_devices(file="paul_trap_basis_RF.dat", type="tecplot")
