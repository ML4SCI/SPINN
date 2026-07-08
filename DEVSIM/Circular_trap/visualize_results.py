import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

e_charge = 1.602e-19  # electron charge [C]
amu = 1.66054e-27     # atomic mass unit [kg]
M_Ca = 40 * amu       # Calcium ion mass [kg]

M = M_Ca

f = 10.2e6  # RF frequency [Hz]
Omega = 2 * np.pi * f

scale = (e_charge**2) / (4 * M * Omega**2)
print(f"Scale value: {scale}")

North_south_contact_voltage = 0.0  # [V]


# ----------------------------
# helpers
# ----------------------------
def read_devsim_mesh(filename, block_index=4):
    mb = pv.get_reader(filename).read()
    mesh = mb[block_index]
    return mesh.cell_data_to_point_data()


def first_existing_file(filenames):
    for filename in filenames:
        if os.path.exists(filename):
            return filename
    raise FileNotFoundError(
        "Could not find any of these files:\n" + "\n".join(filenames)
    )


def npz_get(data, key, default=None):
    return data[key] if key in data.files else default


# ----------------------------
# load DC basis arrays
# ----------------------------
data_DC = np.load("basis_functions_DC.npz")

world_bc_type = str(npz_get(data_DC, "world_bc_type", "unknown"))
world_bc_value = float(npz_get(data_DC, "world_bc_value", 0.0))

print(f"World BC type: {world_bc_type}")
print(f"World BC value: {world_bc_value}")

# After the BC update, there is no longer a world-boundary basis solve
# when the world boundary is Neumann. So use a real electrode basis file
# as the reference geometry.
dc_mesh_file = first_existing_file([
    "paul_trap_basis_DC_North_contact.dat",
    "paul_trap_basis_DC_South_contact.dat",
    "paul_trap_basis_DC_East_contact.dat",
    "paul_trap_basis_DC_West_contact.dat",
    "paul_trap_basis_DC_world_boundary_contact.dat",  # fallback for old files
])

print(f"Using DC mesh file: {dc_mesh_file}")

mesh = read_devsim_mesh(dc_mesh_file)

phi_north = npz_get(data_DC, "phi_north_contact")
phi_south = npz_get(data_DC, "phi_south_contact")

if phi_north is None:
    raise KeyError("Could not find phi_north_contact in basis_functions_DC.npz")

if phi_south is None:
    raise KeyError("Could not find phi_south_contact in basis_functions_DC.npz")

mesh["phi_north_contact"] = phi_north
mesh["phi_south_contact"] = phi_south

# Old scripts may contain this key. New Neumann scripts usually will not.
phi_world = npz_get(data_DC, "phi_world_boundary_contact", None)

if phi_world is None:
    print("No phi_world_boundary_contact found. Using zero world-boundary contribution.")
    phi_world = np.zeros_like(phi_north)
else:
    mesh["phi_world_boundary_contact"] = phi_world

mesh["phi_dc_total"] = (
    North_south_contact_voltage * phi_north
    + North_south_contact_voltage * phi_south
    + phi_world
)


# ----------------------------
# separately load RF solve to get field data
# ----------------------------
rf_mesh = read_devsim_mesh("paul_trap_basis_RF.dat")

mesh["phi_rf"] = rf_mesh["Potential"]

potential_gradx = rf_mesh["Potential_gradx"]  # V/cm
potential_grady = rf_mesh["Potential_grady"]  # V/cm

grad2_vcm2 = potential_gradx**2 + potential_grady**2
grad2_vm2 = 1.0e4 * grad2_vcm2

mesh["abs_potential_grad_vcm2"] = grad2_vcm2
mesh["abs_potential_grad_vm2"] = grad2_vm2

mesh["rf_pseudopotential_J"] = scale * grad2_vm2
mesh["rf_pseudopotential_eV"] = mesh["rf_pseudopotential_J"] / e_charge

# The DC electric potential in volts corresponds to qV in eV for a singly charged ion.
mesh["psi_total_eV"] = mesh["phi_dc_total"] + mesh["rf_pseudopotential_eV"]


# ----------------------------
# plot
# ----------------------------
p = pv.Plotter()
p.add_mesh(mesh, scalars="phi_dc_total", cmap="jet")
p.view_xy()
p.show()

p = pv.Plotter()
p.add_mesh(mesh, scalars="phi_rf", cmap="jet")
p.view_xy()
p.show()

p = pv.Plotter()
p.add_mesh(mesh, scalars="rf_pseudopotential_eV", cmap="jet")
p.view_xy()
p.show()

p = pv.Plotter()
p.add_mesh(mesh, scalars="psi_total_eV", cmap="jet")
p.view_xy()
p.show()

contours = mesh.contour(isosurfaces=20, scalars="psi_total_eV")

p = pv.Plotter()
p.add_mesh(mesh, scalars="psi_total_eV", cmap="jet", opacity=0.6)
p.add_mesh(contours, color="black", line_width=2)
p.view_xy()
p.show()

xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

lo = -0.125
hi = 0.125

print(f"low: {lo}, high: {hi}")

p0 = (lo, lo, 0.0)
p1 = (hi, hi, 0.0)

line = pv.Line(p0, p1, resolution=400)

p = pv.Plotter()
p.add_mesh(mesh, scalars="psi_total_eV", cmap="jet")
p.add_mesh(line, color="red", line_width=4)
p.view_xy()
p.show()

sample = mesh.sample_over_line(p0, p1, resolution=400)

distance = sample["Distance"]
psi = sample["psi_total_eV"]

plt.figure()
plt.plot(distance, psi)
plt.xlabel("Distance along line")
plt.ylabel("psi_total_eV")
plt.title("psi_total_eV along y=x")
plt.grid(True)
plt.show()