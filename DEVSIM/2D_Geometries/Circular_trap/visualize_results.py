import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

e_charge = 1.602E-19 # electron charge [C]
M_e = 9.109E-31 # electron mass [kg]
amu = 1.66054E-27 # atomic mass unit [kg]
M_Ca = 40 * amu # Calcium ion mass

M = M_Ca

f= 10.2e6 # RF frequency [Hz]
Omega = 2*np.pi*f

scale = (e_charge**2)/(4*M*Omega**2)
print(f"Scale value: {scale}")

North_contact_voltage = 0 # [V]
# RF_contact_voltage = 100 # [V]

# ----------------------------
# load basis arrays
# ----------------------------
data_DC = np.load("basis_functions_DC.npz")

# reference mesh for geometry + nodal storage
mesh = pv.get_reader("paul_trap_basis_DC_world_boundary_contact.dat").read()[4]
mesh = mesh.cell_data_to_point_data()

mesh["phi_north_contact"] = data_DC["phi_north_contact"]
mesh["phi_south_contact"] = data_DC["phi_south_contact"]
mesh["phi_world_boundary_contact"] = data_DC["phi_world_boundary_contact"]
mesh["phi_dc_total"] = North_contact_voltage*data_DC["phi_north_contact"] + North_contact_voltage*data_DC["phi_south_contact"] + data_DC["phi_world_boundary_contact"]

# ----------------------------
# separately load RF solve to get field data
# ----------------------------

rf_mesh = pv.get_reader("paul_trap_basis_RF.dat").read()[4]
rf_mesh = rf_mesh.cell_data_to_point_data()

mesh["phi_rf"] = rf_mesh["Potential"]

potential_gradx = rf_mesh["Potential_gradx"]   # V/cm
potential_grady = rf_mesh["Potential_grady"]   # V/cm

grad2_vcm2 = potential_gradx**2 + potential_grady**2
grad2_vm2 = 1.0e4 * grad2_vcm2

mesh["abs_potential_grad_vcm2"] = grad2_vcm2
mesh["abs_potential_grad_vm2"] = grad2_vm2

mesh["rf_pseudopotential_J"] = scale * grad2_vm2
mesh["rf_pseudopotential_eV"] = mesh["rf_pseudopotential_J"] / e_charge

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

print(f'low: {lo}, high: {hi}')
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

