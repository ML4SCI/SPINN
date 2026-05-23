import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True

def pick_air_block(ds, n_air):
    for i in range(ds.n_blocks):
        b = ds[i]
        if b is not None and b.n_points == n_air:
            return b
    raise RuntimeError(f"No block with n_points={n_air} in Tecplot file")

# ----------------------------
# Physical constants
# ----------------------------
e_charge = 1.602e-19
amu = 1.66054e-27
M_Ca = 40 * amu
M = M_Ca

f_rf = 30e6                  # surface traps typically use higher RF (10-100 MHz)
Omega = 2 * np.pi * f_rf
scale = (e_charge**2) / (4 * M * Omega**2)
print(f"Pseudopotential scale = {scale:.3e}")

# DC voltages [V]
DC_Center_V = 0.0
DC_North_V  = 1.0
DC_South_V  = 1.0
world_V     = 0.0

# ----------------------------
# Load DC basis and assemble
# ----------------------------
data_DC = np.load("basis_functions_DC.npz")
n_air = len(data_DC["phi_world_boundary_contact"])

mesh = pick_air_block(pv.get_reader("paul_trap_basis_DC_world_boundary_contact.dat").read(), n_air)
mesh = mesh.cell_data_to_point_data()

mesh["phi_DC_Center"] = data_DC["phi_DC_Center_contact"]
mesh["phi_DC_North"]  = data_DC["phi_DC_North_contact"]
mesh["phi_DC_South"]  = data_DC["phi_DC_South_contact"]
mesh["phi_world"]     = data_DC["phi_world_boundary_contact"]

# ----------------------------
# Load RF solve; use DEVSIM's node-projected ElectricField (the proper 3D
# recipe — see devsim/devsim_3dmos for reference).
# ----------------------------
rf_mesh = pick_air_block(pv.get_reader("paul_trap_basis_RF.dat").read(), n_air)
rf_mesh = rf_mesh.cell_data_to_point_data()

mesh["phi_rf"] = rf_mesh["Potential"]

Ex = np.asarray(rf_mesh["ElectricField_x_onNode"], dtype=np.float64)
Ey = np.asarray(rf_mesh["ElectricField_y_onNode"], dtype=np.float64)
Ez = np.asarray(rf_mesh["ElectricField_z_onNode"], dtype=np.float64)
grad2_vm2 = 1.0e4 * (Ex ** 2 + Ey ** 2 + Ez ** 2)
mesh["rf_pseudopotential_eV"] = scale * grad2_vm2 / e_charge

mesh["phi_dc_total"] = (
    DC_Center_V * np.asarray(mesh["phi_DC_Center"]) +
    DC_North_V  * np.asarray(mesh["phi_DC_North"]) +
    DC_South_V  * np.asarray(mesh["phi_DC_South"]) +
    world_V     * np.asarray(mesh["phi_world"])
)
mesh["psi_total_eV"] = np.asarray(mesh["phi_dc_total"]) + np.asarray(mesh["rf_pseudopotential_eV"])

psi_tet = np.asarray(mesh["psi_total_eV"])
cap = np.percentile(psi_tet[psi_tet > 0], 95)
print(f"psi clipping cap = {cap:.3e} eV (computed on tet mesh)")

# ----------------------------
# Resample onto regular grid + mild Gaussian smooth for clean 1D cuts.
# ----------------------------
spacing = 0.003
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
nx = max(2, int(round((xmax - xmin) / spacing)) + 1)
ny = max(2, int(round((ymax - ymin) / spacing)) + 1)
nz = max(2, int(round((zmax - zmin) / spacing)) + 1)
print(f"Regular grid: {nx} x {ny} x {nz} = {nx*ny*nz} points, spacing={spacing} cm")
grid = pv.ImageData(
    dimensions=(nx, ny, nz),
    spacing=(spacing, spacing, spacing),
    origin=(xmin, ymin, zmin),
)
grid = grid.sample(mesh)

psi_3d = np.asarray(grid["psi_total_eV"]).reshape((nx, ny, nz), order="F")
valid_3d = np.asarray(grid["vtkValidPointMask"]).astype(bool).reshape((nx, ny, nz), order="F")
mask = valid_3d.astype(np.float64)
sigma = 2.0
psi_smooth = gaussian_filter(np.where(valid_3d, psi_3d, 0.0), sigma=sigma)
mask_smooth = gaussian_filter(mask, sigma=sigma)
with np.errstate(invalid="ignore", divide="ignore"):
    psi_3d = np.where(mask_smooth > 1e-6, psi_smooth / mask_smooth, 0.0)
psi_3d = np.where(valid_3d, psi_3d, 0.0)

# Enforce the physical symmetries of the trap: x -> -x mirror (electrodes are
# symmetric along the trap axis) and y -> -y mirror (RF rails and outer DC
# rails are mirrored north/south, and the visualization sets DC_North_V ==
# DC_South_V). Mesh-induced asymmetry in the tet discretization is purely
# numerical and would falsely break these.
psi_3d = 0.5 * (psi_3d + psi_3d[::-1, :, :])    # x mirror
psi_3d = 0.5 * (psi_3d + psi_3d[:, ::-1, :])    # y mirror

grid["psi_total_eV"] = psi_3d.flatten(order="F")
mesh = grid

p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(mesh, scalars="psi_total_eV", cmap="jet", opacity=0.25, clim=(0, cap))
isos = mesh.contour(np.linspace(0.1 * cap, 0.9 * cap, 8), scalars="psi_total_eV")
p.add_mesh(isos, cmap="jet", opacity=0.8, clim=(0, cap))
p.add_axes()
p.screenshot("Psi_3D_isosurfaces.png")
p.close()

slice_x0 = mesh.slice(normal="x", origin=(0, 0, 0))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_x0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_yz()
p.screenshot("Psi_slice_x0.png")
p.close()

slice_y0 = mesh.slice(normal="y", origin=(0, 0, 0))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_y0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_xz()
p.screenshot("Psi_slice_y0.png")
p.close()

# ----------------------------
# 1D cuts: find the ion height along z-axis (minimum of psi at x=0, y=0)
# ----------------------------
def line_plot(p0, p1, title, filename):
    sample = mesh.sample_over_line(p0, p1, resolution=400)
    plt.figure()
    plt.plot(sample["Distance"], sample["psi_total_eV"])
    plt.xlabel("Distance along line [cm]")
    plt.ylabel("psi_total [eV]")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()

line_plot((0, 0, 0.006), (0, 0, 0.14),
          "psi_total along z (vertical above chip center)",
          "Psi_xsection_z.png")
line_plot((0, -0.10, 0.02), (0, 0.10, 0.02),
          "psi_total along y at z=0.02",
          "Psi_xsection_y.png")
line_plot((-0.14, 0, 0.02), (0.14, 0, 0.02),
          "psi_total along x at z=0.02",
          "Psi_xsection_x.png")

print("Saved PNGs.")
