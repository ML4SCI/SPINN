import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True

# ----------------------------
# Physical constants
# ----------------------------
e_charge = 1.602e-19          # electron charge [C]
amu = 1.66054e-27             # atomic mass unit [kg]
M_Ca = 40 * amu               # Calcium-40 ion mass
M = M_Ca

f_rf = 10.2e6                 # RF drive frequency [Hz]
Omega = 2 * np.pi * f_rf
scale = (e_charge**2) / (4 * M * Omega**2)
print(f"Pseudopotential scale (e^2/(4 M Omega^2)) = {scale:.3e}")

# DC electrode voltages [V]
DC_North_V = 0.0
DC_South_V = 0.0
Endcap_Top_V = 1.0
Endcap_Bot_V = 1.0
world_V = 0.0

# ----------------------------
# Load DC base functions and assemble Phi_DC
# ----------------------------
data_DC = np.load("basis_functions_DC.npz")
n_air = len(data_DC["phi_world_boundary_contact"])

def pick_air_block(ds, n_air):
    """Tecplot output is a MultiBlock with one block per DEVSIM region; we want
    the air block. Match by node count rather than hard-coded index because the
    Tecplot block ordering is alphabetical on region name and shifts with the
    number of electrodes."""
    for i in range(ds.n_blocks):
        b = ds[i]
        if b is not None and b.n_points == n_air:
            return b
    raise RuntimeError(f"No block with n_points={n_air} in Tecplot file")

mesh = pick_air_block(pv.get_reader("paul_trap_basis_DC_world_boundary_contact.dat").read(), n_air)
mesh = mesh.cell_data_to_point_data()

mesh["phi_DC_North"]    = data_DC["phi_DC_North_contact"]
mesh["phi_DC_South"]    = data_DC["phi_DC_South_contact"]
mesh["phi_Endcap_Top"]  = data_DC["phi_Endcap_Top_contact"]
mesh["phi_Endcap_Bot"]  = data_DC["phi_Endcap_Bot_contact"]
mesh["phi_world"]       = data_DC["phi_world_boundary_contact"]

# DC superposition is built later on the regular grid after resampling.

# ----------------------------
# Load RF solve and assemble pseudopotential using DEVSIM's node-projected
# ElectricField components. This is the analog of the 2D pipeline's reliance
# on `Potential_gradx`/`Potential_grady` — DEVSIM's `element_from_edge_model`
# produces ElectricField_x/y/z_onNode which are the well-behaved 3D version
# of node-based field components. (DEVSIM's `vector_gradient` is broken in 3D.)
# ----------------------------
rf_mesh = pick_air_block(pv.get_reader("paul_trap_basis_RF.dat").read(), n_air)
rf_mesh = rf_mesh.cell_data_to_point_data()

mesh["phi_rf"] = rf_mesh["Potential"]

Ex = np.asarray(rf_mesh["ElectricField_x_onNode"], dtype=np.float64)
Ey = np.asarray(rf_mesh["ElectricField_y_onNode"], dtype=np.float64)
Ez = np.asarray(rf_mesh["ElectricField_z_onNode"], dtype=np.float64)
grad2_vm2 = 1.0e4 * (Ex ** 2 + Ey ** 2 + Ez ** 2)        # (V/cm)^2 -> (V/m)^2

mesh["rf_pseudopotential_eV"] = scale * grad2_vm2 / e_charge

mesh["phi_dc_total"] = (
    DC_North_V    * np.asarray(mesh["phi_DC_North"]) +
    DC_South_V    * np.asarray(mesh["phi_DC_South"]) +
    Endcap_Top_V  * np.asarray(mesh["phi_Endcap_Top"]) +
    Endcap_Bot_V  * np.asarray(mesh["phi_Endcap_Bot"]) +
    world_V       * np.asarray(mesh["phi_world"])
)
mesh["psi_total_eV"] = np.asarray(mesh["phi_dc_total"]) + np.asarray(mesh["rf_pseudopotential_eV"])

# Compute clipping cap from the refined tet-mesh nodes.
psi_tet = np.asarray(mesh["psi_total_eV"])
cap = np.percentile(psi_tet[psi_tet > 0], 95)
print(f"psi clipping cap = {cap:.3e} eV (computed on tet mesh)")

# ----------------------------
# Resample onto a regular Cartesian grid for symmetric slicing. The gradient
# is now physically smooth (from DEVSIM), so we don't need Gaussian smoothing
# of the Potential — we just sample the already-computed psi field.
# ----------------------------
spacing = 0.005
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

# Boundary-aware Gaussian smoothing (mask-weighted blur). Treats electrode-
# interior points as missing data so smoothing doesn't pull air-side values
# toward zero near electrode boundaries.
psi_3d = np.asarray(grid["psi_total_eV"]).reshape((nx, ny, nz), order="F")
valid_3d = np.asarray(grid["vtkValidPointMask"]).astype(bool).reshape((nx, ny, nz), order="F")
mask = valid_3d.astype(np.float64)
sigma = 2.0
psi_smooth = gaussian_filter(np.where(valid_3d, psi_3d, 0.0), sigma=sigma)
mask_smooth = gaussian_filter(mask, sigma=sigma)
with np.errstate(invalid="ignore", divide="ignore"):
    psi_3d = np.where(mask_smooth > 1e-6, psi_smooth / mask_smooth, 0.0)
psi_3d = np.where(valid_3d, psi_3d, 0.0)

# Enforce the physical symmetries of the trap:
#  x -> -x   (RF_East and RF_West, mirrored)
#  y -> -y   (DC_North and DC_South, mirrored; both also set to same V)
#  z -> -z   (Endcap_Top and Endcap_Bot, mirrored; both set to same V)
# Note: 4-fold rotation around z is BROKEN here (RF at 300 V vs DC at 0 V),
# so we only apply the three mirror reflections.
psi_3d = 0.5 * (psi_3d + psi_3d[::-1, :, :])    # x mirror
psi_3d = 0.5 * (psi_3d + psi_3d[:, ::-1, :])    # y mirror
psi_3d = 0.5 * (psi_3d + psi_3d[:, :, ::-1])    # z mirror

grid["psi_total_eV"] = psi_3d.flatten(order="F")

mesh = grid

# ----------------------------
# 3D visualization: volume render + isosurfaces
# ----------------------------
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(mesh, scalars="psi_total_eV", cmap="jet", opacity=0.25, clim=(0, cap))
isos = mesh.contour(np.linspace(0.05 * cap, 0.95 * cap, 12), scalars="psi_total_eV")
p.add_mesh(isos, cmap="jet", opacity=0.7, clim=(0, cap))
p.add_axes()
p.screenshot("Psi_3D_isosurfaces.png")
p.close()

# Mid-plane slice (z = 0) showing the radial confinement quadrupole.
slice_z0 = mesh.slice(normal="z", origin=(0, 0, 0))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_z0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_xy()
# Zoom onto the trap region (blades occupy ~+/-0.10 cm).
p.camera.zoom(2.5)
p.screenshot("Psi_slice_z0.png")
p.close()

# Axial slice (y = 0) showing axial confinement from endcaps.
slice_y0 = mesh.slice(normal="y", origin=(0, 0, 0))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_y0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_xz()
p.screenshot("Psi_slice_y0.png")
p.close()

# ----------------------------
# 1D cross-sections through the trap center
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

line_plot((-0.04, 0, 0), (0.04, 0, 0),
          "psi_total along x (radial, between RF blades)",
          "Psi_xsection_x.png")
line_plot((0, -0.04, 0), (0, 0.04, 0),
          "psi_total along y (radial, between DC blades)",
          "Psi_xsection_y.png")
line_plot((0, 0, -0.7), (0, 0, 0.7),
          "psi_total along z (axial)",
          "Psi_xsection_z.png")

print("Saved PNGs.")
