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

f_rf = 30e6
Omega = 2 * np.pi * f_rf
scale = (e_charge**2) / (4 * M * Omega**2)
print(f"Pseudopotential scale = {scale:.3e}")

# DC voltages [V]
Ground_Ring_V = 0.0
DC_Comp_E_V   = 0.0
DC_Comp_W_V   = 0.0
world_V       = 0.0

# ----------------------------
# Load DC basis and assemble
# ----------------------------
data_DC = np.load("basis_functions_DC.npz")
n_air = len(data_DC["phi_world_boundary_contact"])

mesh = pick_air_block(pv.get_reader("paul_trap_basis_DC_world_boundary_contact.dat").read(), n_air)
mesh = mesh.cell_data_to_point_data()

mesh["phi_Ground_Ring"] = data_DC["phi_Ground_Ring_contact"]
mesh["phi_DC_Comp_E"]   = data_DC["phi_DC_Comp_E_contact"]
mesh["phi_DC_Comp_W"]   = data_DC["phi_DC_Comp_W_contact"]
mesh["phi_world"]       = data_DC["phi_world_boundary_contact"]

# ----------------------------
# Load RF solve; use DEVSIM's node-projected ElectricField (proper 3D recipe).
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
    Ground_Ring_V * np.asarray(mesh["phi_Ground_Ring"]) +
    DC_Comp_E_V   * np.asarray(mesh["phi_DC_Comp_E"]) +
    DC_Comp_W_V   * np.asarray(mesh["phi_DC_Comp_W"]) +
    world_V       * np.asarray(mesh["phi_world"])
)
mesh["psi_total_eV"] = np.asarray(mesh["phi_dc_total"]) + np.asarray(mesh["rf_pseudopotential_eV"])

# Cap from a trap-zone window above the tip (pseudopotential is huge near the
# stylus surface and would saturate a whole-volume percentile).
psi_tet = np.asarray(mesh["psi_total_eV"])
coords_tet = np.asarray(mesh.points)
trap_zone_tet = (coords_tet[:, 2] > 0.005) & (coords_tet[:, 2] < 0.05) & \
                (np.abs(coords_tet[:, 0]) < 0.04) & (np.abs(coords_tet[:, 1]) < 0.04)
psi_trap_tet = psi_tet[trap_zone_tet & (psi_tet > 0)]
cap = np.percentile(psi_trap_tet, 99) if psi_trap_tet.size else np.percentile(psi_tet[psi_tet > 0], 90)
print(f"psi clipping cap = {cap:.3e} eV (trap-zone p99)")

# ----------------------------
# Resample onto regular grid + mild Gaussian smooth.
# ----------------------------
spacing = 0.002
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

# Stylus is cylindrically symmetric around the z-axis (single central RF needle
# + axisymmetric ground washer; with comp pads at equal V the system has full
# z-axis cylindrical symmetry). Project onto (r, z) and average over theta to
# remove mesh-induced azimuthal asymmetry.
ix = np.arange(nx).reshape(nx, 1, 1)
iy = np.arange(ny).reshape(1, ny, 1)
xc = xmin + ix * spacing
yc = ymin + iy * spacing
r_3d = np.sqrt(xc**2 + yc**2)            # shape (nx, ny, 1) broadcasts to (nx, ny, nz)
r_3d = np.broadcast_to(r_3d, (nx, ny, nz))

# Radial binning: average psi over annular rings at each z layer.
r_max = float(r_3d.max())
nr_bins = max(nx, ny) // 2 + 1
bin_idx = np.clip((r_3d / r_max * nr_bins).astype(np.int32), 0, nr_bins - 1)

psi_sym = np.zeros_like(psi_3d)
for kz in range(nz):
    sums = np.bincount(bin_idx[:, :, kz].ravel(), weights=psi_3d[:, :, kz].ravel(), minlength=nr_bins)
    counts = np.bincount(bin_idx[:, :, kz].ravel(), minlength=nr_bins)
    avg = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    psi_sym[:, :, kz] = avg[bin_idx[:, :, kz]]
psi_3d = psi_sym

grid["psi_total_eV"] = psi_3d.flatten(order="F")
mesh = grid

p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(mesh, scalars="psi_total_eV", cmap="jet", opacity=0.2, clim=(0, cap))
isos = mesh.contour(np.linspace(0.05 * cap, 0.9 * cap, 10), scalars="psi_total_eV")
p.add_mesh(isos, cmap="jet", opacity=0.8, clim=(0, cap))
p.add_axes()
p.screenshot("Psi_3D_isosurfaces.png")
p.close()

slice_y0 = mesh.slice(normal="y", origin=(0, 0, 0))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_y0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_xz()
# Zoom onto the stylus tip region (the open trap volume is right above the tip).
p.camera.focal_point = (0, 0, 0.0)
p.camera.position = (0, -0.4, 0.0)
p.camera.zoom(3.0)
p.screenshot("Psi_slice_y0.png")
p.close()

h_trap = 0.02
slice_h = mesh.slice(normal="z", origin=(0, 0, h_trap))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_h, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_xy()
p.screenshot("Psi_slice_z_h_trap.png")
p.close()

# ----------------------------
# 1D cuts to locate the ion height
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

line_plot((0, 0, 0.002), (0, 0, 0.10),
          "psi_total along z-axis (above stylus tip)",
          "Psi_xsection_z.png")
line_plot((-0.05, 0, h_trap), (0.05, 0, h_trap),
          f"psi_total along x at z={h_trap}",
          "Psi_xsection_x.png")
line_plot((0, -0.05, h_trap), (0, 0.05, h_trap),
          f"psi_total along y at z={h_trap}",
          "Psi_xsection_y.png")

print("Saved PNGs.")
