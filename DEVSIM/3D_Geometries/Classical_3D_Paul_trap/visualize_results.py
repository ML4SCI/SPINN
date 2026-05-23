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

f_rf = 10.2e6
Omega = 2 * np.pi * f_rf
scale = (e_charge**2) / (4 * M * Omega**2)
print(f"Pseudopotential scale = {scale:.3e}")

# DC voltages [V]
Endcap_Top_V = 0.0
Endcap_Bot_V = 0.0
world_V = 0.0

# ----------------------------
# Load DC basis and assemble
# ----------------------------
data_DC = np.load("basis_functions_DC.npz")
n_air = len(data_DC["phi_world_boundary_contact"])

mesh = pick_air_block(pv.get_reader("paul_trap_basis_DC_world_boundary_contact.dat").read(), n_air)
mesh = mesh.cell_data_to_point_data()

mesh["phi_Endcap_Top"] = data_DC["phi_Endcap_Top_contact"]
mesh["phi_Endcap_Bot"] = data_DC["phi_Endcap_Bot_contact"]
mesh["phi_world"]      = data_DC["phi_world_boundary_contact"]

# ----------------------------
# Load RF solve and assemble pseudopotential using DEVSIM's node-projected
# ElectricField components (produced by element_from_edge_model in the RF
# script). This is the proper 3D DEVSIM recipe.
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
    Endcap_Top_V * np.asarray(mesh["phi_Endcap_Top"]) +
    Endcap_Bot_V * np.asarray(mesh["phi_Endcap_Bot"]) +
    world_V      * np.asarray(mesh["phi_world"])
)
mesh["psi_total_eV"] = np.asarray(mesh["phi_dc_total"]) + np.asarray(mesh["rf_pseudopotential_eV"])

psi_tet = np.asarray(mesh["psi_total_eV"])
cap = np.percentile(psi_tet[psi_tet > 0], 95)
print(f"psi clipping cap = {cap:.3e} eV (computed on tet mesh)")

# ----------------------------
# Resample onto a fine regular Cartesian grid for symmetric slicing, then
# apply a boundary-aware smoother that does NOT bleed across electrode
# boundaries (would otherwise produce fuzzy electrode silhouettes).
# ----------------------------
spacing = 0.002          # 20 μm — fine enough to resolve the curved ring boundary
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

# Boundary-aware Gaussian smoothing: smooth psi*mask and the mask separately,
# then divide. This is the standard NaN-safe / mask-aware Gaussian blur trick
# (Stockham/Smith). It treats invalid neighbors as missing data rather than as
# zeros, so the electrode-boundary fuzziness disappears.
mask = valid_3d.astype(np.float64)
psi_masked = np.where(valid_3d, psi_3d, 0.0)
sigma = 2.0
psi_smooth = gaussian_filter(psi_masked, sigma=sigma)
mask_smooth = gaussian_filter(mask, sigma=sigma)
with np.errstate(invalid="ignore", divide="ignore"):
    psi_3d_smooth = np.where(mask_smooth > 1e-6, psi_smooth / mask_smooth, 0.0)
psi_3d_smooth = np.where(valid_3d, psi_3d_smooth, 0.0)

# The classical Paul trap geometry is cylindrically symmetric around z and also
# mirror-symmetric in z (Endcap_Top mirrors Endcap_Bot with equal V). Project
# psi onto (r, z), average over theta, and mirror over z to enforce the
# physical symmetry the tet mesh fails to capture.
ix = np.arange(nx).reshape(nx, 1, 1)
iy = np.arange(ny).reshape(1, ny, 1)
xc = xmin + ix * spacing
yc = ymin + iy * spacing
r_3d = np.sqrt(xc**2 + yc**2)
r_3d = np.broadcast_to(r_3d, (nx, ny, nz))

r_max = float(r_3d.max())
nr_bins = max(nx, ny) // 2 + 1
bin_idx = np.clip((r_3d / r_max * nr_bins).astype(np.int32), 0, nr_bins - 1)

psi_sym = np.zeros_like(psi_3d_smooth)
for kz in range(nz):
    sums = np.bincount(bin_idx[:, :, kz].ravel(), weights=psi_3d_smooth[:, :, kz].ravel(), minlength=nr_bins)
    counts = np.bincount(bin_idx[:, :, kz].ravel(), minlength=nr_bins)
    avg = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    psi_sym[:, :, kz] = avg[bin_idx[:, :, kz]]
# z-mirror
psi_sym = 0.5 * (psi_sym + psi_sym[:, :, ::-1])

grid["psi_total_eV"] = psi_sym.flatten(order="F")
mesh = grid

# ----------------------------
# 3D views
# ----------------------------
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
isos = mesh.contour(np.linspace(0.05 * cap, 0.95 * cap, 10), scalars="psi_total_eV")
p.add_mesh(mesh, scalars="psi_total_eV", cmap="jet", opacity=0.2, clim=(0, cap))
p.add_mesh(isos, cmap="jet", opacity=0.7, clim=(0, cap))
p.add_axes()
p.screenshot("Psi_3D_isosurfaces.png")
p.close()

slice_z0 = mesh.slice(normal="z", origin=(0, 0, 0))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_z0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_xy()
p.screenshot("Psi_slice_z0.png")
p.close()

slice_y0 = mesh.slice(normal="y", origin=(0, 0, 0))
p = pv.Plotter(off_screen=True, window_size=(1024, 768))
p.add_mesh(slice_y0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
p.add_axes()
p.view_xz()
p.screenshot("Psi_slice_y0.png")
p.close()

# ----------------------------
# 1D radial and axial cuts through the trap center
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

line_plot((-0.08, 0, 0), (0.08, 0, 0), "psi_total along x (radial)", "Psi_xsection_x.png")
line_plot((0, 0, -0.08), (0, 0, 0.08), "psi_total along z (axial)",  "Psi_xsection_z.png")

print("Saved PNGs.")
