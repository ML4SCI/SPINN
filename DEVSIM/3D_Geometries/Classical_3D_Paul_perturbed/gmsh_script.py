import gmsh
import math

gmsh.initialize()
gmsh.model.add("paul_trap_perturbed_3D")

# =========================================================
# Parameters (cm) — PERTURBED Paul trap.
# We deliberately break the ideal-Paul ratio r0^2 = 2 z0^2 so that the
# analytic harmonic lift L = V_RF * (1 - eta) / 2 is NO LONGER an exact
# solution to Laplace's equation. The PINN must learn the corrections.
# Concretely: z0 is enlarged by a factor 1.20 relative to ideal.
# =========================================================
r0 = 0.10                                   # saddle radius (unchanged)
z0_ideal = r0 / math.sqrt(2.0)
z0_factor = 1.20                            # 20% larger than ideal (BREAKS ratio)
z0 = z0_ideal * z0_factor
print(f"z0_factor = {z0_factor}, z0_ideal = {z0_ideal:.5f}, z0_used = {z0:.5f}")
print(f"r0^2 / z0^2 = {r0**2/z0**2:.4f}  (ideal Paul: 2.0)")

# How much of each hyperbolic surface to keep (truncation extents).
ring_z_half = 0.06              # ring extends to z = +/- 0.06
ring_r_outer = 0.22             # outer radial wall of ring solid

endcap_r_max = 0.10             # endcap extends radially to r = 0.10
endcap_z_outer = 0.20           # endcap caps off at |z| = 0.20

world_size = 0.6                # cubic world box [-0.3, 0.3]^3
world_mesh = 0.06
electrode_mesh = 0.005
trap_center_mesh = 0.005

n_profile = 50                  # points along each hyperbolic curve

# =========================================================
# Build the (r, z) profiles and revolve them around the z-axis.
# =========================================================
def make_ring_profile():
    """Closed region between the inner hyperbolic boundary (one sheet) and an
    outer cylindrical wall at r = ring_r_outer. Stays in x > 0 so revolution
    produces a proper torus-like solid (no axis singularity)."""
    pts = []
    for i in range(n_profile):
        z = -ring_z_half + (2.0 * ring_z_half) * i / (n_profile - 1)
        r = r0 * math.sqrt(1.0 + (z / z0) ** 2)
        pts.append(gmsh.model.occ.addPoint(r, 0, z, electrode_mesh))
    p_outer_top = gmsh.model.occ.addPoint(ring_r_outer, 0,  ring_z_half, electrode_mesh)
    p_outer_bot = gmsh.model.occ.addPoint(ring_r_outer, 0, -ring_z_half, electrode_mesh)

    inner = gmsh.model.occ.addSpline(pts)
    top   = gmsh.model.occ.addLine(pts[-1], p_outer_top)
    outer = gmsh.model.occ.addLine(p_outer_top, p_outer_bot)
    bot   = gmsh.model.occ.addLine(p_outer_bot, pts[0])

    loop = gmsh.model.occ.addCurveLoop([inner, top, outer, bot])
    face = gmsh.model.occ.addPlaneSurface([loop])
    return face

def make_endcap_profile(sign):
    """sign=+1 for top endcap, -1 for bottom. The profile has its left edge ON
    the z-axis (r=0); after revolution that edge collapses to a line on the axis
    — OCC handles this as a degenerate boundary of the resulting solid."""
    pts = []
    for i in range(n_profile):
        r = endcap_r_max * i / (n_profile - 1)
        z = sign * z0 * math.sqrt(1.0 + (r / r0) ** 2)
        pts.append(gmsh.model.occ.addPoint(r, 0, z, electrode_mesh))

    z_outer = sign * endcap_z_outer
    p_outer = gmsh.model.occ.addPoint(endcap_r_max, 0, z_outer, electrode_mesh)
    p_axis  = gmsh.model.occ.addPoint(0,            0, z_outer, electrode_mesh)

    inner    = gmsh.model.occ.addSpline(pts)
    radial   = gmsh.model.occ.addLine(pts[-1], p_outer)
    flat_top = gmsh.model.occ.addLine(p_outer, p_axis)
    axis     = gmsh.model.occ.addLine(p_axis, pts[0])

    loop = gmsh.model.occ.addCurveLoop([inner, radial, flat_top, axis])
    face = gmsh.model.occ.addPlaneSurface([loop])
    return face

ring_face       = make_ring_profile()
top_endcap_face = make_endcap_profile(+1)
bot_endcap_face = make_endcap_profile(-1)

def revolve_around_z(face_tag):
    out = gmsh.model.occ.revolve([(2, face_tag)], 0, 0, 0, 0, 0, 1, 2 * math.pi)
    return [t for d, t in out if d == 3][0]

ring_volume       = revolve_around_z(ring_face)
top_endcap_volume = revolve_around_z(top_endcap_face)
bot_endcap_volume = revolve_around_z(bot_endcap_face)

# =========================================================
# World box and fragment
# =========================================================
world = gmsh.model.occ.addBox(
    -world_size/2, -world_size/2, -world_size/2,
    world_size, world_size, world_size
)

electrodes = [
    (ring_volume,       "Ring"),
    (top_endcap_volume, "Endcap_Top"),
    (bot_endcap_volume, "Endcap_Bot"),
]
electrode_dim_tags = [(3, t) for t, _ in electrodes]
out, out_map = gmsh.model.occ.fragment([(3, world)], electrode_dim_tags)
gmsh.model.occ.synchronize()

world_frag_tags = [t for d, t in out_map[0] if d == 3]
electrode_new_tags = [out_map[i + 1][0][1] for i in range(len(electrodes))]
air_candidates = [t for t in world_frag_tags if t not in electrode_new_tags]
assert len(air_candidates) == 1, f"Expected 1 air fragment, got {len(air_candidates)}"
air_volume = air_candidates[0]

# =========================================================
# Physical groups
# =========================================================
air_pg = gmsh.model.addPhysicalGroup(3, [air_volume])
gmsh.model.setPhysicalName(3, air_pg, "air")

for (orig_tag, name), new_tag in zip(electrodes, electrode_new_tags):
    vol_pg = gmsh.model.addPhysicalGroup(3, [new_tag])
    gmsh.model.setPhysicalName(3, vol_pg, name)

    boundary = gmsh.model.getBoundary([(3, new_tag)], oriented=False)
    surf_tags = [t for d, t in boundary if d == 2]
    contact_pg = gmsh.model.addPhysicalGroup(2, surf_tags)
    gmsh.model.setPhysicalName(2, contact_pg, f"{name}_contact")

tol = 1e-6
air_boundary = gmsh.model.getBoundary([(3, air_volume)], oriented=False)
world_boundary_tags = []
for d, t in air_boundary:
    if d != 2:
        continue
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, t)
    on_outer = (
        abs(cx - world_size/2) < tol or abs(cx + world_size/2) < tol or
        abs(cy - world_size/2) < tol or abs(cy + world_size/2) < tol or
        abs(cz - world_size/2) < tol or abs(cz + world_size/2) < tol
    )
    if on_outer:
        world_boundary_tags.append(t)
world_boundary_pg = gmsh.model.addPhysicalGroup(2, world_boundary_tags)
gmsh.model.setPhysicalName(2, world_boundary_pg, "world_boundary")

# =========================================================
# Mesh sizing — fine near electrodes and in the trap center.
# =========================================================
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", electrode_mesh)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", world_mesh)

electrode_points = set()
for new_tag in electrode_new_tags:
    for d, t in gmsh.model.getBoundary([(3, new_tag)], recursive=True, oriented=False):
        if d == 0:
            electrode_points.add(t)
gmsh.model.mesh.setSize([(0, p) for p in electrode_points], electrode_mesh)

field_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", list(electrode_points))

field_thr = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field_thr, "InField", field_dist)
gmsh.model.mesh.field.setNumber(field_thr, "SizeMin", trap_center_mesh)
gmsh.model.mesh.field.setNumber(field_thr, "SizeMax", world_mesh)
gmsh.model.mesh.field.setNumber(field_thr, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(field_thr, "DistMax", 0.20)
gmsh.model.mesh.field.setAsBackgroundMesh(field_thr)

gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("paul_trap_perturbed_3D_mesh.msh")

gmsh.finalize()
