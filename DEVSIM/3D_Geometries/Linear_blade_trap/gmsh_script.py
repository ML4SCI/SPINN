import gmsh
import math

gmsh.initialize()
gmsh.model.add("paul_trap_linear_blade")

# =========================================================
# Parameters (cm)
# =========================================================
world_size_x = 1.0
world_size_y = 1.0
world_size_z = 2.0

r0 = 0.05               # ion-to-blade distance
blade_thickness = 0.05  # blade radial thickness
# blade_width MUST be strictly less than 2*r0 so the blade boxes do not touch at
# corner edges (otherwise adjacent contacts share mesh nodes and the gradient
# blows up locally). With r0=0.05, blade_width<0.10 leaves a small gap.
blade_width = 0.08
blade_length = 1.4      # blade extent along z

endcap_z = 0.80         # endcap z-offset from origin
endcap_thickness = 0.05
endcap_radius = 0.15

world_mesh = 0.10
electrode_mesh = 0.008
trap_center_mesh = 0.01

# =========================================================
# Geometry
# Four blades surround the z-axis. RF blades sit on +/-x,
# DC blades on +/-y. Two cylindrical endcaps at +/-z provide
# axial confinement.
# =========================================================
world = gmsh.model.occ.addBox(
    -world_size_x/2, -world_size_y/2, -world_size_z/2,
    world_size_x, world_size_y, world_size_z
)

rf_east = gmsh.model.occ.addBox(
    r0, -blade_width/2, -blade_length/2,
    blade_thickness, blade_width, blade_length
)
rf_west = gmsh.model.occ.addBox(
    -r0 - blade_thickness, -blade_width/2, -blade_length/2,
    blade_thickness, blade_width, blade_length
)
dc_north = gmsh.model.occ.addBox(
    -blade_width/2, r0, -blade_length/2,
    blade_width, blade_thickness, blade_length
)
dc_south = gmsh.model.occ.addBox(
    -blade_width/2, -r0 - blade_thickness, -blade_length/2,
    blade_width, blade_thickness, blade_length
)
endcap_top = gmsh.model.occ.addCylinder(
    0, 0, endcap_z,
    0, 0, endcap_thickness,
    endcap_radius
)
endcap_bot = gmsh.model.occ.addCylinder(
    0, 0, -endcap_z - endcap_thickness,
    0, 0, endcap_thickness,
    endcap_radius
)

electrodes = [
    (rf_east,    "RF_East"),
    (rf_west,    "RF_West"),
    (dc_north,   "DC_North"),
    (dc_south,   "DC_South"),
    (endcap_top, "Endcap_Top"),
    (endcap_bot, "Endcap_Bot"),
]

# Boolean fragment so that air and electrodes share conformal interfaces.
# Out_map[0]   -> fragments originating from world (= air piece + N electrode-shaped pieces).
# Out_map[i+1] -> single fragment originating from electrode i.
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

    # All boundary surfaces of an interior electrode are interfaces with air.
    boundary = gmsh.model.getBoundary([(3, new_tag)], oriented=False)
    surf_tags = [t for d, t in boundary if d == 2]
    contact_pg = gmsh.model.addPhysicalGroup(2, surf_tags)
    gmsh.model.setPhysicalName(2, contact_pg, f"{name}_contact")

# World outer boundary: air-volume faces whose center lies on a world-box face.
tol = 1e-6
air_boundary = gmsh.model.getBoundary([(3, air_volume)], oriented=False)
world_boundary_tags = []
for d, t in air_boundary:
    if d != 2:
        continue
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, t)
    on_outer = (
        abs(cx - world_size_x/2) < tol or abs(cx + world_size_x/2) < tol or
        abs(cy - world_size_y/2) < tol or abs(cy + world_size_y/2) < tol or
        abs(cz - world_size_z/2) < tol or abs(cz + world_size_z/2) < tol
    )
    if on_outer:
        world_boundary_tags.append(t)
world_boundary_pg = gmsh.model.addPhysicalGroup(2, world_boundary_tags)
gmsh.model.setPhysicalName(2, world_boundary_pg, "world_boundary")

# =========================================================
# Mesh sizing — refine on electrode surfaces and near the trap center.
# =========================================================
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", electrode_mesh)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", world_mesh)

electrode_points = set()
for new_tag in electrode_new_tags:
    for d, t in gmsh.model.getBoundary([(3, new_tag)], recursive=True, oriented=False):
        if d == 0:
            electrode_points.add(t)
gmsh.model.mesh.setSize([(0, p) for p in electrode_points], electrode_mesh)

# Distance-based refinement of the inter-blade region (where the ion sits).
field_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", list(electrode_points))

field_thr = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field_thr, "InField", field_dist)
gmsh.model.mesh.field.setNumber(field_thr, "SizeMin", trap_center_mesh)
gmsh.model.mesh.field.setNumber(field_thr, "SizeMax", world_mesh)
gmsh.model.mesh.field.setNumber(field_thr, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(field_thr, "DistMax", 0.30)
gmsh.model.mesh.field.setAsBackgroundMesh(field_thr)

gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("paul_trap_linear_blade_mesh.msh")

gmsh.finalize()
