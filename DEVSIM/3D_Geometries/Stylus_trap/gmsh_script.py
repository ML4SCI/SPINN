import gmsh

gmsh.initialize()
gmsh.model.add("paul_trap_stylus")

# =========================================================
# Parameters (cm) — stylus trap inspired by Maiwald et al. (2009).
# A single central RF needle is surrounded by a grounded washer; two DC
# compensation pads sit beneath the ring for transverse DC trim. The ion
# is trapped in vacuum above the needle tip (z > 0) — the open geometry
# above z = 0 gives ~4π optical access.
# =========================================================
world_size = 0.4           # cubic world [-0.2, +0.2]^3

# Central RF stylus: thin cylinder along z, tip at z = 0.
r_stylus = 0.005           # 50 μm radius
h_stylus = 0.15            # stylus extends from z = -0.15 to z = 0 (fits in world)

# Ground washer (annular plate) around the stylus.
ring_inner_r = 0.020       # clears the stylus
ring_outer_r = 0.200
ring_z_lo = -0.025
ring_z_hi = -0.015         # 100 μm thick

# Two DC compensation pads — small squares at +x and -x.
comp_pad_side = 0.020
comp_pad_z_lo = -0.045
comp_pad_z_hi = -0.035
comp_pad_offset = 0.100    # radial offset from z-axis

world_mesh = 0.04
electrode_mesh = 0.004
trap_center_mesh = 0.004

# =========================================================
# Geometry
# =========================================================
stylus = gmsh.model.occ.addCylinder(
    0, 0, -h_stylus,
    0, 0, h_stylus,
    r_stylus
)

# Washer = outer cylinder minus inner cylinder.
ring_outer = gmsh.model.occ.addCylinder(
    0, 0, ring_z_lo,
    0, 0, ring_z_hi - ring_z_lo,
    ring_outer_r
)
ring_inner = gmsh.model.occ.addCylinder(
    0, 0, ring_z_lo - 0.005,
    0, 0, (ring_z_hi - ring_z_lo) + 0.010,
    ring_inner_r
)
ring_cut, _ = gmsh.model.occ.cut([(3, ring_outer)], [(3, ring_inner)])
ground_ring = ring_cut[0][1]

comp_e = gmsh.model.occ.addBox(
    comp_pad_offset - comp_pad_side/2, -comp_pad_side/2, comp_pad_z_lo,
    comp_pad_side, comp_pad_side, comp_pad_z_hi - comp_pad_z_lo
)
comp_w = gmsh.model.occ.addBox(
    -comp_pad_offset - comp_pad_side/2, -comp_pad_side/2, comp_pad_z_lo,
    comp_pad_side, comp_pad_side, comp_pad_z_hi - comp_pad_z_lo
)

electrodes = [
    (stylus,       "Stylus_RF"),
    (ground_ring,  "Ground_Ring"),
    (comp_e,       "DC_Comp_E"),
    (comp_w,       "DC_Comp_W"),
]

world = gmsh.model.occ.addBox(
    -world_size/2, -world_size/2, -world_size/2,
    world_size, world_size, world_size
)

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
tol = 1e-6
def on_world_face(cx, cy, cz):
    return (
        abs(cx - world_size/2) < tol or abs(cx + world_size/2) < tol or
        abs(cy - world_size/2) < tol or abs(cy + world_size/2) < tol or
        abs(cz - world_size/2) < tol or abs(cz + world_size/2) < tol
    )

air_pg = gmsh.model.addPhysicalGroup(3, [air_volume])
gmsh.model.setPhysicalName(3, air_pg, "air")

for (orig_tag, name), new_tag in zip(electrodes, electrode_new_tags):
    vol_pg = gmsh.model.addPhysicalGroup(3, [new_tag])
    gmsh.model.setPhysicalName(3, vol_pg, name)

    surf_tags = []
    for d, t in gmsh.model.getBoundary([(3, new_tag)], oriented=False):
        if d != 2:
            continue
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, t)
        if on_world_face(cx, cy, cz):
            continue
        surf_tags.append(t)
    contact_pg = gmsh.model.addPhysicalGroup(2, surf_tags)
    gmsh.model.setPhysicalName(2, contact_pg, f"{name}_contact")

world_boundary_tags = []
for d, t in gmsh.model.getBoundary([(3, air_volume)], oriented=False):
    if d != 2:
        continue
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, t)
    if on_world_face(cx, cy, cz):
        world_boundary_tags.append(t)
world_boundary_pg = gmsh.model.addPhysicalGroup(2, world_boundary_tags)
gmsh.model.setPhysicalName(2, world_boundary_pg, "world_boundary")

# =========================================================
# Mesh sizing — extremely fine near the tip and in the trap region above it.
# =========================================================
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", electrode_mesh)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", world_mesh)

electrode_points = set()
for new_tag in electrode_new_tags:
    for d, t in gmsh.model.getBoundary([(3, new_tag)], recursive=True, oriented=False):
        if d == 0:
            electrode_points.add(t)
gmsh.model.mesh.setSize([(0, p) for p in electrode_points], electrode_mesh)

# Refine the trap region just above the stylus tip.
field_box = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field_box, "VIn",  trap_center_mesh)
gmsh.model.mesh.field.setNumber(field_box, "VOut", world_mesh)
gmsh.model.mesh.field.setNumber(field_box, "XMin", -0.05)
gmsh.model.mesh.field.setNumber(field_box, "XMax",  0.05)
gmsh.model.mesh.field.setNumber(field_box, "YMin", -0.05)
gmsh.model.mesh.field.setNumber(field_box, "YMax",  0.05)
gmsh.model.mesh.field.setNumber(field_box, "ZMin",  0.0)
gmsh.model.mesh.field.setNumber(field_box, "ZMax",  0.05)
gmsh.model.mesh.field.setAsBackgroundMesh(field_box)

gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("paul_trap_stylus_mesh.msh")

gmsh.finalize()
