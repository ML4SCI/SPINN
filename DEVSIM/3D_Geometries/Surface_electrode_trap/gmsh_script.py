import gmsh

gmsh.initialize()
gmsh.model.add("paul_trap_surface_electrode")

# =========================================================
# Parameters (cm) — symmetric 5-wire surface-electrode trap.
# Electrode pads sit on the chip plane (z=0). Vacuum (air) occupies
# z in [0, world_z_top]. The ion floats above the chip at z ~ a few
# hundred micrometres, determined self-consistently by the RF rails.
# =========================================================
world_size_xy = 0.40
world_z_top = 0.15

t_e = 0.005             # electrode thickness (50 μm — exaggerated for meshability)
L = 0.15                # electrode length along x

w_c  = 0.04             # center DC strip width
w_rf = 0.03             # RF rail width
w_o  = 0.05             # outer DC rail width
g    = 0.002            # gap between neighboring electrodes (20 μm)

# y-extents (north side; south side is mirrored).
y_c_top   = +w_c / 2
y_rf_in   = y_c_top + g
y_rf_out  = y_rf_in + w_rf
y_o_in    = y_rf_out + g
y_o_out   = y_o_in + w_o

world_mesh = 0.04
electrode_mesh = 0.005
trap_center_mesh = 0.004

# (name, x0, y0, dx, dy)
electrodes_spec = [
    ("DC_Center", -L, -w_c/2,  2*L, w_c),
    ("RF_North",  -L,  y_rf_in, 2*L, w_rf),
    ("RF_South",  -L, -y_rf_out, 2*L, w_rf),
    ("DC_North",  -L,  y_o_in,  2*L, w_o),
    ("DC_South",  -L, -y_o_out, 2*L, w_o),
]

# =========================================================
# Geometry
# =========================================================
world = gmsh.model.occ.addBox(
    -world_size_xy/2, -world_size_xy/2, 0.0,
    world_size_xy, world_size_xy, world_z_top
)

electrodes = []
for name, x0, y0, dx, dy in electrodes_spec:
    tag = gmsh.model.occ.addBox(x0, y0, 0.0, dx, dy, t_e)
    electrodes.append((tag, name))

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
#
# Electrodes touch the world bottom face (z=0). For each electrode we tag
# only the surfaces that interface with air (top + sides) — its bottom face
# at z=0 is not part of air's boundary, so it must be excluded.
#
# The remaining z=0 plane (the chip gaps between electrodes) becomes part
# of "world_boundary" — i.e., it's grounded. This is the standard "grounded
# gap" approximation used in BEM/FEM surface trap simulations.
# =========================================================
tol = 1e-6

def on_world_face(cx, cy, cz):
    return (
        abs(cx - world_size_xy/2) < tol or abs(cx + world_size_xy/2) < tol or
        abs(cy - world_size_xy/2) < tol or abs(cy + world_size_xy/2) < tol or
        abs(cz - world_z_top)     < tol or abs(cz - 0.0)             < tol
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
# Mesh sizing — fine on electrodes and within ~1 ion-height of the chip.
# =========================================================
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", trap_center_mesh)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", world_mesh)

electrode_points = set()
for new_tag in electrode_new_tags:
    for d, t in gmsh.model.getBoundary([(3, new_tag)], recursive=True, oriented=False):
        if d == 0:
            electrode_points.add(t)
gmsh.model.mesh.setSize([(0, p) for p in electrode_points], electrode_mesh)

# Refine the vacuum region just above the chip where the ion sits.
field_box = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field_box, "VIn",  trap_center_mesh)
gmsh.model.mesh.field.setNumber(field_box, "VOut", world_mesh)
gmsh.model.mesh.field.setNumber(field_box, "XMin", -L)
gmsh.model.mesh.field.setNumber(field_box, "XMax",  L)
gmsh.model.mesh.field.setNumber(field_box, "YMin", -y_o_out)
gmsh.model.mesh.field.setNumber(field_box, "YMax",  y_o_out)
gmsh.model.mesh.field.setNumber(field_box, "ZMin",  t_e)
gmsh.model.mesh.field.setNumber(field_box, "ZMax",  0.06)
gmsh.model.mesh.field.setAsBackgroundMesh(field_box)

gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("paul_trap_surface_electrode_mesh.msh")

gmsh.finalize()
