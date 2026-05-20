import gmsh
import sys
import math

gmsh.initialize()
gmsh.model.add("paul_trap_mesh")


def make_rectangle_with_physical_name(cx, cy, length, width, mesh_size, surface_name, interface_name):
    """
    Creates a rectangular region with a physical 2D surface and 1D boundary interface.

    Parameters:
        cx, cy         : center coordinates
        length         : rectangle length in x-direction
        width          : rectangle width in y-direction
        mesh_size      : local mesh size
        surface_name   : name for the 2D physical group
        interface_name : name for the 1D physical boundary group

    Returns:
        (surface_id, loop_id, surf_tag, interface_tag)
    """

    hx = length / 2.0
    hy = width / 2.0

    # Rectangle corner points (clockwise starting bottom-left)
    p1 = gmsh.model.geo.addPoint(cx - hx, cy - hy, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(cx + hx, cy - hy, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(cx + hx, cy + hy, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(cx - hx, cy + hy, 0, mesh_size)

    # Rectangle edges
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Closed loop and surface
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Physical groups
    surf_tag = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, surf_tag, surface_name)

    interface_tag = gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4])
    gmsh.model.setPhysicalName(1, interface_tag, interface_name)

    return surface, loop, surf_tag, interface_tag


#########

# Parameters
world_size_x = 2.0
world_size_y = 1.5

rect_length = 1.0e-1   # 1 mm
rect_width  = 0.25e-1   # 1 mm

world_mesh_size = 0.025
mesh_size = 0.0025

N_distance = .75e-1
S_distance = .75e-1
W_distance = 1.e-1
E_distance = 1.e-1

######################
####### WORLD ########
######################

# Clockwise starting bottom left
p1 = gmsh.model.geo.addPoint(-world_size_x / 2, -world_size_y / 2, 0, world_mesh_size)
p2 = gmsh.model.geo.addPoint( world_size_x / 2, -world_size_y / 2, 0, world_mesh_size)
p3 = gmsh.model.geo.addPoint( world_size_x / 2,  world_size_y / 2, 0, world_mesh_size)
p4 = gmsh.model.geo.addPoint(-world_size_x / 2,  world_size_y / 2, 0, world_mesh_size)

# Outer frame
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

#######################
##### RECTANGLES ######
#######################

N_surface, N_loop, N_tag, N_interface = make_rectangle_with_physical_name(
    -W_distance, N_distance, rect_length, rect_width, mesh_size, "Northwest", "Northwest_contact"
)

S_surface, S_loop, S_tag, S_interface = make_rectangle_with_physical_name(
    -W_distance, -S_distance, rect_length, rect_width, mesh_size, "Southwest", "Southwest_contact"
)

W_surface, W_loop, W_tag, W_interface = make_rectangle_with_physical_name(
    E_distance, N_distance, rect_length, rect_width, mesh_size, "Northeast", "Northeast_contact"
)

E_surface, E_loop, E_tag, E_interface = make_rectangle_with_physical_name(
    E_distance, -S_distance, rect_length, rect_width, mesh_size, "Southeast", "Southeast_contact"
)

######################
#### ADD Surfaces ####
######################

world_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
world_surface = gmsh.model.geo.addPlaneSurface([world_loop, N_loop, S_loop, W_loop, E_loop])

# Synchronize
gmsh.model.geo.synchronize()

# Physical regions
air_tag = gmsh.model.addPhysicalGroup(2, [world_surface])
gmsh.model.setPhysicalName(2, air_tag, "air")

world_boundary_tag = gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4])
gmsh.model.setPhysicalName(1, world_boundary_tag, "world_boundary")

# Mesh
gmsh.model.mesh.generate(2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

gmsh.write("paul_trap_mesh.msh")

gmsh.finalize()