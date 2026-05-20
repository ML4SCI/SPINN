import gmsh
import sys
import math
import numpy as np

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
world_size_x = 3.0
world_size_y = 2.5

rect_length = 1.0e-1   # 1 mm
rect_width  = 0.25e-1   # 1 mm

world_mesh_size = 0.025
mesh_size = 0.0025

# N_distance = .75e-1
# S_distance = .75e-1
# W_distance = 1.e-1
# E_distance = 1.e-1

left_edge = -3.e-1
right_edge = -left_edge
number_rect = 5

horizontal_range = np.arange(left_edge, right_edge, (2*right_edge)/number_rect)
print(f"Horizontal range: {horizontal_range}")

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

First_surface, First_loop, First_tag, First_interface = make_rectangle_with_physical_name(
    horizontal_range[0], -1e-1, rect_length, rect_width, mesh_size, "First", "First_contact"
)

Second_surface, Second_loop, Second_tag, Second_interface = make_rectangle_with_physical_name(
    horizontal_range[1], -1e-1, rect_length, rect_width, mesh_size, "Second", "Second_contact"
)

Third_surface, Third_loop, Third_tag, Third_interface = make_rectangle_with_physical_name(
    horizontal_range[2], -1e-1, rect_length, rect_width, mesh_size, "Third", "Third_contact"
)

Fourth_surface, Fourth_loop, Fourth_tag, Fourth_interface = make_rectangle_with_physical_name(
    horizontal_range[3], -1e-1, rect_length, rect_width, mesh_size, "Fourth", "Fourth_contact"
)

Fifth_surface, Fifth_loop, Fifth_tag, Fifth_interface = make_rectangle_with_physical_name(
    horizontal_range[4], -1e-1, rect_length, rect_width, mesh_size, "Fifth", "Fifth_contact"
)

######################
#### ADD Surfaces ####
######################

world_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
world_surface = gmsh.model.geo.addPlaneSurface([world_loop, First_loop, Second_loop, Third_loop, Fourth_loop, Fifth_loop])

# ----------------------------
# Extra embedded points above rectangles
# ----------------------------
y_probe = -1e-1 + rect_width/2 + 0.5

x_probe_min = horizontal_range[0] - rect_length/2
x_probe_max = horizontal_range[-1] + rect_length/2

n_probe = 41
x_probe_vals = np.linspace(x_probe_min, x_probe_max, n_probe)

probe_points = []
for x in x_probe_vals:
    p = gmsh.model.geo.addPoint(float(x), float(y_probe), 0, mesh_size)
    probe_points.append(p)

# Synchronize
gmsh.model.geo.synchronize()

# Force those points into the air mesh
gmsh.model.mesh.embed(0, probe_points, 2, world_surface)

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