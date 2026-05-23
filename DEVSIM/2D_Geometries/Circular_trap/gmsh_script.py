import gmsh
import sys
import math

gmsh.initialize()
gmsh.model.add("paul_trap_mesh")


def make_circle_with_physical_name(cx, cy, r, mesh_size, surface_name, interface_name):
    """
    Creates a circular region with a physical 2D surface and 1D boundary interface.
    
    Parameters:
        cx, cy        – center coordinates
        r             – radius
        mesh_size     – local mesh size
        surface_name  – name for the 2D physical group
        interface_name– name for the 1D physical boundary group

    Returns:
        (surface_id, interface_id)
    """

    # Define center and four points (N, E, S, W)
    c = gmsh.model.geo.addPoint(cx, cy, 0, mesh_size)
    pN = gmsh.model.geo.addPoint(cx, cy + r, 0, mesh_size)
    pE = gmsh.model.geo.addPoint(cx + r, cy, 0, mesh_size)
    pS = gmsh.model.geo.addPoint(cx, cy - r, 0, mesh_size)
    pW = gmsh.model.geo.addPoint(cx - r, cy, 0, mesh_size)

    # Define arcs
    a1 = gmsh.model.geo.addCircleArc(pN, c, pE)
    a2 = gmsh.model.geo.addCircleArc(pE, c, pS)
    a3 = gmsh.model.geo.addCircleArc(pS, c, pW)
    a4 = gmsh.model.geo.addCircleArc(pW, c, pN)

    # Build closed loop and surface
    loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # --- Physical groups ---
    # 2D surface
    surf_tag = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, surf_tag, surface_name)

    # 1D interface (the arcs)
    interface_tag = gmsh.model.addPhysicalGroup(1, [a1, a2, a3, a4])
    gmsh.model.setPhysicalName(1, interface_tag, interface_name)

    return surface, loop, surf_tag, interface_tag


#########

# Parameters
world_size_x = 2.5
world_size_y = 2.0
radius = 0.5e-1 # r=0.5 mm
offset = 1e-1 # offset=

world_mesh_size = 0.025
mesh_size = 0.0025

N_distance = 1.25e-1 # 1mm
S_distance = 1.25e-1 # 1mm
W_distance = 1.25e-1 # 1mm
E_distance = 1.25e-1 # 1mm

######################
####### WORLD ########
######################

# Clockwise starting bottom left
p1 = gmsh.model.geo.addPoint(- world_size_x/2, - world_size_y/2, 0, world_mesh_size)
p2 = gmsh.model.geo.addPoint( world_size_x/2, - world_size_y/2, 0, world_mesh_size)
p3 = gmsh.model.geo.addPoint( world_size_x/2,  world_size_y/2, 0, world_mesh_size)
p4 = gmsh.model.geo.addPoint(- world_size_x/2,  world_size_y/2, 0, world_mesh_size)

# Outer frame
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

#######################
### Bottom Solenoid ###
#######################

N_surface, N_loop, N_tag, N_interface = make_circle_with_physical_name(0, N_distance, radius, mesh_size, "North", "North_contact")

S_surface, S_loop, S_tag, S_interface = make_circle_with_physical_name(0, - S_distance, radius, mesh_size, "South", "South_contact")

W_surface, W_loop, W_tag, W_interface = make_circle_with_physical_name(- W_distance, 0, radius, mesh_size, "West", "West_contact")

E_surface, E_loop, E_tag, E_interface = make_circle_with_physical_name(E_distance, 0, radius, mesh_size, "East", "East_contact")

######################
#### ADD Surfaces ####
######################

# WORLD region
world_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
world_surface = gmsh.model.geo.addPlaneSurface([world_loop, N_loop, S_loop, W_loop, E_loop])

# Synchronize
gmsh.model.geo.synchronize()

# === Define physical regions ===
air_tag = gmsh.model.addPhysicalGroup(2, [world_surface])
gmsh.model.setPhysicalName(2, air_tag, "air") #2 corresponds to 2 dimensions

world_boundary_tag = gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4])
gmsh.model.setPhysicalName(1, world_boundary_tag, "world_boundary") #1 corresponds to 1 dimensions

# === Mesh ===
gmsh.model.mesh.generate(2) #2 corresponds to 2 dimensions

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

gmsh.write("paul_trap_mesh.msh")