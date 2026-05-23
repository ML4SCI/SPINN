import gmsh
import math

gmsh.initialize()
gmsh.model.add("paul_trap_hyperbola_mesh")


def rotate_translate(x, y, theta, cx, cy):
    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)
    return cx + xr, cy + yr


def make_hyperbola_with_physical_name(
    cx, cy,
    theta,
    a, b,
    half_height,
    mesh_size,
    surface_name,
    interface_name,
    npts=60
):
    """
    Create a closed 2D region whose front boundary is one branch of a hyperbola
    and whose back boundary is a straight line joining the two endpoints.

    Local hyperbola:
        ((x + a)^2 / a^2) - (y^2 / b^2) = 1
    or equivalently
        x(u) = a*cosh(u) - a
        y(u) = b*sinh(u)

    This makes the tip of the hyperbola occur at local (0, 0), and the branch
    opens in the +x direction.

    Parameters
    ----------
    cx, cy : float
        Location of the hyperbola tip in global coordinates.
    theta : float
        Rotation angle [radians]. 0 means pointing in +x.
    a, b : float
        Hyperbola shape parameters.
    half_height : float
        Half-height of the truncated hyperbola.
    mesh_size : float
        Local mesh size.
    surface_name : str
        Name for physical surface.
    interface_name : str
        Name for physical boundary.
    npts : int
        Number of sample points along the hyperbola branch.

    Returns
    -------
    surface, loop, surf_tag, interface_tag
    """

    # Determine parameter range so that y spans [-half_height, +half_height]
    umax = math.asinh(half_height / b)

    point_tags = []
    for i in range(npts):
        u = -umax + (2.0 * umax) * i / (npts - 1)
        x_local = a * math.cosh(u) - a
        y_local = b * math.sinh(u)
        xg, yg = rotate_translate(x_local, y_local, theta, cx, cy)
        p = gmsh.model.geo.addPoint(xg, yg, 0, mesh_size)
        point_tags.append(p)

    # Hyperbolic front face
    hyperbola_curve = gmsh.model.geo.addSpline(point_tags)

    # Straight line closing the back
    closing_line = gmsh.model.geo.addLine(point_tags[-1], point_tags[0])

    # Closed loop + surface
    loop = gmsh.model.geo.addCurveLoop([hyperbola_curve, closing_line])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Physical groups
    surf_tag = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, surf_tag, surface_name)

    interface_tag = gmsh.model.addPhysicalGroup(1, [hyperbola_curve, closing_line])
    gmsh.model.setPhysicalName(1, interface_tag, interface_name)

    return surface, loop, surf_tag, interface_tag


# =========================================================
# Parameters
# =========================================================
world_size_x = 2.5
world_size_y = 2.0

world_mesh_size = 0.025
mesh_size = 0.0025

# Hyperbola parameters
a = 0.06         # controls nose curvature / how quickly it opens
b = 0.04         # controls lateral spread
half_height = 0.09

N_distance = 0.03
S_distance = 0.03
W_distance = 0.03
E_distance = 0.03


# =========================================================
# WORLD
# =========================================================
p1 = gmsh.model.geo.addPoint(-world_size_x / 2, -world_size_y / 2, 0, world_mesh_size)
p2 = gmsh.model.geo.addPoint( world_size_x / 2, -world_size_y / 2, 0, world_mesh_size)
p3 = gmsh.model.geo.addPoint( world_size_x / 2,  world_size_y / 2, 0, world_mesh_size)
p4 = gmsh.model.geo.addPoint(-world_size_x / 2,  world_size_y / 2, 0, world_mesh_size)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)


# =========================================================
# Four inward-facing hyperbolic electrodes
# Local +x is the pointing direction
# =========================================================

# North points downward toward origin
N_surface, N_loop, N_tag, N_interface = make_hyperbola_with_physical_name(
    0, N_distance, math.pi / 2,
    a, b, half_height, mesh_size,
    "North", "North_contact"
)

# South points upward toward origin
S_surface, S_loop, S_tag, S_interface = make_hyperbola_with_physical_name(
    0, -S_distance, - math.pi / 2,
    a, b, half_height, mesh_size,
    "South", "South_contact"
)

# West points right toward origin
W_surface, W_loop, W_tag, W_interface = make_hyperbola_with_physical_name(
    -W_distance, 0, math.pi,
    a, b, half_height, mesh_size,
    "West", "West_contact"
)

# East points left toward origin
E_surface, E_loop, E_tag, E_interface = make_hyperbola_with_physical_name(
    E_distance, 0, 0.0,
    a, b, half_height, mesh_size,
    "East", "East_contact"
)


# =========================================================
# AIR / WORLD SURFACE
# =========================================================
world_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
world_surface = gmsh.model.geo.addPlaneSurface([world_loop, N_loop, S_loop, W_loop, E_loop])

gmsh.model.geo.synchronize()

air_tag = gmsh.model.addPhysicalGroup(2, [world_surface])
gmsh.model.setPhysicalName(2, air_tag, "air")

world_boundary_tag = gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4])
gmsh.model.setPhysicalName(1, world_boundary_tag, "world_boundary")


# =========================================================
# Mesh + write
# =========================================================
gmsh.model.mesh.generate(2)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("paul_trap_hyperbola_mesh.msh")

gmsh.finalize()