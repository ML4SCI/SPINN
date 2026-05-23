"""Boundary-surface point clouds for hard-BC enforcement.

Two groups:
  A = ring electrode surface  (potential = V_RF)
  B = endcap surfaces + world box outer faces  (potential = 0)

We approximate the signed distance from x to a group as the minimum Euclidean
distance to that group's point cloud. This is sufficient for hard-BC enforcement
because the BCs only need to hold at points ON the boundary (where the distance
to the corresponding point cloud is exactly zero by construction) — the
approximation only matters for the smooth lift function in the interior.

Also samples interior collocation points uniformly in the air region for the
PDE residual loss.
"""
import numpy as np
from . import geometry as g


def _grid(n_total):
    """Pick (n1, n2) close to sqrt(n_total) each so that n1*n2 ~= n_total."""
    n1 = int(np.ceil(np.sqrt(n_total)))
    n2 = max(1, n_total // n1)
    return n1, n2


def sample_ring_surface(n_per_face=2500, rng=None):
    """Points on the ring electrode surfaces (V = V_RF).

    Four surface pieces:
      1. Inner hyperboloid: r = r0 * sqrt(1 + (z/z0)^2), z in [-z_half, +z_half]
      2. Outer cylinder:    r = r_outer, z in [-z_half, +z_half]
      3. Top flat ring:     z = +z_half, r in [r_inner(z_half), r_outer]
      4. Bottom flat ring:  z = -z_half, r in [r_inner(z_half), r_outer]
    """
    pieces = []

    # 1. Inner hyperboloid
    n_z, n_t = _grid(n_per_face)
    z_arr = np.linspace(-g.ring_z_half, g.ring_z_half, n_z)
    t_arr = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
    R = g.r0 * np.sqrt(1.0 + (Z / g.z0) ** 2)
    pieces.append(np.stack([R * np.cos(T), R * np.sin(T), Z], axis=-1).reshape(-1, 3))

    # 2. Outer cylinder
    n_z, n_t = _grid(n_per_face)
    z_arr = np.linspace(-g.ring_z_half, g.ring_z_half, n_z)
    t_arr = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
    pieces.append(
        np.stack([g.ring_r_outer * np.cos(T), g.ring_r_outer * np.sin(T), Z], axis=-1).reshape(-1, 3)
    )

    # 3 & 4. Flat top and bottom rings
    r_inner_top = g.r0 * np.sqrt(1.0 + (g.ring_z_half / g.z0) ** 2)
    n_r, n_t = _grid(n_per_face)
    r_arr = np.linspace(r_inner_top, g.ring_r_outer, n_r)
    t_arr = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    R, T = np.meshgrid(r_arr, t_arr, indexing="ij")
    X = (R * np.cos(T)).ravel()
    Y = (R * np.sin(T)).ravel()
    pieces.append(np.stack([X, Y, np.full_like(X, +g.ring_z_half)], axis=-1))
    pieces.append(np.stack([X, Y, np.full_like(X, -g.ring_z_half)], axis=-1))

    return np.concatenate(pieces, axis=0).astype(np.float32)


def sample_endcap_surface(sign, n_per_face=2500):
    """Points on one endcap (sign = +1 for top, -1 for bottom), V = 0.

    Three surface pieces:
      1. Inner hyperboloid: z = sign * z0 * sqrt(1 + (r/r0)^2), r in [0, r_max]
      2. Outer cylinder:    r = r_max, z in [sign*z_inner_at_rmax, sign*z_outer]
      3. Flat top/bottom:   z = sign * z_outer, r in [0, r_max]
    """
    pieces = []

    # 1. Inner hyperboloid
    n_r, n_t = _grid(n_per_face)
    r_arr = np.linspace(0.0, g.endcap_r_max, n_r)
    t_arr = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    R, T = np.meshgrid(r_arr, t_arr, indexing="ij")
    Z = sign * g.z0 * np.sqrt(1.0 + (R / g.r0) ** 2)
    pieces.append(np.stack([R * np.cos(T), R * np.sin(T), Z], axis=-1).reshape(-1, 3))

    # 2. Outer cylinder at r = endcap_r_max
    z_inner_at_rmax = sign * g.z0 * np.sqrt(1.0 + (g.endcap_r_max / g.r0) ** 2)
    z_outer = sign * g.endcap_z_outer
    z_lo, z_hi = sorted([z_inner_at_rmax, z_outer])
    n_z, n_t = _grid(n_per_face)
    z_arr = np.linspace(z_lo, z_hi, n_z)
    t_arr = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
    pieces.append(
        np.stack([g.endcap_r_max * np.cos(T), g.endcap_r_max * np.sin(T), Z], axis=-1).reshape(-1, 3)
    )

    # 3. Flat outer face (top of top endcap, bottom of bot endcap)
    n_r, n_t = _grid(n_per_face)
    r_arr = np.linspace(0.0, g.endcap_r_max, n_r)
    t_arr = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    R, T = np.meshgrid(r_arr, t_arr, indexing="ij")
    X = (R * np.cos(T)).ravel()
    Y = (R * np.sin(T)).ravel()
    pieces.append(np.stack([X, Y, np.full_like(X, z_outer)], axis=-1))

    return np.concatenate(pieces, axis=0).astype(np.float32)


def sample_world_surface(n_per_face=2500):
    """Points on the six faces of the cubic world box (V = 0)."""
    pieces = []
    n = int(np.ceil(np.sqrt(n_per_face)))
    u = np.linspace(-g.world_half, g.world_half, n)
    U, V = np.meshgrid(u, u, indexing="ij")
    U, V = U.ravel(), V.ravel()
    plane = np.full_like(U, +g.world_half)
    nplane = np.full_like(U, -g.world_half)

    pieces.append(np.stack([U, V, plane], axis=-1))    # z = +
    pieces.append(np.stack([U, V, nplane], axis=-1))   # z = -
    pieces.append(np.stack([U, plane, V], axis=-1))    # y = +
    pieces.append(np.stack([U, nplane, V], axis=-1))   # y = -
    pieces.append(np.stack([plane, U, V], axis=-1))    # x = +
    pieces.append(np.stack([nplane, U, V], axis=-1))   # x = -

    return np.concatenate(pieces, axis=0).astype(np.float32)


def _in_ring_electrode(x, y, z, eps=0.0):
    """True if (x,y,z) is inside the ring electrode body."""
    r2 = x * x + y * y
    # Ring volume: r_inner(z) <= r <= r_outer  AND  |z| <= z_half
    r_inner2 = (g.r0 ** 2) * (1.0 + (z / g.z0) ** 2)
    inside_z = np.abs(z) <= g.ring_z_half + eps
    inside_r = (r2 >= r_inner2 - eps) & (r2 <= g.ring_r_outer ** 2 + eps)
    return inside_z & inside_r


def _in_endcap_body(x, y, z, sign, eps=0.0):
    """True if (x,y,z) is inside the top (sign=+1) or bottom (sign=-1) endcap body."""
    r2 = x * x + y * y
    # Endcap volume: z is "above" the hyperboloid and below the flat cap
    z_inner = sign * g.z0 * np.sqrt(1.0 + r2 / g.r0 ** 2)
    z_outer = sign * g.endcap_z_outer
    # Need r <= r_max and z between z_inner and z_outer (sign-dependent)
    if sign > 0:
        in_z = (z >= z_inner - eps) & (z <= z_outer + eps)
    else:
        in_z = (z <= z_inner + eps) & (z >= z_outer - eps)
    in_r = r2 <= g.endcap_r_max ** 2 + eps
    return in_z & in_r


def is_in_air(points, eps=1e-4):
    """Vectorized: which of `points` (N, 3) are at least `eps` inside the air region?

    Uses `eps` both to shrink the world and to expand each electrode body. Points
    within `eps` of any boundary are rejected.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside_world = (np.abs(x) <= g.world_half - eps) & (np.abs(y) <= g.world_half - eps) & (np.abs(z) <= g.world_half - eps)
    inside_ring = _in_ring_electrode(x, y, z, eps=eps)
    inside_endcap_top = _in_endcap_body(x, y, z, +1, eps=eps)
    inside_endcap_bot = _in_endcap_body(x, y, z, -1, eps=eps)
    return inside_world & ~inside_ring & ~inside_endcap_top & ~inside_endcap_bot


def sample_air_points(n, rng, boundary_buffer=0.005):
    """Reject-sample n points uniformly in the air region.

    `boundary_buffer` (cm) excludes points within this distance of any boundary.
    Boundary-adjacent points have very large Laplacians of the lift function and
    would dominate the residual loss; excluding them gives a much better
    optimization signal in the bulk.
    """
    out = []
    n_have = 0
    while n_have < n:
        batch_size = int(2.5 * (n - n_have))
        candidates = rng.uniform(-g.world_half, g.world_half, size=(batch_size, 3)).astype(np.float32)
        keep = is_in_air(candidates, eps=boundary_buffer)
        good = candidates[keep]
        out.append(good)
        n_have += len(good)
    return np.concatenate(out, axis=0)[:n]


if __name__ == "__main__":
    # Quick sanity check.
    print("Sampling boundary point clouds...")
    pA = sample_ring_surface()
    pB = np.concatenate(
        [sample_endcap_surface(+1), sample_endcap_surface(-1), sample_world_surface()],
        axis=0,
    )
    rng = np.random.default_rng(0)
    air = sample_air_points(5000, rng)
    print(f"  Group A (ring, V=V_RF): {pA.shape[0]} points")
    print(f"  Group B (endcap+world, V=0): {pB.shape[0]} points")
    print(f"  Air collocation points: {air.shape[0]}")
    print(f"  Air bounds: x [{air[:,0].min():.3f}, {air[:,0].max():.3f}], "
          f"y [{air[:,1].min():.3f}, {air[:,1].max():.3f}], "
          f"z [{air[:,2].min():.3f}, {air[:,2].max():.3f}]")
