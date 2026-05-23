"""Geometric constants for the classical 3D Paul trap.

Mirrors the parameters used in DEVSIM/3D_Geometries/Classical_3D_Paul_trap/gmsh_script.py
so the PINN solves the same boundary-value problem as the FEM reference.
"""
import numpy as np

# ----------------------------
# Trap geometry (cm) — classical Wolfgang Paul, r0^2 = 2 z0^2.
# ----------------------------
r0 = 0.10                       # saddle radius
z0 = r0 / np.sqrt(2.0)          # ideal axial spacing

# Ring electrode (hyperboloid of one sheet, revolved around z).
# Truncated and closed off by an outer cylindrical wall + top/bottom flat rings.
ring_z_half = 0.06
ring_r_outer = 0.22

# Endcap electrodes (hyperboloid of two sheets, revolved around z).
# Capped by a flat disc at z = +/- endcap_z_outer.
endcap_r_max = 0.10
endcap_z_outer = 0.20

# Cubic world (vacuum extent). Outer faces are grounded.
world_size = 0.6
world_half = world_size / 2.0

# Drive voltage on the ring electrode (peak RF amplitude, V).
V_RF = 300.0
