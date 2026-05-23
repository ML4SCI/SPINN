"""SIREN MLP wrapped with hard Dirichlet BC enforcement.

The output of the wrapped model `phi(x)` satisfies, by construction:
    phi(x) = V_RF on every Group-A (ring) boundary point
    phi(x) = 0   on every Group-B (endcap + world) boundary point

Construction (Shepard-style two-group hard BC):
    d_A(x) = min distance from x to Group A point cloud  (≈ SDF to RF surface)
    d_B(x) = min distance from x to Group B point cloud  (≈ SDF to ground surface)

    w_A(x) = d_B / (d_A + d_B + eps)
    w_B(x) = d_A / (d_A + d_B + eps)
    L(x)   = V_RF * w_A(x) + 0 * w_B(x)         # smooth lift satisfying BCs
    B(x)   = d_A * d_B / (d_A + d_B + eps)      # vanishes on BOTH boundary groups
    phi(x) = L(x) + B(x) * NN(x)

On a Group-A point  -> d_A = 0  -> L = V_RF, B = 0  -> phi = V_RF.
On a Group-B point  -> d_B = 0  -> L = 0,    B = 0  -> phi = 0.
The PINN therefore only has to learn the bulk Laplace residual; the
boundary conditions are exact for any NN weights.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn

from . import geometry as g


# ----------------------------
# SIREN layer (Sitzmann et al., 2020) — sin activation, careful init.
# ----------------------------
def _siren_kernel_init(in_dim, w0, is_first):
    """Uniform init that keeps activations ~ N(0, 1) after sin(w0 * z)."""
    if is_first:
        bound = 1.0 / in_dim
    else:
        bound = jnp.sqrt(6.0 / in_dim) / w0

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


class SIREN(nn.Module):
    hidden: tuple[int, ...] = (128, 128, 128, 128)
    w0: float = 30.0
    w0_first: float = 30.0
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        # First layer
        x = nn.Dense(
            self.hidden[0],
            kernel_init=_siren_kernel_init(in_dim, self.w0_first, is_first=True),
        )(x)
        x = jnp.sin(self.w0_first * x)
        # Hidden layers
        for n_out in self.hidden[1:]:
            in_dim_h = x.shape[-1]
            x = nn.Dense(
                n_out,
                kernel_init=_siren_kernel_init(in_dim_h, self.w0, is_first=False),
            )(x)
            x = jnp.sin(self.w0 * x)
        # Output: scalar (default) or vector
        x = nn.Dense(
            self.output_dim,
            kernel_init=_siren_kernel_init(x.shape[-1], self.w0, is_first=False),
        )(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


# ----------------------------
# Hard-BC wrapper
# ----------------------------
@dataclasses.dataclass
class HardBC:
    """Holds boundary point clouds; offers d_A, d_B, lift, bump, and phi().

    Solves for the NORMALIZED potential (V_ring = 1, V_ground = 0). Multiply by
    V_RF after the fact to get the physical RF amplitude. Working at unit scale
    keeps the loss at O(1/L^2) instead of O(V_RF^2/L^2) and the NN outputs at
    O(1/B(x)) instead of O(V_RF/B(x)) — both much friendlier for optimization.
    """
    pA: jnp.ndarray     # (Na, 3) — ring surface (V = 1, will be scaled by V_RF later)
    pB: jnp.ndarray     # (Nb, 3) — endcap + world surface (V = 0)
    eps: float = 1e-6

    def d_A(self, x):
        # x: (3,)
        return jnp.min(jnp.linalg.norm(self.pA - x, axis=-1))

    def d_B(self, x):
        return jnp.min(jnp.linalg.norm(self.pB - x, axis=-1))

    def phi(self, nn_apply, params, x, x_normalizer):
        """Normalized wrapped potential at single point x (shape (3,)).

        Returns a scalar in [0, 1] on boundaries (exactly):
          on ring   (d_A=0): 1
          on ground (d_B=0): 0
        Caller multiplies by V_RF to get the physical potential.
        """
        dA = self.d_A(x)
        dB = self.d_B(x)
        denom = dA + dB + self.eps
        L = dB / denom                                  # lift, in [0, 1]
        B = dA * dB / denom                             # bump, vanishes on every boundary
        x_norm = x_normalizer(x)
        nn_out = nn_apply(params, x_norm[None, :])[0]
        return L + B * nn_out


@dataclasses.dataclass
class AnalyticHyperbolicBC:
    """Hard-BC wrapper using the trap's natural hyperbolic coordinate.

    Define eta(x) = z^2/z0^2 - r^2/r0^2  with r = sqrt(x^2 + y^2). Then:
      ring inner hyperboloid:    eta = -1     (BC: phi = 1, normalized)
      endcap inner hyperboloid:  eta = +1     (BC: phi = 0)
    For the IDEAL Paul ratio r0^2 = 2*z0^2, eta is harmonic (Laplace eta = 0),
    so the linear lift L = (1 - eta)/2 is exactly the analytic ideal-Paul
    solution — the PINN starts at zero PDE residual for the ideal geometry.

    The bump (1 - eta^2)/2 vanishes on both hyperbolic surfaces, so any NN
    output is silenced on the electrode hyperboloids. Other parts of the
    electrode (flat ring faces, outer cylinder, endcap caps) and the world
    box are handled by a SOFT-BC loss term in the training script.
    """
    r0: float
    z0: float

    def eta(self, x):
        r2 = x[0] ** 2 + x[1] ** 2
        return x[2] ** 2 / (self.z0 ** 2) - r2 / (self.r0 ** 2)

    def phi(self, nn_apply, params, x, x_normalizer):
        e = self.eta(x)
        L = (1.0 - e) / 2.0               # = 1 on ring (eta=-1), 0 on endcap (eta=+1)
        # Bump vanishes on |eta|=1 (electrode hyperboloids) and is bounded.
        # Without clamping, |eta| can reach ~18 at the cubic world corners,
        # which would make (1 - eta^2)/2 ~ -160 and amplify NN noise into
        # field values of thousands of volts there. Clamp keeps the bump
        # bounded in [-1, 1] so the NN can't blow up the field anywhere.
        B_raw = (1.0 - e ** 2) / 2.0
        B = jnp.clip(B_raw, -1.0, 1.0)
        x_norm = x_normalizer(x)
        nn_out = nn_apply(params, x_norm[None, :])[0]
        return L + B * nn_out


class TanhMLP(nn.Module):
    """Plain tanh MLP — the 'vanilla PINN' architecture (PyTorch port).

    Maps a 3D point to a scalar (normalized) potential. No analytic lift, no
    hard-BC factoring — the network output IS phi directly. Boundary conditions
    must be enforced as soft penalty terms in the loss. This mirrors
    PINN/Vanilla_Paul_Trap_PINN/Paul_Trap_PINN.py's MLP, ported to Flax for an
    apples-to-apples comparison against the SIREN / analytic-lift variants.
    """
    hidden: tuple[int, ...] = (128, 128, 128, 128)

    @nn.compact
    def __call__(self, x):
        # x: (3,) single point — normalized to ~[-1, 1] by the caller
        for n_out in self.hidden:
            x = nn.Dense(n_out)(x)
            x = jnp.tanh(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


@dataclasses.dataclass
class VanillaSoftBC:
    """Wrapper for the vanilla soft-BC PINN: phi(x) = NN(x_normalized).

    No lift, no bump. The network output is the (normalized) potential directly.
    All boundary conditions are enforced softly in the training loss. Provided
    so the rest of the eval tooling (validate_rf, visualize_rf, trap_quality)
    can treat it with the same `phi(nn_apply, params, x, normalizer)` interface
    as the hard-BC variants.
    """
    def phi(self, nn_apply, params, x, x_normalizer):
        x_norm = x_normalizer(x)
        return nn_apply(params, x_norm[None, :])[0]


class IsotopicTrainableShape(nn.Module):
    """Isotopic PINN with TRAINABLE lift shape parameters (r0, z0).

    Where IsotopicHyperbolicBC treats r0 and z0 as fixed constants of the lift,
    this module makes them flax parameters that the optimizer updates alongside
    the SIREN weights. They are log-parameterized to guarantee positivity:
        r0 = exp(log_r0_param)
        z0 = exp(log_z0_param)

    Lift / bump / displacement structure is the same as IsotopicHyperbolicBC:

        phi(x) = (1 - eta(x + delta(x))) / 2
        eta(x) = z^2 / z0^2 - (x^2 + y^2) / r0^2
        delta(x) = delta_scale * tanh(NN_3D(x)) * clip((1 - eta^2)/2, -1, 1)

    With r0, z0 now in the param dict, the BC loss has a direct gradient
    pathway to move the lift's hyperboloid surface globally (rather than just
    locally via the bounded displacement). This is the architecture change
    required to enable actual SHAPE DISCOVERY.
    """
    r0_init: float = 0.10
    z0_init: float = 0.0707
    delta_scale: float = 0.02
    siren_hidden: tuple = (128, 128, 128, 128)
    w0: float = 15.0
    world_half: float = 0.30

    @nn.compact
    def __call__(self, x):
        # x: (3,) — single 3D point
        log_r0 = self.param("log_r0",
                            lambda key: jnp.array(jnp.log(self.r0_init), dtype=jnp.float32))
        log_z0 = self.param("log_z0",
                            lambda key: jnp.array(jnp.log(self.z0_init), dtype=jnp.float32))
        r0 = jnp.exp(log_r0)
        z0 = jnp.exp(log_z0)

        # eta with trainable r0, z0
        r2 = x[0] ** 2 + x[1] ** 2
        eta = x[2] ** 2 / (z0 ** 2) - r2 / (r0 ** 2)
        B = jnp.clip((1.0 - eta ** 2) / 2.0, -1.0, 1.0)

        # SIREN displacement field
        x_norm = x / self.world_half
        siren_out = SIREN(
            hidden=self.siren_hidden,
            w0=self.w0,
            w0_first=self.w0,
            output_dim=3,
        )(x_norm[None, :])[0]
        delta = self.delta_scale * jnp.tanh(siren_out) * B

        # Evaluate analytic lift at the deformed coord
        x_def = x + delta
        r2_def = x_def[0] ** 2 + x_def[1] ** 2
        eta_def = x_def[2] ** 2 / (z0 ** 2) - r2_def / (r0 ** 2)
        return (1.0 - eta_def) / 2.0


class IsotopicSeededHoles(nn.Module):
    """Isotopic shape-discovery PINN with NO hyperbolic prior.

    Instead of baking in the analytic Paul form eta = z^2/z0^2 - r^2/r0^2, the
    electrodes are defined as the level surfaces of simple analytic "holes"
    seeded at the electrode locations, which a topology-preserving diffeomorphism
    then deforms into the discovered electrode shapes:

      - Ring (phi = 1): a torus of revolution about z, waist at r = ring_R,
        tube radius ring_a, sitting at the equator (z = 0).
      - Endcaps (phi = 0): two axial blobs centered at z = +/- cap_d, radius cap_a.

    Hard BCs are enforced on the seeds' (deformed) surfaces via the Shepard
    two-group construction (same as HardBC, but with smooth analytic signed
    distances instead of point clouds):

        d_A = |s_ring(X)|,   d_B = softmin(|s_cap+|, |s_cap-|)
        L    = d_B / (d_A + d_B)              # 1 on ring, 0 on endcaps
        bump = d_A d_B / (d_A + d_B)          # vanishes on both surfaces
        phi  = L + bump * NN_bulk(x)

    where X = x + delta(x) is the deformed coordinate. Because the seeds are
    composed with a diffeomorphism, the level-set TOPOLOGY (one ring hole + two
    endcap holes) is preserved while the GEOMETRY is free — this relaxes the
    shape prior to a topological + symmetry prior only.

    Reflection symmetry across the x, y, z planes is enforced STRUCTURALLY: the
    SIREN sees only even features (x/H)^2, (y/H)^2, (z/H)^2, and each displacement
    component is given the correct parity by multiplying an even scalar by its own
    axis coordinate:
        delta_x = ds * tanh(a) * (x/H)   (odd in x, even in y,z)
        delta_y = ds * tanh(b) * (y/H)   (odd in y, even in x,z)
        delta_z = ds * tanh(c) * (z/H)   (odd in z, even in x,y)
    so delta(Rx) = R delta(x) for every reflection R, and the seeded-symmetric
    holes deform into shapes with the same symmetry. The bulk correction reuses a
    4th even SIREN output, making the entire field phi(Rx) = phi(x) by
    construction.

    Seeds are FIXED (deform-only): ring_R, ring_a, cap_d, cap_a are constants.
    """
    ring_R: float = 0.10
    ring_a: float = 0.03
    cap_d: float = 0.0707
    cap_a: float = 0.04
    delta_scale: float = 0.10
    siren_hidden: tuple = (128, 128, 128, 128)
    w0: float = 15.0
    world_half: float = 0.30
    eps: float = 1e-8           # softabs floor (~1um); keeps hard BC ~0.1% tight

    @nn.compact
    def __call__(self, x, return_aux=False):
        H = self.world_half
        # Even features -> network cannot distinguish +/- on any axis.
        u = jnp.stack([(x[0] / H) ** 2, (x[1] / H) ** 2, (x[2] / H) ** 2])
        out = SIREN(
            hidden=self.siren_hidden,
            w0=self.w0,
            w0_first=self.w0,
            output_dim=4,
        )(u[None, :])[0]

        # Parity-correct displacement: delta_i odd in axis i, even in the others.
        dx = self.delta_scale * jnp.tanh(out[0]) * (x[0] / H)
        dy = self.delta_scale * jnp.tanh(out[1]) * (x[1] / H)
        dz = self.delta_scale * jnp.tanh(out[2]) * (x[2] / H)
        X = x + jnp.stack([dx, dy, dz])
        bulk = out[3]                            # even scalar -> symmetric field

        # Smooth signed distances to the seeded surfaces (at deformed coord).
        def softabs(s):
            return jnp.sqrt(s * s + self.eps)

        # Ring torus distance via a SMOOTH polynomial implicit (no on-axis cone).
        # F = 0 on the torus surface; F and |grad F|^2 are polynomials in X, so
        # they are smooth on the z-axis (unlike sqrt(x^2+y^2), whose 1/rho second
        # derivative spikes the Laplacian on axis). Gradient-normalizing F gives a
        # signed distance that equals the true distance away from the ring and
        # ~ (ring_R - ring_a) at the origin; the constant floor G0 keeps it finite
        # where grad F vanishes (the ring core circle / center). Crucially d(rho)
        # has zero rho-derivative on the axis, so the PDE can be enforced THROUGH
        # the trap center -- no axis-tube exclusion needed.
        q = X[0] ** 2 + X[1] ** 2 + X[2] ** 2
        s_xy = X[0] ** 2 + X[1] ** 2
        F = (q + self.ring_R ** 2 - self.ring_a ** 2) ** 2 - 4.0 * self.ring_R ** 2 * s_xy
        gradF2 = 16.0 * (s_xy * (q - self.ring_R ** 2 - self.ring_a ** 2) ** 2
                         + X[2] ** 2 * (q + self.ring_R ** 2 - self.ring_a ** 2) ** 2)
        G0 = (self.ring_R - self.ring_a) * (self.ring_R + self.ring_a) ** 2
        d_ring = F / jnp.sqrt(gradF2 + G0 * G0)
        dA = softabs(d_ring)

        # Endcap blobs are spheres -> distance is smooth on the axis already
        # (sqrt of all three coords; rho-derivative -> 0 on axis).
        s_cap_p = jnp.sqrt(X[0] ** 2 + X[1] ** 2 + (X[2] - self.cap_d) ** 2 + self.eps) - self.cap_a
        s_cap_m = jnp.sqrt(X[0] ** 2 + X[1] ** 2 + (X[2] + self.cap_d) ** 2 + self.eps) - self.cap_a
        # Distance to the endcap PAIR via the smooth product/sum blend (same form
        # as the Shepard bump): vanishes on either cap, C-infinity, and -- unlike a
        # soft-min -- has no high-curvature ridge on the z=0 mid-plane between caps.
        a, b = softabs(s_cap_p), softabs(s_cap_m)
        dB = a * b / (a + b + self.eps)

        denom = dA + dB + self.eps
        L = dB / denom
        bump = dA * dB / denom
        phi = L + bump * bulk
        if return_aux:
            # Signed distance to the nearest DISCOVERED (deformed) electrode
            # surface: < 0 inside the deformed torus tube or either endcap blob.
            # Used by visualizers to draw the discovered-shape outline.
            electrode_sdf = jnp.minimum(d_ring, jnp.minimum(s_cap_p, s_cap_m))
            return phi, electrode_sdf
        return phi


@dataclasses.dataclass
class IsotopicHyperbolicBC:
    """Isotopy / topology-preserving variant of the analytic lift.

    The field is the analytic ideal-Paul lift evaluated at a *deformed*
    coordinate Phi(x) = x + delta(x), where delta is a bounded displacement
    learned by the NN. This is a smooth diffeomorphism (for small enough
    delta), so the level surfaces of phi are *diffeomorphic* to those of the
    ideal lift — same topology (one-sheet ring hyperboloid + two-sheet endcap
    hyperboloid), but allowed to deform smoothly.

    Hard BCs are preserved on the IDEAL hyperbolic surfaces (eta = ±1) by
    multiplying delta by a bump that vanishes there:

        phi(x) = (1 - eta(x + delta_field(x))) / 2
        delta_field(x) = delta_scale * tanh(NN_3D(x)) * bump(eta(x))
        bump(eta) = clip((1 - eta^2)/2, -1, 1)

    Properties:
      - On eta(x) = ±1: bump = 0 -> delta_field = 0 -> phi = (1-eta(x))/2 (exact BC)
      - In the bulk: delta_field is a small, smooth, bounded displacement
      - The Jacobian of x -> x + delta_field stays close to identity, so the
        map is locally invertible; level surfaces don't tear or merge.

    The NN's "job" is much smaller than in the from-scratch PINN: it only has
    to learn a 3D displacement field that's already zero on the electrode
    surfaces and bounded everywhere. Starting from delta = 0 (random NN init
    with output O(1) tanh'd to ~O(delta_scale)) gives the analytic ideal-Paul
    solution — already 0.07% accurate in the trap center.
    """
    r0: float
    z0: float
    delta_scale: float = 0.02   # max displacement [cm] (small fraction of trap size)

    def eta(self, x):
        r2 = x[0] ** 2 + x[1] ** 2
        return x[2] ** 2 / (self.z0 ** 2) - r2 / (self.r0 ** 2)

    def phi(self, nn_apply, params, x, x_normalizer):
        # Bump that vanishes on the electrode hyperboloids
        e_in = self.eta(x)
        B = jnp.clip((1.0 - e_in ** 2) / 2.0, -1.0, 1.0)

        # NN-predicted 3D displacement (bounded by delta_scale * tanh * bump)
        x_norm = x_normalizer(x)
        delta_raw = nn_apply(params, x_norm[None, :])[0]    # shape (3,)
        delta = self.delta_scale * jnp.tanh(delta_raw) * B

        # Deformed coordinate
        x_def = x + delta

        # Analytic lift evaluated at the deformed coordinate
        e_def = self.eta(x_def)
        return (1.0 - e_def) / 2.0


@dataclasses.dataclass
class AnalyticHyperbolicBC_Faded:
    """Approach 2: analytic hyperbolic lift multiplied by a world-fading factor.

    phi(x) = fade(x) * (L_analytic(x) + NN(x))

      L_analytic(x) = (1 - eta(x)) / 2          # ideal-Paul harmonic
      fade(x)       = (1 - x^2/H^2) * (1 - y^2/H^2) * (1 - z^2/H^2)

    Properties:
      - On the world box (|x| or |y| or |z| = H): fade = 0 -> phi = 0 EXACTLY
        (hard BC for the world boundary)
      - On the electrode hyperboloids: fade != 0; L_analytic equals the target
        BC value but is scaled by fade(electrode) ~ 0.9. Soft-BC loss in the
        training script pushes the NN to make up the residual ~10%.
      - The NN has full headroom (bump = fade, which is O(1) in the trap region)
        so it can correct any field discrepancy without blowing up.
    """
    r0: float
    z0: float
    world_half: float

    def eta(self, x):
        r2 = x[0] ** 2 + x[1] ** 2
        return x[2] ** 2 / (self.z0 ** 2) - r2 / (self.r0 ** 2)

    def fade(self, x):
        H = self.world_half
        return ((1.0 - (x[0] / H) ** 2)
                * (1.0 - (x[1] / H) ** 2)
                * (1.0 - (x[2] / H) ** 2))

    def phi(self, nn_apply, params, x, x_normalizer):
        e = self.eta(x)
        f = self.fade(x)
        L = (1.0 - e) / 2.0
        x_norm = x_normalizer(x)
        nn_out = nn_apply(params, x_norm[None, :])[0]
        return f * (L + nn_out)
