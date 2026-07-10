"""Deposition kernels from ST-FLIP (Braun et al. 2026), Section 3.8.

Spatial: separable 1D poly6 variant  W_S(r) = (35/32)^3 * prod_d w6(r_d),
         w6(r) = max(0, (1 - r^2))^3, support |r| < 1 (grid units).

Temporal: one-sided, half-step shifted poly6 (Eq. 19)
         W_T(tau) = (35/16) * w6(tau - 1/2) * 1{tau <= 1/2}
peaking at tau = +1/2 so the most recent samples in the time slab receive the
most weight.  Both kernels integrate to 1 over their (grid-normalised)
support, which makes the P2G weight accumulators reusable as a phase field.
"""

from __future__ import annotations

# 1D normalisation for the poly6 ramp: integral of (1-r^2)^3 over [-1,1] is 32/35.
NORM_1D = 35.0 / 32.0
# One-sided temporal kernel integrates the past half only, hence twice the norm.
NORM_T = 35.0 / 16.0


def w6(xp, r):
    """Unnormalised 1D poly6 ramp, support |r| < 1."""
    q = 1.0 - r * r
    q = xp.maximum(q, 0.0)
    return q * q * q


def w_spatial_1d(xp, r):
    """Normalised per-axis spatial weight."""
    return NORM_1D * w6(xp, r)


def w_temporal(xp, tau):
    """One-sided temporal kernel W_T (Eq. 19). tau is slab-normalised time."""
    return xp.where(tau <= 0.5, NORM_T * w6(xp, tau - 0.5), 0.0)


def smoothstep(xp, a, b, x):
    """Standard cubic ramp used for adaptive jitter attenuation."""
    s = xp.clip((x - a) / (b - a), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)
