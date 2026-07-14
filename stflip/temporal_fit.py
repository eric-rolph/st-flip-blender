"""W_T-weighted linear-in-time moment fit (roadmap TIME-M1).

Pure estimator math for the two-time-level slab experiment: nothing in the
solver imports this module.  The question it exists to answer is whether
retaining first temporal moments during P2G -- algebraically equivalent to
two time levels at the slab ends -- and reconstructing linearly in tau buys
anything over today's estimator, which collapses the slab to the one-sided
W_T-weighted mean (effective evaluation time tau ~ +0.2266, effective sample
fraction ~ 0.613; both cross-checked against
``validation.temporal_quadrature_coverage``).

The paper already reports a negative preliminary result for a related idea
(thin-4D-grid linear-in-time reconstruction, p.6, and time-interpolated G2P,
p.9), so this study is the KILL POINT: solver integration (TIME-M2) only
happens on a GO verdict from ``tools/run_temporal_study.py``.

Estimator.  Given per-sample weights ``w_i`` (temporal kernel times any
spatial kernel weight), sample times ``tau_i`` and values ``q_i``:

    qbar = S0q / S0w            (today's estimator, the lam -> inf limit)
    tbar = S1w / S0w
    var  = S2w / S0w - tbar**2
    cov  = S1q / S0w - qbar * tbar
    b    = cov / (var + lam)
    q(tau_eval) = qbar + b * (tau_eval - tbar)

``lam`` is a dimensionless ridge regulariser in units of the reference tau
variance 1/12; ``lam -> inf`` degrades EXACTLY to today's one-sided mean
(never to the symmetric in-slab mean the paper rejected for oscillatory
artifacts).
"""

from __future__ import annotations

REFERENCE_TAU_VARIANCE = 1.0 / 12.0


def accumulate_moments(xp, tau, q, weights, axis=-1):
    """Weighted temporal moments over ``axis`` (default: the sample axis).

    Returns a dict of S0w, S0q, S1w, S2w, S1q -- exactly the accumulators a
    TIME-M2 P2G would scatter (three extra per velocity component).
    """

    w = weights
    return {
        "S0w": w.sum(axis=axis),
        "S0q": (w * q).sum(axis=axis),
        "S1w": (w * tau).sum(axis=axis),
        "S2w": (w * tau * tau).sum(axis=axis),
        "S1q": (w * tau * q).sum(axis=axis),
    }


def one_sided_mean(xp, moments, eps=1e-12):
    """Today's estimator: the plain W_T-weighted mean."""

    d = xp.maximum(moments["S0w"], eps)
    return moments["S0q"] / d


def fit_evaluate(xp, moments, lam, tau_eval=0.5, eps=1e-12):
    """Ridge-regularised linear-in-tau reconstruction at ``tau_eval``.

    Returns ``(value, slope)``.  ``lam`` is dimensionless (multiplied by the
    reference variance 1/12 internally).
    """

    if lam < 0.0:
        raise ValueError("lam must not be negative")
    d = xp.maximum(moments["S0w"], eps)
    qbar = moments["S0q"] / d
    tbar = moments["S1w"] / d
    var = xp.maximum(moments["S2w"] / d - tbar * tbar, 0.0)
    cov = moments["S1q"] / d - qbar * tbar
    slope = cov / (var + lam * REFERENCE_TAU_VARIANCE + eps)
    return qbar + slope * (tau_eval - tbar), slope


def simulate_jitter_taus(rng, particles, steps, gamma, dt_ratio=1.0):
    """Simulate the ACTUAL Eq. 10-11 residual process, vectorized.

    Returns a ``(steps, particles)`` array of deposit times
    ``theta = -dt_resid / dt_prev`` INCLUDING the clamp and an optional
    abrupt adaptive-dt change (``dt_ratio`` scales dt at the midpoint,
    exercising the clamping bias the paper bounds at under one percent of
    updates).  ``gamma`` may be a scalar or a per-particle array.
    """

    import numpy as np

    dt = 1.0
    resid = np.zeros(particles, dtype=np.float64)
    thetas = np.empty((steps, particles), dtype=np.float64)
    for k in range(steps):
        dt_prev = dt
        if dt_ratio != 1.0 and k == steps // 2:
            dt = dt * dt_ratio
        thetas[k] = -resid / dt_prev
        xi = rng.random(particles) - 0.5
        dt_act = np.clip(dt + resid + gamma * xi * dt, 0.0, 2.0 * dt)
        resid = dt + resid - dt_act
    return thetas
