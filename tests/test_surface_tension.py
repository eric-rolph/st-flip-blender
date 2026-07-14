"""Surface tension (CSF) forces and coefficients (paper Sec 3.9)."""

import numpy as np

from stflip import Params, STFLIPSolver, surface_tension


def _sphere_phase(n, r, cx=None):
    cx = cx if cx is not None else (n / 2.0)
    ax = np.arange(n) + 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.sqrt((x - cx) ** 2 + (y - cx) ** 2 + (z - cx) ** 2)
    return (d < r).astype(np.float32)


def test_smoothing_reduces_variance():
    rng = np.random.default_rng(0)
    noisy = rng.random((16, 16, 16)).astype(np.float32)
    smooth = surface_tension.smooth_phase(np, noisy, iters=3)
    assert smooth.var() < noisy.var()


def test_csf_force_points_inward_on_a_blob():
    """For a liquid sphere the CSF force must pull the interface inward
    (toward the centre), i.e. oppose the outward normal."""
    n = 32
    phi = _sphere_phase(n, r=9.0)
    F = surface_tension.cell_force(np, phi, dx=1.0, sigma=1.0, iters=2)
    c = n / 2.0
    ax = np.arange(n) + 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    rvec = np.stack([x - c, y - c, z - c], axis=-1)
    rnorm = np.linalg.norm(rvec, axis=-1, keepdims=True) + 1e-9
    outward = rvec / rnorm
    radial = (F * outward).sum(axis=-1)
    d = np.sqrt((rvec ** 2).sum(-1))
    shell = (d > 7) & (d < 11)  # the diffuse interface band
    assert radial[shell].mean() < 0, "surface tension should compress the blob"


def test_surface_tension_stays_finite():
    n = 16
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=4.0, seed=1, surface_tension=0.02,
               transfer="apic")
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[4:12, 4:12, 4:12] = True
    s.add_liquid_mask(m)
    for _ in range(3):
        s.step_frame()
    vel = s.be.to_numpy(s.vel)
    assert np.all(np.isfinite(vel))


def test_zero_sigma_is_a_noop():
    """sigma = 0 must be bitwise plain, even with the stabilizer enabled.

    The original form compared run(0.0) to run(0.0) -- a tautology that
    guarded nothing (roadmap CAP review).  The real guard: enabling
    st_implicit with zero surface tension must not perturb one bit.
    """
    def run(**st_kwargs):
        n = 16
        p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
                   frame_dt=1 / 24, cfl_target=4.0, seed=2, **st_kwargs)
        s = STFLIPSolver(p, "cpu")
        m = np.zeros((n, n, n), bool)
        m[:n // 2, :, :n // 2] = True
        s.add_liquid_mask(m)
        for _ in range(3):
            s.step_frame()
        return s.be.to_numpy(s.pos)
    assert np.array_equal(
        run(), run(surface_tension=0.0, st_implicit=True))


def test_capillary_time_step_clamp():
    """With sigma > 0 the adaptive step must respect the Brackbill capillary
    limit dt <= sqrt(rho dx^3 / (4 pi sigma)) even when the velocity CFL
    would permit a far larger step (calm pool, large CFL target)."""
    import math
    n = 16
    sigma = 100.0
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=8.0, seed=0,
               surface_tension=sigma)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[:, :, :n // 2] = True
    s.add_liquid_mask(m)
    stats = s.step_frame()
    dt_cap = math.sqrt(p.rho * p.dx ** 3 / (4.0 * math.pi * sigma))
    assert stats.steps > 1  # the clamp forced substepping of the calm frame
    assert max(stats.dt_values) <= dt_cap + 1e-9
    assert np.all(np.isfinite(s.be.to_numpy(s.vel)))
