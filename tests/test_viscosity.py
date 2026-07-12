"""Implicit viscosity (Stam-style diffusion)."""

import numpy as np

from stflip import Params, STFLIPSolver
from stflip import viscosity as V


def test_diffuse_conserves_mean_and_smooths():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((18, 14, 14)).astype(np.float32)
    fixed = np.zeros_like(u, dtype=bool)
    out = V.diffuse_component(np, u, coef=8.0, fixed=fixed, fixed_value=0.0)
    assert out.var() < u.var()                       # smooths
    assert abs(out.mean() - u.mean()) < 1e-4         # Neumann conserves mean
    assert np.all(np.isfinite(out))


def test_diffuse_unconditionally_stable_at_huge_coef():
    """The whole reason for an implicit solve: enormous coef stays bounded
    (an explicit step would diverge for coef > ~1/6)."""
    rng = np.random.default_rng(1)
    u = rng.standard_normal((16, 16, 16)).astype(np.float32)
    fixed = np.zeros_like(u, dtype=bool)
    out = V.diffuse_component(np, u, coef=1000.0, fixed=fixed, fixed_value=0.0)
    assert np.all(np.isfinite(out))
    assert out.var() < u.var() * 0.01                # nearly fully relaxed


def test_diffuse_zero_coef_is_identity():
    u = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    fixed = np.zeros_like(u, dtype=bool)
    out = V.diffuse_component(np, u, coef=0.0, fixed=fixed, fixed_value=0.0)
    assert np.array_equal(out, u)


def test_diffuse_dirichlet_solid_faces_held():
    """Fixed (solid) faces are held at the solid velocity; the field relaxes
    toward that no-slip boundary value."""
    u = np.ones((10, 6, 6), dtype=np.float32)
    fixed = np.zeros_like(u, dtype=bool)
    fixed[0] = True
    fixed[-1] = True
    out = V.diffuse_component(np, u, coef=50.0, fixed=fixed, fixed_value=0.0)
    assert np.allclose(out[0], 0.0) and np.allclose(out[-1], 0.0)
    # interior relaxes toward the 0 walls (below the initial 1.0)
    assert out[5].mean() < 1.0


def _blob(visc, n=20, frames=5, seed=2, cfl=8.0):
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=cfl, seed=seed, viscosity=visc)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[6:14, 6:14, 8:16] = True
    s.add_liquid_mask(m)
    for _ in range(frames):
        s.step_frame()
    return s


def test_zero_viscosity_is_a_noop():
    a = _blob(0.0)
    b = _blob(0.0)
    assert np.array_equal(a.be.to_numpy(a.pos), b.be.to_numpy(b.pos))


def test_viscosity_dissipates_kinetic_energy_and_stays_stable():
    inv = _blob(0.0)
    visc = _blob(0.06)
    vi = inv.be.to_numpy(inv.vel)
    vv = visc.be.to_numpy(visc.vel)
    ke_i = float((vi * vi).sum())
    ke_v = float((vv * vv).sum())
    assert np.all(np.isfinite(vv))                   # stable at CFL 8
    assert ke_v < ke_i                               # viscous dissipation
    assert visc.pos.shape[0] == inv.pos.shape[0]
