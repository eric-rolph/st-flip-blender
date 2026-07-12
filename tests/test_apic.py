"""APIC affine transfer (paper Sec 3.9)."""

import numpy as np
import pytest

from stflip import Params, STFLIPSolver
from stflip.solver import _inv3x3


def _solver(n=16, **kw):
    kw.setdefault("gravity", (0, 0, -9.81))
    kw.setdefault("seed", 0)
    p = Params(resolution=(n, n, n), dx=1.0 / n,
               frame_dt=1 / 24, cfl_target=6.0, **kw)
    return STFLIPSolver(p, "cpu")


def test_inv3x3_matches_numpy():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(50, 3, 3)).astype(np.float32)
    # Make SPD-ish so it is well-conditioned.
    M = np.einsum("nij,nkj->nik", M, M) + np.eye(3)[None] * 0.5
    inv = _inv3x3(np, M)
    ref = np.linalg.inv(M)
    assert np.allclose(inv, ref, atol=1e-3)


def test_apic_recovers_linear_velocity_gradient():
    """C_p must reconstruct the velocity gradient of a linear field exactly:
    for u(x) = A x + b the affine matrix equals A (this is APIC's defining
    property and validates the MAC-grid B*D^-1 reconstruction)."""
    n = 16
    s = _solver(n=n, transfer="apic")
    dx = s.p.dx
    A = np.array([[0.0, 0.5, -0.3],
                  [0.2, 0.0, 0.4],
                  [-0.1, 0.6, 0.0]], dtype=np.float64)  # traceless-ish
    b = np.array([0.3, -0.2, 0.1])

    offsets = {"u": (0.0, 0.5, 0.5), "v": (0.5, 0.0, 0.5), "w": (0.5, 0.5, 0.0)}
    shapes = {"u": (n + 1, n, n), "v": (n, n + 1, n), "w": (n, n, n + 1)}
    grids = {}
    for axis, g in enumerate(("u", "v", "w")):
        sh = shapes[g]
        off = offsets[g]
        ii, jj, kk = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]),
                                 np.arange(sh[2]), indexing="ij")
        x = (ii + off[0]) * dx
        y = (jj + off[1]) * dx
        z = (kk + off[2]) * dx
        # component `axis` of A x + b
        val = A[axis, 0] * x + A[axis, 1] * y + A[axis, 2] * z + b[axis]
        grids[g] = val.astype(np.float32)

    # Particles well inside the domain so all taps are valid.
    rng = np.random.default_rng(1)
    pos = (0.3 + 0.4 * rng.random((200, 3))).astype(np.float32)
    s.pos = s.be.from_numpy(pos)
    u_new = s._sample_faces(grids, s.pos)
    _v, C = s._g2p_apic(grids, s.pos, u_new)
    C = s.be.to_numpy(C)
    # Interior particles should recover A to good accuracy.
    err = np.abs(C - A[None]).max(axis=(1, 2))
    assert np.median(err) < 0.05, f"median C error {np.median(err):.4f}"


def test_apic_runs_stable_and_carries_C():
    n = 20
    s = _solver(n=n, transfer="apic")
    m = np.zeros((n, n, n), bool); m[:n // 3, :, :n // 2] = True
    n0 = s.add_liquid_mask(m)
    for _ in range(4):
        s.step_frame()
    assert s.pos.shape[0] == n0
    assert s.C.shape == (n0, 3, 3)
    vel = s.be.to_numpy(s.vel)
    assert np.all(np.isfinite(vel)) and np.all(np.isfinite(s.be.to_numpy(s.C)))
    # APIC is low-dissipation but must not blow up on a dam break.
    assert np.linalg.norm(vel, axis=1).max() < 8.0


def test_apic_less_dissipative_than_pic():
    """A rotating velocity blob keeps more kinetic energy under APIC than PIC."""
    def spin_energy(mode):
        n = 16
        s = _solver(n=n, gravity=(0, 0, 0), transfer=mode, seed=3)
        m = np.zeros((n, n, n), bool); m[4:12, 4:12, 4:12] = True
        s.add_liquid_mask(m)
        xp = s.be.xp
        pos = s.be.to_numpy(s.pos)
        c = np.array([0.5, 0.5, 0.5])
        r = pos - c
        # solid-body rotation about z
        vel = np.zeros_like(pos)
        vel[:, 0] = -r[:, 1] * 6.0
        vel[:, 1] = r[:, 0] * 6.0
        s.vel = s.be.from_numpy(vel.astype(np.float32))
        for _ in range(3):
            s.step_frame()
        v = s.be.to_numpy(s.vel)
        return float((v * v).sum())
    e_apic = spin_energy("apic")
    e_pic = spin_energy("pic")
    assert e_apic > e_pic, f"APIC {e_apic:.2f} should exceed PIC {e_pic:.2f}"
