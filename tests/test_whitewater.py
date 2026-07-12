"""Whitewater secondary particles (foam/bubble/spray)."""

import numpy as np

from stflip import Params, STFLIPSolver
from stflip.whitewater import BUBBLE, FOAM, SPRAY, Whitewater, WhitewaterParams


def _splashy_dam(n=20, **kw):
    kw.setdefault("seed", 0)
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=6.0, **kw)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[:n // 3, :, :n // 2] = True
    s.add_liquid_mask(m)
    return s


def test_dam_break_emits_and_stays_finite():
    s = _splashy_dam()
    ww = Whitewater(s, WhitewaterParams(
        energy_min=0.05, speed_min=0.05, seed=1))
    emitted_total = 0
    for _ in range(5):
        s.step_frame()
        out = ww.step(s.p.frame_dt)
        emitted_total += out.get("emitted", 0)
    assert emitted_total > 0, "an energetic dam break must emit whitewater"
    pos, vel, kind, life = ww.get_render_particles()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))
    size = np.asarray(s.size)
    assert np.all(pos >= 0.0) and np.all(pos <= size[None, :])
    assert set(np.unique(kind)).issubset({FOAM, BUBBLE, SPRAY})


def test_classification_by_phase_field():
    """A secondary deep inside liquid must classify as bubble; one in the
    empty region as spray."""
    n = 16
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=4.0, seed=2)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[:, :, :n // 2] = True
    s.add_liquid_mask(m)
    s.step_frame()
    ww = Whitewater(s)
    xp = s.be.xp
    ww.pos = xp.asarray([[0.5, 0.5, 0.15],   # deep in the pool
                         [0.5, 0.5, 0.9]],   # in the air
                        dtype=xp.float32)
    ww.vel = xp.zeros((2, 3), dtype=xp.float32)
    ww.life = xp.asarray([5.0, 5.0], dtype=xp.float32)
    ww.kind = xp.zeros((2,), dtype=xp.int8)
    ww._classify()
    kind = s.be.to_numpy(ww.kind)
    assert kind[0] == BUBBLE
    assert kind[1] == SPRAY


def test_spray_falls_and_bubble_rises():
    n = 16
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=4.0, seed=3)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[:, :, :n // 2] = True
    s.add_liquid_mask(m)
    s.step_frame()
    ww = Whitewater(s, WhitewaterParams(drag=0.5))
    xp = s.be.xp
    ww.pos = xp.asarray([[0.5, 0.5, 0.85],   # spray high in the air
                         [0.5, 0.5, 0.15]],  # bubble deep in the pool
                        dtype=xp.float32)
    ww.vel = xp.zeros((2, 3), dtype=xp.float32)
    ww.life = xp.asarray([10.0, 10.0], dtype=xp.float32)
    ww.kind = xp.zeros((2,), dtype=xp.int8)
    z0 = s.be.to_numpy(ww.pos)[:, 2].copy()
    for _ in range(3):
        ww.step(s.p.frame_dt)
    pos, _vel, kind, _life = ww.get_render_particles()
    assert pos[0, 2] < z0[0] - 0.01, "spray must fall under gravity"
    assert pos[1, 2] > z0[1] + 0.005, "bubble must rise by buoyancy"


def test_lifetime_cull_and_cap():
    s = _splashy_dam(n=16)
    s.step_frame()
    ww = Whitewater(s, WhitewaterParams(max_particles=50))
    xp = s.be.xp
    n0 = 200
    ww.pos = xp.asarray(
        0.4 + 0.2 * np.random.default_rng(0).random((n0, 3)),
        dtype=xp.float32)
    ww.vel = xp.zeros((n0, 3), dtype=xp.float32)
    ww.life = xp.full((n0,), 0.01, dtype=xp.float32)  # about to expire
    ww.kind = xp.zeros((n0,), dtype=xp.int8)
    ww.step(s.p.frame_dt)
    assert ww.pos.shape[0] <= 50
    # A second step with everything expired empties the system.
    ww.life = xp.zeros_like(ww.life)
    ww.step(s.p.frame_dt)
    assert ww.pos.shape[0] == 0
