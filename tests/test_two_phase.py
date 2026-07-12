"""Two-phase gas coupling (paper Sec 3.1, 3.6-3.7)."""

import numpy as np

from stflip import Params, STFLIPSolver


def _two_phase(n=24, **kw):
    kw.setdefault("seed", 5)
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=3.0, two_phase=True, rho_gas=1.2,
               **kw)
    return STFLIPSolver(p, "cpu")


def test_fill_gas_only_when_two_phase():
    n = 16
    p = Params(resolution=(n, n, n), dx=1.0 / n, two_phase=False)
    s = STFLIPSolver(p, "cpu")
    s.add_liquid_mask(np.ones((n, n, n), bool))
    assert s.fill_gas() == 0  # no-op in free-surface mode


def test_gas_fills_free_cells():
    n = 16
    s = _two_phase(n=n)
    liq = np.zeros((n, n, n), bool); liq[:, :, n // 2:] = True
    nl = s.add_liquid_mask(liq)
    ng = s.fill_gas()
    ph = s.be.to_numpy(s.phase)
    assert (ph > 0.5).sum() == nl and ng > 0
    # roughly half the domain becomes gas
    assert (ph < 0.5).sum() == ng


def test_heavy_blob_sinks_through_gas():
    n = 24
    s = _two_phase(n=n)
    blob = np.zeros((n, n, n), bool); blob[9:15, 9:15, 15:21] = True
    s.add_liquid_mask(blob)
    s.fill_gas()
    ph = s.be.to_numpy(s.phase)
    z0 = s.be.to_numpy(s.pos)[ph > 0.5][:, 2].mean()
    for _ in range(6):
        s.step_frame()
    pos = s.be.to_numpy(s.pos)
    assert np.all(np.isfinite(pos))
    z1 = pos[ph > 0.5][:, 2].mean()
    assert z1 < z0 - 0.05, f"heavy liquid should sink: {z0:.3f} -> {z1:.3f}"


def test_light_bubble_rises_through_liquid():
    n = 24
    s = _two_phase(n=n, seed=6)
    liq = np.ones((n, n, n), bool); liq[9:15, 9:15, 4:10] = False
    s.add_liquid_mask(liq)
    gas = np.zeros((n, n, n), bool); gas[9:15, 9:15, 4:10] = True
    s.add_gas_mask(gas)
    ph = s.be.to_numpy(s.phase)
    zg0 = s.be.to_numpy(s.pos)[ph < 0.5][:, 2].mean()
    for _ in range(6):
        s.step_frame()
    pos = s.be.to_numpy(s.pos)
    assert np.all(np.isfinite(pos))
    zg1 = pos[ph < 0.5][:, 2].mean()
    assert zg1 > zg0 + 0.03, f"light gas should rise: {zg0:.3f} -> {zg1:.3f}"


def test_particle_count_conserved_closed_box():
    n = 20
    s = _two_phase(n=n, seed=7)
    s.add_liquid_mask((lambda m: (m.__setitem__((slice(None), slice(None),
                     slice(n // 2, None)), True) or m))(np.zeros((n, n, n), bool)))
    s.fill_gas()
    n0 = s.pos.shape[0]
    for _ in range(4):
        s.step_frame()
    # No inflow/culling: the fixed particle set is preserved.
    assert s.pos.shape[0] == n0
    assert np.all(s.be.to_numpy(s.phase) >= 0.0)
