"""Two-phase gas coupling (paper Sec 3.1, 3.6-3.7)."""

import numpy as np
import pytest

from stflip import Params, STFLIPSolver


def _two_phase(n=24, **kw):
    kw.setdefault("seed", 5)
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=3.0, two_phase=True, rho_gas=1.2,
               **kw)
    return STFLIPSolver(p, "cpu")


def _coincident_p2g(*, phases, x_velocities, liquid_ppc=1, gas_ppc=1,
                    backend="cpu"):
    params = Params(
        resolution=(2, 2, 2),
        dx=1.0,
        gravity=(0.0, 0.0, 0.0),
        particles_per_cell=liquid_ppc,
        gas_particles_per_cell=gas_ppc,
        st_enabled=False,
        two_phase=True,
        rho=1000.0,
        rho_gas=1.2,
    )
    solver = STFLIPSolver(params, backend)
    count = len(phases)
    positions = np.tile(
        np.asarray([[1.0, 0.5, 0.5]], dtype=np.float32), (count, 1))
    velocities = np.zeros((count, 3), dtype=np.float32)
    velocities[:, 0] = np.asarray(x_velocities, dtype=np.float32)
    solver.pos = solver.be.from_numpy(positions)
    solver.vel = solver.be.from_numpy(velocities)
    solver.phase = solver.be.from_numpy(np.asarray(phases, dtype=np.float32))
    solver.dt_resid = solver.be.from_numpy(
        np.zeros((count,), dtype=np.float32))
    return solver, solver._p2g(params.frame_dt)


def test_two_phase_face_velocity_is_weighted_by_particle_mass():
    phases = [1.0] * 2 + [0.0] * 8
    solver, grids = _coincident_p2g(
        phases=phases,
        x_velocities=[1.0] * 2 + [0.0] * 8,
        liquid_ppc=2,
        gas_ppc=8,
    )

    velocity = solver.be.to_numpy(grids["u"])[1, 0, 0]
    expected = 1000.0 / (1000.0 + 1.2)

    assert velocity == pytest.approx(expected, rel=2e-6)


def test_two_phase_volume_fraction_is_invariant_to_phase_ppc():
    phases = [1.0] * 2 + [0.0] * 8
    solver, grids = _coincident_p2g(
        phases=phases,
        x_velocities=[0.0] * len(phases),
        liquid_ppc=2,
        gas_ppc=8,
    )

    support = solver.be.to_numpy(grids["c_m"]) > solver.p.eps_m
    phi = solver.be.to_numpy(grids["c_phi"])[support]

    assert phi.size > 0
    np.testing.assert_allclose(phi, 0.5, rtol=2e-6, atol=1e-7)


def test_equal_phase_ppc_keeps_existing_volume_fraction_ratio():
    phases = [1.0, 1.0, 0.0, 0.0]
    solver, grids = _coincident_p2g(
        phases=phases,
        x_velocities=[0.0] * len(phases),
        liquid_ppc=4,
        gas_ppc=4,
    )

    total_volume = solver.be.to_numpy(grids["u_m"])
    liquid_volume = solver.be.to_numpy(grids["u_ml"])
    support = total_volume > solver.p.eps_m
    phi = solver.be.to_numpy(grids["u_phi"])

    np.testing.assert_allclose(
        total_volume[support], 2.0 * liquid_volume[support],
        rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(phi[support], 0.5, rtol=1e-7, atol=0.0)


@pytest.mark.parametrize("phase", [True, -1.0, 0.25, 2.0, np.nan])
def test_inflow_phase_must_be_a_binary_tag(phase):
    solver = _two_phase(n=2)
    mask = np.ones(solver.shape, dtype=bool)

    with pytest.raises(ValueError, match="inflow phase must be 0 or 1"):
        solver.add_inflow(mask, phase=phase)


def test_checkpoint_restore_rejects_non_binary_phase_tags():
    source = _two_phase(n=2)
    mask = np.zeros(source.shape, dtype=bool)
    mask[0, 0, 0] = True
    source.add_liquid_mask(mask)
    state = source.checkpoint_state()
    state["phase"][0] = np.float32(0.5)

    restored = _two_phase(n=2)
    with pytest.raises(ValueError, match="checkpoint phase must contain 0 or 1"):
        restored.restore_state(state)


def test_single_phase_p2g_is_exactly_independent_of_gas_parameters():
    def deposit(rho_gas, gas_ppc):
        params = Params(
            resolution=(2, 2, 2),
            dx=1.0,
            particles_per_cell=2,
            gas_particles_per_cell=gas_ppc,
            rho_gas=rho_gas,
            st_enabled=False,
            two_phase=False,
        )
        solver = STFLIPSolver(params, "cpu")
        solver.pos = np.asarray(
            [[1.0, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=np.float32)
        solver.vel = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        solver.phase = np.asarray([1.0, 0.0], dtype=np.float32)
        solver.dt_resid = np.zeros((2,), dtype=np.float32)
        return solver._p2g(params.frame_dt)

    baseline = deposit(rho_gas=1.2, gas_ppc=2)
    changed_gas_settings = deposit(rho_gas=900.0, gas_ppc=64)

    assert "u_ml" not in baseline
    assert baseline["u"][1, 0, 0] == np.float32(0.5)
    assert baseline.keys() == changed_gas_settings.keys()
    for name in baseline:
        np.testing.assert_array_equal(baseline[name], changed_gas_settings[name])


@pytest.mark.gpu
def test_two_phase_ppc_weighting_matches_cuda_backend():
    from stflip.backend import cuda_diagnostics

    pytest.importorskip("cupy")
    available, reason = cuda_diagnostics(force=True)
    assert available, reason
    phases = [1.0] * 2 + [0.0] * 8
    velocities = [1.0] * 2 + [0.0] * 8
    cpu, cpu_grids = _coincident_p2g(
        phases=phases,
        x_velocities=velocities,
        liquid_ppc=2,
        gas_ppc=8,
        backend="cpu",
    )
    cuda, cuda_grids = _coincident_p2g(
        phases=phases,
        x_velocities=velocities,
        liquid_ppc=2,
        gas_ppc=8,
        backend="cuda",
    )

    for name in ("u", "u_m", "u_ml", "u_p", "u_phi", "c_m", "c_phi"):
        np.testing.assert_allclose(
            cuda.be.to_numpy(cuda_grids[name]),
            cpu.be.to_numpy(cpu_grids[name]),
            rtol=2e-6,
            atol=1e-7,
        )


def test_fill_gas_only_when_two_phase():
    n = 16
    p = Params(resolution=(n, n, n), dx=1.0 / n, two_phase=False)
    s = STFLIPSolver(p, "cpu")
    s.add_liquid_mask(np.ones((n, n, n), bool))
    assert s.fill_gas() == 0  # no-op in free-surface mode


def test_gas_fills_free_cells():
    n = 16
    s = _two_phase(n=n)
    liq = np.zeros((n, n, n), bool)
    liq[:, :, n // 2:] = True
    nl = s.add_liquid_mask(liq)
    ng = s.fill_gas()
    ph = s.be.to_numpy(s.phase)
    assert (ph > 0.5).sum() == nl and ng > 0
    # roughly half the domain becomes gas
    assert (ph < 0.5).sum() == ng


def test_heavy_blob_sinks_through_gas():
    n = 24
    s = _two_phase(n=n)
    blob = np.zeros((n, n, n), bool)
    blob[9:15, 9:15, 15:21] = True
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
    liq = np.ones((n, n, n), bool)
    liq[9:15, 9:15, 4:10] = False
    s.add_liquid_mask(liq)
    gas = np.zeros((n, n, n), bool)
    gas[9:15, 9:15, 4:10] = True
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


def test_render_particles_exclude_gas():
    """Gas is simulation state, not water: the render export must contain
    only liquid particles or the surface mesher solidifies the air region."""
    n = 16
    s = _two_phase(n=n, seed=9)
    liq = np.zeros((n, n, n), bool)
    liq[:, :, : n // 2] = True
    n_liquid = s.add_liquid_mask(liq)
    s.fill_gas()
    assert s.pos.shape[0] > n_liquid  # gas actually seeded
    s.step_frame()
    pos, vel = s.get_render_particles()
    assert len(pos) == len(vel)
    assert 0 < len(pos) <= n_liquid
    assert len(pos) < s.pos.shape[0]
