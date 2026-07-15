"""ENER-M2b: advection-reflection scheme."""

import hashlib

import numpy as np
import pytest

from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver


def _state_hash(solver) -> str:
    digest = hashlib.sha256()
    for arr in (solver.pos, solver.vel, solver.dt_resid):
        host = solver.be.to_numpy(arr)
        digest.update(np.ascontiguousarray(host).tobytes())
    return digest.hexdigest()


def _params(n=12, **overrides):
    base = dict(
        resolution=(n, n, n), gravity=(0.0, 0.0, -9.81), dx=1.0 / n,
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
        st_enabled=True, seed=7,
    )
    base.update(overrides)
    return Params(**base)


def _pool_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    return mask


def _dam_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    return mask


def _run(params, mask, frames=3):
    solver = STFLIPSolver(params, get_backend("cpu"))
    solver.add_liquid_mask(mask)
    for _ in range(frames):
        solver.step_frame()
    return solver


class TestGating:
    def test_rejects_non_bool(self):
        with pytest.raises(TypeError):
            Params(resolution=(8, 8, 8), dx=0.125, reflection="yes")

    def test_off_by_default_is_bitwise_plain(self):
        n = 12
        plain = _run(_params(n), _dam_mask(n))
        explicit = _run(_params(n, reflection=False), _dam_mask(n))
        assert _state_hash(plain) == _state_hash(explicit)

    def test_reflection_changes_the_result(self):
        n = 12
        plain = _run(_params(n), _dam_mask(n))
        reflected = _run(_params(n, reflection=True), _dam_mask(n))
        assert _state_hash(plain) != _state_hash(reflected)


class TestHydrostaticTrace:
    def test_still_pool_stays_still(self):
        # The trace the review demanded: gravity enters particles via the
        # reflected impulse, transport through u1 = 0 moves nothing, the
        # second projection cancels the uniform impulse against the floor.
        n = 12
        solver = _run(
            _params(n, reflection=True), _pool_mask(n), frames=3)
        speeds = np.linalg.norm(solver.be.to_numpy(solver.vel), axis=1)
        assert float(speeds.max()) < 0.5

    def test_dam_still_falls_under_gravity(self):
        # The counterpart guard: gravity must not be dropped (the failure
        # of the original, review-rejected step order).
        n = 12
        solver = _run(
            _params(n, reflection=True), _dam_mask(n), frames=3)
        vel = solver.be.to_numpy(solver.vel)
        assert float(np.abs(vel).max()) > 0.5
        pos = solver.be.to_numpy(solver.pos)
        # The dam front has advanced beyond the initial x extent.
        assert float(pos[:, 0].max()) > 1.0 / 3.0 + 0.05


class TestResidualInduction:
    def test_randomized_adaptive_dt_stress(self):
        # Drive the composed half-clips through abrupt dt changes and
        # assert the documented invariant |r| <= max(dt, dt_prev)/2 with
        # geometric excess contraction (docs/design/advection-reflection).
        rng = np.random.default_rng(0)
        particles = 4096
        resid = np.zeros(particles)
        dt = 1.0
        dt_prev = 1.0
        for step in range(400):
            if step % 17 == 0:
                dt = float(rng.choice([0.5, 1.0, 2.0]))
            dt_act1 = np.clip(dt / 2.0 + resid, 0.0, dt)
            r1 = dt / 2.0 + resid - dt_act1
            jit = dt * (rng.random(particles) - 0.5)
            dt_act2 = np.clip(dt / 2.0 + r1 + jit, 0.0, dt)
            resid = dt / 2.0 + r1 - dt_act2
            bound = max(dt, dt_prev) / 2.0
            assert float(np.abs(resid).max()) <= bound + 1e-12
            assert float((dt_act1 + dt_act2).max()) <= 2.0 * dt + 1e-12
            dt_prev = dt

    def test_solver_drift_bound_with_reflection(self):
        n = 12
        solver = STFLIPSolver(
            _params(n, reflection=True), get_backend("cpu"))
        solver.add_liquid_mask(_dam_mask(n))
        dt_max = 0.0
        for _ in range(4):
            stats = solver.step_frame()
            dt_max = max(dt_max, *(stats.dt_values or (0.0,)))
            resid = solver.be.to_numpy(solver.dt_resid)
            assert np.all(np.isfinite(resid))
            assert float(np.abs(resid).max()) <= 0.5 * dt_max + 1e-9


class TestCompatibility:
    @pytest.mark.parametrize("transfer", ["apic", "pic"])
    def test_other_transfers_run(self, transfer):
        n = 12
        solver = _run(
            _params(n, reflection=True, transfer=transfer), _dam_mask(n))
        vel = solver.be.to_numpy(solver.vel)
        assert np.all(np.isfinite(vel))

    def test_two_phase_smoke(self):
        n = 12
        params = _params(
            n, reflection=True, two_phase=True, rho_gas=1.25,
            pressure_solver="multigrid")
        solver = STFLIPSolver(params, get_backend("cpu"))
        solver.add_liquid_mask(_pool_mask(n))
        solver.fill_gas()
        for _ in range(2):
            solver.step_frame()
        vel = solver.be.to_numpy(solver.vel)
        assert np.all(np.isfinite(vel))

    def test_outflow_keeps_arrays_consistent(self):
        n = 12
        params = _params(n, reflection=True)
        solver = STFLIPSolver(params, get_backend("cpu"))
        solver.add_liquid_mask(_dam_mask(n))
        sink = np.zeros((n,) * 3, dtype=bool)
        sink[-1, :, :] = True
        solver.add_outflow(sink, mode="VOLUME")
        for _ in range(4):
            solver.step_frame()
        assert solver.dt_resid.shape[0] == solver.pos.shape[0]
        assert solver.particle_id.shape[0] == solver.pos.shape[0]

    def test_checkpoint_resume_bit_identical(self):
        n = 12
        params = _params(n, reflection=True)
        straight = _run(params, _dam_mask(n), frames=4)

        first = STFLIPSolver(params, get_backend("cpu"))
        first.add_liquid_mask(_dam_mask(n))
        for _ in range(2):
            first.step_frame()
        state = first.checkpoint_state()

        resumed = STFLIPSolver(params, get_backend("cpu"))
        resumed.restore_state(state)
        for _ in range(2):
            resumed.step_frame()
        assert _state_hash(resumed) == _state_hash(straight)


class TestEnergyRetention:
    @pytest.mark.slow
    def test_rotating_tank_reflection_beats_plain_at_same_cfl(self):
        # The quick in-suite version of the ENER-M2b gate: at CFL 8 over
        # 12 frames of the zero-g rotating tank, reflection must retain
        # MORE angular momentum than plain stepping.  The full CFL-16
        # study against the ENER-M0 floor-relative baselines runs in
        # tools/ and is recorded in the PR.
        from stflip.velocity import SolidBodyRotation

        n = 16
        field = SolidBodyRotation(
            center=(0.5, 0.5, 0.5), angular_velocity=(0.0, 0.0, 6.0))

        def run(reflection):
            params = _params(
                n, reflection=reflection, gravity=(0.0, 0.0, 0.0),
                cfl_target=8.0, particles_per_cell=4)
            solver = STFLIPSolver(params, get_backend("cpu"))
            solver.add_liquid_mask(
                np.ones((n,) * 3, dtype=bool), velocity=field)
            pos0 = solver.be.to_numpy(solver.pos) - 0.5
            vel0 = solver.be.to_numpy(solver.vel)
            lz0 = float(np.mean(
                pos0[:, 0] * vel0[:, 1] - pos0[:, 1] * vel0[:, 0]))
            for _ in range(12):
                solver.step_frame()
            pos = solver.be.to_numpy(solver.pos) - 0.5
            vel = solver.be.to_numpy(solver.vel)
            lz = float(np.mean(
                pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
            return lz / lz0

        plain = run(False)
        reflected = run(True)
        assert reflected > plain
