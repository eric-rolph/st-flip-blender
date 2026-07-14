"""ERR-M1: step-control signals, controller law, and solver diagnostics."""

import hashlib

import numpy as np
import pytest

from stflip import step_control
from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver


def _state_hash(solver) -> str:
    digest = hashlib.sha256()
    for arr in (solver.pos, solver.vel, solver.dt_resid):
        host = solver.be.to_numpy(arr)
        digest.update(np.ascontiguousarray(host).tobytes())
    return digest.hexdigest()


def _dam_params(n=12, **overrides):
    base = dict(
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
        st_enabled=True, seed=6,
    )
    base.update(overrides)
    return Params(**base)


def _dam_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    return mask


class TestControllerLaw:
    def test_bounds_and_ratio(self):
        state = step_control.ControllerState.fresh()
        floor = 0.25
        for level in (0.0, 0.3, 1.0, 0.6, 1.0, 1.0, 1.0):
            before = state.r
            r = step_control.update(
                state, {"clamp_bind_fraction": level * 0.1}, 1.0, floor)
            assert floor <= r <= 1.0
            assert r <= before + 1e-12 or state.quiet_streak >= (
                step_control.QUIET_STEPS)

    @pytest.mark.parametrize("level", [0.1, 0.2])
    def test_steady_state_converges_to_one_below_release(self, level):
        # Chatter below A_RELEASE must be pure quiet: r recovers to 1 and
        # stays there (the floor-pinning failure mode the review caught).
        signals = {"clamp_bind_fraction":
                   step_control.SIGNAL_BANDS["clamp_bind_fraction"][0]
                   + level * 0.08}
        assert step_control.combined_alarm(signals) < step_control.A_RELEASE
        state = step_control.ControllerState(r=0.3)
        for _ in range(40):
            r = step_control.update(state, signals, 1.0, 0.25)
        assert r == 1.0

    def test_steady_state_converges_to_floor_above_release(self):
        signals = {"clamp_bind_fraction": 0.06}  # alarm ~ 0.5
        assert step_control.combined_alarm(signals) >= step_control.A_RELEASE
        state = step_control.ControllerState.fresh()
        for _ in range(40):
            r = step_control.update(state, signals, 1.0, 0.25)
        assert r == 0.25

    def test_quiet_counter_does_not_reset_on_raise(self):
        state = step_control.ControllerState(r=0.2)
        quiet = {"clamp_bind_fraction": 0.0}
        rs = [step_control.update(state, quiet, 1.0, 0.1)
              for _ in range(10)]
        # After arming (3 quiet substeps), every subsequent step raises:
        # geometric recovery, not one raise per three steps.
        armed = rs[step_control.QUIET_STEPS - 1 :]
        for before, after in zip(armed[:-1], armed[1:]):
            assert after == pytest.approx(
                min(1.0, before * step_control.BETA_UP))

    def test_init_asymmetry(self):
        assert step_control.ControllerState.fresh().r == 1.0
        assert step_control.ControllerState.restored(0.25).r == 0.25

    def test_dt_decrease_masks_contaminated_signals(self):
        state = step_control.ControllerState.fresh()
        step_control.update(state, {}, 1.0, 0.25)
        # A halved dt would spike clamp-bind purely as a controller
        # artifact; the mask must suppress it for that substep.
        masked = step_control.masked_signals(
            {"clamp_bind_fraction": 1.0, "near_solid_fast_fraction": 0.5},
            state, 0.5)
        assert "clamp_bind_fraction" not in masked
        assert "near_solid_fast_fraction" in masked
        unmasked = step_control.masked_signals(
            {"clamp_bind_fraction": 1.0}, state, 0.9)
        assert "clamp_bind_fraction" in unmasked


class TestVmaxPredictor:
    def test_growth_capped_and_floored(self):
        state = step_control.ControllerState.fresh()
        assert step_control.predicted_vmax(state, 1.0) == 1.0
        assert step_control.predicted_vmax(state, 10.0) == pytest.approx(
            10.0 * step_control.G_CAP)
        # Decaying velocity must not shrink the prediction below vmax.
        assert step_control.predicted_vmax(state, 1.0) == 1.0

    def test_combined_floor_bounds_reduction(self):
        # The combined controller + predictor reduction must never drop dt
        # below the slider floor relative to the UNPREDICTED velocity.
        state = step_control.ControllerState(r=0.25)
        state.vmax_history = [1.0]
        dt = step_control.effective_dt_candidate(
            state, 2.0, 16.0, 1.0 / 32.0, 1.0, strength=0.5)
        floor = step_control.r_floor(16.0, 0.5)
        assert dt >= floor * 16.0 * (1.0 / 32.0) / 2.0 - 1e-12


class TestSolverDiagnostics:
    def test_off_by_default_and_bitwise_identical(self):
        n = 12
        plain = STFLIPSolver(_dam_params(n), get_backend("cpu"))
        plain.add_liquid_mask(_dam_mask(n))
        stats = None
        for _ in range(2):
            stats = plain.step_frame()
        assert stats.clamp_bind_fractions == []
        assert stats.capillary_clamped_steps == []
        baseline = _state_hash(plain)

        diag = STFLIPSolver(_dam_params(n), get_backend("cpu"))
        diag.add_liquid_mask(_dam_mask(n))
        diag._collect_step_diagnostics = True
        for _ in range(2):
            stats = diag.step_frame()
        assert _state_hash(diag) == baseline
        assert len(stats.clamp_bind_fractions) == stats.steps
        assert len(stats.undersampled_marginal_fractions) == stats.steps
        assert len(stats.near_solid_fast_fractions) == stats.steps
        assert all(0.0 <= v <= 1.0 for v in stats.clamp_bind_fractions)
        assert all(
            0.0 <= v <= 1.0
            for v in stats.undersampled_marginal_fractions)

    def test_capillary_regime_flagged(self):
        n = 12
        params = _dam_params(
            n, surface_tension=50.0, gravity=(0.0, 0.0, 0.0))
        solver = STFLIPSolver(params, get_backend("cpu"))
        solver.add_liquid_mask(_dam_mask(n))
        solver._collect_step_diagnostics = True
        stats = solver.step_frame()
        assert len(stats.capillary_clamped_steps) == stats.steps
        assert any(stats.capillary_clamped_steps)

    def test_near_solid_signal_fires_for_fast_jet_at_wall(self):
        n = 16
        params = _dam_params(n, gravity=(0.0, 0.0, 0.0), cfl_target=16.0)
        solver = STFLIPSolver(params, get_backend("cpu"))
        mask = np.zeros((n,) * 3, dtype=bool)
        mask[2:5, 6:10, 6:10] = True
        solver.add_liquid_mask(mask, velocity=(6.0, 0.0, 0.0))
        cells = (np.stack(np.meshgrid(
            *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
        sdf = (0.55 - cells[..., 0]).astype(np.float32)  # wall at x=0.55
        solver.set_solid_sdf(-sdf)
        solver._collect_step_diagnostics = True
        fired = 0.0
        for _ in range(3):
            stats = solver.step_frame()
            fired = max(fired, max(stats.near_solid_fast_fractions))
        assert fired > 0.0
