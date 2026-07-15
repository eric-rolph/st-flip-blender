"""CAP-M3: surface-tension physics validation.

Calibration evidence (2026-07-14, v0.38): Rayleigh n=2 period measured 34
frames vs 31.8 theoretical (7 percent); parasitic currents at scale 8 with
the stabilizer measured 0.63x the scale-1 explicit reference (gate allows
3x).  Overdamping at high clamp scales is the accepted accuracy-for-speed
trade; GROWTH is the failure mode.
"""

import math

import numpy as np
import pytest

from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver

RHO = 1000.0


def _sphere_mask(n, radius, stretch_z=1.0, squeeze_xy=1.0,
                 centre=(0.5, 0.5, 0.5)):
    cells = (np.stack(np.meshgrid(
        *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
    d = cells - np.asarray(centre)
    r_eff = np.sqrt(
        (d[..., 0] / squeeze_xy) ** 2 + (d[..., 1] / squeeze_xy) ** 2
        + (d[..., 2] / stretch_z) ** 2)
    return r_eff <= radius


def _params(n, sigma, scale, implicit, **overrides):
    base = dict(
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, 0.0),
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=4,
        st_enabled=True, seed=1, surface_tension=sigma,
        st_clamp_scale=scale, st_implicit=implicit,
    )
    base.update(overrides)
    return Params(**base)


def _brackbill_dt(sigma, dx, rho=RHO):
    return math.sqrt(rho * dx ** 3 / (4.0 * math.pi * sigma))


class TestClampArithmetic:
    def test_substep_count_matches_scaled_clamp(self):
        # Quantized assertion from the roadmap: on a clamp-bound config
        # the substep count is exactly ceil(frame_dt / min(dt_CFL,
        # scale * dt_cap)).
        n = 32
        sigma = 100.0
        frame_dt = 1.0 / 24.0
        for scale in (1.0, 4.0):
            params = _params(n, sigma, scale, False, frame_dt=frame_dt)
            solver = STFLIPSolver(params, get_backend("cpu"))
            solver.add_liquid_mask(_sphere_mask(n, 0.2))
            stats = solver.step_frame()
            dt_cap = scale * _brackbill_dt(sigma, params.dx)
            # The droplet starts at rest, so dt_CFL is frame-bound.
            expected = math.ceil(frame_dt / min(frame_dt, dt_cap))
            assert stats.steps == expected


@pytest.mark.slow
class TestLaplaceJump:
    @pytest.mark.parametrize("scale,implicit", [(1.0, False), (8.0, True)])
    def test_pressure_jump_within_25_percent(self, scale, implicit):
        # Young-Laplace: delta_p = 2 sigma / R across a spherical
        # interface; air is the Dirichlet p = 0 gauge, so the interior
        # plateau IS the jump.  The droplet must be resolved WELL past
        # the phase-smoothing width or the CSF curvature systematically
        # under-reads (measured: a 3.5-cell radius at 16^3 reports 68
        # percent of theory; a 7-cell radius at 24^3 sits inside the
        # gate).
        n = 24
        sigma = 30.0
        radius = 0.3
        solver = STFLIPSolver(
            _params(n, sigma, scale, implicit), get_backend("cpu"))
        solver.add_liquid_mask(_sphere_mask(n, radius))
        solver._collect_step_diagnostics = True
        for _ in range(2):
            solver.step_frame()
        pressure = solver.be.to_numpy(solver._last_pressure)
        cells = (np.stack(np.meshgrid(
            *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
        r = np.linalg.norm(cells - 0.5, axis=-1)
        interior = pressure[r < 0.5 * radius]
        assert interior.size
        jump = float(interior.mean())
        theory = 2.0 * sigma / radius
        assert jump == pytest.approx(theory, rel=0.25)


@pytest.mark.slow
class TestRayleighOscillation:
    def test_mode2_period_within_30_percent_and_decaying(self):
        n = 16
        sigma = 30.0
        radius = 0.22
        omega = math.sqrt(8.0 * sigma / (RHO * radius ** 3))
        theory_frames = 2.0 * math.pi / omega * 24.0
        solver = STFLIPSolver(
            _params(n, sigma, 4.0, True), get_backend("cpu"))
        solver.add_liquid_mask(_sphere_mask(
            n, radius, stretch_z=1.15, squeeze_xy=0.87))
        aspects = []
        for _ in range(40):
            solver.step_frame()
            pos = solver.be.to_numpy(solver.pos) - 0.5
            izz = float((pos[:, 2] ** 2).mean())
            ixx = float((pos[:, 0] ** 2 + pos[:, 1] ** 2).mean() / 2.0)
            aspects.append(math.sqrt(izz / ixx))
        trace = np.asarray(aspects)
        crossings = np.nonzero(np.diff(np.sign(trace - 1.0)))[0]
        assert len(crossings) >= 2, "no oscillation detected"
        period_frames = 2.0 * float(np.diff(crossings).mean())
        assert period_frames == pytest.approx(theory_frames, rel=0.30)
        # Overdamping is the accepted trade; growth is the failure.
        early = float(np.abs(trace[:10] - 1.0).max())
        late = float(np.abs(trace[-10:] - 1.0).max())
        assert late < early


@pytest.mark.slow
class TestParasiticCurrents:
    def test_relaxed_clamp_bounded_by_reference(self):
        n = 16

        def peak_speed(scale, implicit):
            solver = STFLIPSolver(
                _params(n, 30.0, scale, implicit), get_backend("cpu"))
            solver.add_liquid_mask(_sphere_mask(n, 0.22))
            peaks = []
            for _ in range(12):
                solver.step_frame()
                v = np.linalg.norm(
                    solver.be.to_numpy(solver.vel), axis=1)
                peaks.append(float(v.max()))
            return float(np.mean(peaks[-4:]))

        reference = peak_speed(1.0, False)
        relaxed = peak_speed(8.0, True)
        assert relaxed <= 3.0 * reference


@pytest.mark.slow
class TestTwoPhaseStability:
    def test_800_to_1_ratio_at_scale_8_stays_bounded(self):
        # Density ratios enter the stabilizer only through R; the CG must
        # tolerate the diagonal contrast at production ratios with bounded
        # iteration counts.
        n = 16
        params = _params(
            n, 30.0, 8.0, True, two_phase=True, rho_gas=1.25,
            gravity=(0.0, 0.0, -9.81), pressure_solver="multigrid")
        solver = STFLIPSolver(params, get_backend("cpu"))
        mask = np.zeros((n,) * 3, dtype=bool)
        mask[:, :, : n // 2] = True
        solver.add_liquid_mask(mask)
        solver.fill_gas()
        iters = []
        for _ in range(4):
            stats = solver.step_frame()
            iters.extend(stats.st_cg_iters)
        vel = solver.be.to_numpy(solver.vel)
        assert np.all(np.isfinite(vel))
        assert iters, "stabilizer never ran"
        assert max(iters) <= 200
