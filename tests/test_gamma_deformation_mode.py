"""CALM-M4: deformation-aware jitter attenuation."""

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
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
        st_enabled=True, seed=10,
    )
    base.update(overrides)
    return Params(**base)


def _pool_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    return mask


def _run(params, setup, frames=3):
    solver = STFLIPSolver(params, get_backend("cpu"))
    setup(solver)
    for _ in range(frames):
        solver.step_frame()
    return solver


class TestParams:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            Params(resolution=(8, 8, 8), dx=0.125, gamma_mode="psychic")

    def test_speed_mode_still_bitwise_default(self):
        n = 12
        default = _run(_params(n), lambda s: s.add_liquid_mask(
            _pool_mask(n)))
        explicit = _run(_params(n, gamma_mode="speed"),
                        lambda s: s.add_liquid_mask(_pool_mask(n)))
        assert _state_hash(default) == _state_hash(explicit)


class TestStirredPoolMechanism:
    def _surface_gamma_mean(self, solver):
        gamma = solver.be.to_numpy(solver._gamma_prev)
        pos = solver.be.to_numpy(solver.pos)
        near_surface = (pos[:, 2] > 0.42) & (pos[:, 2] < 0.5)
        assert near_surface.any()
        return float(gamma[near_surface].mean())

    def test_river_surface_damps_under_deformation_gate(self):
        # The mode's real target, found empirically while building the
        # stirred-pool variant: a stirred POOL cannot reach speed-gate-
        # pinning velocities at accessible scale without destroying
        # itself, but a RIVER (uniform translation at local CFL >= 1)
        # pins the speed gate at gamma = 1 on a surface whose motion is
        # entirely tangential -- v.n ~ 0, zero strain, quasi-steady grid.
        # The deformation gate must damp that skin.
        n = 12

        def setup(s):
            mask = np.zeros((n,) * 3, dtype=bool)
            mask[n // 8 : n // 2, :, : (2 * n) // 5] = True
            s.add_liquid_mask(mask, velocity=(2.5, 0.0, 0.0))

        def top_skin(solver):
            pos = solver.be.to_numpy(solver.pos)
            return (pos[:, 2] > 0.30) & (pos[:, 2] < 0.40)

        river = dict(gravity=(0.0, 0.0, 0.0), cfl_target=8.0)
        deform = _run(
            _params(n, gamma_mode="deformation", **river), setup,
            frames=3)
        skin = top_skin(deform)
        assert skin.any()
        gamma_deform = float(
            deform.be.to_numpy(deform._gamma_prev)[skin].mean())

        speed = _run(_params(n, **river), setup, frames=3)
        skin_s = top_skin(speed)
        gamma_speed = float(speed.be.to_numpy(speed._jitter_gamma(
            float(speed._dt_prev)))[skin_s].mean())
        # Precondition: the speed gate really is pinned open (local CFL
        # = 2.5 * dt / dx >= 1 at this configuration).
        assert gamma_speed > 0.9
        assert gamma_deform < 0.7 * gamma_speed

    def test_solid_body_rotation_reads_low_strain(self):
        # The symmetric-tensor review fix: rigid rotation has D = 0, so
        # the strain term must NOT keep the gate open (the full gradient
        # norm would read sqrt(2) * omega and pin gamma at one).
        from stflip.velocity import SolidBodyRotation

        n = 12
        field = SolidBodyRotation(
            center=(0.5, 0.5, 0.5), angular_velocity=(0.0, 0.0, 3.0))
        solver = STFLIPSolver(
            _params(n, gamma_mode="deformation", gravity=(0.0, 0.0, 0.0)),
            get_backend("cpu"))
        solver.add_liquid_mask(np.ones((n,) * 3, dtype=bool),
                               velocity=field)
        for _ in range(2):
            solver.step_frame()
        gamma = solver.be.to_numpy(solver._gamma_prev)
        pos = solver.be.to_numpy(solver.pos)
        # Domain-boundary cells legitimately lose interiorness (kernel
        # clipping depresses c_phi at walls; walls are not SDF solids, so
        # the solid mask does not apply) -- assert on the interior, where
        # the claim lives: rigid rotation has D = 0 and must not collapse
        # the gate through spurious strain.
        interior = np.all(np.abs(pos - 0.5) < 0.32, axis=1)
        assert interior.any()
        assert float(gamma[interior].mean()) > 0.95


class TestFastFlowKeepsJitter:
    def _slug_front_gamma(self, n, x_slice, speed=4.0):
        params = _params(
            n, gamma_mode="deformation", gravity=(0.0, 0.0, 0.0),
            cfl_target=16.0)
        solver = STFLIPSolver(params, get_backend("cpu"))
        mask = np.zeros((n,) * 3, dtype=bool)
        mask[x_slice, n // 4 : (3 * n) // 4, n // 4 : (3 * n) // 4] = True
        solver.add_liquid_mask(mask, velocity=(speed, 0.0, 0.0))
        for _ in range(2):
            solver.step_frame()
        gamma = solver.be.to_numpy(solver._gamma_prev)
        pos = solver.be.to_numpy(solver.pos)
        edge = pos[:, 0] > pos[:, 0].max() - 1.5 / n
        assert edge.any()
        return float(gamma[edge].mean())

    def test_resolved_front_keeps_full_jitter(self):
        # The S3 guard for RESOLVED features: the phase-flux term reads
        # the advancing front's contour-crossing rate and keeps the gate
        # open (measured 0.979 for a 6-cell slug).
        assert self._slug_front_gamma(16, slice(2, 8)) > 0.9

    def test_thin_feature_undergating_is_the_documented_limitation(self):
        # KNOWN LIMITATION, measured and pinned: features thinner than
        # ~4 cells under-gate because the smoothed phase (and hence the
        # flux and interiorness inputs) washes out -- the same
        # thin-feature contract CALM-M2 documents for the render denoise.
        # Deformation mode ships default-off partly for this reason; if
        # an improvement lifts this value above 0.9, promote the mode's
        # documentation accordingly.
        thin = self._slug_front_gamma(16, slice(2, 4))
        assert 0.5 < thin < 0.9


class TestPlumbing:
    def test_drift_bound_holds(self):
        n = 12
        solver = STFLIPSolver(
            _params(n, gamma_mode="deformation"), get_backend("cpu"))
        solver.add_liquid_mask(_pool_mask(n))
        dt_max = 0.0
        for _ in range(4):
            stats = solver.step_frame()
            dt_max = max(dt_max, *(stats.dt_values or (0.0,)))
            resid = solver.be.to_numpy(solver.dt_resid)
            assert float(np.abs(resid).max()) <= 0.5 * dt_max + 1e-9

    def test_m0_plateau_holds_with_persisted_gamma(self):
        n = 12
        params = _params(
            n, gravity=(0.0, 0.0, 0.0), gamma_mode="deformation",
            particles_per_cell=8, cfl_target=4.0)
        solver = _run(
            params,
            lambda s: s.add_liquid_mask(np.ones((n,) * 3, dtype=bool)),
            frames=2)
        grids = solver._p2g(solver._dt_prev)
        interior = solver.be.to_numpy(
            grids["c_m"])[2:-2, 2:-2, 2:-2] / solver.m0
        assert float(interior.mean()) == pytest.approx(1.0, rel=0.01)

    def test_checkpoint_carries_gamma_prev(self):
        n = 12
        solver = _run(
            _params(n, gamma_mode="deformation"),
            lambda s: s.add_liquid_mask(_pool_mask(n)), frames=2)
        state = solver.checkpoint_state()
        assert "gamma_prev" in state

    def test_reflection_composes(self):
        n = 12
        solver = _run(
            _params(n, gamma_mode="deformation", reflection=True),
            lambda s: s.add_liquid_mask(_pool_mask(n)), frames=2)
        vel = solver.be.to_numpy(solver.vel)
        assert np.all(np.isfinite(vel))
