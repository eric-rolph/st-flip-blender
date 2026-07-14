"""Tests for SAMP-M1 stable particle ids and the checkpoint v3 bump."""

import numpy as np
import pytest

from stflip import cache
from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver


def _solver(**overrides):
    n = overrides.pop("resolution", 12)
    values = {
        "resolution": (n, n, n),
        "dx": 1.0 / n,
        "gravity": (0.0, 0.0, -9.81),
        "frame_dt": 1.0 / 24.0,
        "cfl_target": 8.0,
        "particles_per_cell": 2,
        "st_enabled": True,
        "seed": 0,
    }
    values.update(overrides)
    solver = STFLIPSolver(Params(**values), get_backend("cpu"))
    return solver


def _ids(solver):
    return solver.be.to_numpy(solver.particle_id)


def _pool_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    return mask


class TestIdAllocation:
    def test_seeding_allocates_monotonic_unique_ids(self):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        ids = _ids(solver)
        assert ids.dtype == np.int64
        assert len(ids) == solver.pos.shape[0]
        np.testing.assert_array_equal(ids, np.arange(len(ids)))
        assert solver._next_particle_id == len(ids)

    def test_two_sources_never_share_ids(self):
        solver = _solver()
        n = 12
        lower = np.zeros((n,) * 3, dtype=bool)
        lower[:, :, : n // 4] = True
        upper = np.zeros((n,) * 3, dtype=bool)
        upper[:, :, n // 2: (3 * n) // 4] = True
        solver.add_liquid_mask(lower)
        solver.add_liquid_mask(upper)
        ids = _ids(solver)
        assert np.unique(ids).size == ids.size

    def test_ids_survive_stepping_without_duplication(self):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        before = set(_ids(solver).tolist())
        for _ in range(2):
            solver.step_frame()
        after = _ids(solver)
        assert np.unique(after).size == after.size
        assert set(after.tolist()) <= before  # no outflow: same particles

    def test_outflow_compaction_keeps_id_alignment(self):
        # Remove a slice of particles through the shared keep-mask path and
        # check ids stay aligned with positions (same subset, same order).
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        xp = solver.be.xp
        n = solver.pos.shape[0]
        ids_before = _ids(solver)
        pos_before = solver.be.to_numpy(solver.pos)
        volume_removed = xp.zeros((n,), dtype=bool)
        volume_removed[::3] = True
        pressure_removed = xp.zeros((n,), dtype=bool)
        solver._apply_outflow_filter(
            solver.pos, volume_removed, pressure_removed)
        keep = ~solver.be.to_numpy(volume_removed)
        np.testing.assert_array_equal(_ids(solver), ids_before[keep])
        np.testing.assert_array_equal(
            solver.be.to_numpy(solver.pos), pos_before[keep])

    def test_reconcile_pads_with_fresh_ids(self):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        xp = solver.be.xp
        ids_before = _ids(solver)
        # Simulate a caller appending raw positions without attributes.
        extra = xp.zeros((5, 3), dtype=xp.float32) + 0.5
        solver.pos = xp.concatenate([solver.pos, extra])
        solver.vel = xp.concatenate(
            [solver.vel, xp.zeros((5, 3), dtype=xp.float32)])
        solver.dt_resid = xp.concatenate(
            [solver.dt_resid, xp.zeros((5,), dtype=xp.float32)])
        solver._reconcile_particle_attrs()
        ids_after = _ids(solver)
        assert len(ids_after) == len(ids_before) + 5
        assert np.unique(ids_after).size == ids_after.size
        assert ids_after[-5:].min() > ids_before.max()

    def test_substep_counter_advances_per_substep(self):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        assert solver._substep_index == 0
        stats = solver.step_frame()
        assert solver._substep_index == stats.steps


class TestCheckpointV3:
    def test_round_trip_preserves_ids_and_counters(self, tmp_path):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        solver.step_frame()
        state = solver.checkpoint_state()
        cache.write_checkpoint(str(tmp_path), 1, state)
        restored_state = cache.read_checkpoint(str(tmp_path), 1)

        twin = _solver()
        twin.restore_state(restored_state)
        np.testing.assert_array_equal(_ids(twin), _ids(solver))
        assert twin._next_particle_id == solver._next_particle_id
        assert twin._substep_index == solver._substep_index

    def test_archive_is_version_3(self, tmp_path):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        cache.write_checkpoint(str(tmp_path), 0, solver.checkpoint_state())
        with np.load(cache.checkpoint_path(str(tmp_path), 0)) as data:
            assert int(data["version"]) == 3 == cache.CHECKPOINT_VERSION
            assert "particle_id" in data.files
            assert "gamma_prev" not in data.files  # reserved, mode-gated

    def test_v2_archive_restores_with_synthesized_ids(self, tmp_path):
        # A pre-bump archive (no id members) must restore in the documented
        # fallback mode: ids 0..n-1, counter n, substep 0.
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        solver.step_frame()
        state = solver.checkpoint_state()
        for key in ("particle_id", "next_particle_id", "substep_index"):
            state.pop(key)
        twin = _solver()
        twin.restore_state(state)
        count = twin.pos.shape[0]
        np.testing.assert_array_equal(_ids(twin), np.arange(count))
        assert twin._next_particle_id == count
        assert twin._substep_index == 0

    def test_duplicate_ids_are_rejected(self):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        state = solver.checkpoint_state()
        state["particle_id"][1] = state["particle_id"][0]
        twin = _solver()
        with pytest.raises(ValueError):
            twin.restore_state(state)
        with pytest.raises(cache.CheckpointError):
            cache.validate_checkpoint_state(state)

    def test_resume_continues_allocating_unique_ids(self, tmp_path):
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        solver.step_frame()
        state = solver.checkpoint_state()

        twin = _solver()
        twin.restore_state(state)
        restored = set(_ids(twin).tolist())
        n = 12
        upper = np.zeros((n,) * 3, dtype=bool)
        upper[:, :, (3 * n) // 4:] = True
        twin.add_liquid_mask(upper)
        ids = _ids(twin)
        assert np.unique(ids).size == ids.size
        fresh = set(ids.tolist()) - restored
        assert fresh and min(fresh) >= twin.pos.shape[0] - len(fresh)

    def test_gamma_prev_reserved_key_round_trips(self, tmp_path):
        # The reserved mode-gated member validates and round-trips when a
        # future milestone writes it, and its range is enforced.
        solver = _solver()
        solver.add_liquid_mask(_pool_mask(12))
        state = solver.checkpoint_state()
        count = state["pos"].shape[0]
        state["gamma_prev"] = np.full((count,), 0.5, dtype=np.float32)
        cache.write_checkpoint(str(tmp_path), 2, state)
        restored = cache.read_checkpoint(str(tmp_path), 2)
        np.testing.assert_array_equal(
            restored["gamma_prev"], state["gamma_prev"])
        bad = dict(state)
        bad["gamma_prev"] = np.full((count,), 1.5, dtype=np.float32)
        with pytest.raises(cache.CheckpointError):
            cache.validate_checkpoint_state(bad)

    def test_trajectory_unchanged_by_id_maintenance(self):
        # Ids are bookkeeping: two identically seeded solvers, one with ids
        # sliced mid-run (forcing reconcile padding), must still agree --
        # the physics arrays never read particle_id.
        a = _solver()
        a.add_liquid_mask(_pool_mask(12))
        b = _solver()
        b.add_liquid_mask(_pool_mask(12))
        b.particle_id = b.particle_id[: b.particle_id.shape[0] // 2]
        b._reconcile_particle_attrs()
        for _ in range(2):
            a.step_frame()
            b.step_frame()
        np.testing.assert_array_equal(
            a.be.to_numpy(a.pos), b.be.to_numpy(b.pos))
        np.testing.assert_array_equal(
            a.be.to_numpy(a.vel), b.be.to_numpy(b.vel))
