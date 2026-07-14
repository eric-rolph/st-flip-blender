"""Tests for the CALM-M1 calm-surface height metrics and scene runner."""

import math
import sys

import numpy as np
import pytest

from stflip.metrics import (
    height_map_stats,
    particle_height_map,
    process_peak_memory_bytes,
    surface_height_map,
)
from stflip.validation import (
    CALM_SURFACE_SCENES,
    CALM_SURFACE_SCHEMA,
    CalmSurfaceConfig,
    run_calm_surface_validation,
)


def _lattice_positions(resolution: int, fill_z_cells: int, ppc_axis: int):
    """Regular lattice with ppc_axis**3 particles per cell below a plane."""

    dx = 1.0 / resolution
    per_axis = np.arange(ppc_axis) + 0.5
    offsets = np.stack(np.meshgrid(
        per_axis, per_axis, per_axis, indexing="ij"), axis=-1)
    offsets = offsets.reshape(-1, 3) / ppc_axis
    cells = np.stack(np.meshgrid(
        np.arange(resolution),
        np.arange(resolution),
        np.arange(fill_z_cells),
        indexing="ij"), axis=-1).reshape(-1, 3)
    positions = (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 3) * dx
    return positions.astype(np.float64)


class TestParticleHeightMap:
    def test_absolute_height_of_half_filled_box(self):
        # A half-filled unit box must report its analytic fill depth to
        # within half a cell; an off-by-ppc scale error cannot slip past.
        resolution = 8
        dx = 1.0 / resolution
        positions = _lattice_positions(resolution, resolution // 2, 2)
        heights = particle_height_map(
            positions,
            dx=dx,
            resolution=(resolution,) * 3,
            particles_per_cell=8,
        )
        assert heights.shape == (resolution, resolution)
        assert np.all(np.abs(heights - 0.5) <= dx / 2 + 1e-12)

    def test_flat_lattice_surface_has_zero_rms(self):
        # Lattice-regular seeding gives every column the same count; the
        # flatness bound below assumes that regularity.
        positions = _lattice_positions(8, 4, 2)
        heights = particle_height_map(
            positions, dx=1.0 / 8, resolution=(8, 8, 8),
            particles_per_cell=8)
        stats = height_map_stats(heights)
        assert stats["wet_column_count"] == 64
        assert stats["height_rms_spatial"] < 1e-9

    def test_single_particle_moves_height_subvoxel(self):
        resolution = 8
        dx = 1.0 / resolution
        base = _lattice_positions(resolution, 4, 2)
        extra = np.array([[0.5 * dx, 0.5 * dx, 0.6]])
        with_extra = np.concatenate([base, extra], axis=0)
        h0 = particle_height_map(
            base, dx=dx, resolution=(resolution,) * 3, particles_per_cell=8)
        h1 = particle_height_map(
            with_extra, dx=dx, resolution=(resolution,) * 3,
            particles_per_cell=8)
        delta = h1 - h0
        assert delta[0, 0] == pytest.approx(dx / 8)
        assert np.count_nonzero(delta) == 1

    def test_floor_z_excludes_particles_below(self):
        positions = np.array([[0.05, 0.05, -0.5], [0.05, 0.05, 0.5]])
        heights = particle_height_map(
            positions, dx=0.1, resolution=(10, 10, 10),
            particles_per_cell=1, floor_z=0.0)
        assert heights[0, 0] == pytest.approx(0.1)

    def test_rejects_bad_inputs(self):
        good = np.zeros((1, 3))
        with pytest.raises(ValueError):
            particle_height_map(
                np.zeros((3,)), dx=0.1, resolution=(4, 4, 4),
                particles_per_cell=1)
        with pytest.raises(ValueError):
            particle_height_map(
                good, dx=0.0, resolution=(4, 4, 4), particles_per_cell=1)
        with pytest.raises(ValueError):
            particle_height_map(
                good, dx=0.1, resolution=(4, 4), particles_per_cell=1)
        with pytest.raises(ValueError):
            particle_height_map(
                good, dx=0.1, resolution=(4, 4, 4), particles_per_cell=0)
        with pytest.raises(ValueError):
            particle_height_map(
                np.full((1, 3), np.nan), dx=0.1, resolution=(4, 4, 4),
                particles_per_cell=1)


class TestSurfaceHeightMap:
    def test_recovers_plane_by_linear_interpolation(self):
        nx, ny, nz = 4, 5, 20
        voxel = 0.05
        origin = (0.0, 0.0, 0.0)
        density = np.zeros((nx, ny, nz), dtype=np.float64)
        density[:, :, :10] = 1.0
        density[:, :, 10] = 0.3
        heights = surface_height_map(
            density, origin=origin, voxel_size=voxel)
        # Crossing sits between voxel centres 9 (1.0) and 10 (0.3).
        expected = (9 + 0.5 + (1.0 - 0.5) / (1.0 - 0.3)) * voxel
        assert np.allclose(heights, expected, atol=1e-12)

    def test_dry_column_is_nan(self):
        density = np.zeros((2, 2, 6))
        density[0, 0, :2] = 1.0
        heights = surface_height_map(
            density, origin=(0.0, 0.0, 0.0), voxel_size=1.0)
        assert math.isfinite(heights[0, 0])
        assert np.isnan(heights[1, 1])

    def test_saturated_top_reports_upper_face(self):
        nz = 4
        density = np.ones((2, 2, nz))
        heights = surface_height_map(
            density, origin=(0.0, 0.0, 0.0), voxel_size=1.0)
        assert np.allclose(heights, float(nz))

    def test_origin_offset_is_applied(self):
        density = np.zeros((1, 1, 4))
        density[0, 0, :2] = 1.0
        shifted = surface_height_map(
            density, origin=(0.0, 0.0, 3.0), voxel_size=1.0)
        base = surface_height_map(
            density, origin=(0.0, 0.0, 0.0), voxel_size=1.0)
        assert shifted[0, 0] == pytest.approx(base[0, 0] + 3.0)

    def test_rejects_bad_inputs(self):
        density = np.zeros((2, 2, 2))
        with pytest.raises(ValueError):
            surface_height_map(
                np.zeros((2, 2)), origin=(0, 0, 0), voxel_size=1.0)
        with pytest.raises(ValueError):
            surface_height_map(density, origin=(0, 0), voxel_size=1.0)
        with pytest.raises(ValueError):
            surface_height_map(density, origin=(0, 0, 0), voxel_size=0.0)


class TestHeightMapStats:
    def test_reference_subtraction(self):
        values = np.array([[1.0, 2.0], [3.0, np.nan]])
        reference = np.array([[1.0, 1.0], [1.0, 1.0]])
        stats = height_map_stats(values, reference_map=reference)
        deviations = np.array([0.0, 1.0, 2.0])
        assert stats["wet_column_count"] == 3
        assert stats["height_mean"] == pytest.approx(deviations.mean())
        assert stats["height_rms_spatial"] == pytest.approx(
            np.sqrt(np.mean((deviations - deviations.mean()) ** 2)))

    def test_all_dry_returns_none(self):
        stats = height_map_stats(np.full((2, 2), np.nan))
        assert stats["wet_column_count"] == 0
        assert stats["height_mean"] is None
        assert stats["height_rms_spatial"] is None

    def test_reference_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            height_map_stats(
                np.zeros((2, 2)), reference_map=np.zeros((3, 2)))


def test_process_peak_memory_probe():
    peak = process_peak_memory_bytes()
    # Windows, Linux, and macOS all have a cheap peak-RSS source; only
    # exotic platforms may return None.
    if sys.platform in {"win32", "linux", "darwin"}:
        assert isinstance(peak, int) and peak > 0
    else:
        assert peak is None or (isinstance(peak, int) and peak > 0)


class TestCalmSurfaceConfig:
    def test_rejects_unknown_scene(self):
        with pytest.raises(ValueError):
            CalmSurfaceConfig(scenes=("still_pool", "tsunami"))

    def test_rejects_bad_integers(self):
        with pytest.raises(ValueError):
            CalmSurfaceConfig(resolution=4)
        with pytest.raises(ValueError):
            CalmSurfaceConfig(frames=0)
        with pytest.raises(ValueError):
            CalmSurfaceConfig(surface_iterations=-1)
        with pytest.raises(ValueError):
            CalmSurfaceConfig(resolution=True)

    def test_scene_registry_is_stable(self):
        assert CALM_SURFACE_SCENES == (
            "still_pool",
            "stirred_pool",
            "ballistic_droplet",
            "translating_slab",
        )


@pytest.fixture(scope="module")
def still_pool_report():
    config = CalmSurfaceConfig(
        resolution=12,
        frames=2,
        particles_per_cell=2,
        cfl_target=16.0,
        scenes=("still_pool",),
        probe_count=4,
        surface_iterations=2,
    )
    return run_calm_surface_validation(config)


class TestRunner:
    def test_report_structure(self, still_pool_report):
        report = still_pool_report
        assert report["schema"] == CALM_SURFACE_SCHEMA
        scene = report["scenes"]["still_pool"]
        assert len(scene["frames"]) == 2
        assert len(scene["probe_columns"]) == 4
        frame = scene["frames"][0]
        assert frame["particle_count"] > 0
        assert frame["particle_height"]["wet_column_count"] == 144
        assert frame["render_height"]["wet_column_count"] > 0
        for key in ("step_wall_s", "render_resynchronization_wall_s",
                    "surface_reconstruction_wall_s"):
            assert frame["timing"][key] >= 0.0
        assert len(frame["probe_heights_particle"]) == 4
        assert frame["whitewater_counts"] is None

    def test_still_pool_stays_near_fill_height(self, still_pool_report):
        scene = still_pool_report["scenes"]["still_pool"]
        summary = scene["summary"]
        dx = 1.0 / 12
        for label in ("particle", "render"):
            block = summary[label]
            assert block["wet_column_count"] > 0
            # A resting pool must not drift by more than a cell over the
            # short run, and its temporal wobble stays sub-voxel.
            assert abs(block["height_drift"]) < dx
            assert block["height_std_temporal_mean"] < dx
        for frame in scene["frames"]:
            mean = frame["particle_height"]["height_mean"]
            assert abs(mean - 0.5) < 2 * dx

    def test_whitewater_arm_reports_counts(self):
        config = CalmSurfaceConfig(
            resolution=12,
            frames=1,
            particles_per_cell=2,
            scenes=("still_pool",),
            probe_count=2,
            surface_iterations=0,
            whitewater=True,
        )
        report = run_calm_surface_validation(config)
        counts = report["scenes"]["still_pool"]["frames"][0][
            "whitewater_counts"]
        assert set(counts) >= {"total", "foam", "bubble", "spray"}

    def test_translating_slab_scene_builds(self):
        config = CalmSurfaceConfig(
            resolution=12,
            frames=1,
            particles_per_cell=2,
            scenes=("translating_slab", "ballistic_droplet"),
            probe_count=2,
            surface_iterations=0,
        )
        report = run_calm_surface_validation(config)
        for name in ("translating_slab", "ballistic_droplet"):
            scene = report["scenes"][name]
            assert scene["params"]["gravity_z"] == 0.0
            assert scene["frames"][0]["particle_count"] > 0
