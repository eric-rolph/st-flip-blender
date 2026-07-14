"""CALM-M2: render-path calm-region denoise."""

import numpy as np
import pytest

from stflip import metrics, surface


def _noisy_plane(n=24, level=None, noise=0.18, seed=2):
    """A bounded density field: liquid below a plane, with surface noise."""

    level = n // 2 if level is None else level
    rng = np.random.default_rng(seed)
    density = np.zeros((n, n, n), dtype=np.float32)
    density[:, :, :level] = 1.0
    band = slice(max(level - 2, 0), min(level + 2, n))
    density[:, :, band] += rng.normal(
        0.0, noise, density[:, :, band].shape).astype(np.float32)
    return np.clip(density, 0.0, 1.0)


def _flat_mask(shape):
    return np.ones(shape, dtype=np.float32)


class TestBounds:
    def test_erode_dilate_envelopes(self):
        rng = np.random.default_rng(0)
        values = rng.random((8, 9, 10)).astype(np.float32)
        lower, upper = surface.erode_dilate_bounds(values)
        assert np.all(lower <= values)
        assert np.all(values <= upper)
        # 3x3x3 neighbourhood extremes at an interior point.
        assert lower[4, 4, 4] == values[3:6, 3:6, 3:6].min()
        assert upper[4, 4, 4] == values[3:6, 3:6, 3:6].max()


class TestCalmRegionSmooth:
    def test_zero_iterations_is_identity(self):
        density = _noisy_plane()
        out = surface.calm_region_smooth(
            density, _flat_mask(density.shape), extra_iterations=0)
        assert np.array_equal(out, density)

    def test_result_stays_inside_band(self):
        density = _noisy_plane()
        lower, upper = surface.erode_dilate_bounds(density)
        out = surface.calm_region_smooth(
            density, _flat_mask(density.shape), extra_iterations=20)
        assert np.all(out >= lower - 1e-6)
        assert np.all(out <= upper + 1e-6)

    def test_mean_surface_position_preserved(self):
        # Individual noise-spike columns legitimately move more than a
        # voxel (that IS the denoising); the band clamp's promise is that
        # the surface as a whole does not migrate.
        density = _noisy_plane()
        out = surface.calm_region_smooth(
            density, _flat_mask(density.shape), extra_iterations=20)
        before = metrics.surface_height_map(
            density, origin=(0.0, 0.0, 0.0), voxel_size=1.0)
        after = metrics.surface_height_map(
            out, origin=(0.0, 0.0, 0.0), voxel_size=1.0)
        both = np.isfinite(before) & np.isfinite(after)
        assert both.any()
        drift = float(np.abs(
            after[both].mean() - before[both].mean()))
        assert drift <= 0.5

    def test_reduces_calm_surface_noise(self):
        density = _noisy_plane()
        out = surface.calm_region_smooth(
            density, _flat_mask(density.shape), extra_iterations=20)
        before = metrics.height_map_stats(metrics.surface_height_map(
            density, origin=(0.0, 0.0, 0.0), voxel_size=1.0))
        after = metrics.height_map_stats(metrics.surface_height_map(
            out, origin=(0.0, 0.0, 0.0), voxel_size=1.0))
        assert after["height_rms_spatial"] < 0.5 * before["height_rms_spatial"]

    @staticmethod
    def _sphere(n, r_vox):
        cells = (np.stack(np.meshgrid(
            *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5)
        radius = np.linalg.norm(cells - n / 2.0, axis=-1)
        return np.clip((r_vox - radius) / 2.0 + 0.5, 0.0, 1.0).astype(
            np.float32)

    @staticmethod
    def _real_mask(density):
        blurred = surface.gaussian_blur(
            density, sigma_voxels=surface.GAUSSIAN_SIGMA_DX
            / surface.VOXEL_SIZE_DX)
        return surface.feature_mask(density, blurred)

    def test_small_droplet_protected_by_feature_mask(self):
        # A droplet smaller than the blur kernel has a high self-quotient,
        # so the feature mask (not the clamp) keeps it nearly intact.
        density = self._sphere(20, 2.5)
        out = surface.calm_region_smooth(
            density, self._real_mask(density), extra_iterations=20)
        before = int((density >= 0.5).sum())
        after = int((out >= 0.5).sum())
        assert before > 0
        assert after >= 0.95 * before

    def test_resolved_sphere_erodes_at_most_the_clamp_band(self):
        # A resolved sphere reads as calm (quotient ~ 1) and curvature flow
        # wants to shrink it; the erode envelope is the guaranteed floor --
        # the isosurface retreats at most one voxel.
        density = self._sphere(24, 6.0)
        lower, _upper = surface.erode_dilate_bounds(density)
        out = surface.calm_region_smooth(
            density, self._real_mask(density), extra_iterations=20)
        floor = int((lower >= 0.5).sum())
        after = int((out >= 0.5).sum())
        assert floor > 0
        assert after >= floor

    def test_thin_sheet_survives(self):
        # A two-voxel sheet: the band clamp bounds drift, and the REAL
        # feature mask (self-quotient) must classify it as a feature so the
        # calm pass barely touches it.
        n = 24
        density = np.zeros((n, n, n), dtype=np.float32)
        density[:, :, 11:13] = 1.0
        blurred = surface.gaussian_blur(
            density, sigma_voxels=surface.GAUSSIAN_SIGMA_DX
            / surface.VOXEL_SIZE_DX)
        mask = surface.feature_mask(density, blurred)
        out = surface.calm_region_smooth(
            density, mask, extra_iterations=20)
        before = int((density >= 0.5).sum())
        after = int((out >= 0.5).sum())
        assert after >= 0.9 * before


class TestReconstructIntegration:
    def _positions(self, seed=1):
        rng = np.random.default_rng(seed)
        base = rng.random((600, 3)).astype(np.float32)
        base[:, 2] *= 0.5
        return base * 0.5

    def test_default_zero_is_bitwise_identical(self):
        positions = self._positions()
        plain = surface.reconstruct_surface(positions, 1.0 / 16.0)
        explicit = surface.reconstruct_surface(
            positions, 1.0 / 16.0, calm_iterations=0)
        assert np.array_equal(plain.density, explicit.density)
        assert plain.origin == explicit.origin

    def test_calm_iterations_run_and_report(self):
        positions = self._positions()
        result = surface.reconstruct_surface(
            positions, 1.0 / 16.0, calm_iterations=30)
        assert result.diagnostics["calm_iterations_completed"] == 30
        # The calm pass widens the diffusion band, so padding must grow.
        plain = surface.reconstruct_surface(positions, 1.0 / 16.0)
        assert (result.diagnostics["padding_voxels"]
                > plain.diagnostics["padding_voxels"])

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            surface.reconstruct_surface(
                self._positions(), 1.0 / 16.0, calm_iterations=-1)
