import math

import numpy as np
import pytest

from stflip.surface import (
    DEFAULT_MCF_ITERATIONS,
    FEATURE_EPSILON,
    FEATURE_SLOPE,
    FEATURE_THRESHOLD,
    GAUSSIAN_SIGMA_DX,
    GAUSSIAN_TRUNCATE,
    SPHERE_RADIUS_DX,
    VOXEL_SIZE_DX,
    feature_mask,
    gaussian_blur,
    mean_curvature_flow,
    reconstruct_surface,
    sphere_ramp,
)


def _index_at(result, point):
    index = np.rint(
        (np.asarray(point) - np.asarray(result.origin)) / result.voxel_size
    ).astype(int)
    sample = np.asarray(result.origin) + index * result.voxel_size
    np.testing.assert_allclose(sample, point, rtol=0.0, atol=1e-7)
    return tuple(index)


def test_appendix_b_constants_and_exact_feature_mask_formula():
    assert VOXEL_SIZE_DX == 0.5
    assert SPHERE_RADIUS_DX == 0.5
    assert GAUSSIAN_SIGMA_DX == 2.0
    assert GAUSSIAN_TRUNCATE == 4.0
    assert FEATURE_THRESHOLD == 2.0
    assert FEATURE_SLOPE == 5.0
    assert FEATURE_EPSILON == 1e-10
    assert DEFAULT_MCF_ITERATIONS == 30

    density = np.asarray([0.0, 0.2, 0.35, 0.5, 1.0], dtype=np.float32)
    density = density.reshape(5, 1, 1)
    blurred = np.full_like(density, 0.1)
    quotient = density / (blurred + np.float32(FEATURE_EPSILON))
    t = np.clip(
        (quotient - FEATURE_THRESHOLD) / (FEATURE_SLOPE - FEATURE_THRESHOLD),
        0.0,
        1.0,
    )
    expected = 1.0 - t * t * (3.0 - 2.0 * t)

    actual = feature_mask(density, blurred)

    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    assert actual[0, 0, 0] == 1.0
    assert actual[-1, 0, 0] == 0.0


def test_sphere_ramp_is_half_exactly_at_radius_and_has_one_voxel_width():
    radius = 0.5
    h = 0.5
    distance = np.asarray(
        [radius - h / 2, radius, radius + h / 2], dtype=np.float32)

    density = sphere_ramp(distance, radius, h)

    np.testing.assert_array_equal(density, [1.0, 0.5, 0.0])


def test_single_particle_reconstruction_places_half_iso_at_sphere_radius():
    result = reconstruct_surface(
        np.zeros((1, 3), dtype=np.float32),
        dx=1.0,
        iterations=0,
        padding_voxels=1,
        max_voxels=10_000,
    )
    center = _index_at(result, (0.0, 0.0, 0.0))
    surface = _index_at(result, (0.5, 0.0, 0.0))
    outside = _index_at(result, (1.0, 0.0, 0.0))

    assert result.voxel_size == 0.5
    assert result.density.dtype == np.float32
    assert result.density[center] == 1.0
    assert result.density[surface] == 0.5
    assert result.density[outside] == 0.0
    assert result.diagnostics["sphere_radius"] == 0.5
    assert result.diagnostics["sphere_ramp_width_voxels"] == 1.0


def test_splat_is_union_max_not_overlap_sum_and_is_chunk_invariant():
    one = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    repeated = np.repeat(one, 9, axis=0)
    baseline = reconstruct_surface(
        one, 1.0, iterations=0, padding_voxels=1, particle_chunk_size=1)
    chunked = reconstruct_surface(
        repeated, 1.0, iterations=0, padding_voxels=1,
        particle_chunk_size=4)

    assert baseline.origin == chunked.origin
    np.testing.assert_array_equal(chunked.density, baseline.density)
    assert float(chunked.density.max()) == 1.0


def test_gaussian_blur_is_normalized_symmetric_and_uses_four_voxel_sigma():
    impulse = np.zeros((41, 41, 41), dtype=np.float32)
    impulse[20, 20, 20] = 1.0

    blurred = gaussian_blur(impulse)
    marginal = blurred.sum(axis=(1, 2), dtype=np.float64)
    coordinates = np.arange(-20, 21, dtype=np.float64)
    sigma = math.sqrt(float((marginal * coordinates**2).sum()))

    assert float(blurred.sum(dtype=np.float64)) == pytest.approx(1.0, abs=2e-6)
    np.testing.assert_allclose(blurred, blurred[::-1], atol=1e-8)
    np.testing.assert_allclose(blurred, blurred[:, ::-1], atol=1e-8)
    np.testing.assert_allclose(blurred, blurred[:, :, ::-1], atol=1e-8)
    assert sigma == pytest.approx(4.0, abs=0.003)


def test_default_reconstruction_is_deterministic_finite_and_paper_configured():
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.4, 0.1, -0.2], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    first = reconstruct_surface(positions, 1.0, max_voxels=100_000)
    second = reconstruct_surface(positions, 1.0, max_voxels=100_000)

    assert first.origin == second.origin
    assert first.diagnostics == second.diagnostics
    np.testing.assert_array_equal(first.density, second.density)
    assert np.all(np.isfinite(first.density))
    assert float(first.density.min()) >= 0.0
    assert float(first.density.max()) <= 1.0
    assert first.diagnostics["iterations_completed"] == 30
    assert first.diagnostics["gaussian_sigma_voxels"] == 4.0
    assert first.diagnostics["pseudo_time_step"] == pytest.approx(1.0 / 6.0)
    assert first.diagnostics["feature_threshold"] == 2.0
    assert first.diagnostics["feature_slope"] == 5.0


def test_feature_mask_preserves_thin_protrusion_more_than_unmasked_flow():
    density = np.zeros((25, 25, 25), dtype=np.float32)
    density[7:17, 7:17, 7:17] = 1.0
    density[17:22, 12, 12] = 1.0
    mask = feature_mask(density, gaussian_blur(density))

    protected = mean_curvature_flow(density, mask, iterations=5)
    unmasked = mean_curvature_flow(
        density, np.ones_like(density), iterations=5)
    fixed_feature = mask == 0.0

    assert np.any(fixed_feature)
    np.testing.assert_array_equal(protected[fixed_feature], density[fixed_feature])
    assert np.max(np.abs(unmasked[fixed_feature] - density[fixed_feature])) > 0.1
    assert protected[21, 12, 12] > unmasked[21, 12, 12]


def test_empty_input_returns_backend_field_without_allocating_a_crop():
    result = reconstruct_surface(
        np.empty((0, 3), dtype=np.float32),
        dx=0.25,
        grid_anchor=(1.0, 2.0, 3.0),
        max_voxels=1,
    )

    assert result.density.shape == (0, 0, 0)
    assert result.density.dtype == np.float32
    assert result.origin == (1.0, 2.0, 3.0)
    assert result.voxel_size == 0.125
    assert result.diagnostics["empty"] is True
    assert result.diagnostics["voxel_count"] == 0
    assert result.diagnostics["iterations_completed"] == 0


def test_zero_iterations_returns_the_unsmoothed_union_density():
    result = reconstruct_surface(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        2.0,
        iterations=0,
        padding_voxels=0,
        max_voxels=10_000,
    )

    assert result.diagnostics["iterations_completed"] == 0
    assert result.density[_index_at(result, (1.0, 0.0, 0.0))] == 0.5


@pytest.mark.parametrize(
    "positions, message",
    [
        (np.zeros((3,), dtype=np.float32), r"shape \(N, 3\)"),
        (np.zeros((2, 4), dtype=np.float32), r"shape \(N, 3\)"),
        (np.asarray([[0.0, np.nan, 0.0]]), "finite"),
        (np.asarray([[0.0j, 0.0j, 0.0j]]), "finite"),
        ("particles", r"shape \(N, 3\)"),
    ],
)
def test_position_validation_is_strict_and_atomic(positions, message):
    with pytest.raises(ValueError, match=message):
        reconstruct_surface(positions, 1.0)


@pytest.mark.parametrize("dx", [0.0, -1.0, np.nan, np.inf])
def test_dx_must_be_finite_and_positive(dx):
    with pytest.raises((TypeError, ValueError), match="dx"):
        reconstruct_surface(np.zeros((1, 3), dtype=np.float32), dx)


def test_options_and_helpers_reject_invalid_values():
    positions = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="iterations"):
        reconstruct_surface(positions, 1.0, iterations=True)
    with pytest.raises(ValueError, match="iterations"):
        reconstruct_surface(positions, 1.0, iterations=-1)
    with pytest.raises(ValueError, match="padding_voxels"):
        reconstruct_surface(positions, 1.0, padding_voxels=-1)
    with pytest.raises(ValueError, match="grid_anchor"):
        reconstruct_surface(positions, 1.0, grid_anchor=(0.0, np.nan, 0.0))
    with pytest.raises(ValueError, match="identical shapes"):
        feature_mask(np.zeros((2, 2, 2)), np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mean_curvature_flow(
            np.zeros((2, 2, 2)), np.full((2, 2, 2), 1.1), iterations=1)


def test_max_voxel_guard_runs_before_dense_allocation():
    positions = np.asarray(
        [[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]],
        dtype=np.float32,
    )

    with pytest.raises(MemoryError, match="exceeding max_voxels=1000"):
        reconstruct_surface(
            positions, 1.0, iterations=0, padding_voxels=0,
            max_voxels=1_000)


@pytest.mark.gpu
def test_gpu_reconstruction_matches_cpu_within_float32_tolerance():
    cupy = pytest.importorskip("cupy")
    from stflip.backend import cuda_diagnostics

    available, reason = cuda_diagnostics(force=True)
    assert available, reason
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.55, 0.1, -0.2], [-0.4, 0.25, 0.3]],
        dtype=np.float32,
    )
    cpu = reconstruct_surface(
        positions, 1.0, iterations=3, padding_voxels=3,
        particle_chunk_size=2)
    gpu = reconstruct_surface(
        cupy.asarray(positions), 1.0, iterations=3, padding_voxels=3,
        particle_chunk_size=2, array_module=cupy)

    assert gpu.origin == cpu.origin
    assert gpu.density.dtype == cupy.float32
    np.testing.assert_allclose(
        cupy.asnumpy(gpu.density), cpu.density, rtol=2e-5, atol=2e-6)
