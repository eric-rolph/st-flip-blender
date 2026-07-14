"""SAMP-M2: hash-based Owen-scrambled temporal sampling library."""

import numpy as np
import pytest

from stflip import sampling


def _xi_series(particle_id, indices, seed=0):
    ids = np.full(len(indices), particle_id, dtype=np.int64)
    return sampling.temporal_xi(
        np, ids, np.asarray(indices, dtype=np.int64), seed)


class TestPrimitives:
    def test_reverse_bits32_involution_and_known_values(self):
        v = np.array([0, 1, 0x80000000, 0xDEADBEEF], dtype=np.uint32)
        rev = sampling.reverse_bits32(np, v)
        assert rev[0] == 0
        assert rev[1] == 0x80000000
        assert rev[2] == 1
        assert np.array_equal(sampling.reverse_bits32(np, rev), v)

    def test_laine_karras_low_bits_permute_within_blocks(self):
        # Owen dependency structure: output bit b depends only on input
        # bits <= b, so the low k output bits are a bijection of the low k
        # input bits.
        k = 12
        v = np.arange(2 ** k, dtype=np.uint32)
        key = np.full(v.shape, 0xB5297A4D, dtype=np.uint32)
        out = sampling.laine_karras_permutation(np, v, key)
        low = out & np.uint32(2 ** k - 1)
        assert len(np.unique(low)) == 2 ** k

    def test_nested_uniform_scramble_maps_aligned_blocks_aligned(self):
        # The composed rev/LK/rev scramble must send an aligned 2^k index
        # block bijectively onto another aligned 2^k block -- this is what
        # preserves dyadic stratification under index shuffling.
        k = 10
        base = 5 * 2 ** k
        v = np.arange(base, base + 2 ** k, dtype=np.uint32)
        key = np.full(v.shape, 0xB5297A4D, dtype=np.uint32)
        out = sampling.reverse_bits32(
            np, sampling.laine_karras_permutation(
                np, sampling.reverse_bits32(np, v), key))
        assert len(np.unique(out)) == 2 ** k
        assert len(np.unique(out >> np.uint32(k))) == 1

    def test_particle_keys_distinct_for_adjacent_ids(self):
        ids = np.arange(10_000, dtype=np.int64)
        keys = sampling.particle_keys(np, ids, 0, sampling._POINT_SALT)
        assert len(np.unique(keys)) == len(ids)


class TestSequenceProperties:
    @pytest.mark.parametrize("k", [2, 4, 6])
    @pytest.mark.parametrize("block", [0, 3, 17])
    def test_aligned_windows_stratify_exactly(self, k, block):
        # Over any 2^k-aligned window of consecutive substeps, one deviate
        # falls in each of the 2^k equal slab strata (the point of using a
        # (0, 2)-sequence; index shuffling weakens this to ALIGNED windows
        # only, which is why the window starts at block * 2^k).
        for particle in (1, 999, 123_456_789):
            start = block * 2 ** k
            xi = _xi_series(particle, range(start, start + 2 ** k))
            strata = np.floor(
                (xi.astype(np.float64) + 0.5) * 2 ** k).astype(np.int64)
            strata = np.clip(strata, 0, 2 ** k - 1)
            assert len(np.unique(strata)) == 2 ** k

    def test_marginal_uniformity_across_particles(self):
        # At one fixed substep the deviate across particles must be
        # uniform on [-1/2, 1/2): Kolmogorov-Smirnov style bound.
        n = 100_000
        ids = np.arange(n, dtype=np.int64)
        xi = sampling.temporal_xi(np, ids, 7, 3)
        sorted_xi = np.sort(xi.astype(np.float64)) + 0.5
        ecdf = (np.arange(1, n + 1)) / n
        d = float(np.abs(sorted_xi - ecdf).max())
        assert d < 5.0 / np.sqrt(n)

    def test_cross_particle_sequences_decorrelated(self):
        steps = np.arange(4096, dtype=np.int64)
        a = _xi_series(42, steps).astype(np.float64)
        for other in (43, 44, 10_042):
            b = _xi_series(other, steps).astype(np.float64)
            r = float(np.corrcoef(a, b)[0, 1])
            assert abs(r) < 4.0 / np.sqrt(len(steps))

    def test_range_and_dtype(self):
        xi = _xi_series(5, range(1024))
        assert xi.dtype == np.float32
        assert float(xi.min()) >= -0.5
        assert float(xi.max()) <= 0.5

    def test_deterministic_and_seed_sensitive(self):
        ids = np.arange(64, dtype=np.int64)
        a = sampling.temporal_xi(np, ids, 11, 0)
        b = sampling.temporal_xi(np, ids, 11, 0)
        c = sampling.temporal_xi(np, ids, 11, 1)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_scalar_index_matches_array_index(self):
        ids = np.arange(64, dtype=np.int64)
        scalar = sampling.temporal_xi(np, ids, 9, 0)
        array = sampling.temporal_xi(
            np, ids, np.full(64, 9, dtype=np.int64), 0)
        assert np.array_equal(scalar, array)

    def test_no_python_scalar_overflow_warnings(self):
        ids = np.array([2 ** 62, 2 ** 63 - 1], dtype=np.int64)
        with np.errstate(all="raise"):
            xi = sampling.temporal_xi(np, ids, 2 ** 31, 0)
        assert np.all(np.isfinite(xi))


@pytest.mark.gpu
class TestGpuParity:
    def test_bitwise_identical_to_cpu(self):
        cupy = pytest.importorskip("cupy")
        ids = np.arange(100_000, dtype=np.int64)
        cpu = sampling.temporal_xi(np, ids, 1234, 5)
        gpu = sampling.temporal_xi(cupy, cupy.asarray(ids), 1234, 5)
        assert np.array_equal(cpu, cupy.asnumpy(gpu))
