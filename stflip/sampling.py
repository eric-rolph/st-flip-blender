"""Stateless low-discrepancy temporal sampling (roadmap SAMP-M2).

Owen-scrambled, index-shuffled Sobol dimension-0 deviates for the per-particle
temporal jitter, following Burley 2020 ("Practical Hash-based Owen
Scrambling").  ``(particle_id, substep_index, seed)`` fully determine every
deviate: there is no engine state, so checkpoint/resume needs nothing beyond
the ids and substep counter that SAMP-M1 already persists, and skipped draws
(``jitter_strength == 0`` paths) cannot desynchronize anything.

Sobol dimension 0 is the bit-reversed index (van der Corput base 2), so both
the index shuffle and the point scramble reduce to a Laine-Karras hash applied
in the reversed-bit domain.  Each ``v ^= v * c`` step only propagates bits
upward, so in that domain every output bit depends only on equal-or-lower
input bits -- a valid nested-uniform (Owen-style) scramble that preserves the
(0, 2)-sequence stratification per key.  Index shuffling under an independent
per-particle key decorrelates the per-key scrambles (different-key scrambles
of the same base point are only approximately independent); its cost is that
exact dyadic stratification holds for index windows ALIGNED at multiples of
``2**k`` rather than for arbitrary windows.

Everything is unsigned 32-bit ARRAY arithmetic (wraparound is well-defined
and warning-free for NumPy arrays, and bit-exact between NumPy and CuPy), so
CPU and GPU produce identical deviates.
"""

from __future__ import annotations

import numbers


_POINT_SALT = 0x9E3779B97F4A7C15
_SHUFFLE_SALT = 0xD6E8FEB86659FD93


def particle_keys(xp, particle_id, seed, salt):
    """Per-particle uint32 hash keys from 64-bit ids, a seed, and a salt.

    ``particle_id`` is the SAMP-M1 int64 array (device-side under CuPy).
    The mix runs in uint64 and keeps the high 32 bits, so ids that differ
    only in low bits still receive well-separated keys.
    """

    ids = xp.asarray(particle_id).astype(xp.uint64)
    salted = xp.uint64((int(seed) ^ int(salt)) & 0xFFFFFFFFFFFFFFFF)
    mixed = _splitmix64_xp(xp, ids ^ salted)
    return (mixed >> xp.uint64(32)).astype(xp.uint32)


def _splitmix64_xp(xp, x):
    mask = xp.uint64(0xFFFFFFFFFFFFFFFF)
    x = (x + xp.uint64(0x9E3779B97F4A7C15)) & mask
    x = ((x ^ (x >> xp.uint64(30))) * xp.uint64(0xBF58476D1CE4E5B9)) & mask
    x = ((x ^ (x >> xp.uint64(27))) * xp.uint64(0x94D049BB133111EB)) & mask
    return x ^ (x >> xp.uint64(31))


def reverse_bits32(xp, v):
    """Bit-reverse each element of a uint32 array."""

    v = v.astype(xp.uint32)
    v = ((v >> xp.uint32(1)) & xp.uint32(0x55555555)) | (
        (v & xp.uint32(0x55555555)) << xp.uint32(1))
    v = ((v >> xp.uint32(2)) & xp.uint32(0x33333333)) | (
        (v & xp.uint32(0x33333333)) << xp.uint32(2))
    v = ((v >> xp.uint32(4)) & xp.uint32(0x0F0F0F0F)) | (
        (v & xp.uint32(0x0F0F0F0F)) << xp.uint32(4))
    v = ((v >> xp.uint32(8)) & xp.uint32(0x00FF00FF)) | (
        (v & xp.uint32(0x00FF00FF)) << xp.uint32(8))
    return (v >> xp.uint32(16)) | (v << xp.uint32(16))


def laine_karras_permutation(xp, v, key):
    """Laine-Karras-style hash: a nested-uniform permutation in-place.

    Operates in the reversed-bit domain.  Every step only propagates bits
    upward (``v ^= v * c`` with even ``c``), so each output bit depends only
    on equal-or-lower input bits, which is exactly the Owen-scramble
    dependency structure.
    """

    v = v.astype(xp.uint32)
    key = key.astype(xp.uint32)
    v = v ^ (v * xp.uint32(0x3D20ADEA))
    v = v + key
    v = v * ((key >> xp.uint32(16)) | xp.uint32(1))
    v = v ^ (v * xp.uint32(0x05526C56))
    v = v ^ (v * xp.uint32(0x53A22864))
    return v


def _as_index_array(xp, substep_index, shape):
    if isinstance(substep_index, numbers.Integral):
        return xp.full(shape, int(substep_index) & 0xFFFFFFFF,
                       dtype=xp.uint32)
    index = xp.asarray(substep_index)
    return (index.astype(xp.uint64) & xp.uint64(0xFFFFFFFF)).astype(
        xp.uint32)


def temporal_xi(xp, particle_id, substep_index, seed):
    """Owen-scrambled, index-shuffled Sobol dim-0 deviate in [-1/2, 1/2).

    ``particle_id``: int64 array of stable ids (SAMP-M1).
    ``substep_index``: scalar or array; the global substep counter.
    ``seed``: the simulation seed.

    Replaces ``rng.random(n) - 0.5`` in the temporal jitter: over any
    2^k-aligned window of consecutive substeps, each particle's deviates
    exactly stratify the slab, so the W_T-weighted temporal quadrature error
    along a trajectory decays as O(log K / K) instead of O(K**-0.5) while
    per-step cross-particle scatter is unchanged (independent per-particle
    keys).
    """

    ids = xp.asarray(particle_id)
    key_point = particle_keys(xp, ids, seed, _POINT_SALT)
    key_shuffle = particle_keys(xp, ids, seed, _SHUFFLE_SALT)
    index = _as_index_array(xp, substep_index, ids.shape)

    # Burley 2020: shuffle the index with one nested-uniform scramble, then
    # scramble the Sobol point with another.  For dimension 0 the point bits
    # are the bit-reversed index, so both stages live in the reversed domain.
    shuffled = reverse_bits32(
        xp, laine_karras_permutation(
            xp, reverse_bits32(xp, index), key_shuffle))
    scrambled = laine_karras_permutation(xp, shuffled, key_point)
    bits = reverse_bits32(xp, scrambled)
    return (bits.astype(xp.float64) * (2.0 ** -32) - 0.5).astype(xp.float32)


def temporal_xi_cp_rot(xp, particle_id, substep_index, seed):
    """Cranley-Patterson rotated van der Corput deviate in [-1/2, 1/2).

    Experimental A/B comparison arm ONLY (roadmap SAMP): each particle's
    offset is frozen for all steps, so pairwise phase relationships persist
    across the whole bake and can surface as structured artifacts --
    exactly why the Owen-scrambled sampler is the primary choice.  The
    uint32 addition wraps, which IS the mod-1 rotation.
    """

    ids = xp.asarray(particle_id)
    key = particle_keys(xp, ids, seed, _POINT_SALT)
    index = _as_index_array(xp, substep_index, ids.shape)
    rotated = reverse_bits32(xp, index) + key
    return (rotated.astype(xp.float64) * (2.0 ** -32) - 0.5).astype(
        xp.float32)
