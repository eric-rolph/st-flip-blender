import json
from pathlib import Path

import numpy as np
import pytest

from stflip import cache


def test_frame_and_metadata_round_trip(tmp_path):
    positions = np.arange(15, dtype=np.float64).reshape(5, 3)
    velocities = -positions
    meta = {"backend": "cuda", "frame_end_baked": 4}

    frame = cache.write_frame(str(tmp_path), 4, positions, velocities)
    cache.write_meta(str(tmp_path), meta)

    loaded_positions, loaded_velocities = cache.read_frame(str(tmp_path), 4)
    assert frame == cache.frame_path(str(tmp_path), 4)
    assert loaded_positions.dtype == np.float32
    np.testing.assert_array_equal(loaded_positions, positions.astype(np.float32))
    np.testing.assert_array_equal(loaded_velocities, velocities.astype(np.float32))
    assert cache.read_meta(str(tmp_path)) == meta
    assert cache.baked_frames(str(tmp_path)) == [4]
    assert not list(tmp_path.glob(".stflip-writing-*"))


def test_failed_atomic_replace_keeps_previous_frame(monkeypatch, tmp_path):
    original = np.ones((2, 3), dtype=np.float32)
    cache.write_frame(str(tmp_path), 1, original, original)

    def fail_replace(source, destination):
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(cache.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        cache.write_frame(str(tmp_path), 1, original * 9, original * 9)

    positions, velocities = cache.read_frame(str(tmp_path), 1)
    np.testing.assert_array_equal(positions, original)
    np.testing.assert_array_equal(velocities, original)
    assert not list(tmp_path.glob(".stflip-writing-*"))


def test_corrupt_cache_files_are_treated_as_missing(tmp_path):
    Path(cache.frame_path(str(tmp_path), 7)).write_bytes(b"not an npz")
    (tmp_path / cache.META_NAME).write_text("{not json", encoding="utf-8")

    assert cache.read_frame(str(tmp_path), 7) is None
    assert cache.read_meta(str(tmp_path)) is None


def test_wrong_numpy_container_and_array_schema_are_treated_as_missing(tmp_path):
    frame_path = Path(cache.frame_path(str(tmp_path), 7))
    with frame_path.open("wb") as stream:
        np.save(stream, np.ones((3, 3), dtype=np.float32))
    assert cache.read_frame(str(tmp_path), 7) is None

    np.savez(
        frame_path,
        positions=np.ones((3, 2), dtype=np.float32),
        velocities=np.ones((3, 2), dtype=np.float32),
    )
    assert cache.read_frame(str(tmp_path), 7) is None


@pytest.mark.parametrize("value", [[], None, "metadata"])
def test_non_mapping_metadata_is_treated_as_missing(tmp_path, value):
    (tmp_path / cache.META_NAME).write_text(
        json.dumps(value), encoding="utf-8")
    assert cache.read_meta(str(tmp_path)) is None


def test_clear_removes_stale_atomic_temporary_files(tmp_path):
    (tmp_path / ".stflip-writing-orphan.npz").write_bytes(b"partial")
    (tmp_path / cache.META_NAME).write_text(json.dumps({}), encoding="utf-8")

    assert cache.clear(str(tmp_path)) == 2
    assert list(tmp_path.iterdir()) == []
