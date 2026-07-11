import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stflip import cache
from stflip.metrics import FRAME_FIELD_ORDER, METRICS_SCHEMA, measure_frame


def _metric_record(frame, simulation_time_s=0.0):
    empty = np.empty((0, 3), dtype=np.float32)
    params = SimpleNamespace(
        dx=0.25,
        rho=1000.0,
        particles_per_cell=8,
        pcg_tol=1e-4,
        cfl_target=8.0,
    )
    return measure_frame(
        frame=frame,
        simulation_time_s=simulation_time_s,
        params=params,
        stats=None,
        positions_local=empty,
        velocities=empty,
    )


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


def test_metric_jsonl_round_trip_ignores_partial_lines_and_dedupes_frames(
        tmp_path):
    first = _metric_record(4, 0.1)
    replacement = _metric_record(4, 0.2)
    second = _metric_record(5, 0.3)
    path = cache.append_metric(str(tmp_path), first)
    cache.append_metric(str(tmp_path), second)
    cache.append_metric(str(tmp_path), replacement)
    with open(path, "ab") as stream:
        stream.write(b'{"schema_version":1,"frame":6')
    cache.append_metric(str(tmp_path), _metric_record(6, 0.4))

    records = cache.read_metrics(str(tmp_path))

    assert [record["frame"] for record in records] == [4, 5, 6]
    assert records[0]["simulation_time_s"] == 0.2
    assert records[2]["simulation_time_s"] == 0.4
    assert not cache.read_metrics(str(tmp_path / "missing"))


def test_append_metric_rejects_nonfinite_or_nonflat_records(tmp_path):
    record = _metric_record(1)
    record["compute_wall_s"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        cache.append_metric(str(tmp_path), record)

    record = _metric_record(1)
    record["extra"] = []
    with pytest.raises(ValueError, match="extra"):
        cache.append_metric(str(tmp_path), record)


def test_metric_json_and_csv_exports_are_atomic_and_self_contained(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "exports"
    meta = {
        "backend": "cuda",
        "frame_start": 2,
        "frame_end_baked": 2,
        "settings": {"target_cfl": 8.0, "seed": 3},
    }
    cache.write_meta(str(cache_dir), meta)
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(cache_dir), 2, empty, empty)
    cache.append_metric(str(cache_dir), _metric_record(2, 0.25))

    json_path = cache.export_metrics(
        str(cache_dir), str(output_dir / "metrics.json"), "JSON")
    csv_path = cache.export_metrics(
        str(cache_dir), str(output_dir / "metrics.csv"), "csv")

    with open(json_path, encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload["schema"] == METRICS_SCHEMA
    assert payload["version"] == 1
    assert payload["run"] == meta
    assert payload["frames"][0]["frame"] == 2

    with open(csv_path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == ["run_metadata_json", *FRAME_FIELD_ORDER]
    assert len(rows) == 1
    assert json.loads(rows[0]["run_metadata_json"]) == meta
    assert rows[0]["frame"] == "2"
    assert not list(output_dir.glob(".stflip-exporting-*"))


def test_failed_atomic_metric_export_keeps_previous_file(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(cache_dir), 1, empty, empty)
    cache.append_metric(str(cache_dir), _metric_record(1))
    cache.write_meta(
        str(cache_dir), {"frame_start": 1, "frame_end_baked": 1})
    destination = tmp_path / "metrics.json"
    destination.write_text("previous", encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("simulated interrupted export")

    monkeypatch.setattr(cache.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        cache.export_metrics(
            str(cache_dir), str(destination), "json")

    assert destination.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".stflip-exporting-*"))


def test_metric_export_rejects_records_without_baked_frames(tmp_path):
    cache.append_metric(str(tmp_path), _metric_record(9))

    with pytest.raises(ValueError, match="baked frames"):
        cache.export_metrics(
            str(tmp_path), str(tmp_path / "metrics.csv"), "csv")


def test_metric_export_uses_committed_readable_frame_intersection(tmp_path):
    empty = np.empty((0, 3), dtype=np.float32)
    for frame in (1, 2):
        cache.write_frame(str(tmp_path), frame, empty, empty)
        cache.append_metric(str(tmp_path), _metric_record(frame))
    cache.write_meta(
        str(tmp_path), {"frame_start": 1, "frame_end_baked": 1})

    destination = cache.export_metrics(
        str(tmp_path), str(tmp_path / "export.json"), "json")
    payload = json.loads(Path(destination).read_text(encoding="utf-8"))
    assert [record["frame"] for record in payload["frames"]] == [1]

    Path(cache.frame_path(str(tmp_path), 1)).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="baked frames"):
        cache.export_metrics(
            str(tmp_path), str(tmp_path / "corrupt.json"), "json")


def test_metric_export_override_does_not_replace_commit_authority(tmp_path):
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(tmp_path), 3, empty, empty)
    cache.append_metric(str(tmp_path), _metric_record(3))
    cache.write_meta(
        str(tmp_path), {"frame_start": 3, "frame_end_baked": 3})

    destination = cache.export_metrics(
        str(tmp_path),
        str(tmp_path / "override.json"),
        "json",
        run_meta={"label": "custom"},
    )

    payload = json.loads(Path(destination).read_text(encoding="utf-8"))
    assert payload["run"] == {"label": "custom"}
    assert [record["frame"] for record in payload["frames"]] == [3]


@pytest.mark.parametrize(
    "name", [cache.META_NAME, cache.METRICS_NAME, "stflip_000001.npz"])
def test_metric_export_refuses_to_overwrite_cache_control_files(tmp_path, name):
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(tmp_path), 1, empty, empty)
    cache.append_metric(str(tmp_path), _metric_record(1))
    cache.write_meta(
        str(tmp_path), {"frame_start": 1, "frame_end_baked": 1})

    with pytest.raises(ValueError, match="cache control"):
        cache.export_metrics(
            str(tmp_path), str(tmp_path / name), "json")


def test_clear_removes_stale_atomic_temporary_files(tmp_path):
    (tmp_path / ".stflip-writing-orphan.npz").write_bytes(b"partial")
    (tmp_path / cache.META_NAME).write_text(json.dumps({}), encoding="utf-8")
    cache.append_metric(str(tmp_path), _metric_record(1))

    assert cache.clear(str(tmp_path)) == 3
    assert list(tmp_path.iterdir()) == []
