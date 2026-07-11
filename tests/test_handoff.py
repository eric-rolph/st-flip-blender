from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import zipfile

import numpy as np
import pytest

from stflip import cache, handoff
from stflip.metrics import FRAME_FIELD_ORDER, measure_frame


ROOT = Path(__file__).parents[1]


def _metric(frame: int) -> dict:
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
        simulation_time_s=(frame - 1) / 24.0,
        params=params,
        stats=None,
        positions_local=empty,
        velocities=empty,
    )


def _metadata(frame_end_baked: int = 2) -> dict:
    return {
        "frame_start": 1,
        "frame_end": 3,
        "frame_end_baked": frame_end_baked,
        "dx": 0.125,
        "cache_owner_id": "must-not-leak",
        "checkpoint": {"fingerprint": "must-not-leak"},
        "scene_units": {
            "length_unit": "blender_unit",
            "system": "METRIC",
            "scale_length": 0.01,
        },
        "settings": {
            "resolution": 64,
            "grid_dims": [64, 32, 16],
            "target_cfl": 8.0,
            "particles_per_cell": 8,
            "seed": 7,
            "fps": 24.0,
            "gravity": [0.0, 0.0, -9.81],
            "api_key": "must-not-leak",
            "cache_path": "must-not-leak",
        },
    }


def _write_cache(path: Path) -> None:
    cache.write_meta(str(path), _metadata())
    cache.write_frame(
        str(path),
        1,
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
    )
    cache.write_frame(
        str(path),
        2,
        np.asarray(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64
        ),
        np.asarray(
            [[0.5, 0.25, 0.125], [1.0, 2.0, 3.0]], dtype=np.float64
        ),
    )
    # This readable output and metric are beyond the committed range.
    cache.write_frame(
        str(path), 3,
        np.zeros((3, 3), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
    )
    cache.append_metric(str(path), _metric(1))
    cache.append_metric(str(path), _metric(3))
    (path / "stflip_checkpoint_000001.npz").write_bytes(
        b"raw restart state must not be exported"
    )


def test_handoff_contains_only_committed_playback_and_sanitized_metrics(
    tmp_path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_cache(cache_dir)

    destination = handoff.export_handoff(
        str(cache_dir), tmp_path / "handoff.zip"
    )

    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            handoff.MANIFEST_NAME,
            handoff.METRICS_ARCHIVE_NAME,
            "frames/stflip_000001.npz",
            "frames/stflip_000002.npz",
        }
        assert not any("checkpoint" in name for name in archive.namelist())
        for info in archive.infolist():
            expected = (
                zipfile.ZIP_STORED
                if info.filename.startswith("frames/")
                else zipfile.ZIP_DEFLATED
            )
            assert info.compress_type == expected
        manifest = json.loads(archive.read(handoff.MANIFEST_NAME))
        handoff.validate_manifest(manifest)
        assert manifest["schema"] == handoff.HANDOFF_SCHEMA
        assert manifest["version"] == handoff.HANDOFF_VERSION
        assert manifest["playback_only"] is True
        assert manifest["contains_solver_checkpoints"] is False
        assert manifest["coordinate_space"]["positions"] == "BLENDER_WORLD"
        assert manifest["coordinate_space"]["particle_order"] \
            == "UNORDERED_PER_FRAME"
        assert manifest["units"] == {
            "length_unit": "blender_unit",
            "time_unit": "second",
            "velocity_unit": "blender_unit_per_second",
            "scene_unit_system": "METRIC",
            "meters_per_blender_unit": 0.01,
        }
        assert manifest["timing"]["frame_count"] == 2
        assert manifest["timing"]["seconds_per_frame"] \
            == pytest.approx(1.0 / 24.0)
        assert "api_key" not in manifest["settings"]
        assert "cache_path" not in manifest["settings"]
        assert "cache_owner_id" not in manifest
        assert manifest["settings"]["dx"] == pytest.approx(0.125)
        assert set(manifest["limitations"].values()) == {False}

        for record in manifest["frames"]:
            payload = archive.read(record["path"])
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                assert set(arrays.files) == {"positions", "velocities"}
                assert arrays["positions"].dtype == np.float32
                assert arrays["velocities"].dtype == np.float32
                assert arrays["positions"].shape == (
                    record["particle_count"], 3
                )

        metrics_payload = archive.read(handoff.METRICS_ARCHIVE_NAME)
        assert hashlib.sha256(metrics_payload).hexdigest() \
            == manifest["metrics"]["sha256"]
        metrics = json.loads(metrics_payload)
        assert [record["frame"] for record in metrics["records"]] == [1]
        assert set(metrics["records"][0]) == set(FRAME_FIELD_ORDER)


def test_handoff_without_metrics_omits_metrics_member(tmp_path):
    cache_dir = tmp_path / "cache"
    cache.write_meta(str(cache_dir), _metadata(frame_end_baked=1))
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(cache_dir), 1, empty, empty)

    destination = handoff.export_handoff(
        str(cache_dir), tmp_path / "handoff.zip"
    )

    with zipfile.ZipFile(destination) as archive:
        assert handoff.METRICS_ARCHIVE_NAME not in archive.namelist()
        manifest = json.loads(archive.read(handoff.MANIFEST_NAME))
    assert manifest["metrics"] == {
        "included": False,
        "path": None,
        "schema": None,
        "version": None,
        "record_count": 0,
        "sha256": None,
    }


def test_handoff_rejects_nonfinite_committed_frame(tmp_path):
    cache.write_meta(str(tmp_path), _metadata(frame_end_baked=1))
    cache.write_frame(
        str(tmp_path), 1,
        np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )

    with pytest.raises(handoff.HandoffError, match="finite"):
        handoff.export_handoff(str(tmp_path), tmp_path / "handoff.zip")


def test_handoff_requires_committed_readable_frames(tmp_path):
    cache.write_meta(str(tmp_path), _metadata(frame_end_baked=1))
    Path(cache.frame_path(str(tmp_path), 1)).write_bytes(b"corrupt")

    with pytest.raises(handoff.HandoffError, match="committed frame 1 is unreadable"):
        handoff.export_handoff(str(tmp_path), tmp_path / "handoff.zip")


def test_failed_atomic_handoff_keeps_previous_destination(
    monkeypatch, tmp_path,
):
    cache_dir = tmp_path / "cache"
    cache.write_meta(str(cache_dir), _metadata(frame_end_baked=1))
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(cache_dir), 1, empty, empty)
    destination = tmp_path / "handoff.zip"
    destination.write_bytes(b"previous")

    def fail_replace(source, target):
        raise OSError("simulated interrupted handoff")

    monkeypatch.setattr(handoff.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        handoff.export_handoff(str(cache_dir), destination)

    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".stflip-handoff-exporting-*"))


@pytest.mark.parametrize(
    "name",
    [
        cache.META_NAME,
        cache.METRICS_NAME,
        "stflip_000001.npz",
        "stflip_checkpoint_000001.npz",
    ],
)
def test_handoff_refuses_cache_control_overwrite(tmp_path, name):
    with pytest.raises(handoff.HandoffError, match="cache control"):
        handoff.export_handoff(str(tmp_path), tmp_path / name)


def test_manifest_validator_rejects_extra_or_misleading_capabilities(
    tmp_path,
):
    cache.write_meta(str(tmp_path), _metadata(frame_end_baked=1))
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(tmp_path), 1, empty, empty)
    destination = handoff.export_handoff(
        str(tmp_path), tmp_path / "handoff.zip"
    )
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read(handoff.MANIFEST_NAME))

    extra = {**manifest, "checkpoint": {}}
    with pytest.raises(handoff.HandoffError, match="top-level"):
        handoff.validate_manifest(extra)

    misleading = deepcopy(manifest)
    misleading["limitations"]["stable_particle_ids_included"] = True
    with pytest.raises(handoff.HandoffError, match="limitations"):
        handoff.validate_manifest(misleading)


def test_blender_handoff_operator_registers_and_exports(monkeypatch, tmp_path):
    root = "_stflip_handoff_operator_test"
    for name in (root, f"{root}.addon", f"{root}.stflip"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, f"{root}.stflip.cache", cache)
    monkeypatch.setitem(sys.modules, f"{root}.stflip.handoff", handoff)
    experiments = types.ModuleType(f"{root}.stflip.experiments")
    experiments.get_profile = lambda name: None
    monkeypatch.setitem(sys.modules, experiments.__name__, experiments)

    cache_dir = tmp_path / "cache"
    cache.write_meta(str(cache_dir), _metadata(frame_end_baked=1))
    empty = np.empty((0, 3), dtype=np.float32)
    cache.write_frame(str(cache_dir), 1, empty, empty)
    handlers = types.ModuleType(f"{root}.addon.handlers")
    handlers.resolve_cache_dir = lambda scene: str(cache_dir)
    handlers.scene_cache_ownership = (
        lambda scene, metadata: cache.OWNERSHIP_OWNED)
    monkeypatch.setitem(sys.modules, handlers.__name__, handlers)
    operators_module = types.ModuleType(f"{root}.addon.operators")
    operators_module._BAKE = {}
    monkeypatch.setitem(sys.modules, operators_module.__name__, operators_module)

    reports = []
    registered = []

    class Operator:
        def report(self, level, message):
            reports.append((level, message))

    class ExportHelper:
        pass

    bpy = types.ModuleType("bpy")
    bpy.types = SimpleNamespace(Operator=Operator)
    bpy.app = SimpleNamespace(background=False)
    bpy.path = SimpleNamespace(abspath=lambda value: value)
    bpy.utils = SimpleNamespace(
        register_class=lambda cls: registered.append(cls),
        unregister_class=lambda cls: registered.remove(cls),
    )
    props = types.ModuleType("bpy.props")
    props.EnumProperty = lambda **kwargs: None
    props.StringProperty = lambda **kwargs: None
    bpy.props = props
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    bpy_extras = types.ModuleType("bpy_extras")
    io_utils = types.ModuleType("bpy_extras.io_utils")
    io_utils.ExportHelper = ExportHelper
    bpy_extras.io_utils = io_utils
    monkeypatch.setitem(sys.modules, "bpy_extras", bpy_extras)
    monkeypatch.setitem(sys.modules, "bpy_extras.io_utils", io_utils)

    module_name = f"{root}.addon.experiment"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "addon" / "experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.STFLIP_OT_export_handoff in module.CLASSES
    module.register()
    assert registered == list(module.CLASSES)

    operator = module.STFLIP_OT_export_handoff()
    operator.filepath = str(tmp_path / "portable")
    context = SimpleNamespace(scene=object())
    assert module.STFLIP_OT_export_handoff.poll(context) is True
    result = operator.execute(context)

    assert result == {"FINISHED"}
    assert (tmp_path / "portable.zip").is_file()
    assert reports[-1][0] == {"INFO"}

    cache.write_meta(str(cache_dir), _metadata(frame_end_baked=2))
    assert module.STFLIP_OT_export_handoff.poll(context) is False
    ready, reason = module._handoff_cache_ready(context.scene)
    assert ready is False
    assert "missing" in reason
    cache.write_meta(str(cache_dir), _metadata(frame_end_baked=1))

    handlers.scene_cache_ownership = (
        lambda scene, metadata: cache.OWNERSHIP_FOREIGN)
    assert module.STFLIP_OT_export_handoff.poll(context) is False
    operator.filepath = str(tmp_path / "refused")
    assert operator.execute(context) == {"CANCELLED"}
    assert not (tmp_path / "refused.zip").exists()
    assert "ownership is foreign" in reports[-1][1]

    module.unregister()
    assert registered == []
