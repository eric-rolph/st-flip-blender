"""Portable, playback-only exports for downstream visual enhancement tools.

The handoff deliberately contains rendered particle snapshots rather than raw
solver checkpoints.  Every packaged frame is reconstructed from validated
cache arrays, hashed, and described by a strict versioned manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import math
from numbers import Integral, Real
import os
import re
import tempfile
import zipfile

import numpy as np

from . import cache
from .metrics import FRAME_FIELD_ORDER, METRICS_SCHEMA, SCHEMA_VERSION


HANDOFF_SCHEMA = "stflip.downstream-playback-handoff"
HANDOFF_VERSION = 1
MANIFEST_NAME = "manifest.json"
METRICS_ARCHIVE_NAME = "metrics.json"
HASH_ALGORITHM = "sha256"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SETTING_KEYS = frozenset({
    "adaptive_gamma",
    "collect_enstrophy",
    "collect_metrics",
    "create_surface",
    "density",
    "dx",
    "eps_m",
    "eps_rho_relative",
    "flip_fraction",
    "fps",
    "gravity",
    "grid_dims",
    "jitter_strength",
    "local_advection_cfl",
    "particles_per_cell",
    "pcg_max_iterations",
    "pcg_tolerance",
    "resolution",
    "seed",
    "spatiotemporal_sampling",
    "surface_geometric_smoothing",
    "surface_method",
    "surface_particle_radius_dx",
    "paper_mcf_iterations",
    "paper_mesh_adaptivity",
    "paper_max_reconstruction_voxels",
    "surface_smoothing_factor",
    "surface_smoothing_iterations",
    "surface_voxel_size_dx",
    "target_cfl",
})

_COORDINATE_SPACE = {
    "positions": "BLENDER_WORLD",
    "velocities": "BLENDER_WORLD_AXES_PER_SECOND",
    "up_axis": "+Z",
    "handedness": "RIGHT_HANDED",
    "particle_order": "UNORDERED_PER_FRAME",
}

_LIMITATIONS = {
    "stable_particle_ids_included": False,
    "foam_or_spray_labels_included": False,
    "inferred_microdetail_included": False,
    "ai_model_included": False,
}

_TOP_LEVEL_KEYS = frozenset({
    "schema",
    "version",
    "package_type",
    "playback_only",
    "contains_solver_checkpoints",
    "hash_algorithm",
    "coordinate_space",
    "units",
    "timing",
    "settings",
    "frames",
    "metrics",
    "limitations",
})
_FRAME_KEYS = frozenset({
    "frame", "path", "particle_count", "positions_dtype",
    "velocities_dtype", "sha256",
})
_METRICS_KEYS = frozenset({
    "included", "path", "schema", "version", "record_count", "sha256",
})


class HandoffError(ValueError):
    """The cache cannot be represented by the portable handoff contract."""


def _json_value(value, name: str):
    """Return a JSON-native, finite copy without arbitrary object coercion."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise HandoffError(f"{name} must be finite")
        return result
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise HandoffError(f"{name} keys must be strings")
            result[key] = _json_value(value[key], f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise HandoffError(f"{name} is not JSON-safe")


def _sanitized_settings(metadata: Mapping) -> dict:
    source = metadata.get("settings", {})
    if not isinstance(source, Mapping):
        raise HandoffError("cache settings metadata must be a mapping")
    result = {
        key: _json_value(source[key], f"settings.{key}")
        for key in sorted(_SETTING_KEYS.intersection(source))
    }
    if "dx" in metadata:
        result["dx"] = _json_value(metadata["dx"], "dx")
        if isinstance(result["dx"], bool) \
                or not isinstance(result["dx"], (int, float)) \
                or result["dx"] <= 0.0:
            raise HandoffError("dx must be finite and positive")
    if "grid_dims" not in result and "dims" in metadata:
        result["grid_dims"] = _json_value(metadata["dims"], "dims")
    return dict(sorted(result.items()))


def _units(metadata: Mapping) -> dict:
    source = metadata.get("scene_units", {})
    if not isinstance(source, Mapping):
        raise HandoffError("scene_units metadata must be a mapping")
    length = source.get("length_unit", "blender_unit")
    system = source.get("system", "NONE")
    scale = source.get("scale_length")
    if not isinstance(length, str) or not length:
        raise HandoffError("scene_units.length_unit must be a non-empty string")
    if not isinstance(system, str):
        raise HandoffError("scene_units.system must be a string")
    if scale is not None:
        scale = _json_value(scale, "scene_units.scale_length")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) \
                or scale <= 0.0:
            raise HandoffError("scene_units.scale_length must be positive")
    return {
        "length_unit": length,
        "time_unit": "second",
        "velocity_unit": f"{length}_per_second",
        "scene_unit_system": system,
        "meters_per_blender_unit": scale,
    }


def _frame_arrays(value, frame: int) -> tuple[np.ndarray, np.ndarray]:
    if value is None:
        raise HandoffError(f"committed frame {frame} is unreadable")
    positions, velocities = value
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise HandoffError(f"frame {frame} positions must have shape (N, 3)")
    if velocities.shape != positions.shape:
        raise HandoffError(f"frame {frame} velocities must match positions")
    if not np.issubdtype(positions.dtype, np.number) \
            or not np.issubdtype(velocities.dtype, np.number):
        raise HandoffError(f"frame {frame} arrays must be numeric")
    if not bool(np.all(np.isfinite(positions))) \
            or not bool(np.all(np.isfinite(velocities))):
        raise HandoffError(f"frame {frame} arrays must contain finite values")
    # Cache frames are already float32/C-contiguous in normal operation. Avoid
    # an unconditional second raw-frame copy; malformed legacy numeric dtypes
    # are normalized only when necessary.
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    velocities = np.ascontiguousarray(velocities, dtype=np.float32)
    if not bool(np.all(np.isfinite(positions))) \
            or not bool(np.all(np.isfinite(velocities))):
        raise HandoffError(f"frame {frame} exceeds the float32 finite range")
    return positions, velocities


def _npz_payload(positions: np.ndarray, velocities: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        positions=positions,
        velocities=velocities,
    )
    return stream.getvalue()


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_destination(cache_dir: str, destination: str) -> None:
    cache_root = os.path.normcase(os.path.realpath(cache_dir))
    resolved = os.path.normcase(os.path.realpath(destination))
    if os.path.dirname(resolved) == cache_root:
        name = os.path.basename(resolved)
        if (
            name in {cache.META_NAME, cache.METRICS_NAME}
            or (name.startswith("stflip_") and name.endswith(".npz"))
            or name.startswith(".stflip-")
        ):
            raise HandoffError(
                f"refusing to overwrite cache control file {name!r}"
            )
    if os.path.splitext(destination)[1].lower() != ".zip":
        raise HandoffError("handoff destination must use the .zip extension")


def _timing(metadata: Mapping, frames: list[int], settings: Mapping) -> dict:
    fps = settings.get("fps")
    if fps is not None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) \
                or not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise HandoffError("settings.fps must be finite and positive")
        fps = float(fps)
    return {
        "frame_start": frames[0],
        "frame_end": frames[-1],
        "frame_count": len(frames),
        "frames_per_second": fps,
        "seconds_per_frame": None if fps is None else 1.0 / fps,
        "source_frame_end_requested": _json_value(
            metadata.get("frame_end"), "frame_end"
        ),
        "source_frame_end_committed": _json_value(
            metadata.get("frame_end_baked"), "frame_end_baked"
        ),
    }


def _metrics_payload(cache_dir: str, frames: list[int]):
    records = cache.read_metrics(cache_dir, frames)
    if not records:
        return None, {
            "included": False,
            "path": None,
            "schema": None,
            "version": None,
            "record_count": 0,
            "sha256": None,
        }
    sanitized = [
        {name: _json_value(record[name], f"metrics.{name}")
         for name in FRAME_FIELD_ORDER}
        for record in records
    ]
    payload = _json_bytes({
        "schema": METRICS_SCHEMA,
        "version": SCHEMA_VERSION,
        "records": sanitized,
    })
    return payload, {
        "included": True,
        "path": METRICS_ARCHIVE_NAME,
        "schema": METRICS_SCHEMA,
        "version": SCHEMA_VERSION,
        "record_count": len(sanitized),
        "sha256": _sha256(payload),
    }


def _committed_frame_range(metadata: Mapping) -> list[int]:
    """Return the contiguous frame range authorized by the commit marker."""
    start = metadata.get("frame_start")
    end = metadata.get("frame_end_baked")
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, Integral)
            or not isinstance(end, Integral) or int(end) < int(start)):
        raise HandoffError("cache has no valid committed frame range")
    return list(range(int(start), int(end) + 1))


def validate_manifest(manifest: Mapping) -> None:
    """Validate the exact version-1 portable manifest structure."""
    if not isinstance(manifest, Mapping) or set(manifest) != _TOP_LEVEL_KEYS:
        raise HandoffError("handoff manifest top-level fields do not match schema")
    if manifest["schema"] != HANDOFF_SCHEMA \
            or manifest["version"] != HANDOFF_VERSION:
        raise HandoffError("handoff manifest schema/version is unsupported")
    if manifest["package_type"] != "downstream_visual_enhancement_playback":
        raise HandoffError("handoff package_type is invalid")
    if manifest["playback_only"] is not True \
            or manifest["contains_solver_checkpoints"] is not False:
        raise HandoffError("handoff must be playback-only without checkpoints")
    if manifest["hash_algorithm"] != HASH_ALGORITHM:
        raise HandoffError("handoff hash algorithm is unsupported")
    if manifest["coordinate_space"] != _COORDINATE_SPACE:
        raise HandoffError("handoff coordinate-space declaration is invalid")
    if manifest["limitations"] != _LIMITATIONS:
        raise HandoffError("handoff limitations declaration is invalid")
    if not isinstance(manifest["units"], Mapping) \
            or set(manifest["units"]) != {
                "length_unit", "time_unit", "velocity_unit",
                "scene_unit_system", "meters_per_blender_unit",
            }:
        raise HandoffError("handoff units declaration is invalid")
    units = manifest["units"]
    if any(
        not isinstance(units[name], str) or not units[name]
        for name in (
            "length_unit", "time_unit", "velocity_unit", "scene_unit_system"
        )
    ):
        raise HandoffError("handoff unit names must be non-empty strings")
    scale = units["meters_per_blender_unit"]
    if scale is not None and (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or scale <= 0.0
    ):
        raise HandoffError("handoff unit scale must be finite and positive")
    if not isinstance(manifest["settings"], Mapping) \
            or not set(manifest["settings"]).issubset(_SETTING_KEYS):
        raise HandoffError("handoff settings are invalid")
    timing = manifest["timing"]
    if not isinstance(timing, Mapping) or set(timing) != {
        "frame_start", "frame_end", "frame_count", "frames_per_second",
        "seconds_per_frame", "source_frame_end_requested",
        "source_frame_end_committed",
    }:
        raise HandoffError("handoff timing declaration is invalid")
    for name in (
        "frame_start", "frame_end", "frame_count",
        "source_frame_end_requested", "source_frame_end_committed",
    ):
        value = timing[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise HandoffError(f"handoff timing {name} must be an integer or null")
    fps = timing["frames_per_second"]
    seconds = timing["seconds_per_frame"]
    if (fps is None) != (seconds is None):
        raise HandoffError("handoff frame-rate timing fields must agree")
    if fps is not None and (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or fps <= 0.0
        or not isinstance(seconds, (int, float))
        or not math.isclose(float(seconds), 1.0 / float(fps), rel_tol=1e-12)
    ):
        raise HandoffError("handoff frame-rate timing is invalid")
    frame_records = manifest["frames"]
    if not isinstance(frame_records, list) or not frame_records:
        raise HandoffError("handoff frames must be a non-empty list")
    numbers = []
    paths = set()
    for record in frame_records:
        if not isinstance(record, Mapping) or set(record) != _FRAME_KEYS:
            raise HandoffError("handoff frame fields do not match schema")
        frame = record["frame"]
        count = record["particle_count"]
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise HandoffError("handoff frame number must be an integer")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise HandoffError("handoff particle_count must be non-negative")
        if record["positions_dtype"] != "float32" \
                or record["velocities_dtype"] != "float32":
            raise HandoffError("handoff frame dtypes must be float32")
        path = record["path"]
        expected_path = f"frames/stflip_{frame:06d}.npz"
        if path != expected_path or path in paths:
            raise HandoffError("handoff frame path is invalid")
        if not isinstance(record["sha256"], str) \
                or _HASH_RE.fullmatch(record["sha256"]) is None:
            raise HandoffError("handoff frame hash is invalid")
        numbers.append(frame)
        paths.add(path)
    if numbers != sorted(set(numbers)):
        raise HandoffError("handoff frame numbers must be unique and sorted")
    if (
        timing["frame_start"] != numbers[0]
        or timing["frame_end"] != numbers[-1]
        or timing["frame_count"] != len(numbers)
    ):
        raise HandoffError("handoff timing does not match frame records")
    metrics = manifest["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != _METRICS_KEYS:
        raise HandoffError("handoff metrics declaration is invalid")
    if metrics["included"] is True:
        if metrics["path"] != METRICS_ARCHIVE_NAME \
                or metrics["schema"] != METRICS_SCHEMA \
                or metrics["version"] != SCHEMA_VERSION \
                or not isinstance(metrics["record_count"], int) \
                or metrics["record_count"] <= 0 \
                or metrics["record_count"] > len(frame_records) \
                or not isinstance(metrics["sha256"], str) \
                or _HASH_RE.fullmatch(metrics["sha256"]) is None:
            raise HandoffError("included handoff metrics declaration is invalid")
    elif metrics != {
        "included": False,
        "path": None,
        "schema": None,
        "version": None,
        "record_count": 0,
        "sha256": None,
    }:
        raise HandoffError("excluded handoff metrics declaration is invalid")


def export_handoff(
    cache_dir: str,
    path: str | os.PathLike,
    *,
    metadata: Mapping | None = None,
) -> str:
    """Atomically export committed playback frames as a portable ZIP.

    Commit authority always comes from the cache metadata on disk.  The
    optional ``metadata`` argument may override only the sanitized descriptive
    fields in the manifest; it cannot expand the committed frame range.
    """
    destination = os.path.abspath(os.fspath(path))
    _validate_destination(cache_dir, destination)
    committed_metadata = cache.read_meta(cache_dir)
    if not isinstance(committed_metadata, dict):
        raise HandoffError("cache has no valid committed metadata")
    frames = _committed_frame_range(committed_metadata)
    descriptive_metadata = committed_metadata if metadata is None else metadata
    if not isinstance(descriptive_metadata, Mapping):
        raise HandoffError("handoff metadata must be a mapping")
    settings = _sanitized_settings(descriptive_metadata)

    metrics_payload, metrics_record = _metrics_payload(cache_dir, frames)

    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=parent, prefix=".stflip-handoff-exporting-", suffix=".zip"
    )
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w") as archive:
                # Serialize and write one frame at a time. Production caches
                # can contain many gigabytes of playback data, so retaining
                # every reconstructed NPZ in memory would defeat the purpose
                # of a portable handoff. NPZ payloads are already compressed;
                # storing them avoids a redundant outer deflate pass.
                frame_records = []
                for frame in frames:
                    positions, velocities = _frame_arrays(
                        cache.read_frame(cache_dir, frame), frame
                    )
                    payload = _npz_payload(positions, velocities)
                    archive_path = f"frames/stflip_{frame:06d}.npz"
                    archive.writestr(
                        archive_path, payload,
                        compress_type=zipfile.ZIP_STORED,
                    )
                    frame_records.append({
                        "frame": int(frame),
                        "path": archive_path,
                        "particle_count": int(positions.shape[0]),
                        "positions_dtype": "float32",
                        "velocities_dtype": "float32",
                        "sha256": _sha256(payload),
                    })
                if metrics_payload is not None:
                    archive.writestr(
                        METRICS_ARCHIVE_NAME,
                        metrics_payload,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                manifest = {
                    "schema": HANDOFF_SCHEMA,
                    "version": HANDOFF_VERSION,
                    "package_type": (
                        "downstream_visual_enhancement_playback"),
                    "playback_only": True,
                    "contains_solver_checkpoints": False,
                    "hash_algorithm": HASH_ALGORITHM,
                    "coordinate_space": dict(_COORDINATE_SPACE),
                    "units": _units(descriptive_metadata),
                    "timing": _timing(
                        descriptive_metadata, frames, settings),
                    "settings": settings,
                    "frames": frame_records,
                    "metrics": metrics_record,
                    "limitations": dict(_LIMITATIONS),
                }
                validate_manifest(manifest)
                archive.writestr(
                    MANIFEST_NAME,
                    _json_bytes(manifest),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            stream.flush()
            os.fsync(stream.fileno())
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.testzip() is not None:
                raise HandoffError("handoff ZIP failed integrity validation")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination


__all__ = [
    "HANDOFF_SCHEMA",
    "HANDOFF_VERSION",
    "HASH_ALGORITHM",
    "HandoffError",
    "MANIFEST_NAME",
    "METRICS_ARCHIVE_NAME",
    "export_handoff",
    "validate_manifest",
]
