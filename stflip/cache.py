"""Disk cache for baked frames: one compressed .npz per frame + a meta file."""

from __future__ import annotations

import csv
import json
from numbers import Integral
import os
import tempfile

import numpy as np

from .metrics import (
    FRAME_FIELD_ORDER,
    METRICS_SCHEMA,
    SCHEMA_VERSION,
    validate_frame_record,
)

META_NAME = "stflip_meta.json"
METRICS_NAME = "stflip_metrics.jsonl"


def _atomic_path(cache_dir: str, suffix: str):
    """Yield a same-directory temporary path suitable for ``os.replace``."""
    os.makedirs(cache_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(
        dir=cache_dir, prefix=".stflip-writing-", suffix=suffix)
    return fd, path


def frame_path(cache_dir: str, frame: int) -> str:
    return os.path.join(cache_dir, f"stflip_{frame:06d}.npz")


def metrics_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, METRICS_NAME)


def write_frame(cache_dir: str, frame: int, positions: np.ndarray,
                velocities: np.ndarray) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = frame_path(cache_dir, frame)
    fd, temporary = _atomic_path(cache_dir, ".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                positions=positions.astype(np.float32),
                velocities=velocities.astype(np.float32),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def read_frame(cache_dir: str, frame: int):
    path = frame_path(cache_dir, frame)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as data:
            positions = data["positions"]
            velocities = data["velocities"]
            if (positions.ndim != 2 or positions.shape[1:] != (3,)
                    or velocities.shape != positions.shape
                    or not np.issubdtype(positions.dtype, np.number)
                    or not np.issubdtype(velocities.dtype, np.number)):
                return None
            return positions, velocities
    except (OSError, TypeError, ValueError, KeyError):
        return None


def write_meta(cache_dir: str, meta: dict) -> None:
    path = os.path.join(cache_dir, META_NAME)
    fd, temporary = _atomic_path(cache_dir, ".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(meta, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def read_meta(cache_dir: str) -> dict | None:
    path = os.path.join(cache_dir, META_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
            return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def committed_frames(cache_dir: str, metadata: dict | None = None) -> list[int]:
    """Return readable frame files inside the atomically committed range."""
    metadata = read_meta(cache_dir) if metadata is None else metadata
    if not isinstance(metadata, dict):
        return []
    lo = metadata.get("frame_start")
    hi = metadata.get("frame_end_baked")
    if (isinstance(lo, bool) or isinstance(hi, bool)
            or not isinstance(lo, Integral)
            or not isinstance(hi, Integral)
            or hi < lo):
        return []
    return [frame for frame in baked_frames(cache_dir)
            if lo <= frame <= hi and read_frame(cache_dir, frame) is not None]


def append_metric(cache_dir: str, record: dict) -> str:
    """Durably append one strict frame record to the canonical JSONL file."""
    validate_frame_record(record)
    line = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    path = metrics_path(cache_dir)
    encoded = (line + "\n").encode("utf-8")
    with open(path, "a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                # Separate a trailing interrupted record from the next valid
                # append so recovery can continue in a later Blender session.
                stream.write(b"\n")
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def read_metrics(cache_dir: str, valid_frames=None) -> list[dict]:
    """Read valid frame records, ignoring damage and keeping the last frame."""
    path = metrics_path(cache_dir)
    if not os.path.isfile(path):
        return []
    by_frame: dict[int, dict] = {}
    try:
        with open(path, "rb") as stream:
            for raw_line in stream:
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                    validate_frame_record(value)
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    # Appending is deliberately recoverable: an interrupted
                    # final write, or a damaged individual line, cannot hide
                    # earlier completed frame records.
                    continue
                by_frame[value["frame"]] = value
    except OSError:
        return []
    allowed = None if valid_frames is None else {int(v) for v in valid_frames}
    return [by_frame[frame] for frame in sorted(by_frame)
            if allowed is None or frame in allowed]


def _atomic_export_path(path: str):
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    suffix = os.path.splitext(destination)[1] or ".tmp"
    fd, temporary = tempfile.mkstemp(
        dir=parent, prefix=".stflip-exporting-", suffix=suffix)
    return destination, fd, temporary


def _validate_export_destination(cache_dir: str, path: str) -> None:
    cache_root = os.path.normcase(os.path.realpath(cache_dir))
    destination = os.path.normcase(os.path.realpath(path))
    if os.path.dirname(destination) != cache_root:
        return
    name = os.path.basename(destination)
    is_frame = name.startswith("stflip_") and name.endswith(".npz")
    if name in {META_NAME, METRICS_NAME} or is_frame:
        raise ValueError(f"refusing to overwrite cache control file {name!r}")


def export_metrics(
    cache_dir: str,
    path: str,
    fmt: str,
    *,
    run_meta: dict | None = None,
) -> str:
    """Atomically export canonical metrics to self-contained JSON or CSV."""
    output_format = fmt.lower()
    if output_format not in {"json", "csv"}:
        raise ValueError("metric export format must be 'json' or 'csv'")
    destination = os.path.abspath(path)
    _validate_export_destination(cache_dir, destination)
    committed_meta = read_meta(cache_dir)
    records = read_metrics(
        cache_dir, committed_frames(cache_dir, committed_meta))
    if not records:
        raise ValueError("no recorded metrics exist for baked frames")
    metadata = committed_meta if run_meta is None else run_meta
    metadata = metadata or {}
    # Validate metadata before creating the temporary export.  Current cache
    # metadata is JSON-native; rejecting NaN/non-serializable values keeps both
    # output formats deterministic and self-contained.
    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    destination, fd, temporary = _atomic_export_path(destination)
    try:
        if output_format == "json":
            payload = {
                "schema": METRICS_SCHEMA,
                "version": SCHEMA_VERSION,
                "run": metadata,
                "frames": records,
            }
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        else:
            fieldnames = ("run_metadata_json", *FRAME_FIELD_ORDER)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow({"run_metadata_json": metadata_json, **record})
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination


def baked_frames(cache_dir: str) -> list[int]:
    if not os.path.isdir(cache_dir):
        return []
    out = []
    for name in os.listdir(cache_dir):
        if name.startswith("stflip_") and name.endswith(".npz"):
            try:
                out.append(int(name[7:13]))
            except ValueError:
                pass
    return sorted(out)


def clear(cache_dir: str) -> int:
    n = 0
    if not os.path.isdir(cache_dir):
        return n
    for name in os.listdir(cache_dir):
        if ((name.startswith("stflip_") and name.endswith(".npz"))
                or name == META_NAME
                or name == METRICS_NAME
                or name.startswith(".stflip-writing-")
                or name.startswith(".stflip-exporting-")):
            os.remove(os.path.join(cache_dir, name))
            n += 1
    return n
