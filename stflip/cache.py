"""Disk cache for baked frames: one compressed .npz per frame + a meta file."""

from __future__ import annotations

import csv
import hashlib
import json
from numbers import Integral
import os
import re
import tempfile
import zipfile

import numpy as np

from .metrics import (
    FRAME_FIELD_ORDER,
    METRICS_SCHEMA,
    SCHEMA_VERSION,
    validate_frame_record,
)

META_NAME = "stflip_meta.json"
METRICS_NAME = "stflip_metrics.jsonl"
CHECKPOINT_SCHEMA = "stflip-solver-checkpoint"
CHECKPOINT_VERSION = 3
SURFACE_SCHEMA = "stflip-paper-surface"
SURFACE_VERSION = 2
SURFACE_LEGACY_VERSION = 1
SURFACE_CONFIG_SCHEMA = "stflip-paper-surface-config"
SURFACE_CONFIG_VERSION = 2

_FRAME_RE = re.compile(r"^stflip_(-?\d+)\.npz$")
_CHECKPOINT_RE = re.compile(r"^stflip_checkpoint_(-?\d+)\.npz$")
_SURFACE_RE = re.compile(
    r"^stflip_surface_([0-9a-f]{64})_(-?\d+)\.npz$")
# Version 1 stored only base trajectory state. Version 2 additionally persists
# the two-phase tag, APIC affine matrix, and shading attributes as separate
# archive members, so a resumed two-phase/APIC bake continues without a visible
# discontinuity. Version-1 archives are still readable: their missing members
# restore to the historical defaults (liquid phase, empty affine, zero shading).
_CHECKPOINT_KEYS_V1 = frozenset({
    "schema",
    "version",
    "frame",
    "fingerprint",
    "pos",
    "vel",
    "dt_resid",
    "time",
    "dt_prev",
    "rng_state_json",
    "outflow_removed_total",
    "volume_outflow_removed_total",
    "pressure_outflow_removed_total",
})
# Extra per-particle members introduced by version 2. ``affine_c`` is the APIC
# 3x3 matrix per particle; it is stored with shape (0, 3, 3) when APIC is off.
_CHECKPOINT_EXTRA_KEYS = frozenset({
    "phase",
    "affine_c",
    "age",
    "source_id",
})
_CHECKPOINT_KEYS_V2 = _CHECKPOINT_KEYS_V1 | _CHECKPOINT_EXTRA_KEYS
# Version 3 (roadmap SAMP-M1) adds stable per-particle ids, the id
# allocation counter, and the global substep counter, so id-keyed sampling
# schemes survive resume.  Version-2 archives are still readable: their
# missing members restore with synthesized ids 0..n-1 and substep 0.
_CHECKPOINT_V3_EXTRA_KEYS = frozenset({
    "particle_id",
    "next_particle_id",
    "substep_index",
})
_CHECKPOINT_KEYS_V3 = _CHECKPOINT_KEYS_V2 | _CHECKPOINT_V3_EXTRA_KEYS
_CHECKPOINT_KEYS_BY_VERSION = {
    1: _CHECKPOINT_KEYS_V1,
    2: _CHECKPOINT_KEYS_V2,
    3: _CHECKPOINT_KEYS_V3,
}
# Reserved, mode-gated OPTIONAL archive members per version (roadmap
# Decision 3: the checkpoint schema is bumped exactly once, so members that
# later milestones may need are reserved here).  ``gamma_prev`` is written
# only when a bake combines CALM-M3's surface gamma mode with NORM's exact
# temporal normalization; it is absent otherwise.
_CHECKPOINT_OPTIONAL_KEYS_BY_VERSION = {
    1: frozenset(),
    2: frozenset(),
    3: frozenset({"gamma_prev"}),
}
# Optional keys in the in-memory checkpoint-state mapping (npz ``affine_c`` maps
# to state key ``C`` to match the solver's attribute name).
_CHECKPOINT_OPTIONAL_STATE_KEYS = frozenset({
    "phase", "C", "age", "source_id",
    "particle_id", "next_particle_id", "substep_index", "gamma_prev",
})
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_SURFACE_KEYS = frozenset({
    "schema",
    "version",
    "frame",
    "fingerprint",
    "source_positions_sha256",
    "mesh_sha256",
    "vertices",
    "triangles",
    "quads",
})

OWNER_KEY = "cache_owner_id"
OWNERSHIP_OWNED = "owned"
OWNERSHIP_LEGACY = "legacy"
OWNERSHIP_FOREIGN = "foreign"
OWNERSHIP_INVALID = "invalid"
OWNERSHIP_MISSING = "missing"


def ownership_status(metadata: dict | None, expected_owner_id: str) -> str:
    """Classify cache metadata relative to one scene's persistent owner ID.

    ``owned`` means a non-empty owner is an exact match. ``legacy`` means the
    metadata predates ownership IDs and is safe to read for compatibility.
    ``foreign`` is a valid, non-empty owner belonging to another scene.
    ``invalid`` covers malformed owner fields or a missing expected ID, while
    ``missing`` means there is no metadata mapping to authorize at all.

    Callers may read ``owned`` and ``legacy`` caches. Destructive operations
    should refuse ``foreign`` and ``invalid`` so a custom path shared by two
    scenes cannot delete the other scene's bake.
    """
    if not isinstance(metadata, dict):
        return OWNERSHIP_MISSING
    if not isinstance(expected_owner_id, str) or not expected_owner_id.strip():
        return OWNERSHIP_INVALID
    if OWNER_KEY not in metadata:
        return OWNERSHIP_LEGACY
    recorded = metadata[OWNER_KEY]
    if not isinstance(recorded, str) or not recorded.strip():
        return OWNERSHIP_INVALID
    if recorded.strip() == expected_owner_id.strip():
        return OWNERSHIP_OWNED
    return OWNERSHIP_FOREIGN


def _atomic_path(cache_dir: str, suffix: str):
    """Yield a same-directory temporary path suitable for ``os.replace``."""
    os.makedirs(cache_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(
        dir=cache_dir, prefix=".stflip-writing-", suffix=suffix)
    return fd, path


def frame_path(cache_dir: str, frame: int) -> str:
    return os.path.join(cache_dir, f"stflip_{frame:06d}.npz")


def checkpoint_path(cache_dir: str, frame: int) -> str:
    """Return the raw, restart-capable solver checkpoint path for ``frame``."""
    if isinstance(frame, bool) or not isinstance(frame, Integral):
        raise ValueError("checkpoint frame must be an integer")
    return os.path.join(cache_dir, f"stflip_checkpoint_{int(frame):06d}.npz")


def surface_path(cache_dir: str, frame: int, fingerprint: str) -> str:
    """Return the immutable paper-surface cache path for one configuration."""
    if isinstance(frame, bool) or not isinstance(frame, Integral):
        raise ValueError("surface frame must be an integer")
    fingerprint = _checkpoint_fingerprint(fingerprint, allow_empty=False)
    return os.path.join(
        cache_dir,
        f"stflip_surface_{fingerprint}_{int(frame):06d}.npz",
    )


class SurfaceCacheError(ValueError):
    """A paper-surface cache entry exists but fails its strict schema."""


def surface_config_fingerprint(config: dict) -> str:
    """Hash one canonical, JSON-compatible reconstruction configuration."""
    if not isinstance(config, dict):
        raise SurfaceCacheError("surface configuration must be a mapping")
    try:
        encoded = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise SurfaceCacheError(
            "surface configuration is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_surface_metadata(metadata: dict) -> str:
    """Validate paper-cache provenance and return its configuration hash."""
    if not isinstance(metadata, dict):
        raise SurfaceCacheError("surface metadata must be a mapping")
    if metadata.get("schema") != SURFACE_SCHEMA:
        raise SurfaceCacheError("surface metadata schema is invalid")
    version = metadata.get("version")
    if (isinstance(version, bool) or not isinstance(version, Integral)
            or int(version) not in {SURFACE_LEGACY_VERSION, SURFACE_VERSION}):
        raise SurfaceCacheError("surface metadata version is unsupported")
    if metadata.get("mode") != "PAPER_MCF":
        raise SurfaceCacheError("surface metadata mode is invalid")

    config = metadata.get("config")
    if not isinstance(config, dict):
        raise SurfaceCacheError("surface metadata configuration is invalid")
    if config.get("schema") != SURFACE_CONFIG_SCHEMA:
        raise SurfaceCacheError("surface configuration schema is invalid")
    config_version = config.get("version")
    if (isinstance(config_version, bool)
            or not isinstance(config_version, Integral)
            or int(config_version) != SURFACE_CONFIG_VERSION):
        raise SurfaceCacheError("surface configuration version is unsupported")
    try:
        fingerprint = _checkpoint_fingerprint(
            metadata.get("fingerprint"), allow_empty=False)
    except CheckpointError as exc:
        raise SurfaceCacheError(
            "surface metadata fingerprint is invalid") from exc
    if surface_config_fingerprint(config) != fingerprint:
        raise SurfaceCacheError(
            "surface metadata configuration fingerprint does not match")
    return fingerprint


def _surface_indices(value, width: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise SurfaceCacheError(f"surface {name} is not an index array") from exc
    if array.ndim != 2 or array.shape[1:] != (width,):
        raise SurfaceCacheError(
            f"surface {name} must have shape (N, {width})")
    if not np.issubdtype(array.dtype, np.integer):
        raise SurfaceCacheError(f"surface {name} must contain integers")
    owned = np.array(array, dtype=np.int32, order="C", copy=True)
    if array.size and not np.array_equal(array, owned):
        raise SurfaceCacheError(f"surface {name} indices exceed int32")
    return owned


def validate_surface_mesh(vertices, triangles, quads) -> tuple:
    """Return owned canonical arrays for a Blender-compatible surface mesh."""
    try:
        vertices = np.asarray(vertices)
    except (TypeError, ValueError) as exc:
        raise SurfaceCacheError("surface vertices are not numeric") from exc
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or not np.issubdtype(vertices.dtype, np.number)
            or not np.isrealobj(vertices)):
        raise SurfaceCacheError("surface vertices must have shape (N, 3)")
    vertices = np.array(vertices, dtype=np.float32, order="C", copy=True)
    if not bool(np.all(np.isfinite(vertices))):
        raise SurfaceCacheError("surface vertices must be finite float32")
    triangles = _surface_indices(triangles, 3, "triangles")
    quads = _surface_indices(quads, 4, "quads")
    vertex_count = int(vertices.shape[0])
    for name, faces in (("triangles", triangles), ("quads", quads)):
        if faces.size and (int(faces.min()) < 0
                           or int(faces.max()) >= vertex_count):
            raise SurfaceCacheError(
                f"surface {name} reference missing vertices")
    return vertices, triangles, quads


def surface_source_fingerprint(positions) -> str:
    """Hash the canonical float32 particle positions used by reconstruction."""
    try:
        array = np.asarray(positions)
    except (TypeError, ValueError) as exc:
        raise SurfaceCacheError("surface source positions are not numeric") from exc
    if (array.ndim != 2 or array.shape[1:] != (3,)
            or not np.issubdtype(array.dtype, np.number)
            or not np.isrealobj(array)):
        raise SurfaceCacheError(
            "surface source positions must have shape (N, 3)")
    owned = np.array(array, dtype="<f4", order="C", copy=True)
    if not bool(np.all(np.isfinite(owned))):
        raise SurfaceCacheError("surface source positions must be finite")
    digest = hashlib.sha256()
    digest.update(b"stflip-paper-surface-source-v1\0")
    digest.update(np.asarray(owned.shape, dtype="<i8").tobytes())
    digest.update(owned.tobytes(order="C"))
    return digest.hexdigest()


def _surface_mesh_digest(vertices, triangles, quads) -> str:
    digest = hashlib.sha256()
    digest.update(b"stflip-paper-surface-mesh-v1\0")
    for label, array, dtype in (
        (b"vertices", vertices, "<f4"),
        (b"triangles", triangles, "<i4"),
        (b"quads", quads, "<i4"),
    ):
        canonical = np.asarray(array, dtype=dtype, order="C")
        digest.update(label + b"\0")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def surface_mesh_fingerprint(vertices, triangles, quads) -> str:
    """Hash one validated mesh independently of NPZ container bytes."""
    mesh = validate_surface_mesh(vertices, triangles, quads)
    return _surface_mesh_digest(*mesh)


def write_surface(
    cache_dir: str,
    frame: int,
    fingerprint: str,
    vertices,
    triangles,
    quads,
    *,
    source_positions=None,
) -> str:
    """Atomically cache one immutable Appendix-B output mesh.

    Version 2 binds the mesh to the solver-local float32 particle positions
    actually consumed by reconstruction. Version 1 archives remain readable
    and are interpreted as bound to the legacy world-space particle frame.
    """
    fingerprint = _checkpoint_fingerprint(fingerprint, allow_empty=False)
    vertices, triangles, quads = validate_surface_mesh(
        vertices, triangles, quads)
    if source_positions is None:
        source_positions = np.empty((0, 3), dtype=np.float32)
    source_fingerprint = surface_source_fingerprint(source_positions)
    mesh_fingerprint = _surface_mesh_digest(vertices, triangles, quads)
    path = surface_path(cache_dir, frame, fingerprint)
    fd, temporary = _atomic_path(cache_dir, ".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(SURFACE_SCHEMA),
                version=np.asarray(SURFACE_VERSION, dtype=np.int64),
                frame=np.asarray(int(frame), dtype=np.int64),
                fingerprint=np.asarray(fingerprint),
                source_positions_sha256=np.asarray(source_fingerprint),
                mesh_sha256=np.asarray(mesh_fingerprint),
                vertices=vertices,
                triangles=triangles,
                quads=quads,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def read_surface(
    cache_dir: str,
    frame: int,
    fingerprint: str,
    *,
    expected_source_positions=None,
    expected_legacy_source_positions=None,
):
    """Read a strict cached paper mesh, returning ``None`` only if absent."""
    fingerprint = _checkpoint_fingerprint(fingerprint, allow_empty=False)
    path = surface_path(cache_dir, frame, fingerprint)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != _SURFACE_KEYS:
                raise SurfaceCacheError("surface archive keys do not match schema")
            schema = data["schema"]
            version = data["version"]
            archived_frame = data["frame"]
            archived_fingerprint = data["fingerprint"]
            source_fingerprint = data["source_positions_sha256"]
            mesh_fingerprint = data["mesh_sha256"]
            if (schema.shape != () or schema.dtype.kind not in {"U", "S"}
                    or str(schema.item()) != SURFACE_SCHEMA):
                raise SurfaceCacheError("surface schema identifier is invalid")
            if version.shape != () or version.dtype != np.dtype(np.int64):
                raise SurfaceCacheError("surface schema version is unsupported")
            archived_version = int(version)
            if archived_version not in {
                    SURFACE_LEGACY_VERSION, SURFACE_VERSION}:
                raise SurfaceCacheError("surface schema version is unsupported")
            if (archived_frame.shape != ()
                    or archived_frame.dtype != np.dtype(np.int64)
                    or int(archived_frame) != int(frame)):
                raise SurfaceCacheError(
                    "surface frame binding does not match filename")
            archived_fingerprint = _checkpoint_fingerprint(
                archived_fingerprint, allow_empty=False)
            if archived_fingerprint != fingerprint:
                raise SurfaceCacheError(
                    "surface fingerprint binding does not match filename")
            source_fingerprint = _checkpoint_fingerprint(
                source_fingerprint, allow_empty=False)
            expected_positions = expected_source_positions
            if archived_version == SURFACE_LEGACY_VERSION:
                expected_positions = (
                    expected_legacy_source_positions
                    if expected_legacy_source_positions is not None
                    else expected_source_positions)
            elif (expected_positions is None
                  and expected_legacy_source_positions is not None):
                raise SurfaceCacheError(
                    "surface local source particle positions are unavailable")
            if expected_positions is not None:
                expected_source = surface_source_fingerprint(
                    expected_positions)
                if source_fingerprint != expected_source:
                    raise SurfaceCacheError(
                        "surface source particle positions do not match")
            mesh = validate_surface_mesh(
                data["vertices"], data["triangles"], data["quads"])
            mesh_fingerprint = _checkpoint_fingerprint(
                mesh_fingerprint, allow_empty=False)
            if mesh_fingerprint != _surface_mesh_digest(*mesh):
                raise SurfaceCacheError("surface mesh fingerprint does not match")
            return mesh
    except SurfaceCacheError:
        raise
    except (OSError, TypeError, ValueError, KeyError, EOFError,
            zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise SurfaceCacheError(f"surface {frame} is corrupt") from exc


def metrics_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, METRICS_NAME)


def write_frame(cache_dir: str, frame: int, positions: np.ndarray,
                velocities: np.ndarray, attributes: dict | None = None) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = frame_path(cache_dir, frame)
    fd, temporary = _atomic_path(cache_dir, ".npz")
    # Optional per-particle shading attributes (age, source, speed, ...) are
    # stored under attr_<name> keys.  read_frame ignores unknown keys, so this
    # stays backward-compatible with caches and readers that predate them.
    extra = {}
    if attributes:
        for name, values in attributes.items():
            arr = np.asarray(values)
            if arr.shape[0] == positions.shape[0]:
                extra[f"attr_{name}"] = arr
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                positions=positions.astype(np.float32),
                velocities=velocities.astype(np.float32),
                **extra,
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
    except (OSError, TypeError, ValueError, KeyError,
            zipfile.BadZipFile, zipfile.LargeZipFile):
        return None


def read_frame_attributes(cache_dir: str, frame: int) -> dict:
    """Return the per-particle shading attributes stored for ``frame``.

    Keys are the attribute names (``age``, ``source``, ``speed``, ...) without
    the ``attr_`` prefix.  Empty dict when the frame is missing or predates the
    attribute format."""
    path = frame_path(cache_dir, frame)
    if not os.path.isfile(path):
        return {}
    try:
        out = {}
        with np.load(path) as data:
            for key in data.files:
                if key.startswith("attr_"):
                    out[key[len("attr_"):]] = data[key]
        return out
    except (OSError, TypeError, ValueError, KeyError,
            zipfile.BadZipFile, zipfile.LargeZipFile):
        return {}


class CheckpointError(ValueError):
    """A checkpoint exists but cannot safely restore solver state."""


def _checkpoint_array(value, shape, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint {name} is not a numeric array") from exc
    if array.shape != shape or array.dtype != np.dtype(np.float32):
        raise CheckpointError(
            f"checkpoint {name} must be float32 with shape {shape}")
    if not bool(np.all(np.isfinite(array))):
        raise CheckpointError(f"checkpoint {name} contains non-finite values")
    return np.array(array, dtype=np.float32, order="C", copy=True)


def _checkpoint_scalar(value, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(np.float64):
        raise CheckpointError(f"checkpoint {name} must be a float64 scalar")
    result = float(array)
    if not np.isfinite(result) or result < 0.0:
        raise CheckpointError(f"checkpoint {name} must be finite and non-negative")
    return result


def _checkpoint_counter(value, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(np.int64):
        raise CheckpointError(f"checkpoint {name} must be an int64 scalar")
    result = int(array)
    if result < 0:
        raise CheckpointError(f"checkpoint {name} must be non-negative")
    return result


def _checkpoint_affine(value, count: int) -> np.ndarray:
    """Validate an APIC affine matrix stack: (N, 3, 3) or empty (0, 3, 3)."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint C is not a numeric array") from exc
    if array.dtype != np.dtype(np.float32) or array.ndim != 3 \
            or array.shape[1:] != (3, 3) or array.shape[0] not in (0, count):
        raise CheckpointError(
            f"checkpoint C must be float32 with shape ({count}, 3, 3) "
            "or (0, 3, 3)")
    if not bool(np.all(np.isfinite(array))):
        raise CheckpointError("checkpoint C contains non-finite values")
    return np.array(array, dtype=np.float32, order="C", copy=True)


def _checkpoint_particle_id(value, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64) or array.shape != (count,):
        raise CheckpointError(
            "checkpoint particle_id must be an int64 array of shape (N,)")
    if array.size:
        if int(array.min()) < 0:
            raise CheckpointError(
                "checkpoint particle_id must be non-negative")
        if np.unique(array).size != count:
            raise CheckpointError("checkpoint particle_id must be unique")
    return np.array(array, dtype=np.int64, order="C", copy=True)


def _checkpoint_source_id(value, count: int) -> np.ndarray:
    """Validate the per-particle source id array: int32, shape (N,), >= 0."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            "checkpoint source_id is not a numeric array") from exc
    if array.dtype != np.dtype(np.int32) or array.shape != (count,):
        raise CheckpointError(
            f"checkpoint source_id must be int32 with shape ({count},)")
    if array.size and int(array.min()) < 0:
        raise CheckpointError("checkpoint source_id must be non-negative")
    return np.array(array, dtype=np.int32, order="C", copy=True)


def _checkpoint_rng_json(value) -> tuple[dict, np.ndarray]:
    """Validate and canonically encode one NumPy bit-generator state."""
    if not isinstance(value, dict):
        raise CheckpointError("checkpoint rng_state must be a mapping")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        decoded = json.loads(encoded.decode("utf-8"))
        probe = np.random.default_rng()
        probe.bit_generator.state = decoded
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError("checkpoint rng_state is invalid") from exc
    return decoded, np.frombuffer(encoded, dtype=np.uint8).copy()


def validate_checkpoint_state(state: dict) -> dict:
    """Return an owned, normalized copy of a complete solver checkpoint.

    The restart format intentionally excludes grids and derived geometry. They
    are rebuilt by the next solver step; only trajectory-defining mutable state
    is persisted.  Array dtypes are fixed so CPU and CUDA restore identically.
    """
    if not isinstance(state, dict):
        raise CheckpointError("checkpoint state must be a mapping")
    required = {
        "pos", "vel", "dt_resid", "time", "dt_prev",
        "rng_state", "outflow_removed_total", "volume_outflow_removed_total",
        "pressure_outflow_removed_total",
    }
    allowed = required | _CHECKPOINT_OPTIONAL_STATE_KEYS
    if not required <= set(state) or not set(state) <= allowed:
        missing = sorted(required - set(state))
        extra = sorted(set(state) - allowed)
        raise CheckpointError(
            f"checkpoint state keys mismatch (missing={missing}, extra={extra})")

    positions_value = np.asarray(state["pos"])
    if positions_value.ndim != 2 or positions_value.shape[1:] != (3,):
        raise CheckpointError("checkpoint pos must have shape (N, 3)")
    positions = _checkpoint_array(
        positions_value, positions_value.shape, "pos")
    count = positions.shape[0]
    velocities = _checkpoint_array(
        state["vel"], positions.shape, "vel")
    dt_resid = _checkpoint_array(
        state["dt_resid"], (count,), "dt_resid")
    rng_state, _ = _checkpoint_rng_json(state["rng_state"])
    normalized = {
        "pos": positions,
        "vel": velocities,
        "dt_resid": dt_resid,
        "time": _checkpoint_scalar(state["time"], "time"),
        "dt_prev": _checkpoint_scalar(state["dt_prev"], "dt_prev"),
        "rng_state": rng_state,
        "outflow_removed_total": _checkpoint_counter(
            state["outflow_removed_total"], "outflow_removed_total"),
        "volume_outflow_removed_total": _checkpoint_counter(
            state["volume_outflow_removed_total"],
            "volume_outflow_removed_total"),
        "pressure_outflow_removed_total": _checkpoint_counter(
            state["pressure_outflow_removed_total"],
            "pressure_outflow_removed_total"),
    }
    # Optional per-particle members are validated only when present so that a
    # version-1 checkpoint (which omits them) still normalizes cleanly.
    if "phase" in state:
        normalized["phase"] = _checkpoint_array(state["phase"], (count,), "phase")
    if "C" in state:
        normalized["C"] = _checkpoint_affine(state["C"], count)
    if "age" in state:
        normalized["age"] = _checkpoint_array(state["age"], (count,), "age")
    if "source_id" in state:
        normalized["source_id"] = _checkpoint_source_id(state["source_id"], count)
    if "particle_id" in state:
        normalized["particle_id"] = _checkpoint_particle_id(
            state["particle_id"], count)
    if "next_particle_id" in state:
        normalized["next_particle_id"] = _checkpoint_counter(
            state["next_particle_id"], "next_particle_id")
    if "substep_index" in state:
        normalized["substep_index"] = _checkpoint_counter(
            state["substep_index"], "substep_index")
    if "gamma_prev" in state:
        gamma_prev = _checkpoint_array(
            state["gamma_prev"], (count,), "gamma_prev")
        if gamma_prev.size and (
                float(gamma_prev.min()) < 0.0
                or float(gamma_prev.max()) > 1.0):
            raise CheckpointError(
                "checkpoint gamma_prev must lie in [0, 1]")
        normalized["gamma_prev"] = gamma_prev
    return normalized


def _checkpoint_fingerprint(value, *, allow_empty: bool) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != () or value.dtype.kind not in {"U", "S"}:
            raise CheckpointError("checkpoint fingerprint is invalid")
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise CheckpointError("checkpoint fingerprint is invalid") from exc
    if not isinstance(value, str):
        raise CheckpointError("checkpoint fingerprint is invalid")
    if value == "" and allow_empty:
        return value
    if _FINGERPRINT_RE.fullmatch(value) is None:
        raise CheckpointError("checkpoint fingerprint is invalid")
    return value


def write_checkpoint(
    cache_dir: str,
    frame: int,
    state: dict,
    *,
    fingerprint: str = "",
) -> str:
    """Atomically write a raw restart checkpoint bound to frame and inputs."""
    normalized = validate_checkpoint_state(state)
    _, rng_bytes = _checkpoint_rng_json(normalized["rng_state"])
    path = checkpoint_path(cache_dir, frame)
    fingerprint = _checkpoint_fingerprint(fingerprint, allow_empty=True)
    # A version-2+ archive always carries the extra per-particle members.
    # When a caller supplies only base state, persist the historical defaults
    # so the archive still round-trips (liquid phase, empty affine, zero
    # shading, synthesized ids 0..n-1 -- the same identities a version-2
    # restore would synthesize).
    count = normalized["pos"].shape[0]
    phase = normalized.get(
        "phase", np.ones((count,), dtype=np.float32))
    affine_c = normalized.get(
        "C", np.zeros((0, 3, 3), dtype=np.float32))
    age = normalized.get(
        "age", np.zeros((count,), dtype=np.float32))
    source_id = normalized.get(
        "source_id", np.zeros((count,), dtype=np.int32))
    particle_id = normalized.get(
        "particle_id", np.arange(count, dtype=np.int64))
    next_particle_id = normalized.get("next_particle_id", count)
    substep_index = normalized.get("substep_index", 0)
    optional_members = {}
    if "gamma_prev" in normalized:
        optional_members["gamma_prev"] = normalized["gamma_prev"]
    fd, temporary = _atomic_path(cache_dir, ".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(
                stream,
                schema=np.asarray(CHECKPOINT_SCHEMA),
                version=np.asarray(CHECKPOINT_VERSION, dtype=np.int64),
                frame=np.asarray(int(frame), dtype=np.int64),
                fingerprint=np.asarray(fingerprint),
                pos=normalized["pos"],
                vel=normalized["vel"],
                dt_resid=normalized["dt_resid"],
                time=np.asarray(normalized["time"], dtype=np.float64),
                dt_prev=np.asarray(normalized["dt_prev"], dtype=np.float64),
                rng_state_json=rng_bytes,
                outflow_removed_total=np.asarray(
                    normalized["outflow_removed_total"], dtype=np.int64),
                volume_outflow_removed_total=np.asarray(
                    normalized["volume_outflow_removed_total"], dtype=np.int64),
                pressure_outflow_removed_total=np.asarray(
                    normalized["pressure_outflow_removed_total"], dtype=np.int64),
                phase=phase,
                affine_c=affine_c,
                age=age,
                source_id=source_id,
                particle_id=particle_id,
                next_particle_id=np.asarray(
                    int(next_particle_id), dtype=np.int64),
                substep_index=np.asarray(
                    int(substep_index), dtype=np.int64),
                **optional_members,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def read_checkpoint(
    cache_dir: str,
    frame: int,
    *,
    expected_fingerprint: str | None = None,
) -> dict | None:
    """Read a strict restart checkpoint, returning ``None`` only if absent.

    Existing but malformed files raise :class:`CheckpointError`, allowing the
    Blender resume operator to distinguish corruption from a cache that was
    created by an older add-on without checkpoints.
    """
    path = checkpoint_path(cache_dir, frame)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            if "version" not in files or "schema" not in files:
                raise CheckpointError(
                    "checkpoint archive keys do not match schema")
            version = data["version"]
            if (version.shape != () or version.dtype != np.dtype(np.int64)
                    or int(version) not in _CHECKPOINT_KEYS_BY_VERSION):
                raise CheckpointError("checkpoint schema version is unsupported")
            archived_version = int(version)
            required_keys = _CHECKPOINT_KEYS_BY_VERSION[archived_version]
            optional_keys = _CHECKPOINT_OPTIONAL_KEYS_BY_VERSION[
                archived_version]
            if not required_keys <= files <= (required_keys | optional_keys):
                raise CheckpointError(
                    "checkpoint archive keys do not match schema")
            schema = data["schema"]
            if (schema.shape != () or schema.dtype.kind not in {"U", "S"}
                    or str(schema.item()) != CHECKPOINT_SCHEMA):
                raise CheckpointError("checkpoint schema identifier is invalid")
            archived_frame = data["frame"]
            if (archived_frame.shape != ()
                    or archived_frame.dtype != np.dtype(np.int64)
                    or int(archived_frame) != int(frame)):
                raise CheckpointError("checkpoint frame binding does not match filename")
            archived_fingerprint = _checkpoint_fingerprint(
                data["fingerprint"], allow_empty=True)
            if expected_fingerprint is not None:
                expected = _checkpoint_fingerprint(
                    expected_fingerprint, allow_empty=False)
                if archived_fingerprint != expected:
                    raise CheckpointError(
                        "checkpoint simulation fingerprint does not match")
            rng_bytes = data["rng_state_json"]
            if rng_bytes.ndim != 1 or rng_bytes.dtype != np.dtype(np.uint8):
                raise CheckpointError("checkpoint rng_state_json is invalid")
            try:
                rng_state = json.loads(rng_bytes.tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CheckpointError("checkpoint rng_state_json is invalid") from exc
            state = {
                "pos": data["pos"],
                "vel": data["vel"],
                "dt_resid": data["dt_resid"],
                "time": data["time"],
                "dt_prev": data["dt_prev"],
                "rng_state": rng_state,
                "outflow_removed_total": data["outflow_removed_total"],
                "volume_outflow_removed_total": data[
                    "volume_outflow_removed_total"],
                "pressure_outflow_removed_total": data[
                    "pressure_outflow_removed_total"],
            }
            if archived_version >= 2:
                # ``affine_c`` maps to the solver's ``C`` attribute name.
                state["phase"] = data["phase"]
                state["C"] = data["affine_c"]
                state["age"] = data["age"]
                state["source_id"] = data["source_id"]
            if archived_version >= 3:
                state["particle_id"] = data["particle_id"]
                state["next_particle_id"] = int(data["next_particle_id"])
                state["substep_index"] = int(data["substep_index"])
                if "gamma_prev" in files:
                    state["gamma_prev"] = data["gamma_prev"]
            return validate_checkpoint_state(state)
    except CheckpointError:
        raise
    except (OSError, TypeError, ValueError, KeyError, EOFError,
            zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise CheckpointError(f"checkpoint {frame} is corrupt") from exc


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
        match = _FRAME_RE.fullmatch(name)
        if match is not None:
            out.append(int(match.group(1)))
    return sorted(out)


def checkpoint_frames(cache_dir: str) -> list[int]:
    """Return frames whose checkpoint files pass the complete strict schema."""
    if not os.path.isdir(cache_dir):
        return []
    out = []
    for name in os.listdir(cache_dir):
        match = _CHECKPOINT_RE.fullmatch(name)
        if match is None:
            continue
        frame = int(match.group(1))
        try:
            if read_checkpoint(cache_dir, frame) is not None:
                out.append(frame)
        except CheckpointError:
            continue
    return sorted(set(out))


def surface_frames(cache_dir: str, fingerprint: str) -> list[int]:
    """Return strict paper-surface frames for one reconstruction config."""
    fingerprint = _checkpoint_fingerprint(fingerprint, allow_empty=False)
    if not os.path.isdir(cache_dir):
        return []
    out = []
    for name in os.listdir(cache_dir):
        match = _SURFACE_RE.fullmatch(name)
        if match is None or match.group(1) != fingerprint:
            continue
        frame = int(match.group(2))
        try:
            if read_surface(cache_dir, frame, fingerprint) is not None:
                out.append(frame)
        except SurfaceCacheError:
            continue
    return sorted(set(out))


def resumable_frames(cache_dir: str, metadata: dict | None = None) -> list[int]:
    """Return atomically committed frames with both output and solver state."""
    readable_output = set(committed_frames(cache_dir, metadata))
    return sorted(readable_output.intersection(checkpoint_frames(cache_dir)))


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
