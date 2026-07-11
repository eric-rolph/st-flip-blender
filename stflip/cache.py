"""Disk cache for baked frames: one compressed .npz per frame + a meta file."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

META_NAME = "stflip_meta.json"


def _atomic_path(cache_dir: str, suffix: str):
    """Yield a same-directory temporary path suitable for ``os.replace``."""
    os.makedirs(cache_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(
        dir=cache_dir, prefix=".stflip-writing-", suffix=suffix)
    return fd, path


def frame_path(cache_dir: str, frame: int) -> str:
    return os.path.join(cache_dir, f"stflip_{frame:06d}.npz")


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
                or name.startswith(".stflip-writing-")):
            os.remove(os.path.join(cache_dir, name))
            n += 1
    return n
