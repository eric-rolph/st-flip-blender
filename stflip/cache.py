"""Disk cache for baked frames: one compressed .npz per frame + a meta file."""

from __future__ import annotations

import json
import os

import numpy as np

META_NAME = "stflip_meta.json"


def frame_path(cache_dir: str, frame: int) -> str:
    return os.path.join(cache_dir, f"stflip_{frame:06d}.npz")


def write_frame(cache_dir: str, frame: int, positions: np.ndarray,
                velocities: np.ndarray) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = frame_path(cache_dir, frame)
    np.savez_compressed(path,
                        positions=positions.astype(np.float32),
                        velocities=velocities.astype(np.float32))
    return path


def read_frame(cache_dir: str, frame: int):
    path = frame_path(cache_dir, frame)
    if not os.path.isfile(path):
        return None
    with np.load(path) as data:
        return data["positions"], data["velocities"]


def write_meta(cache_dir: str, meta: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, META_NAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def read_meta(cache_dir: str) -> dict | None:
    path = os.path.join(cache_dir, META_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        if (name.startswith("stflip_") and name.endswith(".npz")) or name == META_NAME:
            os.remove(os.path.join(cache_dir, name))
            n += 1
    return n
