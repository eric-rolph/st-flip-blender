"""Deterministic tiled-grid layout primitives and sparsity telemetry.

This module is the first, deliberately solver-independent increment toward a
packed tiled grid.  It turns a dense active-cell mask into a stable set of
``tile_size**3`` blocks, provides the tile-coordinate lookup needed by future
stencil kernels, and packs/unpacks arbitrary cell-centred fields.

No simulation path depends on this module yet.  Keeping the representation
standalone lets every later tiled operator prove parity with the trusted dense
solver before it is allowed into a production step.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable

import numpy as np


FACE_NEIGHBOR_OFFSETS = np.asarray(
    [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
     (0, 1, 0), (0, 0, -1), (0, 0, 1)],
    dtype=np.int32,
)
ALL_NEIGHBOR_OFFSETS = np.asarray(
    [offset for offset in itertools.product((-1, 0, 1), repeat=3)
     if offset != (0, 0, 0)],
    dtype=np.int32,
)


def _positive_shape(shape: Iterable[int], name: str) -> tuple[int, int, int]:
    values = tuple(shape)
    if len(values) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0 for value in values):
        raise ValueError(f"{name} must contain three positive integers")
    return tuple(int(value) for value in values)


def _positive_int(value: int, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _sorted_unique(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    unique = np.unique(np.asarray(coords, dtype=np.int32), axis=0)
    order = np.lexsort((unique[:, 2], unique[:, 1], unique[:, 0]))
    return np.ascontiguousarray(unique[order], dtype=np.int32)


def _tile_footprint_cells(
    coords: np.ndarray,
    shape: tuple[int, int, int],
    tile_size: int,
) -> int:
    total = 0
    for coord in coords:
        lo = coord.astype(np.int64) * tile_size
        hi = np.minimum(lo + tile_size, np.asarray(shape, dtype=np.int64))
        total += int(np.prod(hi - lo, dtype=np.int64))
    return total


@dataclass(frozen=True, slots=True)
class TileTelemetry:
    """Storage opportunity measured from one active-cell mask."""

    domain_cells: int
    active_cells: int
    active_fraction: float
    bounding_box_cells: int
    bounding_box_fraction: float
    core_tiles: int
    core_tile_cells: int
    core_tile_fraction: float
    halo_tiles: int
    halo_tile_cells: int
    halo_tile_fraction: float
    bbox_to_core_tile_ratio: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class TileLayout:
    """Packed tile coordinates and a dense coarse lookup table.

    ``core_coords`` contains tiles touched by active cells. ``coords`` also
    contains the requested tile halo. Both arrays are lexicographically sorted,
    so slot assignment and serialized diagnostics are deterministic.
    """

    shape: tuple[int, int, int]
    tile_size: int
    halo_tiles: int
    tile_grid_shape: tuple[int, int, int]
    core_coords: np.ndarray
    coords: np.ndarray
    table: np.ndarray

    @property
    def tile_count(self) -> int:
        return int(self.coords.shape[0])

    @property
    def core_tile_count(self) -> int:
        return int(self.core_coords.shape[0])

    def slot(self, coord: Iterable[int]) -> int:
        values = tuple(int(value) for value in coord)
        if len(values) != 3:
            raise ValueError("tile coordinate must contain three integers")
        if any(value < 0 or value >= self.tile_grid_shape[axis]
               for axis, value in enumerate(values)):
            return -1
        return int(self.table[values])

    def neighbor_slots(self, *, include_diagonals: bool = True) -> np.ndarray:
        offsets = (ALL_NEIGHBOR_OFFSETS if include_diagonals
                   else FACE_NEIGHBOR_OFFSETS)
        result = np.full((self.tile_count, len(offsets)), -1, dtype=np.int32)
        limits = np.asarray(self.tile_grid_shape, dtype=np.int32)
        for slot, coord in enumerate(self.coords):
            neighbors = coord[None, :] + offsets
            valid = np.all((neighbors >= 0) & (neighbors < limits), axis=1)
            if np.any(valid):
                selected = neighbors[valid]
                result[slot, valid] = self.table[
                    selected[:, 0], selected[:, 1], selected[:, 2]
                ]
        return result

    def with_halo(self, packed: np.ndarray) -> np.ndarray:
        """Return packed tiles with a one-cell 26-neighbour halo.

        Missing neighbours and cells beyond a partial Domain-edge tile are
        zero.  The operation reads only the packed field and tile table; it does
        not reconstruct a dense Domain array, which makes it the exchange
        primitive later tiled stencil operators can reuse on CPU or GPU.
        """

        values = np.asarray(packed)
        prefix = (self.tile_count,) + (self.tile_size,) * 3
        if values.ndim < 4 or tuple(values.shape[:4]) != prefix:
            raise ValueError(
                f"packed field must begin with shape {prefix}, got "
                f"{values.shape}")
        halo_shape = (
            self.tile_count,
            self.tile_size + 2,
            self.tile_size + 2,
            self.tile_size + 2,
        ) + values.shape[4:]
        halo = np.zeros(halo_shape, dtype=values.dtype)
        interior = (slice(None),) + (slice(1, self.tile_size + 1),) * 3
        halo[interior] = values
        neighbors = self.neighbor_slots(include_diagonals=True)
        for target_slot in range(self.tile_count):
            for offset_index, offset in enumerate(ALL_NEIGHBOR_OFFSETS):
                source_slot = int(neighbors[target_slot, offset_index])
                if source_slot < 0:
                    continue
                target_slices = []
                source_slices = []
                for delta in offset:
                    if delta < 0:
                        target_slices.append(slice(0, 1))
                        source_slices.append(slice(self.tile_size - 1,
                                                   self.tile_size))
                    elif delta > 0:
                        target_slices.append(slice(self.tile_size + 1,
                                                   self.tile_size + 2))
                        source_slices.append(slice(0, 1))
                    else:
                        target_slices.append(slice(1, self.tile_size + 1))
                        source_slices.append(slice(0, self.tile_size))
                halo[(target_slot, *target_slices)] = values[
                    (source_slot, *source_slices)]
        return halo

    def pack(self, field: np.ndarray) -> np.ndarray:
        """Pack a cell-centred field over ``coords``, padding edge tiles."""

        values = np.asarray(field)
        if values.ndim < 3 or tuple(values.shape[:3]) != self.shape:
            raise ValueError(
                f"field must begin with domain shape {self.shape}, got "
                f"{values.shape}")
        packed_shape = (
            self.tile_count,
            self.tile_size,
            self.tile_size,
            self.tile_size,
        ) + values.shape[3:]
        packed = np.zeros(packed_shape, dtype=values.dtype)
        for slot, coord in enumerate(self.coords):
            lo = coord.astype(np.int64) * self.tile_size
            hi = np.minimum(lo + self.tile_size, np.asarray(self.shape))
            source = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
            target = tuple(slice(0, int(hi[a] - lo[a])) for a in range(3))
            packed[(slot,) + target] = values[source]
        return packed

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        """Scatter a packed field into a zero-filled dense domain array."""

        values = np.asarray(packed)
        prefix = (self.tile_count,) + (self.tile_size,) * 3
        if values.ndim < 4 or tuple(values.shape[:4]) != prefix:
            raise ValueError(
                f"packed field must begin with shape {prefix}, got "
                f"{values.shape}")
        dense = np.zeros(self.shape + values.shape[4:], dtype=values.dtype)
        for slot, coord in enumerate(self.coords):
            lo = coord.astype(np.int64) * self.tile_size
            hi = np.minimum(lo + self.tile_size, np.asarray(self.shape))
            target = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
            source = tuple(slice(0, int(hi[a] - lo[a])) for a in range(3))
            dense[target] = values[(slot,) + source]
        return dense

    def telemetry(self, active_mask: np.ndarray) -> TileTelemetry:
        mask = np.asarray(active_mask)
        if mask.dtype != np.bool_ or mask.shape != self.shape:
            raise ValueError(
                f"active_mask must be bool with shape {self.shape}")
        domain_cells = int(mask.size)
        active_cells = int(np.count_nonzero(mask))
        if active_cells:
            active = np.argwhere(mask)
            expected_core = _sorted_unique(active // self.tile_size)
            if not np.array_equal(expected_core, self.core_coords):
                raise ValueError(
                    "active_mask activates different core tiles than this layout")
            extent = active.max(axis=0) - active.min(axis=0) + 1
            bbox_cells = int(np.prod(extent, dtype=np.int64))
        else:
            if self.core_tile_count:
                raise ValueError(
                    "active_mask activates different core tiles than this layout")
            bbox_cells = 0
        core_cells = _tile_footprint_cells(
            self.core_coords, self.shape, self.tile_size)
        halo_cells = _tile_footprint_cells(
            self.coords, self.shape, self.tile_size)
        return TileTelemetry(
            domain_cells=domain_cells,
            active_cells=active_cells,
            active_fraction=active_cells / domain_cells,
            bounding_box_cells=bbox_cells,
            bounding_box_fraction=bbox_cells / domain_cells,
            core_tiles=self.core_tile_count,
            core_tile_cells=core_cells,
            core_tile_fraction=core_cells / domain_cells,
            halo_tiles=self.tile_count,
            halo_tile_cells=halo_cells,
            halo_tile_fraction=halo_cells / domain_cells,
            bbox_to_core_tile_ratio=(
                bbox_cells / core_cells if core_cells else None),
        )


def build_tile_layout(
    active_mask: np.ndarray,
    *,
    tile_size: int = 8,
    halo_tiles: int = 1,
) -> TileLayout:
    """Return a deterministic tile layout for a three-dimensional bool mask."""

    mask = np.asarray(active_mask)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("active_mask must be a three-dimensional bool array")
    shape = _positive_shape(mask.shape, "active_mask shape")
    tile_size = _positive_int(tile_size, "tile_size")
    halo_tiles = _nonnegative_int(halo_tiles, "halo_tiles")
    tile_grid_shape = tuple(math.ceil(size / tile_size) for size in shape)

    active_cells = np.argwhere(mask)
    core = _sorted_unique(active_cells // tile_size)
    if core.size and halo_tiles:
        offsets = np.asarray(
            list(itertools.product(
                range(-halo_tiles, halo_tiles + 1), repeat=3)),
            dtype=np.int32,
        )
        expanded = (core[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        limits = np.asarray(tile_grid_shape, dtype=np.int32)
        expanded = expanded[np.all(
            (expanded >= 0) & (expanded < limits), axis=1)]
        coords = _sorted_unique(expanded)
    else:
        coords = core.copy()

    table = np.full(tile_grid_shape, -1, dtype=np.int32)
    if coords.size:
        table[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(
            len(coords), dtype=np.int32)
    core.setflags(write=False)
    coords.setflags(write=False)
    table.setflags(write=False)
    return TileLayout(
        shape=shape,
        tile_size=tile_size,
        halo_tiles=halo_tiles,
        tile_grid_shape=tile_grid_shape,
        core_coords=core,
        coords=coords,
        table=table,
    )


__all__ = [
    "ALL_NEIGHBOR_OFFSETS",
    "FACE_NEIGHBOR_OFFSETS",
    "TileLayout",
    "TileTelemetry",
    "build_tile_layout",
]
