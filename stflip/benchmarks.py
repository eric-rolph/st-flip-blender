"""Reproducible paper-reference scene geometry and validation metrics.

The benchmark definitions in this module are deliberately independent of
Blender.  Published dimensions are kept separate from discretisation choices,
and every derived grid reports its effective extent.  Experimental samples are
never embedded or digitised from figures: callers may provide an attributable
CSV and receive deterministic, interpolation-based error metrics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


KLEEFSMAN_DOI = "10.1016/j.jcp.2004.12.007"
STFLIP_DOI = "10.1145/3811289"


def _finite_positive(value: float, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _inside_box_sdf(points: np.ndarray, lo, hi) -> np.ndarray:
    """Positive-inside signed distance for an axis-aligned box."""

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    q = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return -(outside + inside)


def _inside_z_cylinder_sdf(
    points: np.ndarray,
    center_xy: tuple[float, float],
    z_range: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Positive-inside signed distance for a capped z-axis cylinder."""

    radial = np.linalg.norm(
        points[..., :2] - np.asarray(center_xy, dtype=np.float64), axis=-1)
    center_z = 0.5 * (z_range[0] + z_range[1])
    half_z = 0.5 * (z_range[1] - z_range[0])
    d = np.stack((radial - radius, np.abs(points[..., 2] - center_z) - half_z),
                 axis=-1)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return -(outside + inside)


@dataclass(frozen=True, slots=True)
class BenchmarkGrid:
    """Uniform cubic-cell discretisation of a physical benchmark extent."""

    requested_extent: tuple[float, float, float]
    shape: tuple[int, int, int]
    dx: float

    @property
    def effective_extent(self) -> tuple[float, float, float]:
        return tuple(size * self.dx for size in self.shape)

    def _points(self, *, nodes: bool) -> np.ndarray:
        axes = [
            np.arange(size + (1 if nodes else 0), dtype=np.float64) * self.dx
            + (0.0 if nodes else 0.5 * self.dx)
            for size in self.shape
        ]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def cell_points(self) -> np.ndarray:
        return self._points(nodes=False)

    def node_points(self) -> np.ndarray:
        return self._points(nodes=True)


def benchmark_grid(
    extent: Iterable[float],
    *,
    longest_resolution: int | None = None,
    dx: float | None = None,
) -> BenchmarkGrid:
    values = tuple(_finite_positive(value, "extent") for value in extent)
    if len(values) != 3:
        raise ValueError("extent must contain three positive values")
    if (longest_resolution is None) == (dx is None):
        raise ValueError("provide exactly one of longest_resolution or dx")
    if dx is None:
        if (isinstance(longest_resolution, bool)
                or not isinstance(longest_resolution, (int, np.integer))
                or int(longest_resolution) < 8):
            raise ValueError("longest_resolution must be an integer >= 8")
        dx = max(values) / int(longest_resolution)
    else:
        dx = _finite_positive(dx, "dx")
    shape = tuple(max(1, int(math.ceil(value / dx - 1e-12))) for value in values)
    return BenchmarkGrid(values, shape, float(dx))


@dataclass(frozen=True, slots=True)
class KleefsmanBenchmark:
    """Kleefsman et al. three-dimensional dam-break geometry in metres.

    Published and solver coordinates both use x=0 at the downstream/left wall.
    Water initially occupies the high-x upstream/right end of the tank and
    flows toward negative x after gate removal.
    """

    tank: tuple[float, float, float] = (3.22, 1.0, 1.0)
    initial_water_length: float = 1.228
    initial_water_height: float = 0.55
    obstacle_center: tuple[float, float, float] = (0.744, 0.5, 0.0805)
    obstacle_size: tuple[float, float, float] = (0.161, 0.403, 0.161)
    gauge_x: tuple[tuple[str, float], ...] = (
        ("H2", 0.992),
        ("H4", 2.636),
    )
    gauge_sample_y: float = 0.5

    def grid(self, *, longest_resolution: int | None = None,
             dx: float | None = None) -> BenchmarkGrid:
        return benchmark_grid(
            self.tank, longest_resolution=longest_resolution, dx=dx)

    @property
    def gate_x(self) -> float:
        """Gate location in the published and solver coordinates."""
        return self.tank[0] - self.initial_water_length

    @property
    def gauges_xy(self) -> tuple[tuple[str, tuple[float, float]], ...]:
        # The widely reproduced benchmark publishes H2/H4 x locations.  A
        # centreline y is an explicit sampling choice for this implementation,
        # not claimed as an independently published probe coordinate.
        return tuple(
            (name, (x, self.gauge_sample_y)) for name, x in self.gauge_x)

    def liquid_mask(self, grid: BenchmarkGrid) -> np.ndarray:
        points = grid.cell_points()
        return ((points[..., 0] >= self.gate_x)
                & (points[..., 0] <= self.tank[0])
                & (points[..., 1] <= self.tank[1])
                & (points[..., 2] <= self.initial_water_height))

    def _obstacle_inside(self, points: np.ndarray) -> np.ndarray:
        center = np.asarray(self.obstacle_center)
        half = 0.5 * np.asarray(self.obstacle_size)
        return _inside_box_sdf(points, center - half, center + half)

    def solid_sdf_cells(self, grid: BenchmarkGrid) -> np.ndarray:
        # The solver convention is negative in solid, positive in fluid.
        return np.asarray(-self._obstacle_inside(grid.cell_points()), dtype=np.float32)

    def solid_sdf_nodes(self, grid: BenchmarkGrid) -> np.ndarray:
        return np.asarray(-self._obstacle_inside(grid.node_points()), dtype=np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "name": "Kleefsman 3-D dam break with obstacle",
            "source_doi": KLEEFSMAN_DOI,
            "units": "metres",
            "tank": self.tank,
            "initial_water_length": self.initial_water_length,
            "initial_water_height": self.initial_water_height,
            "obstacle_size": self.obstacle_size,
            "coordinate_convention": {
                "published": (
                    "x=0 at the downstream/left wall; x increases toward the "
                    "upstream/right wall"
                ),
                "solver": (
                    "identical to the published benchmark coordinate system"
                ),
                "transform": (
                    "identity: x_solver = x_published; y and z unchanged"
                ),
            },
            "published_coordinates": {
                "initial_water_x_range": (self.gate_x, self.tank[0]),
                "gate_x": self.gate_x,
                "obstacle_center": self.obstacle_center,
                "gauge_x": dict(self.gauge_x),
            },
            "solver_coordinates": {
                "initial_water_x_range": (self.gate_x, self.tank[0]),
                "gate_x": self.gate_x,
                "obstacle_center": self.obstacle_center,
                "gauge_x": dict(self.gauge_x),
            },
            "gauge_sampling_assumption": {
                "y": self.gauge_sample_y,
                "description": "centreline cylindrical particle footprint",
            },
        }


@dataclass(frozen=True, slots=True)
class GlugBenchmark:
    """Paper-constrained two-container glugging scene.

    The ST-FLIP paper publishes the square side, connector radius/length, water
    height, and scale L, but not wall thickness or complete container placement.
    The layout below makes those remaining choices explicit and reproducible;
    it must be described as *paper-constrained*, not pixel-identical geometry.
    """

    length_scale: float = 1.0
    container_side_ratio: float = 0.5
    container_height_ratio: float = 0.5
    connector_radius_ratio: float = 0.05
    connector_length_ratio: float = 0.05
    margin_ratio: float = 0.05

    @property
    def extent(self) -> tuple[float, float, float]:
        scale = self.length_scale
        side = self.container_side_ratio * scale
        total_height = (2 * self.container_height_ratio
                        + self.connector_length_ratio) * scale
        margin = self.margin_ratio * scale
        return side + 2 * margin, side + 2 * margin, total_height + 2 * margin

    @property
    def published_dimensions(self) -> dict[str, float]:
        scale = self.length_scale
        return {
            "L": scale,
            "container_square_side": self.container_side_ratio * scale,
            "connector_radius": self.connector_radius_ratio * scale,
            "connector_length": self.connector_length_ratio * scale,
            "initial_upper_water_height": 0.5 * scale,
        }

    def grid(self, *, longest_resolution: int | None = None,
             dx: float | None = None) -> BenchmarkGrid:
        return benchmark_grid(
            self.extent, longest_resolution=longest_resolution, dx=dx)

    def _cavity_sdf(self, points: np.ndarray) -> np.ndarray:
        scale = self.length_scale
        margin = self.margin_ratio * scale
        side = self.container_side_ratio * scale
        height = self.container_height_ratio * scale
        neck = self.connector_length_ratio * scale
        x0 = y0 = margin
        x1 = y1 = margin + side
        lower_z0 = margin
        lower_z1 = lower_z0 + height
        upper_z0 = lower_z1 + neck
        upper_z1 = upper_z0 + height
        center_xy = (margin + 0.5 * side, margin + 0.5 * side)
        return np.maximum.reduce((
            _inside_box_sdf(points, (x0, y0, lower_z0), (x1, y1, lower_z1)),
            _inside_z_cylinder_sdf(
                points, center_xy, (lower_z1, upper_z0),
                self.connector_radius_ratio * scale),
            _inside_box_sdf(points, (x0, y0, upper_z0), (x1, y1, upper_z1)),
        ))

    def solid_sdf_cells(self, grid: BenchmarkGrid) -> np.ndarray:
        return np.asarray(self._cavity_sdf(grid.cell_points()), dtype=np.float32)

    def solid_sdf_nodes(self, grid: BenchmarkGrid) -> np.ndarray:
        return np.asarray(self._cavity_sdf(grid.node_points()), dtype=np.float32)

    def liquid_mask(self, grid: BenchmarkGrid) -> np.ndarray:
        scale = self.length_scale
        margin = self.margin_ratio * scale
        height = self.container_height_ratio * scale
        neck = self.connector_length_ratio * scale
        upper_z0 = margin + height + neck
        upper_water_top = upper_z0 + 0.5 * scale
        points = grid.cell_points()
        return ((self._cavity_sdf(points) > 0.0)
                & (points[..., 2] >= upper_z0)
                & (points[..., 2] <= upper_water_top))

    def region_masks(self, grid: BenchmarkGrid) -> dict[str, np.ndarray]:
        scale = self.length_scale
        margin = self.margin_ratio * scale
        height = self.container_height_ratio * scale
        neck = self.connector_length_ratio * scale
        z = grid.cell_points()[..., 2]
        return {
            "lower": (z >= margin) & (z <= margin + height),
            "connector": ((z > margin + height)
                          & (z < margin + height + neck)),
            "upper": ((z >= margin + height + neck)
                      & (z <= margin + 2 * height + neck)),
        }

    def metadata(self) -> dict[str, object]:
        return {
            "name": "ST-FLIP paper-constrained glug",
            "source_doi": STFLIP_DOI,
            "units": "metres",
            "published_dimensions": self.published_dimensions,
            "layout_assumptions": {
                "container_internal_height": (
                    self.container_height_ratio * self.length_scale),
                "container_margin": self.margin_ratio * self.length_scale,
                "wall_model": "all domain cells outside the connected cavity are solid",
            },
        }


def water_height_from_particles(
    positions: np.ndarray,
    gauge_xy: Iterable[float],
    *,
    dx: float,
    radius: float | None = None,
    floor_z: float = 0.0,
) -> float:
    """Return a grid-resolved, bottom-connected water height at one gauge.

    A vertical layer is wet when it contains at least one particle inside the
    gauge footprint.  Starting at the first layer above ``floor_z``, one empty
    layer is tolerated to avoid a single Monte-Carlo sampling hole; spray above
    a larger gap is ignored.
    """

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("positions must be a finite (N, 3) array")
    xy = tuple(float(value) for value in gauge_xy)
    if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
        raise ValueError("gauge_xy must contain two finite values")
    dx = _finite_positive(dx, "dx")
    radius = 1.5 * dx if radius is None else _finite_positive(radius, "radius")
    if not math.isfinite(floor_z):
        raise ValueError("floor_z must be finite")
    if not len(values):
        return 0.0
    horizontal = np.linalg.norm(values[:, :2] - np.asarray(xy), axis=1)
    selected = values[(horizontal <= radius) & (values[:, 2] >= floor_z)]
    if not len(selected):
        return 0.0
    layers = set(np.floor((selected[:, 2] - floor_z) / dx).astype(np.int64))
    highest = -1
    gap = 0
    layer = 0
    while gap <= 1 and layer <= max(layers):
        if layer in layers:
            highest = layer
            gap = 0
        else:
            gap += 1
        layer += 1
    return float(floor_z + (highest + 1) * dx) if highest >= 0 else 0.0


def load_gauge_reference(path: str | Path) -> dict[str, np.ndarray | str]:
    """Load an attributable ``time_s,<gauge>_m...`` reference CSV."""

    source = Path(path)
    payload = source.read_bytes()
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    fieldnames = reader.fieldnames
    if not fieldnames or len(fieldnames) < 2 or fieldnames[0] != "time_s":
        raise ValueError("reference CSV needs time_s and at least one gauge column")
    if (any(name is None or not name for name in fieldnames)
            or len(set(fieldnames)) != len(fieldnames)):
        raise ValueError("reference CSV columns must be named and unique")
    gauge_columns = fieldnames[1:]
    if any(not name.endswith("_m") or not name[:-2]
           for name in gauge_columns):
        raise ValueError("gauge columns must use names such as H2_m")
    rows = list(reader)
    if not rows:
        raise ValueError("reference CSV needs at least one data row")
    if any(
            None in row
            or any(row.get(name) is None or not row[name].strip()
                   for name in fieldnames)
            for row in rows):
        raise ValueError("reference columns must be complete with no extra values")
    result: dict[str, np.ndarray | str] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    for name in ("time_s", *gauge_columns):
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"reference column {name} must be complete and numeric") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"reference column {name} must be finite")
        result[name] = values
    times = result["time_s"]
    assert isinstance(times, np.ndarray)
    if np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("reference time_s must be non-negative and strictly increasing")
    for name in gauge_columns:
        values = result[name]
        assert isinstance(values, np.ndarray)
        if np.any(values < 0.0):
            raise ValueError(f"reference column {name} cannot contain negative heights")
    return result


def compare_gauge_series(
    simulated_time_s: Iterable[float],
    simulated: Mapping[str, Iterable[float]],
    reference: Mapping[str, np.ndarray | str],
) -> dict[str, dict[str, float | int]]:
    """Compare simulated gauges on the reference timestamps in their overlap."""

    sim_time = np.asarray(tuple(simulated_time_s), dtype=np.float64)
    if (sim_time.ndim != 1 or len(sim_time) < 2
            or not np.all(np.isfinite(sim_time))
            or np.any(np.diff(sim_time) <= 0.0)):
        raise ValueError("simulated_time_s must be finite and strictly increasing")
    try:
        ref_time = np.asarray(reference["time_s"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reference time_s must be a numeric series") from exc
    if (ref_time.ndim != 1 or len(ref_time) < 2
            or not np.all(np.isfinite(ref_time))
            or np.any(ref_time < 0.0)
            or np.any(np.diff(ref_time) <= 0.0)):
        raise ValueError(
            "reference time_s must be finite, non-negative, and strictly increasing")
    result: dict[str, dict[str, float | int]] = {}
    for gauge, series in simulated.items():
        key = f"{gauge}_m"
        if key not in reference:
            continue
        sim_values = np.asarray(tuple(series), dtype=np.float64)
        ref_values = np.asarray(reference[key], dtype=np.float64)
        if sim_values.shape != sim_time.shape or not np.all(np.isfinite(sim_values)):
            raise ValueError(f"simulated gauge {gauge} must align with simulated_time_s")
        if (ref_values.shape != ref_time.shape
                or not np.all(np.isfinite(ref_values))
                or np.any(ref_values < 0.0)):
            raise ValueError(
                f"reference gauge {key} must align with reference time_s")
        time_scale = max(
            1.0,
            abs(float(sim_time[0])),
            abs(float(sim_time[-1])),
            abs(float(ref_time[0])),
            abs(float(ref_time[-1])),
        )
        endpoint_tolerance = max(
            1e-12, 64.0 * np.finfo(np.float64).eps * time_scale)
        overlap = (
            (ref_time >= sim_time[0] - endpoint_tolerance)
            & (ref_time <= sim_time[-1] + endpoint_tolerance)
        )
        if np.count_nonzero(overlap) < 2:
            raise ValueError(f"gauge {gauge} has fewer than two overlapping samples")
        # Decimal CSV timestamps may round a frame boundary a few ulps outside
        # the simulated interval. Accepted endpoints are clipped so np.interp
        # does not silently extrapolate a tolerance-admitted sample.
        times = np.clip(ref_time[overlap], sim_time[0], sim_time[-1])
        observed = ref_values[overlap]
        predicted = np.interp(times, sim_time, sim_values)
        error = predicted - observed
        sim_peak = int(np.argmax(predicted))
        ref_peak = int(np.argmax(observed))
        result[gauge] = {
            "samples": int(len(times)),
            "rmse_m": float(np.sqrt(np.mean(error * error))),
            "mae_m": float(np.mean(np.abs(error))),
            "bias_m": float(np.mean(error)),
            "peak_height_error_m": float(
                np.max(predicted) - np.max(observed)),
            "peak_time_error_s": float(times[sim_peak] - times[ref_peak]),
        }
    if not result:
        raise ValueError("reference contains none of the simulated gauges")
    return result


__all__ = [
    "BenchmarkGrid",
    "GlugBenchmark",
    "KLEEFSMAN_DOI",
    "KleefsmanBenchmark",
    "STFLIP_DOI",
    "benchmark_grid",
    "compare_gauge_series",
    "load_gauge_reference",
    "water_height_from_particles",
]
