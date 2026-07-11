"""Deterministic initial and inflow velocity fields.

Fields are evaluated on host NumPy arrays before particle state is uploaded to
the selected solver backend.  Keeping this setup path in float32 on the host
makes seeded CPU and CUDA particle states bitwise identical for a fixed seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


VelocityVector: TypeAlias = Sequence[float] | np.ndarray


def _vec3(value: VelocityVector, name: str) -> tuple[float, float, float]:
    try:
        untyped = np.asarray(value)
        if np.iscomplexobj(untyped):
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(untyped, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain exactly three finite values"
        ) from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite values")
    return tuple(float(component) for component in array)


def _positions(value: np.ndarray) -> np.ndarray:
    try:
        untyped = np.asarray(value)
        if np.iscomplexobj(untyped):
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            positions = np.asarray(untyped, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "positions_local must have shape (N, 3) and contain finite values"
        ) from exc
    if (positions.ndim != 2 or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))):
        raise ValueError(
            "positions_local must have shape (N, 3) and contain finite values"
        )
    return np.ascontiguousarray(positions)


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(samples)):
        raise ValueError("velocity field produced non-finite values")
    return np.ascontiguousarray(samples, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class UniformVelocity:
    """A constant velocity along the solver-local grid axes."""

    value: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _vec3(self.value, "value"))

    def sample(self, positions_local: np.ndarray) -> np.ndarray:
        """Evaluate the field at particle positions as an owned float32 array."""
        positions = _positions(positions_local)
        samples = np.empty(positions.shape, dtype=np.float32)
        samples[:] = np.asarray(self.value, dtype=np.float32)
        return samples


@dataclass(frozen=True, slots=True)
class SolidBodyRotation:
    """Rigid velocity field ``linear + omega x (position - center)``.

    ``center`` uses solver-local distance units.  ``angular_velocity`` is a
    right-handed vector in radians per simulation-time unit; its magnitude is
    angular speed.  ``linear_velocity`` is superposed in solver distance per
    simulation-time unit.
    """

    center: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center, "center"))
        object.__setattr__(
            self,
            "angular_velocity",
            _vec3(self.angular_velocity, "angular_velocity"),
        )
        object.__setattr__(
            self,
            "linear_velocity",
            _vec3(self.linear_velocity, "linear_velocity"),
        )

    def sample(self, positions_local: np.ndarray) -> np.ndarray:
        """Evaluate the field at actual particle positions on the host."""
        positions = _positions(positions_local)
        center = np.asarray(self.center, dtype=np.float32)
        omega = np.asarray(self.angular_velocity, dtype=np.float32)
        linear = np.asarray(self.linear_velocity, dtype=np.float32)
        samples = np.empty(positions.shape, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            relative = positions - center
            samples[:, 0] = (
                omega[1] * relative[:, 2] - omega[2] * relative[:, 1]
            )
            samples[:, 1] = (
                omega[2] * relative[:, 0] - omega[0] * relative[:, 2]
            )
            samples[:, 2] = (
                omega[0] * relative[:, 1] - omega[1] * relative[:, 0]
            )
            samples += linear
        return _validate_samples(samples)


VelocityField: TypeAlias = UniformVelocity | SolidBodyRotation
VelocityInput: TypeAlias = VelocityField | VelocityVector


def as_velocity_field(value: VelocityInput) -> VelocityField:
    """Normalize a legacy constant vector or return a built-in field."""
    if isinstance(value, (UniformVelocity, SolidBodyRotation)):
        return value
    return UniformVelocity(value)
