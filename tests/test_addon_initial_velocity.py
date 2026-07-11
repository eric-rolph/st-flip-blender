"""Bpy-free tests for Blender initial-velocity field resolution."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_operators(monkeypatch, tmp_path):
    root = "_stflip_initial_velocity_test"
    for name in (root, f"{root}.addon", f"{root}.stflip"):
        monkeypatch.setitem(sys.modules, name, _package(name))

    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(Operator=object)
    bpy.utils = types.SimpleNamespace(
        user_resource=lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    velocity = _load_module(
        monkeypatch,
        f"{root}.stflip.velocity",
        ROOT / "stflip" / "velocity.py",
    )
    cache = types.ModuleType(f"{root}.stflip.cache")
    backend = types.ModuleType(f"{root}.stflip.backend")
    backend.cuda_device_name = lambda: None
    backend.cuda_diagnostics = lambda *args, **kwargs: {
        "available": False,
        "error": "not installed",
    }
    backend.get_backend = lambda name="auto": None
    solver = types.ModuleType(f"{root}.stflip.solver")
    solver.Params = object
    solver.STFLIPSolver = object
    monkeypatch.setitem(sys.modules, cache.__name__, cache)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, solver.__name__, solver)

    for leaf in ("handlers", "mesher", "voxelize"):
        module = types.ModuleType(f"{root}.addon.{leaf}")
        if leaf == "handlers":
            module.resolve_cache_dir = lambda scene: ""
        monkeypatch.setitem(sys.modules, module.__name__, module)

    operators = _load_module(
        monkeypatch,
        f"{root}.addon.operators",
        ROOT / "addon" / "operators.py",
    )
    return operators, velocity


def _settings(**overrides):
    values = {
        "initial_velocity_mode": "UNIFORM",
        "initial_velocity": (1.25, -2.0, 3.5),
        "rotation_center_world": (0.0, 0.0, 0.0),
        "rotation_axis_world": (0.0, 0.0, 1.0),
        "angular_speed": 0.1,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_uniform_resolution_preserves_legacy_world_vector(
        monkeypatch, tmp_path):
    operators, velocity = _load_operators(monkeypatch, tmp_path)

    field, descriptor = operators.resolve_liquid_initial_velocity(
        _settings(), (10.0, 20.0, 30.0), "Uniform Source")

    assert isinstance(field, velocity.UniformVelocity)
    assert field.value == pytest.approx((1.25, -2.0, 3.5))
    assert descriptor == {
        "name": "Uniform Source",
        "initial_velocity": list(field.value),
        "initial_velocity_mode": "UNIFORM",
    }


def test_solid_body_resolution_subtracts_origin_normalizes_axis_and_keeps_sign(
        monkeypatch, tmp_path):
    operators, velocity = _load_operators(monkeypatch, tmp_path)
    settings = _settings(
        initial_velocity_mode="SOLID_BODY",
        rotation_center_world=(11.0, 22.0, 33.0),
        rotation_axis_world=(0.0, 0.0, -2.0),
        angular_speed=-0.5,
    )

    field, descriptor = operators.resolve_liquid_initial_velocity(
        settings, (10.0, 20.0, 30.0), "Vortex Source")

    assert isinstance(field, velocity.SolidBodyRotation)
    assert field.center == pytest.approx((1.0, 2.0, 3.0))
    assert field.angular_velocity == pytest.approx((0.0, 0.0, 0.5))
    assert field.linear_velocity == pytest.approx((1.25, -2.0, 3.5))
    rotation = descriptor["solid_body_rotation"]
    assert descriptor["initial_velocity"] == list(field.linear_velocity)
    assert descriptor["initial_velocity_mode"] == "SOLID_BODY"
    assert rotation == {
        "center_world": [11.0, 22.0, 33.0],
        "center_solver_local": list(field.center),
        "axis_world_authored": [0.0, 0.0, -2.0],
        "axis_world_unit": [0.0, 0.0, -1.0],
        "angular_speed_radians_per_second": -0.5,
        "angular_velocity_world": list(field.angular_velocity),
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initial_velocity": (0.0, np.nan, 0.0)}, "Initial Velocity"),
        ({"rotation_center_world": (0.0, np.inf, 0.0)}, "Rotation Center"),
        ({"rotation_axis_world": (0.0, 0.0, 0.0)}, "non-zero"),
        ({"rotation_axis_world": (0.0, np.inf, 1.0)}, "Rotation Axis"),
        ({"angular_speed": np.nan}, "Angular Speed"),
    ],
)
def test_solid_body_resolution_rejects_invalid_active_controls(
        monkeypatch, tmp_path, overrides, message):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    settings = _settings(initial_velocity_mode="SOLID_BODY", **overrides)

    with pytest.raises(ValueError, match=message) as error:
        operators.resolve_liquid_initial_velocity(
            settings, (0.0, 0.0, 0.0), "Broken Vortex")

    assert "Broken Vortex" in str(error.value)


def test_unknown_mode_and_invalid_domain_origin_are_source_specific(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="Unknown Source.*unknown"):
        operators.resolve_liquid_initial_velocity(
            _settings(initial_velocity_mode="OTHER"),
            (0.0, 0.0, 0.0),
            "Unknown Source",
        )
    with pytest.raises(ValueError, match="Origin Source.*Domain Origin"):
        operators.resolve_liquid_initial_velocity(
            _settings(),
            (0.0, 0.0),
            "Origin Source",
        )
