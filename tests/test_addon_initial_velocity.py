"""Bpy-free tests for Blender source-velocity and schedule resolution."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from stflip import cache as core_cache
from stflip import surface as core_surface


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
    for attribute in (
        "CHECKPOINT_SCHEMA",
        "CHECKPOINT_VERSION",
        "META_NAME",
        "METRICS_NAME",
        "SURFACE_CONFIG_SCHEMA",
        "SURFACE_CONFIG_VERSION",
        "SURFACE_SCHEMA",
        "SURFACE_VERSION",
        "surface_config_fingerprint",
    ):
        setattr(cache, attribute, getattr(core_cache, attribute))
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
    monkeypatch.setitem(sys.modules, f"{root}.stflip.surface", core_surface)
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
        "inflow_velocity_mode": "UNIFORM",
        "inflow_velocity": (1.25, -2.0, 3.5),
        "inflow_use_frame_range": False,
        "inflow_start_frame": 1,
        "inflow_end_frame": 250,
        "rotation_center_world": (0.0, 0.0, 0.0),
        "rotation_axis_world": (0.0, 0.0, 1.0),
        "angular_speed": 0.1,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _force_settings(**overrides):
    values = {
        "force_type": "VORTEX",
        "force_strength": 4.5,
        "force_scale": 0.5,
        "force_radius": 2.0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _fingerprint_inputs():
    params = types.SimpleNamespace(
        resolution=(4, 3, 2),
        dx=0.25,
        gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 24.0,
        seed=17,
        particles_per_cell=4,
    )
    sources = [{
        "role": "LIQUID",
        "mode": "UNIFORM",
        "velocity": [1.0, 0.0, 0.0],
        "mask": np.arange(24).reshape(4, 3, 2) % 3 == 0,
    }]
    solid = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(4, 3, 2)
    return params, sources, solid


def test_force_object_discovery_accepts_meshes_and_empties_only(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    def obj(object_type, role):
        return types.SimpleNamespace(
            type=object_type,
            stflip=types.SimpleNamespace(role=role),
        )

    mesh_force = obj("MESH", "FORCE")
    empty_force = obj("EMPTY", "FORCE")
    camera_force = obj("CAMERA", "FORCE")
    mesh_liquid = obj("MESH", "LIQUID")
    empty_liquid = obj("EMPTY", "LIQUID")
    scene = types.SimpleNamespace(objects=[
        mesh_force, empty_force, camera_force, mesh_liquid, empty_liquid,
    ])

    assert operators._fluid_objects(scene, "FORCE") == [
        mesh_force, empty_force]
    assert operators._fluid_objects(scene, "LIQUID") == [mesh_liquid]


def test_vortex_force_resolution_converts_world_center_to_solver_local(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    matrix_world = np.eye(4, dtype=np.float64)
    matrix_world[:3, 2] = (0.0, 0.0, 2.0)
    matrix_world[:3, 3] = (11.0, 22.0, 33.0)

    force, descriptor = operators.resolve_force_field(
        _force_settings(), matrix_world, (10.0, 20.0, 30.0), 17,
        "Vortex Guide")

    assert force == {
        "force_type": "VORTEX",
        "strength": 4.5,
        "axis": pytest.approx((0.0, 0.0, 1.0)),
        "center": pytest.approx((1.0, 2.0, 3.0)),
        "radius": 2.0,
    }
    assert descriptor == {
        "name": "Vortex Guide",
        "force_type": "VORTEX",
        "strength": 4.5,
        "axis_world_unit": pytest.approx([0.0, 0.0, 1.0]),
        "center_world": [11.0, 22.0, 33.0],
        "center_solver_local": pytest.approx([1.0, 2.0, 3.0]),
        "radius": 2.0,
    }


def test_turbulence_force_seed_is_stable_when_python_hash_salt_changes(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    settings = _force_settings(
        force_type="TURBULENCE", force_strength=8.0, force_scale=0.75)

    monkeypatch.setattr("builtins.hash", lambda _value: 1)
    first, first_descriptor = operators.resolve_force_field(
        settings, np.eye(4), (0.0, 0.0, 0.0), 17, "Storm Guide")
    monkeypatch.setattr("builtins.hash", lambda _value: 999_999)
    second, second_descriptor = operators.resolve_force_field(
        settings, np.eye(4), (0.0, 0.0, 0.0), 17, "Storm Guide")

    assert first == second
    assert first_descriptor == second_descriptor
    assert first["force_type"] == "TURBULENCE"
    assert first["strength"] == 8.0
    assert first["scale"] == 0.75
    assert isinstance(first["seed"], int)
    assert first["seed"] >= 0


@pytest.mark.parametrize("scene_seed", [True, -1, 1.5, np.nan])
def test_turbulence_force_rejects_invalid_scene_seed(
        monkeypatch, tmp_path, scene_seed):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="Storm Guide: Random Seed"):
        operators.resolve_force_field(
            _force_settings(force_type="TURBULENCE"),
            np.eye(4),
            (0.0, 0.0, 0.0),
            scene_seed,
            "Storm Guide",
        )


def test_directional_force_resolution_preserves_normalized_world_direction(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    matrix_world = np.eye(4, dtype=np.float64)
    matrix_world[:3, 2] = (0.0, -3.0, 0.0)

    force, descriptor = operators.resolve_force_field(
        _force_settings(force_type="DIRECTIONAL", force_strength=-2.0),
        matrix_world, (10.0, 20.0, 30.0), 17, "Wind Guide")

    assert force == {
        "force_type": "DIRECTIONAL",
        "strength": -2.0,
        "direction": pytest.approx((0.0, -1.0, 0.0)),
    }
    assert descriptor == {
        "name": "Wind Guide",
        "force_type": "DIRECTIONAL",
        "strength": -2.0,
        "direction_world_unit": pytest.approx([0.0, -1.0, 0.0]),
    }


@pytest.mark.parametrize(
    ("settings", "matrix_world", "domain_origin", "expected_label"),
    [
        (_force_settings(force_type="MAGNET"), np.eye(4), (0.0, 0.0, 0.0),
         "Force Type"),
        (_force_settings(), np.eye(3), (0.0, 0.0, 0.0), "Transform"),
        (_force_settings(), np.full((4, 4), np.nan), (0.0, 0.0, 0.0),
         "Transform"),
        (_force_settings(), np.zeros((4, 4)), (0.0, 0.0, 0.0),
         "Force Axis"),
        (_force_settings(force_strength=np.nan), np.eye(4), (0.0, 0.0, 0.0),
         "Force Strength"),
        (_force_settings(force_radius=0.0), np.eye(4), (0.0, 0.0, 0.0),
         "Vortex Radius"),
        (_force_settings(force_radius=np.inf), np.eye(4), (0.0, 0.0, 0.0),
         "Vortex Radius"),
        (_force_settings(force_type="TURBULENCE", force_scale=0.0), np.eye(4),
         (0.0, 0.0, 0.0), "Turbulence Scale"),
        (_force_settings(force_type="TURBULENCE", force_scale=np.inf),
         np.eye(4), (0.0, 0.0, 0.0), "Turbulence Scale"),
        (_force_settings(), np.eye(4), (np.nan, 0.0, 0.0), "Domain Origin"),
    ],
)
def test_force_resolution_rejects_invalid_or_non_finite_inputs(
        monkeypatch, tmp_path, settings, matrix_world, domain_origin,
        expected_label):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    with pytest.raises(ValueError) as excinfo:
        operators.resolve_force_field(
            settings, matrix_world, domain_origin, 17, "Bad Guide")

    assert "Bad Guide" in str(excinfo.value)
    assert expected_label in str(excinfo.value)


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


def test_inflow_solid_body_uses_shared_resolver_and_actual_field_values(
        monkeypatch, tmp_path):
    operators, velocity = _load_operators(monkeypatch, tmp_path)
    settings = _settings(
        inflow_velocity_mode="SOLID_BODY",
        inflow_velocity=(2.0, -1.0, 0.5),
        rotation_center_world=(11.0, 22.0, 33.0),
        rotation_axis_world=(0.0, 3.0, 0.0),
        angular_speed=0.75,
    )

    field, descriptor = operators.resolve_inflow_velocity(
        settings, (10.0, 20.0, 30.0), "Rotating Inflow")

    assert isinstance(field, velocity.SolidBodyRotation)
    assert field.center == pytest.approx((1.0, 2.0, 3.0))
    assert field.angular_velocity == pytest.approx((0.0, 0.75, 0.0))
    assert field.linear_velocity == pytest.approx((2.0, -1.0, 0.5))
    assert descriptor["velocity"] == list(field.linear_velocity)
    assert descriptor["velocity_mode"] == "SOLID_BODY"
    assert descriptor["solid_body_rotation"][
        "angular_velocity_world"] == list(field.angular_velocity)
    assert operators._resolved_velocity_fingerprint(
        descriptor, "velocity", "velocity_mode") == {
            "velocity": list(field.linear_velocity),
            "velocity_mode": "SOLID_BODY",
            "solid_body_rotation": {
                "center_solver_local": list(field.center),
                "angular_velocity_world": list(field.angular_velocity),
            },
        }


def test_inflow_uniform_mode_preserves_legacy_vector(monkeypatch, tmp_path):
    operators, velocity = _load_operators(monkeypatch, tmp_path)

    field, descriptor = operators.resolve_inflow_velocity(
        _settings(), (10.0, 20.0, 30.0), "Uniform Inflow")

    assert isinstance(field, velocity.UniformVelocity)
    assert descriptor == {
        "name": "Uniform Inflow",
        "velocity": list(field.value),
        "velocity_mode": "UNIFORM",
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

    with pytest.raises(ValueError, match="Broken Inflow.*Inflow Velocity"):
        operators.resolve_inflow_velocity(
            _settings(inflow_velocity=(0.0, np.inf, 0.0)),
            (0.0, 0.0, 0.0),
            "Broken Inflow",
        )
    with pytest.raises(ValueError, match="Origin Source.*Domain Origin"):
        operators.resolve_liquid_initial_velocity(
            _settings(),
            (0.0, 0.0),
            "Origin Source",
        )


def test_inclusive_inflow_frames_translate_to_half_open_solver_time(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    settings = _settings(
        inflow_use_frame_range=True,
        inflow_start_frame=8,
        inflow_end_frame=10,
    )

    start, end, descriptor, warning = operators.resolve_inflow_schedule(
        settings, 5, 20, 0.25, "Scheduled Inflow")

    assert start == pytest.approx(0.5)
    assert end == pytest.approx(1.25)
    assert warning is None
    assert descriptor["active_frame_range"] == {
        "mode": "LIMITED",
        "authored_inclusive": [8, 10],
        "effective_inclusive": [8, 10],
        "solver_time_seconds": {
            "start_inclusive": pytest.approx(0.5),
            "end_exclusive": pytest.approx(1.25),
        },
    }


def test_inflow_schedule_clamps_before_bake_and_warns_outside_request(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    partial = _settings(
        inflow_use_frame_range=True,
        inflow_start_frame=2,
        inflow_end_frame=6,
    )
    start, end, descriptor, warning = operators.resolve_inflow_schedule(
        partial, 5, 20, 0.25, "Partial")
    assert (start, end) == pytest.approx((0.0, 0.25))
    assert descriptor["active_frame_range"][
        "effective_inclusive"] == [6, 6]
    assert "inclusive overlap 6-6" in warning

    inactive = _settings(
        inflow_use_frame_range=True,
        inflow_start_frame=1,
        inflow_end_frame=4,
    )
    start, end, descriptor, warning = operators.resolve_inflow_schedule(
        inactive, 5, 20, 0.25, "Inactive")
    assert (start, end) == (0.0, 0.0)
    assert descriptor["active_frame_range"]["effective_inclusive"] is None
    assert "deliberately inactive" in warning


def test_unbounded_and_invalid_inflow_schedules(monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    start, end, descriptor, warning = operators.resolve_inflow_schedule(
        _settings(), 1, 48, 1.0 / 24.0, "Unbounded")
    assert (start, end, warning) == (0.0, None, None)
    assert descriptor["active_frame_range"]["mode"] == "UNBOUNDED"

    assert operators._inflow_schedule_overlaps(0.0, None, 1.0)
    assert operators._inflow_schedule_overlaps(0.25, 0.5, 1.0)
    assert not operators._inflow_schedule_overlaps(1.0, None, 1.0)
    assert not operators._inflow_schedule_overlaps(0.0, 0.0, 1.0)
    assert not operators._inflow_schedule_overlaps(2.0, 3.0, 1.0)

    with pytest.raises(ValueError, match="Start Frame.*exceed"):
        operators.resolve_inflow_schedule(
            _settings(
                inflow_use_frame_range=True,
                inflow_start_frame=12,
                inflow_end_frame=11,
            ),
            1,
            48,
            1.0 / 24.0,
            "Backwards",
        )


def test_source_mask_applies_solids_counts_cells_and_rejects_shape_mismatch(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    obj = types.SimpleNamespace(name="Outlet")
    raw = np.array([[[True], [True]], [[False], [True]]])
    operators.voxelize.mask_from_object = lambda *args: raw.copy()
    not_solid = np.array([[[True], [False]], [[True], [True]]])

    mask, count = operators._source_mask(
        obj, object(), np.zeros(3), 0.5, (2, 2, 1), not_solid)

    assert mask.dtype == np.bool_
    assert count == 2
    assert np.array_equal(
        mask, np.array([[[True], [False]], [[False], [True]]]))

    with pytest.raises(ValueError, match="Outlet.*does not match"):
        operators._source_mask(
            obj, object(), np.zeros(3), 0.5, (2, 1, 2), None)


def test_bake_lifecycle_keeps_error_and_progress_coherent(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    settings = types.SimpleNamespace(
        bake_state="IDLE", bake_status="", bake_error="", bake_progress=0.0)

    operators._set_bake_lifecycle(
        settings, "RUNNING", "Frame 2", progress=1.7)
    assert settings.bake_state == "RUNNING"
    assert settings.bake_status == "Frame 2"
    assert settings.bake_error == ""
    assert settings.bake_progress == 1.0

    operators._fail_bake(settings, "pressure diverged")
    assert settings.bake_state == "FAILED"
    assert settings.bake_error == "pressure diverged"
    assert settings.bake_status == "Bake failed: pressure diverged"


def test_solver_params_include_advanced_blender_controls(monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    captured = {}

    def fake_params(**kwargs):
        captured.update(kwargs)
        return captured

    operators.Params = fake_params
    settings = types.SimpleNamespace(
        cfl_target=12.0,
        particles_per_cell=6,
        seed=17,
        flip_blend=0.97,
        st_enabled=True,
        jitter_strength=0.8,
        adaptive_gamma=False,
        eta_phi=0.4,
        density=998.2,
        local_cfl=0.6,
        pcg_tolerance=2e-5,
        pcg_max_iterations=321,
        pressure_solver="multigrid",
        density_floor_relative=7e-4,
        transfer="apic",
        two_phase=True,
        rho_gas=1.3,
        gas_particles_per_cell=5,
        surface_tension=0.02,
        sparse=True,
        viscosity=0.03,
        sheeting=0.4,
    )

    result = operators._solver_params(
        settings, (8, 6, 4), 0.25, (0.0, 0.0, -9.81), 30.0)

    assert result is captured
    assert captured == {
        "resolution": (8, 6, 4),
        "dx": 0.25,
        "gravity": (0.0, 0.0, -9.81),
        "frame_dt": pytest.approx(1.0 / 30.0),
        "cfl_target": 12.0,
        "particles_per_cell": 6,
        "seed": 17,
        "flip_blend": 0.97,
        "st_enabled": True,
        "jitter_strength": 0.8,
        "adaptive_gamma": False,
        "eta_phi": 0.4,
        "rho": 998.2,
        "cfl_local": 0.6,
        "pcg_tol": 2e-5,
        "pcg_max_iter": 321,
        "pressure_solver": "multigrid",
        "eps_rho_rel": 7e-4,
        "transfer": "apic",
        "two_phase": True,
        "rho_gas": 1.3,
        "gas_particles_per_cell": 5,
        "surface_tension": 0.02,
        "sparse": True,
        "viscosity": 0.03,
        "sheeting": 0.4,
    }


def test_simulation_fingerprint_is_stable_and_input_complete(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    params, sources, solid = _fingerprint_inputs()
    arguments = (
        params, (4, 3, 2), 0.25, (1.0, 2.0, 3.0), "cpu", sources,
        solid, None,
    )

    first = operators.simulation_fingerprint(*arguments)
    second = operators.simulation_fingerprint(*arguments)

    assert first == second
    assert len(first) == 64
    assert operators.fingerprint_matches(first, second)
    assert not operators.fingerprint_matches(first, "not-a-fingerprint")

    mutations = []
    changed_params = types.SimpleNamespace(**vars(params))
    changed_params.seed += 1
    mutations.append((changed_params, arguments[1:]))
    changed_mask_sources = [{**sources[0], "mask": sources[0]["mask"].copy()}]
    changed_mask_sources[0]["mask"][0, 0, 0] ^= True
    mutations.append((params, (*arguments[1:5], changed_mask_sources,
                               *arguments[6:])))
    changed_velocity_sources = [{**sources[0], "velocity": [2.0, 0.0, 0.0]}]
    mutations.append((params, (*arguments[1:5], changed_velocity_sources,
                               *arguments[6:])))
    changed_solid = solid.copy()
    changed_solid[0, 0, 0] += 0.5
    mutations.append((params, (*arguments[1:6], changed_solid, None)))

    for changed_first, changed_rest in mutations:
        assert operators.simulation_fingerprint(
            changed_first, *changed_rest) != first


def test_simulation_fingerprint_tracks_source_order_and_outflow_mode(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    params, sources, solid = _fingerprint_inputs()
    outflow = {
        "role": "OUTFLOW",
        "mode": "VOLUME",
        "mask": np.zeros((4, 3, 2), dtype=bool),
    }
    outflow["mask"][0, 0, 0] = True

    baseline = operators.simulation_fingerprint(
        params, (4, 3, 2), 0.25, (0, 0, 0), "cpu",
        [sources[0], outflow], solid,
    )
    reordered = operators.simulation_fingerprint(
        params, (4, 3, 2), 0.25, (0, 0, 0), "cpu",
        [outflow, sources[0]], solid,
    )
    pressure = operators.simulation_fingerprint(
        params, (4, 3, 2), 0.25, (0, 0, 0), "cpu",
        [sources[0], {**outflow, "mode": "PRESSURE"}], solid,
    )

    assert baseline != reordered
    assert baseline != pressure


def test_simulation_fingerprint_changes_with_normalized_force_inputs(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    params, sources, solid = _fingerprint_inputs()
    force = {
        "name": "Vortex Guide",
        "force_type": "VORTEX",
        "strength": 4.5,
        "axis_world_unit": [0.0, 0.0, 1.0],
        "center_world": [11.0, 22.0, 33.0],
        "center_solver_local": [1.0, 2.0, 3.0],
        "radius": 2.0,
    }

    baseline = operators.simulation_fingerprint(
        params, (4, 3, 2), 0.25, (10.0, 20.0, 30.0), "cpu",
        sources, solid, forces=[force])
    changed = operators.simulation_fingerprint(
        params, (4, 3, 2), 0.25, (10.0, 20.0, 30.0), "cpu",
        sources, solid, forces=[{**force, "strength": 5.0}])
    renamed = operators.simulation_fingerprint(
        params, (4, 3, 2), 0.25, (10.0, 20.0, 30.0), "cpu",
        sources, solid, forces=[{**force, "name": "Display Name Only"}])

    assert baseline != changed
    assert baseline == renamed


def test_inflow_fingerprint_uses_resolved_field_and_inclusive_schedule(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    params, sources, solid = _fingerprint_inputs()
    mask = sources[0]["mask"]

    def fingerprint(settings):
        _field, descriptor = operators.resolve_inflow_velocity(
            settings, (1.0, 2.0, 3.0), "Fingerprint Inflow")
        _start, _end, schedule, _warning = operators.resolve_inflow_schedule(
            settings, 1, 48, params.frame_dt, "Fingerprint Inflow")
        source = {
            "role": "INFLOW",
            "velocity": operators._resolved_velocity_fingerprint(
                descriptor, "velocity", "velocity_mode"),
            "active_frame_range": schedule["active_frame_range"],
            "mask": mask,
        }
        return operators.simulation_fingerprint(
            params, (4, 3, 2), 0.25, (1.0, 2.0, 3.0), "cpu",
            [source], solid,
        )

    baseline = _settings(
        inflow_velocity_mode="SOLID_BODY",
        inflow_use_frame_range=True,
        inflow_start_frame=3,
        inflow_end_frame=5,
        rotation_axis_world=(0.0, 0.0, 2.0),
        angular_speed=1.25,
    )
    # Axis magnitude is authored metadata but normalizes to the same field.
    equivalent = _settings(**{
        **vars(baseline),
        "rotation_axis_world": (0.0, 0.0, 4.0),
    })
    changed_speed = _settings(**{
        **vars(baseline),
        "angular_speed": 1.5,
    })
    changed_schedule = _settings(**{
        **vars(baseline),
        "inflow_end_frame": 6,
    })

    assert fingerprint(equivalent) == fingerprint(baseline)
    assert fingerprint(changed_speed) != fingerprint(baseline)
    assert fingerprint(changed_schedule) != fingerprint(baseline)


def test_resume_metadata_requires_matching_fingerprint_and_extension(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    operators.cache.CHECKPOINT_SCHEMA = "stflip-solver-checkpoint"
    operators.cache.CHECKPOINT_VERSION = 1
    fingerprint = "a" * 64
    metadata = {
        "frame_start": 1,
        "frame_end_baked": 2,
        "checkpoint": {
            "schema": operators.cache.CHECKPOINT_SCHEMA,
            "version": operators.cache.CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "latest_frame": 2,
            "state": "CANCELLED",
        },
    }

    assert operators.validate_resume_metadata(
        metadata, fingerprint, 1, 3) == 2
    with pytest.raises(ValueError, match="inputs changed"):
        operators.validate_resume_metadata(metadata, "b" * 64, 1, 3)
    with pytest.raises(ValueError, match="extend Scene End"):
        operators.validate_resume_metadata(metadata, fingerprint, 1, 2)


def test_resume_operator_is_an_independent_blender_rna_type(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    assert operators.STFLIP_OT_resume_bake.__bases__ == (
        operators.bpy.types.Operator,)
    for callback in ("execute", "invoke", "modal"):
        assert callback in operators.STFLIP_OT_resume_bake.__dict__


@pytest.mark.parametrize(
    ("cache_dir", "blend_filepath", "expected"),
    [
        ("//stflip_cache", "", True),
        ("relative/cache", "", True),
        ("C:/absolute/cache", "", False),
        ("//stflip_cache", "C:/project/scene.blend", False),
    ],
)
def test_relative_cache_requires_saved_blend(
        monkeypatch, tmp_path, cache_dir, blend_filepath, expected):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    assert operators.relative_cache_needs_saved_blend(
        cache_dir, blend_filepath) is expected


def test_replacing_setup_clears_owned_bake_and_refuses_foreign_cache(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    settings = types.SimpleNamespace(
        bake_state="COMPLETE", bake_status="old", bake_error="",
        bake_progress=1.0,
    )
    scene = types.SimpleNamespace(stflip=settings)
    cleared = []
    outputs = []
    operators.handlers.ensure_scene_cache_id = lambda value: "scene-id"
    operators.handlers.scene_cache_ownership = lambda value: "owned"
    operators.handlers.clear_scene_output = lambda value: outputs.append(value)
    operators.resolve_cache_dir = lambda value: str(tmp_path / "cache")
    operators.cache.clear = lambda value: cleared.append(value) or 3

    assert operators._clear_bake_for_new_setup(scene) == 3
    assert cleared == [str(tmp_path / "cache")]
    assert outputs == [scene]
    assert settings.bake_state == "IDLE"
    assert settings.bake_status == ""
    assert settings.bake_progress == 0.0

    operators.handlers.scene_cache_ownership = lambda value: "foreign"
    with pytest.raises(ValueError, match="ownership is foreign"):
        operators._clear_bake_for_new_setup(scene)
    assert len(cleared) == 1


def test_setup_replacement_counts_owned_files_and_requires_confirmation(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for name in (
        "stflip_000001.npz",
        "stflip_checkpoint_000001.npz",
        "stflip_meta.json",
        "stflip_metrics.jsonl",
        "keep-me.txt",
    ):
        (cache_dir / name).write_bytes(b"x")
    scene = types.SimpleNamespace()
    operators.cache.META_NAME = "stflip_meta.json"
    operators.cache.METRICS_NAME = "stflip_metrics.jsonl"
    operators.resolve_cache_dir = lambda value: str(cache_dir)
    operators.handlers.scene_cache_ownership = lambda value: "owned"

    assert operators._owned_setup_cache_file_count(scene) == 4

    confirmations = []
    window_manager = types.SimpleNamespace(
        invoke_confirm=lambda *args, **kwargs: (
            confirmations.append((args, kwargs)) or {"RUNNING_MODAL"}
        )
    )
    context = types.SimpleNamespace(
        scene=scene, window_manager=window_manager)
    operator = types.SimpleNamespace(
        execute=lambda value: pytest.fail("execute bypassed confirmation"))

    assert operators._invoke_setup_replace_confirmation(
        operator, context, object()) == {"RUNNING_MODAL"}
    assert "cannot restore" in confirmations[0][1]["message"]

    operators.handlers.scene_cache_ownership = lambda value: "missing"
    assert operators._owned_setup_cache_file_count(scene) == 4

    operators.handlers.scene_cache_ownership = lambda value: "foreign"
    assert operators._owned_setup_cache_file_count(scene) == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(frame_start=0), "range"),
        (lambda value: value["checkpoint"].update(version=999), "schema"),
        (lambda value: value["checkpoint"].update(latest_frame=1), "marker"),
        (lambda value: value["checkpoint"].update(state="UNKNOWN"), "state"),
    ],
)
def test_resume_metadata_rejects_corrupt_commit_markers(
        monkeypatch, tmp_path, mutation, match):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)
    operators.cache.CHECKPOINT_SCHEMA = "stflip-solver-checkpoint"
    operators.cache.CHECKPOINT_VERSION = 1
    metadata = {
        "frame_start": 1,
        "frame_end_baked": 2,
        "checkpoint": {
            "schema": operators.cache.CHECKPOINT_SCHEMA,
            "version": operators.cache.CHECKPOINT_VERSION,
            "fingerprint": "c" * 64,
            "latest_frame": 2,
            "state": "COMPLETE",
        },
    }
    mutation(metadata)

    with pytest.raises(ValueError, match=match):
        operators.validate_resume_metadata(metadata, "c" * 64, 1, 3)


def test_whirlpool_provenance_is_explicitly_approximate(monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    class Scene(dict):
        stflip = types.SimpleNamespace(resolution=48)

    assert operators._scene_setup_provenance(Scene()) is None
    scene = Scene(stflip_setup="WHIRLPOOL_PREVIEW_APPROXIMATE")

    provenance = operators._scene_setup_provenance(scene)

    assert provenance["exact_reproduction"] is False
    assert provenance["published_constraints"] == {
        "domain_dimensions_m": [200.0, 200.0, 80.0],
        "outlet_diameter_m": 20.0,
        "outlet_length_m": 10.0,
        "angular_speed_radians_per_second": 0.1,
    }
    assert provenance["preview_resolution_longest_axis"] == 48
    assert any("not paper MCF" in item for item in provenance["limitations"])


def test_high_cfl_jet_provenance_separates_published_and_preview_values(
        monkeypatch, tmp_path):
    operators, _velocity = _load_operators(monkeypatch, tmp_path)

    class Obj(dict):
        def __init__(
            self,
            role,
            dimensions,
            *,
            location=(0.0, 0.0, 0.0),
            rotation_euler=(0.0, 0.0, 0.0),
            **settings,
        ):
            super().__init__(stflip_generated_setup="HIGH_CFL_JET_LEAK")
            self.dimensions = dimensions
            self.location = location
            self.rotation_euler = rotation_euler
            self.stflip = types.SimpleNamespace(role=role, **settings)

    domain = types.SimpleNamespace(
        dimensions=(6.0, 6.0, 6.0),
        location=(0.0, 0.0, 3.0),
    )
    inflow = Obj(
        "INFLOW", (1.0, 1.0, 0.5),
        location=(0.0, 0.0, 5.5),
        inflow_velocity=(0.0, 0.0, -48.0),
        inflow_velocity_mode="UNIFORM",
        inflow_use_frame_range=True,
        inflow_start_frame=2,
        inflow_end_frame=48,
        rotation_center_world=(0.0, 0.0, 0.0),
        rotation_axis_world=(0.0, 0.0, 1.0),
        angular_speed=0.1,
    )
    plate = Obj(
        "OBSTACLE", (4.0, 4.0, 0.125), location=(0.0, 0.0, 2.0))
    outlet = Obj(
        "OUTFLOW",
        (5.75, 5.75, 0.125),
        location=(0.0, 0.0, 0.0625),
        outflow_mode="PRESSURE",
    )

    class Scene(dict):
        stflip = types.SimpleNamespace(
            resolution=48,
            domain=domain,
            cfl_target=16.0,
            local_cfl=1.0,
            particles_per_cell=8,
            st_enabled=True,
            jitter_strength=1.0,
            adaptive_gamma=True,
            eta_phi=0.5,
            flip_blend=0.98,
        )
        render = types.SimpleNamespace(fps=24, fps_base=1.0)
        objects = (inflow, plate, outlet)
        use_gravity = True
        gravity = (0.0, 0.0, -9.81)

    scene = Scene(stflip_setup="HIGH_CFL_JET_LEAK_APPROXIMATE")
    provenance = operators._scene_setup_provenance(scene)

    assert provenance["kind"] == "HIGH_CFL_JET_LEAK_APPROXIMATE"
    assert provenance["exact_reproduction"] is False
    assert provenance["preset_intact"] is True
    assert provenance["paper_figure"] == 21
    assert provenance["published_constraints"] == {
        "target_cfl": 16.0,
        "obstacle_thickness_grid_cells": 1.0,
        "local_collision_cfl": 1.0,
    }
    assert provenance["preview_choices"]["jet_speed_meters_per_second"] == 48.0
    assert provenance["preview_choices"]["resolution_longest_axis"] == 48
    assert provenance["current_values"]["jet_velocity_meters_per_second"] \
        == [0.0, 0.0, -48.0]
    assert provenance["current_values"]["nominal_jet_cells_per_frame"] \
        == pytest.approx(16.0)
    assert provenance["current_values"]["plate_thickness_grid_cells"] \
        == pytest.approx(1.0)

    scene.stflip.resolution = 64
    changed = operators._scene_setup_provenance(scene)
    assert changed["preset_intact"] is False
    assert changed["current_values"]["plate_thickness_grid_cells"] \
        == pytest.approx(4.0 / 3.0)
    scene.stflip.resolution = 48
    inflow.stflip.inflow_velocity = (48.0, 0.0, 0.0)
    redirected = operators._scene_setup_provenance(scene)
    assert redirected["preset_intact"] is False
    inflow.stflip.inflow_velocity = (0.0, 0.0, -48.0)
    inflow.stflip.inflow_velocity_mode = "SOLID_BODY"
    rotational = operators._scene_setup_provenance(scene)
    assert rotational["preset_intact"] is False
    inflow.stflip.inflow_velocity_mode = "UNIFORM"
    scene.stflip.st_enabled = False
    instantaneous = operators._scene_setup_provenance(scene)
    assert instantaneous["preset_intact"] is False
    assert any(
        "does not publish" in limitation
        for limitation in provenance["limitations"]
    )
