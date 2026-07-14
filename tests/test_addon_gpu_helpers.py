"""Pure regression tests for Blender add-on GPU setup decisions.

The production module is loaded with a very small fake ``bpy`` surface so
these tests remain runnable in ordinary CPython.  Blender operations are not
executed here; only deterministic helper logic is exercised.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from stflip import cache as core_cache
from stflip import surface as core_surface


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_operators(monkeypatch, tmp_path):
    root = "_stflip_addon_test"
    for name in (root, f"{root}.addon", f"{root}.stflip"):
        monkeypatch.setitem(sys.modules, name, _package(name))

    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(Operator=object)
    bpy.utils = types.SimpleNamespace(
        user_resource=lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)

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
        "SurfaceCacheError",
        "surface_config_fingerprint",
        "surface_mesh_fingerprint",
        "surface_source_fingerprint",
        "validate_surface_metadata",
    ):
        setattr(cache, attribute, getattr(core_cache, attribute))
    backend = types.ModuleType(f"{root}.stflip.backend")
    backend.cuda_available = lambda: False
    backend.cuda_device_name = lambda: None
    backend.cuda_diagnostics = lambda *args, **kwargs: {
        "available": False,
        "error": "not installed",
    }
    backend.get_backend = lambda name="auto": None
    solver = types.ModuleType(f"{root}.stflip.solver")
    solver.Params = object
    solver.STFLIPSolver = object
    velocity = types.ModuleType(f"{root}.stflip.velocity")
    velocity.SolidBodyRotation = object
    velocity.UniformVelocity = object
    monkeypatch.setitem(sys.modules, cache.__name__, cache)
    monkeypatch.setitem(sys.modules, f"{root}.stflip.surface", core_surface)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, solver.__name__, solver)
    monkeypatch.setitem(sys.modules, velocity.__name__, velocity)

    for leaf in ("handlers", "mesher", "voxelize"):
        module = types.ModuleType(f"{root}.addon.{leaf}")
        if leaf == "handlers":
            module.resolve_cache_dir = lambda scene: ""
        elif leaf == "mesher":
            module.place_paper_surface_object = (
                lambda obj, origin: setattr(
                    obj, "location", tuple(origin)) or obj)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    path = Path(__file__).parents[1] / "addon" / "operators.py"
    name = f"{root}.addon.operators"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_install_candidates_are_pinned_and_ordered(monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)

    requirements = [item["requirement"]
                    for item in operators.GPU_INSTALL_CANDIDATES]
    assert requirements == [
        "cupy-cuda13x[ctk]==14.1.1",
        "cupy-cuda12x==14.1.1",
    ]
    assert len({item["slug"] for item in operators.GPU_INSTALL_CANDIDATES}) == 2


def test_candidate_cleanup_removes_only_shadow_numpy(monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    candidate = tmp_path / "runtime" / "candidate"
    for name in ("numpy", "numpy.libs", "numpy-2.5.1.dist-info", "cupy"):
        (candidate / name).mkdir(parents=True)
        (candidate / name / "sentinel.txt").write_text(name)

    removed = operators.remove_shadow_numpy(candidate)

    assert set(removed) == {"numpy", "numpy.libs", "numpy-2.5.1.dist-info"}
    assert not (candidate / "numpy").exists()
    assert not (candidate / "numpy.libs").exists()
    assert not (candidate / "numpy-2.5.1.dist-info").exists()
    assert (candidate / "cupy" / "sentinel.txt").read_text() == "cupy"


def test_cuda_runtime_cleanup_preserves_active_candidate(monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    root = tmp_path / "stflip_cuda_runtime"
    active = root / "cuda13-active"
    obsolete = root / "cuda12-obsolete"
    installing = root / "cuda13-installing"
    active.mkdir(parents=True)
    obsolete.mkdir()
    installing.mkdir()
    (active / "keep.txt").write_text("active")
    (obsolete / "large-wheel.bin").write_bytes(b"old")
    (installing / operators._INSTALLING_RUNTIME_FILE).write_text("pid=123")

    removed = operators.cleanup_inactive_cuda_runtimes(active)

    assert removed == [obsolete.name]
    assert (active / "keep.txt").read_text() == "active"
    assert not obsolete.exists()
    assert installing.exists()


def test_memory_estimate_allows_128_but_rejects_predictable_512_oom(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    gib = 1024 ** 3

    normal = operators.estimate_bake_memory((128, 128, 128), 8)
    huge = operators.estimate_bake_memory((512, 512, 512), 8)

    assert huge["working_set_bytes"] > normal["working_set_bytes"] * 50
    assert operators.memory_guard_reason(
        normal, backend_name="cpu", ram_available=16 * gib,
        vram_available=None,
    ) is None
    reason = operators.memory_guard_reason(
        huge, backend_name="cpu", ram_available=64 * gib,
        vram_available=None,
    )
    assert reason is not None
    assert "RAM" in reason
    assert "512" in reason


def test_memory_guard_uses_vram_for_cuda_without_blocking_normal_gpu(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    gib = 1024 ** 3
    estimate = operators.estimate_bake_memory((128, 128, 128), 8)

    assert operators.memory_guard_reason(
        estimate, backend_name="cuda", ram_available=32 * gib,
        vram_available=16 * gib,
    ) is None
    reason = operators.memory_guard_reason(
        estimate, backend_name="cuda", ram_available=32 * gib,
        vram_available=2 * gib,
    )
    assert reason is not None
    assert "VRAM" in reason


def test_paper_surface_memory_estimate_includes_dense_fields_and_splat_scratch(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    voxels = 16_777_216
    chunk = core_surface.DEFAULT_PARTICLE_CHUNK_SIZE

    estimate = operators.estimate_paper_surface_memory(voxels)
    with_particles = operators.estimate_paper_surface_memory(
        voxels, particle_count=1_000_000)

    assert estimate["stencil_candidate_count"] == 125
    assert estimate["splat_scratch_bytes"] == chunk * 125 * 128
    assert estimate["device_working_set_bytes"] == (
        voxels * np.dtype(np.float32).itemsize * 40
        + estimate["splat_scratch_bytes"])
    assert estimate["cpu_working_set_bytes"] == (
        estimate["device_working_set_bytes"])
    assert estimate["cuda_host_working_set_bytes"] == (
        voxels * np.dtype(np.float32).itemsize * 8)
    assert (with_particles["device_working_set_bytes"]
            - estimate["device_working_set_bytes"]) == 12_000_000
    assert (with_particles["cpu_working_set_bytes"]
            - estimate["cpu_working_set_bytes"]) == 24_000_000
    assert (with_particles["cuda_host_working_set_bytes"]
            - estimate["cuda_host_working_set_bytes"]) == 24_000_000


def test_paper_surface_backend_decision_uses_one_backend_for_complete_cache(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    gib = 1024 ** 3
    estimate = operators.estimate_paper_surface_memory(16_777_216)

    enough = operators.paper_surface_backend_decision(
        "cuda",
        estimate,
        ram_available=32 * gib,
        vram_available=8 * gib,
        reserved_ram_bytes=1 * gib,
        reserved_vram_bytes=2 * gib,
    )
    fallback = operators.paper_surface_backend_decision(
        "cuda",
        estimate,
        ram_available=16 * gib,
        vram_available=4 * gib,
        reserved_ram_bytes=1 * gib,
        reserved_vram_bytes=2 * gib,
    )
    impossible = operators.paper_surface_backend_decision(
        "cuda",
        estimate,
        ram_available=1 * gib,
        vram_available=None,
    )

    assert enough == {"backend": "cuda", "warning": "", "error": ""}
    assert fallback["backend"] == "cpu"
    assert "complete Paper MCF cache" in fallback["warning"]
    assert fallback["error"] == ""
    assert impossible["backend"] is None
    assert "CPU fallback is also unsafe" in impossible["error"]


def test_normalize_cuda_diagnostics_accepts_mapping_and_object(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)

    mapping = operators.normalize_cuda_diagnostics({
        "available": True,
        "device_name": "RTX Test",
        "free_bytes": 12,
        "total_bytes": 20,
    })
    assert mapping == {
        "available": True,
        "device": "RTX Test",
        "free_bytes": 12,
        "total_bytes": 20,
        "error": "",
    }

    diagnostic = types.SimpleNamespace(
        ok=False,
        device="Unavailable GPU",
        memory_free=3,
        memory_total=5,
        message="kernel preflight failed",
    )
    normalized = operators.normalize_cuda_diagnostics(diagnostic)
    assert normalized["available"] is False
    assert normalized["device"] == "Unavailable GPU"
    assert normalized["free_bytes"] == 3
    assert normalized["total_bytes"] == 5
    assert normalized["error"] == "kernel preflight failed"


def test_paper_surface_config_pins_paper_and_discretization_constants(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)

    config = operators.paper_surface_config(0.125, 30, 0.0, "cuda")

    assert config["algorithm"] == "appendix_b_feature_preserving_mcf_v2"
    assert config["rasterizer"] == "linear_subvoxel_union_v1"
    assert config["schema"] == core_cache.SURFACE_CONFIG_SCHEMA
    assert config["version"] == core_cache.SURFACE_CONFIG_VERSION
    assert config["boundary"] == {
        "gaussian": "constant_zero_extension_v1",
        "mcf": "edge_neumann_v1",
    }
    assert config["particle_radius_dx"] == core_surface.SPHERE_RADIUS_DX
    assert config["reconstruction_voxel_dx"] == core_surface.VOXEL_SIZE_DX
    assert config["sphere_ramp_width_voxels"] == (
        core_surface.SPHERE_RAMP_WIDTH_VOXELS)
    assert config["gaussian_sigma_dx"] == core_surface.GAUSSIAN_SIGMA_DX
    assert config["gaussian_truncate_sigma"] == core_surface.GAUSSIAN_TRUNCATE
    assert config["feature_theta"] == core_surface.FEATURE_THRESHOLD
    assert config["feature_zeta"] == core_surface.FEATURE_SLOPE
    assert config["feature_epsilon"] == core_surface.FEATURE_EPSILON
    assert config["gradient_epsilon"] == core_surface.NORMAL_EPSILON
    assert config["isovalue"] == core_surface.SURFACE_ISOVALUE
    assert config["mcf_iterations"] == 30
    assert config["mesh_adaptivity"] == 0.0
    assert config["backend"] == "cuda"
    assert config["particle_coordinate_frame"] == "solver_local_float32"
    assert config["mesh_coordinate_frame"] == (
        "domain_local_object_translation_v1")
    assert config["blender_object_translation"] == (
        "domain_world_origin_float64")


def test_large_world_paper_reconstruction_keeps_subcell_geometry_local(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    local = np.array(
        ((0.125, 0.0, 0.0), (0.375, 0.0, 0.0)), dtype=np.float32)
    origin = np.array((100_000_000.0, -100_000_000.0, 0.0))
    collapsed_world = (local.astype(np.float64) + origin).astype(np.float32)
    assert collapsed_world[0, 0] == collapsed_world[1, 0]
    observed = {}

    def reconstruct(positions, _dx, **_kwargs):
        observed["positions"] = np.asarray(positions).copy()
        return types.SimpleNamespace(
            density=np.ones((2, 2, 2), dtype=np.float32),
            origin=(0.0, 0.0, 0.0),
            voxel_size=0.5,
            diagnostics={"particle_count": 2},
        )

    monkeypatch.setattr(operators.surface_core, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(
        operators.mesher,
        "density_field_to_polygons",
        lambda *_args, **_kwargs: (
            local.copy(),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 4), dtype=np.int32),
        ),
        raising=False,
    )
    backend = types.SimpleNamespace(
        name="cpu",
        xp=np,
        synchronize=lambda: None,
        to_numpy=np.asarray,
    )
    config = operators.paper_surface_config(1.0, 30, 0.0, "cpu")

    vertices, _triangles, _quads, _diagnostics = (
        operators._reconstruct_paper_surface(
            local, 1.0, config, 1000, backend))
    surface_obj = types.SimpleNamespace(location=None)
    operators._set_paper_surface_origin(surface_obj, origin)

    np.testing.assert_array_equal(observed["positions"], local)
    assert vertices[1, 0] - vertices[0, 0] == pytest.approx(0.25)
    assert surface_obj.location == pytest.approx(tuple(origin))
    world_x = np.asarray(surface_obj.location[0], dtype=np.float64) + (
        vertices[:, 0].astype(np.float64))
    assert world_x[1] - world_x[0] == pytest.approx(0.25)


def test_large_origin_frames_preserve_local_positions_for_surface_rebuild(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    local = np.array(
        ((0.125, 0.0, 0.0), (0.375, 0.0, 0.0)), dtype=np.float32)
    origin = np.array((100_000_000.0, 0.0, 0.0))
    world = (local.astype(np.float64) + origin).astype(np.float32)

    attributes = operators._frame_attributes_with_surface_local_positions(
        {"age": np.zeros(2, dtype=np.float32)}, local, origin, 1.0)
    monkeypatch.setattr(
        operators.cache,
        "read_frame_attributes",
        lambda _path, _frame: attributes,
        raising=False,
    )
    recovered, source = operators._cached_surface_local_positions(
        str(tmp_path), 7, world, origin, 1.0)

    assert operators._PAPER_LOCAL_POSITION_ATTRIBUTE in attributes
    np.testing.assert_array_equal(recovered, local)
    assert source == "synchronized_solver_local_cache_attribute"


def test_live_paper_surface_forces_exact_local_source_at_near_origin(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    local = np.array(
        ((0.125, 0.25, 0.5), (0.375, 0.75, 0.875)), dtype=np.float32)
    origin = np.array((0.1, -0.2, 0.3), dtype=np.float64)

    attributes = operators._frame_attributes_with_surface_local_positions(
        {}, local, origin, 1.0, force=True)

    np.testing.assert_array_equal(
        attributes["stflip_surface_local_position"], local)


def test_large_origin_rebuild_refuses_lossy_legacy_world_only_cache(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    origin = np.array((100_000_000.0, 0.0, 0.0))
    world = np.full((2, 3), origin, dtype=np.float32)
    monkeypatch.setattr(
        operators.cache,
        "read_frame_attributes",
        lambda _path, _frame: {},
        raising=False,
    )

    with pytest.raises(ValueError, match="rebake.*current add-on"):
        operators._cached_surface_local_positions(
            str(tmp_path), 7, world, origin, 1.0)


def test_final_paper_fidelity_preset_is_explicit_and_non_destructive(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    settings = types.SimpleNamespace(
        experiment_profile="DAM_BREAK_ST_CFL_16",
        bake_status="existing cache remains",
    )
    expected = {
        "experiment_profile": "CUSTOM",
        "cfl_target": 8.0,
        "particles_per_cell": 8,
        "st_enabled": True,
        "jitter_strength": 1.0,
        "adaptive_gamma": True,
        "eta_phi": 0.5,
        "transfer": "flip",
        "flip_blend": 0.98,
        "local_cfl": 1.0,
        "pcg_tolerance": 1e-4,
        "create_surface": True,
        "surface_method": "PAPER_MCF",
        "paper_mcf_iterations": 30,
        "paper_mesh_adaptivity": 0.0,
    }

    operators.apply_paper_fidelity_settings(settings)

    for name, value in expected.items():
        assert getattr(settings, name) == value
    assert settings.experiment_profile == "CUSTOM"


def test_paper_surface_config_reads_live_surface_core_constants(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    monkeypatch.setattr(operators.surface_core, "FEATURE_THRESHOLD", 3.25)

    config = operators.paper_surface_config(0.125, 30, 0.0, "cpu")

    assert config["feature_theta"] == 3.25


def test_paper_surface_fingerprint_is_config_specific_not_simulation_state(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    base = operators.paper_surface_config(0.25, 30, 0.0, "cpu")

    first = operators.paper_surface_fingerprint(base)
    second = operators.paper_surface_fingerprint(dict(base))
    changed = dict(base, mcf_iterations=31)

    assert first == second
    assert len(first) == 64
    assert first != operators.paper_surface_fingerprint(changed)


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 30, 0.0, "cpu"),
        (0.1, 0, 0.0, "cpu"),
        (0.1, -1, 0.0, "cpu"),
        (0.1, 30, 1.1, "cpu"),
        (0.1, 30, 0.0, "metal"),
    ],
)
def test_paper_surface_config_rejects_invalid_values(
        monkeypatch, tmp_path, args):
    operators = _load_operators(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        operators.paper_surface_config(*args)


def test_resume_extends_only_an_exact_current_paper_configuration(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    config = operators.paper_surface_config(0.25, 30, 0.0, "cpu")
    fingerprint = operators.paper_surface_fingerprint(config)
    metadata = operators.paper_surface_metadata(
        config, fingerprint, 262_144, latest_frame=2, state="COMPLETE")

    selected = operators.matching_resume_paper_surface_config(
        metadata,
        requested=True,
        dx=0.25,
        iterations=30,
        adaptivity=0.0,
    )
    changed = operators.matching_resume_paper_surface_config(
        metadata,
        requested=True,
        dx=0.25,
        iterations=31,
        adaptivity=0.0,
    )
    disabled = operators.matching_resume_paper_surface_config(
        metadata,
        requested=False,
        dx=0.25,
        iterations=30,
        adaptivity=0.0,
    )

    assert selected == (config, fingerprint, "")
    assert changed[0:2] == (None, None)
    assert "will not be extended" in changed[2]
    assert disabled == (None, None, "")


def test_resume_ignores_failed_or_inconsistent_paper_cache(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    config = operators.paper_surface_config(0.25, 30, 0.0, "cpu")
    fingerprint = operators.paper_surface_fingerprint(config)
    failed = operators.paper_surface_metadata(
        config, fingerprint, 262_144, latest_frame=1, state="FAILED")
    inconsistent = operators.paper_surface_metadata(
        config, "f" * 64, 262_144, latest_frame=1, state="COMPLETE")

    for metadata in (failed, inconsistent, None):
        selected = operators.matching_resume_paper_surface_config(
            metadata,
            requested=True,
            dx=0.25,
            iterations=30,
            adaptivity=0.0,
        )
        assert selected[0:2] == (None, None)
        assert "Rebuild Paper Surface Cache" in selected[2]


def test_runtime_paper_failure_does_not_abort_particle_frame_commit(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    reports = []
    writes = []
    monkeypatch.setattr(
        operators.cache,
        "write_frame",
        lambda *args, **kwargs: writes.append("frame"),
        raising=False,
    )
    monkeypatch.setattr(
        operators.cache,
        "write_checkpoint",
        lambda *args, **kwargs: writes.append("checkpoint"),
        raising=False,
    )
    monkeypatch.setattr(
        operators.cache,
        "write_meta",
        lambda *args, **kwargs: writes.append("metadata"),
        raising=False,
    )
    monkeypatch.setattr(
        operators,
        "_write_paper_surface_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MemoryError("surface allocation failed")),
    )

    stats = types.SimpleNamespace(
        n_particles=1, steps=1, particles_removed=0)
    solver = types.SimpleNamespace(
        p=types.SimpleNamespace(dx=0.25),
        be=types.SimpleNamespace(name="cuda"),
        step_frame=lambda: stats,
        get_render_particles=lambda: (
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        ),
        get_render_particles_ex=lambda: (
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            {},
        ),
        checkpoint_state=lambda: {},
    )
    settings = types.SimpleNamespace(
        bake_state="RUNNING", bake_status="", bake_error="", bake_progress=0.0)
    scene = types.SimpleNamespace(
        stflip=settings,
        frame_set=lambda frame: writes.append(("display", frame)),
    )
    surface_meta = {"state": "RUNNING"}
    operators._BAKE.update({
        "solver": solver,
        "scene": scene,
        "cache_dir": str(tmp_path),
        "frame": 1,
        "end": 2,
        "origin": np.zeros(3, dtype=np.float32),
        "meta": {
            "frame_start": 1,
            "frame_end_baked": 1,
            "checkpoint": {"fingerprint": "a" * 64},
            "surface_reconstruction": surface_meta,
        },
        "paper_surface_config": {"backend": "cuda"},
        "paper_surface_fingerprint": "b" * 64,
        "paper_surface_backend": solver.be,
        "paper_surface_max_voxels": 262_144,
        "backend_label": "CUDA",
        "collect_metrics": False,
    })
    operator = operators.STFLIP_OT_bake()
    operator.report = lambda levels, message: reports.append((levels, message))

    assert operator._bake_next_frame() is False

    assert writes[:3] == ["frame", "checkpoint", "metadata"]
    assert writes[-1] == ("display", 2)
    assert operators._BAKE["meta"]["frame_end_baked"] == 2
    assert operators._BAKE["meta"]["checkpoint"]["latest_frame"] == 2
    assert surface_meta["state"] == "FAILED"
    assert surface_meta["error"] == "surface allocation failed"
    assert operators._BAKE["paper_surface_config"] is None
    assert reports and reports[-1][0] == {"WARNING"}
    operators._BAKE.clear()


def test_reconstruction_honors_configured_backend_instead_of_caller_backend(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    selected = types.SimpleNamespace(
        name="cpu",
        xp="cpu-array-module",
        synchronize=lambda: None,
    )
    passed = types.SimpleNamespace(name="cuda", xp="cuda-array-module")
    observed = {}

    def reconstruct(positions, dx, **kwargs):
        observed["array_module"] = kwargs["array_module"]
        return types.SimpleNamespace(
            density=np.empty((0, 0, 0), dtype=np.float32),
            origin=np.zeros(3, dtype=np.float32),
            voxel_size=0.5,
            diagnostics={"particle_count": 0},
        )

    monkeypatch.setattr(operators, "get_backend", lambda name: selected)
    monkeypatch.setattr(operators.surface_core, "reconstruct_surface", reconstruct)
    config = operators.paper_surface_config(0.25, 30, 0.0, "cpu")

    operators._reconstruct_paper_surface(
        np.empty((0, 3), dtype=np.float32),
        0.25,
        config,
        1000,
        passed,
    )

    assert observed["array_module"] == "cpu-array-module"


def test_surface_rebuild_preflight_pins_cpu_fallback_in_config_and_state(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    gib = 1024 ** 3
    reports = []
    metadata = {
        "dx": 0.25,
        "origin": [0.0, 0.0, 0.0],
        "frame_start": 1,
        "frame_end_baked": 2,
        "settings": {
            "grid_dims": [48, 48, 48],
            "particles_per_cell": 8,
        },
    }
    settings = types.SimpleNamespace(
        create_surface=True,
        surface_method="PAPER_MCF",
        backend="cuda",
        paper_mcf_iterations=30,
        paper_mesh_adaptivity=0.0,
        paper_max_reconstruction_voxels=16_777_216,
        bake_status="",
    )
    scene = types.SimpleNamespace(stflip=settings)
    context = types.SimpleNamespace(scene=scene)
    monkeypatch.setattr(
        operators.cache, "read_meta", lambda path: metadata, raising=False)
    monkeypatch.setattr(
        operators.cache, "committed_frames", lambda path, meta: [1, 2],
        raising=False,
    )
    monkeypatch.setattr(
        operators.handlers, "scene_cache_ownership", lambda scene, meta: "owned",
        raising=False,
    )
    monkeypatch.setattr(operators, "resolve_cache_dir", lambda scene: str(tmp_path))
    monkeypatch.setattr(
        operators,
        "current_cuda_diagnostics",
        lambda: {"available": True, "free_bytes": 3 * gib},
    )
    monkeypatch.setattr(
        operators, "_system_available_memory_bytes", lambda: 16 * gib)
    monkeypatch.setattr(
        operators,
        "get_backend",
        lambda name: types.SimpleNamespace(name=name),
    )
    operator = operators.STFLIP_OT_rebuild_paper_surfaces()
    operator.report = lambda level, message: reports.append((level, message))

    assert operator._setup(context) is True

    assert operators._SURFACE_BAKE["backend"].name == "cpu"
    assert operators._SURFACE_BAKE["config"]["backend"] == "cpu"
    assert operators._SURFACE_BAKE["fingerprint"] == (
        operators.paper_surface_fingerprint(
            operators._SURFACE_BAKE["config"]))
    assert any(
        "complete Paper MCF cache" in message
        for _level, message in reports
    )


def test_surface_rebuild_refuses_noncontiguous_committed_particles(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    metadata = {
        "dx": 0.25,
        "origin": [0.0, 0.0, 0.0],
        "frame_start": 1,
        "frame_end_baked": 3,
    }
    settings = types.SimpleNamespace(
        create_surface=True,
        surface_method="PAPER_MCF",
        backend="cpu",
        paper_mcf_iterations=30,
        paper_mesh_adaptivity=0.0,
        paper_max_reconstruction_voxels=262_144,
        bake_status="",
    )
    scene = types.SimpleNamespace(stflip=settings)
    context = types.SimpleNamespace(scene=scene)
    reports = []
    monkeypatch.setattr(
        operators.cache, "read_meta", lambda path: metadata, raising=False)
    monkeypatch.setattr(
        operators.cache,
        "committed_frames",
        lambda path, meta: [1, 3],
        raising=False,
    )
    monkeypatch.setattr(
        operators.handlers,
        "scene_cache_ownership",
        lambda scene, meta: "owned",
        raising=False,
    )
    monkeypatch.setattr(operators, "resolve_cache_dir", lambda scene: str(tmp_path))
    operator = operators.STFLIP_OT_rebuild_paper_surfaces()
    operator.report = lambda levels, message: reports.append((levels, message))

    assert operator._setup(context) is False
    assert "missing or corrupt" in reports[-1][1]
    assert operators._SURFACE_BAKE == {}


def test_surface_visibility_never_reenables_gn_for_plain_paper_mesh(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    modifier = types.SimpleNamespace(show_viewport=True, show_render=True)

    class Surface(dict):
        hide_render = True
        hide_viewport = True
        modifiers = {"STFLIP Surface": modifier}

        def hide_set(self, hidden):
            self.hidden = hidden

    surface = Surface(stflip_surface_method="PAPER_MCF")

    operators._set_surface_enabled(surface, True)

    assert surface.hide_render is False
    assert surface.hide_viewport is False
    assert surface.hidden is False
    assert modifier.show_viewport is False
    assert modifier.show_render is False

    surface["stflip_surface_method"] = "FAST_PREVIEW"
    operators._set_surface_enabled(surface, True)
    assert modifier.show_viewport is True
    assert modifier.show_render is True

    operators._set_surface_enabled(surface, False)
    assert modifier.show_viewport is False
    assert modifier.show_render is False


@pytest.mark.parametrize(
    ("create_surface", "surface_method"),
    [(False, "PAPER_MCF"), (True, "FAST_PREVIEW")],
)
def test_surface_rebuild_completion_does_not_apply_mesh_after_ui_mode_change(
        monkeypatch, tmp_path, create_surface, surface_method):
    operators = _load_operators(monkeypatch, tmp_path)
    operators.cache.SURFACE_SCHEMA = "stflip-paper-surface"
    operators.cache.SURFACE_VERSION = 1
    metadata_writes = []
    operators.cache.write_meta = (
        lambda path, value: metadata_writes.append((path, value.copy())))
    operators.cache.read_frame = lambda *args: pytest.fail(
        "viewport particle frame was read after surface display was disabled")
    operators.cache.read_surface = lambda *args, **kwargs: pytest.fail(
        "viewport paper mesh was read after surface mode changed")
    operators.mesher.ensure_paper_surface_object = (
        lambda *args, **kwargs: pytest.fail(
            "paper surface object was applied after surface mode changed"))

    settings = types.SimpleNamespace(
        create_surface=create_surface,
        surface_method=surface_method,
        surface_object=object(),
        bake_status="",
    )
    scene = types.SimpleNamespace(
        stflip=settings,
        frame_current=1,
    )
    context = types.SimpleNamespace(
        scene=scene,
        window_manager=types.SimpleNamespace(),
    )
    operators._SURFACE_BAKE.update({
        "running": True,
        "scene": scene,
        "frames": [1],
        "config": {"schema": "test"},
        "fingerprint": "a" * 64,
        "max_voxels": 16_777_216,
        "meta": {"frame_start": 1, "frame_end_baked": 1},
        "cache_dir": str(tmp_path),
        "backend": types.SimpleNamespace(name="cpu"),
        "world_origin": np.zeros(3, dtype=np.float64),
        "latest_diagnostics": None,
    })
    operator = operators.STFLIP_OT_rebuild_paper_surfaces()

    result = operator._finish(context, "COMPLETE")

    assert result == {"FINISHED"}
    assert len(metadata_writes) == 1
    assert metadata_writes[0][1]["surface_reconstruction"]["state"] == (
        "COMPLETE")
    assert settings.bake_status == "Paper surface cache complete (1 frames; cpu)"
    assert operators._SURFACE_BAKE == {}


def _active_surface_finish_state(operators, tmp_path):
    config = operators.paper_surface_config(0.25, 30, 0.0, "cpu")
    fingerprint = operators.paper_surface_fingerprint(config)
    settings = types.SimpleNamespace(
        create_surface=True,
        surface_method="PAPER_MCF",
        surface_object=object(),
        bake_status="",
    )
    scene = types.SimpleNamespace(stflip=settings, frame_current=1)
    context = types.SimpleNamespace(
        scene=scene,
        window_manager=types.SimpleNamespace(),
    )
    operators._SURFACE_BAKE.update({
        "running": True,
        "scene": scene,
        "frames": [1],
        "config": config,
        "fingerprint": fingerprint,
        "max_voxels": 262_144,
        "meta": {"frame_start": 1, "frame_end_baked": 1},
        "cache_dir": str(tmp_path),
        "backend": types.SimpleNamespace(name="cpu"),
        "world_origin": np.zeros(3, dtype=np.float64),
        "latest_diagnostics": {"voxel_count": 12},
    })
    operator = operators.STFLIP_OT_rebuild_paper_surfaces()
    reports = []
    operator.report = lambda levels, message: reports.append((levels, message))
    return operator, context, settings, reports


def test_surface_activation_survives_viewport_refresh_failure(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    operator, context, settings, reports = _active_surface_finish_state(
        operators, tmp_path)
    metadata_writes = []
    monkeypatch.setattr(
        operators.cache,
        "write_meta",
        lambda path, value: metadata_writes.append(value.copy()),
        raising=False,
    )
    monkeypatch.setattr(
        operators.cache,
        "read_frame",
        lambda *args: (np.zeros((1, 3), dtype=np.float32), object()),
        raising=False,
    )
    monkeypatch.setattr(
        operators.cache,
        "read_surface",
        lambda *args, **kwargs: (object(), object(), object()),
        raising=False,
    )
    monkeypatch.setattr(
        operators.mesher,
        "ensure_paper_surface_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("viewport unavailable")),
        raising=False,
    )

    result = operator._finish(context, "COMPLETE")

    assert result == {"FINISHED"}
    assert metadata_writes[0]["surface_reconstruction"]["state"] == "COMPLETE"
    assert "Paper surface cache complete" in settings.bake_status
    assert reports[-1][0] == {"WARNING"}
    assert "cache is active" in reports[-1][1]
    assert operators._SURFACE_BAKE == {}


def test_surface_activation_write_failure_is_caught_and_cleans_state(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    operator, context, settings, reports = _active_surface_finish_state(
        operators, tmp_path)
    monkeypatch.setattr(
        operators.cache,
        "write_meta",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
        raising=False,
    )

    result = operator._finish(context, "COMPLETE")

    assert result == {"CANCELLED"}
    assert "activation failed: disk full" in settings.bake_status
    assert reports[-1][0] == {"ERROR"}
    assert operators._SURFACE_BAKE == {}
