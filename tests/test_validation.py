import json

import numpy as np
import pytest
import stflip.validation as validation_module

from stflip.metrics import FRAME_FIELD_ORDER, SCHEMA_VERSION
from stflip.validation import (
    MATCHED_CASES,
    VALIDATION_SCHEMA,
    ValidationConfig,
    _hash_initial_checkpoint,
    _run_case,
    detect_high_cfl,
    run_matched_validation,
    run_multi_seed_validation,
    temporal_quadrature_coverage,
    write_validation_artifact,
)


@pytest.fixture(scope="module")
def artifact():
    return run_matched_validation(ValidationConfig(
        resolution=6,
        frames=2,
        particles_per_cell=1,
        seed=19,
        high_cfl_threshold=0.05,
    ))


def test_case_matrix_varies_only_temporal_sampling_and_target_cfl():
    assert [case.identifier for case in MATCHED_CASES] == [
        "st_cfl_1",
        "instantaneous_cfl_1",
        "st_cfl_16",
        "instantaneous_cfl_16",
    ]


def test_initial_checkpoint_hash_covers_rng_and_m0_not_only_particle_arrays():
    state = {
        "pos": np.zeros((1, 3), dtype=np.float32),
        "vel": np.zeros((1, 3), dtype=np.float32),
        "dt_resid": np.zeros((1,), dtype=np.float32),
        "time": 0.0,
        "dt_prev": 0.1,
        "rng_state": {"bit_generator": "test", "state": {"state": 1}},
        "outflow_removed_total": 0,
        "volume_outflow_removed_total": 0,
        "pressure_outflow_removed_total": 0,
    }

    baseline = _hash_initial_checkpoint(state, 1.0)
    changed_rng = {**state, "rng_state": {
        "bit_generator": "test", "state": {"state": 2}}}

    assert baseline != _hash_initial_checkpoint(changed_rng, 1.0)
    assert baseline != _hash_initial_checkpoint(state, 2.0)
    assert [
        (case.st_enabled, case.cfl_target) for case in MATCHED_CASES
    ] == [
        (True, 1.0),
        (False, 1.0),
        (True, 16.0),
        (False, 16.0),
    ]


def test_real_solver_matrix_has_same_initial_state_and_strict_metrics(artifact):
    assert artifact["schema"] == VALIDATION_SCHEMA
    assert artifact["acceptance"]["same_initial_particle_state"] is True
    assert artifact["acceptance"]["same_initial_shared_state"] is True
    assert artifact["acceptance"][
        "complete_checkpoints_match_within_cfl"] is True
    checkpoint_hashes = artifact["acceptance"][
        "initial_checkpoint_sha256_by_case"]
    assert checkpoint_hashes == {
        identifier: case["initial_checkpoint_sha256"]
        for identifier, case in artifact["cases"].items()
    }
    assert checkpoint_hashes["st_cfl_1"] == checkpoint_hashes[
        "instantaneous_cfl_1"]
    assert checkpoint_hashes["st_cfl_16"] == checkpoint_hashes[
        "instantaneous_cfl_16"]
    assert checkpoint_hashes["st_cfl_1"] != checkpoint_hashes["st_cfl_16"]
    particle_hashes = {
        case["initial_particle_state_sha256"]
        for case in artifact["cases"].values()
    }
    assert particle_hashes == {
        artifact["acceptance"]["initial_particle_state_sha256"]}

    shared_parameters = []
    for identifier, case in artifact["cases"].items():
        assert len(case["frames"]) == 2
        assert all(len(frame["output_sha256"]) == 64
                   for frame in case["frames"])
        assert all(len(frame["deposition_sha256"]) == 64
                   for frame in case["frames"])
        assert [frame["metrics"]["frame"] for frame in case["frames"]] == [1, 2]
        for frame in case["frames"]:
            assert tuple(frame["metrics"]) == FRAME_FIELD_ORDER
            assert frame["metrics"]["schema_version"] == SCHEMA_VERSION
            assert frame["metrics"]["particle_count"] > 0
        parameters = dict(case["parameters"])
        assert parameters.pop("st_enabled") == identifier.startswith("st_")
        assert parameters.pop("cfl_target") == (
            16.0 if identifier.endswith("16") else 1.0)
        shared_parameters.append(parameters)
    assert all(value == shared_parameters[0] for value in shared_parameters)


def test_temporal_ablation_and_st_residual_checks_are_explicit(artifact):
    checks = artifact["acceptance"]["temporal_residual_checks"]
    assert artifact["acceptance"]["core_checks_passed"] is True
    assert checks["st_cfl_1"]["passed"] is True
    assert checks["st_cfl_16"]["passed"] is True
    assert checks["instantaneous_cfl_1"] == {
        "passed": True,
        "dt_resid_abs_max_s": 0.0,
    }
    assert checks["instantaneous_cfl_16"] == {
        "passed": True,
        "dt_resid_abs_max_s": 0.0,
    }
    assert checks["st_cfl_16"]["dt_resid_abs_max_s"] > 0.0
    assert artifact["acceptance"]["eq7_8_temporal_quadrature_passed"] is True
    for identifier, case in artifact["cases"].items():
        modes = {
            frame["temporal_state"]["quadrature"]["deposition_weight_mode"]
            for frame in case["frames"]
        }
        assert modes == ({"one_sided_temporal_kernel_exact_norm"}
                         if identifier.startswith("st_")
                         else {"instantaneous_unit_weight"})


def test_eq7_8_temporal_quadrature_covers_slab_and_converges_deterministically():
    coarse = temporal_quadrature_coverage(128, 16)
    fine = temporal_quadrature_coverage(256, 16)

    assert fine["passed"] is True
    assert fine["occupied_bins"] == fine["bin_count"] == 16
    assert fine["resolution_doubling_reduces_error"] is True
    assert fine["normalization_error"] < coarse["normalization_error"]
    assert fine["weighted_mean_tau"] > 0.0
    assert fine["instantaneous_control"]["occupied_bins"] == 1


def test_quality_comparison_is_separate_from_backend_timing(artifact):
    quality = artifact["quality"]
    assert set(quality) == {
        "interpretation",
        "st_high_vs_st_low",
        "instantaneous_high_vs_instantaneous_low",
        "st_high_vs_instantaneous_high",
        "st_high_vs_instantaneous_low_diagnostic",
        "primary",
        "secondary_evidence",
    }
    assert quality["primary"]["measure"] == (
        "phase_density_rmse_mean")
    assert "particle" not in quality["primary"]["basis"].lower()
    for comparison_name in (
        "st_high_vs_st_low", "instantaneous_high_vs_instantaneous_low"
    ):
        comparison = quality[comparison_name]
        assert comparison["field_metrics"][
            "coherence_primary_measure"] == "phase_density_rmse_mean"
        assert "Diagnostic only" in comparison[
            "trajectory_diagnostics"]["interpretation"]
    raw_mass = quality["secondary_evidence"][
        "normalized_deposited_mass_rmse_mean"]
    assert raw_mass["acceptance_gate"] is False
    assert "Monte Carlo" in raw_mass["interpretation"]
    assert quality["secondary_evidence"]["phase_threshold_iou_mean"][
        "required_for_coherence_gate"] is True
    assert "timing" in artifact
    assert not any("wall" in key for key in _nested_keys(quality))
    assert set(artifact["timing"]) == set(artifact["cases"])
    for timing in artifact["timing"].values():
        assert timing["step_wall_s_total"] >= 0.0
        assert timing["render_resynchronization_wall_s_total"] >= 0.0


def _nested_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def test_high_cfl_detection_uses_observed_metrics_not_target_label(artifact):
    maxima = artifact["acceptance"]["high_cfl"]["actual_max_by_case"]
    below_observed = min(maxima.values()) * 0.5
    above_observed = max(maxima.values()) + 1.0

    reached = detect_high_cfl(artifact["cases"], below_observed)
    missed = detect_high_cfl(artifact["cases"], above_observed)

    assert reached["reached"] is True
    assert missed["reached"] is False
    assert missed["reached_by_case"] == {
        "st_cfl_16": False,
        "instantaneous_cfl_16": False,
    }


def test_output_hashes_and_quality_are_deterministic_while_timing_is_not_used():
    config = ValidationConfig(
        resolution=4, frames=1, particles_per_cell=1, seed=23,
        high_cfl_threshold=0.01,
    )
    first = run_matched_validation(config)
    second = run_matched_validation(config)

    assert first["acceptance"]["initial_checkpoint_sha256_by_case"] == second[
        "acceptance"]["initial_checkpoint_sha256_by_case"]
    for identifier in first["cases"]:
        assert [frame["output_sha256"] for frame in first["cases"][
            identifier]["frames"]] == [
                frame["output_sha256"] for frame in second["cases"][
                    identifier]["frames"]]
        assert [frame["deposition_sha256"] for frame in first["cases"][
            identifier]["frames"]] == [
                frame["deposition_sha256"] for frame in second["cases"][
                    identifier]["frames"]]
    assert first["quality"] == second["quality"]


def test_case_output_is_invariant_to_prior_case_execution_order():
    config = ValidationConfig(
        resolution=4, frames=1, particles_per_cell=1, seed=29,
        high_cfl_threshold=0.01,
    )
    target = next(
        case for case in MATCHED_CASES if case.identifier == "st_cfl_16")
    first = _run_case(config, target)
    for case in reversed(MATCHED_CASES):
        _run_case(config, case)
    second = _run_case(config, target)

    assert [frame.output_sha256 for frame in first.frames] == [
        frame.output_sha256 for frame in second.frames]
    assert [frame.normalized_deposited_mass.tobytes()
            for frame in first.frames] == [
                frame.normalized_deposited_mass.tobytes()
                for frame in second.frames]


def test_validation_rejects_sources_changed_during_execution(monkeypatch):
    calls = 0

    def changing_hashes():
        nonlocal calls
        calls += 1
        return {"solver.py": "a" * 64 if calls == 1 else "b" * 64}

    monkeypatch.setattr(
        validation_module, "_source_file_hashes", changing_hashes)

    with pytest.raises(RuntimeError, match="sources changed"):
        run_matched_validation(ValidationConfig(
            resolution=4, frames=1, particles_per_cell=1, seed=31,
            high_cfl_threshold=0.01,
        ))


def test_cli_validation_ready_gate_includes_core_checks(monkeypatch, tmp_path):
    from tools import run_validation as cli

    aggregate = {
        "high_cfl_reached_all": True,
        "core_checks_passed_all": False,
        "primary_measure": "phase_density_rmse_mean",
        "st_wins": 3,
        "phase_threshold_iou_st_wins": 3,
        "seed_count": 3,
        "mean_st_error_is_lower": True,
        "mean_phase_threshold_iou_is_higher": True,
        "raw_mass_diagnostic": {"acceptance_gate": False},
        "internal_coherence_improved": True,
        "validation_ready": False,
    }
    artifact = {
        "seeds": [0, 1, 2],
        "aggregate": aggregate,
    }
    monkeypatch.setattr(
        cli, "run_multi_seed_validation",
        lambda config, seeds: artifact,
    )
    monkeypatch.setattr(
        cli, "write_validation_artifact",
        lambda path, value: str(path),
    )

    result = cli.main([
        "--seeds", "0", "1", "2",
        "--output", str(tmp_path / "validation.json"),
        "--require-validation-ready",
    ])

    assert result == 4


def test_artifact_writer_round_trips_strict_json_atomically(artifact, tmp_path):
    destination = tmp_path / "nested" / "validation.json"

    result = write_validation_artifact(destination, artifact)

    assert result == str(destination.resolve())
    assert json.loads(destination.read_text("utf-8")) == artifact
    assert not list(destination.parent.glob(".stflip-validation-*"))


def test_multi_seed_artifact_aggregates_without_hiding_individual_runs():
    result = run_multi_seed_validation(
        ValidationConfig(
            resolution=4, frames=1, particles_per_cell=1,
            high_cfl_threshold=0.01,
        ),
        seeds=(3, 7),
    )

    assert result["schema"] == "stflip.multi_seed_matched_validation"
    assert result["seeds"] == [3, 7]
    assert [run["scenario"]["seed"] for run in result["runs"]] == [3, 7]
    assert result["aggregate"]["seed_count"] == 2
    assert result["aggregate"]["strict_majority_required"] == 2
    assert result["aggregate"]["st_wins"] == sum(
        run["quality"]["primary"]["st_has_lower_error"]
        for run in result["runs"]
    )
    assert result["aggregate"]["phase_threshold_iou_st_wins"] == sum(
        run["quality"]["secondary_evidence"][
            "phase_threshold_iou_mean"]["st_has_higher_value"]
        for run in result["runs"]
    )
    assert result["aggregate"]["raw_mass_diagnostic"][
        "acceptance_gate"] is False


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1, 2), ("bad",)])
def test_multi_seed_validation_rejects_ambiguous_seed_sets(seeds):
    with pytest.raises(ValueError, match="seeds"):
        run_multi_seed_validation(ValidationConfig(resolution=4), seeds)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolution": 3},
        {"frames": 0},
        {"particles_per_cell": 0},
        {"seed": -1},
        {"backend": "auto"},
        {"high_cfl_threshold": 0.0},
        {"frame_rate": float("nan")},
    ],
)
def test_validation_configuration_rejects_ambiguous_or_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        ValidationConfig(**kwargs)
