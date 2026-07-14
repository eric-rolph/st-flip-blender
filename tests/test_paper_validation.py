import json

import pytest

from stflip import __version__ as stflip_version
from stflip.paper_validation import (
    DEFAULT_GLUG_SURFACE_TENSION,
    PAPER_VALIDATION_SCHEMA,
    PaperRunConfig,
    run_glug_benchmark,
    run_kleefsman_benchmark,
    write_paper_artifact,
)


def _quick(**changes):
    values = dict(
        longest_resolution=12,
        frames=1,
        particles_per_cell=1,
        gas_particles_per_cell=1,
        cfl_target=4.0,
        seed=3,
    )
    values.update(changes)
    return PaperRunConfig(**values)


def test_kleefsman_quick_run_records_gauges_without_claiming_reference():
    artifact = run_kleefsman_benchmark(_quick())

    assert artifact["schema"] == PAPER_VALIDATION_SCHEMA
    assert artifact["case"] == "kleefsman"
    assert artifact["reference"] is None
    assert artifact["comparison"] is None
    assert artifact["implementation"]["version"] == stflip_version
    assert len(artifact["implementation"]["python_source_sha256"]) == 64
    runtime = artifact["implementation"]["runtime"]
    assert runtime["python_version"]
    assert runtime["numpy_version"]
    assert runtime["platform"]
    assert artifact["solver"]["backend_runtime"] == {
        "array_library": "numpy",
        "array_library_version": runtime["numpy_version"],
        "cupy_version": None,
        "cuda_device": None,
    }
    assert artifact["solver"]["normalized_params"]["surface_tension"] == 0.0
    outflow = artifact["boundary_conditions"]["registered_outflow"]
    assert outflow["pressure_open_face_counts"]["z_max"] == (
        artifact["grid"]["shape"][0] * artifact["grid"]["shape"][1])
    assert outflow["pressure_open_face_count"] == (
        outflow["pressure_open_face_counts"]["z_max"])
    assert len(artifact["samples"]["time_s"]) == 2
    assert set(artifact["samples"]["water_height_m"]) == {"H2", "H4"}
    assert artifact["frames"][0]["pressure_relative_residuals"]


def test_glug_quick_run_records_two_phase_transfer_metrics():
    artifact = run_glug_benchmark(_quick())

    assert artifact["schema"] == PAPER_VALIDATION_SCHEMA
    assert artifact["case"] == "glug"
    assert len(artifact["samples"]) == 2
    first = artifact["samples"][0]
    assert first["liquid_particles"] > 0
    assert first["gas_particles"] > 0
    # A coarse cell that straddles the connector can seed below the analytic
    # boundary; the evidence-scale grid makes this discretisation error vanish.
    assert first["liquid_upper_fraction"] > 0.9
    assert artifact["solver"]["normalized_params"]["surface_tension"] == (
        DEFAULT_GLUG_SURFACE_TENSION)
    assert artifact["physics_assumptions"]["surface_tension_n_per_m"] == (
        DEFAULT_GLUG_SURFACE_TENSION)
    assert "re-synchronized" in artifact["sampling_method"]["positions"]
    assert "does not publish" in artifact["physics_assumptions"][
        "surface_tension_provenance"]


def test_paper_run_config_rejects_invalid_glug_surface_tension():
    with pytest.raises(ValueError, match="finite and non-negative"):
        _quick(glug_surface_tension=float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        _quick(glug_surface_tension=-0.01)


def test_artifact_write_is_strict_json_and_atomic(tmp_path):
    artifact = run_kleefsman_benchmark(_quick(frames=0))
    output = write_paper_artifact(tmp_path / "nested" / "artifact.json", artifact)

    assert json.loads(output.read_text(encoding="utf-8"))["case"] == "kleefsman"
    assert not list(output.parent.glob("*.tmp"))


def test_kleefsman_reference_csv_and_citation_are_a_strict_pair(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "time_s,H2_m,H4_m\n0,0,0\n0.0416666666667,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires reference_csv"):
        run_kleefsman_benchmark(
            _quick(frames=0), reference_citation="Kleefsman et al.")
    with pytest.raises(ValueError, match="nonblank"):
        run_kleefsman_benchmark(
            _quick(frames=0),
            reference_csv=reference,
            reference_citation="   ",
        )

    artifact = run_kleefsman_benchmark(
        _quick(frames=1),
        reference_csv=reference,
        reference_citation="  Kleefsman et al.  ",
    )
    assert artifact["reference"]["citation"] == "Kleefsman et al."
