from dataclasses import FrozenInstanceError

import pytest

from stflip.experiments import (
    CUSTOM_PROFILE_ID,
    DAM_BREAK_INSTANTANEOUS_CFL_16,
    DAM_BREAK_PROFILES,
    ENSTROPHY_PROFILES,
    LAMINAR_DAM_BREAK_PROFILES,
    PAPER_DEFAULTS,
    PARTICLE_COUNT_PROFILES,
    PROFILES,
    PROFILE_BY_ID,
    PROFILE_ENUM_ITEMS,
    get_profile,
    profile_provenance,
)


COMMON_SETTINGS = {
    "jitter_strength": 1.0,
    "adaptive_gamma": True,
    "eta_phi": 0.5,
    "seed": 0,
    "collect_metrics": True,
}


def _settings_without(profile, *axes):
    settings = profile.settings()
    for axis in axes:
        settings.pop(axis)
    return settings


def test_profile_identifiers_and_enum_items_are_unique_and_stable():
    identifiers = [profile.identifier for profile in PROFILES]
    assert len(identifiers) == len(set(identifiers))
    assert set(PROFILE_BY_ID) == set(identifiers)

    enum_identifiers = [item[0] for item in PROFILE_ENUM_ITEMS]
    assert PROFILE_ENUM_ITEMS[0][0] == CUSTOM_PROFILE_ID
    assert len(enum_identifiers) == len(set(enum_identifiers))
    assert enum_identifiers[1:] == identifiers
    assert all(len(item) == 3 for item in PROFILE_ENUM_ITEMS)


def test_custom_is_not_a_fixed_profile_and_unknown_ids_fail_clearly():
    assert get_profile(CUSTOM_PROFILE_ID) is None
    with pytest.raises(ValueError, match="unknown experiment profile"):
        get_profile("NOT_A_PROFILE")


def test_profiles_are_immutable_and_parameter_only():
    with pytest.raises(FrozenInstanceError):
        PAPER_DEFAULTS.seed = 42
    assert all(not profile.exact_paper_geometry for profile in PROFILES)
    assert all("not an exact reproduction" in profile.description
               for profile in PROFILES)


def test_every_profile_uses_reproducible_paper_defaults():
    for profile in PROFILES:
        for name, expected in COMMON_SETTINGS.items():
            assert getattr(profile, name) == expected
        assert profile.description
        assert profile.paper_reference

    assert PAPER_DEFAULTS.cfl_target == 8.0
    assert PAPER_DEFAULTS.particles_per_cell == 8
    assert PAPER_DEFAULTS.flip_blend == 0.98
    assert PAPER_DEFAULTS.st_enabled is True
    assert PAPER_DEFAULTS.collect_enstrophy is False
    assert PAPER_DEFAULTS.identifier == "METHOD_DEFAULTS_PREVIEW"
    assert "does not publish its seed" in PAPER_DEFAULTS.description


def test_all_profiles_stay_within_blender_property_ranges():
    for profile in PROFILES:
        assert 0.5 <= profile.cfl_target <= 30.0
        assert 1 <= profile.particles_per_cell <= 64
        assert 0.0 <= profile.jitter_strength <= 1.0
        assert 0.1 <= profile.eta_phi <= 2.0
        assert 0.0 <= profile.flip_blend <= 1.0
        assert 0 <= profile.seed <= 2_147_483_647
        assert not profile.collect_enstrophy or profile.collect_metrics


def test_dam_break_sweep_changes_only_target_cfl():
    assert [profile.cfl_target for profile in DAM_BREAK_PROFILES] == [
        1.0, 2.0, 4.0, 8.0, 16.0]
    snapshots = {
        tuple(sorted(_settings_without(profile, "cfl_target").items()))
        for profile in DAM_BREAK_PROFILES
    }
    assert len(snapshots) == 1
    assert all(profile.family == "dam_break" for profile in DAM_BREAK_PROFILES)


def test_laminar_dam_break_sweep_matches_figure_8_targets():
    assert [profile.cfl_target for profile in LAMINAR_DAM_BREAK_PROFILES] == [
        1.0, 3.0, 5.0, 10.0, 20.0]
    snapshots = {
        tuple(sorted(_settings_without(profile, "cfl_target").items()))
        for profile in LAMINAR_DAM_BREAK_PROFILES
    }
    assert len(snapshots) == 1
    assert all(profile.family == "laminar_dam_break"
               for profile in LAMINAR_DAM_BREAK_PROFILES)


def test_instantaneous_profile_changes_only_temporal_sampling_at_cfl16():
    st_cfl16 = DAM_BREAK_PROFILES[-1]
    ablation = DAM_BREAK_INSTANTANEOUS_CFL_16

    assert ablation.cfl_target == st_cfl16.cfl_target == 16.0
    assert ablation.st_enabled is False
    assert st_cfl16.st_enabled is True
    assert _settings_without(ablation, "st_enabled") == _settings_without(
        st_cfl16, "st_enabled")
    assert "not the paper's standard-FLIP/GFM" in ablation.description


def test_enstrophy_matrix_changes_only_cfl_and_flip_fraction():
    expected = [
        (1.0, 0.99),
        (5.0, 0.99),
        (10.0, 0.99),
        (1.0, 0.95),
        (1.0, 0.90),
    ]
    assert [(profile.cfl_target, profile.flip_blend)
            for profile in ENSTROPHY_PROFILES] == expected
    snapshots = {
        tuple(sorted(_settings_without(
            profile, "cfl_target", "flip_blend").items()))
        for profile in ENSTROPHY_PROFILES
    }
    assert len(snapshots) == 1
    assert all(profile.st_enabled for profile in ENSTROPHY_PROFILES)
    assert all("standard-FLIP/GFM baseline" in profile.description
               for profile in ENSTROPHY_PROFILES)
    assert all(profile.collect_metrics and profile.collect_enstrophy
               for profile in ENSTROPHY_PROFILES)


def test_particle_count_sweep_changes_only_particles_per_cell():
    assert [profile.particles_per_cell
            for profile in PARTICLE_COUNT_PROFILES] == [1, 2, 4, 8, 16, 50]
    assert all(profile.cfl_target == 10.0
               for profile in PARTICLE_COUNT_PROFILES)
    snapshots = {
        tuple(sorted(_settings_without(
            profile, "particles_per_cell").items()))
        for profile in PARTICLE_COUNT_PROFILES
    }
    assert len(snapshots) == 1
    assert all(not profile.collect_enstrophy
               for profile in PARTICLE_COUNT_PROFILES)
    assert "reference" in PARTICLE_COUNT_PROFILES[-1].label.lower()
    assert all("standard-FLIP/CFL 1 branch is unavailable"
               in profile.description for profile in PARTICLE_COUNT_PROFILES)


def test_profile_provenance_rejects_selected_but_modified_controls():
    profile = DAM_BREAK_PROFILES[-1]
    matching = profile.settings()

    exact = profile_provenance(profile.identifier, matching)
    assert exact == {
        "selected": profile.identifier,
        "matched": profile.identifier,
        "matches_selected": True,
        "mismatched_controls": [],
    }

    matching["cfl_target"] = 3.0
    changed = profile_provenance(profile.identifier, matching)
    assert changed["matched"] == CUSTOM_PROFILE_ID
    assert changed["matches_selected"] is False
    assert changed["mismatched_controls"] == ["cfl_target"]

    assert profile_provenance(CUSTOM_PROFILE_ID, {})["matches_selected"] is True
