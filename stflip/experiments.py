"""Paper-inspired, parameter-only experiment profiles.

These profiles reproduce selected *parameter sweeps* from the ST-FLIP paper.
They deliberately do not claim to reproduce the paper's scene geometry,
reference solvers, reconstruction pipeline, or validation metrics.  The module
is independent of :mod:`bpy` so profile discovery and validation work in
ordinary Python and CI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType


CUSTOM_PROFILE_ID = "CUSTOM"
PAPER_INSPIRED_GEOMETRY_NOTE = (
    "Parameter-only, paper-inspired profile; scene geometry is not an exact "
    "reproduction of the paper."
)


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """A reproducible snapshot of selected solver controls."""

    identifier: str
    label: str
    description: str
    family: str
    paper_reference: str
    cfl_target: float
    particles_per_cell: int
    st_enabled: bool
    jitter_strength: float
    adaptive_gamma: bool
    eta_phi: float
    flip_blend: float
    seed: int
    collect_metrics: bool
    collect_enstrophy: bool
    exact_paper_geometry: bool = False

    def __post_init__(self) -> None:
        if not self.identifier or self.identifier == CUSTOM_PROFILE_ID:
            raise ValueError("a fixed profile needs a non-CUSTOM identifier")
        if not 0.5 <= self.cfl_target <= 30.0:
            raise ValueError("cfl_target must be in Blender's [0.5, 30] range")
        if not 1 <= self.particles_per_cell <= 64:
            raise ValueError(
                "particles_per_cell must be in Blender's [1, 64] range")
        if not 0.0 <= self.jitter_strength <= 1.0:
            raise ValueError("jitter_strength must be in [0, 1]")
        if not 0.1 <= self.eta_phi <= 2.0:
            raise ValueError("eta_phi must be in Blender's [0.1, 2] range")
        if not 0.0 <= self.flip_blend <= 1.0:
            raise ValueError("flip_blend must be in [0, 1]")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("seed must be a non-negative 32-bit integer")
        if self.collect_enstrophy and not self.collect_metrics:
            raise ValueError("enstrophy collection requires metrics collection")
        if self.exact_paper_geometry:
            raise ValueError(
                "built-in profiles are parameter-only, not exact geometry")

    def settings(self) -> dict[str, object]:
        """Return values keyed by the corresponding Blender properties."""
        return {
            "cfl_target": self.cfl_target,
            "particles_per_cell": self.particles_per_cell,
            "st_enabled": self.st_enabled,
            "jitter_strength": self.jitter_strength,
            "adaptive_gamma": self.adaptive_gamma,
            "eta_phi": self.eta_phi,
            "flip_blend": self.flip_blend,
            "seed": self.seed,
            "collect_metrics": self.collect_metrics,
            "collect_enstrophy": self.collect_enstrophy,
        }

    def blender_enum_item(self) -> tuple[str, str, str]:
        """Return an item accepted directly by ``bpy.props.EnumProperty``."""
        return self.identifier, self.label, self.description


_COMMON = {
    "particles_per_cell": 8,
    "st_enabled": True,
    "jitter_strength": 1.0,
    "adaptive_gamma": True,
    "eta_phi": 0.5,
    "flip_blend": 0.98,
    "seed": 0,
    "collect_metrics": True,
    "collect_enstrophy": False,
}


def _profile(
    identifier: str,
    label: str,
    description: str,
    family: str,
    paper_reference: str,
    cfl_target: float,
    **overrides: object,
) -> ExperimentProfile:
    values = {**_COMMON, **overrides}
    return ExperimentProfile(
        identifier=identifier,
        label=label,
        description=f"{description} {PAPER_INSPIRED_GEOMETRY_NOTE}",
        family=family,
        paper_reference=paper_reference,
        cfl_target=cfl_target,
        **values,
    )


PAPER_DEFAULTS = _profile(
    "METHOD_DEFAULTS_PREVIEW",
    "Method defaults preview",
    "Paper-wide method defaults, using CFL 8 and 8 particles/cell as a "
    "practical preview configuration and implementation seed 0 (the paper "
    "does not publish its seed).",
    "paper_defaults",
    "Section 4, Parameters",
    8.0,
)

DAM_BREAK_PROFILES = tuple(
    _profile(
        f"DAM_BREAK_ST_CFL_{cfl}",
        f"Dam break - ST-FLIP CFL {cfl}",
        f"ST-FLIP dam-break parameter profile at target CFL {cfl}.",
        "dam_break",
        "Section 4.3, Figure 2",
        float(cfl),
    )
    for cfl in (1, 2, 4, 8, 16)
)

DAM_BREAK_INSTANTANEOUS_PROFILES = tuple(
    _profile(
        f"DAM_BREAK_INSTANTANEOUS_CFL_{cfl}",
        f"Dam break - instantaneous P2G CFL {cfl}",
        f"Instantaneous-P2G temporal ablation at target CFL {cfl}. This is "
        "not the paper's standard-FLIP/GFM reference solver.",
        "dam_break_ablation",
        "Section 4.3 inspired ablation",
        float(cfl),
        st_enabled=False,
    )
    for cfl in (1, 16)
)
DAM_BREAK_INSTANTANEOUS_CFL_1 = DAM_BREAK_INSTANTANEOUS_PROFILES[0]
DAM_BREAK_INSTANTANEOUS_CFL_16 = DAM_BREAK_INSTANTANEOUS_PROFILES[1]

LAMINAR_DAM_BREAK_PROFILES = tuple(
    _profile(
        f"LAMINAR_DAM_BREAK_ST_CFL_{cfl}",
        f"Laminar dam break - ST-FLIP CFL {cfl}",
        f"Early dam-break parameter profile at target CFL {cfl}.",
        "laminar_dam_break",
        "Section 4.1, Figure 8",
        float(cfl),
    )
    for cfl in (1, 3, 5, 10, 20)
)

_ENSTROPHY_MATRIX = (
    (1, 0.99),
    (5, 0.99),
    (10, 0.99),
    (1, 0.95),
    (1, 0.90),
)
ENSTROPHY_PROFILES = tuple(
    _profile(
        f"ENSTROPHY_CFL_{cfl}_FLIP_{int(round(blend * 100)):02d}",
        f"Enstrophy - CFL {cfl}, FLIP {blend:.2f}",
        f"ST-FLIP enstrophy analog at target CFL {cfl} and FLIP fraction "
        f"{blend:.2f}; the paper's CFL 1 branches use an unavailable true "
        "standard-FLIP/GFM baseline.",
        "enstrophy_st_analog",
        "Section 4.4, Figures 14-15 (ST-FLIP analog)",
        float(cfl),
        flip_blend=blend,
        collect_enstrophy=True,
    )
    for cfl, blend in _ENSTROPHY_MATRIX
)

PARTICLE_COUNT_PROFILES = tuple(
    _profile(
        f"PARTICLE_COUNT_CFL_10_PPC_{ppc}",
        (f"Particle count - CFL 10, {ppc} PPC"
         if ppc != 50 else "Particle count - CFL 10, 50 PPC reference"),
        (f"ST-FLIP particle-count study at target CFL 10 with {ppc} "
         f"particles/cell{' (reference)' if ppc == 50 else ''}; the paper's "
         "standard-FLIP/CFL 1 branch is unavailable."),
        "particle_count",
        "Section 4.5, Figure 16",
        10.0,
        particles_per_cell=ppc,
    )
    for ppc in (1, 2, 4, 8, 16, 50)
)

PROFILES = (
    PAPER_DEFAULTS,
    *LAMINAR_DAM_BREAK_PROFILES,
    *DAM_BREAK_PROFILES,
    *DAM_BREAK_INSTANTANEOUS_PROFILES,
    *ENSTROPHY_PROFILES,
    *PARTICLE_COUNT_PROFILES,
)
PROFILE_BY_ID = MappingProxyType(
    {profile.identifier: profile for profile in PROFILES})

PROFILE_ENUM_ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        CUSTOM_PROFILE_ID,
        "Custom",
        "Keep manually selected parameters; no paper profile is applied.",
    ),
    *(profile.blender_enum_item() for profile in PROFILES),
)

# Explicit aliases make the public intent discoverable to Blender-side code.
EXPERIMENT_PROFILES = PROFILE_BY_ID
EXPERIMENT_PROFILE_ITEMS = PROFILE_ENUM_ITEMS


def get_profile(identifier: str) -> ExperimentProfile | None:
    """Look up a fixed profile; ``CUSTOM`` intentionally resolves to None."""
    if identifier == CUSTOM_PROFILE_ID:
        return None
    try:
        return PROFILE_BY_ID[identifier]
    except KeyError as exc:
        raise ValueError(f"unknown experiment profile: {identifier!r}") from exc


def profile_provenance(identifier: str, values) -> dict[str, object]:
    """Describe whether current controls still match the selected profile."""
    if identifier == CUSTOM_PROFILE_ID:
        return {
            "selected": CUSTOM_PROFILE_ID,
            "matched": CUSTOM_PROFILE_ID,
            "matches_selected": True,
            "mismatched_controls": [],
        }
    try:
        profile = get_profile(identifier)
    except ValueError:
        return {
            "selected": identifier,
            "matched": CUSTOM_PROFILE_ID,
            "matches_selected": False,
            "mismatched_controls": ["unknown_profile"],
        }
    mismatches = []
    for name, expected in profile.settings().items():
        actual = (values.get(name) if isinstance(values, dict)
                  else getattr(values, name))
        if isinstance(expected, float):
            matches = math.isclose(
                float(actual), expected, rel_tol=1e-6, abs_tol=1e-6)
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(name)
    return {
        "selected": identifier,
        "matched": identifier if not mismatches else CUSTOM_PROFILE_ID,
        "matches_selected": not mismatches,
        "mismatched_controls": mismatches,
    }


__all__ = [
    "CUSTOM_PROFILE_ID",
    "DAM_BREAK_INSTANTANEOUS_CFL_1",
    "DAM_BREAK_INSTANTANEOUS_CFL_16",
    "DAM_BREAK_INSTANTANEOUS_PROFILES",
    "DAM_BREAK_PROFILES",
    "ENSTROPHY_PROFILES",
    "EXPERIMENT_PROFILES",
    "EXPERIMENT_PROFILE_ITEMS",
    "ExperimentProfile",
    "LAMINAR_DAM_BREAK_PROFILES",
    "PAPER_DEFAULTS",
    "PAPER_INSPIRED_GEOMETRY_NOTE",
    "PARTICLE_COUNT_PROFILES",
    "PROFILES",
    "PROFILE_BY_ID",
    "PROFILE_ENUM_ITEMS",
    "get_profile",
    "profile_provenance",
]
