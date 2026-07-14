"""Error-aware step-control signals and controller law (roadmap ERR).

Pure functions plus a small host-side state dataclass; bpy-free and
xp-agnostic, importable by tools and tests without the solver.  ERR-M1
lands ONLY diagnostics (the solver captures signals behind an internal
non-Params attribute, bit-identical by default); the controller below is
exercised by unit tests and the study tool, and is wired into the dt
decision only in ERR-M2, after the diagnostics study proves (or kills)
the idea.  No step rejection ever: predictor-only, or the Eq. 10-11
residual carryover, the RNG stream contract, and the equal-subdivision
invariant would all break.

Controller law (review-corrected):
- Per signal, alarm level ``a_i = clip((s_i - lo_i)/(hi_i - lo_i), 0, 1)``;
  combined ``A = max_i(a_i)`` (signals are correlated; worst-case response).
- Decay and release are MUTUALLY EXCLUSIVE: only ``A >= A_RELEASE`` decays
  ``r <- max(r_floor, r * BETA_DOWN ** A)``; ``A < A_RELEASE`` is pure
  quiet.  After ``QUIET_STEPS`` consecutive quiet substeps the controller
  arms and raises ``r <- min(1, r * BETA_UP)`` EVERY quiet substep (the
  counter does not reset on a raise).  Without the exclusivity, mild
  sub-threshold chatter monotonically pins r at the floor.
- Fresh bakes start at r = 1 (trust the user's Target CFL until evidence
  arrives); restores re-initialize at r = r_floor (signal history was
  lost).  The asymmetry is deliberate and tested.
- Signals partially CAUSED by dt decreases (clamp-bind, under-sampled
  faces) are masked for one substep after any dt drop below
  ``MASK_DT_RATIO`` times the previous dt, from any cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

BETA_DOWN = 0.5
BETA_UP = 1.25
A_RELEASE = 0.25
QUIET_STEPS = 3
G_CAP = 2.0
G_SMOOTH = 3
MASK_DT_RATIO = 0.75

# Per-signal (lo, hi) alarm normalisation. Dead-bands (lo) absorb healthy
# baselines; ERR-M1's study output is the calibration evidence for these.
SIGNAL_BANDS = {
    "clamp_bind_fraction": (0.02, 0.10),
    "undersampled_face_fraction": (0.10, 0.35),
    "interface_noise_rms": (0.15, 0.40),
    "near_solid_fast_fraction": (0.002, 0.02),
    "vmax_growth": (1.25, 2.0),
}


def alarm_level(name: str, value: float) -> float:
    """Normalised alarm in [0, 1] for one signal."""

    lo, hi = SIGNAL_BANDS[name]
    if not math.isfinite(value):
        return 1.0
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def combined_alarm(signals: dict) -> float:
    """Worst-case combination of the available signals."""

    level = 0.0
    for name, value in signals.items():
        if name in SIGNAL_BANDS and value is not None:
            level = max(level, alarm_level(name, float(value)))
    return level


@dataclass
class ControllerState:
    """Host-side controller memory (never checkpointed; Decision 3)."""

    r: float = 1.0
    quiet_streak: int = 0
    last_dt: float | None = None
    vmax_history: list = field(default_factory=list)

    @classmethod
    def fresh(cls) -> "ControllerState":
        """Start of a new bake: trust the user's Target CFL."""

        return cls(r=1.0)

    @classmethod
    def restored(cls, r_floor: float) -> "ControllerState":
        """After checkpoint restore: signal history was lost, start low."""

        return cls(r=max(min(r_floor, 1.0), 0.0))


def r_floor(cfl_target: float, strength: float) -> float:
    """Slider mapping: floor = cfl_target ** -strength, never below CFL 1."""

    if cfl_target <= 1.0:
        return 1.0
    floor = cfl_target ** (-max(min(strength, 1.0), 0.0))
    return max(floor, 1.0 / cfl_target)


def masked_signals(signals: dict, state: ControllerState,
                   dt: float) -> dict:
    """Suppress dt-decrease-contaminated signals for one substep."""

    if state.last_dt is not None and dt < MASK_DT_RATIO * state.last_dt:
        signals = dict(signals)
        signals.pop("clamp_bind_fraction", None)
        signals.pop("undersampled_face_fraction", None)
    return signals


def update(state: ControllerState, signals: dict, dt: float,
           floor: float) -> float:
    """Advance the controller one substep; returns the new r."""

    level = combined_alarm(masked_signals(signals, state, dt))
    if level >= A_RELEASE:
        state.quiet_streak = 0
        state.r = max(floor, state.r * (BETA_DOWN ** level))
    else:
        state.quiet_streak += 1
        if state.quiet_streak >= QUIET_STEPS:
            state.r = min(1.0, state.r * BETA_UP)
    state.last_dt = float(dt)
    return state.r


def predicted_vmax(state: ControllerState, vmax: float) -> float:
    """Growth-extrapolated velocity bound (paper Fig. 7 underestimate fix).

    The growth ratio is smoothed over ``G_SMOOTH`` substeps to de-bias
    max-statistic rectification noise, and capped at ``G_CAP``.
    """

    history = state.vmax_history
    history.append(float(vmax))
    del history[:-G_SMOOTH]
    growth = 1.0
    if len(history) >= 2 and history[0] > 1e-9:
        # Geometric endpoint growth: a spike followed by decay reads as
        # flat instead of rectifying upward like a mean of ratios would.
        span = len(history) - 1
        growth = (history[-1] / history[0]) ** (1.0 / span)
    return vmax * min(max(growth, 1.0), G_CAP)


def effective_dt_candidate(state: ControllerState, vmax: float,
                           cfl_target: float, dx: float, t_rem: float,
                           strength: float) -> float:
    """The guarded dt candidate for step_frame (ERR-M2 wiring point).

    Combines the controller multiplier with the vmax predictor, floored so
    the COMBINED reduction never exceeds the slider floor (without this the
    worst case is 8x substeps, not 4x), never exceeds the user's Target
    CFL, and never drops the effective CFL below min(cfl_target, 1)
    with respect to the predicted velocity.
    """

    floor = r_floor(cfl_target, strength)
    vpred = predicted_vmax(state, vmax)
    cfl_eff = max(cfl_target * state.r, min(cfl_target, 1.0))
    dt = cfl_eff * dx / max(vpred, 1e-6)
    dt = max(dt, floor * cfl_target * dx / max(vmax, 1e-6))
    return min(dt, t_rem)
