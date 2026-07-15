"""ST-FLIP solver core (Braun, Winchenbach, Bender, Thuerey - SIGGRAPH 2026).

Implements Algorithm 1 of the paper: a FLIP/PIC liquid solver on a MAC grid
where particles are treated as Monte Carlo samples in 4D space-time.  Each
particle carries a time residual dt_resid = t_global - t_particle; particle-
to-grid deposition multiplies the separable spatial poly6 kernel by a
one-sided temporal kernel evaluated at the particle's slab-normalised sample
time, and per-particle advection times are jittered each step with residual
carryover (Eq. 10-12).  The P2G weight accumulators double as a space-time
phase field providing the variable pressure-projection coefficients (Eq. 13,
15), eliminating per-step surface reconstruction.

The module is bpy-free and runs on NumPy or CuPy via stflip.backend.
"""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from . import (
    apertures,
    forces,
    kernels,
    multigrid,
    pressure,
    sampling,
    st_implicit,
    surface_tension,
    viscosity,
)
from .backend import Backend, get_backend
from .velocity import VelocityField, VelocityInput, as_velocity_field

# Face-grid offsets (in cell units) of node (i,j,k) for each MAC grid.
_OFFSETS = {
    "u": (0.0, 0.5, 0.5),
    "v": (0.5, 0.0, 0.5),
    "w": (0.5, 0.5, 0.0),
    "c": (0.5, 0.5, 0.5),
}

_PRESSURE_OUTFLOW_SIDES = (
    "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
)

_TAPS = [(di, dj, dk) for di in (0, 1) for dj in (0, 1) for dk in (0, 1)]


def _norm_rows(xp, a):
    """Row-wise euclidean norm without cupy.linalg (avoids cuBLAS)."""
    return xp.sqrt((a * a).sum(axis=1))


def _inv3x3(xp, M):
    """Batched analytic inverse of (N,3,3) matrices via the adjugate.

    Avoids cupy.linalg (Windows CuPy wheels ship no cuBLAS); the matrices are
    small and SPD-with-regularisation, so a closed-form cofactor inverse is
    both fast and numerically adequate for the APIC affine reconstruction."""
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    g, h, i = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(b * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - b * g)
    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d
    det = a * c00 + b * c01 + c * c02
    inv_det = 1.0 / xp.where(xp.abs(det) > 1e-30, det, 1e-30)
    out = xp.empty_like(M)
    out[:, 0, 0] = c00 * inv_det
    out[:, 0, 1] = c10 * inv_det
    out[:, 0, 2] = c20 * inv_det
    out[:, 1, 0] = c01 * inv_det
    out[:, 1, 1] = c11 * inv_det
    out[:, 1, 2] = c21 * inv_det
    out[:, 2, 0] = c02 * inv_det
    out[:, 2, 1] = c12 * inv_det
    out[:, 2, 2] = c22 * inv_det
    return out


@dataclass
class Params:
    resolution: tuple[int, int, int] = (64, 64, 64)
    dx: float = 1.0 / 64.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    rho: float = 1000.0
    frame_dt: float = 1.0 / 24.0
    cfl_target: float = 8.0
    particles_per_cell: int = 8
    flip_blend: float = 0.98          # alpha_FLIP
    # False is an instantaneous-P2G temporal ablation. It is not a complete
    # standard-FLIP/GFM implementation because the spatial phase projection
    # remains active.
    st_enabled: bool = True
    jitter_strength: float = 1.0      # base gamma
    adaptive_gamma: bool = True       # attenuate jitter in calm regions (Sec 3.10)
    # Divide each particle's temporal weight by the paper's closed-form mean
    # weight mu(gamma_p) = (945 + 105 g^2 - 21 g^4 - 5 g^6) / 1024 so that
    # E[w] = 1 at every gamma.  The paper (Sec 3.10) skips this and accepts a
    # bounded error: up to 7.7 percent mean weight and 3.9 percent phi_st in
    # attenuated regions.  False reproduces that paper-faithful legacy
    # behavior.  gamma == 1 runs are bitwise identical either way.
    exact_temporal_norm: bool = True
    eta_phi: float = 0.5              # phase-transition steepness
    eps_m: float = 1e-9               # under-sampled face threshold
    eps_rho_rel: float = 1e-3         # eps_rho = eps_rho_rel * rho
    pcg_tol: float = 1e-4
    pcg_max_iter: int = 400
    # PPE preconditioner: "jacobi" is the diagonal-preconditioned CG; "multigrid"
    # wraps it in a geometric V-cycle that makes the iteration count nearly
    # grid-independent (a large win at production resolutions, a wash on small
    # grids, where it transparently falls back to the diagonal path).
    pressure_solver: str = "jacobi"
    # Temporal jitter deviate source (roadmap SAMP-M3).  "pseudo" is the
    # paper-faithful PRNG draw; "sobol_owen" replaces it with the stateless
    # Owen-scrambled low-discrepancy sequence keyed by stable particle ids
    # (opt-in until the SAMP-M5 A/B passes); "cp_rot" is an experimental
    # Cranley-Patterson comparison arm, never a production choice.
    temporal_sampling: str = "pseudo"
    # Jitter attenuation gate (roadmap CALM-M3/M4).  "speed" is the
    # paper's Sec 3.10 local-CFL gate, bit-exact with earlier releases.
    # "surface" ADDS an interiorness term (never subtracts) so deep-liquid
    # and deep-gas particles keep full jitter; where local CFL ~ 0 the
    # field is temporally constant and restored jitter has no aliasing to
    # fight, so that tier is hygiene and uniform-stratification
    # groundwork.  "deformation" (CALM-M4, speculative) replaces the
    # speed activity with max(normal-displacement CFL, normal strain
    # rate, grid-frame unsteadiness) plus the interiorness term, so a
    # calm surface above a MOVING bulk (stirred pool, river) can damp
    # where the speed gate cannot; a measured miss ships default-off.
    gamma_mode: str = "speed"
    cfl_local: float = 1.0            # advection sub-step bound
    seed: int = 0

    # --- Velocity transfer (Sec 3.9) -------------------------------------
    transfer: str = "flip"            # "flip" | "apic" | "pic"
    apic_reg: float = 1e-2            # Tikhonov reg (in dx^2) for the APIC D^-1

    # --- Two-phase gas coupling (Sec 3.1, 3.6-3.7) -----------------------
    two_phase: bool = False           # couple a light gas phase to the liquid
    rho_gas: float = 1.2              # gas density rho_g (air ~= 1.2 kg/m^3)
    gas_particles_per_cell: int = 8   # ppc used to fill the gas region

    # --- Surface tension (Sec 3.9, CSF model) ----------------------------
    surface_tension: float = 0.0      # sigma (N/m); 0 disables the CSF force
    st_smooth_iters: int = 2          # B-spline smoothing passes for curvature
    # Capillary clamp relaxation (roadmap CAP-M0).  Scales the Brackbill dt
    # cap; above 1 the explicit CSF feedback is no longer provably stable,
    # so pair modest scales (2-4) with the st_max_dv_cells limiter.  The
    # clamp itself is never removed, only scaled by this bounded factor.
    st_clamp_scale: float = 1.0       # 1 = paper-faithful Brackbill clamp
    # Semi-implicit capillary stabilizer (roadmap CAP-M2): an
    # interface-concentrated Helmholtz solve per velocity component that
    # damps the stiff capillary-wave feedback, letting st_clamp_scale be
    # raised (recommended 4-16 for droplet/glugging scenes) without the
    # explicit CSF loop blowing up.  No unconditional-stability proof is
    # claimed; the Brackbill clamp is scaled, never removed.
    st_implicit: bool = False
    # Advection-reflection (roadmap ENER-M2b, Zehnder et al. 2018 adapted
    # to slab-integrated P2G): two half-advections through PROJECTED
    # fields with a mid-step reflected momentum update, halving the
    # first-order splitting's energy/angular-momentum loss per unit
    # physical time at roughly the grid cost of two substeps.  Enable it
    # when RAISING Target CFL (reflection at 2N costs about the same
    # grid work as plain at N); pair with pressure_solver="multigrid".
    # Where temporal MC noise dominates (calm scenes) it can lose --
    # reflection reduces SPLITTING dissipation only.
    reflection: bool = False
    # Per-face limiter on the explicit CSF kick: the velocity change per
    # substep is clipped so it can displace at most this many cells per
    # step (dv_max = st_max_dv_cells * dx / dt).  0 disables the limiter.
    # Above the Brackbill limit the clipped feedback saturates as bounded
    # grid-scale interface chatter instead of blowing up -- robustness
    # insurance, not accuracy.
    st_max_dv_cells: float = 0.0

    # --- Viscosity (implicit diffusion) ----------------------------------
    viscosity: float = 0.0            # kinematic viscosity (dx^2/s); 0 = inviscid
    visc_tol: float = 1e-5            # implicit-diffusion CG tolerance
    visc_max_iter: int = 200

    # --- Particle sheeting / anti-clumping (position-only) ---------------
    sheeting: float = 0.0             # 0..~1 strength; spreads clumps, fills voids

    # --- Sparse production grid (active-block domain cropping) -----------
    sparse: bool = False              # crop grids to the active particle region
    block_size: int = 8               # active-block granularity (cells)
    sparse_pad: int = 0               # extra halo blocks beyond the fluid band

    def __post_init__(self) -> None:
        """Validate and normalize all public solver parameters eagerly.

        Invalid values otherwise tend to fail much later in array allocation,
        pressure projection, or the adaptive-step loop.  Keeping this check on
        the bpy-free boundary also gives CPU and CUDA callers identical errors.
        """
        try:
            resolution = tuple(self.resolution)
        except TypeError as exc:
            raise ValueError("resolution must contain three positive integers") from exc
        if (
            len(resolution) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) <= 0
                for value in resolution
            )
        ):
            raise ValueError("resolution must contain three positive integers")
        self.resolution = tuple(int(value) for value in resolution)

        try:
            gravity = tuple(float(value) for value in self.gravity)
        except (TypeError, ValueError) as exc:
            raise ValueError("gravity must contain three finite values") from exc
        if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
            raise ValueError("gravity must contain three finite values")
        self.gravity = gravity

        scalar_rules = {
            "dx": (0.0, None),
            "rho": (0.0, None),
            "frame_dt": (0.0, None),
            "cfl_target": (0.0, None),
            "flip_blend": (0.0, 1.0),
            "jitter_strength": (0.0, 1.0),
            "eta_phi": (0.0, None),
            "eps_m": (0.0, None),
            "eps_rho_rel": (0.0, None),
            "pcg_tol": (0.0, None),
            "cfl_local": (0.0, None),
        }
        for name, (lower, upper) in scalar_rules.items():
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if upper is None:
                if value <= lower:
                    raise ValueError(f"{name} must be positive")
            elif not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
            setattr(self, name, value)

        for name in ("particles_per_cell", "pcg_max_iter"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        for name in ("st_enabled", "adaptive_gamma", "exact_temporal_norm",
                     "st_implicit", "reflection"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
            raise ValueError("seed must be an integer")
        self.seed = int(self.seed)
        if self.seed < 0:
            raise ValueError("seed must not be negative")

        # --- Extension parameters (transfer / two-phase / ST / sparse) ----
        if self.transfer not in ("flip", "apic", "pic"):
            raise ValueError("transfer must be 'flip', 'apic', or 'pic'")
        if self.pressure_solver not in ("jacobi", "multigrid"):
            raise ValueError("pressure_solver must be 'jacobi' or 'multigrid'")
        if self.temporal_sampling not in ("pseudo", "sobol_owen", "cp_rot"):
            raise ValueError(
                "temporal_sampling must be 'pseudo', 'sobol_owen', "
                "or 'cp_rot'")
        if self.gamma_mode not in ("speed", "surface", "deformation"):
            raise ValueError(
                "gamma_mode must be 'speed', 'surface', or 'deformation'")
        for name in ("two_phase", "sparse"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("apic_reg", "rho_gas"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        st = float(self.surface_tension)
        if not math.isfinite(st) or st < 0.0:
            raise ValueError("surface_tension must be non-negative")
        self.surface_tension = st
        clamp_scale = float(self.st_clamp_scale)
        if not math.isfinite(clamp_scale) or not 1.0 <= clamp_scale <= 16.0:
            raise ValueError("st_clamp_scale must lie in [1, 16]")
        self.st_clamp_scale = clamp_scale
        max_dv = float(self.st_max_dv_cells)
        if not math.isfinite(max_dv) or max_dv < 0.0:
            raise ValueError("st_max_dv_cells must be non-negative")
        self.st_max_dv_cells = max_dv
        visc = float(self.viscosity)
        if not math.isfinite(visc) or visc < 0.0:
            raise ValueError("viscosity must be non-negative")
        self.viscosity = visc
        sh = float(self.sheeting)
        if not math.isfinite(sh) or sh < 0.0:
            raise ValueError("sheeting must be non-negative")
        self.sheeting = sh
        vt = float(self.visc_tol)
        if not math.isfinite(vt) or vt <= 0.0:
            raise ValueError("visc_tol must be positive")
        self.visc_tol = vt
        if (isinstance(self.visc_max_iter, bool)
                or not isinstance(self.visc_max_iter, numbers.Integral)
                or int(self.visc_max_iter) <= 0):
            raise ValueError("visc_max_iter must be a positive integer")
        self.visc_max_iter = int(self.visc_max_iter)
        for name in ("gas_particles_per_cell", "block_size"):
            value = getattr(self, name)
            if (isinstance(value, bool)
                    or not isinstance(value, numbers.Integral)
                    or int(value) <= 0):
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        for name in ("st_smooth_iters", "sparse_pad"):
            value = getattr(self, name)
            if (isinstance(value, bool)
                    or not isinstance(value, numbers.Integral)
                    or int(value) < 0):
                raise ValueError(f"{name} must be a non-negative integer")
            setattr(self, name, int(value))


@dataclass
class FrameStats:
    steps: int = 0
    dt_values: list = field(default_factory=list)
    inactive_time_s: float = 0.0
    # This solver's adaptive step uses maximum particle speed. These names are
    # deliberately not "grid CFL": the paper's diagnostic may be computed
    # from a different MaxVelocity implementation in another host solver.
    particle_cfl_estimated_values: list = field(default_factory=list)
    particle_cfl_actual_values: list = field(default_factory=list)
    pcg_iters: list = field(default_factory=list)
    pcg_rel_residuals: list = field(default_factory=list)
    n_particles: int = 0
    max_speed: float = 0.0
    particles_removed: int = 0
    volume_outflow_removed: int = 0
    pressure_outflow_removed: int = 0
    # ERR-M1 step-control diagnostics (populated only when the solver's
    # internal _collect_step_diagnostics attribute is set; plain FrameStats
    # payload, NOT the strict metrics frame-record schema).
    st_cg_iters: list = field(default_factory=list)
    st_cg_rel_residuals: list = field(default_factory=list)
    clamp_bind_fractions: list = field(default_factory=list)
    undersampled_marginal_fractions: list = field(default_factory=list)
    undersampled_subthreshold_fractions: list = field(default_factory=list)
    near_solid_fast_fractions: list = field(default_factory=list)
    capillary_clamped_steps: list = field(default_factory=list)


class STFLIPSolver:
    def __init__(self, params: Params, backend: Backend | str = "auto"):
        self.p = params
        self.be = get_backend(backend) if isinstance(backend, str) else backend
        xp = self.be.xp
        nx, ny, nz = params.resolution
        self.shape = (nx, ny, nz)
        self.size = (nx * params.dx, ny * params.dx, nz * params.dx)

        # Particle state (device arrays).
        self.pos = xp.zeros((0, 3), dtype=xp.float32)
        self.vel = xp.zeros((0, 3), dtype=xp.float32)
        self.dt_resid = xp.zeros((0,), dtype=xp.float32)
        # Phase indicator chi_l in {0, 1}: 1 = liquid, 0 = gas.  Only the gas
        # column is populated when two_phase is enabled.
        self.phase = xp.zeros((0,), dtype=xp.float32)
        # APIC affine velocity matrix C (Jiang et al. 2015); one 3x3 per
        # particle, only advanced in "apic" transfer mode.
        self.C = xp.zeros((0, 3, 3), dtype=xp.float32)
        # Shading attributes: seconds since a particle was seeded, and the
        # 0-based id of the source (liquid/inflow) that seeded it.  Exported
        # as point attributes for age-fade, speed, and per-source colouring.
        self.age = xp.zeros((0,), dtype=xp.float32)
        self.source_id = xp.zeros((0,), dtype=xp.int32)
        self._next_source = 0
        # Stable per-particle identity (roadmap SAMP-M1): ids are allocated
        # monotonically, survive outflow compaction, and are never reused --
        # a duplicated id would silently correlate two particles in any
        # id-keyed sampling scheme.  Physics never reads them.
        self.particle_id = xp.zeros((0,), dtype=xp.int64)
        self._next_particle_id = 0
        self._gamma_prev = None
        # Global substep counter, committed once per completed substep.
        self._substep_index = 0

        # Device-resident constants (allocating these per call would force
        # host->device transfers inside the advection hot loop on GPU).
        self._offsets_dev = {g: xp.asarray(off, dtype=xp.float32)
                             for g, off in _OFFSETS.items()}
        eps = 1e-3 * params.dx
        self._clamp_lo = xp.asarray([eps] * 3, dtype=xp.float32)
        self._clamp_hi = xp.asarray([s - eps for s in self.size],
                                    dtype=xp.float32)
        self._domain_size_dev = xp.asarray(self.size, dtype=xp.float32)

        # Solid signed distance, cell-centred; positive outside solids.  A
        # node-centred SDF is optional: when present it defines fractional
        # face apertures for the pressure projection, while the cell-centred
        # field remains the collision/push-out representation.
        self.sdf = xp.full(self.shape, 1e9, dtype=xp.float32)
        self._solid_node_sdf = None
        self._sdf_grad = None
        self._solid_faces = None  # apertures + fully-solid cells, built lazily
        self._solid_exterior_apertures = None
        # Cell-centred solid velocity for animated (moving-wall) obstacles;
        # None means every solid is static (u_solid = 0).
        self.solid_vel = None
        # Cell offset of the last stored (windowed) grids for sparse resync.
        self._full_shape = (nx, ny, nz)
        self._grid_origin = None
        # Art-directable body forces (wind/vortex/turbulence); each is a spec
        # dict consumed in _step_core.  The frame origin is the sparse-window
        # cell offset so solver-local forces stay put when the window moves.
        self._forces: list[dict] = []
        self._frame_origin_cells = np.zeros(3, dtype=np.float64)

        # Masks stay on-device for occupancy checks; immutable velocity fields
        # stay on the host so each refill is sampled deterministically at its
        # newly jittered particle positions.
        # Inflows are static source masks with an immutable velocity field and
        # an optional active interval in solver seconds. Scheduling depends
        # only on ``self.time``, so exact checkpoints need no extra emitter
        # state beyond the simulation clock they already store.
        self._inflows: list[
            tuple[object, VelocityField, float, float | None, float, int]
        ] = []
        # Outflow masks are lazy: two dense 512^3 boolean allocations would
        # otherwise cost ~256 MiB on every no-outflow CUDA simulation.
        self._volume_outflow = None
        self._pressure_outflow = None
        self._has_volume_outflow = False
        self._has_pressure_outflow = False
        self._pressure_outflow_side_masks = {
            side: None for side in _PRESSURE_OUTFLOW_SIDES
        }
        self._pressure_outflow_faces = None
        self._outflow_geometry_stats_cache = None
        self._outflow_removed_total = 0
        self._volume_outflow_removed_total = 0
        self._pressure_outflow_removed_total = 0
        self._rng = np.random.default_rng(params.seed)
        self._dt_prev = params.frame_dt / max(params.cfl_target, 1.0)
        self.time = 0.0

        self._grids: dict = {}
        self.m0 = self._calibrate_m0()

    def _reconcile_particle_attrs(self) -> None:
        """Keep phase (and, in APIC mode, C) length-consistent with pos.

        Most mutations go through _seed_cells / _apply_outflow_filter which
        already maintain them, but callers may assign self.pos directly; this
        pads new particles as liquid with zero affine and drops stragglers."""
        xp = self.be.xp
        n = self.pos.shape[0]
        m = int(self.phase.shape[0])
        if m != n:
            self.phase = (
                xp.concatenate([self.phase, xp.ones((n - m,), dtype=xp.float32)])
                if m < n else self.phase[:n])
        if self._use_apic:
            cm = int(self.C.shape[0])
            if cm != n:
                self.C = (
                    xp.concatenate(
                        [self.C, xp.zeros((n - cm, 3, 3), dtype=xp.float32)])
                    if cm < n else self.C[:n])
        am = int(self.age.shape[0])
        if am != n:
            self.age = (
                xp.concatenate([self.age, xp.zeros((n - am,), dtype=xp.float32)])
                if am < n else self.age[:n])
        sm = int(self.source_id.shape[0])
        if sm != n:
            self.source_id = (
                xp.concatenate(
                    [self.source_id, xp.zeros((n - sm,), dtype=xp.int32)])
                if sm < n else self.source_id[:n])
        pm = int(self.particle_id.shape[0])
        if pm != n:
            if pm < n:
                # Pad with FRESH ids, never zeros: a duplicated id would
                # silently correlate two particles in id-keyed sampling.
                fresh = xp.arange(
                    self._next_particle_id,
                    self._next_particle_id + (n - pm), dtype=xp.int64)
                self._next_particle_id += n - pm
                self.particle_id = xp.concatenate([self.particle_id, fresh])
            else:
                self.particle_id = self.particle_id[:n]

    def checkpoint_state(self) -> dict:
        """Return an owned, backend-neutral snapshot sufficient to restart.

        Grid velocities, aperture caches, and gradients are derived from the
        configured scene plus particle state and are rebuilt on the next step.
        The NumPy RNG state and previous adaptive timestep are trajectory state
        and therefore must be captured alongside particle arrays.

        The two-phase gas tag, APIC affine matrix, and shading attributes (age,
        source id) are persisted so a resumed two-phase/APIC bake continues
        without a discontinuity at the resume frame. ``C`` keeps its live shape:
        (N, 3, 3) under APIC, or (0, 3, 3) otherwise.
        """
        state = {
            "pos": np.array(
                self.be.to_numpy(self.pos), dtype=np.float32, order="C",
                copy=True),
            "vel": np.array(
                self.be.to_numpy(self.vel), dtype=np.float32, order="C",
                copy=True),
            "dt_resid": np.array(
                self.be.to_numpy(self.dt_resid), dtype=np.float32, order="C",
                copy=True),
            "time": float(self.time),
            "dt_prev": float(self._dt_prev),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "outflow_removed_total": int(self._outflow_removed_total),
            "volume_outflow_removed_total": int(
                self._volume_outflow_removed_total),
            "pressure_outflow_removed_total": int(
                self._pressure_outflow_removed_total),
            "phase": np.array(
                self.be.to_numpy(self.phase), dtype=np.float32, order="C",
                copy=True),
            "C": np.array(
                self.be.to_numpy(self.C), dtype=np.float32, order="C",
                copy=True),
            "age": np.array(
                self.be.to_numpy(self.age), dtype=np.float32, order="C",
                copy=True),
            "source_id": np.array(
                self.be.to_numpy(self.source_id), dtype=np.int32, order="C",
                copy=True),
            "particle_id": np.array(
                self.be.to_numpy(self.particle_id), dtype=np.int64, order="C",
                copy=True),
            "next_particle_id": int(self._next_particle_id),
            "substep_index": int(self._substep_index),
        }
        if (self.p.gamma_mode != "speed" and self.p.exact_temporal_norm
                and self._gamma_prev is not None
                and self._gamma_prev.shape[0] == self.pos.shape[0]):
            # Surface-mode exact normalization keys on the draw-time gamma,
            # whose interiorness input no longer exists after the substep;
            # persist it via the mode-gated optional checkpoint member
            # reserved in the SAMP-M1 schema bump.
            state["gamma_prev"] = np.array(
                self.be.to_numpy(self._gamma_prev), dtype=np.float32,
                order="C", copy=True)
        return state

    def restore_state(self, state: dict) -> None:
        """Strictly restore a snapshot into this configured solver instance.

        Version-2 checkpoints carry the phase tag, APIC affine matrix, and
        shading attributes; version-1 checkpoints omit them and restore to the
        historical defaults (liquid phase, empty affine, zero age/source).
        """
        required = {
            "pos", "vel", "dt_resid", "time", "dt_prev", "rng_state",
            "outflow_removed_total", "volume_outflow_removed_total",
            "pressure_outflow_removed_total",
        }
        optional = {
            "phase", "C", "age", "source_id",
            "particle_id", "next_particle_id", "substep_index",
            "gamma_prev",
        }
        if (not isinstance(state, dict)
                or not required <= set(state)
                or not set(state) <= (required | optional)):
            raise ValueError("checkpoint state has an incompatible schema")

        def particle_array(name, shape=None):
            try:
                array = np.asarray(state[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"checkpoint {name} must be a float32 array") from exc
            if array.dtype != np.dtype(np.float32):
                raise ValueError(f"checkpoint {name} must be a float32 array")
            if shape is not None and array.shape != shape:
                raise ValueError(f"checkpoint {name} has an incompatible shape")
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"checkpoint {name} contains non-finite values")
            return np.array(array, dtype=np.float32, order="C", copy=True)

        pos = particle_array("pos")
        if pos.ndim != 2 or pos.shape[1:] != (3,):
            raise ValueError("checkpoint pos must have shape (N, 3)")
        vel = particle_array("vel", pos.shape)
        dt_resid = particle_array("dt_resid", (pos.shape[0],))
        domain_size = np.asarray(self.size, dtype=np.float32)
        if np.any(pos < 0.0) or np.any(pos > domain_size[None, :]):
            raise ValueError("checkpoint particles lie outside the solver domain")

        def finite_scalar(name, *, positive=False):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"checkpoint {name} must be finite")
            result = float(value)
            if not math.isfinite(result) or result < 0.0:
                raise ValueError(f"checkpoint {name} must be finite")
            if positive and result <= 0.0:
                raise ValueError(f"checkpoint {name} must be positive")
            return result

        time_value = finite_scalar("time")
        dt_prev = finite_scalar("dt_prev", positive=True)

        counters = {}
        for name in (
            "outflow_removed_total",
            "volume_outflow_removed_total",
            "pressure_outflow_removed_total",
        ):
            value = state[name]
            if (isinstance(value, bool)
                    or not isinstance(value, numbers.Integral)
                    or int(value) < 0):
                raise ValueError(
                    f"checkpoint {name} must be a non-negative integer")
            counters[name] = int(value)
        if counters["outflow_removed_total"] != (
                counters["volume_outflow_removed_total"]
                + counters["pressure_outflow_removed_total"]):
            raise ValueError("checkpoint outflow counters are inconsistent")

        if not isinstance(state["rng_state"], dict):
            raise ValueError("checkpoint rng_state is invalid")
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("checkpoint rng_state is invalid") from exc

        # Validate the optional per-particle members (if present) before any
        # mutation, so a malformed extra cannot leave the solver half-restored.
        xp = self.be.xp
        count = pos.shape[0]
        phase_restore = None
        if "phase" in state:
            phase_np = particle_array("phase", (count,))
            if not bool(np.all((phase_np == 0.0) | (phase_np == 1.0))):
                raise ValueError("checkpoint phase must contain 0 or 1")
            phase_restore = self.be.from_numpy(phase_np)
        affine_restore = None
        if "C" in state and self._use_apic:
            affine_np = np.asarray(state["C"])
            if (affine_np.dtype != np.dtype(np.float32) or affine_np.ndim != 3
                    or affine_np.shape[1:] != (3, 3)
                    or affine_np.shape[0] not in (0, count)):
                raise ValueError("checkpoint C has an incompatible shape")
            if not bool(np.all(np.isfinite(affine_np))):
                raise ValueError("checkpoint C contains non-finite values")
            affine_restore = self.be.from_numpy(
                np.array(affine_np, dtype=np.float32, order="C", copy=True))
        age_restore = None
        if "age" in state:
            age_np = particle_array("age", (count,))
            age_restore = self.be.from_numpy(age_np)
        source_restore = None
        if "source_id" in state:
            source_np = np.asarray(state["source_id"])
            if source_np.dtype != np.dtype(np.int32) \
                    or source_np.shape != (count,):
                raise ValueError("checkpoint source_id has an incompatible shape")
            if source_np.size and int(source_np.min()) < 0:
                raise ValueError("checkpoint source_id must be non-negative")
            source_restore = self.be.from_numpy(
                np.array(source_np, dtype=np.int32, order="C", copy=True))
        gamma_prev_restore = None
        if "gamma_prev" in state:
            gamma_np = particle_array("gamma_prev", (count,))
            if gamma_np.size and (
                    float(gamma_np.min()) < 0.0
                    or float(gamma_np.max()) > 1.0):
                raise ValueError("checkpoint gamma_prev must lie in [0, 1]")
            gamma_prev_restore = self.be.from_numpy(gamma_np)
        particle_id_restore = None
        next_particle_id = None
        if "particle_id" in state:
            id_np = np.asarray(state["particle_id"])
            if id_np.dtype != np.dtype(np.int64) or id_np.shape != (count,):
                raise ValueError(
                    "checkpoint particle_id has an incompatible shape")
            if id_np.size:
                if int(id_np.min()) < 0:
                    raise ValueError(
                        "checkpoint particle_id must be non-negative")
                if np.unique(id_np).size != count:
                    raise ValueError("checkpoint particle_id must be unique")
            particle_id_restore = self.be.from_numpy(
                np.array(id_np, dtype=np.int64, order="C", copy=True))
            next_value = state.get("next_particle_id", 0)
            if (isinstance(next_value, bool)
                    or not isinstance(next_value, numbers.Integral)
                    or int(next_value) < 0):
                raise ValueError(
                    "checkpoint next_particle_id must be a non-negative "
                    "integer")
            next_particle_id = int(next_value)
        substep_index = state.get("substep_index", 0)
        if (isinstance(substep_index, bool)
                or not isinstance(substep_index, numbers.Integral)
                or int(substep_index) < 0):
            raise ValueError(
                "checkpoint substep_index must be a non-negative integer")

        # Commit only after the complete state validates, so a rejected restore
        # cannot leave a running solver partially mutated.
        self.pos = self.be.from_numpy(pos)
        self.vel = self.be.from_numpy(vel)
        self.dt_resid = self.be.from_numpy(dt_resid)
        # Restore the two-phase tag when persisted; otherwise (version-1
        # checkpoint) fall back to single-phase liquid.
        self.phase = (
            phase_restore if phase_restore is not None
            else xp.ones((count,), dtype=xp.float32))
        # Under APIC, restore the affine field: a stored (N, 3, 3) is used as
        # is, while an empty (0, 3, 3) is regrown to zeros on the next step.
        # Without APIC the field stays empty regardless of what was stored.
        self.C = (
            affine_restore if affine_restore is not None
            else xp.zeros((0, 3, 3), dtype=xp.float32))
        # Restore shading attributes when persisted; version-1 checkpoints
        # restart ages at zero and source ids at zero.
        self.age = (
            age_restore if age_restore is not None
            else xp.zeros((count,), dtype=xp.float32))
        self.source_id = (
            source_restore if source_restore is not None
            else xp.zeros((count,), dtype=xp.int32))
        # New sources seeded after a resume must not reuse a restored id, so
        # advance the counter past the largest restored source id.
        if source_restore is not None and count:
            self._next_source = max(
                self._next_source, int(self.be.to_numpy(self.source_id).max()) + 1)
        # Restore stable particle ids when persisted (checkpoint v3);
        # pre-v3 checkpoints synthesize ids 0..n-1 and restart the substep
        # counter -- permitted ONLY for the pseudo sampler, whose deviates
        # never key on ids.  An id-keyed low-discrepancy bake cannot resume
        # from synthesized identities without silently changing every
        # subsequent deviate.
        if particle_id_restore is None and self.p.temporal_sampling != "pseudo":
            raise ValueError(
                "checkpoint predates stable particle ids (schema v3); "
                "resuming a temporal_sampling != 'pseudo' bake requires a "
                "v3 checkpoint - re-bake from frame zero")
        if particle_id_restore is not None:
            self.particle_id = particle_id_restore
            floor = int(next_particle_id)
            if count:
                floor = max(
                    floor,
                    int(self.be.to_numpy(self.particle_id).max()) + 1)
            self._next_particle_id = floor
        else:
            self.particle_id = xp.arange(count, dtype=xp.int64)
            self._next_particle_id = count
        self._substep_index = int(substep_index)
        self._gamma_prev = gamma_prev_restore
        self.time = time_value
        self._dt_prev = dt_prev
        self._rng = restored_rng
        self._outflow_removed_total = counters["outflow_removed_total"]
        self._volume_outflow_removed_total = counters[
            "volume_outflow_removed_total"]
        self._pressure_outflow_removed_total = counters[
            "pressure_outflow_removed_total"]
        self._grids = {}

    # ------------------------------------------------------------------ setup

    @property
    def _use_apic(self) -> bool:
        return self.p.transfer == "apic"

    def set_solid_sdf(
        self,
        sdf_cells: np.ndarray,
        node_sdf: np.ndarray | None = None,
        solid_vel: np.ndarray | None = None,
    ) -> None:
        """Set the collision SDF and, optionally, fractional solid geometry.

        ``sdf_cells`` preserves the original binary cell-centred API and is
        always used for particle collision push-out.  Supplying
        ``node_sdf`` with shape ``(nx + 1, ny + 1, nz + 1)`` enables
        cut-cell face apertures for pressure projection.  Omitting it keeps
        the previous binary rule: a face is blocked when either adjacent
        cell is solid.  ``solid_vel`` is an optional ``(nx, ny, nz, 3)``
        cell-centred solid velocity for animated moving-wall obstacles; call
        this once per frame with the current obstacle pose to drive kinematic
        boundaries.
        """
        cells = self._validate_sdf_array(sdf_cells, self.shape, "sdf_cells")
        nodes = None
        if node_sdf is not None:
            expected = tuple(n + 1 for n in self.shape)
            nodes = self._validate_sdf_array(node_sdf, expected, "node_sdf")

        xp = self.be.xp
        self.sdf = self.be.from_numpy(cells)
        if nodes is None:
            self._solid_node_sdf = None
        else:
            self._solid_node_sdf = self.be.from_numpy(nodes)
        if solid_vel is None:
            self.solid_vel = None
        else:
            vel = self._validate_sdf_array(
                np.asarray(solid_vel), self.shape + (3,), "solid_vel")
            self.solid_vel = self.be.from_numpy(vel)
        self._sdf_grad = tuple(
            xp.gradient(self.sdf, self.p.dx, axis=axis)
            if extent > 1
            else xp.zeros_like(self.sdf)
            for axis, extent in enumerate(self.shape)
        )
        self._solid_faces = None  # rebuild on next step
        self._solid_exterior_apertures = None
        self._pressure_outflow_faces = None
        self._outflow_geometry_stats_cache = None

    @staticmethod
    def _validate_sdf_array(value, expected_shape, name: str) -> np.ndarray:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a finite numeric array with shape "
                f"{expected_shape}"
            ) from exc
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        try:
            finite = bool(np.all(np.isfinite(array)))
        except TypeError as exc:
            raise TypeError(f"{name} must contain numeric values") from exc
        if not finite:
            raise ValueError(f"{name} must contain only finite values")
        owned = np.array(array, dtype=np.float32, order="C", copy=True)
        if not np.all(np.isfinite(owned)):
            raise ValueError(f"{name} values exceed the float32 finite range")
        return owned

    def add_liquid_mask(
        self,
        cell_mask: np.ndarray,
        velocity: VelocityInput = (0.0, 0.0, 0.0),
    ) -> int:
        """Seed jittered particles and sample velocity at their positions."""
        field = as_velocity_field(velocity)
        mask = self._validate_cell_mask(cell_mask)
        cells = np.argwhere(mask)
        sid = self._next_source
        self._next_source += 1
        return self._seed_cells(cells, field, source_id=sid)

    def add_inflow(
        self,
        cell_mask: np.ndarray,
        velocity: VelocityInput = (0.0, 0.0, 0.0),
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
        phase: float = 1.0,
    ) -> None:
        """Register a refill source, optionally limited to a time interval.

        ``start_time`` is inclusive and ``end_time`` is exclusive, both in
        solver seconds relative to the bake start. The default remains an
        always-active source. Refill is occupancy based, not a prescribed
        volumetric flow rate. Equal endpoints define a valid inactive source.
        """
        field = as_velocity_field(velocity)
        mask = self._validate_cell_mask(cell_mask)
        if (isinstance(start_time, bool)
                or not isinstance(start_time, numbers.Real)
                or not math.isfinite(float(start_time))
                or float(start_time) < 0.0):
            raise ValueError("inflow start_time must be finite and non-negative")
        start = float(start_time)
        end = None
        if end_time is not None:
            if (isinstance(end_time, bool)
                    or not isinstance(end_time, numbers.Real)
                    or not math.isfinite(float(end_time))
                    or float(end_time) < start):
                raise ValueError(
                    "inflow end_time must be finite and not precede start_time"
                )
            end = float(end_time)
        if (isinstance(phase, bool)
                or not isinstance(phase, numbers.Real)
                or float(phase) not in (0.0, 1.0)):
            raise ValueError("inflow phase must be 0 or 1")
        phase_value = float(phase)
        sid = self._next_source
        self._next_source += 1
        self._inflows.append(
            (self.be.from_numpy(mask), field, start, end, phase_value, sid)
        )

    def add_outflow(
        self,
        cell_mask: np.ndarray,
        mode: str = "VOLUME",
        *,
        faces=None,
    ) -> None:
        """Register a particle sink or an exterior pressure/open boundary.

        ``VOLUME`` removes particles entering any marked cell, with the mask
        tested after every local RK advection substep. ``PRESSURE`` opens only
        the simulation-domain faces intersected by marked boundary cells,
        imposes exterior ``p = 0`` at half-cell distance, and removes particles
        after they cross one of those faces.  ``faces`` may restrict a PRESSURE
        mask to named faces (``x_min`` through ``z_max``); omitting it preserves
        the historical behavior of opening every exterior face touched by the
        mask.
        """
        if not isinstance(mode, str):
            raise ValueError("outflow mode must be 'VOLUME' or 'PRESSURE'")
        normalized = mode.strip().upper()
        if normalized not in {"VOLUME", "PRESSURE"}:
            raise ValueError("outflow mode must be 'VOLUME' or 'PRESSURE'")
        if faces is not None and normalized != "PRESSURE":
            raise ValueError("outflow faces apply only to PRESSURE mode")
        if faces is None:
            selected_sides = _PRESSURE_OUTFLOW_SIDES
        else:
            try:
                values = (faces,) if isinstance(faces, str) else tuple(faces)
                selected_sides = tuple(
                    str(side).strip().lower() for side in values)
            except TypeError as exc:
                raise ValueError(
                    "pressure outflow faces must be exterior face names") from exc
            if (not selected_sides
                    or len(set(selected_sides)) != len(selected_sides)
                    or any(side not in _PRESSURE_OUTFLOW_SIDES
                           for side in selected_sides)):
                raise ValueError(
                    "pressure outflow faces must be unique exterior face names")
        mask = self._validate_cell_mask(cell_mask)
        if normalized == "PRESSURE" and np.any(mask) and not (
            np.any(mask[0])
            or np.any(mask[-1])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
            or np.any(mask[:, :, 0])
            or np.any(mask[:, :, -1])
        ):
            raise ValueError(
                "PRESSURE outflow mask must intersect the domain exterior"
            )
        boundary_views = {
            "x_min": mask[0],
            "x_max": mask[-1],
            "y_min": mask[:, 0],
            "y_max": mask[:, -1],
            "z_min": mask[:, :, 0],
            "z_max": mask[:, :, -1],
        }
        if normalized == "PRESSURE":
            missing = [
                side for side in selected_sides
                if not np.any(boundary_views[side])
            ]
            if faces is not None and missing:
                raise ValueError(
                    "PRESSURE outflow mask has no marked cells on requested "
                    f"side(s): {', '.join(missing)}")
        has_cells = bool(np.any(mask))
        if not has_cells:
            return
        device_mask = self.be.from_numpy(mask)
        if normalized == "VOLUME":
            self._volume_outflow = (
                device_mask
                if self._volume_outflow is None
                else self._volume_outflow | device_mask
            )
            self._has_volume_outflow = True
        else:
            self._pressure_outflow = (
                device_mask
                if self._pressure_outflow is None
                else self._pressure_outflow | device_mask
            )
            device_boundaries = {
                "x_min": device_mask[0],
                "x_max": device_mask[-1],
                "y_min": device_mask[:, 0],
                "y_max": device_mask[:, -1],
                "z_min": device_mask[:, :, 0],
                "z_max": device_mask[:, :, -1],
            }
            for side in selected_sides:
                boundary = device_boundaries[side].copy()
                existing = self._pressure_outflow_side_masks[side]
                self._pressure_outflow_side_masks[side] = (
                    boundary if existing is None else existing | boundary)
            self._has_pressure_outflow = True
            self._pressure_outflow_faces = None
        self._outflow_geometry_stats_cache = None

    def add_force(self, force_type: str, strength: float, *,
                  direction=(0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0),
                  axis=(0.0, 0.0, 1.0), radius: float = 1e9,
                  scale: float = 1.0, seed: int = 0) -> None:
        """Register an art-directable body force applied like gravity.

        ``force_type`` is 'DIRECTIONAL' (wind along ``direction``), 'VORTEX'
        (swirl about ``axis`` through ``center`` with ``radius`` falloff),
        'TURBULENCE' (divergence-free curl noise of wavelength ``scale``), or
        'CONFINEMENT' (vorticity confinement -- re-energizes existing swirls
        from the flow's own vorticity; a look control that injects energy).
        ``strength`` is the acceleration magnitude. ``center`` is solver-local;
        ``direction`` and ``axis`` are normalized world-oriented vectors.
        """
        ft = str(force_type).strip().upper()
        if ft not in {"DIRECTIONAL", "VORTEX", "TURBULENCE", "CONFINEMENT"}:
            raise ValueError(
                "force_type must be DIRECTIONAL, VORTEX, TURBULENCE, "
                "or CONFINEMENT")
        if not math.isfinite(float(strength)):
            raise ValueError("force strength must be finite")
        self._forces.append({
            "type": ft, "strength": float(strength),
            "direction": tuple(float(v) for v in direction),
            "center": tuple(float(v) for v in center),
            "axis": tuple(float(v) for v in axis),
            "radius": float(radius), "scale": float(scale), "seed": int(seed),
        })

    def _apply_forces(self, grids, dt: float) -> None:
        """Add each registered body force to the face velocities (like gravity),
        using solver-local coordinates that respect the active sparse window."""
        if not self._forces:
            return
        xp = self.be.xp
        origin = self._frame_origin_cells
        accel = None
        for f in self._forces:
            ft = f["type"]
            if ft == "DIRECTIONAL":
                a = forces.directional_accel(
                    xp, self.shape, self.p.dx, f["direction"], f["strength"])
            elif ft == "VORTEX":
                a = forces.vortex_accel(
                    xp, self.shape, self.p.dx, f["center"], f["axis"],
                    f["strength"], f["radius"], origin)
            elif ft == "CONFINEMENT":
                a = forces.confinement_accel(
                    xp, self.p.dx, grids, f["strength"])
            else:
                a = forces.turbulence_accel(
                    xp, self.shape, self.p.dx, f["strength"], f["scale"],
                    self.time, f["seed"], origin)
            if a is None:
                continue
            accel = a if accel is None else accel + a
        if accel is None:
            return
        # Cell-centred acceleration -> face acceleration -> velocity increment.
        au = xp.zeros_like(grids["u"])
        au[1:-1] = 0.5 * (accel[1:, :, :, 0] + accel[:-1, :, :, 0])
        av = xp.zeros_like(grids["v"])
        av[:, 1:-1] = 0.5 * (accel[:, 1:, :, 1] + accel[:, :-1, :, 1])
        aw = xp.zeros_like(grids["w"])
        aw[:, :, 1:-1] = 0.5 * (accel[:, :, 1:, 2] + accel[:, :, :-1, 2])
        grids["u"] = grids["u"] + dt * au
        grids["v"] = grids["v"] + dt * av
        grids["w"] = grids["w"] + dt * aw

    def _validate_cell_mask(self, cell_mask: np.ndarray) -> np.ndarray:
        try:
            mask = np.asarray(cell_mask)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cell_mask must have solver shape {self.shape}"
            ) from exc
        if mask.shape != self.shape:
            raise ValueError(f"cell_mask must have solver shape {self.shape}")
        # Own the mask: CPU Backend.from_numpy intentionally avoids copies,
        # while CUDA upload inherently copies. Retaining caller storage here
        # would let later mutations change CPU inflows only.
        return np.array(mask, dtype=bool, order="C", copy=True)

    def _seed_cells(self, cells: np.ndarray, velocity: VelocityInput,
                    phase: float = 1.0, ppc: int | None = None,
                    source_id: int = 0) -> int:
        field = as_velocity_field(velocity)
        if len(cells) == 0:
            return 0
        xp = self.be.xp
        if ppc is None:
            ppc = self.p.particles_per_cell
        n = len(cells) * ppc
        jitter = self._rng.random((n, 3), dtype=np.float32)
        base = np.repeat(cells.astype(np.float32), ppc, axis=0)
        # 2x2x2 stratification when ppc == 8 keeps the initial field even.
        if ppc == 8:
            sub = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                           dtype=np.float32)
            sub = np.tile(sub, (len(cells), 1))
            pts = (base + (sub + jitter) * 0.5) * self.p.dx
        else:
            pts = (base + jitter) * self.p.dx
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        vel = field.sample(pts)
        self.pos = xp.concatenate([self.pos, self.be.from_numpy(pts)])
        self.vel = xp.concatenate([self.vel, self.be.from_numpy(vel)])
        self.dt_resid = xp.concatenate(
            [self.dt_resid, xp.zeros((n,), dtype=xp.float32)])
        self.phase = xp.concatenate(
            [self.phase, xp.full((n,), float(phase), dtype=xp.float32)])
        if self._use_apic:
            self.C = xp.concatenate(
                [self.C, xp.zeros((n, 3, 3), dtype=xp.float32)])
        self.age = xp.concatenate(
            [self.age, xp.zeros((n,), dtype=xp.float32)])
        self.source_id = xp.concatenate(
            [self.source_id, xp.full((n,), int(source_id), dtype=xp.int32)])
        self.particle_id = xp.concatenate([
            self.particle_id,
            xp.arange(self._next_particle_id,
                      self._next_particle_id + n, dtype=xp.int64),
        ])
        self._next_particle_id += n
        return n

    def add_gas_mask(
        self,
        cell_mask: np.ndarray,
        velocity: VelocityInput = (0.0, 0.0, 0.0),
    ) -> int:
        """Seed gas particles (phase = 0) into every masked cell.

        A no-op unless ``two_phase`` is enabled, so callers may seed gas
        unconditionally."""
        if not self.p.two_phase:
            return 0
        mask = self._validate_cell_mask(cell_mask)
        cells = np.argwhere(mask)
        sid = self._next_source
        self._next_source += 1
        return self._seed_cells(cells, velocity, phase=0.0,
                                ppc=self.p.gas_particles_per_cell,
                                source_id=sid)

    def fill_gas(self) -> int:
        """Fill every non-solid cell not already occupied by liquid with gas
        particles (two-phase only).  Call after all liquid/inflow seeding."""
        if not self.p.two_phase:
            return 0
        occupied = self._cell_counts() > 0.5
        solid = self.sdf < 0.0
        free = self.be.to_numpy(~occupied & ~solid)
        return self.add_gas_mask(free)

    def _cell_counts(self):
        """Per-cell particle occupancy count (device grid)."""
        xp = self.be.xp
        nx, ny, nz = self.shape
        counts = xp.zeros((nx * ny * nz,), dtype=xp.float32)
        if self.pos.shape[0]:
            idx = xp.clip((self.pos / self.p.dx).astype(xp.int32), 0,
                          xp.asarray([nx - 1, ny - 1, nz - 1]))
            flat = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
            self.be.scatter_add(counts, flat, xp.ones_like(flat, dtype=xp.float32))
        return counts.reshape(self.shape)

    # ------------------------------------------------------------- deposition

    def _jitter_gamma(self, dt: float, phi_gate=None):
        """Per-particle jitter strength gamma_p in [0, 1] (Eq. 10, Sec 3.10).

        Called at the jitter draw (end of a substep, with that substep's dt)
        and again during the next substep's P2G (with the committed dt_prev)
        by the exact temporal-weight normalization.  Both calls see the same
        velocities because nothing between the draw and the next deposit
        modifies self.vel; freshly seeded particles that never drew get a
        velocity-recomputed value for exactly one substep (bounded by the
        legacy error, self-healing after their first draw).
        """

        xp = self.be.xp
        p = self.p
        gamma = p.jitter_strength * xp.ones(
            (self.pos.shape[0],), dtype=xp.float32)
        if p.adaptive_gamma:
            local_cfl = _norm_rows(xp, self.vel) * dt / p.dx
            gate = kernels.smoothstep(xp, 0.0, 1.0, local_cfl)
            if phi_gate is not None:
                # Surface mode (CALM-M3): interiorness ADDS to the speed
                # gate and is clipped, so it can only restore jitter, never
                # remove it.  phi_gate has solid-occluded cells forced to
                # one so wall-adjacent bulk behaves like deep bulk.
                interior = kernels.smoothstep(
                    xp, 0.85, 0.98,
                    self._sample_cells(phi_gate, self.pos))
                gate = xp.clip(gate + interior, 0.0, 1.0)
            gamma = gamma * gate
        return gamma


    def _deformation_gamma(self, dt: float, grids, solid_c):
        """CALM-M4 deformation-aware jitter strength (speculative tier).

        Activity per particle: max of the normal-displacement CFL
        |v_p . n_hat| dt/dx, the normal strain rate |n . D . n| dt, and the
        grid-frame unsteadiness ||u_now - u_prev|| dt/dx; the surface-mode
        interiorness term is added and the gate clipped to one.  The
        SYMMETRIC strain-rate tensor is load-bearing: the full
        velocity-gradient norm reads sqrt(2) omega under solid-body
        rotation, which would keep gamma high on exactly the stirred-pool
        surface this mode exists to damp.  Unsteadiness compares against
        the previous substep's committed grids; after a restore or a
        sparse-window shape change the term is zero for one substep, so
        deformation-mode resume is deterministic but not bit-exact for
        that single substep (documented; default-off mode).
        """

        xp = self.be.xp
        p = self.p
        dx = p.dx
        phi_gate = xp.where(solid_c, xp.float32(1.0), grids["c_phi"])
        phi_s = surface_tension.smooth_phase(xp, phi_gate, 2)
        gx, gy, gz = xp.gradient(phi_s, dx)
        mag = xp.sqrt(gx * gx + gy * gy + gz * gz)
        inv = 1.0 / xp.maximum(mag, 1e-12)
        nx_, ny_, nz_ = gx * inv, gy * inv, gz * inv

        uc = 0.5 * (grids["u"][1:, :, :] + grids["u"][:-1, :, :])
        vc = 0.5 * (grids["v"][:, 1:, :] + grids["v"][:, :-1, :])
        wc = 0.5 * (grids["w"][:, :, 1:] + grids["w"][:, :, :-1])
        dux, duy, duz = xp.gradient(uc, dx)
        dvx, dvy, dvz = xp.gradient(vc, dx)
        dwx, dwy, dwz = xp.gradient(wc, dx)
        ndn = xp.abs(
            nx_ * nx_ * dux + ny_ * ny_ * dvy + nz_ * nz_ * dwz
            + nx_ * ny_ * (duy + dvx)
            + nx_ * nz_ * (duz + dwx)
            + ny_ * nz_ * (dvz + dwy))

        unsteady = xp.zeros_like(uc)
        prev = self._grids
        if (prev and prev.get("u") is not None
                and prev["u"].shape == grids["u"].shape):
            pu = 0.5 * (prev["u"][1:, :, :] + prev["u"][:-1, :, :])
            pv = 0.5 * (prev["v"][:, 1:, :] + prev["v"][:, :-1, :])
            pw = 0.5 * (prev["w"][:, :, 1:] + prev["w"][:, :, :-1])
            unsteady = xp.sqrt(
                (uc - pu) ** 2 + (vc - pv) ** 2 + (wc - pw) ** 2)

        # Normal displacement measured as PHASE FLUX |v . grad phi_s| dt
        # rather than v . n_hat: the normalized direction is noise where
        # the gradient is small (jet-front caps measured gamma 0.59 from
        # direction noise), while the unnormalized flux is zero for
        # tangential flow (the river case), zero in the interior, and
        # large exactly when a particle crosses phase contours.  PHI_STEP
        # is the phase change per substep considered fully active.
        phi_step = 0.25
        ghx = self._sample_cells(gx.astype(xp.float32), self.pos)
        ghy = self._sample_cells(gy.astype(xp.float32), self.pos)
        ghz = self._sample_cells(gz.astype(xp.float32), self.pos)
        term1 = xp.abs(
            self.vel[:, 0] * ghx + self.vel[:, 1] * ghy
            + self.vel[:, 2] * ghz) * (dt / phi_step)
        term2 = self._sample_cells(ndn.astype(xp.float32), self.pos) * dt
        term3 = self._sample_cells(
            unsteady.astype(xp.float32), self.pos) * (dt / dx)
        activity = xp.maximum(term1, xp.maximum(term2, term3))
        # Interiorness reads the SMOOTHED phase: raw c_phi carries
        # Monte-Carlo density holes (measured: 0.70 cells in the middle of
        # a uniformly filled box at ppc 2) that the (0.85, 0.98) band on
        # the raw field misreads as near-surface.  The smoothed field sits
        # at 0.93+ in true interior and ~0.3-0.6 across a real interface,
        # so a slightly lower band separates them cleanly.
        interior = kernels.smoothstep(
            xp, 0.80, 0.92,
            self._sample_cells(phi_s.astype(xp.float32), self.pos))
        gate = xp.clip(
            kernels.smoothstep(xp, 0.0, 1.0, activity) + interior,
            0.0, 1.0)
        return (p.jitter_strength * gate).astype(xp.float32)

    def _calibrate_m0(self) -> float:
        """Reference mass: expected accumulator value for a uniformly filled
        patch with ppc particles per cell and tau ~ U(-1/2, 1/2) (Sec 3.6).

        Both the separable spatial kernel and temporal kernel are analytically
        normalized to unit integral. For a uniform particle number density of
        ``ppc`` per cell, the expected accumulator is therefore exactly
        ``ppc``. The previous eight-sample Monte Carlo calibration introduced
        a large, method-dependent error at low PPC (despite using the same
        physical particle density), which confounded phase comparisons.
        """
        return float(self.p.particles_per_cell)

    def _p2g(self, dt_prev: float, force_instantaneous: bool = False):
        """4D->3D particle-to-grid transfer (Eq. 8-9). Returns face grids.

        In two-phase mode, ``*_m``/``*_ml`` remain volume-normalized sampling
        support for validity, activity, and liquid volume fraction. Face
        momentum and its denominator instead use phase-density mass weights.
        """
        xp = self.be.xp
        p = self.p
        nx, ny, nz = self.shape

        gp = self.pos / p.dx  # positions in grid units
        # No clipping: W_T is zero outside the slab on BOTH sides, so
        # out-of-slab samples (possible after abrupt adaptive-dt changes)
        # correctly receive zero weight instead of the clipped peak weight.
        theta = -self.dt_resid / max(dt_prev, 1e-12)
        if p.st_enabled and not force_instantaneous:
            wt = kernels.w_temporal(xp, theta)
            if p.exact_temporal_norm:
                # Exact Sec 3.10 conditioning (roadmap NORM-M1): dividing by
                # each particle's own mean weight mu(gamma_p) restores
                # E[wt] = 1 at every gamma.  This removes the paper's
                # accepted mixed-gamma face-velocity bias (up to ~2 percent
                # of the local velocity contrast toward fast particles) and
                # the up-to-3.9-percent phi_st depression on attenuated calm
                # surfaces.  gamma_p is recomputed from unmodified state; see
                # the draw-site contract comment.
                if p.gamma_mode != "speed":
                    # Surface/deformation gamma was persisted at the draw
                    # (its grid inputs no longer exist).  Fresh-seeded
                    # particles that never drew get the full-jitter divisor
                    # for exactly one substep: bounded in [1, 1024/945],
                    # self-healing after their first draw.
                    gamma = self._gamma_prev
                    n_now = theta.shape[0]
                    if gamma is None:
                        gamma = xp.full(
                            (n_now,), p.jitter_strength, dtype=xp.float32)
                    elif gamma.shape[0] < n_now:
                        gamma = xp.concatenate([gamma, xp.full(
                            (n_now - gamma.shape[0],), p.jitter_strength,
                            dtype=xp.float32)])
                    elif gamma.shape[0] > n_now:
                        gamma = gamma[:n_now]
                else:
                    gamma = self._jitter_gamma(dt_prev)
                wt = wt / kernels.w_temporal_mean(xp, gamma)
            wt = wt.astype(xp.float32)
        else:
            # Instantaneous deposition (st_enabled off, or the reflected
            # scheme's mid-step P2G): wt = 1 is m0-consistent because m0
            # is calibrated to E[m] with E[W_T] = 1 under uniform tau, and
            # has strictly less temporal variance -- no recalibration.
            wt = xp.ones_like(theta)

        shapes = {"u": (nx + 1, ny, nz), "v": (nx, ny + 1, nz),
                  "w": (nx, ny, nz + 1), "c": (nx, ny, nz)}
        vel_axis = {"u": 0, "v": 1, "w": 2}
        apic = self._use_apic
        two = p.two_phase
        phase = self.phase
        if two:
            # Normalize by one liquid particle's volume/mass. The omitted
            # common factors (cell volume / liquid PPC, and liquid density)
            # cancel in both volume fractions and velocity averages.
            gas_volume_weight = (
                float(p.particles_per_cell) / p.gas_particles_per_cell)
            gas_density_ratio = p.rho_gas / p.rho
            gas_mass_weight = gas_volume_weight * gas_density_ratio
            particle_volume_weight = (
                phase + (1.0 - phase) * gas_volume_weight)
            particle_mass_weight = (
                phase + (1.0 - phase) * gas_mass_weight)

        grids = {}
        for g, off in _OFFSETS.items():
            sh = shapes[g]
            volume = xp.zeros(sh, dtype=xp.float32).ravel()
            liquid_volume = (
                xp.zeros(sh, dtype=xp.float32).ravel() if two else None)
            mom = (xp.zeros(sh, dtype=xp.float32).ravel()
                   if g != "c" else None)
            face_mass = (
                xp.zeros(sh, dtype=xp.float32).ravel()
                if two and mom is not None else None)
            axis = vel_axis.get(g)
            xi = gp - self._offsets_dev[g]
            base = xp.floor(xi).astype(xp.int32)
            frac = xi - base
            for (di, dj, dk) in _TAPS:
                w = (kernels.w_spatial_1d(xp, frac[:, 0] - di)
                     * kernels.w_spatial_1d(xp, frac[:, 1] - dj)
                     * kernels.w_spatial_1d(xp, frac[:, 2] - dk) * wt)
                ii = base[:, 0] + di
                jj = base[:, 1] + dj
                kk = base[:, 2] + dk
                ok = ((ii >= 0) & (ii < sh[0]) & (jj >= 0) & (jj < sh[1])
                      & (kk >= 0) & (kk < sh[2]))
                w = xp.where(ok, w, 0.0).astype(xp.float32)
                flat = ((xp.clip(ii, 0, sh[0] - 1) * sh[1]
                         + xp.clip(jj, 0, sh[1] - 1)) * sh[2]
                        + xp.clip(kk, 0, sh[2] - 1))
                weighted_volume = w * particle_volume_weight if two else w
                self.be.scatter_add(volume, flat, weighted_volume)
                if liquid_volume is not None:
                    self.be.scatter_add(
                        liquid_volume, flat, weighted_volume * phase)
                if mom is not None:
                    vel_a = self.vel[:, axis]
                    if apic:
                        # C.(x_f - x_p): displacement (node - particle) in grid
                        # units is (d - frac) per axis, scaled to world by dx.
                        rx = (di - frac[:, 0])
                        ry = (dj - frac[:, 1])
                        rz = (dk - frac[:, 2])
                        vel_a = vel_a + p.dx * (self.C[:, axis, 0] * rx
                                                + self.C[:, axis, 1] * ry
                                                + self.C[:, axis, 2] * rz)
                    momentum_weight = (
                        w * particle_mass_weight if two else w)
                    if face_mass is not None:
                        self.be.scatter_add(face_mass, flat, momentum_weight)
                    self.be.scatter_add(mom, flat, momentum_weight * vel_a)
            grids[g + "_m"] = volume.reshape(sh)
            if liquid_volume is not None:
                grids[g + "_ml"] = liquid_volume.reshape(sh)
            if mom is not None:
                grids[g + "_p"] = mom.reshape(sh)
            if face_mass is not None:
                # Volume support controls validity; physical mass controls the
                # velocity average. Finalize now so this transient accumulator
                # can be reused for the next face grid instead of retaining
                # three production-sized arrays.
                valid = grids[g + "_m"] > p.eps_m
                xp.maximum(
                    face_mass, np.finfo(np.float32).tiny, out=face_mass)
                grids[g] = xp.where(
                    valid,
                    grids[g + "_p"]
                    / face_mass.reshape(sh),
                    0.0,
                )
                grids[g + "_valid"] = valid
                grids[g + "_phi"] = xp.clip(
                    grids[g + "_ml"]
                    / xp.maximum(grids[g + "_m"], p.eps_m),
                    0.0,
                    1.0,
                )
                face_mass = None

        if not two:
            # Keep the historical free-surface transfer arithmetic isolated:
            # its sampling weights serve as both momentum mass and phase mass.
            for g in ("u", "v", "w"):
                m = grids[g + "_m"]
                valid = m > p.eps_m
                grids[g] = xp.where(
                    valid,
                    grids[g + "_p"] / xp.maximum(m, p.eps_m),
                    0.0,
                )
                grids[g + "_valid"] = valid
                # Space-time phase field from the weight accumulators (Eq. 13):
                # phi = C(m / (eta_phi * m0)), C(x) = min(sqrt(x), 1).
                grids[g + "_phi"] = xp.minimum(
                    xp.sqrt(m / (p.eta_phi * self.m0)), 1.0)
            grids["c_phi"] = xp.minimum(
                xp.sqrt(grids["c_m"] / (p.eta_phi * self.m0)), 1.0)
        else:
            # Eq. 7: liquid volume support over total volume support. The
            # common liquid-particle normalization cancels in this ratio.
            grids["c_phi"] = xp.clip(
                grids["c_ml"] / xp.maximum(grids["c_m"], p.eps_m),
                0.0,
                1.0,
            )
        return grids

    # ------------------------------------------------------------- grid utils

    def _solid_face_apertures(self):
        """Return open-area fractions on MAC faces and fully-solid cells.

        The binary fallback deliberately matches the original solver: domain
        faces are closed and an internal face is closed when either adjacent
        cell-centred SDF sample is negative.  With a node SDF, cut-cell area
        fractions are computed geometrically; domain faces remain closed.
        """
        if self._solid_faces is not None:
            return self._solid_faces
        xp = self.be.xp
        nx, ny, nz = self.shape

        if self._solid_node_sdf is None:
            solid_c = self.sdf < 0.0
            alpha_u = xp.zeros((nx + 1, ny, nz), dtype=xp.float32)
            alpha_u[1:-1] = (~(solid_c[1:] | solid_c[:-1])).astype(
                xp.float32)
            alpha_v = xp.zeros((nx, ny + 1, nz), dtype=xp.float32)
            alpha_v[:, 1:-1] = (~(solid_c[:, 1:] | solid_c[:, :-1])).astype(
                xp.float32)
            alpha_w = xp.zeros((nx, ny, nz + 1), dtype=xp.float32)
            alpha_w[:, :, 1:-1] = (
                ~(solid_c[:, :, 1:] | solid_c[:, :, :-1])
            ).astype(xp.float32)
            exterior = (
                (~solid_c[0]).astype(xp.float32),
                (~solid_c[-1]).astype(xp.float32),
                (~solid_c[:, 0]).astype(xp.float32),
                (~solid_c[:, -1]).astype(xp.float32),
                (~solid_c[:, :, 0]).astype(xp.float32),
                (~solid_c[:, :, -1]).astype(xp.float32),
            )
        else:
            alpha_u, alpha_v, alpha_w = (
                apertures.face_apertures_from_node_sdf(
                    self._solid_node_sdf, array_module=xp)
            )
            solid_c = apertures.solid_cells_from_node_sdf(
                self._solid_node_sdf, array_module=xp)
            alpha_u = xp.clip(alpha_u, 0.0, 1.0).astype(xp.float32)
            alpha_v = xp.clip(alpha_v, 0.0, 1.0).astype(xp.float32)
            alpha_w = xp.clip(alpha_w, 0.0, 1.0).astype(xp.float32)
            exterior = (
                alpha_u[0].copy(),
                alpha_u[-1].copy(),
                alpha_v[:, 0].copy(),
                alpha_v[:, -1].copy(),
                alpha_w[:, :, 0].copy(),
                alpha_w[:, :, -1].copy(),
            )

            # The simulation domain is a closed box even when every node on
            # an exterior face is outside the embedded solid.
            alpha_u[0] = 0.0
            alpha_u[-1] = 0.0
            alpha_v[:, 0] = 0.0
            alpha_v[:, -1] = 0.0
            alpha_w[:, :, 0] = 0.0
            alpha_w[:, :, -1] = 0.0

        self._solid_exterior_apertures = exterior
        self._solid_faces = (alpha_u, alpha_v, alpha_w, solid_c)
        return self._solid_faces

    def _solid_face_masks(self):
        """Compatibility view of the fully blocked solid faces."""
        alpha_u, alpha_v, alpha_w, solid_c = self._solid_face_apertures()
        return alpha_u <= 0.0, alpha_v <= 0.0, alpha_w <= 0.0, solid_c

    def _solid_face_vel(self):
        """(u,v,w) face grids of the prescribed solid velocity, or 0-scalars.

        Interior faces take the average of the two adjacent cell solid
        velocities; the static container walls stay at zero."""
        xp = self.be.xp
        nx, ny, nz = self.shape
        if self.solid_vel is None:
            z = xp.float32(0.0)
            return z, z, z
        sv_c = self.solid_vel  # (nx, ny, nz, 3)
        us = xp.zeros((nx + 1, ny, nz), dtype=xp.float32)
        us[1:-1] = 0.5 * (sv_c[1:, :, :, 0] + sv_c[:-1, :, :, 0])
        vs = xp.zeros((nx, ny + 1, nz), dtype=xp.float32)
        vs[:, 1:-1] = 0.5 * (sv_c[:, 1:, :, 1] + sv_c[:, :-1, :, 1])
        ws = xp.zeros((nx, ny, nz + 1), dtype=xp.float32)
        ws[:, :, 1:-1] = 0.5 * (sv_c[:, :, 1:, 2] + sv_c[:, :, :-1, 2])
        return us, vs, ws

    def _pressure_face_masks(self):
        """Full MAC masks for exterior faces opened by PRESSURE outflows."""
        if self._pressure_outflow_faces is not None:
            return self._pressure_outflow_faces
        xp = self.be.xp
        nx, ny, nz = self.shape
        _, _, _, solid_c = self._solid_face_apertures()
        exterior = self._solid_exterior_apertures
        face_u = xp.zeros((nx + 1, ny, nz), dtype=bool)
        face_v = xp.zeros((nx, ny + 1, nz), dtype=bool)
        face_w = xp.zeros((nx, ny, nz + 1), dtype=bool)
        side_masks = self._pressure_outflow_side_masks
        if side_masks["x_min"] is not None:
            face_u[0] = (side_masks["x_min"] & (~solid_c[0])
                         & (exterior[0] > 0.0))
        if side_masks["x_max"] is not None:
            face_u[-1] = (side_masks["x_max"] & (~solid_c[-1])
                          & (exterior[1] > 0.0))
        if side_masks["y_min"] is not None:
            face_v[:, 0] = (side_masks["y_min"] & (~solid_c[:, 0])
                            & (exterior[2] > 0.0))
        if side_masks["y_max"] is not None:
            face_v[:, -1] = (side_masks["y_max"] & (~solid_c[:, -1])
                             & (exterior[3] > 0.0))
        if side_masks["z_min"] is not None:
            face_w[:, :, 0] = (side_masks["z_min"] & (~solid_c[:, :, 0])
                               & (exterior[4] > 0.0))
        if side_masks["z_max"] is not None:
            face_w[:, :, -1] = (side_masks["z_max"] & (~solid_c[:, :, -1])
                                & (exterior[5] > 0.0))
        self._pressure_outflow_faces = (face_u, face_v, face_w)
        return self._pressure_outflow_faces

    def _active_face_apertures(self):
        """Solid apertures with explicitly opened pressure-outflow faces."""
        xp = self.be.xp
        alpha_u, alpha_v, alpha_w, solid_c = self._solid_face_apertures()
        if not self._has_pressure_outflow:
            return alpha_u, alpha_v, alpha_w, solid_c
        pressure_u, pressure_v, pressure_w = self._pressure_face_masks()
        if not self.be.is_gpu and not bool(
            xp.any(pressure_u) | xp.any(pressure_v) | xp.any(pressure_w)
        ):
            return alpha_u, alpha_v, alpha_w, solid_c
        exterior = self._solid_exterior_apertures
        active_u = alpha_u.copy()
        active_v = alpha_v.copy()
        active_w = alpha_w.copy()
        active_u[0] = xp.where(pressure_u[0], exterior[0], active_u[0])
        active_u[-1] = xp.where(pressure_u[-1], exterior[1], active_u[-1])
        active_v[:, 0] = xp.where(
            pressure_v[:, 0], exterior[2], active_v[:, 0]
        )
        active_v[:, -1] = xp.where(
            pressure_v[:, -1], exterior[3], active_v[:, -1]
        )
        active_w[:, :, 0] = xp.where(
            pressure_w[:, :, 0], exterior[4], active_w[:, :, 0]
        )
        active_w[:, :, -1] = xp.where(
            pressure_w[:, :, -1], exterior[5], active_w[:, :, -1]
        )
        return active_u, active_v, active_w, solid_c

    def outflow_stats(self) -> dict:
        """Return registered mask/open-face counts and cumulative removals."""
        if self._outflow_geometry_stats_cache is None:
            if not (self._has_volume_outflow or self._has_pressure_outflow):
                counts = [0] * 8
            else:
                xp = self.be.xp
                zero = xp.asarray(0, dtype=xp.int64)
                if self._has_pressure_outflow:
                    face_u, face_v, face_w = self._pressure_face_masks()
                    face_counts = (
                        face_u[0].sum(),
                        face_u[-1].sum(),
                        face_v[:, 0].sum(),
                        face_v[:, -1].sum(),
                        face_w[:, :, 0].sum(),
                        face_w[:, :, -1].sum(),
                    )
                else:
                    face_counts = (zero,) * 6
                device_counts = xp.stack((
                    self._volume_outflow.sum()
                    if self._has_volume_outflow else zero,
                    self._pressure_outflow.sum()
                    if self._has_pressure_outflow else zero,
                    *face_counts,
                ))
                counts = [
                    int(value)
                    for value in self.be.to_numpy(device_counts).tolist()
                ]
            side_counts = dict(zip(
                ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
                counts[2:],
                strict=True,
            ))
            self._outflow_geometry_stats_cache = {
                "volume_cell_count": counts[0],
                "pressure_cell_count": counts[1],
                "pressure_open_face_count": sum(side_counts.values()),
                "pressure_open_face_counts": side_counts,
            }
        geometry = self._outflow_geometry_stats_cache
        return {
            **geometry,
            "pressure_open_face_counts": dict(
                geometry["pressure_open_face_counts"]
            ),
            "particles_removed_total": self._outflow_removed_total,
            "volume_outflow_removed_total": (
                self._volume_outflow_removed_total
            ),
            "pressure_outflow_removed_total": (
                self._pressure_outflow_removed_total
            ),
        }

    def _apply_outflow_filter(
        self,
        positions,
        volume_removed,
        pressure_removed,
        stats: FrameStats | None = None,
    ) -> dict[str, int]:
        """Apply one shared keep mask to every particle-state array."""
        xp = self.be.xp
        self._reconcile_particle_attrs()
        keep = ~(volume_removed | pressure_removed)
        counts = self.be.to_numpy(xp.stack((
            volume_removed.sum(), pressure_removed.sum()
        )))
        volume_count, pressure_count = (int(value) for value in counts.tolist())
        removed = volume_count + pressure_count
        self.pos = positions[keep]
        self.vel = self.vel[keep]
        self.dt_resid = self.dt_resid[keep]
        self.phase = self.phase[keep]
        if self.C.shape[0]:
            self.C = self.C[keep]
        self.age = self.age[keep]
        self.source_id = self.source_id[keep]
        self.particle_id = self.particle_id[keep]
        if (self._gamma_prev is not None
                and self._gamma_prev.shape[0] == keep.shape[0]):
            self._gamma_prev = self._gamma_prev[keep]
        if stats is not None:
            stats.volume_outflow_removed += volume_count
            stats.pressure_outflow_removed += pressure_count
            stats.particles_removed += removed
        self._volume_outflow_removed_total += volume_count
        self._pressure_outflow_removed_total += pressure_count
        self._outflow_removed_total += removed
        return {
            "particles_removed": removed,
            "volume_outflow_removed": volume_count,
            "pressure_outflow_removed": pressure_count,
        }

    def cull_outflows(self) -> dict[str, int]:
        """Immediately remove particles already captured by outflow regions.

        This is intended for setup-time cleanup before exporting the initial
        frame. VOLUME masks take precedence; remaining particles already
        outside the domain are removed only when every crossed side is an
        opened PRESSURE face. The operation is idempotent and updates the same
        cumulative counters as stepping.
        """
        if not (self._has_volume_outflow or self._has_pressure_outflow):
            return {
                "particles_removed": 0,
                "volume_outflow_removed": 0,
                "pressure_outflow_removed": 0,
            }
        xp = self.be.xp
        zero = xp.zeros((self.pos.shape[0],), dtype=bool)
        volume_removed = (
            self._positions_in_mask(self._volume_outflow, self.pos)
            if self._has_volume_outflow
            else zero
        )
        pressure_removed = (
            (~volume_removed) & self._pressure_exit_allowed(self.pos)
            if self._has_pressure_outflow
            else zero
        )
        return self._apply_outflow_filter(
            self.pos, volume_removed, pressure_removed
        )

    def solid_aperture_stats(self) -> dict[str, int | str]:
        """Summarize the active solid-boundary representation.

        Counts cover all three MAC face grids, including closed simulation
        domain faces, and therefore partition ``total_face_count`` exactly.
        This call may synchronize a GPU backend and is intended for run
        metadata, not the stepping hot path.
        """
        xp = self.be.xp
        alpha_u, alpha_v, alpha_w, _ = self._solid_face_apertures()

        def count(predicate) -> int:
            value = self.be.to_numpy(xp.asarray(predicate.sum()))
            return int(value.item())

        faces = (alpha_u, alpha_v, alpha_w)
        total = sum(int(alpha.size) for alpha in faces)
        blocked = sum(count(alpha <= 0.0) for alpha in faces)
        fractional = sum(
            count((alpha > 0.0) & (alpha < 1.0)) for alpha in faces
        )
        opened = total - blocked - fractional
        return {
            "model": (
                "fractional_node_sdf"
                if self._solid_node_sdf is not None
                else "binary_cell_center"
            ),
            "total_face_count": total,
            "blocked_face_count": blocked,
            "fractional_face_count": fractional,
            "open_face_count": opened,
        }

    def _extrapolate(self, u, valid, layers: int, allowed=None):
        """Propagate velocities into invalid faces by neighbour averaging.

        Shifted-slice accumulation: no wrap-around to correct and roughly a
        third of the kernel launches of an xp.roll formulation.  ``allowed``
        prevents propagation onto or through zero-aperture faces.
        """
        xp = self.be.xp
        if allowed is None:
            allowed = xp.ones_like(valid, dtype=bool)
        valid = valid & allowed
        for _ in range(layers):
            vf = valid.astype(u.dtype)
            uf = u * vf
            s = xp.zeros_like(u)
            c = xp.zeros_like(u)
            s[:-1] += uf[1:]
            c[:-1] += vf[1:]
            s[1:] += uf[:-1]
            c[1:] += vf[:-1]
            s[:, :-1] += uf[:, 1:]
            c[:, :-1] += vf[:, 1:]
            s[:, 1:] += uf[:, :-1]
            c[:, 1:] += vf[:, :-1]
            s[:, :, :-1] += uf[:, :, 1:]
            c[:, :, :-1] += vf[:, :, 1:]
            s[:, :, 1:] += uf[:, :, :-1]
            c[:, :, 1:] += vf[:, :, :-1]
            newly = (~valid) & allowed & (c > 0)
            u = xp.where(newly, s / xp.maximum(c, 1.0), u)
            valid = valid | newly
        return u, valid

    def _sample_faces(self, grids, pos):
        """Trilinear G2P gather of the three face grids at positions."""
        xp = self.be.xp
        gp = pos / self.p.dx
        out = xp.zeros((pos.shape[0], 3), dtype=xp.float32)
        for ax, g in enumerate(("u", "v", "w")):
            arr = grids[g]
            sh = arr.shape
            xi = gp - self._offsets_dev[g]
            base = xp.floor(xi).astype(xp.int32)
            frac = (xi - base).astype(xp.float32)
            val = xp.zeros((pos.shape[0],), dtype=xp.float32)
            for (di, dj, dk) in _TAPS:
                wx = frac[:, 0] * di + (1 - frac[:, 0]) * (1 - di)
                wy = frac[:, 1] * dj + (1 - frac[:, 1]) * (1 - dj)
                wz = frac[:, 2] * dk + (1 - frac[:, 2]) * (1 - dk)
                ii = xp.clip(base[:, 0] + di, 0, sh[0] - 1)
                jj = xp.clip(base[:, 1] + dj, 0, sh[1] - 1)
                kk = xp.clip(base[:, 2] + dk, 0, sh[2] - 1)
                val += wx * wy * wz * arr[ii, jj, kk]
            out[:, ax] = val
        return out

    def _g2p_apic(self, grids, pos, u_new):
        """APIC grid-to-particle (Jiang et al. 2015) on the MAC grid.

        Returns interpolated velocities (== ``u_new``) and the reconstructed
        affine matrices C.  Per axis a we form B_a = sum_i w_i u_i (x_i - x_p)
        and D_a = sum_i w_i (x_i - x_p)(x_i - x_p)^T with the same trilinear
        weights used for interpolation, then set the a-th row of C to
        D_a^{-1} B_a.  D_a is Tikhonov-regularised so isolated or node-aligned
        particles stay well-conditioned (no cupy.linalg)."""
        xp = self.be.xp
        n = pos.shape[0]
        gp = pos / self.p.dx
        reg = self.p.apic_reg * self.p.dx * self.p.dx
        C = xp.zeros((n, 3, 3), dtype=xp.float32)
        for axis, gname in enumerate(("u", "v", "w")):
            arr = grids[gname]
            sh = arr.shape
            xi = gp - self._offsets_dev[gname]
            base = xp.floor(xi).astype(xp.int32)
            frac = (xi - base).astype(xp.float32)
            b = xp.zeros((n, 3), dtype=xp.float32)
            D = xp.zeros((n, 3, 3), dtype=xp.float32)
            for (di, dj, dk) in _TAPS:
                wx = frac[:, 0] * di + (1 - frac[:, 0]) * (1 - di)
                wy = frac[:, 1] * dj + (1 - frac[:, 1]) * (1 - dj)
                wz = frac[:, 2] * dk + (1 - frac[:, 2]) * (1 - dk)
                w = wx * wy * wz
                ii = xp.clip(base[:, 0] + di, 0, sh[0] - 1)
                jj = xp.clip(base[:, 1] + dj, 0, sh[1] - 1)
                kk = xp.clip(base[:, 2] + dk, 0, sh[2] - 1)
                uval = arr[ii, jj, kk]
                rx = (di - frac[:, 0]) * self.p.dx
                ry = (dj - frac[:, 1]) * self.p.dx
                rz = (dk - frac[:, 2]) * self.p.dx
                wu = w * uval
                b[:, 0] += wu * rx
                b[:, 1] += wu * ry
                b[:, 2] += wu * rz
                D[:, 0, 0] += w * rx * rx
                D[:, 0, 1] += w * rx * ry
                D[:, 0, 2] += w * rx * rz
                D[:, 1, 1] += w * ry * ry
                D[:, 1, 2] += w * ry * rz
                D[:, 2, 2] += w * rz * rz
            D[:, 1, 0] = D[:, 0, 1]
            D[:, 2, 0] = D[:, 0, 2]
            D[:, 2, 1] = D[:, 1, 2]
            for d in range(3):
                D[:, d, d] += reg
            Dinv = _inv3x3(xp, D)
            C[:, axis, 0] = (Dinv[:, 0, 0] * b[:, 0] + Dinv[:, 0, 1] * b[:, 1]
                             + Dinv[:, 0, 2] * b[:, 2])
            C[:, axis, 1] = (Dinv[:, 1, 0] * b[:, 0] + Dinv[:, 1, 1] * b[:, 1]
                             + Dinv[:, 1, 2] * b[:, 2])
            C[:, axis, 2] = (Dinv[:, 2, 0] * b[:, 0] + Dinv[:, 2, 1] * b[:, 1]
                             + Dinv[:, 2, 2] * b[:, 2])
        return u_new, C

    def _apply_sheeting(self) -> None:
        """Position-only anti-clumping (sheeting).

        Nudges particles a small clamped step down the local density gradient,
        spreading over-dense clumps and filling voids so thin sheets and
        splashes hold together.  Touches only positions, not velocity, so it
        adds no kinetic energy and cannot destabilise the sim."""
        if self.pos.shape[0] == 0:
            return
        xp = self.be.xp
        dx = self.p.dx
        # Smooth the (aliased) per-cell count before differentiating so the
        # gradient captures the macro dense->sparse trend, not per-cell noise.
        counts = surface_tension.smooth_phase(
            xp, self._cell_counts().astype(xp.float32), iters=2)
        gx, gy, gz = xp.gradient(counts, dx)
        g = xp.stack([self._sample_cells(gx, self.pos),
                      self._sample_cells(gy, self.pos),
                      self._sample_cells(gz, self.pos)], axis=1)
        ref = max(float(self.p.particles_per_cell), 1.0)
        disp = -(self.p.sheeting * dx * dx / ref) * g
        mag = _norm_rows(xp, disp)
        scale = xp.minimum(1.0, (0.3 * dx) / xp.maximum(mag, 1e-12))
        # Gate to genuinely over-dense interior: cells below the reference
        # count (surface, spray) get ~no push, so redistribution spreads bulk
        # clumps without inflating the free surface down its density cliff.
        local = self._sample_cells(counts, self.pos)
        excess = xp.clip((local - ref) / ref, 0.0, 1.0)
        weight = (scale * excess)[:, None]
        self.pos = self._clamp_domain(
            self._push_out_of_solids(self.pos + disp * weight))

    def _enforce_solid_velocity(self) -> None:
        """Remove the penetrating (inward) relative normal velocity of
        particles inside the solid band so animated obstacles push (rather than
        trap) the fluid, while still allowing free separation."""
        xp = self.be.xp
        if self.solid_vel is None or self._sdf_grad is None:
            return
        d = self._sample_cells(self.sdf, self.pos)
        near = d < (1.0 * self.p.dx)
        if not self.be.is_gpu and not bool(xp.any(near)):
            return
        gx = self._sample_cells(self._sdf_grad[0], self.pos)
        gy = self._sample_cells(self._sdf_grad[1], self.pos)
        gz = self._sample_cells(self._sdf_grad[2], self.pos)
        nrm = xp.stack([gx, gy, gz], axis=1)
        nrm = nrm / xp.maximum(_norm_rows(xp, nrm)[:, None], 1e-9)
        us = xp.stack([self._sample_cells(self.solid_vel[..., i], self.pos)
                       for i in range(3)], axis=1)
        vrel = self.vel - us
        vn_s = (vrel * nrm).sum(axis=1)
        penetrating = vn_s < 0.0
        vn = xp.where(penetrating[:, None], vn_s[:, None] * nrm, 0.0)
        self.vel = xp.where(near[:, None], self.vel - vn, self.vel)

    def _sample_cells(self, arr, pos):
        """Trilinear sample of a cell-centred grid at positions."""
        xp = self.be.xp
        sh = arr.shape
        xi = pos / self.p.dx - 0.5
        base = xp.floor(xi).astype(xp.int32)
        frac = (xi - base).astype(xp.float32)
        val = xp.zeros((pos.shape[0],), dtype=xp.float32)
        for (di, dj, dk) in _TAPS:
            wx = frac[:, 0] * di + (1 - frac[:, 0]) * (1 - di)
            wy = frac[:, 1] * dj + (1 - frac[:, 1]) * (1 - dj)
            wz = frac[:, 2] * dk + (1 - frac[:, 2]) * (1 - dk)
            ii = xp.clip(base[:, 0] + di, 0, sh[0] - 1)
            jj = xp.clip(base[:, 1] + dj, 0, sh[1] - 1)
            kk = xp.clip(base[:, 2] + dk, 0, sh[2] - 1)
            val += wx * wy * wz * arr[ii, jj, kk]
        return val

    # -------------------------------------------------------------- advection

    def _push_out_of_solids(self, pos):
        xp = self.be.xp
        if self._sdf_grad is None:
            return pos
        d = self._sample_cells(self.sdf, pos)
        margin = 0.1 * self.p.dx
        need = d < margin
        # The early-out saves real time on CPU, but on GPU the any() forces
        # a blocking device sync inside every RK substep; compute through.
        if not self.be.is_gpu and not bool(xp.any(need)):
            return pos
        gx = self._sample_cells(self._sdf_grad[0], pos)
        gy = self._sample_cells(self._sdf_grad[1], pos)
        gz = self._sample_cells(self._sdf_grad[2], pos)
        g = xp.stack([gx, gy, gz], axis=1)
        norm = _norm_rows(xp, g)[:, None]
        n = g / xp.maximum(norm, 1e-9)
        push = xp.maximum(margin - d, 0.0)[:, None] * n
        return xp.where(need[:, None], pos + push, pos)

    def _clamp_domain(self, pos):
        return self.be.xp.clip(pos, self._clamp_lo, self._clamp_hi)

    def _grid_velocity_bound(self, grids):
        """Strict bound on the magnitude of the sampled MAC advector.

        Trilinear interpolation is a convex combination for each component,
        so the norm of any sampled velocity is bounded by the Euclidean norm
        of the three component-wise absolute maxima.
        """
        xp = self.be.xp
        maxima = [xp.max(xp.abs(grids[name])) for name in ("u", "v", "w")]
        return xp.sqrt(sum(value * value for value in maxima))

    def _positions_in_mask(self, mask, pos):
        xp = self.be.xp
        in_domain = xp.all((pos >= 0.0) & (pos < self._domain_size_dev), axis=1)
        hi = xp.asarray(self.shape, dtype=xp.int32) - 1
        indices = xp.clip((pos / self.p.dx).astype(xp.int32), 0, hi)
        selected = mask[indices[:, 0], indices[:, 1], indices[:, 2]]
        return in_domain & selected

    def _segments_hit_mask(self, mask, start, end):
        """Exact grid traversal for swept particle segments.

        A vectorized Amanatides-Woo traversal visits every positive-length
        voxel entered by each segment. The fixed iteration bound follows from
        the strict local-CFL displacement bound, avoiding a GPU synchronization
        to discover a data-dependent loop count.
        """
        xp = self.be.xp
        hit = self._positions_in_mask(mask, start) | self._positions_in_mask(
            mask, end
        )
        if start.shape[0] == 0:
            return hit
        delta = end - start
        moving = xp.abs(delta) > 1e-30
        safe_delta = xp.where(moving, delta, 1.0)
        direction = xp.sign(delta).astype(xp.int32)
        cell = xp.floor(start / self.p.dx).astype(xp.int32)
        next_boundary = xp.where(
            direction > 0,
            (cell + 1).astype(xp.float32) * self.p.dx,
            cell.astype(xp.float32) * self.p.dx,
        )
        beyond_segment = xp.asarray(2.0, dtype=xp.float32)
        t_max = xp.where(
            moving, (next_boundary - start) / safe_delta, beyond_segment
        ).astype(xp.float32)
        t_delta = xp.where(
            moving, self.p.dx / xp.abs(safe_delta), 0.0
        ).astype(xp.float32)
        hi = xp.asarray(self.shape, dtype=xp.int32) - 1
        max_crossings = int(math.ceil(math.sqrt(3.0) * self.p.cfl_local)) + 3
        for _ in range(max_crossings):
            t_next = xp.min(t_max, axis=1)
            active = (~hit) & (t_next <= 1.0 + 1e-6)
            tolerance = 1e-6 * xp.maximum(1.0, xp.abs(t_next))
            advance = active[:, None] & (
                xp.abs(t_max - t_next[:, None]) <= tolerance[:, None]
            )
            cell = cell + direction * advance.astype(xp.int32)
            inside = active & xp.all((cell >= 0) & (cell <= hi), axis=1)
            lookup = xp.clip(cell, 0, hi)
            entered = mask[lookup[:, 0], lookup[:, 1], lookup[:, 2]]
            hit = hit | (inside & entered)
            t_max = t_max + xp.where(advance, t_delta, 0.0)
        return hit

    def _pressure_exit_allowed(self, pos, start=None):
        """Whether every crossed domain side is an opened outflow face.

        During advection, tangential face indices are evaluated at the actual
        segment/boundary intersection. ``start=None`` supports setup-time
        culling of particles that are already outside.
        """
        xp = self.be.xp
        face_u, face_v, face_w = self._pressure_face_masks()
        nx, ny, nz = self.shape

        def boundary_cells(axis: int, boundary: float):
            if start is None:
                intersection = pos
            else:
                delta = pos - start
                denominator = delta[:, axis]
                safe = xp.where(xp.abs(denominator) > 1e-30, denominator, 1.0)
                t = (boundary - start[:, axis]) / safe
                intersection = start + t[:, None] * delta
            cell = xp.floor(intersection / self.p.dx).astype(xp.int32)
            return (
                xp.clip(cell[:, 0], 0, nx - 1),
                xp.clip(cell[:, 1], 0, ny - 1),
                xp.clip(cell[:, 2], 0, nz - 1),
            )

        low_x = pos[:, 0] < 0.0
        high_x = pos[:, 0] >= self.size[0]
        low_y = pos[:, 1] < 0.0
        high_y = pos[:, 1] >= self.size[1]
        low_z = pos[:, 2] < 0.0
        high_z = pos[:, 2] >= self.size[2]
        outside = low_x | high_x | low_y | high_y | low_z | high_z
        _, iy_x0, iz_x0 = boundary_cells(0, 0.0)
        _, iy_x1, iz_x1 = boundary_cells(0, self.size[0])
        ix_y0, _, iz_y0 = boundary_cells(1, 0.0)
        ix_y1, _, iz_y1 = boundary_cells(1, self.size[1])
        ix_z0, iy_z0, _ = boundary_cells(2, 0.0)
        ix_z1, iy_z1, _ = boundary_cells(2, self.size[2])
        blocked = (
            (low_x & (~face_u[0, iy_x0, iz_x0]))
            | (high_x & (~face_u[-1, iy_x1, iz_x1]))
            | (low_y & (~face_v[ix_y0, 0, iz_y0]))
            | (high_y & (~face_v[ix_y1, -1, iz_y1]))
            | (low_z & (~face_w[ix_z0, iy_z0, 0]))
            | (high_z & (~face_w[ix_z1, iy_z1, -1]))
        )
        return outside & (~blocked)

    def _advect(self, grids, pos, dt_act, *, track_outflows: bool = False):
        """Sub-stepped RK3 (Ralston) through the grid velocity field.

        ``dt_act`` is per-particle and may be negative (used for
        un-jittering).  When ``track_outflows`` is true, the returned tuple is
        ``(positions, volume_removed, pressure_removed)``; sink tests occur at
        every local-CFL substep before closed-domain clamping.
        """
        xp = self.be.xp
        p = self.p
        speed_bound = self._grid_velocity_bound(grids)
        nsub = xp.ceil(
            speed_bound * xp.abs(dt_act) / (p.dx * p.cfl_local)
        ).astype(xp.int32)
        # Never cap this count: a global cap can violate the documented local
        # CFL bound when large global CFL targets and temporal jitter combine.
        nsub = xp.maximum(nsub, 1)
        h = dt_act / nsub.astype(xp.float32)
        max_n = int(nsub.max()) if nsub.size else 0
        if not track_outflows:
            # Preserve the legacy allocation-free path when no sink
            # classification is requested (the normal no-outflow case).
            for s in range(max_n):
                he = xp.where(s < nsub, h, 0.0).astype(xp.float32)[:, None]
                k1 = self._sample_faces(grids, pos)
                k2 = self._sample_faces(grids, pos + 0.5 * he * k1)
                k3 = self._sample_faces(grids, pos + 0.75 * he * k2)
                pos = pos + he * (
                    2.0 * k1 + 3.0 * k2 + 4.0 * k3
                ) / 9.0
                pos = self._clamp_domain(self._push_out_of_solids(pos))
            return pos

        volume_removed = xp.zeros((pos.shape[0],), dtype=bool)
        pressure_removed = xp.zeros((pos.shape[0],), dtype=bool)
        alive = xp.ones((pos.shape[0],), dtype=bool)
        if self._has_volume_outflow and pos.shape[0]:
            volume_removed = self._positions_in_mask(
                self._volume_outflow, pos
            )
            alive = ~volume_removed
        for s in range(max_n):
            he = xp.where((s < nsub) & alive, h, 0.0).astype(xp.float32)[:, None]
            k1 = self._sample_faces(grids, pos)
            k2 = self._sample_faces(grids, pos + 0.5 * he * k1)
            k3 = self._sample_faces(grids, pos + 0.75 * he * k2)
            proposed = pos + he * (2.0 * k1 + 3.0 * k2 + 4.0 * k3) / 9.0
            if self._has_volume_outflow:
                hit_volume = alive & self._segments_hit_mask(
                    self._volume_outflow, pos, proposed
                )
                volume_removed = volume_removed | hit_volume
                alive = alive & (~hit_volume)
            collision_adjusted = self._push_out_of_solids(proposed)
            if self._has_pressure_outflow:
                exited = alive & self._pressure_exit_allowed(
                    collision_adjusted, start=pos
                )
                pressure_removed = pressure_removed | exited
                alive = alive & (~exited)
            constrained = self._clamp_domain(collision_adjusted)
            pos = xp.where(alive[:, None], constrained, collision_adjusted)
        return pos, volume_removed, pressure_removed

    # -------------------------------------------------- sparse active window

    def _band(self) -> int:
        """Halo, in cells, the fluid may reach in one step: the same velocity-
        extrapolation band advection is guaranteed to stay inside."""
        return int(math.ceil(2.0 * self.p.cfl_target)) + 2

    def _sparse_engaged(self) -> bool:
        # The window crops the solid/collision fields; the cut-cell node-SDF
        # apertures and outflow machinery are not windowed, so fall back to the
        # dense solve when any of them is active.
        return (self.p.sparse and not self._has_volume_outflow
                and not self._has_pressure_outflow
                and self._solid_node_sdf is None)

    def _active_window(self):
        """Block-aligned (lo, sub_shape) covering every particle plus a one-
        step halo, clamped to the domain (sparse production grid)."""
        xp = self.be.xp
        nx, ny, nz = self._full_shape
        gp = self.pos / self.p.dx
        lo_p = self.be.to_numpy(xp.floor(gp.min(axis=0))).astype(np.int64)
        hi_p = self.be.to_numpy(xp.ceil(gp.max(axis=0))).astype(np.int64)
        band = self._band()
        bs = max(int(self.p.block_size), 1)
        pad = int(self.p.sparse_pad) * bs
        full = np.array([nx, ny, nz], dtype=np.int64)
        lo = np.maximum(lo_p - band - pad, 0)
        hi = np.minimum(hi_p + band + pad, full)
        lo = (lo // bs) * bs
        hi = np.minimum(((hi + bs - 1) // bs) * bs, full)
        hi = np.maximum(hi, lo + 1)
        return lo, tuple(int(v) for v in (hi - lo))

    def _enter_window(self, lo, sub_shape):
        """Swap grid state to the cropped window frame (does not touch pos)."""
        xp = self.be.xp
        sl = (slice(lo[0], lo[0] + sub_shape[0]),
              slice(lo[1], lo[1] + sub_shape[1]),
              slice(lo[2], lo[2] + sub_shape[2]))
        saved = (self.shape, self.size, self.sdf, self._sdf_grad,
                 self.solid_vel, self._solid_faces,
                 self._solid_exterior_apertures, self._clamp_lo,
                 self._clamp_hi, self._domain_size_dev)
        self.shape = sub_shape
        self.size = tuple(n * self.p.dx for n in sub_shape)
        self.sdf = self.sdf[sl]
        self._sdf_grad = (None if self._sdf_grad is None
                          else tuple(g[sl] for g in self._sdf_grad))
        self.solid_vel = (None if self.solid_vel is None
                          else self.solid_vel[sl[0], sl[1], sl[2], :])
        self._solid_faces = None
        self._solid_exterior_apertures = None
        eps = 1e-3 * self.p.dx
        self._clamp_lo = xp.asarray([eps] * 3, dtype=xp.float32)
        self._clamp_hi = xp.asarray([n - eps for n in self.size],
                                    dtype=xp.float32)
        self._domain_size_dev = xp.asarray(self.size, dtype=xp.float32)
        # World-space forces need the window's cell offset to stay put.
        self._frame_origin_cells = np.asarray(lo, dtype=np.float64)
        return saved

    def _exit_window(self, saved):
        (self.shape, self.size, self.sdf, self._sdf_grad, self.solid_vel,
         self._solid_faces, self._solid_exterior_apertures, self._clamp_lo,
         self._clamp_hi, self._domain_size_dev) = saved
        self._frame_origin_cells = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------- step

    def _project(self, grids, dt, stats, alpha, open_faces, solid_vel,
                 inv_rho, active) -> None:
        """Pressure-project grids[u/v/w] in place (Eq. 14-16).

        The projected velocity is invariant to the ``dt`` used here: the
        pressure scales as 1/dt, the PPE coefficients as dt, and the
        correction as dt, so dt cancels in the corrected velocity.  This
        makes the method reusable for a second projection within one substep
        (advection-reflection, roadmap ENER-M2b) without recalibration.
        """

        xp = self.be.xp
        p = self.p
        alpha_u, alpha_v, alpha_w = alpha
        open_u, open_v, open_w = open_faces
        us_sol, vs_sol, ws_sol = solid_vel
        inv_rho_u, inv_rho_v, inv_rho_w = inv_rho

        # PPE coefficients are aperture-weighted, while the velocity update
        # itself is not: k_f = dt * alpha_f / rho_f, followed by
        # u_f <- u_f - dt/rho_f * grad(p) on every alpha_f > 0 face.
        kx = dt * alpha_u * inv_rho_u
        ky = dt * alpha_v * inv_rho_v
        kz = dt * alpha_w * inv_rho_w

        div = apertures.weighted_divergence(
            grids["u"], grids["v"], grids["w"],
            alpha_u, alpha_v, alpha_w, p.dx, array_module=xp,
        )
        if self.solid_vel is not None:
            # Moving-wall flux: the blocked fraction (1 - alpha) of each face
            # carries the solid velocity, so its divergence enters the RHS.
            div = div + apertures.weighted_divergence(
                us_sol, vs_sol, ws_sol,
                1.0 - alpha_u, 1.0 - alpha_v, 1.0 - alpha_w, p.dx,
                array_module=xp,
            )

        # Solve sum_f k_f (p_c - p_nb)/dx^2 = -(div u*)_c on active cells.
        rhs = -(div) * active
        kx2, ky2, kz2 = kx / p.dx**2, ky / p.dx**2, kz / p.dx**2
        ppe_solve = (
            multigrid.solve if p.pressure_solver == "multigrid"
            else pressure.solve)
        pr, iters, rel = ppe_solve(
            xp, rhs, kx2, ky2, kz2, active, tol=p.pcg_tol,
            max_iter=p.pcg_max_iter)
        stats.pcg_iters.append(iters)
        stats.pcg_rel_residuals.append(rel)
        if not math.isfinite(rel) or rel > p.pcg_tol:
            raise pressure.PressureSolveError(iters, rel, p.pcg_tol)

        pm = pr * active
        if getattr(self, "_collect_step_diagnostics", False):
            # Physics-validation capture (CAP-M3): the Laplace-jump test
            # reads the solved pressure directly.  Diagnostics-gated, so
            # the default path allocates nothing.
            self._last_pressure = pm
        gradx = (pm[1:, :, :] - pm[:-1, :, :]) / p.dx
        correction = dt * inv_rho_u[1:-1, :, :] * gradx
        grids["u"][1:-1, :, :] -= xp.where(
            open_u[1:-1, :, :], correction, 0.0)
        grady = (pm[:, 1:, :] - pm[:, :-1, :]) / p.dx
        correction = dt * inv_rho_v[:, 1:-1, :] * grady
        grids["v"][:, 1:-1, :] -= xp.where(
            open_v[:, 1:-1, :], correction, 0.0)
        gradz = (pm[:, :, 1:] - pm[:, :, :-1]) / p.dx
        correction = dt * inv_rho_w[:, :, 1:-1] * gradz
        grids["w"][:, :, 1:-1] -= xp.where(
            open_w[:, :, 1:-1], correction, 0.0)

        if self._has_pressure_outflow:
            # Exterior pressure-outflow faces place p=0 half a cell from the
            # adjacent cell centre. The factor two matches the half-cell
            # pressure gradient and the boundary terms in pressure.py.
            pressure_u, pressure_v, pressure_w = self._pressure_face_masks()
            grids["u"][0] -= xp.where(
                pressure_u[0],
                dt * inv_rho_u[0] * (2.0 * pm[0] / p.dx),
                0.0,
            )
            grids["u"][-1] -= xp.where(
                pressure_u[-1],
                dt * inv_rho_u[-1] * (-2.0 * pm[-1] / p.dx),
                0.0,
            )
            grids["v"][:, 0] -= xp.where(
                pressure_v[:, 0],
                dt * inv_rho_v[:, 0] * (2.0 * pm[:, 0] / p.dx),
                0.0,
            )
            grids["v"][:, -1] -= xp.where(
                pressure_v[:, -1],
                dt * inv_rho_v[:, -1] * (-2.0 * pm[:, -1] / p.dx),
                0.0,
            )
            grids["w"][:, :, 0] -= xp.where(
                pressure_w[:, :, 0],
                dt * inv_rho_w[:, :, 0] * (2.0 * pm[:, :, 0] / p.dx),
                0.0,
            )
            grids["w"][:, :, -1] -= xp.where(
                pressure_w[:, :, -1],
                dt * inv_rho_w[:, :, -1] * (-2.0 * pm[:, :, -1] / p.dx),
                0.0,
            )

        grids["u"] = xp.where(open_u, grids["u"], us_sol)
        grids["v"] = xp.where(open_v, grids["v"], vs_sol)
        grids["w"] = xp.where(open_w, grids["w"], ws_sol)

    def _extrapolate_valid_faces(self, grids, old, open_faces, solid_vel,
                                 layers: int) -> None:
        """Extrapolate grids (and the FLIP baseline) into invalid faces.

        Jittered advection can move a particle up to 2*dt, i.e. 2*CFL cells,
        so the band must cover that worst case.  The saved pre-force field
        ``old`` (None for PIC/APIC transfers) is extrapolated with the SAME
        mask: forming the FLIP delta between an extrapolated new field and
        hard zeros would hand isolated spray particles their neighbours'
        full velocity as a spurious energy kick (temporal weights can
        invalidate all of a particle's own faces when theta lands in the
        kernel's zero tail).
        """

        xp = self.be.xp
        open_u, open_v, open_w = open_faces
        us_sol, vs_sol, ws_sol = solid_vel
        for gname, face_open, sfv in (
            ("u", open_u, us_sol), ("v", open_v, vs_sol),
            ("w", open_w, ws_sol),
        ):
            valid = grids[gname + "_valid"] & face_open
            grids[gname], _ = self._extrapolate(
                grids[gname], valid, layers, allowed=face_open)
            # Re-enforce no-through defensively after extrapolation.
            grids[gname] = xp.where(face_open, grids[gname], sfv)
            if old is not None:
                old[gname], _ = self._extrapolate(
                    old[gname], valid, layers, allowed=face_open)
                old[gname] = xp.where(face_open, old[gname], sfv)

    def _step(self, dt: float, stats: FrameStats) -> None:
        if self._sparse_engaged() and self.pos.shape[0]:
            xp = self.be.xp
            lo, sub = self._active_window()
            dxlo = xp.asarray(lo.astype(np.float32) * self.p.dx,
                              dtype=xp.float32)
            saved = self._enter_window(lo, sub)
            self.pos = self.pos - dxlo
            try:
                self._step_core(dt, stats)
            finally:
                self.pos = self.pos + dxlo
                self._exit_window(saved)
            self._grid_origin = lo
        else:
            self._grid_origin = None
            self._step_core(dt, stats)

    def _step_core(self, dt: float, stats: FrameStats) -> None:
        if self.p.reflection:
            self._step_core_reflected(dt, stats)
            return
        xp = self.be.xp
        p = self.p
        dt_prev = self._dt_prev
        flip = (p.transfer == "flip")
        self._reconcile_particle_attrs()

        grids = self._p2g(dt_prev)
        alpha_u, alpha_v, alpha_w, solid_c = self._active_face_apertures()
        open_u = alpha_u > 0.0
        open_v = alpha_v > 0.0
        open_w = alpha_w > 0.0
        us_sol, vs_sol, ws_sol = self._solid_face_vel()

        # Save post-P2G velocities for the FLIP delta (FLIP transfer only).
        old = ({g: grids[g].copy() for g in ("u", "v", "w")} if flip else None)

        # External forces (grid-based, Sec 3.3): gravity, then art-directable
        # body forces (wind/vortex/turbulence).
        g = p.gravity
        grids["u"] = grids["u"] + g[0] * dt
        grids["v"] = grids["v"] + g[1] * dt
        grids["w"] = grids["w"] + g[2] * dt
        self._apply_forces(grids, dt)

        # Face densities rho(phi_f) = rho_l phi_f + rho_g (1 - phi_f).
        # Free-surface: phi_u is the space-time accumulator so rho collapses to
        # rho_l phi_f and air is a Dirichlet p = 0 boundary.  Two-phase: phi_u
        # is a liquid volume fraction and both phases are solved together.
        eps_rho = p.eps_rho_rel * p.rho
        if p.two_phase:
            rho_u = p.rho * grids["u_phi"] + p.rho_gas * (1.0 - grids["u_phi"])
            rho_v = p.rho * grids["v_phi"] + p.rho_gas * (1.0 - grids["v_phi"])
            rho_w = p.rho * grids["w_phi"] + p.rho_gas * (1.0 - grids["w_phi"])
            active = (grids["c_m"] > p.eps_m) & (~solid_c)
        else:
            rho_u = p.rho * grids["u_phi"]
            rho_v = p.rho * grids["v_phi"]
            rho_w = p.rho * grids["w_phi"]
            active = (grids["c_phi"] >= 0.5) & (~solid_c)
        inv_rho_u = 1.0 / xp.maximum(rho_u, eps_rho)
        inv_rho_v = 1.0 / xp.maximum(rho_v, eps_rho)
        inv_rho_w = 1.0 / xp.maximum(rho_w, eps_rho)

        # Surface tension (CSF, Sec 3.9): sigma * kappa * grad(phi) as a face
        # acceleration before projection.  The smoothed-phase gradient is
        # computed once and shared between the explicit predictor and the
        # optional semi-implicit stabilizer so both see the same interface
        # delta function.
        if p.surface_tension > 0.0:
            _phi_s, st_gx, st_gy, st_gz, st_mag = (
                surface_tension.smoothed_phase_gradient(
                    xp, grids["c_phi"], p.dx, p.st_smooth_iters))
            F = surface_tension.cell_force_from_gradient(
                xp, st_gx, st_gy, st_gz, st_mag, p.dx, p.surface_tension)
            fu = xp.zeros_like(grids["u"])
            fu[1:-1] = 0.5 * (F[1:, :, :, 0] + F[:-1, :, :, 0])
            fv = xp.zeros_like(grids["v"])
            fv[:, 1:-1] = 0.5 * (F[:, 1:, :, 1] + F[:, :-1, :, 1])
            fw = xp.zeros_like(grids["w"])
            fw[:, :, 1:-1] = 0.5 * (F[:, :, 1:, 2] + F[:, :, :-1, 2])
            ku = dt * fu * inv_rho_u
            kv = dt * fv * inv_rho_v
            kw = dt * fw * inv_rho_w
            if p.st_max_dv_cells > 0.0:
                # Clip the kick so one substep displaces at most
                # st_max_dv_cells cells; above the Brackbill limit the
                # explicit feedback then saturates instead of growing.
                dv_max = p.st_max_dv_cells * p.dx / dt
                ku = xp.clip(ku, -dv_max, dv_max)
                kv = xp.clip(kv, -dv_max, dv_max)
                kw = xp.clip(kw, -dv_max, dv_max)
            grids["u"] = grids["u"] + ku
            grids["v"] = grids["v"] + kv
            grids["w"] = grids["w"] + kw

            if p.st_implicit:
                # CAP-M2: (R + A) u = R * u_hat per component. Edges are
                # gated to VALID open faces only -- invalid faces hold
                # pre-extrapolation values and must not drag interface
                # velocities (spurious drag masquerading as damping).
                for axis, gname, rho_f, open_f in (
                    (0, "u", rho_u, open_u),
                    (1, "v", rho_v, open_v),
                    (2, "w", rho_w, open_w),
                ):
                    delta = st_implicit.face_delta(xp, st_mag, axis)
                    dof = grids[gname + "_valid"] & open_f
                    ekx, eky, ekz = st_implicit.edge_coefficients(
                        xp, delta, dof, dt, p.surface_tension, p.dx)
                    grids[gname], st_iters, st_rel = (
                        st_implicit.stabilize_component(
                            xp, grids[gname],
                            xp.maximum(rho_f, eps_rho), ekx, eky, ekz,
                            dof, tol=p.pcg_tol,
                            max_iter=p.pcg_max_iter))
                    stats.st_cg_iters.append(st_iters)
                    stats.st_cg_rel_residuals.append(st_rel)

        # Implicit viscosity (Stam-style diffusion): unconditionally stable, so
        # it preserves large time steps for thick fluids.  Fully-blocked solid
        # faces are no-slip Dirichlet at the solid velocity.
        if p.viscosity > 0.0:
            coef = dt * p.viscosity / (p.dx * p.dx)
            grids["u"] = viscosity.diffuse_component(
                xp, grids["u"], coef, ~open_u, us_sol,
                tol=p.visc_tol, max_iter=p.visc_max_iter)
            grids["v"] = viscosity.diffuse_component(
                xp, grids["v"], coef, ~open_v, vs_sol,
                tol=p.visc_tol, max_iter=p.visc_max_iter)
            grids["w"] = viscosity.diffuse_component(
                xp, grids["w"], coef, ~open_w, ws_sol,
                tol=p.visc_tol, max_iter=p.visc_max_iter)

        # No-through on fully blocked faces (u . n = u_solid . n) before the
        # aperture-weighted flux divergence.  A partially open face stores the
        # fluid velocity over its open area and remains active.
        grids["u"] = xp.where(open_u, grids["u"], us_sol)
        grids["v"] = xp.where(open_v, grids["v"], vs_sol)
        grids["w"] = xp.where(open_w, grids["w"], ws_sol)

        self._project(
            grids, dt, stats,
            (alpha_u, alpha_v, alpha_w),
            (open_u, open_v, open_w),
            (us_sol, vs_sol, ws_sol),
            (inv_rho_u, inv_rho_v, inv_rho_w),
            active,
        )

        layers = int(math.ceil(2.0 * p.cfl_target)) + 2
        self._extrapolate_valid_faces(
            grids, old,
            (open_u, open_v, open_w),
            (us_sol, vs_sol, ws_sol),
            layers,
        )

        # G2P (Sec 3.9): FLIP/PIC blend, pure PIC, or APIC affine transfer.
        u_new = self._sample_faces(grids, self.pos)
        if p.transfer == "apic":
            self.vel, self.C = self._g2p_apic(grids, self.pos, u_new)
        elif p.transfer == "pic":
            self.vel = u_new
        else:
            u_old = self._sample_faces(old, self.pos)
            a = p.flip_blend
            self.vel = (a * (self.vel + (u_new - u_old)) + (1.0 - a) * u_new)

        # Impart moving-wall velocity: kill the penetrating relative normal
        # velocity for particles inside the solid band.
        if self.solid_vel is not None:
            self._enforce_solid_velocity()

        # Temporal jitter with residual carryover (Eq. 10-11, Alg. 1 l.23-28).
        n = self.pos.shape[0]
        if p.st_enabled and p.jitter_strength > 0.0:
            # Contract: self.vel is NOT modified between this draw and the
            # next substep's P2G (sheeting is position-only, solid-velocity
            # enforcement runs before the draw, compaction and appends keep
            # all particle arrays consistent), so _jitter_gamma called with
            # the committed dt reproduces this gamma exactly at deposit
            # time.  The exact temporal normalization (Sec 3.10, roadmap
            # NORM-M1) relies on that; changes to the step tail must
            # preserve it or persist gamma instead.  Surface mode DOES
            # persist it: the interiorness term reads this substep's
            # c_phi, which next substep's P2G cannot reproduce.
            if p.gamma_mode == "surface":
                phi_gate = xp.where(
                    solid_c, xp.float32(1.0), grids["c_phi"])
                gamma = self._jitter_gamma(dt, phi_gate)
                if p.exact_temporal_norm:
                    self._gamma_prev = gamma
            elif p.gamma_mode == "deformation":
                gamma = self._deformation_gamma(dt, grids, solid_c)
                if p.exact_temporal_norm:
                    self._gamma_prev = gamma
            else:
                gamma = self._jitter_gamma(dt)
            if p.temporal_sampling == "sobol_owen":
                xi = sampling.temporal_xi(
                    xp, self.particle_id, self._substep_index, p.seed)
            elif p.temporal_sampling == "cp_rot":
                xi = sampling.temporal_xi_cp_rot(
                    xp, self.particle_id, self._substep_index, p.seed)
            else:
                xi = self.be.from_numpy(
                    self._rng.random(n, dtype=np.float32)) - 0.5
            jit = gamma * xi * dt
        else:
            jit = xp.zeros((n,), dtype=xp.float32)
        dt_act = xp.clip(dt + self.dt_resid + jit, 0.0, 2.0 * dt)
        self.dt_resid = (dt + self.dt_resid - dt_act).astype(xp.float32)

        has_outflows = self._has_volume_outflow or self._has_pressure_outflow
        if has_outflows and self.pos.shape[0]:
            advected, volume_removed, pressure_removed = self._advect(
                grids, self.pos, dt_act, track_outflows=True
            )
            self._apply_outflow_filter(
                advected, volume_removed, pressure_removed, stats
            )
        else:
            self.pos = self._advect(grids, self.pos, dt_act)
        if p.sheeting > 0.0:
            self._apply_sheeting()
        if getattr(self, "_collect_step_diagnostics", False):
            # ERR-M1: read-only signal capture, away from the jitter block
            # that SAMP-M3 / CALM-M3 / NORM-M1 own.  The clamp-bind test on
            # dt_act is mathematically identical to testing the pre-clip
            # operands.
            self._capture_step_diagnostics(grids, dt, dt_act, stats)
        self._grids = grids
        self._dt_prev = dt
        self._substep_index += 1
        self.time += dt
        if self.age.shape[0] == self.pos.shape[0]:
            self.age = self.age + np.float32(dt)

    def _step_core_reflected(self, dt: float, stats: FrameStats) -> None:
        """Advection-reflection substep (roadmap ENER-M2b).

        The corrected scheme of record from the roadmap review: forces
        enter particle momentum exactly once (via the mid-step reflected
        G2P), transport ALWAYS uses a projected field (never the
        non-solenoidal reflected field), and each half-advection travels at
        most dt so the half-band extrapolation contract is honest.  The
        residual-carryover induction for the composed half-clips is written
        out in docs/design/advection-reflection.md; the randomized
        adaptive-dt stress test in tests/test_reflection.py exercises it.

        Deliberate duplication: the force/density block mirrors
        _step_core so the default path stays byte-identical with
        reflection off.
        """

        xp = self.be.xp
        p = self.p
        dt_prev = self._dt_prev
        flip = (p.transfer == "flip")
        self._reconcile_particle_attrs()

        # (1) Slab-integrated first P2G; FLIP baseline snapshot.
        grids = self._p2g(dt_prev)
        alpha_u, alpha_v, alpha_w, solid_c = self._active_face_apertures()
        open_u = alpha_u > 0.0
        open_v = alpha_v > 0.0
        open_w = alpha_w > 0.0
        us_sol, vs_sol, ws_sol = self._solid_face_vel()
        old = ({g_: grids[g_].copy() for g_ in ("u", "v", "w")}
               if flip else None)

        # (2) Forces, densities, surface tension, viscosity, no-through --
        # the _step_core block, then a u_star snapshot for the mirror.
        g = p.gravity
        grids["u"] = grids["u"] + g[0] * dt
        grids["v"] = grids["v"] + g[1] * dt
        grids["w"] = grids["w"] + g[2] * dt
        self._apply_forces(grids, dt)

        eps_rho = p.eps_rho_rel * p.rho
        if p.two_phase:
            rho_u = p.rho * grids["u_phi"] + p.rho_gas * (
                1.0 - grids["u_phi"])
            rho_v = p.rho * grids["v_phi"] + p.rho_gas * (
                1.0 - grids["v_phi"])
            rho_w = p.rho * grids["w_phi"] + p.rho_gas * (
                1.0 - grids["w_phi"])
            active = (grids["c_m"] > p.eps_m) & (~solid_c)
        else:
            rho_u = p.rho * grids["u_phi"]
            rho_v = p.rho * grids["v_phi"]
            rho_w = p.rho * grids["w_phi"]
            active = (grids["c_phi"] >= 0.5) & (~solid_c)
        inv_rho_u = 1.0 / xp.maximum(rho_u, eps_rho)
        inv_rho_v = 1.0 / xp.maximum(rho_v, eps_rho)
        inv_rho_w = 1.0 / xp.maximum(rho_w, eps_rho)

        if p.surface_tension > 0.0:
            _phi_s, st_gx, st_gy, st_gz, st_mag = (
                surface_tension.smoothed_phase_gradient(
                    xp, grids["c_phi"], p.dx, p.st_smooth_iters))
            F = surface_tension.cell_force_from_gradient(
                xp, st_gx, st_gy, st_gz, st_mag, p.dx, p.surface_tension)
            fu = xp.zeros_like(grids["u"])
            fu[1:-1] = 0.5 * (F[1:, :, :, 0] + F[:-1, :, :, 0])
            fv = xp.zeros_like(grids["v"])
            fv[:, 1:-1] = 0.5 * (F[:, 1:, :, 1] + F[:, :-1, :, 1])
            fw = xp.zeros_like(grids["w"])
            fw[:, :, 1:-1] = 0.5 * (F[:, :, 1:, 2] + F[:, :, :-1, 2])
            ku = dt * fu * inv_rho_u
            kv = dt * fv * inv_rho_v
            kw = dt * fw * inv_rho_w
            if p.st_max_dv_cells > 0.0:
                dv_max = p.st_max_dv_cells * p.dx / dt
                ku = xp.clip(ku, -dv_max, dv_max)
                kv = xp.clip(kv, -dv_max, dv_max)
                kw = xp.clip(kw, -dv_max, dv_max)
            grids["u"] = grids["u"] + ku
            grids["v"] = grids["v"] + kv
            grids["w"] = grids["w"] + kw
            if p.st_implicit:
                for axis, gname, rho_f, open_f in (
                    (0, "u", rho_u, open_u),
                    (1, "v", rho_v, open_v),
                    (2, "w", rho_w, open_w),
                ):
                    delta = st_implicit.face_delta(xp, st_mag, axis)
                    dof = grids[gname + "_valid"] & open_f
                    ekx, eky, ekz = st_implicit.edge_coefficients(
                        xp, delta, dof, dt, p.surface_tension, p.dx)
                    grids[gname], st_iters, st_rel = (
                        st_implicit.stabilize_component(
                            xp, grids[gname],
                            xp.maximum(rho_f, eps_rho), ekx, eky, ekz,
                            dof, tol=p.pcg_tol,
                            max_iter=p.pcg_max_iter))
                    stats.st_cg_iters.append(st_iters)
                    stats.st_cg_rel_residuals.append(st_rel)

        if p.viscosity > 0.0:
            coef = dt * p.viscosity / (p.dx * p.dx)
            grids["u"] = viscosity.diffuse_component(
                xp, grids["u"], coef, ~open_u, us_sol,
                tol=p.visc_tol, max_iter=p.visc_max_iter)
            grids["v"] = viscosity.diffuse_component(
                xp, grids["v"], coef, ~open_v, vs_sol,
                tol=p.visc_tol, max_iter=p.visc_max_iter)
            grids["w"] = viscosity.diffuse_component(
                xp, grids["w"], coef, ~open_w, ws_sol,
                tol=p.visc_tol, max_iter=p.visc_max_iter)

        grids["u"] = xp.where(open_u, grids["u"], us_sol)
        grids["v"] = xp.where(open_v, grids["v"], vs_sol)
        grids["w"] = xp.where(open_w, grids["w"], ws_sol)
        u_star = {g_: grids[g_].copy() for g_ in ("u", "v", "w")}

        # (3) First projection: u1, the (solenoidal) transport field.
        self._project(
            grids, dt, stats,
            (alpha_u, alpha_v, alpha_w),
            (open_u, open_v, open_w),
            (us_sol, vs_sol, ws_sol),
            (inv_rho_u, inv_rho_v, inv_rho_w),
            active,
        )

        # (4) Reflected field u_hat = 2 u1 - u_star ONLY on open faces
        # that were valid in the first P2G AND carry meaningful phase
        # support; eps_rho-clamped near-interface faces must not get
        # their projection residual doubled.  Never used for transport.
        u_hat = {}
        for gname, open_f, sfv in (
            ("u", open_u, us_sol), ("v", open_v, vs_sol),
            ("w", open_w, ws_sol),
        ):
            mirror = (open_f & grids[gname + "_valid"]
                      & (grids[gname + "_phi"] >= 0.5))
            u_hat[gname] = xp.where(
                mirror, 2.0 * grids[gname] - u_star[gname], grids[gname])
            u_hat[gname] = xp.where(open_f, u_hat[gname], sfv)

        # (5) Half-band extrapolation with the SAME valid masks for the
        # transport field, the reflected field, and the FLIP baseline.
        half_layers = int(math.ceil(p.cfl_target)) + 2
        for gname, face_open, sfv in (
            ("u", open_u, us_sol), ("v", open_v, vs_sol),
            ("w", open_w, ws_sol),
        ):
            valid = grids[gname + "_valid"] & face_open
            grids[gname], _ = self._extrapolate(
                grids[gname], valid, half_layers, allowed=face_open)
            grids[gname] = xp.where(face_open, grids[gname], sfv)
            u_hat[gname], _ = self._extrapolate(
                u_hat[gname], valid, half_layers, allowed=face_open)
            u_hat[gname] = xp.where(face_open, u_hat[gname], sfv)
            if flip:
                old[gname], _ = self._extrapolate(
                    old[gname], valid, half_layers, allowed=face_open)
                old[gname] = xp.where(face_open, old[gname], sfv)

        # (6) Mid-step reflection G2P: forces and the reflected pressure
        # impulse enter particle momentum exactly once.  FLIP takes the
        # pure delta interp(u_hat - old), which equals the plain delta
        # interp(u1 - old) plus the reflected impulse interp(u1 - u_star);
        # PIC and APIC are replacement transfers and read u_hat directly.
        u_hat_p = self._sample_faces(u_hat, self.pos)
        if p.transfer == "apic":
            self.vel, self.C = self._g2p_apic(u_hat, self.pos, u_hat_p)
        elif p.transfer == "pic":
            self.vel = u_hat_p
        else:
            u_old_p = self._sample_faces(old, self.pos)
            self.vel = self.vel + (u_hat_p - u_old_p)
        if self.solid_vel is not None:
            self._enforce_solid_velocity()

        # (7) First half-advection through u1.  The clip upper bound dt
        # keeps each half within cfl_target cells (the half band above)
        # and total per-substep travel <= 2 dt (the _band contract).
        dt_act1 = xp.clip(dt / 2.0 + self.dt_resid, 0.0, dt)
        r1 = (dt / 2.0 + self.dt_resid - dt_act1).astype(xp.float32)
        has_outflows = (
            self._has_volume_outflow or self._has_pressure_outflow)
        if has_outflows and self.pos.shape[0]:
            advected, volume_removed, pressure_removed = self._advect(
                grids, self.pos, dt_act1, track_outflows=True)
            keep = ~(volume_removed | pressure_removed)
            self._apply_outflow_filter(
                advected, volume_removed, pressure_removed, stats)
            r1 = r1[keep]
        else:
            self.pos = self._advect(grids, self.pos, dt_act1)

        # (8) Instantaneous second P2G (wt = 1) + its FLIP baseline.
        grids2 = self._p2g(dt, force_instantaneous=True)
        old2 = ({g_: grids2[g_].copy() for g_ in ("u", "v", "w")}
                if flip else None)

        # (9) Face densities and the active mask from the second deposit.
        # Do NOT re-apply forces: they already reached particle momentum
        # in (6) and are inside this deposit.  Step-start apertures and
        # wall velocities are reused (walls at t^n, O(dt) consistent).
        if p.two_phase:
            rho_u2 = p.rho * grids2["u_phi"] + p.rho_gas * (
                1.0 - grids2["u_phi"])
            rho_v2 = p.rho * grids2["v_phi"] + p.rho_gas * (
                1.0 - grids2["v_phi"])
            rho_w2 = p.rho * grids2["w_phi"] + p.rho_gas * (
                1.0 - grids2["w_phi"])
            active2 = (grids2["c_m"] > p.eps_m) & (~solid_c)
        else:
            rho_u2 = p.rho * grids2["u_phi"]
            rho_v2 = p.rho * grids2["v_phi"]
            rho_w2 = p.rho * grids2["w_phi"]
            active2 = (grids2["c_phi"] >= 0.5) & (~solid_c)
        inv_rho2 = (
            1.0 / xp.maximum(rho_u2, eps_rho),
            1.0 / xp.maximum(rho_v2, eps_rho),
            1.0 / xp.maximum(rho_w2, eps_rho),
        )
        grids2["u"] = xp.where(open_u, grids2["u"], us_sol)
        grids2["v"] = xp.where(open_v, grids2["v"], vs_sol)
        grids2["w"] = xp.where(open_w, grids2["w"], ws_sol)

        # (10) Second projection, extrapolation, final G2P.
        self._project(
            grids2, dt, stats,
            (alpha_u, alpha_v, alpha_w),
            (open_u, open_v, open_w),
            (us_sol, vs_sol, ws_sol),
            inv_rho2,
            active2,
        )
        self._extrapolate_valid_faces(
            grids2, old2,
            (open_u, open_v, open_w),
            (us_sol, vs_sol, ws_sol),
            half_layers,
        )
        u_new = self._sample_faces(grids2, self.pos)
        if p.transfer == "apic":
            self.vel, self.C = self._g2p_apic(grids2, self.pos, u_new)
        elif p.transfer == "pic":
            self.vel = u_new
        else:
            u_old2 = self._sample_faces(old2, self.pos)
            a = p.flip_blend
            self.vel = (a * (self.vel + (u_new - u_old2))
                        + (1.0 - a) * u_new)
        if self.solid_vel is not None:
            self._enforce_solid_velocity()

        # (11) Second-half jitter with fresh velocities.  Unclamped, this
        # gives dt_resid = -gamma xi dt, the paper's stationary
        # distribution; the NORM vel-immutability contract holds because
        # only positions change after this draw.
        n = self.pos.shape[0]
        if p.st_enabled and p.jitter_strength > 0.0:
            if p.gamma_mode == "surface":
                phi_gate = xp.where(
                    solid_c, xp.float32(1.0), grids2["c_phi"])
                gamma = self._jitter_gamma(dt, phi_gate)
                if p.exact_temporal_norm:
                    self._gamma_prev = gamma
            elif p.gamma_mode == "deformation":
                gamma = self._deformation_gamma(dt, grids2, solid_c)
                if p.exact_temporal_norm:
                    self._gamma_prev = gamma
            else:
                gamma = self._jitter_gamma(dt)
            if p.temporal_sampling == "sobol_owen":
                xi = sampling.temporal_xi(
                    xp, self.particle_id, self._substep_index, p.seed)
            elif p.temporal_sampling == "cp_rot":
                xi = sampling.temporal_xi_cp_rot(
                    xp, self.particle_id, self._substep_index, p.seed)
            else:
                xi = self.be.from_numpy(
                    self._rng.random(n, dtype=np.float32)) - 0.5
            jit = gamma * xi * dt
        else:
            jit = xp.zeros((n,), dtype=xp.float32)
        dt_act2 = xp.clip(dt / 2.0 + r1 + jit, 0.0, dt)
        self.dt_resid = (dt / 2.0 + r1 - dt_act2).astype(xp.float32)

        # (12) Second half-advection through grids2; commit.
        if has_outflows and self.pos.shape[0]:
            advected, volume_removed, pressure_removed = self._advect(
                grids2, self.pos, dt_act2, track_outflows=True)
            self._apply_outflow_filter(
                advected, volume_removed, pressure_removed, stats)
        else:
            self.pos = self._advect(grids2, self.pos, dt_act2)
        if p.sheeting > 0.0:
            self._apply_sheeting()
        if getattr(self, "_collect_step_diagnostics", False):
            self._capture_step_diagnostics(grids2, dt, dt_act2, stats)
        self._grids = grids2
        self._dt_prev = dt
        self._substep_index += 1
        self.time += dt
        if self.age.shape[0] == self.pos.shape[0]:
            self.age = self.age + np.float32(dt)

    def _capture_step_diagnostics(self, grids, dt: float, dt_act,
                                  stats: FrameStats) -> None:
        """Append the ERR-M1 step-control signals for this substep.

        Gated behind the internal (non-Params) ``_collect_step_diagnostics``
        attribute so the default path is bit-identical and pays nothing.
        Fractions are normalised by valid-face counts, never by total face
        count (window-shape dependent).
        """

        xp = self.be.xp
        p = self.p
        n = self.pos.shape[0]
        if n:
            bind = (dt_act <= 0.0) | (dt_act >= 2.0 * dt)
            stats.clamp_bind_fractions.append(
                float(bind.sum()) / float(n))
        else:
            stats.clamp_bind_fractions.append(0.0)

        marginal = 0.0
        subthreshold = 0.0
        valid_total = 0.0
        for g in ("u", "v", "w"):
            m = grids[g + "_m"]
            valid = grids[g + "_valid"]
            count = float(valid.sum())
            valid_total += count
            marginal += float((valid & (m < 0.25 * self.m0)).sum())
            subthreshold += float(((m > 0.0) & (m <= p.eps_m)).sum())
        denom = max(valid_total, 1.0)
        stats.undersampled_marginal_fractions.append(marginal / denom)
        stats.undersampled_subthreshold_fractions.append(
            subthreshold / denom)

        if n and float(self.sdf.min()) < 1e8:
            sdf_at = self._sample_cells(self.sdf, self.pos)
            speed_cells = _norm_rows(xp, self.vel) * dt / p.dx
            near_fast = (sdf_at < 2.0 * p.dx) & (speed_cells > 2.0)
            stats.near_solid_fast_fractions.append(
                float(near_fast.sum()) / float(n))
        else:
            stats.near_solid_fast_fractions.append(0.0)

    def step_frame(self) -> FrameStats:
        """Advance one video frame (Algorithm 1 outer loop)."""
        xp = self.be.xp
        p = self.p
        stats = FrameStats()
        self._seed_inflows()
        t_rem = p.frame_dt
        vmax = (float(xp.max(_norm_rows(xp, self.vel)))
                if self.pos.shape[0] else 0.0)
        while t_rem > 1e-9 * p.frame_dt:
            if self.pos.shape[0] == 0:
                # An emitter can become active between output-frame
                # boundaries. Jump an otherwise idle solver exactly to the
                # next start time so that the remaining interval is evolved.
                next_start = self._next_inflow_start(self.time + t_rem)
                if next_start is not None:
                    idle_dt = next_start - self.time
                    self.time += idle_dt
                    t_rem -= idle_dt
                    stats.inactive_time_s += idle_dt
                    self._seed_inflows()
                    if self.pos.shape[0]:
                        vmax = float(xp.max(_norm_rows(xp, self.vel)))
                    continue
                # No state remains to evolve. Advancing the clock directly
                # avoids an otherwise catastrophic empty O(grid) projection
                # and extrapolation pass on production-resolution domains.
                self.time += t_rem
                stats.inactive_time_s += t_rem
                t_rem = 0.0
                break
            next_start = self._next_inflow_start(self.time + t_rem)
            segment_rem = (
                next_start - self.time if next_start is not None else t_rem
            )
            dt = min(
                p.cfl_target * p.dx / max(vmax, 1e-6), segment_rem
            )
            dt_before_capillary = dt
            if p.surface_tension > 0.0:
                # Capillary stability limit (Brackbill et al. 1992): surface
                # tension caps dt at O(dx^{3/2}) independently of the velocity
                # CFL (paper Sec 5), so large sigma and large CFL targets
                # cannot be combined without this clamp blowing up the sim.
                # st_clamp_scale (CAP-M0) relaxes the cap by a bounded user
                # factor; it trades capillary accuracy/stability margin for
                # substeps and is meant to pair with st_max_dv_cells.
                rho_sum = p.rho + (p.rho_gas if p.two_phase else 0.0)
                dt = min(dt, p.st_clamp_scale * math.sqrt(
                    rho_sum * p.dx ** 3
                    / (4.0 * math.pi * p.surface_tension)))
            if getattr(self, "_collect_step_diagnostics", False):
                # ERR signal 6: when the capillary clamp set dt, a CFL-side
                # guard is inert by construction -- record which regime the
                # substep ran in.
                stats.capillary_clamped_steps.append(
                    bool(dt < dt_before_capillary))
            # Subdivide the remaining frame time into even parts (Alg. 1 l.7).
            dt = segment_rem / math.ceil(segment_rem / dt)
            stats.particle_cfl_estimated_values.append(vmax * dt / p.dx)
            self._step(dt, stats)
            vmax = (float(xp.max(_norm_rows(xp, self.vel)))
                    if self.pos.shape[0] else 0.0)
            stats.particle_cfl_actual_values.append(vmax * dt / p.dx)
            stats.steps += 1
            stats.dt_values.append(dt)
            t_rem -= dt
            if t_rem > 1e-9 * p.frame_dt:
                # An inflow is a continuous source, not an output-frame event.
                # Refill at the current global time before the next solver
                # step; occupancy in _seed_inflows prevents duplicate filling.
                n_before = self.pos.shape[0]
                self._seed_inflows()
                if self.pos.shape[0] != n_before:
                    # Newly emitted particles may be faster than the particles
                    # used for the post-step CFL measurement above.
                    vmax = float(xp.max(_norm_rows(xp, self.vel)))
        stats.n_particles = int(self.pos.shape[0])
        stats.max_speed = vmax
        return stats

    def _next_inflow_start(self, horizon: float) -> float | None:
        """Return the next nonzero-duration start strictly before horizon."""
        tolerance = max(1e-12, 1e-9 * self.p.frame_dt)
        starts = (
            start_time
            for _, _, start_time, end_time, _, _ in self._inflows
            if (end_time is None or end_time > start_time + tolerance)
            and self.time + tolerance < start_time < horizon - tolerance
        )
        return min(starts, default=None)

    def _seed_inflows(self) -> None:
        if not self._inflows:
            return
        # Filter schedules before allocating a domain-sized occupancy grid.
        # A future or expired source should preserve the empty-frame fast path,
        # especially at production resolutions where one 512^3 float grid is
        # 512 MiB.  The tolerance prevents accumulated frame-time roundoff from
        # activating a source for one extra interval at its exclusive endpoint.
        tolerance = max(1e-12, 1e-9 * self.p.frame_dt)
        active_inflows = []
        for inflow in self._inflows:
            _, _, start_time, end_time, _phase, _sid = inflow
            if self.time + tolerance < start_time:
                continue
            if (end_time is not None
                    and self.time + tolerance >= end_time):
                continue
            active_inflows.append(inflow)
        if not active_inflows:
            return
        xp = self.be.xp
        nx, ny, nz = self.shape
        # Occupancy from particle binning.
        idx = xp.clip((self.pos / self.p.dx).astype(xp.int32), 0,
                      xp.asarray([nx - 1, ny - 1, nz - 1]))
        flat = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
        counts = xp.zeros((nx * ny * nz,), dtype=xp.float32)
        if flat.shape[0]:
            self.be.scatter_add(counts, flat, xp.ones_like(flat, dtype=xp.float32))
        counts = counts.reshape(self.shape)
        for mask, velocity_field, _start, _end, phase, sid in active_inflows:
            ppc = (self.p.gas_particles_per_cell if phase < 0.5
                   else self.p.particles_per_cell)
            refill_device = mask & (counts < 0.5 * ppc)
            refill = self.be.to_numpy(refill_device)
            cells = np.argwhere(refill)
            seeded = self._seed_cells(cells, velocity_field, phase=phase,
                                      ppc=ppc, source_id=sid)
            if seeded:
                # Later sources in registration order observe particles just
                # seeded by earlier sources, preventing overlapping masks from
                # filling the same cell twice. Each selected cell receives ppc
                # particles, exactly matching _seed_cells.
                counts = counts + refill_device.astype(xp.float32) * ppc

    # ----------------------------------------------------------------- export

    def _resynced_positions_and_keep(self):
        """Render positions (host) re-synchronised to the global time, plus the
        outflow-survivor keep mask (host, or None).  Shared by all exporters."""
        if self.pos.shape[0] == 0 or not self._grids:
            return self.be.to_numpy(self.pos), None
        if self._has_volume_outflow or self._has_pressure_outflow:
            pos, vr, pr = self._advect(
                self._grids, self.pos.copy(), self.dt_resid,
                track_outflows=True)
            keep = ~(vr | pr)
            return self.be.to_numpy(pos[keep]), self.be.to_numpy(keep)
        if self._grid_origin is not None:
            xp = self.be.xp
            lo = self._grid_origin
            sub = tuple(self._grids["c_m"].shape)
            dxlo = xp.asarray(lo.astype(np.float32) * self.p.dx,
                              dtype=xp.float32)
            saved = self._enter_window(lo, sub)
            try:
                pos = self._advect(self._grids, self.pos.copy() - dxlo,
                                   self.dt_resid) + dxlo
            finally:
                self._exit_window(saved)
            return self.be.to_numpy(pos), None
        return self.be.to_numpy(
            self._advect(self._grids, self.pos.copy(), self.dt_resid)), None

    def get_render_particles(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions re-synchronised to the global time (Alg. 1 l.31-34) and
        velocities, as host arrays.  Two-phase gas particles are excluded."""
        pos, vel, _attrs = self.get_render_particles_ex()
        return pos, vel

    def get_render_phase_particles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return global-time positions and aligned phase values on the host.

        Unlike :meth:`get_render_particles`, this snapshot retains both phases
        so two-phase validation can measure liquid/gas motion at one common
        output time. Phase is 1 for liquid and 0 for gas; single-phase solvers
        report every surviving particle as liquid.
        """
        self._reconcile_particle_attrs()
        pos, keep = self._resynced_positions_and_keep()
        if self.p.two_phase and self.phase.shape[0] == self.pos.shape[0]:
            phase = self.be.to_numpy(self.phase)
        else:
            phase = np.ones((self.pos.shape[0],), dtype=np.float32)
        if keep is not None:
            phase = phase[keep]
        return pos, np.asarray(phase, dtype=np.float32)

    def get_render_particles_ex(self):
        """Render positions, velocities, and a dict of shading attributes
        ``{"age", "source", "speed"}`` (host arrays), all aligned and with
        two-phase gas particles excluded.  ``speed`` is |velocity|; ``age`` is
        seconds since seeding; ``source`` is the 0-based seeding-source id."""
        self._reconcile_particle_attrs()
        pos, keep = self._resynced_positions_and_keep()
        vel = self.be.to_numpy(self.vel)
        age = self.be.to_numpy(self.age)
        src = self.be.to_numpy(self.source_id)
        if keep is not None:
            vel, age, src = vel[keep], age[keep], src[keep]
        if self.p.two_phase and self.phase.shape[0] == self.pos.shape[0]:
            liquid = self.be.to_numpy(self.phase) > 0.5
            if keep is not None:
                liquid = liquid[keep]
            pos, vel, age, src = pos[liquid], vel[liquid], age[liquid], src[liquid]
        speed = np.sqrt((vel * vel).sum(axis=1)) if len(vel) else \
            np.zeros((0,), dtype=np.float32)
        return pos, vel, {"age": age.astype(np.float32),
                          "source": src.astype(np.int32),
                          "speed": speed.astype(np.float32)}
