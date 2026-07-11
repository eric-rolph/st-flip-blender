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

from . import apertures, kernels, pressure
from .backend import Backend, get_backend
from .velocity import VelocityField, VelocityInput, as_velocity_field

# Face-grid offsets (in cell units) of node (i,j,k) for each MAC grid.
_OFFSETS = {
    "u": (0.0, 0.5, 0.5),
    "v": (0.5, 0.0, 0.5),
    "w": (0.5, 0.5, 0.0),
    "c": (0.5, 0.5, 0.5),
}

_TAPS = [(di, dj, dk) for di in (0, 1) for dj in (0, 1) for dk in (0, 1)]


def _norm_rows(xp, a):
    """Row-wise euclidean norm without cupy.linalg (avoids cuBLAS)."""
    return xp.sqrt((a * a).sum(axis=1))


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
    eta_phi: float = 0.5              # phase-transition steepness
    eps_m: float = 1e-9               # under-sampled face threshold
    eps_rho_rel: float = 1e-3         # eps_rho = eps_rho_rel * rho
    pcg_tol: float = 1e-4
    pcg_max_iter: int = 400
    cfl_local: float = 1.0            # advection sub-step bound
    seed: int = 0

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
        for name in ("st_enabled", "adaptive_gamma"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
            raise ValueError("seed must be an integer")
        self.seed = int(self.seed)
        if self.seed < 0:
            raise ValueError("seed must not be negative")


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

        # Masks stay on-device for occupancy checks; immutable velocity fields
        # stay on the host so each refill is sampled deterministically at its
        # newly jittered particle positions.
        self._inflows: list[tuple[object, VelocityField]] = []
        # Outflow masks are lazy: two dense 512^3 boolean allocations would
        # otherwise cost ~256 MiB on every no-outflow CUDA simulation.
        self._volume_outflow = None
        self._pressure_outflow = None
        self._has_volume_outflow = False
        self._has_pressure_outflow = False
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

    def checkpoint_state(self) -> dict:
        """Return an owned, backend-neutral snapshot sufficient to restart.

        Grid velocities, aperture caches, and gradients are derived from the
        configured scene plus particle state and are rebuilt on the next step.
        The NumPy RNG state and previous adaptive timestep are trajectory state
        and therefore must be captured alongside particle arrays.
        """
        return {
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
        }

    def restore_state(self, state: dict) -> None:
        """Strictly restore a snapshot into this configured solver instance."""
        required = {
            "pos", "vel", "dt_resid", "time", "dt_prev", "rng_state",
            "outflow_removed_total", "volume_outflow_removed_total",
            "pressure_outflow_removed_total",
        }
        if not isinstance(state, dict) or set(state) != required:
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

        # Commit only after the complete state validates, so a rejected restore
        # cannot leave a running solver partially mutated.
        self.pos = self.be.from_numpy(pos)
        self.vel = self.be.from_numpy(vel)
        self.dt_resid = self.be.from_numpy(dt_resid)
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

    def set_solid_sdf(
        self,
        sdf_cells: np.ndarray,
        node_sdf: np.ndarray | None = None,
    ) -> None:
        """Set the collision SDF and, optionally, fractional solid geometry.

        ``sdf_cells`` preserves the original binary cell-centred API and is
        always used for particle collision push-out.  Supplying
        ``node_sdf`` with shape ``(nx + 1, ny + 1, nz + 1)`` enables
        cut-cell face apertures for pressure projection.  Omitting it keeps
        the previous binary rule: a face is blocked when either adjacent
        cell is solid.
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
        return self._seed_cells(cells, field)

    def add_inflow(
        self,
        cell_mask: np.ndarray,
        velocity: VelocityInput = (0.0, 0.0, 0.0),
    ) -> None:
        field = as_velocity_field(velocity)
        mask = self._validate_cell_mask(cell_mask)
        self._inflows.append(
            (self.be.from_numpy(mask),
             field)
        )

    def add_outflow(self, cell_mask: np.ndarray, mode: str = "VOLUME") -> None:
        """Register a particle sink or an exterior pressure/open boundary.

        ``VOLUME`` removes particles entering any marked cell, with the mask
        tested after every local RK advection substep. ``PRESSURE`` opens only
        the simulation-domain faces intersected by marked boundary cells,
        imposes exterior ``p = 0`` at half-cell distance, and removes particles
        after they cross one of those faces.
        """
        if not isinstance(mode, str):
            raise ValueError("outflow mode must be 'VOLUME' or 'PRESSURE'")
        normalized = mode.strip().upper()
        if normalized not in {"VOLUME", "PRESSURE"}:
            raise ValueError("outflow mode must be 'VOLUME' or 'PRESSURE'")
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
            self._has_pressure_outflow = True
            self._pressure_outflow_faces = None
        self._outflow_geometry_stats_cache = None

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

    def _seed_cells(self, cells: np.ndarray, velocity: VelocityInput) -> int:
        field = as_velocity_field(velocity)
        if len(cells) == 0:
            return 0
        xp = self.be.xp
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
        return n

    # ------------------------------------------------------------- deposition

    def _calibrate_m0(self) -> float:
        """Reference mass: expected accumulator value for a uniformly filled
        patch with ppc particles per cell and tau ~ U(-1/2, 1/2) (Sec 3.6)."""
        rng = np.random.default_rng(12345)
        ppc = self.p.particles_per_cell
        n_cells, trials = 6, 8
        acc = 0.0
        for _ in range(trials):
            pts = rng.random((n_cells**3 * ppc, 3)) * n_cells
            tau = rng.random(len(pts)) - 0.5
            wt = (kernels.w_temporal(np, tau) if self.p.st_enabled
                  else np.ones_like(tau))
            centre = np.array([n_cells / 2.0] * 3)
            r = (pts - centre)
            w = (kernels.w_spatial_1d(np, r[:, 0])
                 * kernels.w_spatial_1d(np, r[:, 1])
                 * kernels.w_spatial_1d(np, r[:, 2]) * wt)
            acc += float(w.sum())
        return acc / trials

    def _p2g(self, dt_prev: float):
        """4D->3D particle-to-grid transfer (Eq. 8-9). Returns face grids."""
        xp = self.be.xp
        p = self.p
        nx, ny, nz = self.shape

        gp = self.pos / p.dx  # positions in grid units
        # No clipping: W_T is zero outside the slab on BOTH sides, so
        # out-of-slab samples (possible after abrupt adaptive-dt changes)
        # correctly receive zero weight instead of the clipped peak weight.
        theta = -self.dt_resid / max(dt_prev, 1e-12)
        if p.st_enabled:
            wt = kernels.w_temporal(xp, theta).astype(xp.float32)
        else:
            wt = xp.ones_like(theta)

        shapes = {"u": (nx + 1, ny, nz), "v": (nx, ny + 1, nz),
                  "w": (nx, ny, nz + 1), "c": (nx, ny, nz)}
        vel_axis = {"u": 0, "v": 1, "w": 2}

        grids = {}
        for g, off in _OFFSETS.items():
            sh = shapes[g]
            mass = xp.zeros(sh, dtype=xp.float32).ravel()
            mom = (xp.zeros(sh, dtype=xp.float32).ravel()
                   if g != "c" else None)
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
                self.be.scatter_add(mass, flat, w)
                if mom is not None:
                    self.be.scatter_add(
                        mom, flat, w * self.vel[:, vel_axis[g]])
            grids[g + "_m"] = mass.reshape(sh)
            if mom is not None:
                grids[g + "_p"] = mom.reshape(sh)

        for g in ("u", "v", "w"):
            m = grids[g + "_m"]
            valid = m > p.eps_m
            grids[g] = xp.where(valid, grids[g + "_p"] / xp.maximum(m, p.eps_m), 0.0)
            grids[g + "_valid"] = valid
            # Space-time phase field from the weight accumulators (Eq. 13):
            # phi = C(m / (eta_phi * m0)), C(x) = min(sqrt(x), 1).
            grids[g + "_phi"] = xp.minimum(
                xp.sqrt(m / (p.eta_phi * self.m0)), 1.0)
        grids["c_phi"] = xp.minimum(
            xp.sqrt(grids["c_m"] / (p.eta_phi * self.m0)), 1.0)
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

    def _pressure_face_masks(self):
        """Full MAC masks for exterior faces opened by PRESSURE outflows."""
        if self._pressure_outflow_faces is not None:
            return self._pressure_outflow_faces
        xp = self.be.xp
        nx, ny, nz = self.shape
        _, _, _, solid_c = self._solid_face_apertures()
        boundary_cells = self._pressure_outflow & (~solid_c)
        exterior = self._solid_exterior_apertures
        face_u = xp.zeros((nx + 1, ny, nz), dtype=bool)
        face_v = xp.zeros((nx, ny + 1, nz), dtype=bool)
        face_w = xp.zeros((nx, ny, nz + 1), dtype=bool)
        face_u[0] = boundary_cells[0] & (exterior[0] > 0.0)
        face_u[-1] = boundary_cells[-1] & (exterior[1] > 0.0)
        face_v[:, 0] = boundary_cells[:, 0] & (exterior[2] > 0.0)
        face_v[:, -1] = boundary_cells[:, -1] & (exterior[3] > 0.0)
        face_w[:, :, 0] = boundary_cells[:, :, 0] & (exterior[4] > 0.0)
        face_w[:, :, -1] = boundary_cells[:, :, -1] & (exterior[5] > 0.0)
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
        keep = ~(volume_removed | pressure_removed)
        counts = self.be.to_numpy(xp.stack((
            volume_removed.sum(), pressure_removed.sum()
        )))
        volume_count, pressure_count = (int(value) for value in counts.tolist())
        removed = volume_count + pressure_count
        self.pos = positions[keep]
        self.vel = self.vel[keep]
        self.dt_resid = self.dt_resid[keep]
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

    # ------------------------------------------------------------------- step

    def _step(self, dt: float, stats: FrameStats) -> None:
        xp = self.be.xp
        p = self.p
        dt_prev = self._dt_prev

        grids = self._p2g(dt_prev)
        alpha_u, alpha_v, alpha_w, solid_c = self._active_face_apertures()
        open_u = alpha_u > 0.0
        open_v = alpha_v > 0.0
        open_w = alpha_w > 0.0

        # Save post-P2G velocities for the FLIP delta.
        old = {g: grids[g].copy() for g in ("u", "v", "w")}

        # External forces (grid-based, Sec 3.3).
        g = p.gravity
        grids["u"] = grids["u"] + g[0] * dt
        grids["v"] = grids["v"] + g[1] * dt
        grids["w"] = grids["w"] + g[2] * dt

        # No-through on fully blocked faces before computing the aperture-
        # weighted flux divergence.  A partially open face stores the fluid
        # velocity over its open area and remains active.
        grids["u"] = xp.where(open_u, grids["u"], 0.0)
        grids["v"] = xp.where(open_v, grids["v"], 0.0)
        grids["w"] = xp.where(open_w, grids["w"], 0.0)

        liquid = (grids["c_phi"] >= 0.5) & (~solid_c)

        # PPE coefficients are aperture-weighted, while the velocity update
        # itself is not: k_f = dt * alpha_f / rho_f, followed by
        # u_f <- u_f - dt/rho_f * grad(p) on every alpha_f > 0 face.
        # This distinction is essential for fractional faces: alpha belongs
        # in the flux constraint, not in the physical pressure acceleration.
        eps_rho = p.eps_rho_rel * p.rho
        inv_rho_u = 1.0 / xp.maximum(p.rho * grids["u_phi"], eps_rho)
        inv_rho_v = 1.0 / xp.maximum(p.rho * grids["v_phi"], eps_rho)
        inv_rho_w = 1.0 / xp.maximum(p.rho * grids["w_phi"], eps_rho)
        kx = dt * alpha_u * inv_rho_u
        ky = dt * alpha_v * inv_rho_v
        kz = dt * alpha_w * inv_rho_w

        div = apertures.weighted_divergence(
            grids["u"], grids["v"], grids["w"],
            alpha_u, alpha_v, alpha_w, p.dx, array_module=xp,
        )

        # Solve sum_f k_f (p_c - p_nb)/dx^2 = -(div u*)_c on liquid cells.
        rhs = -(div) * liquid
        kx2, ky2, kz2 = kx / p.dx**2, ky / p.dx**2, kz / p.dx**2
        pr, iters, rel = pressure.solve(
            xp, rhs, kx2, ky2, kz2, liquid, tol=p.pcg_tol,
            max_iter=p.pcg_max_iter)
        stats.pcg_iters.append(iters)
        stats.pcg_rel_residuals.append(rel)

        pm = pr * liquid
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

        grids["u"] = xp.where(open_u, grids["u"], 0.0)
        grids["v"] = xp.where(open_v, grids["v"], 0.0)
        grids["w"] = xp.where(open_w, grids["w"], 0.0)

        # Extrapolate into under-sampled faces so advection sees a full
        # field.  Jittered advection can move a particle up to 2*dt, i.e.
        # 2*CFL cells, so the band must cover that worst case.  The saved
        # pre-force field is extrapolated with the SAME mask: forming the
        # FLIP delta between an extrapolated new field and hard zeros would
        # hand isolated spray particles their neighbours' full velocity as
        # a spurious energy kick (temporal weights can invalidate all of a
        # particle's own faces when theta lands in the kernel's zero tail).
        layers = int(math.ceil(2.0 * p.cfl_target)) + 2
        for gname, face_open in (
            ("u", open_u), ("v", open_v), ("w", open_w),
        ):
            valid = grids[gname + "_valid"] & face_open
            grids[gname], _ = self._extrapolate(
                grids[gname], valid, layers, allowed=face_open)
            old[gname], _ = self._extrapolate(
                old[gname], valid, layers, allowed=face_open)
            # Re-enforce no-through defensively after extrapolation.
            grids[gname] = xp.where(face_open, grids[gname], 0.0)
            old[gname] = xp.where(face_open, old[gname], 0.0)

        # G2P: FLIP/PIC blend (Sec 3.9, standard operator).
        u_new = self._sample_faces(grids, self.pos)
        u_old = self._sample_faces(old, self.pos)
        a = p.flip_blend
        self.vel = (a * (self.vel + (u_new - u_old)) + (1.0 - a) * u_new)

        # Temporal jitter with residual carryover (Eq. 10-11, Alg. 1 l.23-28).
        n = self.pos.shape[0]
        if p.st_enabled and p.jitter_strength > 0.0:
            gamma = p.jitter_strength * xp.ones((n,), dtype=xp.float32)
            if p.adaptive_gamma:
                local_cfl = _norm_rows(xp, self.vel) * dt / p.dx
                gamma = gamma * kernels.smoothstep(xp, 0.0, 1.0, local_cfl)
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
        self._grids = grids
        self._dt_prev = dt
        self.time += dt

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
                # No state remains to evolve. Advancing the clock directly
                # avoids an otherwise catastrophic empty O(grid) projection
                # and extrapolation pass on production-resolution domains.
                self.time += t_rem
                stats.inactive_time_s += t_rem
                t_rem = 0.0
                break
            dt = min(p.cfl_target * p.dx / max(vmax, 1e-6), t_rem)
            # Subdivide the remaining frame time into even parts (Alg. 1 l.7).
            dt = t_rem / math.ceil(t_rem / dt)
            stats.particle_cfl_estimated_values.append(vmax * dt / p.dx)
            self._step(dt, stats)
            vmax = (float(xp.max(_norm_rows(xp, self.vel)))
                    if self.pos.shape[0] else 0.0)
            stats.particle_cfl_actual_values.append(vmax * dt / p.dx)
            stats.steps += 1
            stats.dt_values.append(dt)
            t_rem -= dt
        stats.n_particles = int(self.pos.shape[0])
        stats.max_speed = vmax
        return stats

    def _seed_inflows(self) -> None:
        if not self._inflows:
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
        for mask, velocity_field in self._inflows:
            refill_device = mask & (
                counts < 0.5 * self.p.particles_per_cell
            )
            refill = self.be.to_numpy(refill_device)
            cells = np.argwhere(refill)
            seeded = self._seed_cells(cells, velocity_field)
            if seeded:
                # Later sources in registration order observe particles just
                # seeded by earlier sources, preventing overlapping masks from
                # filling the same cell twice. Each selected cell receives ppc
                # particles, exactly matching _seed_cells.
                counts = counts + refill_device.astype(
                    xp.float32
                ) * self.p.particles_per_cell

    # ----------------------------------------------------------------- export

    def get_render_particles(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions re-synchronised to the global time (Alg. 1 l.31-34) and
        velocities, as host arrays."""
        if self.pos.shape[0] == 0 or not self._grids:
            return (self.be.to_numpy(self.pos), self.be.to_numpy(self.vel))
        has_outflows = self._has_volume_outflow or self._has_pressure_outflow
        if has_outflows:
            pos, volume_removed, pressure_removed = self._advect(
                self._grids,
                self.pos.copy(),
                self.dt_resid,
                track_outflows=True,
            )
            keep = ~(volume_removed | pressure_removed)
            return self.be.to_numpy(pos[keep]), self.be.to_numpy(self.vel[keep])
        pos = self._advect(self._grids, self.pos.copy(), self.dt_resid)
        return self.be.to_numpy(pos), self.be.to_numpy(self.vel)
