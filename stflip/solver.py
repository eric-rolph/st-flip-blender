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

import math
from dataclasses import dataclass, field

import numpy as np

from . import kernels, pressure
from .backend import Backend, get_backend

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
    st_enabled: bool = True           # False -> plain FLIP (gamma = 0, W_T = const)
    jitter_strength: float = 1.0      # base gamma
    adaptive_gamma: bool = True       # attenuate jitter in calm regions (Sec 3.10)
    eta_phi: float = 0.5              # phase-transition steepness
    eps_m: float = 1e-9               # under-sampled face threshold
    eps_rho_rel: float = 1e-3         # eps_rho = eps_rho_rel * rho
    pcg_tol: float = 1e-4
    pcg_max_iter: int = 400
    cfl_local: float = 1.0            # advection sub-step bound
    max_substeps: int = 40
    seed: int = 0


@dataclass
class FrameStats:
    steps: int = 0
    dt_values: list = field(default_factory=list)
    pcg_iters: list = field(default_factory=list)
    n_particles: int = 0
    max_speed: float = 0.0


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

        # Solid signed distance, cell-centred; positive outside solids.
        self.sdf = xp.full(self.shape, 1e9, dtype=xp.float32)
        self._sdf_grad = None
        self._solid_faces = None  # computed lazily from sdf

        self._inflows: list[tuple] = []  # (cell_mask device bool, velocity xp(3,))
        self._rng = np.random.default_rng(params.seed)
        self._dt_prev = params.frame_dt / max(params.cfl_target, 1.0)
        self.time = 0.0

        self._grids: dict = {}
        self.m0 = self._calibrate_m0()

    # ------------------------------------------------------------------ setup

    def set_solid_sdf(self, sdf_cells: np.ndarray) -> None:
        """Cell-centred signed distance to solids (positive = outside)."""
        assert sdf_cells.shape == self.shape
        xp = self.be.xp
        self.sdf = self.be.from_numpy(sdf_cells.astype(np.float32))
        gx, gy, gz = xp.gradient(self.sdf, self.p.dx)
        self._sdf_grad = (gx, gy, gz)
        self._solid_faces = None  # rebuild on next step

    def add_liquid_mask(self, cell_mask: np.ndarray, velocity=(0.0, 0.0, 0.0)) -> int:
        """Seed particles_per_cell jittered particles in every masked cell."""
        cells = np.argwhere(cell_mask)
        return self._seed_cells(cells, velocity)

    def add_inflow(self, cell_mask: np.ndarray, velocity=(0.0, 0.0, 0.0)) -> None:
        xp = self.be.xp
        self._inflows.append(
            (self.be.from_numpy(cell_mask.astype(bool)),
             xp.asarray(velocity, dtype=xp.float32))
        )

    def _seed_cells(self, cells: np.ndarray, velocity) -> int:
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
        vel = np.broadcast_to(np.asarray(velocity, dtype=np.float32), (n, 3)).copy()
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
        theta = xp.clip(-self.dt_resid / max(dt_prev, 1e-12), -0.5, 0.5)
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
            xi = gp - xp.asarray(off, dtype=xp.float32)
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
            # Space-time phase field from the weight accumulators (Eq. 13).
            grids[g + "_phi"] = xp.minimum(
                xp.sqrt((m / self.m0) ** p.eta_phi), 1.0)
        grids["c_phi"] = xp.minimum(
            xp.sqrt((grids["c_m"] / self.m0) ** p.eta_phi), 1.0)
        return grids

    # ------------------------------------------------------------- grid utils

    def _solid_face_masks(self):
        if self._solid_faces is not None:
            return self._solid_faces
        xp = self.be.xp
        nx, ny, nz = self.shape
        solid_c = self.sdf < 0.0

        su = xp.ones((nx + 1, ny, nz), dtype=bool)
        su[1:-1] = solid_c[1:] | solid_c[:-1]
        sv = xp.ones((nx, ny + 1, nz), dtype=bool)
        sv[:, 1:-1] = solid_c[:, 1:] | solid_c[:, :-1]
        sw = xp.ones((nx, ny, nz + 1), dtype=bool)
        sw[:, :, 1:-1] = solid_c[:, :, 1:] | solid_c[:, :, :-1]
        self._solid_faces = (su, sv, sw, solid_c)
        return self._solid_faces

    def _extrapolate(self, u, valid, layers: int):
        """Propagate velocities into invalid faces by neighbour averaging."""
        xp = self.be.xp
        for _ in range(layers):
            vf = valid.astype(u.dtype)
            s = xp.zeros_like(u)
            c = xp.zeros_like(u)
            for axis in range(3):
                for shift in (1, -1):
                    s += xp.roll(u * vf, shift, axis=axis)
                    c += xp.roll(vf, shift, axis=axis)
                # roll wraps around; zero the wrapped slice
                sl = [slice(None)] * 3
                sl[axis] = 0
                s[tuple(sl)] -= (xp.roll(u * vf, 1, axis=axis))[tuple(sl)]
                c[tuple(sl)] -= (xp.roll(vf, 1, axis=axis))[tuple(sl)]
                sl[axis] = -1
                s[tuple(sl)] -= (xp.roll(u * vf, -1, axis=axis))[tuple(sl)]
                c[tuple(sl)] -= (xp.roll(vf, -1, axis=axis))[tuple(sl)]
            newly = (~valid) & (c > 0)
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
            xi = gp - xp.asarray(_OFFSETS[g], dtype=xp.float32)
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
        if not bool(xp.any(need)):
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
        xp = self.be.xp
        eps = 1e-3 * self.p.dx
        lo = xp.asarray([eps] * 3, dtype=xp.float32)
        hi = xp.asarray([s - eps for s in self.size], dtype=xp.float32)
        return xp.clip(pos, lo, hi)

    def _advect(self, grids, pos, dt_act):
        """Sub-stepped RK3 (Ralston) through the grid velocity field.

        dt_act is per-particle and may be negative (used for un-jittering)."""
        xp = self.be.xp
        p = self.p
        speed = _norm_rows(xp, self.vel) if self.vel.shape[0] else dt_act * 0
        nsub = xp.ceil(
            speed * xp.abs(dt_act) / (p.dx * p.cfl_local)).astype(xp.int32)
        nsub = xp.clip(nsub, 1, p.max_substeps)
        h = dt_act / nsub.astype(xp.float32)
        max_n = int(nsub.max()) if nsub.size else 0
        for s in range(max_n):
            he = xp.where(s < nsub, h, 0.0).astype(xp.float32)[:, None]
            k1 = self._sample_faces(grids, pos)
            k2 = self._sample_faces(grids, pos + 0.5 * he * k1)
            k3 = self._sample_faces(grids, pos + 0.75 * he * k2)
            pos = pos + he * (2.0 * k1 + 3.0 * k2 + 4.0 * k3) / 9.0
            pos = self._clamp_domain(self._push_out_of_solids(pos))
        return pos

    # ------------------------------------------------------------------- step

    def _step(self, dt: float, stats: FrameStats) -> None:
        xp = self.be.xp
        p = self.p
        nx, ny, nz = self.shape
        dt_prev = self._dt_prev

        grids = self._p2g(dt_prev)
        su, sv, sw, solid_c = self._solid_face_masks()

        # Save post-P2G velocities for the FLIP delta.
        old = {g: grids[g].copy() for g in ("u", "v", "w")}

        # External forces (grid-based, Sec 3.3).
        g = p.gravity
        grids["u"] = grids["u"] + g[0] * dt
        grids["v"] = grids["v"] + g[1] * dt
        grids["w"] = grids["w"] + g[2] * dt

        # No-through on solid faces before computing the divergence.
        grids["u"] = xp.where(su, 0.0, grids["u"])
        grids["v"] = xp.where(sv, 0.0, grids["v"])
        grids["w"] = xp.where(sw, 0.0, grids["w"])

        liquid = (grids["c_phi"] >= 0.5) & (~solid_c)

        # Face coefficients k_f = dt * alpha_f / max(rho * phi_f, eps) (Eq. 15).
        eps_rho = p.eps_rho_rel * p.rho
        kx = dt / xp.maximum(p.rho * grids["u_phi"], eps_rho)
        ky = dt / xp.maximum(p.rho * grids["v_phi"], eps_rho)
        kz = dt / xp.maximum(p.rho * grids["w_phi"], eps_rho)
        kx = xp.where(su, 0.0, kx)
        ky = xp.where(sv, 0.0, ky)
        kz = xp.where(sw, 0.0, kz)

        div = ((grids["u"][1:, :, :] - grids["u"][:-1, :, :])
               + (grids["v"][:, 1:, :] - grids["v"][:, :-1, :])
               + (grids["w"][:, :, 1:] - grids["w"][:, :, :-1])) / p.dx

        # Solve sum_f k_f (p_c - p_nb)/dx^2 = -(div u*)_c on liquid cells.
        rhs = -(div) * liquid
        kx2, ky2, kz2 = kx / p.dx**2, ky / p.dx**2, kz / p.dx**2
        pr, iters, _rel = pressure.solve(
            xp, rhs, kx2, ky2, kz2, liquid, tol=p.pcg_tol,
            max_iter=p.pcg_max_iter)
        stats.pcg_iters.append(iters)

        pm = pr * liquid
        gradx = (pm[1:, :, :] - pm[:-1, :, :]) / p.dx
        grids["u"][1:-1, :, :] -= kx[1:-1, :, :] * gradx
        grady = (pm[:, 1:, :] - pm[:, :-1, :]) / p.dx
        grids["v"][:, 1:-1, :] -= ky[:, 1:-1, :] * grady
        gradz = (pm[:, :, 1:] - pm[:, :, :-1]) / p.dx
        grids["w"][:, :, 1:-1] -= kz[:, :, 1:-1] * gradz

        grids["u"] = xp.where(su, 0.0, grids["u"])
        grids["v"] = xp.where(sv, 0.0, grids["v"])
        grids["w"] = xp.where(sw, 0.0, grids["w"])

        # Extrapolate into under-sampled faces so advection sees a full field.
        layers = int(math.ceil(p.cfl_target)) + 2
        for gname, sf in (("u", su), ("v", sv), ("w", sw)):
            valid = grids[gname + "_valid"] & (~sf)
            grids[gname], _ = self._extrapolate(grids[gname], valid, layers)

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
        while t_rem > 1e-9 * p.frame_dt:
            if self.pos.shape[0] == 0:
                break
            vmax = float(xp.max(_norm_rows(xp, self.vel)))
            vmax = max(vmax, 1e-6)
            dt = min(p.cfl_target * p.dx / vmax, t_rem)
            # Subdivide the remaining frame time into even parts (Alg. 1 l.7).
            dt = t_rem / math.ceil(t_rem / dt)
            self._step(dt, stats)
            stats.steps += 1
            stats.dt_values.append(dt)
            t_rem -= dt
        stats.n_particles = int(self.pos.shape[0])
        if self.pos.shape[0]:
            stats.max_speed = float(xp.max(_norm_rows(xp, self.vel)))
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
        for mask, vel in self._inflows:
            refill = self.be.to_numpy(
                mask & (counts < 0.5 * self.p.particles_per_cell))
            cells = np.argwhere(refill)
            self._seed_cells(cells, self.be.to_numpy(vel))

    # ----------------------------------------------------------------- export

    def get_render_particles(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions re-synchronised to the global time (Alg. 1 l.31-34) and
        velocities, as host arrays."""
        if self.pos.shape[0] == 0 or not self._grids:
            return (self.be.to_numpy(self.pos), self.be.to_numpy(self.vel))
        pos = self._advect(self._grids, self.pos.copy(), self.dt_resid)
        return self.be.to_numpy(pos), self.be.to_numpy(self.vel)
