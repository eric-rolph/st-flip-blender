"""Whitewater: foam/spray/bubble secondary particles (issue #11).

The paper's stated motivation for two-phase simulation is an air velocity
field "to drive mist and spray droplet dynamics for realistic, physics-based
white water effects" (Sec 4.9).  This module implements that consumer as a
lightweight secondary-particle system in the spirit of Ihmsen et al. 2012:

- Emission sites are liquid particles in the diffuse interface band whose
  trapped-air potential (relative speed against the grid field), wave-crest
  potential (outward motion against the interface normal), and kinetic energy
  exceed user thresholds.
- Secondaries are re-classified every step from the phase field at their
  position: deep in liquid = bubble, outside = spray, in the band = foam.
- Foam rides the liquid surface velocity and ages out; bubbles feel buoyancy
  plus drag toward the local field; spray flies ballistically with drag
  toward the local field -- which, with two_phase enabled, is the *air*
  velocity near the surface, exactly the paper's intended coupling.  In
  single-phase mode the extrapolated liquid field stands in.

Positions live in solver-local coordinates (same frame as ``solver.pos``);
the module is bpy-free and runs on NumPy or CuPy via the solver's backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FOAM, BUBBLE, SPRAY = 0, 1, 2


@dataclass
class WhitewaterParams:
    trapped_air_rate: float = 60.0    # emissions / (particle * potential * s)
    crest_rate: float = 40.0
    energy_min: float = 0.3           # kinetic-energy gate, 0.5 |v|^2
    energy_max: float = 15.0
    speed_min: float = 0.3            # relative-speed gate (trapped air)
    speed_max: float = 4.0
    lifetime_min: float = 0.4         # seconds, scaled by the energy gate
    lifetime_max: float = 3.0
    max_particles: int = 500_000
    buoyancy: float = 2.5             # bubble accel, in multiples of |g|
    drag: float = 5.0                 # 1/s relaxation toward the field
    bubble_phi: float = 0.6           # phi above this -> bubble
    spray_phi: float = 0.25           # phi below this -> spray
    interface_lo: float = 0.15        # emission band in phi
    interface_hi: float = 0.85
    substeps: int = 2
    seed: int = 0


class Whitewater:
    def __init__(self, solver, params: WhitewaterParams | None = None):
        self.solver = solver
        self.p = params or WhitewaterParams()
        xp = solver.be.xp
        self.pos = xp.zeros((0, 3), dtype=xp.float32)
        self.vel = xp.zeros((0, 3), dtype=xp.float32)
        self.life = xp.zeros((0,), dtype=xp.float32)
        self.kind = xp.zeros((0,), dtype=xp.int8)
        self._rng = np.random.default_rng(self.p.seed)

    # ----------------------------------------------------------- field access

    def _local(self, pos):
        """Shift positions into the (possibly sparse-windowed) grid frame."""
        s = self.solver
        if s._grid_origin is not None:
            xp = s.be.xp
            return pos - xp.asarray(
                s._grid_origin.astype(np.float32) * s.p.dx, dtype=xp.float32)
        return pos

    def _sample_vel(self, pos):
        s = self.solver
        return s._sample_faces(s._grids, self._local(pos))

    def _sample_phi(self, pos):
        s = self.solver
        return s._sample_cells(s._grids["c_phi"], self._local(pos))

    # -------------------------------------------------------------- emission

    def _emit(self, dt: float) -> int:
        s = self.solver
        p = self.p
        xp = s.be.xp
        n = s.pos.shape[0]
        if n == 0:
            return 0
        liquid = s.phase > 0.5
        phi = self._sample_phi(s.pos)
        band = liquid & (phi > p.interface_lo) & (phi < p.interface_hi)

        v_field = self._sample_vel(s.pos)
        rel = s.vel - v_field
        rel_speed = xp.sqrt((rel * rel).sum(axis=1))
        pot_ta = xp.clip((rel_speed - p.speed_min)
                         / max(p.speed_max - p.speed_min, 1e-6), 0.0, 1.0)

        # Wave-crest proxy: motion along the outward interface normal
        # (-grad phi points out of the liquid).
        gx = xp.gradient(s._grids["c_phi"], s.p.dx, axis=0)
        gy = xp.gradient(s._grids["c_phi"], s.p.dx, axis=1)
        gz = xp.gradient(s._grids["c_phi"], s.p.dx, axis=2)
        local = self._local(s.pos)
        nvec = xp.stack([s._sample_cells(gx, local),
                         s._sample_cells(gy, local),
                         s._sample_cells(gz, local)], axis=1)
        norm = xp.sqrt((nvec * nvec).sum(axis=1))
        out_n = -nvec / xp.maximum(norm, 1e-9)[:, None]
        v_out = (s.vel * out_n).sum(axis=1)
        pot_wc = xp.clip(v_out / max(p.speed_max, 1e-6), 0.0, 1.0)

        ke = 0.5 * (s.vel * s.vel).sum(axis=1)
        pot_ke = xp.clip((ke - p.energy_min)
                         / max(p.energy_max - p.energy_min, 1e-6), 0.0, 1.0)

        expect = xp.where(
            band,
            dt * (p.trapped_air_rate * pot_ta + p.crest_rate * pot_wc)
            * pot_ke,
            0.0)
        u = s.be.from_numpy(self._rng.random(n, dtype=np.float32))
        count = xp.floor(expect).astype(xp.int32) + (
            u < (expect - xp.floor(expect))).astype(xp.int32)
        total = int(count.sum())
        if total == 0:
            return 0

        # Expand emitters on the host (counts are small relative to n).
        count_h = s.be.to_numpy(count)
        src = np.repeat(np.arange(n), count_h)
        src_d = s.be.from_numpy(src.astype(np.int64))
        jitter = s.be.from_numpy(
            (self._rng.random((total, 3), dtype=np.float32) - 0.5)
            * np.float32(s.p.dx))
        vj = s.be.from_numpy(
            (self._rng.random((total, 3), dtype=np.float32) - 0.5) * 0.2)
        new_pos = s.pos[src_d] + jitter
        new_vel = s.vel[src_d] * (1.0 + vj)
        span = max(p.lifetime_max - p.lifetime_min, 0.0)
        life_u = s.be.from_numpy(self._rng.random(total, dtype=np.float32))
        new_life = (p.lifetime_min + span * life_u) * xp.clip(
            pot_ke[src_d] + 0.2, 0.0, 1.0)
        new_kind = xp.zeros((total,), dtype=xp.int8)

        self.pos = xp.concatenate([self.pos, new_pos.astype(xp.float32)])
        self.vel = xp.concatenate([self.vel, new_vel.astype(xp.float32)])
        self.life = xp.concatenate([self.life, new_life.astype(xp.float32)])
        self.kind = xp.concatenate([self.kind, new_kind])
        return total

    # ---------------------------------------------------------------- update

    def _classify(self) -> None:
        xp = self.solver.be.xp
        phi = self._sample_phi(self.pos)
        self.kind = xp.where(
            phi > self.p.bubble_phi, np.int8(BUBBLE),
            xp.where(phi < self.p.spray_phi, np.int8(SPRAY),
                     np.int8(FOAM))).astype(xp.int8)

    def step(self, dt: float) -> dict:
        """Advance one output frame; call after ``solver.step_frame()``."""
        s = self.solver
        p = self.p
        xp = s.be.xp
        if not s._grids:
            return self.counts()
        emitted = self._emit(dt)
        if self.pos.shape[0] == 0:
            return self.counts()

        g = xp.asarray(s.p.gravity, dtype=xp.float32)
        g_mag = float(np.linalg.norm(s.p.gravity))
        h = dt / max(int(p.substeps), 1)
        for _ in range(max(int(p.substeps), 1)):
            self._classify()
            v_field = self._sample_vel(self.pos)
            foam = self.kind == FOAM
            bubble = self.kind == BUBBLE
            spray = self.kind == SPRAY
            # Foam: ride the surface flow exactly.
            vel = xp.where(foam[:, None], v_field, self.vel)
            # Bubbles: buoyancy against gravity plus drag toward the field.
            up = (-g / max(g_mag, 1e-9)) if g_mag > 0 else g * 0.0
            acc_b = up[None, :] * (p.buoyancy * g_mag)
            vel = xp.where(
                bubble[:, None],
                vel + h * (acc_b + p.drag * (v_field - vel)),
                vel)
            # Spray: ballistic plus drag toward the (air) field.
            vel = xp.where(
                spray[:, None],
                vel + h * (g[None, :] + p.drag * (v_field - vel)),
                vel)
            self.vel = vel.astype(xp.float32)
            self.pos = self.pos + h * self.vel
            # Ageing: foam and spray expire; bubbles persist until they
            # surface (where classification turns them into foam).
            decay = xp.where(self.kind == BUBBLE, 0.1, 1.0)
            self.life = self.life - h * decay

        # Cull: expired or outside the domain; then cap at max_particles by
        # dropping the oldest survivors.
        size = xp.asarray(s.size, dtype=xp.float32)
        alive = (self.life > 0.0) & xp.all(
            (self.pos >= 0.0) & (self.pos <= size[None, :]), axis=1)
        keep = alive
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.life = self.life[keep]
        self.kind = self.kind[keep]
        if self.pos.shape[0] > p.max_particles:
            start = self.pos.shape[0] - p.max_particles
            self.pos = self.pos[start:]
            self.vel = self.vel[start:]
            self.life = self.life[start:]
            self.kind = self.kind[start:]
        out = self.counts()
        out["emitted"] = emitted
        return out

    # ---------------------------------------------------------------- export

    def counts(self) -> dict:
        n = int(self.pos.shape[0])
        if n == 0:
            return {"total": 0, "foam": 0, "bubble": 0, "spray": 0}
        kind = self.solver.be.to_numpy(self.kind)
        return {
            "total": n,
            "foam": int((kind == FOAM).sum()),
            "bubble": int((kind == BUBBLE).sum()),
            "spray": int((kind == SPRAY).sum()),
        }

    def get_render_particles(self):
        """Host arrays (pos, vel, kind, life) in solver-local coordinates."""
        be = self.solver.be
        return (be.to_numpy(self.pos), be.to_numpy(self.vel),
                be.to_numpy(self.kind), be.to_numpy(self.life))
