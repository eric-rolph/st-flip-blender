# Object roles

A scene is defined by a **Domain** plus objects carrying **Roles** in *ST-FLIP
→ Active Object*. Domain, Liquid, Inflow, Outflow and Obstacle are mesh-only;
Force Field accepts a mesh or Empty.

## Domain

A box that defines the simulation volume and the grid. Everything the solver
does happens inside it; fluid that leaves is gone. **Resolution** counts cells
along the Domain's longest axis, and the other axes get proportional cell
counts, so the Domain's proportions set the grid's proportions.

Only one Domain is used. Changing the Domain size, resolution, or scene FPS
after a bake invalidates resume (the grid would no longer match the checkpoint).

## Liquid

A closed mesh that is **filled with fluid at frame one**. Use it for a starting
body of water — a pool, a block for a dam-break, a sphere to drop.

Liquid objects can be given an **initial velocity**:

- **Uniform** — one velocity vector for the whole volume.
- **Linear / solid-body rotation** — a rotational field about an axis (set the
  centre, axis, and angular speed) for swirls and spinning volumes.

## Inflow

A closed mesh that **emits fluid over time** — a faucet, a jet, a hose. Inflow
is occupancy-based: it keeps its volume topped up with fluid each step, rather
than prescribing a strict volumetric flow rate.

Key options:

- **Inflow velocity** — uniform or solid-body, same as Liquid. A downward
  velocity makes a pour; an upward velocity makes a fountain jet.
- **Frame range** — limit emission to an inclusive `[start, end]` scene-frame
  window so a source turns on and off.
- **Emit as gas** — only visible with **Two-Phase** enabled; emits air instead
  of liquid (used for bubbling/glugging).

## Outflow

A closed mesh that **removes fluid**. Two modes:

- **Volume** — deletes any particle that enters the mesh volume. A simple
  interior drain. It is *not* a pressure boundary.
- **Pressure** — opens the simulation-domain faces it intersects to
  atmospheric pressure (`p = 0` half a cell outside). This is the physically
  correct open boundary for a drain/outlet; it must touch a Domain boundary.

Use **Volume** for "make this fluid disappear here"; use **Pressure** for "the
tank has a real hole to the outside".

## Obstacle

A closed mesh the fluid **flows around**. Obstacles are voxelised into the grid
as solids with cut-cell face apertures, so partially covered cells behave
correctly rather than blocking a whole cell.

- **Animated (Moving Wall)** — re-voxelizes the obstacle once per output frame.
  Rigid transforms use transform differencing. Stable-topology deformation uses
  nearest evaluated-vertex displacement when the object transform is unchanged.

Combined transform-plus-deformation uses the rigid-velocity path. Topology
changes have no deformation velocity. Resume re-samples animated motion and may
differ slightly from an uninterrupted bake.

## Force Field

An **art-directed body force** applied to the fluid. Use a mesh or Empty as the
guide; its geometry is not voxelized. Set **Force Type**:

- **Directional** — a constant push (wind) along the object's local +Z.
- **Vortex** — swirl about the object's +Z axis, within a **radius**.
- **Turbulence** — divergence-free curl-noise for chop and churn, with a
  spatial **scale**.

Directional uses local **+Z**. Vortex uses the origin as its centre and local
**+Z** as its axis. **Strength** scales the effect.

Turbulence is domain-wide and uses **Strength**, **Scale**, and the simulation
seed; the guide transform does not localize it. Forces are how the *Stormy
Pool* preset gets its agitation.

## How roles combine

A typical scene: one **Domain**, one **Liquid** pool or one **Inflow** source,
optional **Obstacles** to interact with, an optional **Outflow** so the domain
does not simply fill up, and optional **Force Fields** for art direction. The
[recipes](recipes.md) page shows concrete combinations.
