# Design: tiled sparse grid

Status: **design + first increment shipped.** This document specifies a fully
tiled sparse grid for the ST-FLIP solver, explains why it is a large
undertaking, and describes the small, safe increment that is already in the
codebase (the pressure-solve active-region crop).

## Motivation

The solver stores every field on dense 3-D arrays sized to the whole Domain.
For free-surface liquid that is wasteful: a splash, a stream, or a thin sheet
occupies a small fraction of the Domain, yet advection, the transfer, and above
all the pressure projection touch every cell.

Two mechanisms already reduce this waste:

1. **Whole-solver sparse crop** (`Params.sparse`). Each step the solver crops
   *all* grids to the axis-aligned bounding box of the active fluid (plus a halo
   of active blocks). This is a large win for a compact splash. It disengages
   when it cannot be applied safely — outflows, cut-cell node-SDF obstacles, or
   Two-Phase filling the Domain — because those need the full grid.

2. **Pressure-solve active crop** (`pressure.crop_to_active`, shipped here).
   Independently of `sparse`, the pressure solve crops its own linear system to
   the active bounding box before the CG runs. This helps in exactly the cases
   where the whole-solver crop is disengaged but the *liquid* is still localized
   (e.g. a localized pour into a Domain that also has a drain).

Both are **bounding-box** strategies. Their weakness is the same: a bounding box
is still dense. An active region that is thin or hollow — a sheet spanning the
Domain, an annulus, a shell of foam around a bubble — has a bounding box nearly
as large as the Domain even though few cells are live. Grid-independent solves
and billion-cell scenes need true sparsity: **store and compute only live
cells.**

## The bounding-box increment (shipped)

`pressure.crop_to_active(xp, rhs, kx, ky, kz, liquid)` computes the tight active
bounding box, pads it by a one-cell inactive halo (clamped at real Domain
boundaries), slices the right-hand side, the three MAC face-coefficient arrays,
and the mask to that box, and returns a `scatter` closure that writes the
sub-solution back into a full-size zero array. Both `pressure.solve` and
`multigrid.solve` call it first and, when the box is meaningfully smaller than
the grid, solve the cropped system and scatter the result.

**Why it is safe.** The halo cells are inactive, so the operator is unchanged:

- Exterior Dirichlet terms in `apply_laplacian` fire on halo cells but multiply
  a zero pressure, contributing nothing.
- Each active cell reaches its true neighbours through the *sliced, unchanged*
  face coefficients; a neighbour just outside the box is inactive (pressure 0),
  exactly as on the full grid.
- Where the box meets a real Domain boundary the halo is clamped away, so
  genuine `p = 0` open-boundary faces are preserved.

The set of active cells and their residual are therefore identical, so the outer
CG's `rel <= tol` contract holds for the scattered pressure. The only difference
from the full-grid solve is the floating-point summation order of the
reductions, which perturbs the result at the float32 rounding level — far below
`tol`. Tests assert this (`tests/test_multigrid.py`): the cropped and full-grid
solutions agree to within `1e-4 * scale`, inactive cells stay zero, and the
result is deterministic.

**Measured effect.** On a `96^3` Domain with a `~16^3` liquid blob (0.5% of
cells active), the pressure solve drops from ~650 ms to ~6 ms (Jacobi) and from
~1770 ms to ~13 ms (multigrid) — a ~100–140× reduction — because the CG now runs
on a `~18^3` box instead of the full grid. This is the ceiling of what a
bounding-box strategy can do; a thin full-span sheet would see little benefit.

## Full tiled sparse grid (proposed)

The target is a grid whose storage and compute are proportional to the number of
**active tiles**, independent of Domain size.

### Data structure

- **Tile.** A fixed `T×T×T` block of cells (e.g. `T = 8`), stored densely. A
  tile is the unit of allocation, activation, and (on GPU) a natural thread
  block.
- **Tile table.** A hash map (or a dense coarse index array of size
  `⌈N/T⌉³` for moderate Domains) from tile coordinate → slot index in a packed
  array of allocated tiles. Empty tiles have no slot.
- **Active set.** The set of tiles that contain fluid, plus a one-tile halo so
  stencils and the transfer can reach neighbouring cells. Rebuilt (incrementally
  updated) each step from particle positions.
- **Packed field arrays.** Each field (`u, v, w, p, phase, masks, apertures, …`)
  is a `(num_active_tiles, …)` array. Cell `(i,j,k)` is addressed by
  `table[tile_of(i,j,k)]` then the in-tile offset.

### Operations

- **Scatter/gather (P2G/G2P).** Particles index their tile via the table; a
  particle whose stencil crosses a tile boundary reads/writes the neighbour tile
  through the halo. Particles are best **sorted by tile** each step (a counting
  sort on tile id) so a tile's particles are contiguous — this is also what
  makes the GPU transfer coalesced.
- **Stencil operators** (divergence, `apply_laplacian`, smoothing). Run per
  active tile; cross-tile neighbours come from the halo. Faces on the boundary
  between an active and an inactive tile use the existing inactive-neighbour
  (Dirichlet-0 / solid) rules — the same logic the bounding-box crop relies on,
  now applied per tile.
- **Pressure solve.** The CG operates over packed tiles. The geometric
  multigrid already implemented coarsens naturally: a coarse tile aggregates
  `2×2×2` fine tiles, and the active set coarsens by "any active child". The
  V-cycle restriction/prolongation become tile-local gather/scatter. The coarse
  levels can switch from tiled back to a small dense grid once the active set is
  small enough — the current dense multigrid becomes the coarse-grid solver.

### Halo exchange

Each step, after the active set is known, fill each active tile's one-cell halo
from its six (or 26, including edges/corners for the transfer) neighbours,
reading zeros/solid for inactive neighbours. On GPU this is a single pass keyed
by the tile table. The bounding-box crop is the degenerate one-tile-per-axis
version of this exchange.

### GPU considerations

- Tiles map to thread blocks; in-tile cells to threads. Shared memory holds a
  tile plus its halo for stencil passes.
- The tile table lives in device memory; lookups are `O(1)` (hash) or a single
  indexed load (dense coarse index).
- Allocation churn is the main cost. Amortize it: keep a free list of tile
  slots, grow the packed arrays geometrically, and only deallocate tiles that
  have been empty for several steps (hysteresis) to avoid thrashing at a moving
  interface.

### Integration points (what a full implementation touches)

- `backend`/new `tiles` module — tile table, active-set maintenance, halo
  exchange, particle-by-tile sort.
- `solver` — replace dense field allocation and every whole-grid op with
  tiled equivalents; maintain the active set from particles each step.
- `pressure` / `multigrid` — packed-tile operators and a tiled V-cycle with a
  dense coarse-grid fallback.
- `apertures`, `surface_tension`, `viscosity`, `forces` — per-tile stencils.
- `cache` — the checkpoint would store the active tile set and packed fields
  (a schema version bump), or reconstruct tiles from particles on load.
- Tests — tiled/dense equivalence at every layer, plus GPU parity.

## Why this is deferred

A correct tiled solver is a substantial rewrite of the solver core and every
stencil, with new failure modes (halo staleness, table/particle desync,
allocation thrash) that must each be tested to the standard the dense path holds
today. It is the right architecture for billion-cell production scenes, but it
is not a small change, and shipping it half-done would regress the reliability
that the current dense-plus-crop path provides.

The bounding-box crop delivered here captures a large fraction of the benefit
for the common case (a localized flow) at negligible risk, and it establishes
the inactive-neighbour semantics the full tiling will reuse per tile. It is the
correct first increment on this path.
