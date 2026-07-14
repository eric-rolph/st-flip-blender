# Validation artifacts

This directory contains auditable validation outputs, not undocumented
reference truth. The checked `stflip_matched_cpu.json` is the separate matched
ST-FLIP/instantaneous-ablation artifact described in the top-level README.

The compact paper-reference runner writes new strict, atomic JSON artifacts for
two reproducible scene definitions:

```bash
python tools/run_paper_validation.py \
  --case kleefsman \
  --resolution 32 \
  --output validation/kleefsman-quick.json

python tools/run_paper_validation.py \
  --case glug \
  --resolution 32 \
  --output validation/glug-quick.json
```

Use `--backend cuda` only on a verified CUDA installation. These compact runs
use the CLI defaults (`resolution=32`, 2 particles/cell) and exercise geometry
and metrics; they are smoke/regression settings, not validation-fidelity or
production-scale timing evidence. Resolution, `dx`, duration, frame rate, and
sampling must be selected for the claim being tested.

## Kleefsman geometry and gauges

The scene encodes the published 3-D tank, initial water block, obstacle, and H2
and H4 x-coordinates from Kleefsman et al., *A Volume-of-Fluid Based Simulation
Method for Wave Impact Problems*,
[DOI 10.1016/j.jcp.2004.12.007](https://doi.org/10.1016/j.jcp.2004.12.007).

The published benchmark and this solver both place `x = 0` at the downstream
(left) wall, with x increasing toward the upstream reservoir. The artifact
records the identical published and solver coordinates for the gate, obstacle
centre, and H2/H4 gauges; no reflection or relabeling is applied. CSV columns
therefore use the published labels `H2_m` and `H4_m` directly.

The widely published x-coordinates do not completely define this particle
sampler. The artifact therefore records the centreline y-coordinate, a
`1.5Δx` cylindrical footprint, and the bottom-connected occupied-layer height
estimator. Detached spray above a gap is ignored. These are reproducible
implementation choices, not experimental probe metadata inferred from a
figure.

For an illustrative user-selected `Δx = 0.02 m`, 8-PPC, 7-second run at 24 FPS:

```bash
python tools/run_paper_validation.py \
  --case kleefsman \
  --dx 0.02 \
  --frames 168 \
  --particles-per-cell 8 \
  --output validation/kleefsman-dx002.json
```

For the published tank extents this requests a `161 × 50 × 50` grid. It is an
explicit run configuration, not a hidden claim that resolution alone reproduces
the experimental curve.

Choose frame count/rate to cover the reference interval you intend to compare;
the runner does not silently extrapolate beyond simulated time.

## External reference CSV

No experimental water-height series is bundled or digitized from a plot. To
make an experimental comparison, provide an attributable numeric CSV and its
citation:

```csv
time_s,H2_m,H4_m
0.00,0.000,0.550
0.05,0.000,0.551
```

The numbers above only illustrate the file shape; they are **not reference
data**. Requirements are:

- `time_s` plus at least one gauge column named like `H2_m` or `H4_m`;
- complete, finite numeric values with non-negative heights;
- non-negative, strictly increasing times;
- at least two reference samples overlapping the simulated interval for each
  compared gauge.

Run a comparison and make the presence of reference data mandatory:

```bash
python tools/run_paper_validation.py \
  --case kleefsman \
  --dx 0.02 \
  --frames 168 \
  --particles-per-cell 8 \
  --reference-csv path/to/kleefsman-gauges.csv \
  --reference-citation "Kleefsman et al., JCP 206(1), DOI:10.1016/j.jcp.2004.12.007; numeric dataset provenance ..." \
  --require-reference \
  --max-gauge-rmse 0.05 \
  --output validation/kleefsman-compared.json
```

`--reference-citation` is required whenever a CSV is supplied. The artifact
stores the citation, source filename, and SHA-256 of the exact bytes. Simulated
heights are interpolated onto overlapping reference timestamps; each matching
gauge receives sample count, RMSE, MAE, bias, peak-height error, and peak-time
error. `--max-gauge-rmse` returns exit code 2 when any compared gauge exceeds
the threshold. Select and justify the threshold before running an experiment.

## Paper-constrained glug scope

The glug case encodes the ST-FLIP paper's published scale ratios: square side
`0.5L`, connector radius and length `0.05L`, and upper water height `0.5L`
([DOI 10.1145/3811289](https://doi.org/10.1145/3811289)). Container internal
height, margin, and the wall/cavity model are explicit assumptions in the JSON.
The samples track liquid/gas distribution and liquid centre of mass.

The paper says surface tension was enabled for this figure but does not publish
its coefficient. The runner therefore records an explicit `0.072 N/m`
water-air assumption by default; override it with `--glug-surface-tension` when
testing another declared value.

This is a fixed-seed two-phase regression scene. A CPU run in the same software
environment is intended to be reproducible; CUDA is checked by numerical
tolerance, not promised bit-for-bit. The scene has no golden reference baseline,
does not claim that unpublished production or Blender geometry was recovered,
does not compare against a PF-FLIP implementation, and does not establish the
paper's production-scale performance.

## Artifact contract

Paper-reference artifacts use schema `stflip-paper-reference-v1` and include:

- declared claim scope and geometry provenance/assumptions;
- requested/effective grid extents, `Δx`, shape, and cell count;
- the normalized solver `Params`, requested/used backend, package version, and
  a platform-normalized SHA-256 over the packaged Python source;
- Python, NumPy, operating-platform, array-library, optional CuPy, and CUDA
  device runtime metadata;
- all run controls, boundary/physics assumptions, and per-frame solver/pressure
  statistics;
- fixed-seed scene-specific samples;
- for Kleefsman, either explicit `null` reference/comparison fields or the
  hashed reference declaration and computed errors.

JSON writing rejects `NaN`/`Inf` and replaces the destination atomically. Keep
the command line, add-on commit, backend/hardware details, and any external
dataset license with an evidence artifact intended for publication.
