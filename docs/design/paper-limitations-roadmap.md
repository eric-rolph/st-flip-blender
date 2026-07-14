# Design: paper-limitations roadmap

Status: **design, revised twice; nothing shipped.** Revision 1: two-lens
adversarial review (feasibility and physics) of all five original initiatives.
Revision 2 (2026-07-14): reconciliation against an independent external plan
(ChatGPT "Sol") that was judged claim-by-claim against the paper and this repo;
the reconciliation added initiative NORM (exact gamma-conditioned weighting --
the external plan's best catch, verified against paper Sec. 3.10 p.9 and
solver.py), initiative ERR (error-aware step control, reviewed and corrected),
and a set of smaller adoptions recorded in place below; everything the external
plan proposed that was rejected is listed under Non-goals with reasons. This
document is the coordinated plan for addressing the limitations (Sec. 5) and
outlook items (Sec. 6) of the paper this addon implements: "Spatiotemporal FLIP
for Fast Free-Surface and Two-Phase Simulation With Very Large Time Steps"
(Braun et al., SIGGRAPH 2026). It contains seven initiatives, each split into
independently landable milestones, plus a global recommended order and explicit
non-goals. Every initiative has been through an adversarial review pass; the
designs here are the corrected versions, with reviewer-mandated changes folded
in and unresolved items promoted to explicit open questions.

**Anchor caveat.** All `file:line` anchors were verified against v0.23.1. A
parallel "codex" agent lands PRs on this repo continuously, so anchors drift:
treat every anchor as a starting hint, run `git fetch` first, and re-verify
with grep before editing. The described functions and invariants are the
stable reference; the line numbers are not.

## Motivation

The paper lists four limitations and three outlook directions:

- **L1** -- quality still degrades as the time step grows. Inherent to the
  method; just more gradual than prior work.
- **L2** -- FLIP-typical, time-step-dependent loss of **energy and angular
  momentum** from the first-order advect-then-project splitting is not
  addressed. The paper says ST-FLIP is in principle compatible with
  countermeasures: advection-reflection (Zehnder et al. 2018), IVOCK (Zhang
  et al. 2015), or vorticity confinement (Fedkiw et al. 2001).
- **L3** -- Monte Carlo **noise**, worst at very high CFL on **calm water
  surfaces** where the signal-to-FLIP-noise ratio is low. Current mitigation
  (paper Sec. 3.10, already implemented here as adaptive attenuation):
  velocity-based damping of the temporal jitter. The noise is mostly
  low-amplitude, unbiased, and uncorrelated -- hence amenable to surface
  denoising.
- **L4** -- less effective for **surface-tension-dominated** flows: surface
  tension imposes dt < O(dx^(3/2)) (Brackbill et al. 1992), capping how far
  the velocity CFL can be raised. This repo already has explicit CSF surface
  tension plus the capillary dt clamp.
- **O1** -- replace pseudo-random temporal jitter with **low-discrepancy
  sequences** (scrambled Sobol/Halton) for variance reduction.
- **O2** -- retain a small number of **distinct time levels** in the slab for
  higher-order temporal reconstruction.
- **O3** -- generalize the spatiotemporal viewpoint to MPM/SPH. **Explicit
  non-goal for this repo** (see Non-goals).

Initiative map: SAMP (O1 -> L3), NORM (Sec. 3.10 exact normalization -> L3),
ENER (L2), TIME (O2 -> L3 + L1), CALM (L3 beyond Sec. 3.10), CAP (L4),
ERR (L1 as a budget instead of an inherent loss).

## Shared conventions and decisions

These bind every milestone below; a future agent should read this section
before picking up any milestone.

**Repo constraints.** Dependencies are NumPy only (no scipy -- Blender's
Python does not ship it). CuPy is optional; all solver math is xp-agnostic;
no `cupy.linalg` anywhere (cuBLAS is missing on the target 5090 setup); no
host-device syncs inside iteration loops; randomness is host-side PCG64
unless a milestone explicitly says otherwise. Code is ASCII-only with no
semicolons; CI runs `uvx ruff check --isolated stflip addon tests tools`
plus a NumPy-only pytest. `stflip/` stays bpy-free. The pytest `gpu` marker
is excluded by default: **there is no GPU CI**; CuPy parity is verified
manually on the local RTX 5090.

**Decision 1 -- Params-fingerprint invalidation is accepted.**
`simulation_fingerprint` (addon/operators.py:495-529) hashes every Params
field via `vars(params)`. Any new Params field therefore changes the
fingerprint of every scene, and resume of any pre-upgrade bake cache is
refused (addon/handlers.py:571-576, operators.py:696-732) even when the new
field holds its default. This is existing repo precedent. We accept it: no
default-elision from the fingerprint (that would weaken the guarantee).
Every milestone that adds a Params field must carry a release-note line
saying "existing bakes cannot resume after upgrade; re-bake", and Params
additions within one initiative should be batched into one PR where
practical so users see one invalidation, not several.

**Decision 2 -- exactly one metrics frame-record schema bump.**
`metrics.FRAME_FIELD_ORDER` (metrics.py:22-61) is a strict schema:
`validate_frame_record` (metrics.py:385-393) rejects any record whose key
set or `schema_version` differs, and `cache.read_metrics`
(cache.py:840-867) silently drops non-validating rows, so a bump makes all
previously cached metrics rows invisible. This roadmap performs **one**
deliberate SCHEMA_VERSION 2 -> 3 bump, in ENER-M0 (angular-momentum
fields). All other new metrics are standalone helper functions or
validation.py-side diagnostics that never touch the frame-record schema.

**Decision 3 -- exactly one checkpoint schema bump.** SAMP-M1 bumps the
checkpoint schema (v2 -> v3 in `_CHECKPOINT_KEYS_BY_VERSION`, cache.py:65)
to add per-particle ids and a substep counter, and RESERVES an optional,
mode-gated `gamma_prev` key (float32 per particle, written only when a bake
combines CALM-M3's `gamma_mode="surface"` with NORM's exact normalization --
see NORM; absent otherwise). No other initiative changes checkpoint state.
Older checkpoints restore with synthesized ids only in pseudo sampling mode
(see SAMP-M1). ERR's controller state is deliberately NOT checkpointed
(restore re-initializes it conservatively; see ERR).

**Decision 4 -- codex coexistence.** Small, atomic PRs. Refactors of hot
solver regions (the `_step_core` tail, `_p2g`) land first and alone with
bit-identical guards, then features build on them. Always `git fetch`
before starting.

**Decision 5 -- one shared calm-surface metric.** CALM-M1 builds the
calm-pool surface metric and scenes. SAMP-M5, TIME-M3, and ERR-M1 consume it.
It is specified once, in CALM-M1; the earlier idea of a separate metric inside
`metrics.measure_frame` for SAMP is dropped (it would have forced a second
schema bump).

**Decision 6 -- reporting frame and statistics (from the reconciliation).**
(a) The HEADLINE reporting metric for every decision writeup (SAMP-M5,
ENER-M2b, CAP-M3, TIME-M3, ERR-M3) is equal-error END-TO-END speedup,
including surface-reconstruction, render-resync, and denoise wall time --
not solver-only step time. ST-FLIP's per-step win partly comes from deleting
per-step surface reconstruction (paper Alg. 1 line 17), and denoise cost
scales with the noise being traded, so solver-only timings overstate wins.
CALM-M1 adds the instrumentation (surfacing + resync wall time, peak-memory
probe). Mechanistic pass/fail gates stay as specified per milestone --
"equal error" needs a reference the addon lacks; the matched-CFL degradation
machinery remains the operable proxy. (b) Gates that make a VARIANCE claim
(SAMP-M5, CALM-M4 acceptance) run 8 seeds with a confidence interval on the
gated quantity (`run_multi_seed_validation`, validation.py:884, already takes
an arbitrary seeds tuple -- config, not code). All other gates stay at seeds
{0,1,2}; CI compute budget is real. (c) Vorticity confinement (ENER-M1) must
be OFF in every validation and decision run of every initiative: it is an
artist look control, never an evaluation configuration. (d) Each adopted
feature's decision doc includes one manual GPU (RTX 5090) wall-clock table
next to the CPU numbers -- CuPy parity alone does not prevent shipping
CPU-only wins.

---

## Initiative SAMP: low-discrepancy temporal sampling (Outlook O1 -> Limitation L3)

### Goal

Replace the pseudo-random per-particle temporal jitter
`xi ~ U(-1/2, 1/2)` (draw site stflip/solver.py:2093-2104, the actual draw
at 2098-2099) with a hash-based Owen-scrambled low-discrepancy sequence.
Because the residual-carryover identity makes next step's sample time
`theta_p = gamma * xi_p` (unclamped), each particle's xi sequence across
steps IS its sample-time sequence across slabs.

**Mechanism (corrected wording).** The gain is NOT that trajectory error
stops random-walking -- Appendix A Lemma A.1 already bounds
`|dt_resid| <= dt_max/2` and prevents drift in both modes. The genuine LDS
gain is: the empirical distribution of each particle's theta over any
window of K steps has star discrepancy O(log K / K) instead of
O(K^(-1/2)), so the W_T-weighted temporal quadrature error along each
trajectory decays faster, and per-face deposited error becomes temporally
anti-correlated (blue) rather than white. Per-step cross-particle scatter
at a fixed face is unchanged by construction (independent per-particle
scrambles), so the honest expectation is a modest, temporally visible gain
on calm surfaces at CFL 8-16 -- gated by an A/B experiment (SAMP-M5) before
any default change. O1 is unproven future work; this ships opt-in.

**Estimator status (corrected wording).** Hash-based Owen under a fixed key
is deterministic: a single bake is consistent QMC with discrepancy-bounded
error, approximately unbiased over the key/seed ensemble (the LK key path
is not an exact bijection in the key, and the flow state at step n depends
on prior deviates, so conditional-on-state uniformity -- which i.i.d. draws
provide -- is only approximate). Per-bake bias is monitored by the m0
regression and the dam-break non-regression gates.

### Design

**Chosen scheme.** 1D Owen-scrambled van der Corput (Sobol dimension 0),
scrambled with the Laine-Karras-style hash from Burley 2020 ("Practical
Hash-based Owen Scrambling"), implemented in pure NumPy/CuPy uint32 integer
arithmetic in a NEW module `stflip/sampling.py`. Stateless:
`(particle_id, substep_index, seed)` fully determine every deviate, which
makes checkpoint/resume trivial and means the
`jitter_strength == 0 / st_enabled == False` skip path (solver.py:2101-2102)
cannot desynchronize anything.

Per particle: `key_p = mix64to32(particle_id, seed)` (splitmix64-style
finalizer). Exploiting `sobol_dim0_bits(n) = reverse_bits32(n)`, Owen
scrambling operates in the reversed-bit domain where the input equals the
index itself:

```
v = n_shuffled
v ^= v * 0x3d20adea
v += key
v *= (key >> 16) | 1
v ^= v * 0x05526c56
v ^= v * 0x53a22864          # all uint32 wraparound
xi = reverse_bits32(v) * 2^-32 - 1/2   # float32
```

Each `v ^= v * c` only propagates bits upward, so in the reversed domain
every output bit depends only on equal-or-lower input bits -- a valid
nested-uniform (Owen-style) scramble preserving the (0,1)-sequence property
per key.

**Index shuffling (review addition, mandatory).** Burley pairs point
scrambling with per-pixel index shuffling because different-key scrambles
of the same base point are only approximately independent. We adopt it up
front rather than as a post-hoc mitigation: `n_shuffled` above is an
Owen-style hash of the global substep counter under a SECOND per-particle
key (same id, different salt). Consequence for testing: the exact
dyadic-stratification property now holds only for index windows aligned at
multiples of 2^k, and the unit test is weakened accordingly.

**Rejected alternatives** (kept for the record): scipy.stats.qmc (scipy is
not a dependency and stateful engines complicate checkpoints); precomputed
Sobol tables (unbounded index, no benefit over a closed-form hash);
unscrambled Sobol/Halton (catastrophic -- all particles share each step's
offset, coherent phase oscillation); Cranley-Patterson rotation (pairwise
offsets frozen for all steps -> persistent structured artifacts; kept only
as an optional experimental A/B arm `cp_rot`); per-cell stratification of
offsets (reduces per-step variance but order-dependent under compaction and
destroys temporal coherence; documented follow-on, out of scope).

**CPU/GPU.** Deviates are computed with xp integer ops directly on the
native backend (device-side under CuPy). This deviates from the host-draw
convention but is justified: particle ids live device-side anyway (outflow
keep-mask compaction at solver.py:1301, 1317-1324) and bit-exact parity
comes from exact uint32 arithmetic. A gpu-marked parity test enforces
bitwise CPU/GPU identity (run manually on the 5090). The host PCG64 RNG
(solver.py:364) remains for spatial seeding and whitewater.

**Property preservation.** Only the distribution of xi changes. Lemma A.1
needs only `xi in [-1/2, 1/2]`, satisfied exactly. Marginal uniformity on
the 2^-32 lattice preserves Eq. 8 unbiasedness (in the ensemble sense
above) and the analytic `m0 = ppc` calibration (solver.py:943-954).
Adaptive gamma (solver.py:2095-2097) rescales xi exactly as today.

### New state and wiring

1. `self.particle_id` (int64, device): allocated monotonically from
   `self._next_particle_id` in `_seed_cells` (solver.py:863-898, next to
   age/source_id), filtered by the same keep-masks in
   `_apply_outflow_filter` (solver.py:1301), padded with FRESH ids (never
   duplicates -- duplicated keys silently correlate two particles) in
   `_reconcile_particle_attrs` (solver.py:371-401).
2. `self._substep_index` (int64): incremented at the commit point next to
   `self._dt_prev = dt` (solver.py:2118-2122).
3. Checkpoints: add `particle_id`, `next_particle_id`, `substep_index` to
   `checkpoint_state` (solver.py:403-446) and `restore_state`
   (solver.py:448-602). The new keys are OPTIONAL in restore with
   mode-gated enforcement: restoring an older-version checkpoint
   synthesizes ids 0..n-1 and substep_index 0 and is permitted ONLY when
   `temporal_sampling == "pseudo"`; resuming a sobol_owen bake from a
   pre-bump checkpoint is refused with a clear error. Bump the cache schema
   (Decision 3) and extend validate/write/read (cache.py:553-611, 633-689,
   692-769).
4. Validation fingerprint: include particle_id bytes and substep_index in
   `_hash_initial_checkpoint` (validation.py:138-182) ONLY when
   `temporal_sampling != "pseudo"`, so stored pseudo-mode validation
   fingerprints remain comparable across the upgrade.
5. Draw-site branch on new `Params.temporal_sampling`
   ("pseudo" default | "sobol_owen" | experimental "cp_rot") with a
   validator next to the pressure_solver check (solver.py:220-221).
   "pseudo" keeps `self._rng.random` byte-identical. Note (release note,
   Decision 1): adding the field invalidates resume of all pre-upgrade
   bakes even in pseudo mode; trajectories are byte-identical, resumability
   is not.

### Cost

Runtime: ~15 uint32 elementwise ops + a 5-step bit reversal + one float
scale per draw (plus one more short hash for index shuffling), O(n), no
reductions -- well under 1 percent of a substep vs P2G and advection. GPU:
tiny elementwise kernels, no host syncs; strictly cheaper than today's
host-draw plus 4n-byte upload. Memory (corrected): core per-particle state
is 40 B (pos 12 + vel 12 + dt_resid 4 + phase 4 + age 4 + source_id 4);
int64 ids add +20 percent on that baseline, ~10 percent under APIC where
the affine C matrix adds another 36 B/particle. Example: +80 MB at 10 M
particles. uint32 ids (+4 B) are a fallback if memory-pressed (wraps after
4.3e9 births). Checkpoints grow by the id array.

### Risks

1. Payoff visually negligible (central risk; O1 is unproven). Detection:
   SAMP-M5 gate. Mitigation: opt-in, documented experimental.
2. Structured aliasing from imperfect scrambling. Detection: PSD
   spectral-spike check on calm-pool height probes (no isolated peak > 3x
   local floor) plus rendered-sequence inspection. Mitigation: index
   shuffling is already in (above); the hash is swappable inside
   sampling.py without touching the solver.
3. Clamping after abrupt adaptive-dt changes locally breaks LDS structure
   exactly as it biases the pseudo scheme (< 1 percent of updates per the
   paper); statelessness means the sequence resumes cleanly. Verify the
   drift-bound test in sobol_owen mode.
4. Physics bias from marginal non-uniformity. Detection: aligned-window
   dyadic-stratification and uniformity unit tests plus an m0 regression in
   an LDS run; dam-break phase-RMSE gate catches downstream drift.
5. Checkpoint schema bump collides with concurrent cache.py work.
   Mitigation: SAMP-M1 is minimal and lands early (see Recommended order).
6. Mode-dependent RNG stream: enabling sobol_owen stops consuming
   `self._rng` for temporal draws, shifting later `_seed_cells` draws
   relative to pseudo mode. Expected; never compare trajectories across
   modes; fingerprints are mode-tagged.
7. Integer semantics: keep everything in array ops (NumPy emits overflow
   warnings only for scalar integer ops); the risk to guard is scalar-op
   leakage and CPU/GPU wraparound parity, covered by the bitwise parity
   test.

### Validation gates

Unit (NEW tests/test_sampling.py): (a) (0,1)-sequence property for windows
aligned at multiples of 2^k (weakened for index shuffling); (b) marginal
uniformity across keys at fixed index (pure-NumPy KS-style bounds); (c)
cross-particle deviate correlation ~ 0; (d) CPU/GPU bitwise identity
(gpu-marked, manual); (e) no scalar-op leakage. Solver: drift bound
`|dt_resid| <= dt_max/2` in sobol_owen mode (pattern of
tests/test_solver.py:442); still-pool guard (pattern of :453); particle_id
uniqueness under outflow compaction, inflow appends, reconcile padding;
checkpoint round-trip incl. old-version fallback semantics.

**SAMP-M5 A/B matrix (restructured per physics review).** The default
adaptive gamma (`Params.adaptive_gamma`, solver.py:100-101) attenuates
jitter 10-1000x in a fully quiescent pool -- exactly where the original
design tried to measure. The matrix therefore isolates the mechanism:

- Arms: (i) calm pool WITH `adaptive_gamma=False` (gamma = 1 everywhere) --
  the controlled test of the sampler itself; (ii) a paper-Fig.6-style drop
  into a quiescent pool (calm surface, local CFL near the surface keeps
  gamma active) -- the actual L3 worst case; (iii) default-config calm pool
  as a do-no-harm check with NO reduction target; (iv) the existing
  dam-break matched-CFL machinery as non-regression.
- Log the per-frame gamma_p histogram alongside the PSD so a null result
  can be attributed to attenuation rather than sampler failure.
- Settings: cfl_target 8 and 16, EIGHT seeds 0..7 with a confidence interval
  on the gated PSD-band reduction (Decision 6b -- this gate makes a variance
  claim), samplers pseudo vs sobol_owen (optional cp_rot arm), normalization
  axis exact vs legacy (NORM-M1's flag -- the reconciliation's adopted
  replacement for the external 720-run factorial). Scenes: the arm matrix
  above plus the CALM-M1 S4 translating slab (gamma fully active by
  construction; gates Galilean surface flatness). Metric: the CALM-M1
  interface-height metric (spatial RMS + temporal PSD of height probes).
- Attribution tooling: the id-parity paired-partition variance probe (built
  once in ERR's step_control.py, hash-bit parity, weighted_volume operand)
  runs off-by-default in the study to attribute null results (sampler vs
  attenuation vs cross-particle scatter). Never a controller input here.
- Writeup adds one denoise-coupling column (Decision 6a payoff channel):
  calm-region MCF iterations needed to reach a fixed surface RMS, per
  sampler/normalization arm -- measured noise reductions should let users
  LOWER denoise iterations, which is the actual end-to-end win.
- Adoption rule: on the gamma-active arms, target >= 30 percent reduction
  in the low-frequency band (below Nyquist/8) of the height PSD with
  spatial flatness no worse; spurious-KE floor no worse; no spectral
  spikes; dam-break phase RMSE and KE trace within a few percent of pseudo.
  All pass at both CFLs -> M6 proceeds (still opt-in first release).
  PSD criterion fails but others pass -> keep documented experimental.
  Spike or dam-break regression -> fix scrambling or reject.

### Milestones

- **SAMP-M1 -- stable particle ids + substep counter** (1-2 days). Zero
  behavior change; checkpoint v3 bump with backward read; fingerprint
  inclusion gated on mode; tests. Independently valuable (handoff.py:76
  `stable_particle_ids_included` can later flip to True). Shippable alone.
- **SAMP-M2 -- sampling library** (1 day). `stflip/sampling.py` with point
  scrambling AND index shuffling; statistical + parity tests. Pure library.
- **SAMP-M3 -- Params.temporal_sampling + draw-site wiring** (0.5-1 day).
  Default "pseudo" is trajectory-byte-identical; release note for the
  fingerprint invalidation (Decision 1).
- **SAMP-M4 -- (absorbed).** The calm-surface metric is CALM-M1 (Decision
  5). SAMP contributes only the PSD/height-probe requirements to that spec.
- **SAMP-M5 -- A/B experiment + decision gate** (1 day compute/writeup).
  Requires CALM-M1 and SAMP-M1..M3. Writes docs/temporal-sampling.md with
  numbers and renders regardless of outcome.
- **SAMP-M6 (conditional on M5)** -- addon toggle, docs, experiments.py
  ablation profiles (frozen profile provenance at experiments.py:246
  untouched; defaults stay "pseudo") (0.5 day).
- **SAMP-M7 (optional)** -- spatial Owen-Sobol seeding at solver.py:873,
  subsuming the ppc==8-only 2x2x2 stratification (solver.py:876-880);
  own mini A/B on inflow scenes (1-2 days).

---

## Initiative NORM: exact gamma-conditioned temporal weighting (Sec. 3.10 -> L3)

Added in revision 2. The external plan's best concrete catch, verified against
the paper and the code; the design below passed both adversarial lenses
(feasibility and physics) without revision.

### Goal

Paper Sec. 3.10 (p.9) admits that adaptive jitter attenuation "in theory
requires also adapting the temporal kernel ... and re-scaling m0" but skips it
because the error is bounded: the mean temporal weight under narrowed jitter is
`mu(g) = (945 + 105 g^2 - 21 g^4 - 5 g^6)/1024 in [945/1024, 1]`, so average
weighting and phi_st are off by at most 7.7 percent and 3.9 percent. The repo
reproduces the approximation faithfully: `_p2g` applies `wt = W_T(theta)` with
no gamma conditioning (solver.py:971-975) and `_calibrate_m0` returns ppc
assuming gamma = 1 (solver.py:943-954). Consequences NORM removes: (a) mixed-
gamma faces bias velocity toward high-gamma (fast) particles -- worst case
~2 percent of the local velocity contrast, every substep, a systematic momentum
leak into calm regions no denoiser removes; (b) a gamma~0 pool's ramp phi
scales by sqrt(945/1024) = 0.961, so the 0.5 isolevel sits sub-voxel low AND
breathes as the activity-driven gamma histogram changes -- polluting exactly
the CALM-M1 drift and low-frequency PSD band that SAMP-M5 gates on; in
two-phase, calm liquid under moving gas reads `phi_f = mu(0)/(mu(0)+1) = 0.480`
instead of 0.500, biasing face density toward gas (material at 800:1);
(c) `m0 = ppc` is exact only at gamma == 1 today (note st_enabled=True with
jitter_strength=0 scales ALL accumulators by 945/1024 right now).

### Design

Per-particle normalized weight `w = W_T(theta) / mu(gamma_p)`, with gamma_p
RECOMPUTED at P2G time rather than persisted. Exact for the shipped speed
gate because self.vel is provably unmodified between the draw (end of step n,
solver.py:2094-2097) and the next P2G: `_apply_sheeting` is position-only
(solver.py:1517-1523, docstring guarantee), `_enforce_solid_velocity` runs
BEFORE the draw (solver.py:2088-2089), outflow compaction subsets all arrays
consistently (solver.py:1319), inflow appends carry dt_resid = 0
(solver.py:887-888), and the draw-time dt is exactly the dt_prev `_p2g`
already receives (committed solver.py:2119, checkpointed as "dt_prev"). Zero
new per-particle state, zero checkpoint changes, Decision 3 intact.

Pieces: (1) kernels.py: `w_temporal_mean(xp, gamma)` in Horner form
`(((-5*g2 - 21)*g2 + 105)*g2 + 945)/1024`, `g2 = gamma*gamma`; identities
mu(1) = 1.0 BITWISE in float32 (945+105-21-5 = 1024 exact -- gamma == 1 runs
are automatically unchanged), mu(0) = W_T(0) = 945/1024. (2) solver.py:
extract `_jitter_gamma(self, dt)` from the draw site (bitwise guard on the
draw path; this is jitter-block edit #0, see Recommended order); condition
inside the existing `if p.st_enabled:` branch only:
`wt = wt / kernels.w_temporal_mean(xp, self._jitter_gamma(dt_prev))` behind
new `Params.exact_temporal_norm: bool = True` (False = paper-faithful legacy).
Because wt multiplies every tap weight for all four accumulator families
(solver.py:1011-1044), the single edit consistently conditions volume,
liquid_volume, momentum, and face-mass -- hence face velocities, `*_phi`,
`c_phi`, `g_valid` thresholds, and m0-relative undersampling flags.
(3) validation.py:295-297: same divisor on the standalone deposition_weights
diagnostic so `effective_weighted_sample_fraction` stays honest (validation-
side only; Decision 2 intact).

Default ON with a legacy flag, decided at introduction: this is exact math
(precedent: the repo's own MC-to-analytic `_calibrate_m0` fix), gamma == 1 is
bitwise unchanged, and Decision 1 means a later default flip would invalidate
defaults-users' bakes a second time.

Edge cases (all argued in review): gamma = 0 divides by 945/1024 (w =
W_T(0)/mu(0) = 1 exactly -- the gamma -> 0 instantaneous-P2G collapse becomes
exact instead of uniformly scaled); out-of-slab theta stays zero (no-clip
policy at solver.py:968-970 preserved); st_enabled=False keeps wt = ones;
ENER-M2b's force_instantaneous second P2G reuses that branch and stays
pristine; clamped Eq. 10-11 updates (< 1 percent per the paper) make the
divisor approximate at the same neglect order as the paper, bounded in
[1, 1024/945]; fresh-seeded particles get a velocity-recomputed divisor for
exactly one substep (at most 7.7 percent underweight, strictly no worse than
status quo, self-heals after their first draw).

**Composition constraint with CALM-M3.** Recompute-at-P2G reproduces the draw
gamma only for the speed gate. CALM-M3's `gamma_mode="surface"` adds an
interiorness term read from the CURRENT step's c_phi, which next step's P2G
cannot reproduce. When (and only when) surface mode is combined with
exact_temporal_norm, the draw site stores gamma_p to `self._gamma_prev` and
checkpoints it via the optional mode-gated `gamma_prev` key RESERVED in
SAMP-M1's v3 schema (Decision 3). Speed mode never writes it.

### Cost, risks, validation

Cost: one `_norm_rows` + one polynomial + one divide per particle per substep,
O(n) elementwise, well under 1 percent; GPU identical (xp ops, no syncs).
Risks: the vel-immutability recompute contract is an INVARIANT future step-
structure changes could silently break -- a debug assertion (draw-site gamma
hash vs recompute, active under `experiments` runs) plus a code comment at the
draw site stating the contract are mandatory; ENER-M2b's restructured tail
keeps the contract (draw happens at its step 11, only positions change after)
and its PR must re-verify. Validation: mu identities + quadrature vs the
actual `w_temporal` (tests/test_kernels.py); bitwise parity at gamma == 1 and
with exact_temporal_norm=False; m0 plateau test -- uniform pool accumulator
mean == ppc at gamma in {0, 0.3, 0.7} (previously only 1.0); mixed-gamma
two-population face-bias test (bias -> 0 with the fix); recompute==draw
stress test across outflow/inflow/sheeting steps; two-phase arm (from the
reconciliation's missed-by-both list): the phi_f = 0.480 -> 0.500 correction
on a calm liquid/moving gas interface at 800:1; params-reject row; gate on
the gamma-active SAMP-M5 arms = phase-volume drift and interface offset
improve or hold, nothing regresses.

### Milestone

- **NORM-M1** (1-3 days, shippable alone): everything above in one PR
  (kernels helper + draw-site extraction + _p2g conditioning + validation
  diagnostic + tests + docs + Decision-1 release note). It also provides the
  "normalization" axis of the SAMP-M5 matrix (exact vs legacy), replacing the
  external plan's proposal to test normalization inside a 720-run factorial.

---

## Initiative ENER: energy and angular momentum (Limitation L2)

### Goal

Counteract the first-order splitting's dt-dependent loss of kinetic energy
and angular momentum, which ST-FLIP's large steps amplify. Deliverables:
(1) an artist-facing vorticity-confinement "liveliness" control (energy
INJECTION, honestly labeled a look control, default off); (2) an opt-in
advection-reflection mode where energy/angular-momentum retention at
CFL 2N matches or beats plain ST-FLIP at CFL N at roughly equal cost;
(3) energy/angular-momentum metrics so all of it is measured. Explicit
non-goal: exact conservation (impossible in this splitting family).

Ordering: metrics first (ENER-M0), confinement (cheap, immediate,
ENER-M1), reflection (the principled fix, ENER-M2a/b), IVOCK as a
time-boxed spike we expect to reject (ENER-M3) -- IVOCK targets
semi-Lagrangian GRID advection loss, whereas this solver's particle
advection loses little vorticity in transport; APIC (already in the repo,
`Params.transfer="apic"`) addresses transfer smoothing more cheaply.

### ENER-M0 -- measure first (1-2 days, shippable alone)

Add `angular_momentum_{x,y,z}_estimate` to `metrics.measure_frame`
(metrics.py:205): sum over particles of mass * cross(x_p - center, v_p).
Review-mandated corrections:

- **This is the roadmap's single frame-record schema bump** (Decision 2):
  SCHEMA_VERSION 2 -> 3, extend FRAME_FIELD_ORDER, update the schema
  equality tests (tests/test_metrics.py, tests/test_validation.py:105,
  tests/test_cache.py:580), and document in the PR that `read_metrics`
  silently drops schema-2 rows, so resumed pre-existing bakes lose
  historical metric rows (accepted, versioned decision).
- **Positions must be resynced**: compute L and KE from
  `_resynced_positions_and_keep` positions (solver.py:2213), not raw
  jittered positions -- otherwise temporal jitter noise (growing with CFL)
  contaminates exactly the decay curves that gate M2b. Velocities remain
  the FLIP-blended particle velocities; document this.
- **Per-phase mass**: use phase-weighted particle mass (reuse the
  two-phase mass logic from `_p2g`), or scope the metric to liquid
  particles and name it accordingly. The uniform `particle_mass` at
  metrics.py:310 is wrong for two-phase bakes.
- New validation scenario: zero-gravity closed-tank cylinder seeded with
  `velocity.SolidBodyRotation` (velocity.py:79-109), ~48 frames, reporting
  L_z(t)/L_z(0) and KE(t)/KE(0) at CFL {1, 4, 8, 16}. **Report the CFL-1
  decay as the aperture-torque floor** (the Cartesian cut-cell cylinder
  exerts stair-step torque) and present all retention numbers relative to
  it -- absolute retention is not pure splitting dissipation.
- Wall-clock: the repo DOES have per-frame wall time
  (`metrics.compute_wall_s`; validation.py:334 records it) -- cost gates
  reuse it; no new profiler.
- Report-only companions (reconciliation adoptions): (i) a Taylor-Green-style
  closed-box energy-decay scene as a shear-dissipation complement to the
  rotating tank (explicitly NOT an order-of-convergence claim -- O(dt)
  temporal jitter voids pointwise temporal order); (ii) a downscaled
  FULL-GAUGE Kleefsman scenario -- NOTE (2026-07-14): substantially covered
  since by stflip/benchmarks.py + stflip/paper_validation.py (v0.24.0,
  PR #39: KleefsmanBenchmark with a multi-gauge layout and gauge-series
  comparison against loadable reference data). Before doing any Kleefsman
  work here, check those modules first; what remains for ENER is only
  wiring the existing benchmark into the ENER-M2b/CAP decision writeups.

### ENER-M1 -- vorticity confinement (2-3 days, shippable alone)

Feasible exactly as designed (both reviewers). New
`forces.confinement_accel(xp, shape, dx, grids, strength, clamp)` in
stflip/forces.py beside directional/vortex/turbulence accel: face->cell
average velocities (pattern metrics.py:178-180), omega = curl via
xp.gradient (pattern metrics.py:182-187), optional one binomial smooth per
component (reuse `surface_tension.smooth_phase`) to tame MC-noise-driven
|omega| gradients, N = grad|omega| normalized with 1e-12 floor,
a = strength * dx * cross(N, omega), masked by `grids["c_phi"] >= 0.5`,
per-cell |a| clamp (default 10*9.81) for unconditional safety. Wiring: add
"CONFINEMENT" to the `add_force` whitelist (solver.py:799-801) and a
dispatch branch in `_apply_forces` (solver.py:820-832; it already receives
grids); the existing cell->face averaging and `u += dt*a` (solver.py:
838-847) apply it like gravity, and the downstream projection removes any
divergence. UI: force_type enum entry (addon/properties.py:66-77) reusing
force_strength. Tests: zero-strength is a bitwise no-op; still pool stays
calm at slider max (clamp works); rotating-tank enstrophy strictly
increases at strength > 0. Docs must say plainly: this injects energy; it
counteracts the LOOK of dissipation, it is not a conservation fix.

Hygiene rule (Decision 6c, resolving the external plan's objection to
shipping confinement first): confinement is OFF in every validation and
decision run of every initiative. The external plan's worry -- that a look
control can visually hide dissipation -- is legitimate exactly once, if
confinement leaked into evaluations; with the rule it cannot, and the
product argument (artists expect the knob; Mantaflow and Houdini ship one;
ENER-M0's metrics land first so the injection is measured) stands.

### ENER-M2a -- projection refactor (1-2 days, shippable alone)

Extract solver.py:1969-2050 (k coefficients, weighted divergence +
moving-wall RHS, ppe_solve, in-place gradient subtraction,
pressure-outflow half-cell corrections, no-through re-enforcement) into a
`_project(grids, dt, ...)` method, and the extrapolation loop
(solver.py:2060-2073) into a helper taking a `layers` argument. Key
identity making reuse trivial (verified in code): the projected VELOCITY is
invariant to the dt used inside `_project` (p scales as 1/dt, k_f =
dt*alpha/rho, correction is dt*inv_rho*grad p -- dt cancels). Gate with a
test asserting bit-identical CPU frame state with reflection off, before
and after. Land fast and atomically (hot region, codex exposure).

### ENER-M2b -- advection-reflection (5-8 days, shippable alone)

**The original step order was rejected by review as a wrong algorithm** --
it never transferred forces or the first projection into particle momentum
(gravity would be dropped; a hydrostatic pool would levitate), and it used
the non-solenoidal reflected field as the transport velocity. The
following CORRECTED scheme is the design of record
("resync-to-midpoint with mid-step reflection G2P"):

1. P2G with W_T as today (`_p2g(dt_prev)`, solver.py:1892); snapshot
   `old` = pre-force grids (as today, solver.py:1900).
2. Apertures, forces (full dt), surface tension, viscosity, no-through as
   today (solver.py:1893-1964). Snapshot `u_star` = {u,v,w} copies after
   that block.
3. First `_project(grids, dt)` -> u1 (projected, solenoidal).
4. Reflected field: `u_hat = 2*u1 - u_star` ONLY on faces that are open
   (alpha > 0), valid in the first P2G, AND have `phi_f` above a threshold
   (first-class mask, not a debug param -- eps_rho-clamped near-interface
   faces must not get their projection residual doubled); elsewhere
   `u_hat = u1`. Re-enforce no-through on u_hat.
5. Extrapolate u1 (transport field), u_hat, and old with the same valid
   masks (same-mask discipline of solver.py:2070-2073), half band
   `layers = ceil(cfl_target) + 2`.
6. **Mid-step reflection G2P**: `u_p <- u_p + interp(u_hat - old)`
   (through the existing G2P blend machinery -- FLIP delta form shown;
   APIC/PIC take their existing branches). This equals the plain FLIP
   delta `interp(u1 - old)` plus the reflected pressure impulse
   `interp(u1 - u_star)`: forces enter particle momentum exactly once, and
   the deposited mid-step field approximates the reflected field as
   Zehnder requires. u_hat is NEVER used for transport.
7. First half-advection through u1: `dt_act1 = clip(dt/2 + dt_resid, 0,
   dt)` -- clip upper bound is dt (changed from 1.5*dt) so each
   half-advection travels at most cfl_target cells, the half band above is
   honest, and total per-step travel <= 2*dt preserves the `_band()`
   sparse-halo contract (solver.py:1800-1802) unchanged. Carry
   `r1 = dt/2 + dt_resid - dt_act1`. Outflow tracking as at
   solver.py:2106-2113; the transient r1 array is filtered with the same
   keep mask (extend `_apply_outflow_filter` with an optional extras list
   or filter locally).
8. Second P2G, instantaneous (wt = 1 via a `force_instantaneous` flag on
   `_p2g` reusing the st_enabled=False branch, solver.py:972-975);
   snapshot `old2`. Justification (corrected): wt = 1 deposition is
   m0-consistent because m0 is calibrated to E[m_hat] with E[W_T] = 1
   under uniform tau, and has strictly less temporal variance; no
   recalibration needed.
9. Rebuild face densities/active from the second P2G's accumulators
   (re-run solver.py:1910-1927); reuse step-start apertures and solid face
   velocities (walls at t^n, O(dt) consistent). Do NOT re-apply forces --
   correct now, because forces already reached particles in step 6 and are
   inside the mid-step deposit.
10. Second `_project(grids2, dt)`; extrapolate (half band); final G2P
    against old2 with same-mask discipline; `_enforce_solid_velocity`.
11. Second-half jitter with fresh velocities: gamma per
    solver.py:2094-2097; one xi draw;
    `dt_act2 = clip(dt/2 + r1 + gamma*xi*dt, 0, dt)`;
    `dt_resid = (dt/2 + r1) - dt_act2`. Unclamped case gives
    `dt_resid = -gamma*xi*dt`, the paper's stationary distribution.
12. Advect through grids2 by dt_act2 with outflow tracking; sheeting;
    commit grids2, `_dt_prev = dt`, `time += dt` (solver.py:2118-2122
    pattern). Render resync (solver.py:2213-2238) unchanged.

Sanity trace (required as a unit test): hydrostatic pool -- particles pick
up +g*dt at midstep via the reflected impulse, transport through u1 = 0
moves nothing, projection 2 of the uniform reflected impulse against the
bottom no-through returns 0, the final delta cancels it: pool exactly
still; a dam break falls under gravity.

**Merge-blocking proof obligation**: the residual-carryover induction for
the composed clips (clip[0,dt] then clip[0,dt] with the r1 carry) must be
written out (3-case analysis mirroring Appendix A) and accompanied by a
randomized adaptive-dt stress test (pattern tests/test_solver.py:440-448),
plus a loud debug assertion that `max(dt_act) * vmax / dx` never exceeds
the extrapolation band.

**Cost (recomputed).** Per reflected substep: 2P + 2G + 2g + 2E + F
(two projections, two P2Gs, two G2P-quality passes, two half-band
extrapolations, forces once). At CFL 2N vs plain at CFL N the per-frame
grid-cost ratio is `1 - F/(2*(P+G+g+E+F)) < 1`: cost-neutral for inviscid
scenes (F small), meaningfully cheaper when implicit viscosity makes F
large. At the SAME CFL, reflection costs ~1.9-2.0x per substep grid-side.
Product framing: "enable Reflection when you raise Target CFL", pair with
`pressure_solver="multigrid"` so the doubled projection stays cheap.
Particle advection work per frame is roughly CFL-independent
(solver.py:1744-1749). RNG note (corrected): reproducibility holds because
rng_state is checkpointed and the draw sequence is deterministic; the draw
COUNT is not invariant across modes in outflow scenes.

**Acceptance gates**: (a) M2a refactor bit-identical; (b) residual bound
under stress; (c) rotating tank: reflection at CFL 16 retains >= the KE
and L_z of plain at CFL 8 (and 8-vs-4), relative to the CFL-1 aperture
floor; (d) **gravity-on dam-break energy/profile co-gate** (the zero-g
tank must not be the only physics gate) plus still-pool and two-phase
smoke with reflection on; (e) wall-clock sanity via the existing
compute_wall_s: reflection@2N within 1.1x of plain@N on the validation
scene with multigrid. Honest framing kept: reflection reduces SPLITTING
dissipation only; where temporal MC noise dominates (calm scenes at 2N,
where the slab widens) it can lose -- the docs say exactly this.

**SOAR arm (conditional, from the reconciliation).** The external plan
proposed starting from second-order advection-reflection (Narain-style
SOAR). Judged: formal second order requires instantaneous midpoint fields
and deterministic sample times; ST-FLIP's P2G deposits W_T-weighted SLAB
integrals at O(dt)-jittered particle times, so the order claim is void
here, and the corrected scheme above is already the FLIP-compatible
structural analogue. The residual delta (transport-field choice, reflection
point) runs as ONE A/B arm inside M2b's harness IF the residual-carryover
induction extends to it; no "second-order" claims are promoted under
jitter.

### ENER-M3 -- IVOCK spike (3-5 days, hard time-box, decision-gated)

Prototype on a branch: advect the previous step's vorticity
semi-Lagrangian, solve lap(psi_i) = -delta_i with `multigrid.solve`
(constant coefficients, liquid mask, tol 1e-2), add curl(psi). Ship only
if it closes an enstrophy gap M2b leaves (> 5 percent at CFL 16 at
<= 1.2x step cost). Review-mandated caveats for the decision memo: naive
Dirichlet psi = 0 at a free surface injects spurious surface vorticity --
use free-slip-consistent psi boundary handling or restrict the correction
to a phi-interior band; and semi-Lagrangian omega advection across 16
cells is itself very diffusive, biasing the spike against IVOCK. A
rejection claim must rest on the interior-band variant, not the naive one.
Expected and acceptable outcome: a documented rejection closing L2's
remaining scope honestly.

---

## Initiative TIME: two-time-level slab representation (Outlook O2 -> L3, hypothesis on L1)

### Goal (reframed per review)

Retain zeroth and first temporal MOMENTS per face during P2G --
algebraically equivalent to two time-level grids at tau = -1/2 and +1/2 --
and reconstruct the field linearly in tau. Today's one-sided kernel W_T
(stflip/kernels.py:33-35) is a weighted mean with effective evaluation
time tau ~ +0.227 (verified analytics: mean tau = 1/2 - 35/128 = 0.2266;
effective sample size 0.613; the repo's own diagnostics report both,
validation.py:575 and :590).

**Honest framing (mandatory).** The 0.273*dt_prev*dq/dt figure is an
offset relative to a HYPOTHESIZED target q(tau = +1/2); the paper's
estimator deliberately targets the W_T-weighted slab integral, and the
slab end is not the step end t^(n+1) either. This initiative therefore
"tests whether residual phase lag contributes to L1", it does not "attack
L1"; only TIME-M3 (scene-level) arbitrates the dynamics question, and the
paper's own negative result on a "thin 4D grid" linear-in-time
reconstruction bears directly on M3. Prior odds are middling; the plan is
gated on a cheap estimator study (TIME-M1) with kill criteria before any
solver code.

### Design (revised)

**Primary arm = W_T-weighted WLS fit** (review-mandated swap). The
temporal fit weight is the existing W_T, so:

- accumulation adds exactly 3 scatters per velocity axis per tap
  (S1w = sum w*W_T*tau, S2w = sum w*W_T*tau^2, S1q = sum w*W_T*tau*q) on
  top of the existing S0w/S0q (tap loop solver.py:1011-1044);
- reconstruction per face: d = max(S0w, eps_m), qbar = S0q/d, tbar = S1w/d,
  var = S2w/d - tbar^2, cov = S1q/d - qbar*tbar, b = cov/(var + lam),
  `grids[g] = qbar + b*(0.5 - tbar)` on valid faces. lam =
  temporal_fit_reg * (1/12) (dimensionless reference scale; M1 sweeps it).
  Closed-form scalar shrinkage -- elementwise only, no linear algebra
  library (mirrors `_inv3x3`, solver.py:53-83);
- the lam -> inf limit is b -> 0 and value -> qbar = today's one-sided W_T
  mean, i.e. the estimator degrades bitwise to the CURRENT estimator, not
  to the symmetric in-slab mean the paper rejected (Fig. 5 oscillatory
  artifacts). This eliminates the dynamical-regression path the review
  found in the uniform-weight arm.

**Validity semantics pinned (all arms)**: `g_valid` stays derived from the
W_T-weighted volume accumulator exactly as today (solver.py:1055/1078 and
the extrapolation contract at 2052-2059). Any validity change is its own
gated experiment with a spray/FLIP-delta regression test. Phase fields
(`*_phi`, `c_phi`) keep the zeroth-moment Eq. 13 / Eq. 7 recipes
(solver.py:1085-1098), so PPE coefficient character is unchanged by
construction in the primary arm.

**Uniform-weight experimental arm** (demoted): uniform in-slab indicator
as fit weight maximizes effective sample size but requires its OWN
S0w/S0q -- +5 scatters per axis-tap (~3x-3.5x scatter work, +40-70 percent
step time under multigrid) since phi/validity stay on W_T moments. If it
survives M1, the reconstruction must shrink toward the ONE-SIDED mean
(evaluate as qbar_WT + b_uniform*(1/2 - tbar_WT)) so no lam reproduces the
rejected symmetric kernel, and it gets a dynamical check (calm pool plus a
mid-energy sloshing case watched for oscillatory surface artifacts, not
just MSE). M1's GO verdict must state which arm was selected AND its
scatter budget.

**Unchanged by construction**: projection consumes the single
reconstructed field (option A -- one instantaneous pressure multiplier,
paper Eq. 17); G2P unchanged (the FLIP baseline `old` automatically
becomes the reconstructed field); jitter Eqs. 10-11, RNG, checkpoints
(tau_p = -dt_resid/dt_prev is derived from already-checkpointed state,
solver.py:971), sparse cropping (transient arrays allocate at the window
shape, solver.py:1867-1883).

**Known temporal inconsistency, kept deliberately**: the projection mixes
u* reconstructed at the slab end with rho(phi_f) and the active mask at
effective time ~0.227. Both derive from the same particle positions, so
spatial support is identical; PPE conditioning is unchanged; active-mask
flicker at CFL 16 goes on the M3 watch list. One paragraph in the results
doc records this.

### Milestones

- **TIME-M1 -- estimator study, the kill point** (3-5 days, shippable
  alone). Land `stflip/temporal_fit.py` (pure xp-agnostic math: moment
  accumulation, fit, evaluate; nothing imports it from the solver),
  `tools/run_temporal_study.py`, tests. The study simulates the actual
  Eq. 10-11 jitter process including clamping and adaptive-dt sequences.
  **Repaired gate matrix**: (a) (s, gamma) cells must be physically
  consistent -- gamma is derived from the same velocity scale that
  generates the temporal signal s (high-s cells run gamma ~ 1); gate
  s >= 2 only at gamma in {0.5, 1} (the old full cross product auto-failed
  at gamma = 0 where no estimator can win). (b) GO requires a single lam
  with MSE(fit) <= 0.95 * MSE(one-sided mean) at s = 2, <= 0.80 at
  s >= 4, AND a universal no-regression bound
  MSE(fit) <= 1.05 * MSE(one-sided mean) at ALL simulated s (0, 0.5, 1,
  2, 4, 8). (c) Gate at face-realistic effective N ~ 30-60
  (spatial-kernel-weighted), with the N = 8..64 sweep reported. Optional
  arm: sweep tau_eval in {0.227, 0.35, 0.5} to expose the bias-variance
  curve instead of presuming +1/2. Unit tests: exact linear-signal
  recovery; the 0.2266 / 0.613 analytics cross-checked against
  `validation.temporal_quadrature_coverage` (validation.py:542-605);
  b -> 0 as tau variance -> 0. Ships value even on NO-GO (documented
  negative result).
- **TIME-M2 -- solver integration behind a default-off flag** (8-12 days,
  only on M1 GO). `Params.temporal_levels` (int, default 1; valid values
  {1, 2}) and `Params.temporal_fit_reg` (finite nonnegative float), both
  with `__post_init__` validation and rows in
  `test_params_reject_invalid_values` (tests/test_solver.py:70-72
  pattern); note the validation artifact's parameters dict gains the keys
  via asdict (benign, expected diff). `_p2g` moment accumulation and
  reconstruction, gated so temporal_levels=1 is bit-identical (dedicated
  parity test hashing pos/vel/dt_resid after N steps). Acceptance:
  bit-parity; linear-in-tau synthetic exactness through the real `_p2g`;
  still-pool with flag on; two-phase 800:1 smoke with unchanged
  `stats.pcg_iters` flag-on vs flag-off (this replaces the earlier,
  vacuous citation of tests/test_multigrid.py:79-134, which builds a
  synthetic PPE and never sees solver phi); a unit test that flag-on/off
  phi and validity grids are identical in the primary arm; checkpoint
  round-trip unchanged (no new keys); CPU/CuPy agreement verified
  manually on the 5090 (no GPU CI marker exists -- corrected claim).
  Cost with the primary arm: P2G scatter work ~2x-2.5x; step time
  +25-45 percent with multigrid projection, +10-20 percent with Jacobi.
  Transient memory: 9 face-grid float32 arrays at the (possibly cropped)
  window shape (~+95 MB dense 128^3, ~+745 MB dense 256^3), freed when
  `_p2g` returns.
- **TIME-M3 -- scene-level evaluation and decision** (4-6 days + ~0.5 day
  for the now-unconditional harness threading). **Unconditional minimal
  validation.py changes** (review fix): add `temporal_levels: int = 1` to
  ValidationConfig with validation, thread through `_params`
  (validation.py:242). MATCHED_CASES (validation.py:46-57) is frozen at
  CFL 1 and 16, so the gate runs at **CFL 16 only** (no CFL-8 claim).
  A/B protocol spelled out: produce TWO artifacts (flag on/off, same
  config and seeds 0/1/2 via run_multi_seed_validation, validation.py:884)
  and compare the `degradation.st_high_vs_st_low` primary errors ACROSS
  artifacts. Additional gates (review additions): total KE and momentum
  trajectories over the full dam-break run with "no systematic divergence
  vs temporal_levels=1 beyond seed spread" (the slope term makes grid
  momentum differ from deposited momentum by O(dt*slope) per face, and
  the 0.98 FLIP blend at solver.py:2082-2084 recycles it -- a long-horizon
  energy gate is mandatory); a >= 200-step smooth-swirl case (rotating
  column) where advective slope is large and feedback can compound;
  calm-pool noise proxy via the CALM-M1 metric (target >= 20 percent
  reduction, no other gate regressing); wall-clock overhead <= 45 percent
  per step (multigrid mode) measured in the tool; active-mask flicker
  watch. Decision recorded in docs/design/temporal-levels.md: PROMOTE
  (adds the frozen-dataclass-defaulted `temporal_levels` field to
  ExperimentProfile, experiments.py:24-84, an ablation profile family,
  one addon IntProperty) / KEEP-EXPERIMENTAL (Params-only) / REVERT
  (remove the `_p2g` branch, keep temporal_fit.py + the doc).

Wording note: success "resolves the K=2 linear case of O2 for this repo;
K > 2 remains open by declared non-goal" (not "closes O2").

---

## Initiative CALM: calm-surface noise suppression beyond Sec. 3.10 (Limitation L3)

### Goal

Four tiers, ordered by measured value per effort: (d) a quantitative
calm-surface metric (the shared instrument, Decision 5); (c) a
render-time-only calm-region denoise in the surface reconstruction path;
(a) surface-scoped jitter gating that restores full interior jitter;
(b) deformation-aware activity so a calm surface above a MOVING bulk
(river, stirred pool -- where the Sec. 3.10 speed gate keeps jitter fully
on) can damp. Honest framing kept: for a truly still pool the existing
gate already suppresses jitter; (a) is hygiene/groundwork rather than a
quality win (see M3 note), and (b) is the only genuinely speculative
piece, shipped last, default-off on a miss.

### CALM-M1 -- calm-pool metric + scenes (1-1.5 days, ship first)

The shared instrument for SAMP-M5, TIME-M3, and CALM-M2..M4.

- **Height map (corrected formula)**: build from per-cell particle COUNTS,
  `h(x, y) = dx * column_count(x, y) / ppc` (np.add.at histogram), NOT the
  boolean `_occupancy` at validation.py:382 (a boolean column sum is off
  by ~ppc and quantized to whole cells -- invisible to exactly the
  sub-voxel noise L3 is about). Add an absolute-height unit test: a
  half-filled box of known depth reports h within half a cell of the
  analytic fill height, so a scale error cannot slip past RMS-only tests.
  The flat-surface < 1e-6 test assumes lattice-regular seeding; say so.
- **Render-path variant is the primary gating metric**: per-column linear
  interpolation of the highest 0.5 crossing of the `reconstruct_surface`
  density (surface.py:381) -- sub-voxel sensitivity, and it measures what
  users see.
- **Resync before binning** (fairness fix): apply the render-path
  re-synchronization (advect a validation-side COPY of positions by
  dt_resid, pattern of `_resynced_positions_and_keep`, solver.py:2213)
  before computing particle-path heights, so cross-mode comparisons
  measure genuine surface irregularity, not removable jitter distortion.
- Statistics: per-frame spatial RMS of (h - mean h); per-column temporal
  std over frames; drift; stored height probes for temporal PSD analysis
  (SAMP-M5 needs the PSD).
- Scenes (new scenario family in stflip/validation.py next to
  MATCHED_CASES, validation.py:46; report-only at first like the
  deposited-mass entries at validation.py:834-850): **S1** still pool
  (test_still_pool_stays_calm geometry, tests/test_solver.py:453);
  **S2** stirred pool -- subsurface VORTEX force via `add_force`
  (solver.py:786, VORTEX at :825-828) so the bulk moves while the surface
  should stay flat. **S2 constraint (mandatory)**: a sustained vortex
  physically dips the surface (centrifugal), so `height_rms_spatial` on
  S2 is physics-contaminated and may NEVER gate; only
  `height_std_temporal` gates on S2, and spatial statistics subtract the
  per-column temporal mean first. **S3** (added for M4 regression): a
  jet or ballistically translating droplet at CFL 16 -- the scene class
  where mis-gating would strobe. **S4** (reconciliation adoption): a
  uniformly translating slab at CFL 8/16 -- gamma fully active by
  construction, gates Galilean surface flatness (a moving flat surface
  must stay flat); also the second scene axis of SAMP-M5 and the natural
  home of NORM's mixed-gamma bias check.
- Instrumentation for Decision 6a (reconciliation adoptions): record
  per-frame `reconstruct_surface` wall time and render-resync advection
  time (solver.py:2213 path) next to compute_wall_s as standalone helpers
  (NOT frame-record fields -- Decision 2), plus a peak-memory probe
  (process RSS high-water per frame) in the validation harness; several
  roadmap features stack transient memory (ids +20 percent, TIME ~+745 MB
  dense 256^3, reflection snapshots) and nothing measures the stack today.
- Whitewater probe (missed by both plans): the repo's whitewater
  secondaries seed from velocity/curvature state, so high-CFL MC noise on
  calm surfaces plausibly drives spurious foam -- likely the FIRST L3
  symptom actual users see. Record whitewater seed counts per frame on
  S1/S2/S4 (whitewater off by default in the scenes; one extra arm with it
  on) so SAMP/CALM/NORM changes can be checked against foam-rate shifts.
- CI config defined explicitly: 24^3, ppc=4, CFL {1, 16}, single seed,
  under 60 s CPU; the full 32^3 x CFL {1, 8, 16} x seeds {0,1,2} matrix
  is the docs baseline table, not CI.
- These are standalone metric functions (stflip/metrics.py helpers +
  validation.py wiring) -- NO frame-record schema change (Decision 2).

### CALM-M2 -- render-path calm denoise (1.5-2 days, shippable alone)

New `calm_region_smooth(density, feature_mask, extra_iterations, xp)` in
stflip/surface.py beside `mean_curvature_flow` (surface.py:242): extra
clipped MCF iterations weighted by a calm mask, then clamp the result
between erode/dilate bounds (3^3 min/max filters) of the pre-pass density
so the isosurface stays within one voxel and volume loss is bounded.
`dtau = 1/(6*max(mask))` matches the existing `_pseudo_time_step` logic.
Review fixes:

- **Calm mask**: for M2, use the existing self-quotient feature mask
  (surface.py:199-220) plus the band clamp only. The proposed
  "low normal variance" AND-term likely excludes the target regions (noisy
  calm surfaces ARE high-normal-variance); if added later it must be
  computed on Gaussian-blurred density (sigma >= 2 dx, above the noise
  wavelength), and only if measured over-smoothing of real waves appears.
- **Thin features**: the band clamp bounds isosurface DRIFT, not
  thin-feature volume -- a <= 1-voxel sheet can erode to an empty lower
  bound. Add a thin-sheet retention test (1-2 voxel slab survives, relying
  on the feature mask). Volume contract refined during implementation (the
  original flat ">= 95 percent sphere volume" is unachievable for RESOLVED
  spheres: they read as calm at self-quotient ~ 1 and the clamp permits one
  voxel of erosion, which is 42 percent of a radius-6-voxel sphere): the
  shipped two-tier contract is (a) droplets smaller than the blur kernel
  are feature-mask-protected at >= 95 percent, and (b) resolved calm
  spheres never erode past the 3x3x3 erosion envelope (one voxel). Both
  are tested.
- Config/UI: `paper_calm_smoothing_iterations` IntProperty (default 0,
  range 0..100 -- 0 allowed, unlike mcf_iterations' 1..100, so it needs
  its own validation in `paper_surface_config`, operators.py:560/578);
  thread through `paper_surface_fingerprint` (operators.py:616-618);
  bump SURFACE_CONFIG_VERSION 2 -> 3 (cache.py:30) and document that ALL
  existing paper-surface caches re-reconstruct once after upgrade
  (derived data, soft-warning path at operators.py:653-684 handles it).
- `_default_padding` (surface.py:277) must use
  iterations + extra_iterations. Default 0 = byte-identical output
  (hash-compared test). Acceptance: on S1 at CFL 8, render-path
  height_rms_spatial reduced >= 2x with 20 calm iterations vs 0.

### CALM-M3 -- surface-scoped gamma (interiorness only) (1 day)

New `Params.gamma_mode` ("speed" default = current behavior bit-exact |
"surface"), validated near solver.py:208; `adaptive_gamma` bool retained
for back-compat. In surface mode, an interiorness term
`smoothstep(0.85, 0.98, c_phi at particle)` is ADDED to the gate (clipped
to 1) so deep-liquid (and symmetric deep-gas in two_phase) particles keep
full jitter. All inside the jitter block solver.py:2093-2100; the xi draw
(2098-2099) and Eq. 10-11 clamp/residual (2103-2104) are untouched, so
RNG draw count, checkpoint rng_state semantics, and CPU/GPU stream parity
hold. Appendix A holds for any per-particle gamma in [0, 1].

Review fixes folded in:

- **Solid masking (mandatory, applies to M3 and M4)**: c_phi cannot
  distinguish free surface from walls (the phase field is depressed along
  all solids). Fold solid fraction into the detection -- treat
  solid-occluded cells as interior (phi = 1) before `smooth_phase`, or
  force interior_p = 1 where solid fraction exceeds a threshold -- so
  wall-adjacent fast flow is never de-jittered by the surface gate.
- **Honesty note (in code comment and docs)**: where local CFL ~ 0 the
  field is temporally constant over the slab, so restored interior jitter
  has no signal to antialias -- this tier is hygiene and uniform
  stratification groundwork (it composes with SAMP), and a null quality
  result on M3 is NOT a failure. The mechanism check (mean |dt_resid|
  over deep-bulk particles returns to ~dt/4, the U(-dt/2, dt/2)
  stationary value) is the acceptance criterion.
- Checkpoint wording (corrected): in-version save/resume round-trip is
  bit-exact (no new arrays, no new draws). Adding gamma_mode to Params
  invalidates resume of pre-upgrade bakes (Decision 1) -- release note.
- NORM composition (revision 2): surface mode's interiorness term reads the
  CURRENT step's c_phi, which NORM's recompute-at-P2G cannot reproduce next
  step. When gamma_mode="surface" runs with exact_temporal_norm, the draw
  site stores gamma_p to `self._gamma_prev` and checkpoints it via the
  optional mode-gated `gamma_prev` key reserved in SAMP-M1's v3 schema
  (Decision 3). Speed mode never writes it. This PR owns wiring that path.

### CALM-M4 -- deformation/normal-velocity activity (2-3 days, last, speculative)

Replace the speed-only activity with (review-corrected):

```
a_p = max( |v_p . n_hat_p| * dt / dx,        # normal-displacement CFL
           |n . D . n|_p * dt,               # normal strain rate
           ||u_grid_now - u_grid_prev||_p * dt / dx )   # grid-frame unsteadiness
gamma_p = jitter_strength * clip(smoothstep(0, 1, a_p) + interior_p, 0, 1)
```

- **D is the SYMMETRIC strain-rate tensor** 0.5*(grad u + grad u^T) -- the
  full velocity-gradient Frobenius norm includes rotation, and near
  solid-body rotation (the S2 vortex) it reads sqrt(2)*omega, which would
  have kept gamma high and made the S2 target unreachable by
  construction. Prefer the surface-relevant scalar |n.D.n|; keep full-D
  as a conservative fallback if n.D.n proves too aggressive in the sweep.
- **Unsteadiness term (added)**: one retained per-cell velocity buffer
  from the previous substep guards the uncovered regression surface --
  the flanks of a fast coherent stream (v tangential, near-zero strain)
  translating many cells per step, exactly the large-CFL aliasing regime
  ST-FLIP exists to fix. The S3 jet scene gates this: no structured flank
  ripple vs main (phase-density RMSE on the jet crop).
- n_hat from the gradient of solid-masked smoothed c_phi
  (`surface_tension.smooth_phase`, iters=2 -- which is 6 axis passes;
  cost accounting says so); cell strain via center<->face averaging idiom
  (solver.py:839-845) and xp.gradient. Roughly 7-8 transient cell grids,
  ~5 `_sample_cells` gathers (8 taps each) per particle; ~5-10 percent
  substep overhead, GPU-friendly, no new RNG.
- **Regression guards (corrected)**: the earlier design cited
  `temporal_quadrature_coverage` (validation.py:542) as a jitter-health
  guard three times -- that function is a static kernel quadrature check,
  run-independent and mode-insensitive (called with no simulation data at
  validation.py:732); as a gate it is vacuous. Replace with the
  run-dependent per-frame `_FrameOutput.temporal_quadrature_state`
  fields (validation.py:295-314, 348-359): gate M4 on
  `effective_weighted_sample_fraction` and `occupied_slab_bins` on the ST
  CFL-16 dam break within 5 percent of main. The phase-Laplacian RMSE
  (validation.py:819-833) is cited as secondary evidence, not a gate
  (only threshold-IoU carries the coherence-gate flag in code).
- Acceptance: S2 height_std_temporal reduced >= 30 percent vs the speed
  gate at equal settings; no S1 regression; S3 jet clean; dam-break
  matched RMSE within 2 percent. On a miss: ships default-off with the
  measured negative result documented.

Effort total for CALM: ~6-8 focused days across four independent PRs.

---

## Initiative CAP: relaxing the capillary time-step restriction (Limitation L4)

### Goal

Any sigma > 0 activates the Brackbill clamp
`dt <= sqrt(rho_sum * dx^3 / (4 * pi * sigma))` (solver.py:2143-2151),
velocity-independent and O(dx^1.5); in surface-tension-dominated scenes it
sets the step and ST-FLIP's large-step advantage evaporates.
**Corrected motivating numbers** (repo defaults rho = 1000,
frame_dt = 1/24): at 16^3 (dx = 1/16), sigma = 100, dt_cap ~ 0.0139 s ->
**3** sub-steps per frame (not ~14 as an earlier draft claimed); at
dx = 1/64 the same sigma gives ~24 sub-steps; production droplet scenes
at fine dx are worse (scaling dx^(-3/2)). With the semi-implicit path the
clamp is relaxed by a bounded user factor (recommended 4-16x), cutting
sub-steps by the same factor in clamp-bound scenes for an expected net
3-10x wall-clock win there; zero change when sigma = 0 (all paths gated,
mirroring the `p.viscosity > 0` precedent at solver.py:1947).

### Design

**Primary (a): semi-implicit surface tension** -- an interface-concentrated
Helmholtz momentum solve (Hysing 2006 style linearized Laplace-Beltrami
stabilizer, lumped scalar-coefficient form), one SPD Jacobi-CG solve per
MAC component, inserted after the explicit CSF kick (solver.py:1931-1942)
and before implicit viscosity (solver.py:1947). The explicit CSF force
stays as the predictor; the implicit operator damps the stiff
capillary-wave feedback. Honest caveat kept: with a lumped operator and a
noisy P2G phase field there is NO unconditional-stability proof, so the
clamp is relaxed by a bounded `st_clamp_scale`, never removed.

Linear system per component d: `(R_d + A_d) u_d = R_d * u_hat_d`, with
R_d = diag(face densities already computed at solver.py:1916-1927,
floored by eps_rho) and A_d a weighted graph Laplacian with edge
coefficient `a_e = dt^2 * sigma * delta_e * gate_e / dx^2`,
delta_e = |grad phi_s| averaged to the edge midpoint (phi_s and its
gradient magnitude refactored out of `surface_tension.cell_force`,
surface_tension.py:74-93, so they are computed once; the existing direct
callers in tests keep the old signature working). SPD because R is a
positive diagonal and A is a symmetric graph Laplacian (density on the
diagonal, never inside the edge coefficient). a_e/R ~ 1/(4*pi) * (dt/dt_B)^2
~ 0.02-0.08 at the Brackbill limit (near-explicit consistency for free)
and O(10) at 16x. Implementation composes `pressure.apply_laplacian` /
`pressure.diagonal` (pressure.py:24-77, verified shape-generic on face
grids with one-larger coefficient arrays) with a Jacobi-CG copied from
the `pressure._solve_core` conventions (pressure.py:310-352: 0-d device
scalars, check_every residual checks, no cupy.linalg). Zeroing the
outermost coefficient layers converts the exterior 2*k Dirichlet terms to
the natural (Neumann) BC a velocity solve needs. Everything lives in a
NEW module `stflip/st_implicit.py` (bpy-free, xp-agnostic).

**Review-mandated corrections to the operator:**

1. **Validity gating (the main physics fix)**: `a_e = 0 unless BOTH
   coupled DOFs have grids[g + "_valid"]` (in addition to the solid /
   aperture gate). At the insertion point, invalid faces hold u = 0 plus
   g*dt -- non-physical values that only get overwritten by extrapolation
   AFTER the projection (solver.py:2052-2073) -- and the smoothed delta
   band extends several cells past particle support, so without this gate
   a shell of invalid air-side faces drags valid free-surface interface
   velocities toward stale values: spurious surface drag masquerading as
   damping, injected into particles via the FLIP delta. With invalid
   faces out of the system, the previously proposed special 0.05*rho
   R-floor is DELETED (no longer needed; keep eps_rho).
   Dedicated test: a uniformly translating droplet (constant velocity +
   sigma, no gravity) at st_clamp_scale = 8 must preserve center-of-mass
   velocity to < 1 percent over 10 frames -- energy tests cannot catch
   drag (drag reduces energy and "passes").
2. **Solid BC spec resolved**: blocked faces are NON-DOFs with
   gate_e = 0; the wall boundary condition is natural/no-flux for the
   capillary stabilizer; solid velocities are enforced by the existing
   downstream no-through write (solver.py:1962-1964). The earlier
   "Dirichlet held at solid velocity, exactly the viscosity
   fixed/fixed_value pattern" sentence is dropped -- it is not
   implementable with the masked pressure stencil (`apply_laplacian`
   zeroes its input via `pm = p * liquid`, pressure.py:31, so fixed
   neighbor values never enter). The corresponding test asserts zero
   capillary flux across blocked edges: the solve is unchanged when
   solid-side values are perturbed.
3. **Consistency test tolerance derived, not guessed**: at
   dt = 0.5x Brackbill the implicit correction is O((dt/dt_B)^2/(4*pi))
   ~ 1-3 percent in the band, so the implicit-vs-explicit comparison
   asserts relative difference <= C * (a/R)_max with C ~ 2-3 (docstring
   shows the derivation), not 1e-3. The exact-identity test (DOFs with
   delta_e = 0 return u_hat exactly) remains the tight check.

Two-phase: density ratios enter only through R; conditioning worsens as
dt^2*sigma/(rho_gas*dx^3) grows (~1e3-1e4 diagonal contrast at 800:1 and
scale 16); Jacobi-CG tolerates it at moderate cost because the system is
identity outside the band -- iterations are recorded in FrameStats
(mirror `stats.pcg_iters`) and the recommended scale is capped where they
blow up.

**Fallback (c): clamp scale + impulse limiter (reworded honestly).**
`st_clamp_scale` without the implicit solve, paired with a per-face
limiter on the explicit kick. **Corrected formula**: the velocity kick
`|dt * F * inv_rho|` is clipped at `dv_max = st_max_dv_cells * dx / dt`
so displacement per step stays under st_max_dv_cells cells (the earlier
draft compared a velocity against a length). **Corrected failure mode**:
above the Brackbill limit the explicit feedback still alternates sign and
grows until the clip binds, saturating at bounded-amplitude interface
chatter (~st_max_dv_cells * dx per step) -- persistent grid-scale surface
noise on a calm pool, not merely "weakened sigma". Documented as
robustness insurance for modest scales (2-4x), NOT a recommended 8-16x
path; its test gains a quiescence metric (RMS interface-band velocity on
a static pool below a threshold tied to the explicit small-dt baseline),
not just "finite and energy-bounded".

**Rejected (kept for the record)**: local sub-stepping of the surface-
tension force in BOTH variants (sharpened in revision 2): frozen-force
sub-stepping delivers identical total impulse with zero stability benefit
-- the instability lives in the force <-> interface-motion feedback; and
RE-EVALUATED-force sub-stepping requires moving the interface at capillary
dt (re-advect band particles, re-deposit and re-smooth phase per subcycle),
which re-buys much of the cost the clamp imposes, conflicts with
per-particle jittered times (advancing a subset breaks the slab-sampling
identity and dt_resid bookkeeping), and lacks a stability argument without
per-subcycle projection. Note the semi-implicit path above already IS the
IMEX treatment (explicit CSF predictor + implicit stabilizer) that the
external plan asked to try "first". Also rejected: Sussman-Ohta 2009
(nonlinear phi solve); folding ST into the PPE coefficients (changes the
meaning of the projection, entangles ST with the multigrid preconditioner);
GFM jump terms on the PPE RHS (accuracy follow-up, not a stability fix).
Optional slow-marked extension only (never a gate): a Rayleigh-Plateau
breakup scene -- expensive and IC-sensitive, report-only if added.

### Cost

Per sub-step when sigma > 0 and st_implicit: one reused smooth_phase, one
gradient-magnitude + edge-averaging pass, 3 Jacobi-CG solves at 10-50
iterations each (identity outside the band), each iteration = one 7-point
stencil + a few reductions. **Stated per scene class (review fix)**:
two-phase, relative to the ~150-iteration Jacobi projection
(docs/performance-and-scaling.md), the three solves are ~0.2-1.0x of one
projection: +20-50 percent per sub-step worst case. Free-surface
projections converge in far fewer iterations (crop boxes, liquid-only
rows), so there the ST solves can reach ~1-2x the projection cost -- still
small against P2G/G2P/advection, and the band cropping in CAP-M4 restores
the bound if needed. Frame-level in clamp-bound scenes: st_clamp_scale=8
cuts full sub-steps ~8x for < 1.5x per-step cost, net ~4-6x wall clock.
Memory: transient only (3 edge-coefficient arrays + ~5 CG work arrays per
component, ~67 MB each at dense 256^3, much less in the sparse window).
GPU: all xp ops, 0-d device scalars, no host syncs, no randomness --
parity is structural. Fallback (c): one multiply + one clip, free.

### Validation

Unit (NEW tests/test_st_implicit.py): operator symmetry/SPD on a 16^3
band problem; consistency with derived tolerance (fix 3 above); identity
far from interface; zero-flux-across-blocked-edges (fix 2). Integration:
32^3 sphere at dt_CFL = 10x dt_Brackbill, scale 10, 10 frames -- finite,
KE non-increasing after frame 1, volume within 5 percent; translating
droplet drag test (fix 1); **bitwise no-op**: sigma=0 + st_implicit=True
reproduces the sigma=0 baseline bit-for-bit (note: the EXISTING
test_zero_sigma_is_a_noop at tests/test_surface_tension.py:56-70 compares
run(0.0) to run(0.0) -- a tautology that guards nothing; the new test is
the real guard, and the old one should be fixed in passing); clamp
arithmetic re-specced on a config where the clamp forces >= 8 sub-steps
(e.g. dx = 1/32 at sigma = 100: ~9 sub-steps) with the quantized
assertion `steps_after == ceil(frame_dt / min(dt_CFL, scale * dt_cap))`;
two-phase 800:1 at scale 8 finite with bounded CG counts; fallback
quiescence metric. Physics (slow-marked): Laplace pressure jump within
25 percent of 2*sigma/R at scale 1 and 8, parasitic currents at scale 8
<= 3x scale 1; droplet n=2 oscillation period within 30 percent of
Rayleigh omega^2 = 8*sigma/(rho*R^3) at scale 4, amplitude decaying
(overdamping documented as the accuracy-for-speed trade; growth = fail).
Scenario: clamp-bound droplet case, explicit scale 1 vs implicit scale 8:
steps/frame reduced >= 4x, phase RMSE within tolerance class, momentum
drift < 2 percent/frame. Headline metric: sub-steps/frame and total
PCG+CG iterations at equal visual acceptance.

### Milestones

- **CAP-M0 -- fallback dial** (0.5-1 day): st_clamp_scale +
  st_max_dv_cells limiter (corrected formula) + clamp scaling + UI +
  tests + honest troubleshooting docs (2-4x framing, chatter caveat).
  Release note: Params fields invalidate resume of pre-upgrade bakes
  (Decision 1; this also resolves the old "confirm resume gating"
  question -- new params are gated automatically by the fingerprint).
- **CAP-M1 -- st_implicit library** (1.5-2 days): stflip/st_implicit.py,
  cell_force refactor, unit tests (1)-(4). Pure additive module.
- **CAP-M2 -- activation** (1-1.5 days): Params.st_implicit, insertion,
  FrameStats CG counters, integration tests incl. the bitwise no-op and
  the translating-droplet drag test, UI toggle, manual GPU parity pass.
- **CAP-M3 -- hardening** (1.5-2 days): physics tests, scenario case +
  run_validation wiring, settings-guide with recommended scales per scene
  class (droplets/glugging 8-16; thin sheets 4-8 with delayed-breakup
  note; crown splashes 2-4).
- **CAP-M4 -- band cropping, contingency only** (1-2 days, only if CAP-M3
  shows CG blow-up at two-phase + scale 16). **Rewritten (review fix)**:
  the earlier "multigrid diag_shift" idea is infeasible as specced --
  `multigrid.build_hierarchy` (multigrid.py:118-123) requires fully even
  dims and MAC face grids are (nx+1, ny, nz), so it would degenerate to
  single-level Jacobi. Instead, reuse the `pressure.crop_boxes`-style
  approach: solve only a tight bounding box of delta_e > threshold (the
  system is identity outside the band), which bounds iteration cost with
  machinery consistent with the existing architecture.

Total CAP: ~5-8 working days for M0-M3.

---

## Initiative ERR: error-aware time-step control (Limitation L1, budgeted)

Added in revision 2 (the external plan's Phase 2, redesigned for this repo
and corrected through two adversarial reviews). The original roadmap treated
L1 -- quality loss as dt grows -- as inherent. ERR reframes it: the user's
Target CFL stays a hard UPPER bound, but cheap per-substep diagnostics may
LOWER the effective CFL for the next dt decision when error indicators spike
(impact overshoot, thin-obstacle approach, clamp-rate spikes, P2G
under-sampling), raising it back with hysteresis when quiet. Violent moments
automatically get more, smaller, still frame-quantized equal substeps; calm
stretches keep the full large-step advantage. Strictly predictor-only: NO
step rejection or re-simulation, ever (rejection breaks Eq. 10-11 residual
carryover, the RNG stream contract, and the equal-subdivision invariant, and
needs full-state rollback snapshots). Honest framing mirroring TIME's: ERR
does not reduce error at a given dt; it reallocates substeps so end-to-end
error meets a budget at lower total cost than the best fixed CFL. Whether
the signals beat plain vmax adaptation is the central open question, so the
initiative is kill-gated on a diagnostics-only study before any controller
code.

### Design

**Single-site bounded multiplier.** cfl_eff = cfl_target * r with
r in [r_floor, 1], applied at exactly one site -- the dt candidate in
step_frame (solver.py:2142) BEFORE the capillary clamp and BEFORE the
equal-substep quantization dt = t_rem/ceil(t_rem/dt) (solver.py:2153) -- so
the paper's Alg. 1 l.7 stepping is preserved verbatim and per-substep CFL
provably never exceeds target. Controller state is host scalars (r, quiet
counter, last vmax) in NEW stflip/step_control.py (bpy-free, xp-agnostic,
pure functions + a small dataclass; per-signal constants are module
constants, not Params). Init (review fix): FRESH bake starts at r = 1
(trust the user's Target CFL until evidence arrives); RESTORE re-initializes
at r = r_floor (signal history was lost; conservative). The asymmetry is
documented and unit-tested on both paths. Resume semantics (review-corrected
wording): a resumed guard-on bake diverges PERMANENTLY from the uninterrupted
trajectory -- any dt perturbation does, identical in kind to changing the
seed, because each substep consumes n RNG draws; the recovery dt sequence is
conservative (each recovery dt <= what the uninterrupted bake would have
chosen at the same state), repeated restores from the same checkpoint are
bit-identical to each other, and guard-off resume stays bit-exact. No
checkpoint keys (Decision 3).

**Signals** (all behind an internal non-Params diagnostics attribute; the
default path is bit-identical; 0-d device scalars batched into the existing
per-substep vmax sync at solver.py:2156):

1. Clamp-bind fraction, computed POST-HOC from dt_act
   ((dt_act <= 0) | (dt_act >= 2*dt) -- mathematically identical to testing
   the pre-clip operands) in a helper called from one line at the end of
   _step_core, away from the jitter block lines 2093-2104 that SAMP-M3 /
   CALM-M3 / NORM-M1 own (review fix 2, option b: ERR-M1 accepts one
   possible rebase against those; ERR-M2 hard-serializes after them). The
   paper reports < 1 percent binding in healthy runs; a spike means the
   temporal slab is mis-sized.
2. Under-sampled-face fractions from accumulators _p2g already builds:
   frac_marginal = count(valid & (m < 0.25*m0)) / max(count(valid), 1) and
   frac_subthreshold normalized by count(valid) -- NEVER by total face count
   (window-shape dependent, review fix 5); window shape recorded alongside.
   The study evaluates frac_marginal both absolute and baseline-relative
   (ratio to trailing median) and lets the ROC pick -- its quiescent value
   scales with interface-area fraction and will not transfer as an absolute
   constant.
3. Predicted max velocity: g_k = vmax_k / vmax_{k-1} smoothed over 2-3
   substeps (de-biases max-statistic rectification noise),
   vmax_pred = vmax_k * clip(g_smooth, 1, G_CAP=2); the dt candidate uses
   vmax_pred. This targets a paper-documented weakness (p.10: MaxStep uses
   previous-step vmax and "can significantly underestimate" during splashy
   intervals, Fig. 7); the repo already logs estimated-vs-actual CFL
   (solver.py:2154/2158) so the gap is measurable today. Bound (review fix
   3): the COMBINED reduction from r and vmax_pred is floored at
   dt >= r_floor * cfl_target * dx / vmax_k -- without this the worst case
   is 8x substeps, not 4x; and free-fall scenes have g clipped at 2 for the
   whole drop, so "quiet segments pay only diagnostic overhead" holds only
   with the combined floor. An accelerating no-alarm scene (free-falling
   block) joins the do-no-harm suite with a substep-count bound.
4. Paired-partition noise probe (needs SAMP-M1 ids -- declared dependency
   for this signal only): s_p from a HASH BIT of the particle id (raw id
   parity is seeding-order stratified and underestimates position-sampling
   noise in ordered calm regions -- review fix 4b); deposit
   d += weighted_volume * s_p (same operand as the c_m scatter,
   solver.py:1024-1025, so m_even + m_odd = c_m holds in BOTH phase modes --
   review fix 4a); signal = RMS over interface cells (0.05 < c_phi < 0.95)
   of |d| / max(c_m, 0.05 * ppc) (floor at a kappa_noise fraction of m0,
   not eps_m, so near-empty cells do not dominate -- review fix 4c). Direct
   Monte Carlo relative-standard-error of interface deposition -- the
   quantity L3 noise IS. Cost: +14 percent of scatter count in free-surface
   mode, +7 percent in two-phase, +2-4 percent of a substep either way
   (review-corrected figures); one transient c-grid accumulator at window
   shape. Doubles as SAMP-M5's attribution probe (built once here).
5. Near-solid fast fraction (thin-obstacle guard, fires BEFORE impact):
   mean((sdf_at_particle < 2*dx) & (|v_p|*dt/dx > 2)), one 8-tap
   _sample_cells gather (pattern solver.py:1556).
6. Capillary/viscous engagement: report-only; when the Brackbill clamp set
   dt the guard is inert by construction (dt already below the CFL cap) --
   and CAP's st_clamp_scale relaxation makes the guard MORE relevant in
   those scenes (declared follow-on, out of scope).

**Controller law** (review fix 1 -- decay and release are mutually
exclusive): per signal, alarm a_i = clip((s_i - lo_i)/(hi_i - lo_i), 0, 1);
combined A = max_i(a_i) (signals are correlated; worst-case response
wanted). If A >= A_RELEASE (0.25): r <- max(r_floor, r * BETA_DOWN**A),
BETA_DOWN = 0.5. If A < A_RELEASE: pure quiet -- no decay; after H = 3
consecutive quiet substeps, r <- min(1, r * BETA_UP), BETA_UP = 1.25, and
the quiet counter does NOT reset after a raise (once armed, raise every
quiet substep; geometric recovery at 1.25/substep). Without this exclusivity
the original law monotonically pinned r at the floor under mild sub-threshold
chatter -- a permanent ~4x substep inflation the no-oscillation test could
not see. Steady-state unit tests at constant A in {0.1, 0.2, 0.3} assert
convergence to 1 below A_RELEASE and to r_floor above it; a gentle-flow
scene (slow pour/slosh) joins do-no-harm with a substep-count-ratio bound,
because the still pool sits at A = 0 exactly and cannot exercise the
chatter band. r_floor = cfl_target**(-quality_guard_strength), additionally
clamped so cfl_eff >= min(cfl_target, 1) (w.r.t. the predicted velocity
estimate; realized CFL against current vmax can sit below 1 by up to G_CAP
-- framing, not a bug).

**Closed-loop contamination** (review fix 2): signals 1-2 are partially
CAUSED by dt decreases -- halving dt pushes carried residuals against both
clip bounds and can push theta outside W_T support (zero-weighted per the
solver.py:968-970 policy), spiking both signals for 1-2 substeps as an
artifact of the controller acting. Mask signals 1-2 for one substep after
any dt decrease exceeding dt_k < 0.75 * dt_{k-1} (from any cause; the dt
sequence is host-side). ERR-M2 acceptance replays the ERR-M1 scenes with
the controller ENABLED and asserts closed-loop false-alarm rate and
time-at-floor within stated factors of the open-loop predictions; the K2
ROC gate is re-evaluated on at least one closed-loop trace.

### Params, stats, cost

ERR-M2 adds Params.quality_guard: bool = False and
Params.quality_guard_strength: float = 0.5 in [0, 1] in ONE PR (one
Decision-1 invalidation + release note). FrameStats (plain dataclass, not
the Decision-2 schema) gains per-step lists: clamp_bind_fraction,
undersampled_face_fraction, interface_noise_rms, near_solid_fast_fraction,
capillary_clamped, cfl_effective_values, window_shape -- consumed by
validation.py as report-only diagnostics (validation.py:834-850 pattern).
Cost: default off = zero (bit-identity is a test); diagnostics on <= 5-8
percent per substep worst case (dominated by signal 4), CPU and GPU alike;
controller math is host-side float ops. Guard on: alarmed segments run up
to cfl_target/(cfl_target*r_floor) more substeps bounded by the combined
floor; quiet segments pay only diagnostics.

### Validation and kill criteria

Unit: controller-law bounds, steady-state convergence, hysteresis
no-oscillation trace, both init paths, vmax_pred cap. Solver: guard-off
bit-identity (TIME-M2 parity pattern); per-substep estimated CFL <=
cfl_target; equal-substep invariant; drift bound under guard-driven varying
dt (tests/test_solver.py:442 pattern); still pool takes the SAME substep
count as guard-off (fresh init r = 1 makes this pass by construction);
restore determinism (two restores identical). ERR-M1 acceptance (kill
point): on three known failure episodes -- (a) dam-break impact frames at
CFL 16 (MATCHED_CASES machinery), (b) a thin-obstacle tunneling scene (1-2
cell wall, jet, CFL 16; tunneled-particle count), (c) calm-pool noise via
the CALM-M1 metric -- each retained signal must show median lead time >= 1
substep before the failure metric spikes, alarms in < 20 percent of
non-failure substeps, and must add lead time or precision OVER the
vmax-growth predictor alone (per-signal redundancy matrix published).
Threshold transfer checked at two resolutions (32^3 and 64^3);
resolution-dependent thresholds are documented as a limitation if found,
not papered over. ERR-M3 A/B (adoption): matched dam-break, guard@16
(strength 0.5) vs fixed CFL {8, 16}; equal-error = guard@16 phase-RMSE and
threshold-IoU within seed spread of fixed@8 on impact frames; efficiency =
end-to-end wall clock (Decision 6a accounting) <= 0.8x fixed@8; do-no-harm
= within 1.05x of fixed@16 on still pool AND the gentle-flow scene AND the
free-fall scene, thin-obstacle tunneling strictly below fixed@16, plus one
conservation-flavored metric (KE half-life on a standing-wave slosh,
guard-on vs guard-off -- review fix 6: extra substeps add PIC-bleed
smoothing that impact-frame phase-RMSE cannot see). KILL: K1 -- if after
ERR-M1 nothing beyond vmax growth adds >= 1 substep median lead time or
precision, collapse the initiative to a one-line vmax_pred-only flag
(documented negative result; study tool retained). K2 -- no threshold
achieves the lead time at < 50 percent false alarms (ROC unusable, checked
on a closed-loop trace too): cancel outright. K3 -- equal-error end-to-end
win over the best fixed CFL chosen WITH HINDSIGHT < 20 percent: keep
default-off experimental or revert (diagnostics stay -- independently
valuable telemetry). K4 -- any do-no-harm regression: fix or revert.

### Milestones

- **ERR-M1 -- diagnostics only, the kill point** (2-3 days + study compute;
  shippable alone). step_control.py signal functions + FrameStats lists +
  gated capture (zero Params change, default path bit-identical) +
  validation report-only wiring + tools/run_step_control_study.py producing
  lead-time/ROC/redundancy tables on the three failure scenes. Depends on
  CALM-M1 (metric); the parity-probe arm alone depends on SAMP-M1 (ids) and
  is appended later if SAMP-M1 slips.
- **ERR-M2 -- controller behind default-off flag** (3-4 days; only on
  ERR-M1 GO). Params pair, dt-decision wiring, conservative restore
  re-init, closed-loop replay acceptance, full test set, manual 5090
  parity + timing (Decision 6d).
- **ERR-M3 -- A/B, adoption decision, UI** (2-3 days; only on ERR-M2).
  Decision recorded in docs/design/error-aware-step-control.md as PROMOTE /
  KEEP-EXPERIMENTAL / REVERT (the TIME-M3 pattern).

ERR non-goals: no step rejection/re-simulation; no per-region or
per-particle dt (breaks the equal-substep structure and the P2G slab
identity); no PID on a scalar error norm (no cheap global error norm
exists; embedded pairs are rejection in disguise); no per-scene threshold
Params (module constants only); no checkpointed controller state; no guard
authority above Target CFL or below CFL 1. Anchors verified against
0eadebd (v0.23.1).

---

## Recommended order

Sequenced across initiatives by value-per-effort and dependency.
Principles: instruments before the experiments that need them; bit-identical
refactors and pure libraries before hot-path wiring; cheap user-visible
value early; kill points for speculative work before big spends; the THREE
edits to the jitter block (NORM-M1, then SAMP-M3, then CALM-M3) are
serialized to avoid textual conflicts; the single checkpoint bump (SAMP-M1)
and the single metrics-schema bump (ENER-M0) land early and alone.

**Batch A -- instruments and cheap wins (first ~1 week)**

1. **CALM-M1** -- shared calm-surface metric + S1/S2/S3 scenes. The
   instrument for SAMP-M5, TIME-M3, and CALM-M2..M4; nothing downstream
   of it can gate without it. No solver change.
2. **ENER-M0** -- energy/angular-momentum metrics (the one metrics schema
   bump), rotating-tank scenario, baseline decay curves incl. the CFL-1
   aperture floor. The acceptance data for all of ENER.
3. **CAP-M0** -- clamp-scale dial + impulse limiter. First user-visible
   win, nearly free, honest 2-4x framing.
4. **SAMP-M1** -- stable particle ids + substep counter (the one
   checkpoint bump). Zero behavior change; lands early to minimize
   codex-schema collision; independently useful for handoff.

**Batch B -- libraries, refactors, immediate features (~1-2 weeks)**

5. **ENER-M2a** -- `_project` + extrapolation refactor with bit-identical
   guard. Hot region; land fast and atomic.
6. **NORM-M1** -- exact gamma-conditioned weighting (jitter-block edit #0:
   its `_jitter_gamma` extraction owns the first edit of solver.py:
   2093-2104). Cheap, exact, default-on; provides SAMP-M5's normalization
   axis, so it must precede the A/B.
7. **ENER-M1** -- vorticity confinement. Immediate artist value, low risk;
   hygiene rule Decision 6c applies from day one.
8. **SAMP-M2** -- sampling library (point scrambling + index shuffling).
   Pure library.
9. **CAP-M1** -- st_implicit library. Pure library.
10. **TIME-M1** -- estimator study, kill point #1. Decides TIME's fate
    before any solver spend; can run in parallel with 5-9.
11. **ERR-M1** -- step-control diagnostics study, kill point #2 (needs
    CALM-M1; parity-probe arm needs SAMP-M1). Solver code is one line at
    the end of _step_core plus gated _p2g reductions -- accepts one
    possible rebase against the jitter-block edits.
12. **CALM-M2** -- render-path calm denoise. Independent render win,
    default byte-identical.

**Batch C -- opt-in features end-to-end (~1-2 weeks)**

13. **SAMP-M3** -- temporal_sampling wiring (jitter-block edit #1, after
    NORM-M1 rebases cleanly).
14. **CALM-M3** -- gamma_mode interiorness (jitter-block edit #2, after
    SAMP-M3; owns the gamma_prev persistence path when combined with
    NORM).
15. **CAP-M2** -- st_implicit activation + drag/no-op tests.
16. **ERR-M2** -- quality-guard controller (only on ERR-M1 GO;
    hard-serialized after items 13-15 and CAP-M0 -- it reads the step_frame
    dt region CAP-M0 edits and sits adjacent to the jitter block).

**Batch D -- experiments, hardening, the big feature (~2-3 weeks)**

17. **SAMP-M5** -- sampling A/B + decision (needs CALM-M1, NORM-M1,
    SAMP-M1..M3; 8 seeds per Decision 6b).
18. **CAP-M3** -- surface-tension hardening + validation scenario.
19. **CALM-M4** -- deformation-aware activity + calibration sweep (needs
    CALM-M1; benefits from CALM-M3; gated by the S2/S3 scenes; 8 seeds per
    Decision 6b; hysteresis is a sweep option only).
20. **ERR-M3** -- quality-guard A/B + adoption decision (only on ERR-M2).
21. **ENER-M2b** -- advection-reflection, corrected scheme (needs ENER-M0
    and ENER-M2a; the largest single item; merge-blocking induction proof
    and hydrostatic trace test; optional SOAR arm; must re-verify NORM's
    vel-immutability contract).

**Batch E -- conditional tail**

22. **TIME-M2 then TIME-M3** -- only on TIME-M1 GO (largest conditional
    spend; TIME-M3 reuses the CALM-M1 metric).
23. **ENER-M3** -- IVOCK spike, only if ENER-M2b leaves a measured
    enstrophy gap; time-boxed; expected documented rejection.
24. **SAMP-M6** -- adoption (on SAMP-M5 pass); **SAMP-M7** -- optional
    spatial Sobol seeding.
25. **CAP-M4** -- band cropping, only on CAP-M3 iteration-count evidence.

Explicit cross-initiative dependencies:

- SAMP-M5 requires CALM-M1 (metric + PSD probes), NORM-M1 (normalization
  axis), SAMP-M1+M2+M3, and ERR-M1's parity probe for attribution.
- TIME-M3 requires TIME-M2 (which requires TIME-M1 GO) and reuses CALM-M1.
- CALM-M4 requires CALM-M1 (S2/S3 gates); CALM-M3 recommended first.
- ENER-M2b requires ENER-M0 (gates) and ENER-M2a (refactor);
  ENER-M3 requires ENER-M2b results.
- CAP milestones are internally sequential (M0 independent; M2 needs M1;
  M3 needs M2; M4 conditional on M3).
- ERR-M1 needs CALM-M1 (+SAMP-M1 for the parity arm); ERR-M2 needs ERR-M1
  GO and serializes after the Batch-C jitter/step_frame edits; ERR-M3
  needs ERR-M2.
- NORM-M1, SAMP-M3, and CALM-M3 all edit solver.py:2093-2104: serialize
  them in that order.

## Non-goals

- **MPM/SPH generalization (paper Outlook O3).** This is a liquid addon;
  the spatiotemporal viewpoint stays FLIP-only here.
- **Exact energy/momentum conservation.** Impossible in this splitting
  family; ENER reduces splitting dissipation and offers a look control,
  nothing more is claimed.
- **K > 2 time levels or quadratic-in-time reconstruction; per-level
  pressure projections; true space-time (4D) projection; time-interpolated
  G2P.** TIME is bounded to the K = 2 linear case; the paper found
  time-interpolated G2P not worthwhile.
- **Per-face adaptive regularization** for the temporal fit (flagged if
  needed, not built).
- **Changing face-validity semantics from the W_T accumulators** as part
  of TIME (any such change is its own gated experiment).
- **Local capillary sub-stepping of the frozen ST force** (identical total
  impulse, zero stability benefit) and **Sussman-Ohta-style implicit
  interface MCF** as the primary L4 fix (nonlinear phi solve; too heavy).
- **Folding surface tension into the PPE coefficients or GFM jump terms**
  in v1 (accuracy follow-up, entangles the preconditioner).
- **Removing the Brackbill clamp.** It is only ever scaled by a bounded,
  validated factor.
- **Multigrid diag_shift on MAC face grids** as originally drafted for
  CAP-M4 -- judged infeasible in review (`multigrid.build_hierarchy`
  cannot coarsen the odd face-grid dimension); replaced by band cropping.
- **The original ENER-M2b step order** (no mid-step particle update,
  u_hat as transport velocity) -- rejected in review as a wrong algorithm;
  superseded by the corrected scheme above.
- **Unscrambled Sobol/Halton and Cranley-Patterson rotation as primary
  samplers**; **scipy or precomputed Sobol tables**; **per-cell temporal
  stratification** (documented follow-on only).
- **Full Bhattacharya 2011 surface denoising** (redistance + biharmonic +
  exact particle-distance constraints) -- the masked-MCF band-clamp
  variant is the scoped version.
- **Default-elision of new Params fields from the simulation
  fingerprint** -- rejected; invalidation is accepted and release-noted
  (Decision 1).

From the revision-2 reconciliation (external-plan items rejected, with
reasons, so they are not re-litigated later):

- **Step rejection / re-simulation in any form** (even "research runs") --
  breaks Eq. 10-11 residual carryover, the RNG stream contract, and the
  equal-subdivision invariant; needs full particle-state rollback. ERR is
  predictor-only.
- **"Fix adaptive-step clamping bias" as a work item** -- the repo already
  exceeds the paper here: equal-subdivision quantization
  (solver.py:2152-2153) plus the deliberate no-clip theta policy
  (solver.py:968-971, out-of-slab samples get ZERO weight, not clipped peak
  weight). Only the clamp-rate counter survives, as ERR signal 1.
- **Second-order advection-reflection as ENER's starting point** -- formal
  order requires instantaneous midpoint fields and deterministic sample
  times; slab-integrated P2G at jittered times provides neither. Optional
  SOAR arm inside ENER-M2b only.
- **"IVOCK as the cheaper baseline"** -- factually wrong (3-component
  vector-Poisson streamfunction solve vs one extra scalar projection);
  ENER-M3's expected-rejection spike stands.
- **Power-PIC and covector/CO-FLIP** -- per-step optimal transport / full
  solver redesign; scope creep for a NumPy/CuPy addon. APIC as a
  comparison "control" needs no work: Params.transfer="apic" exists and
  joins run matrices.
- **PMJ sampler arm** -- PMJ solves 2D-stratified point-set problems; the
  temporal setting is 1D-per-particle across steps, where Sobol dim0 +
  Owen is already optimally dyadically stratified. Pure cost.
- **Antithetic / locally stratified temporal pairs** -- same defects as
  the already-rejected per-cell stratification (order-dependence under
  compaction, temporal-coherence destruction) plus pair decorrelation
  after advection; at most a stateless hash-parity arm if SAMP-M5 leaves
  an unexplained cross-particle-variance signal.
- **A literal 2D solver mode for screening** -- major infra the repo does
  not have; pseudo-2D thin slabs (e.g. 64x4x48) on the existing 3D code
  deliver the screening value at near-zero cost.
- **The 720-run factorial first experiment** -- lacks the adaptive-gamma
  control arms (its calm-pool cells would measure attenuated jitter, the
  exact confound SAMP-M5's arm matrix isolates); its three good deltas
  (translating slab, normalization axis, 8 seeds) are adopted into
  CALM-M1/NORM-M1/SAMP-M5 instead.
- **A ">=2x lower HIGH-frequency calm-surface noise" exit criterion** --
  targets the frequency band LDS cannot improve (per-step cross-particle
  scatter is unchanged by independent per-particle scrambles); the
  low-frequency temporal-PSD gate on gamma-active arms stays.
- **Estimator-variance-driven jitter controller** -- the variance input
  requires the paired estimator in the hot path; CALM-M4's cheaper
  activity signals cover the need; hysteresis noted as a CALM-M4 sweep
  option only.
- **Swept collision detection + trajectory-error-controlled advection
  substeps** (external Phase 5) -- real potential win, real tunneling
  risk; performance-backlog note, revisit on a demonstrated tunneling or
  profiling case, not a roadmap initiative. Same for vague
  advancing-front/extrapolation and density-ratio-solver items already
  covered by multigrid + crop boxes + CAP-M4.

## Open questions

Carried explicitly rather than hidden (a milestone owner should answer
these in the PR that touches them):

1. SAMP: does one lam-free scramble hash suffice, or will the PSD spike
   check force a stronger per-bit nested scramble? (sampling.py isolates
   the swap.)
2. TIME: can a single temporal_fit_reg serve calm and violent regimes
   simultaneously? If not, that is a NO-GO by design (per-face adaptivity
   is a non-goal).
3. CALM-M4: is |n.D.n| sufficient, or does the calibration sweep force
   the conservative full-D fallback? What threshold for the unsteadiness
   term avoids re-enabling jitter on merely-noisy grids?
4. ENER-M2b: the two-phase mirror u_hat = 2*u1 - u_star is exact in the
   rho-weighted metric only for constant rho; the phi_f mask bounds the
   damage, but the 800:1 smoke test decides whether a stricter gas-side
   restriction is needed.
5. CAP: where do CG iteration counts land at two-phase 800:1 and scale
   16 -- is CAP-M4 needed at all?
6. NORM: does the vel-immutability recompute contract survive ENER-M2b's
   restructured step tail in practice (the debug assertion decides), and
   does the two-phase phi_f correction measurably change 800:1 material
   assignment on real scenes?
7. ERR: after ERR-M1, does anything beyond the vmax-growth predictor earn
   its keep (K1), and do open-loop thresholds transfer to closed loop
   (K2's closed-loop re-check)?
