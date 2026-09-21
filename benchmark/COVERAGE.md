# Corpus coverage audit

The corpus tests many feasible inputs. It does not prove that a solver works
for every feasible input. Passing a million broadly sampled cases is not a
substitute for testing singular geometry and its numerical neighborhoods.

## Why Haar and linspace missed cases

The measured source arrays have different blind spots:

| Source | Rows | Observation |
| --- | ---: | --- |
| Haar | 300,000 | No input or target has a pairwise eigenvalue distance below `1e-4`. The minimum distances for C, G, T are approximately `0.00103`, `0.00135`, `0.00186`. |
| Linspace | 761,308 | All stored coordinates are multiples of `1/32`. Many spectra have exact repeated roots, but no observed minimum gap lies between `1e-14` and `1e-4`. |
| Stratified | 13,181 | Includes exact strata and small gaps, but does not independently cross every perturbation scale with every local-matrix mode. |

An eigenvalue distance here is `abs(lambda_i - lambda_j)` on the unit circle,
not a distance between coordinate triples. The measured counts and source
hashes are recorded in [coverage_sources.json](coverage_sources.json).

Exact collisions and shared invariant subspaces occupy lower-dimensional
sets. Continuous random sampling does not deliberately test those sets.
A fixed grid can hit some exact sets, but misses small perturbations around
them and relations that do not align with the grid.

There are also two distinct kinds of boundary. A Weyl wall describes a
repeated eigenvalue of one matrix. A multiplicative Horn wall constrains the
three spectra together. Falbel and Wentworth distinguish outer walls, where
representations are reducible, from inner walls that admit both reducible
and irreducible representations. Thus testing individual Weyl walls does
not establish coverage of all joint boundaries. See their
[introduction and Theorem 3](https://arxiv.org/html/math/0506100v1).
The eigenvalue inequalities themselves are described by
[Agnihotri and Woodward](https://arxiv.org/abs/alg-geom/9712013).

## What the stratified generator covers and misses

The existing generator crosses 29 exact/near/interior families for C and G.
It has 10,933 generated matrix-witness rows, 15 fixed regressions, and 2,233
target-stratum rows selected by a Horn feasibility test.

Its default 13 samples per input-family pair use `sample % 6` to select the
local mode and `sample % 13` to select the scale. Consequently a particular
C-scale is tested in only one mode. The near-identity local mode occurs at
samples 5 and 11, not at all 13 scales.

Canonical matrix creation and target extraction pass through
`LocalEquivalenceClass`, whose coordinate representation uses a `1e-14`
grid. The original C/G coordinates are retained, but the matrix used to
construct the target can lose their smallest offsets. This is acceptable
as approximate data at a `1e-8` test tolerance; it is not evidence that the
smallest intended displacement was exercised faithfully.

The target-stratum section uses the live Horn oracle and a `2e-12`
feasibility tolerance. Those rows do not carry independently constructed
matrix witnesses. All old rows are retained, including failures; this audit
does not silently repair, filter, or rationalize them.

## Solver mechanisms that require separate cases

| Mechanism | Relevant production code | Required construction |
| --- | --- | --- |
| Scalar, 3+1, 2+2, 2+1+1 multiplicities | `problem.rs`, `support_strata.rs`, `resonance.rs` | Exact and split spectra in inputs and target separately |
| Routed product collisions with simple input spectra | `problem.rs`, `resonance.rs` | Equal/inverse input spectra and small perturbations |
| Permutation, single-plane, disjoint-block, fixed-index supports | `support_strata.rs`, `one_plus_three.rs`, `two_plus_two.rs` | Planted frames for every coordinate placement |
| Nearly reducible frames and ill-conditioned root formulas | `charts.rs`, `interior.rs`, `three_givens.rs`, `chart_precision.rs` | Independent angle and eigenvalue-gap ladders |
| Ordering and central target lift | `certificate.rs`, `problem.rs` | Permutation-aware witnesses and both target signs |

For example, `spectrum_kind` compares a squared root distance with machine
epsilon, corresponding to a distance near `1.49e-8`. Other conditioning
checks use different scales. A label such as "near wall" alone does not
show that a test approaches these decisions from both sides.

## Additions and independent feasibility checks

[generate_targeted_cases.py](generate_targeted_cases.py) constructs additional
1,455 cases from explicit real orthogonal witnesses. It does not ask the production
solver to accept a case before adding it. Coordinate conversion does not use
`LocalEquivalenceClass` and does not round coordinates to a decimal grid.

The additions include all 24 permutation placements, all six coordinate
planes, all three disjoint 2+2 partitions, and all four fixed-index 1+3
supports. They include 408 prescribed-target multiplicity cases, 96 cases
with colliding routed products, and a 320-case Cartesian product of input
multiplicity, input gap, local angle, factor role, and frame mode. These
are construction counts, not claims of distinct solver branches.

Repeated target spectra require particular care. The construction chooses
`A = D^2`, a prescribed target diagonal `T`, and a real orthogonal `R`, then
forms `B = D^-1 R T R.T D^-1`. This B is symmetric and unitary, so its real
and imaginary parts commute and have a common real orthogonal eigenbasis.
That basis supplies the sandwich witness. The matrix `AB` is similar to
`R T R.T`, which establishes the prescribed target spectrum in exact
arithmetic. The generator checks the resulting floating-point witness too.

Each added row must pass the independent matrix checker with all errors at
most `1e-12`, tighter than the grader's `1e-8`. The generated witnesses had a
maximum spectral error of `3.283e-15` and maximum Gram/determinant errors
of `1.999e-15`. A construction failure stops
generation; production failures do not remove cases. Witnesses, construction
labels, and measured errors are saved separately from the submitted input.

## One benchmark file and remaining limits

The benchmark reads `corpus/feasible_all.npy`, containing **1,075,950 rows**.
Its first 1,074,495 rows retain
all previous inputs in their original order: basic fixtures, stratified,
linspace, Haar. Targeted additions follow. No deduplication changes the
weighting of the existing suite. `feasible_all.json` records source hashes
and half-open row ranges; the builder checks every copied slice bit for bit.

This audit does not enumerate and independently witness every Horn facet,
every facet intersection, or every resultant/discriminant degeneracy of the
solver's polynomial formulas. Construction labels do not prove that a
particular implementation branch ran. The corpus remains a fixed empirical
benchmark. Fresh seeds and future minimized failures should extend it through
a versioned rebuild, not be described as mathematical completeness.

A reducible witness need not lie on an outer Horn facet. Likewise, a
`nextafter` perturbation need not survive eigenphase extraction or cross the
intended numerical predicate. Actual target gaps are recorded, and the
generator checks the stored coordinates rather than assuming the intended
perturbation survived.

The production replay passed 1,451 of the 1,455 additions. It declined two
near-3+1 target cases and returned two matrices marginally above the `1e-8`
spectral limit. The old cases reproduced their previous coverage exactly.
See [the measured baseline](BASELINE.md) for row IDs and numerical caveats.
