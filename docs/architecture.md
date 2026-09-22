# Source guide

The two solver functions in [`src/lib.rs`](../src/lib.rs) share input
preparation, algebraic constructions, and result verification. If the algebraic
search finds no accepted matrix, they try numerical refinement with restarts.
`solve_with_factors` then uses the eigenbasis from verification to recover the
endpoint factors and check that they reconstruct the requested gate.

For the API and coordinate conventions, see the [README](../README.md) and
[mathematical formulation](research.md). If you want to test a separate
algorithm, the [researcher guide](researcher.md) explains how to use the corpus
runner without working through these internals.

## Source layout

| Source | Responsibility |
|---|---|
| `lib.rs` | Public API, search order, and retry policy |
| `problem.rs` | Shared input preparation, spectra, target branches, and multiplicity classification |
| `spectral.rs` | Spectral matching, final verification, and endpoint factors |
| `numerical.rs` | Levenberg–Marquardt refinement and deterministic starts |
| `diagnostics.rs` | Research entry points and optional stage timing |
| `corpus.rs` | Independent corpus checks, candidate comparison, and reports |
| `algebraic/mod.rs` | Algebraic dispatch and shared frame operations |
| `algebraic/certificate.rs` | Candidate checks, repeated-root repair, and calls to the shared verifier |
| `algebraic/charts.rs`, `interior.rs`, `three_givens.rs`, `chart_precision.rs` | Chart enumeration, reconstruction, root selection, and precision helpers |
| `algebraic/support_strata.rs`, `one_plus_three.rs`, `two_plus_two.rs`, `resonance.rs`, `klein.rs`, `radical.rs` | Specialized constructions |

The algebraic search starts with scalar and rank-one cases, then tries the
specialized constructions and chart search. Every proposed matrix must pass
verification in `spectral.rs` against the original target spectrum, even if
a construction used a nearby spectrum with repeated roots. The numerical
fallback uses the same prepared inputs and spectral checks, with a fixed
iteration budget that can expire before it finds a solution.

Once a matrix passes verification, the solver keeps its spectral state for
endpoint recovery, avoiding another check and diagonalization. Diagnostic
callers can inspect the frame, route, and residual through `Solution`; its
private verification state is populated by the solver functions.

## Working on a construction

The algebraic code remains experimental, particularly the chart search,
which has several local search orders. Some older endpoint routines are also
available through research diagnostics, although the public solver doesn't
call them.

Before removing a specialized construction, use the corpus to measure which
cases depend on it and how its removal affects runtime and accuracy. Routines
with similar formulas may handle different degeneracies, which is why the
corpus checks spectra and endpoint reconstruction independently of the solver's
own tests. Changes to factor extraction also need measurements in GULPS,
where repeated decompositions can affect the total synthesis time.
