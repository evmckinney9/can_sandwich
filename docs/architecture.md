# Source guide

The [README](../README.md) describes the crate and API. The
[research contract](research.md) defines the mathematics and conventions.

Start with [`src/lib.rs`](../src/lib.rs). Both public functions use the same
solve path: prepare the input, try algebraic constructions, verify the result,
and try numerical refinement or fresh starts if needed. `solve_with_factors`
uses the verification eigenbasis to recover the endpoint factors and checks
that they reconstruct the requested gate.

## Source layout

| Source | Responsibility |
|---|---|
| `lib.rs` | Public API, search order, and retry policy |
| `problem.rs` | Shared input preparation, spectra, target branches, and multiplicity classification |
| `spectral.rs` | Spectral matching, final verification, and endpoint factors |
| `numerical.rs` | Levenberg–Marquardt refinement and deterministic starts |
| `diagnostics.rs` | Research entry points and optional stage timing |
| `algebraic/mod.rs` | Algebraic dispatch and shared frame operations |
| `algebraic/certificate.rs` | Cheap candidate checks and algebraic acceptance tests |
| `algebraic/charts.rs`, `interior.rs`, `three_givens.rs`, `chart_precision.rs` | Chart enumeration, reconstruction, root selection, and precision helpers |
| `algebraic/support_strata.rs`, `one_plus_three.rs`, `two_plus_two.rs`, `resonance.rs`, `klein.rs`, `radical.rs` | Specialized constructions |

The algebraic search tries scalar and rank-one cases, specialized constructions,
then the chart search. These routines return candidates. Final acceptance in
`spectral.rs` checks a real SO(4) matrix against the original target spectrum.
Snapping spectra inside a construction does not change the acceptance target.

The numerical fallback uses the same prepared problem and spectral matching.
It has bounded iterations and can fail. Passing the corpus is not a completeness
proof.

## Working on a construction

The algebraic code remains experimental. In particular, the chart search has
several local search orders, and some older endpoint routines remain for
research diagnostics. The public solve path does not use those endpoint routines.

Before deleting a specialized construction, measure its contribution to coverage
and runtime. Similar formulas or names do not establish redundancy. Use the
existing corpus: it checks spectra and endpoint reconstruction independently of
the internal acceptance tests. Also measure GULPS integration when a change
affects factor extraction or repeated work at the crate boundary.
