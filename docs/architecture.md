# Source guide

The [README](../README.md) describes the crate and API. The
[research contract](research.md) defines the mathematics and conventions.
This page describes the implementation and the redesign it needs.

## Current control flow

`lib.rs` exports `solve` and `solve_with_factors` from `cascade.rs`.
Both call `solve_using`, which prepares the problem and calls `solve_inner`.

`solve_inner` tries scalar and rank-one constructions, then `solve_prefix`,
then `charts::solve_full`. The prefix tries support patterns and specialized
algebraic constructions. The chart search has its own dispatch over charts,
repeated spectra, alternative factor roles, and snapped spectra.

Algebraic candidates pass through `certificate::compiler_solution`.
`numerical::verify` then checks the selected result against the original spectrum.
On rejection, `numerical::refine` tries to repair the candidate. If needed,
`numerical::solve` tries deterministic starts, swapped factors, and inverse-factor
problems. A successful numerical candidate goes through the same acceptance path.

`solve_with_factors` reuses the verification eigenbasis for endpoint factors.
It checks their reconstruction before returning them.

## Where the code lives

| Source | Current responsibility |
|---|---|
| `cascade.rs` | Public entry points, dispatch, shared math, profiling, and diagnostic routes |
| `problem.rs` | Spectra, target branches, characteristic coefficients, and multiplicity classification |
| `certificate.rs` | Candidate checks, spectral acceptance, and older diagnostic endpoint routines |
| `numerical.rs` | LM, spectral verification, and production endpoint factors |
| `support_strata.rs`, `one_plus_three.rs`, `two_plus_two.rs`, `resonance.rs`, `klein.rs` | Specialized algebraic constructions |
| `charts.rs`, `interior.rs`, `three_givens.rs`, `chart_precision.rs` | Algebraic chart search, reconstruction, root selection, and numerical precision helpers |
| `radical.rs`, `cpoly.rs` | Radical constructions and polynomial root machinery |

These boundaries overlap. The table is a map of the current code, not an
endorsement of its organization. In particular, `cascade.rs` and `charts.rs`
both control search order, while verification spans several modules.

## Design direction

The production reading path should expose preparation, candidate construction,
refinement, acceptance, and optional factor extraction in that order.

- One place owns the outer search order and retry policy. Each construction
  owns its mathematical subproblem and local enumeration.
- One place owns final acceptance against the original input. Cheap algebraic
  rejection tests remain distinct from that acceptance check.
- Shared spectral and polynomial operations have explicit owners and imports.
  A constructor does not obtain unrelated utilities through its parent.
- A named type represents a mathematical object or a meaningful search record.
  Avoid long positional tuples, integer modes, and wrappers around one matrix.
- Diagnostics stay outside the production reading path.

Before moving every existing algorithm, measure its contribution. Disable one
route at a time and compare coverage and runtime, including difficult boundary
cases. Retain a specialized route for demonstrated correctness or speed benefits.
Do not infer redundancy from similar names or equations.

Use the existing corpus for these changes. It checks actual spectra and endpoint
reconstruction independently of the internal acceptance path. Measure GULPS
integration as well as isolated solver time. The previous endpoint-factor change
showed why those measurements can differ.

This is a design target. The source has not yet been reorganized to satisfy it.
