# Source guide

The two solver functions in [`src/lib.rs`](../src/lib.rs) share input
preparation, algebraic constructions, and result verification. If the algebraic
search finds no accepted matrix, they try numerical refinement with restarts.
`solve_with_factors` uses the eigenbasis from verification to recover the
endpoint factors and check that they reconstruct the requested gate.

For the API and coordinate conventions, see the [README](../README.md) and
[mathematical formulation](research.md). The [researcher guide](researcher.md)
explains how to compare a separate algorithm with the corpus runner.

## Source layout

| Source | Responsibility |
|---|---|
| `lib.rs` | Public API, algebraic search, and numerical fallback |
| `problem.rs` | Input spectra, target branches, and multiplicity classification |
| `spectral.rs` | Spectral matching, final verification, and endpoint factors |
| `numerical.rs` | Levenberg–Marquardt refinement and deterministic starts |
| `diagnostics.rs` | Route reports, candidate verification, and stage timing |
| `corpus.rs` | Independent checks, candidate comparison, and accuracy and latency reports |
| `algebraic/mod.rs` | Scalar and rank-one formulas, dispatch, and shared frame operations |
| `algebraic/certificate.rs` | Candidate checks, repeated-root repair, and calls to the verifier |
| `algebraic/support_strata.rs` | Vertex, edge, face, and confluence dispatch |
| `algebraic/klein.rs` | Quaternion section and eigenframe recovery |
| `algebraic/two_plus_two.rs`, `radical.rs`, `resonance.rs` | Repeated-spectrum constructions |
| `algebraic/interior.rs`, `three_givens.rs` | Three-Givens charts and their polynomial equations |

The search tries scalar and rank-one formulas, vertices, edges, faces, the
Klein section, repeated-spectrum formulas, and three-Givens charts. Numerical
recovery handles the remaining cases. Each search has a bounded budget and
can decline even when a solution exists.

Every accepted frame passes `spectral.rs` against the original spectrum.
`Solution` stores the real frame and its verification state together, so
endpoint recovery uses the same eigenbasis without another decomposition.
A successful result always has that state; failure is `None`.

## Diagnostics

With the `diagnostics` feature, `solve_report` returns `Option<Solution>` with
the real frame, successful `Rung`, and spectral residual. `branch_signature`
classifies the inputs, `init_tables` prepares the three-Givens lookup table,
and `certify_frame` checks a proposed complex frame through the production
acceptance path. Set `PROF=1` and call `prof::dump()` for stage timings.

The retired row-in-plane chart solver, waypoint experiments, and paired-edge
ablation APIs remain in the source snapshot at commit `b3dbb7f`.
They are no longer part of the diagnostic interface. The two public solver
function signatures are unchanged.

## Iterating

Use the corpus to measure correctness, endpoint reconstruction, accuracy,
and latency before and after a change. Entire constructions may be removed
when simpler remaining paths cover their cases. Keep the checker and its
thresholds independent of the experiment. Record rejected approaches as well
as improvements so later work can build on the measurements.
The [September 2026 simplification](optimization.md) records the first set of
stage-removal experiments and their measured results.

Changes to factor extraction also need measurements in GULPS, where different
valid endpoint choices can affect later decompositions and synthesis time.
