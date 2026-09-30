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
| `lib.rs` | Public API, dispatch, polish, the recovery and tail paths, sheet reflections, gauge search, second-order vertex frames |
| `problem.rs` | Input spectra, target branches, and multiplicity classification |
| `spectral.rs` | Spectral matching, sparse and known-spectrum certificates, final verification, and endpoint factors |
| `numerical.rs` | Levenberg–Marquardt refinement, deterministic starts, Horn-wall 3×3 frames, near-wall starts, root refinement, polar orthogonalization |
| `diagnostics.rs` | Route reports, candidate verification, and stage timing |
| `corpus.rs` | Independent checks, candidate comparison, and accuracy and latency reports |
| `algebraic/mod.rs` | Scalar and rank-one formulas, dispatch, frame checks, and shared operations |
| `algebraic/support_strata.rs` | Vertex, edge, face, and radical orientation search |
| `algebraic/klein.rs` | Quaternion section and eigenframe recovery |
| `algebraic/radical.rs` | Repeated-spectrum constructions |
| `algebraic/interior.rs`, `three_givens.rs` | Three-Givens charts and their polynomial equations |

The search tries scalar and rank-one formulas, vertices, edges, faces, the
Klein section, repeated-spectrum formulas, and three-Givens charts. Numerical
recovery handles the remaining cases. The interior search uses sixteen direct
charts, inverse-factor Klein, and transported three-Givens constructions.
QZ root rescue and repeated-root repair protect accuracy near degeneracies.
Each search has a bounded
budget and can decline even when a solution exists.

`witness` has one hot path: the algebraic dispatch, one polish, and a return
when the polished root error is at most `DUAL_TRIGGER = 2e-14`. Everything
else is in two out-of-line functions.

- `recover` runs when the dispatch finds no frame. It tries second-order
  vertex frames (`vertex_quadratic`), then the numerical ladder, then the
  vertex frames again before declining.
- `tail` runs when the polished root error exceeds `DUAL_TRIGGER`. It tries,
  in order, sheet reflections (a quarter turn in the coordinate plane of a
  near-double eigenvalue pair of $D(c)^2$ or $D(g)^2$), the stabilizer gauge
  search (rotations inside each near-degenerate cluster, $SO(k)$ per
  cluster), the transpose dual $(g, c, t)$ chart, and stratum candidates:
  the dispatch rerun without the near-miss rung, spectral retargets, plain
  iteration, Horn-wall split frames, and the numerical ladder. The first
  candidate at or below `DUAL_TRIGGER` returns; otherwise the smallest
  certified root error wins.

The numerical ladder also builds exact Horn-wall 3×3 frames
(`wall_frames`): with one row fixed to a coordinate vector, the remaining
condition is one complex equation, solved over nine two-Givens charts with a
quadratic root isolation. Their frames seed `near_wall` starts for targets
near a wall.

`nalgebra` supplies matrix operations and bounded Schur eigensolves. It is the
only direct dependency.

The spectral acceptance ceiling and polynomial residual limit are `1e-13`;
frame and endpoint checks use `1e-12`. Numerical construction stops at the
spectral ceiling. Final polishing aims for root distances below `1e-14` and
retains useful partial progress. When refinement stalls at a vertex, edge, or
face frame, it restarts along the null space of the root Jacobian. If the
polished frame still misses the target and a routed root $a_ib_j$ lies within
`1e-11` of a target root, the two nearest such routes are refined again with
that root held fixed. It first tries the exact Horn-wall frames, then 2
random starts of the complementary $SO(3)$ block. The numerical ladder's own
split, which has no designated route, uses 8 random starts. This reaches targets on a rank-1 Horn wall, where free refinement
stalls at a singular fiber. Polishing
measures frames with a joint eigenbasis: a basis with a large residual is
finished by Jacobi rotations of the complex matrix. A polished frame replaces
the original only when its error bound is below the original's lower bound,
or below half its error bound when a large residual dominates.

Block rotations use a sine-product formula to avoid cancellation of nearly
equal traces. Numerical refinement solves the damped least-squares system
with QR instead of normal equations and removes accumulated orthogonality
drift with a polar Newton step.

After the polish, a root-matched Levenberg–Marquardt step (`root_lm`) runs
when the verified root error exceeds `1e-14`, and two polar Newton steps
restore orthogonality when the drift exceeds `1e-14`. A stalled step escapes
along the flattest Jacobian direction.

Certification of constructed frames avoids a general eigensolve where the
structure allows. Vertex, edge, and face frames give block-diagonal
sandwiches whose 2×2 blocks are diagonalized by one Jacobi rotation each
(`sparse_state`). Dense constructed frames take eigenvectors from the
adjugates of $(h - \mu_k I)$ at the projected target roots, followed by a
Newton–Schulz step (`targeted_state`). Each falls back to the Schur
eigensolve when its error bound exceeds `2e-15` (sparse) or `5e-15`
(targeted).

Every accepted frame passes `spectral.rs` against the original spectrum.
`Solution` stores the real frame and its verification state together, so
endpoint recovery uses the same eigenbasis without another decomposition.
A successful result always has that state. The README defines how each
function reports a decline.

## Diagnostics

With the `diagnostics` feature, `solve_report` returns `Option<Solution>` with
the real frame, successful `Rung`, and spectral residual. `branch_signature`
classifies the inputs, and `certify_frame` checks a proposed complex frame
through the production acceptance path. Set `PROF=1` and call `prof::dump()`
for stage timings.

## Iterating

The [experiment log](optimization.md) records removed stages, rejected
approaches, and their measurements. `AGENTS.md` gives the rules for changing
the solver.

Changes to factor extraction also need measurements in GULPS, where different
valid endpoint choices can affect later decompositions and synthesis time.
