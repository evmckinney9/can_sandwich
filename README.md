# can_sandwich

**CURRENT RULING (2026-08-14).** The live solver is non-iterative and every
returned frame is forward-certified, but the solver is not yet a complete
realization theorem. Exact spectral signatures own the closed vertex, edge,
face, multiplicity, and routed `1 + 3` strata. Klein, three-Givens, and the
finite cyclic axis orbit are certified algebraic sections used as
accelerators; none is claimed to meet every dense physical fibre.

The remaining generic theorem is action selection. After a projective right
Spin action is fixed, realization is one degree-16 inverse followed by
rational reconstruction and normalization. A fair rational action scheduler
terminates on the regular full-half projection stratum, but no bounded fast
selector is proved. The consistently folded dense rational fixture in the
test suite misses every production section and remains `Unsolved`; its known
fixed action is solved by the degree-16 kernel.

The Spin inverse and unbounded/bounded action-scheduler experiments are gated
behind the `research-spin` Cargo feature. Default builds contain only the live
production realization spine; enable that feature explicitly for Spin census
commands and research benchmarks.

The generic Spin inverse is now reduced to one kernel. Exact finite-field
Macaulay, Singular, and Segre-pencil certificates exclude every possible
dimension of a wholly caustic three-fibre on a nonempty characteristic-zero
Zariski-open family. In the last case, one exact target-dependent projective
action line contains no rank-at-most-two member of the shifted
first-character pencil; projective dimension therefore excludes a
two-dimensional nonintegral-pencil action locus. Consequently every generic
simple fibre contains an off-caustic witness and only the degree-16 full-half
inverse is needed for one-witness realization. The degree-48 mixed inverse is
needed only for a pointwise-complete atlas. This still does not select the
retained action: the theorem proves that the good-action set has interior, not
that a prescribed finite schedule meets it.

The research selector now sends every degree-16 companion through the shared
fallible faer backend. The previous fixed-size nalgebra QR call stalled for
more than 45 seconds on Haar row 20946. The replacement solves that row in
the first two spread actions and makes the eight-action 83-row tail census
terminate. On the known-action benchmark the robust kernel measures 62.98 us
average / 380.62 us slowest over 1,001 runs; the old backend measured
38.12/301.51 us but had no usable worst-case bound. Across the 83 current
Haar AxisQuartic rows, an eight-action prefix solves 54, averaging 2.774 ms
with a 14.259 ms slowest row. This is evidence against promoting the spread
prefix as a production heuristic: the existing axis section is about 104 us
per owned row. The fixed-action algebra is ready; action selection is not.

From-scratch depth-2 can-sandwich solver and the isolated realization engine used
by GULPS. It retains its own nested workspace and depends directly on `nalgebra`,
`faer`, and `dyn-stack`; `gulps-core` depends on this black-box API, while this
crate has no dependency back onto core or any frontend type.

The default library exports only `solve`, `Solution`, `Rung`, and `Mat4`.
Corpus controls, profiler counters, partial-rung entry points, and diagnostic
binaries require the `diagnostics` feature; their instrumentation is compiled
out of the ordinary solver.

Solves the isolated subproblem only: given monodromy coords `(C, G, T)`, find the local
`u` (a real frame `O ∈ SO(4)` in the magic basis) with `weyl(Can(C)·u·Can(G)) =
weyl(T)`, where `Can` first converts each monodromy triple to Weyl coordinates. No
invariants/select/witness/transport -- the problem is assumed already
split into a depth-2 subproblem. Runs the whole `feasible_linspace` triple dataset
end-to-end and reports per-stratum coverage/residual.

## Master object
`M = D_C·O·Λ·Oᵀ·D_C`, the sandwich Makhlin matrix in the magic basis: `D_C =
mb(Can(C))`, `Λ = mb(Can(G))²` (both diagonal), `O = mb(u) ∈ SO(4)`. Matching and
construction uses the symmetric functions `e₁..e₄` of `M` via Newton's identities
on traces, so internal routing does not floor at spectrum degeneracy. The public
compiler boundary additionally root-checks nondegenerate accepted spectra; exactly
repeated targets retain the machine-scale polynomial certificate rather than invoking
an ill-conditioned QR eigensolve. This is the orthostochastic / real
multiplicative-Horn image; reachability is its containment, construction is the
realification.

## Current production algorithm

Although `O in SO(4)` has six parameters, the target spectrum contributes
only three independent equations (the determinant is fixed).  The solver
therefore uses three-dimensional algebraic sections of the `SO(4)` orbit,
chosen by spectral stratum:

1. Build the routed-root support mask, then test only admitted
   signed-permutation vertices.
2. Solve one-Givens edges linearly in `cos(2 theta)`.
3. Solve the input `(3,1)` rank-one secular case when its exact multiplicity
   signature is present.
4. Solve two-disjoint-Givens faces as two independent 2-by-2 blocks.
5. On the remaining repeated-spectrum strata, apply the rank-one/rank-two
   characteristic formulas. These reduce a symmetric unitary to
   `c I + sum rho v v^T`; generic rows never enter this recursion. The 36 even
   skeletons and 28 odd pin pairs exhaust the hyperelliptic characteristics;
   the former split-quartic tier is only an alternate boundary representation
   and has no ownership after vertex/edge/face decline. Intermediate-root
   deflation and first-peel residues are built once per characteristic, then
   shared by all sign-class cofactors. The strict cyclic T4 word is likewise
   built once per two-step problem and reused across its pin pairs.
   On a `(2,2)` gate the two nominal rank-one peel orders are exactly
   identical (`d1=d2`), so production evaluates one representative rather
   than the same characteristic fiber twice.
   Candidate enumeration is certificate-aware: an internally valid spectral
   candidate that fails production frame reconstruction no longer masks the
   remaining characteristics. The enumerator resumes through a caller-supplied
   acceptance function.
   Residue columns are accepted only after a complete four-vector Gram check.
   If that cheap reconstruction is ill-conditioned, production strips the
   known outer diagonal from the already-constructed symmetric sandwich and
   recovers its real orthonormal eigenframe. This is a second reconstruction of
   the same algebraic candidate, not another realization branch.
   First-peel admissibility is evaluated in closed form before any residue
   is built: for a pin-pair candidate `mu = {t1, t2, m, pp/m}` every
   first-peel residue is linear in `x = cos(arg(m) - arg(pp)/2)`, so the
   admissible set is one `x`-interval per pair (`PairGate`). Pairs with an
   empty interval skip the closure rooting entirely; rejected candidates
   cost two comparisons. Margins are three orders wider than the exact
   gates and every uncertain case falls through to the exact evaluation,
   so the accept set is unchanged (shadow-validated at zero disagreements
   over 6.83M candidates on both corpora).
6. When one routed eigenvalue splits off, first test the nine zero-entry walls of
   the residual `SO(3)` block. On each wall the one complex trace equation is
   bilinear, so realization is one quadratic and linear back-substitution. The
   walls are a fast peel, not a completeness claim.
7. Write a spanning-tree chart with three Givens
   rotations. The atlas contains all 96 ordered spanning-tree words (16
   trees times six noncommuting edge orders), with 16 measured hot words
   first. Quotienting by the 48 automorphisms of the parameter cube reduces
   the raw 2,304 `(word, permutation)` descriptions to 216 distinct spectral
   image charts. With `x,y,z = cos^2(theta)`, the first three spectral
   invariants are multi-affine.  For fixed `x`, the three equations are a
   `3x4` affine matrix on the Segre coordinates `[1,y,z,yz]`; its four signed
   cubic cofactors `n_j(x)` give the genuine sextic directly as
   `n0*n3-n1*n2`.  A Bernstein-sign certificate rejects rootless sextics.
   At each retained root, the best-scaled pair of the three original planes
   gives one quadratic in `y`; the strongest row then gives `z` linearly.
   This avoids the old prebuilt biquadratics and fixed recovery pivot.
8. Evaluate the same axis formula in the three cyclic target gauges, ordered
   by the denominators that occur in the exact carrier. These gauges are
   conditioning variants of one construction, not separate coverage branches;
   every transported answer is certified in the original sandwich.
9. Solve the residual image over the 24 row/zero-pair label orbit. This is a
   finite certified peel, not a complete atlas. In each label, the six mixed
   brackets descend to binary-quartic
   invariants `I4,J6`, giving the direct decic
   `T10(u)=4 I4(u)^3-J6(u)^2`. A cubic or degree-drop quadratic Cauer lift,
   linear recovery, and the Heron-kernel frame finish reconstruction. The old
   degree-30 pullback, sampled degree-48 interpolation, fixed-section bank,
   and high-degree certified-isolation path are absent from production.
   Before constructing a chart, the first compound applies an exact support
   gate: the two forced zero entries put `O⊙O` on a 12-vertex face of the
   Birkhoff polytope, so a target trace outside that face cannot occur in the
   chart. This is a necessary convex-hull certificate, not a learned chart
   order or corpus heuristic.  Each zero-weight boundary forces one further
   matrix entry to vanish.  Its orthostochastic image therefore lies on an
   eight-vertex subface; the same trace certificate rejects an empty boundary
   family before constructing or certifying its sextic resultant.
   The
   older degree-12 four-Givens cells, the
   rank-structured degenerate-target tier, and the mid-cascade rank-2/pair22
   emitters were deleted 2026-07-15 under the dataflow gate: zero entries on
   both complete corpora, measured (the live input-(3,1) secular stays).
10. The research Spin tail fixes one projective action and solves the remaining
    `(2,2,4)` intersection with a degree-16 quotient-algebra kernel. The kernel
    is exact, but the bounded action-selection contract is open, so this tail
    is not production coverage.
11. Certify the complete Gram matrix and orientation of every emitted frame,
   then forward-certify the elementary symmetric functions of the original
   `M`.

There are no random starts, generic numerical optimization, or production
search fallbacks. The locked linspace and independent continuous Haar corpora
are empirically complete by default; corpus coverage is not a universal
coverage theorem.

After canonical preparation, the shared frame certificate, deterministic
production dispatch, and the direct decic axis selector, the 2026-08-14
release snapshot solves 761,308/761,308 linspace rows at 3.54 us/triple and
300,000/300,000 Haar rows at 3.81 us/triple. The one-pass maxima were 710.46
us and 238.41 us respectively, but the linspace maximum was scheduler noise:
its row measured 4.98 us average / 78.01 us slowest over 1,001 repetitions.
Across repeated checks of the largest Haar axis rows, the stable worst row
measured 131.08 us average / 299.25 us slowest (125.67 us median) over 301
repetitions. The dense rational miss declines at 38.85 ms average / 48.32 ms
slowest over 31 repetitions and remains `Unsolved`; it is still the dominant
unresolved tail.

Default library results depend only on `(C,G,T)`. Atlas kill switches and
chart-enumeration controls are not part of production behavior; offline atlas
controls require the explicit `diagnostics` feature.

The rungs are sections of one spectral map, not unrelated solvers. With the
determinant fixed, the target has three real invariant coordinates. A
`k`-Givens section therefore has expected fiber dimension `k-3`: vertices,
edges, and faces are boundary restrictions; three Givens gives isolated
degree-six points; four Givens gives a one-dimensional fiber whose projection
folds where a three-Givens boundary section stops meeting it. Repeated spectra
are the confluent limit of the same map, where Cauchy--Binet minors collapse to
rank-one/rank-two secular residues. This explains both dispatch rules:
algebraic multiplicity owns the radical rung, while ordinary near-degeneracy
stays in the generic chart/fiber algebra.

## Triangle reanchoring (2026-07-14)

The radical solver is now closed under the two inverse-sandwich reanchorings,
not only direct and role-swapped orientation.  If the reanchored radical solve
returns

```text
X = D_g V W^-1 V^T D_g = S C^-1 S^T,
```

then inversion gives `D_g S C S^T D_g = V W V^T`; this is the role-swapped
original sandwich, so the required frame is `O=S^T`. The other reanchor

```text
X = D_c V W^-1 V^T D_c = S G^-1 S^T
```

returns `O=S` directly.  Since `X` is symmetric unitary, `S` comes from one
real 4-by-4 symmetric eigendecomposition and fixed-eigenvalue column matching.
Every candidate is re-gated against the original master object.

The same transport also lets bounded three-Givens eliminants run from the
other two vertices of the spectral triangle. These are cyclic boundary charts,
not a separate realization ansatz. On `feasible_linspace`, the old
13-cell 4-Givens sliver cover reduced to two production cells. Those cells
remain in the constructive default and own 27 Haar rows, including all 18
direct three-Givens holes. The earlier search-shaped slow tail
(`pencil`/RANK9, alternate ladder, `cl3`, `peel3`) and bench probes remain
preserved in `src/attic/`.

## Four-Givens algebraic baseline

The following numbers and analysis document the current constructive baseline
that the axis-cubic/quartic reduction must replace.

Complete locked-corpus result (2026-07-14 speed campaign; built with
`target-cpu=native` via `.cargo/config.toml` -- build on the machine that runs):

```text
feasible_linspace stride 1, N=761,308: 761,308/761,308 (100.00%),
    13.5 us/triple, p50 9.3 us, p99 112 us, worst residual 9.11e-10,
    759,415/761,308 (99.75%) below 1e-12
feasible_haar stride 1, N=300,000: 300,000/300,000 (100.0000%),
    15.8 us/triple, p50 10.9 us, p99 121 us, max steady median 1.95 ms
    (row 142340, cell-decline-bound plus reanchor), Sliver-rung mean 0.98 ms,
    worst accepted residual 9.997e-10,
    Interior 297,331 / Sliver 27 / Reanchor 2,642 / Unsolved 0
```
(Corpus maxima above ~3 ms in raw runs are WSL scheduler artifacts; verified
rows run at their steady medians. The four-Givens cell now uses the cascade's
first-certified-exit convention per gamma node -- an ordinary-sheet hit no
longer pays the remaining node eigensolves; decline rows are bit-identical.)

A direct 2026-07-14 profile corrected the earlier tail attribution: row
142340 does enter the 17-node cells, but the companion calls degenerate mostly
to degrees 2--6 and account for only about 0.13 ms.  The dominant cost is up
to five failed fold seeds, each rebuilding the eliminant at `gamma` and
`gamma+h` for as many as 16 Newton steps.  The raw eliminant expression used
hundreds of tiny `Vec` allocations per build.  Its production twin now uses
proved fixed capacities (z degree <=16, eliminated-y degree <=4), while the
old expression remains executable under `ELIM_REFERENCE=1` as an oracle.
`ELIM_FAST_CHECK=1` compared every coefficient on both complete corpora;
coverage, rung counts, residuals, and machine-precision counts are unchanged.
On isolated A/B runs this cuts the 27-row Sliver mean 1.22 -> 0.98 ms and the
named hard-row medians by 21--27%.  This is a genuine constant-factor win, not
the orders-of-magnitude construction still required by the latency bar.

The four-Givens eliminant still uses its exact `E(z)^2` deflation.  The older
three-Givens identity `detx = Res_y(p1,q1) x S6` remains a test oracle, but its
resultant and numerical division have left production.  The direct Segre
cofactor formula constructs the same SEXTIC symmetrically from the original
three trilinears; it is faster and better conditioned.  The interior rung
Bernstein-excludes and isolates roots on that direct sextic.
A proven hull gate (multilinear maps send the cube into the convex hull of
the 8 corner images; box test expanded by ACCEPT + model slack) skips
provably unreachable chart classes before elimination. `init_tables()`
forces the lazy quotient tables so corpus maxima measure the solver, not
the first interior row's table build (the former linspace max was exactly
that artifact).

Certified-separation gates (Frank-Wolfe direction search whose EXCLUSIONS are
verified by 8/16 dot products, so soundness never depends on the search)
now run at both the chart-class level (fires on 54-100% of classes on hard
rows, measured by LP at fixtures) and the cell level (hull of 16 corners
inflated by max|K|). A real-axis-scan replacement for the cell's complex
companion eigensolve was tried and REVERTED: it lost the recorded
rounding-fragile row 257245 -- the complex eig is load-bearing for the
near-fold class, confirming the ledger. 

At that historical point the remaining max was representation-bound: worst rows walked O(hundreds) of
quotient classes at the certified deg-6 floor per class, and the 27 haar
sliver rows pay the 17-node four-Givens cell plus repeated fold collisions. An
orders-of-magnitude max reduction requires an atlas-free construction, not
another constant-factor chart kernel change.  Importantly, the handoff's
claimed "essential degree 2--3 per sheet" is not yet such a construction: no
coefficient table or script recording it exists.  The persisted exact s104
fixed-nu RURs have degree 16; `s107_minpoly_factor_audit.py` factors all 60
modular outputs and finds high-degree components at every sampled good prime,
including `1+15` twice.  A low-degree selector therefore needs additional
proved sheet-label equations; it does not follow from the stored fixed-nu
algebra or from the floating band-limit observation.

The campaign took haar from 54.9 to 19.6 us/triple (2.8x) and linspace from
18.6 to 13.2 us/triple (1.4x) with rung counts and coverage unchanged (one
haar row moved Interior -> Reanchor at the acceptance boundary). Everything
was structural -- no tolerance was touched and no parameter was added:

- `target-cpu=native` build (the baseline x86-64 build had no FMA/AVX).
- Two PROVEN spectral gates. Edge: an edge frame leaves two diagonal entries
  `mu = d^2_k lambda'_k` untouched and those are then exact eigenvalues of M;
  if `mu` is delta-far from every target eigenvalue then `|p_w(mu)| =
  prod_j |mu - w_j| >= delta^4` while `|p_w(mu)| <= 4 max_k |Delta e_k|`, so
  with delta = 1e-2 the smooth residual exceeds 2.5e-9 > ACCEPT and the
  (perm, plane) pair is skipped losslessly. Face: the theta-free block-det
  test was already exact; it now runs FIRST on precomputed unit-modulus
  products (norm_sqr, no sqrt) with the trace algebra built only on a det
  match. Face decline: 18.4 -> 1.6 us/row; edge decline: 4.8 -> 0.6 us/row.
- `dphase` built directly as `diag(exp(i*eigphases))` (the recorded closed
  form) instead of conjugating the full canonical matrix.
- Interior root isolation: Bernstein exclusion via a Pascal-triangle table
  (the binomial-quotient form cost ~250 divisions/call), batch Horner over
  the 64-point grid, and bracketed Illinois refinement (keeps bisection's
  bracket certificate, ~5x fewer evaluations).
- `two_step` / `free_pair_roots` enumeration on fixed-size arrays (no
  per-pair heap allocation), enumeration order preserved exactly.
- `PROF=1` env-gated per-stage instrumentation (zero cost when off) prints
  the profile map after `bench-npy`; `bench_interior_kernel` is an ignored
  microbenchmark for the interior kernel (WSL wall-clock noise swamps
  stride benches at the 10% level).

Remaining profile map (linspace: `two_step` enumeration ~9 us/radical-row,
dominated by 36 mirror-completion calls each -- the recorded
symmetric-evaluation route is the math lever; haar: interior chart tries
~14 us/row -- the f=3 eliminant is the math lever; then rank_perms ~2.5 us
and prelude ~2.8 us per row).

The historical Interior/Sliver/Reanchor attribution could move by a few rows
at the acceptance boundary, while coverage remained complete.

The radical dispatcher uses machine-scale equality to decide ownership.
Previously its internal conditioning radius (`2e-3`) leaked into dispatch:
a generic Haar row with target gap `1.863e-3` paid twelve doomed radical
constructions and took ~5.3 ms steady. Separating exact ownership from internal
confluent conditioning reduces that row to ~0.56 ms without changing either
corpus's coverage or rung counts.

On the identical stride-400 A/B sample, runtime fell from 1,710 to 20.7
us/triple (82.6x) with coverage unchanged at 1,904/1,904.  Former defect rows
268000, 134000, and 170000 now finish in 94, 5, and 27 us respectively.

## Files
- `data.rs` -- minimal `.npy` reader (no dep).
- `can_sandwich.rs` -- the algorithm. Primitives (magic basis, `Can`, `D_C`, `symfn`,
  `M`, smooth verify) DONE + convention-validated against the research rig (`D_C` and
  target spectrum match to 1e-6). Construction rungs converge in here.
- `secular.rs` -- proved support router plus edge/face and exact-confluence entry.
- `klein.rs` -- one-sided quaternion section and its real quartic.
- `main.rs` -- dataset driver.

## Historical pre-reanchor status
Before the transport above, the standalone atlas cascade reached **95.98% solved**, of which **94% machine-precise** (<1e-12); the tail was near-
degenerate where the verify metric itself floors (~1e-9). Every accepted frame is genuine SO(4)
(worst `|det(O)−1|` = 6.7e-16, worst residual 9.6e-8). **No GN, no LM, no Newton polish.**
- [x] vertex (label 3): signed-perm routing -- **100%**, scalar eigenphase (no matmul).
- [x] edge (label 2): single 2×2 block, `e₁` affine in `cos 2θ` -> linear solve -- **100%**.
- [x] face (label 1): two DISJOINT Givens blocks (no cross-term) -> two independent linear solves,
  machine-precise, no eig. Solves 25% of label-1 directly AND re-solves the near-face interior tail
  (the cells the deg-6 chart handled ill-conditioned) -- lifted machine-precise 93.7%->98.7%.
- [x] interior (label 0): deg-6 multilinear chart; `z` linear-elim, resultant in `y`, **real `[0,1]`
  roots via bracketing+bisection (faer-free hot path)**, exact-vs-scan gated -- **99%**.
- [~] sliver (label -1): **79%** (ndist=4 88% / ndist=3 75% / ndist=2 66%) via the 4-Givens peel over
  a **13-cell cover**. The gauge locate is closed-form: the exact-degenerate fold is the tangent of
  the e3-invariant mismatch `h(γ)=Re(tr3−W3)`, an EXACT degree-8 polynomial, so `γ*` is a root of
  `h'(γ)` -- one Chebyshev fit + one companion-eig. **No Newton, no iterative parabola.** The `'sin2'`-
  gauge fold is role-parametrized (`ROLE_FAC['sin2']=1−γ²`) and ndist=2 corners route to a deg-2
  `Y=0` face-solve (e1,e2 bilinear, no scan).
  The dominant lever was **cover completeness**, not construction: on the hard reachable-sliver set,
  78% of the misses were cells the original 6-cell cover simply lacked. Set-cover over the full
  15-word x 24-perm space against the arbiter-reachable slivers picked **7 high-marginal cells** to add
  (94 -> 235 of 257 solvable; the long +1/+2 tail is sample-specific and not added). That lifted the
  sliver rate 34% -> 79% and overall to 95.98%.
  TWO OPEN levers remain, both real (neither is edge-case patching): (1) **perf** -- with 13 cells each
  running the full `coarse_fold_seeds` (NG degree-16 companion-eigs), the sliver path is ~140 ms/triple;
  the per-cell fold cost is now the bottleneck. (2) **construction residue** -- ~14% of hard-set slivers
  are solvable by NO cell (the arbiter reaches them only by descent); that is the genuine no-GN gap,
  where a single-branch continuation through the fold (z-pair/y-root continuity, so the deg-8 fit never
  straddles a branch-min kink) is the only route. The `'sin2'` gauge-boundary sub-case (a_g≈±1, the
  ignored `sin2_fold_solves_k3863` test) is one instance.

Faer is a real dependency (the sliver eliminant + the gauge companion-eig); the other rungs are
faer-free (real-root bracketing). The hot path is scalar corners + `symfn` verify -- no eig outside
the documented companion-eigs, no GN, no Newton polish (Qiskit-clean). The per-cell sliver fold is the
remaining perf hot spot; precision and the closed-form gauge math are done.

## gulps baseline (current production, same dataset)
What current gulps scores on the IDENTICAL stride-40 sample (19,033 triples), via its
real per-segment cascade (`chain::lift_one`: closed-form `strata::solve` rungs, then the
GN backstop). The apples-to-apples column is **GN-FREE (Exact CF)** vs this crate's
GN-free rungs: gulps' closed form covers **71.3%** GN-free (vertex/edge 100%, interior
67%, face 53%, sliver 0%) and only reaches 99.83% SOLVED by routing the other 28.7% to
the Gauss-Newton backstop. This crate (no GN) is at **89.3%** on the same sample, ahead
on interior (+32pp), face (+25pp), sliver (+21pp). One-shot measurement (2026-06-25); the
driver was reverted so gulps stays intact, the numbers are the deliverable:

```
gulps production cascade (chain::solve_segment) on feasible_linspace
loaded 761308 triples (monodromy [C,G,T]); stride 40

OVERALL: 19000/19033 = 99.83% solved; GN-FREE (Exact CF) 13577/19033 = 71.33%; worst residual = 5.8e-9; machine-precise (<1e-12): 16323/19000 = 85.91%; 105.1 us/triple

labels: 3 vertex  2 edge  1 face  0 Givens-interior  -1 sliver
  label      n  solved   GN-free   warm   miss   worst-res  us/triple
    -1    1743   1730 ( 99%)     0 (  0%)     0  1743     3.7e-9    401.60
     0    7272   7270 (100%)  4861 ( 67%)     0  2411     4.5e-9     59.72
     1    2742   2724 ( 99%)  1440 ( 53%)     0  1302     5.8e-9    308.18
     2    5647   5647 (100%)  5647 (100%)     0     0    5.3e-15      1.93
     3    1629   1629 (100%)  1629 (100%)     0     0    4.4e-15      1.76
```
(`warm` = GN warm-started from a CF near-miss; it never fired here, every gulps CF miss
went straight to cold GN. gulps' `machine-precise` floors at ~1e-9 near spectrum
degeneracy where the Makhlin metric over-reports; this crate's symfn-no-eig verify does
not, hence its 98.0% vs gulps' 85.9%.)

## Build / run
`cd crates/can_sandwich && cargo run --release --features diagnostics`
(or `cargo test --release`).
