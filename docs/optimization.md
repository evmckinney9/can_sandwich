# Corpus-driven simplification, 2026-09-23

Baseline: `b3dbb7fed701257fab7a7b0a3e8b9da9a6247ed4`.
Corpus: 1,093,691 rows, SHA-256
`75508dd43fb7e462c9af3e59e070fb6f2defb06d8a344f5a1a7f1691850df5cd`.
Compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`.

## First-pass experiments

Each removal retained numerical recovery and the production spectral verifier.
The comparison runner checked returned matrices independently at the existing
`1e-8` threshold. The corpus and checker equations were not changed.

| Experiment | Coverage | Timing observation | Decision |
|---|---|---|---|
| Remove row-in-plane chart fallback and its precision helper | Full corpus passes | 8.58 → 7.89 s in the paired run | Keep deletion |
| Also omit the first resonance search; retain confluence dispatch's fallback | Full corpus passes | 9.63 → 6.74 s | Keep deletion |
| Omit separate 1+3 wall and dense selectors | Full corpus passes | 9.40 → 9.21 s; small difference | Keep deletion as part of the combined change |
| Omit three-Givens boundary accelerators | Full corpus passes | 9.20 → 12.10 s | Retain these accelerators |
| Replace confluence and interior machinery with row-in-plane charts | 20,636-row sample passes | 0.164 → 2.638 s | Reject slowdown |
| Keep support/Klein formulas, use numerical recovery for everything else | 10 failures in the same sample | 0.152 → 0.393 s | Reject |
| Omit confluence dispatch while keeping the early resonance search | 2 failures in the same sample | 0.178 → 0.193 s | Reject |

The sample selected every 53rd corpus row. It was used only to screen costly
alternatives; accepted solver changes were checked on the full corpus.
Individual experiments used the diagnostic build and are not additive timing
predictions. The final comparison below measures the combined implementation.

## First-pass implementation

The retained search uses scalar and rank-one formulas, support strata, Klein,
confluence formulas, and three-Givens charts, then bounded numerical recovery.
The separate 1+3 and row-in-plane implementations were deleted. Resonance is
searched through the confluence dispatcher once, without a second competing
candidate-selection policy in the outer dispatcher.

Successful solutions now store a real frame and a required verification state.
The old identity/`Unsolved` diagnostic sentinel and optional verification state
were removed. Numerical output goes directly through the shared spectral
verifier, and endpoint extraction consumes the same state. Historical waypoint
and paired-edge experiment APIs and their private endpoint code were retired;
the optional diagnostics retain route reports, classification, candidate
verification, table warmup, and stage counters.

The unused direct `dyn-stack` dependency was removed; it remains transitive
through the numerical library. The native comparison runner now reports
mean, median, p95, p99, p99.9, and worst latency, including the worst row's index.
Its summaries were independently recomputed from the per-call CSV.

## First-pass measurements

The source shrank from 18 Rust files / 15,717 lines to 15 files / 10,363 lines:
5,354 lines removed (34.1%). Givens rotations now update their two affected
rows directly instead of using dense complex matrix products. Their isolated
timing effect was small; the measurements below cover the whole change.

Three separate paired runs used release builds with fat LTO, one codegen
unit, and only the `corpus` feature. The machine was an AMD Ryzen 5 5600X;
the executable was pinned to CPU 2. Both solver versions ran in one process,
with the runner's usual warmup and alternating call order. The baseline used
the same reporting changes; its solver and the independent checker were
unchanged. Each cell below is the range across the three runs.

| Metric | Baseline | Revised solver |
|---|---|---|
| Total solver time (s) | 8.224–8.260 | 6.110–6.130 |
| Mean (µs) | 7.520–7.552 | 5.586–5.605 |
| Median (µs) | 2.580–2.590 | 2.540–2.560 |
| p95 (µs) | 24.580–24.720 | 14.910–15.000 |
| p99 (µs) | 73.080–73.699 | 33.510–33.790 |
| p99.9 (µs) | 168.500–169.990 | 120.770–121.690 |
| Maximum (ms) | 176.201–182.188 | 20.915–21.585 |

The worst baseline row was 24 in every run; the revised solver's was
1,086,413. Total solver time fell by 25.7–26.0%. Median latency was close to
unchanged; the larger gains were at p95, p99, and the worst cases. These are
measurements across the corpus, not worst-case execution-time bounds.

All 1,093,691 cases passed for both solvers in every run. Spectral error p99
was `8.204e-13` before and `1.235e-12` after; maximum error was `7.989e-9`
before and `7.993e-9` after. Every returned frame met the unchanged `1e-8`
checker threshold. 49,583 rows had smaller spectral error, 1,019,109 had equal
error, and 24,999 had larger error.

The external comparison executable used this entry point, with the original
source snapshot renamed to `can_sandwich_baseline` in its manifest:

```rust
fn main() -> std::process::ExitCode {
    can_sandwich_baseline::corpus::compare(can_sandwich::solve)
}
```

It was run as `taskset -c 2 <comparison-executable> --corpus tests/cases.bin`.
Use `--report result.csv` to retain per-call measurements. The native runner's
mean, median, percentiles, maximum, and worst row were independently checked
against CSV output for the full corpus, an even-sized sample, and a singleton.

`make test`, `make lint`, and `make build` passed. The corpus test also
verified endpoint factors by multiplying each decomposition back together.

## Second pass

The next pass froze the first pass (`8db86e1`) as a separate comparison
dependency. Its
route census attributed about 2.17 seconds to rows solved by radical formulas
and 0.90 seconds to the 1,690 rows that reached numerical recovery. These
figures include work before the winning route; they are not isolated stage
costs.

| Experiment | Evidence | Decision |
|---|---|---|
| Remove the remaining resonance search and held-candidate policy | Full corpus passes; about the same total time | Delete |
| Remove the separate two-pair solver | Full corpus passes alone and with the preceding deletions; about the same total time | Delete |
| Stop radical search after the shallow pass | Sample passes, but sends more work to slower fallbacks | Retain the complete pass |
| Try near-commuting starts before random starts | Sample solver time 0.113 → 0.119 s; sample maximum 5.25 → 8.25 ms | Reject |
| Omit numerical vertex/edge checks | Sample passes, but total and worst time increase | Retain |
| Alternate fixed target signs in the first restart sweep | All 1,690 numerical cases pass; less useful than shorter searches | Retain existing sign order |
| Refine rejected algebraic candidates before continuing dispatch | All 1,690 numerical cases pass; little change in total time | Reject added work |
| Limit each numerical start to 40 iterations and each orientation to four starts | Full corpus passes with the module deletions; worst time 22.7 → 3.9 ms in the screening run | Keep |
| Remove radical construction's duplicate matrix-power and eigenvalue checks | Full corpus passes through the original final verifier | Delete |
| Compute phases directly from monodromy coordinates and use real matrix products in spectral projection | Combined full corpus passes | Keep |

The numerical-only screening set contains every row that reached numerical
recovery in the frozen first pass. It does not replace full-corpus validation:
removing algebraic paths can change which rows reach that recovery.

The second pass deletes `resonance.rs` and `two_plus_two.rs`, their dispatch
branches, and unused certificate wrappers. Radical candidates now contain
peel vectors directly, without an optional frame or a second result wrapper.
Their symmetric matrix is built only when orientation transport or eigenframe
recovery needs it. Removed construction checks had repeated work performed by
the original-problem frame and spectral checks; final verification and corpus
tolerances are unchanged.

The search still has finite budgets. Switching numerical parameterizations
sooner improves the measured tail, but neither these budgets nor the finite
corpus establish completeness outside the tested cases. The discarded
algebraic implementations remain in the original baseline commit.

Source size after both passes: 13 Rust files and 7,727 lines. The second pass
removes another 2,636 lines; the total reduction is 7,990 lines (50.8%).

### Second-pass release measurements

Three paired runs compared the frozen first pass with the final second pass.
They used the same release settings and CPU affinity as above, with diagnostics
disabled. Each cell is the range across those three runs; the percentage
change uses matched runs, not opposite ends of the ranges.

| Metric | First pass | Second pass |
|---|---|---|
| Total solver time (s) | 6.480–6.913 | 5.785–6.223 |
| Mean (µs) | 5.925–6.321 | 5.289–5.690 |
| Median (µs) | 2.730–2.910 | 2.670–2.830 |
| p95 (µs) | 15.890–17.610 | 14.890–16.470 |
| p99 (µs) | 35.700–39.080 | 34.250–37.290 |
| p99.9 (µs) | 129.690–142.190 | 115.330–128.620 |
| Maximum (ms) | 21.250–27.649 | 3.695–8.528 |

Total solver time fell by another 10.0–10.7%. All rows passed in every run.
Spectral error p99 changed from `1.235e-12` to `9.783e-13`, and maximum error
from `7.993e-9` to `7.978e-9`. Relative to the first pass, 133,436 rows had
smaller spectral error, 823,736 had equal error, and 136,519 had larger error.
The changes do not improve every individual witness.

One additional paired run against the original `b3dbb7f` snapshot measured
9.054 → 6.014 seconds (33.6% less total time), p99 of 81.310 → 35.820 µs,
and maximum of 182.173 → 10.117 ms. Both versions passed all rows. This is
one run, separate from the three-run table above.

The observed maxima include isolated timing spikes. Row 912,829 measured
8.528 ms in one full run; 30 subsequent calls had median 5.77 µs and maximum
14.06 µs. Row 1,083,563 measured 10.117 ms in the original-baseline comparison;
its replay median was 28.80 µs and maximum 40.74 µs. Row 1,083,144 was the
slowest row in the other two final runs; its replay median was 3.975 ms.
Keep the observed maxima in reports: these wall-clock measurements include
variation beyond deterministic solver work.

`make test` passed all five tests, including the full corpus's independent
spectral checks and endpoint reconstruction. `make lint` passed. The final
comparison executable built without warnings. The corpus, checker, and
acceptance thresholds were unchanged.

## Accuracy correction after the third pass

Baseline: `4492fad`. The initial third-pass deletion experiment passed the old
`1e-8` corpus bound and reduced total time by 6.2–6.6%, but spectral error p99
rose from `9.783e-13` to `2.270e-12`. That tradeoff was rejected. Passing a
loose threshold did not establish that an accuracy mechanism was unnecessary.

The final implementation retains QZ root rescue, admissible-component root
isolation, transported three-Givens charts, and repeated-root repair. It keeps
the independent cleanup: static chart schedules, consolidated frame checks,
reused projected eigenmatrices, and the nalgebra companion fallback. Removing
faer leaves nalgebra as the sole direct dependency and reduces the lockfile
from 84 packages to 25, without package upgrades.

Acceptance limits are now stricter:

| Check | Before | Current |
|---|---|---|
| Spectral roots | `8e-9` | `1e-12` |
| Numerical acceptance | `4e-9` | `1e-13` |
| Polynomial residual | `1e-9` | `1e-13` |
| Final frame checks | `1e-11` | `1e-12` |
| Endpoint reconstruction | `8e-9` | `1e-12` |
| Independent corpus checker | `1e-8` | `1e-12` |

Tightening exposed cancellation in the one- and two-block rotation formulas.
They now use products of sines instead of differences of nearly equal traces.
The two-block search verifies each candidate before leaving its chart scan.
Numerical refinement uses QR on the damped least-squares system, avoiding the
squared condition number of normal equations, and a polar Newton step removes
rounding drift from successive rotations. Constructed frames can be refined
before the strict spectral check rejects them. A routed target root can also
be held fixed while refining its complementary SO(3) block. The inactive
rotation increments are explicitly zeroed after the linear solve, so roundoff
cannot move the boundary root. Numerical budgets are 120 iterations and up
to 24 starts per orientation; the earlier smaller budgets were not retained
at the tighter tolerances.

The checker still computes complex Schur eigenvalues independently of the
solver. Shifting and scaling its matrix fixes convergence near scalar spectra;
a quarter-turn gives a second QR path without changing the spectrum. The
corpus now also reports p99.9 spectral errors and counts above `1e-13`,
`1e-12`, and `1e-10`.

Four original regression rows (1–4) violate the rank-two Horn inequalities
already present in `tests/generate.py`. Their phase-inequality violations are
`4.323e-10`, `1.844e-10`, `2.803e-10`, and `1.100e-10`, respectively. Their
old approximate witnesses cannot satisfy the stricter contract. The grader
now applies that necessary inequality to inputs in the ordered alcove and
requires rejection when it certifies infeasibility. It uses no row lookup;
the corpus bytes are unchanged. The numerical fallback rejects a certified
violation before spending its restart budget. The generator’s feasibility
slack is now `1e-13`, reduced from `1e-7`; the corpus was not regenerated.
See the [grader contract](researcher.md).

Rejected experiments included replacing QZ rescue with local polynomial
Newton steps, removing repeated-root repair, and unstructured perturbations
of accepted frames. Increasing search budgets alone did not fix the last
boundary failures. Removing the spectral-projection regularization also
reduced coverage and was discarded.

### Strict release measurements

Three paired runs used release mode, fat LTO, one codegen unit, the `corpus`
feature, and CPU 2 affinity, with diagnostics disabled and no concurrent
builds or tests. Both solvers were graded with the updated independent checker.

| Metric | Committed `4492fad` | Strict candidate |
|---|---|---|
| Total solver time (s) | 5.293–5.312 | 7.416–7.443 |
| Mean (µs) | 4.839–4.857 | 6.781–6.805 |
| Median (µs) | 2.470–2.490 | 2.700–2.720 |
| p95 (µs) | 13.210–13.220 | 19.030–19.130 |
| p99 (µs) | 30.400–30.440 | 40.130–40.230 |
| p99.9 (µs) | 109.379–109.830 | 383.529–388.909 |
| Maximum (ms) | 3.660–3.744 | 18.620–19.027 |

The stronger contract costs 40.1–40.4% more total solver time in these runs.
The slowest candidate row was 1,075,377 in all three runs. These costs are
recorded rather than traded for larger accepted errors.

The candidate passes every row: 1,093,687 verified witnesses and four certified
infeasible rejections. Spectral error p50 is `1.047e-15`, p99 `4.591e-14`,
p99.9 `9.170e-14`, and maximum `9.762e-13`. There are 645 errors above
`1e-13`, and none above `1e-12`. Maximum orthogonality defect is `1.334e-13`;
maximum determinant error is `1.041e-13`.

Under this stricter grader, the committed solver fails 10,957 rows. Its
spectral p99 is `1.004e-12`. Its reported maximum is `inf` because the stricter
frame checks skip spectral evaluation for some matrices; the earlier checker
measured a maximum spectral error of `7.978e-9`. The comparisons count 161,821
smaller, 760,387 equal, and 171,033 larger finite spectral errors. Small
roundoff differences remain; the new hard error bound applies to every witness.

`make test` passed all five tests, including every corpus row and endpoint
reconstruction. `make lint` passed. The generator passed a syntax check.
The corpus SHA-256 remains
`75508dd43fb7e462c9af3e59e070fb6f2defb06d8a344f5a1a7f1691850df5cd`.
The GULPS pin and release tag were not changed.

## Iteration at the stricter tolerances

The baseline for this iteration was the strict candidate above, frozen before
further edits. The retained changes use squared distances for comparisons in
spectrum classification and repeated-spectrum construction. Spectral checking
now takes a square root after each maximum-distance reduction instead of for
each entry. Acceptance constants and search budgets are unchanged.

The full comparison found identical status, spectral error, orthogonality
error, and determinant error for every row. All 1,093,687 finite spectral
errors were exactly equal, without a rounding threshold. The four infeasible
inputs still return no witness. Worst spectral error remains `9.762e-13`.

Three final paired runs used release mode, fat LTO, one codegen unit, CPU 2,
no diagnostics, and no concurrent builds, tests, or other benchmark runs.
Total solver time fell by 5.3–5.5%. Median latency fell by about 9–10%; the
worst individual timing varied between runs and does not show a consistent gain.

| Metric | Strict baseline | Retained optimization |
|---|---|---|
| Total solver time (s) | 7.861–7.941 | 7.439–7.518 |
| Mean (µs) | 7.188–7.261 | 6.802–6.874 |
| Median (µs) | 2.860–2.890 | 2.590–2.620 |
| p95 (µs) | 20.430–20.690 | 19.770–20.080 |
| p99 (µs) | 44.220–44.990 | 42.210–43.080 |
| p99.9 (µs) | 401.630–403.969 | 375.959–381.479 |
| Maximum (ms) | 19.431–20.116 | 18.504–20.814 |

Several larger changes were tested and rejected:

| Experiment | Result | Decision |
|---|---|---|
| Switch numerical orientations after four starts | Roughly halved the worst latency, but changed individual errors | Revert |
| Reduce fixed-root searches to four starts | Slower on the difficult subset | Revert |
| Solve only the three active rotations for a fixed root | Preserved coverage and reduced latency, but one combined search change raised error from `6.206e-16` to `9.874e-14` | Revert |
| Share the two polynomial-root routines | Preserved coverage, but increased the maximum orthogonality defect and gave no clear speed gain | Revert |
| Tighten numerical acceptance to `1e-14` with the search changes | Improved spectral p99 to `7.820e-15`, but increased errors above `1e-13` from 645 to 879 and increased total time | Revert |

These trials show why aggregate accuracy alone is insufficient. The retained
optimization preserves every independently measured error in the corpus.

Three endpoint-test assertions still used `1e-8`, although production endpoint
verification already used `1e-12`. They now use the independent checker's
`1e-12` constant for orthogonality, determinant, and reconstruction. `make test`
passes all five tests, including the full corpus and these stricter endpoint
assertions. `make lint` passes.

## Refinement below the acceptance ceiling

Baseline: `8e10ac9`. The spectral acceptance ceiling is tightened from
`1e-12` to `1e-13`. Construction refinement keeps its existing `1e-13`
stopping criterion. Final polishing aims for root distances below `1e-14`
and can retain partial progress when it cannot reach that target. It replaces
a frame only when the new root-error estimate, plus its eigenbasis residual
allowance, is below the old estimate minus that allowance. This avoids
replacing a witness on the strength of a difference within basis uncertainty.

Requiring every construction search to reach `1e-14` was rejected. It improved
the aggregate distribution but abandoned intermediate witnesses and produced
four confirmed error increases of about `1.5e-14` to `3.2e-14`. Separating
construction acceptance from final polishing preserves those four witnesses.
Tightening the basis projection's stopping condition did not repair them and
added runtime. Reducing random starts to four also changed witnesses without
a sufficient measured benefit; the 24-start budgets remain.

The checker now tries a third Schur rotation, `0.6 + 0.8i`, when both existing
paths exhaust their iteration budgets. It keeps the same iteration limit,
convergence setting, and acceptance threshold. Independent LAPACK and
80-decimal-digit calculations confirmed that the motivating frame has error
about `6.8e-15`; a fixed regression test covers that checker failure. Both
baseline and candidate use the same updated checker in the comparisons below.

All rows pass: 1,093,687 witnesses and four infeasible rejections. The full
comparison records 59,924 smaller, 1,033,751 equal, and 12 larger spectral
errors. An independent 80-digit eigensolve checked all 12 apparent increases:
the largest actual increase was `9.24e-19`; the two largest reported increases
were instead actual improvements of `2.97e-14` and `4.16e-14`. This audit checks
those individual differences; it is not a general error proof.

| Spectral metric | Baseline | Revised solver |
|---|---|---|
| p50 | `1.047e-15` | `1.024e-15` |
| p99 | `4.591e-14` | `1.224e-14` |
| p99.9 | `9.170e-14` | `3.418e-14` |
| Maximum | `9.762e-13` | `1.216e-13` |
| Count above `1e-13` | 645 | 2 |

These are the independent corpus checker's readings, including its rounding
error. Both remaining readings above `1e-13` are Schur artifacts: high-precision
checks of rows 1,071,777 and 1,071,780 give `6.62e-15` and `6.83e-15`.
The report retains the original readings rather than substituting special
results for those rows. Maximum orthogonality defect falls from `1.334e-13`
to `1.033e-13`; maximum determinant error falls from `1.041e-13` to `1.910e-14`.

Three sequential paired runs used release mode, fat LTO, one codegen unit,
CPU 2 affinity, and no diagnostics or concurrent builds, tests, or benchmarks.
The stronger precision target costs 7.1–7.4% more total solver time.

| Metric | Baseline | Revised solver |
|---|---|---|
| Total solver time (s) | 6.915–7.570 | 7.425–8.109 |
| Mean (µs) | 6.323–6.921 | 6.789–7.414 |
| Median (µs) | 2.420–2.660 | 2.510–2.770 |
| p95 (µs) | 18.080–20.150 | 18.750–20.990 |
| p99 (µs) | 37.890–42.610 | 40.840–45.870 |
| p99.9 (µs) | 365.690–385.989 | 395.260–429.100 |
| Maximum (ms) | 17.517–20.390 | 17.503–17.602 |

Row 1,075,377 is slowest for both solvers in all three runs. These timings do
not establish a consistent improvement in maximum latency. `make test` passes
all five tests, including the full corpus and endpoint reconstruction;
`make lint` passes. The corpus bytes and SHA-256 are unchanged.

## Core algorithm experiments and retained deletion

Baseline: `f8c3374`. This pass removes 449 lines of radical-search filters:
the reduced residue interval filter, skeleton parity calibration, and the
U-residue ray and magnitude filters. Their candidate construction, residue
checks, and final spectral verification remain. The simple residue interval
and strict-word filters remain because their removal increased runtime or
changed witness accuracy. Three unused profiling counters were also removed.

The full paired comparison passes all 1,093,691 rows, including the same four
infeasible rejections. Every row has identical status, spectral error,
orthogonality error, and determinant error. No acceptance tolerance or corpus
byte changed. The retained solver source matches the measured deletion.

One paired release run used fat LTO, one codegen unit, CPU 2 affinity, and no
diagnostics or concurrent builds, tests, or benchmarks:

| Metric | Baseline | Retained deletion |
|---|---|---|
| Total solver time (s) | 7.643 | 7.558 |
| Mean (µs) | 6.989 | 6.910 |
| Median (µs) | 2.580 | 2.570 |
| p95 (µs) | 19.650 | 19.240 |
| p99 (µs) | 43.110 | 42.140 |
| p99.9 (µs) | 394.830 | 392.819 |
| Maximum (ms) | 17.392 | 17.566 |

The 1.1% total-time difference is small and comes from one run; this is a
complexity reduction, not evidence of a substantial speedup. Row 1,075,377
remains the slowest. `make test` and `make lint` pass, including the full
corpus and endpoint reconstruction.

Larger algorithm experiments were rejected:

| Experiment | Evidence | Decision |
|---|---|---|
| Replace radical dispatch with numerical recovery | Sample passes but takes about 10.5 times as long | Revert |
| Direct quadratic charts for a routed root's SO(3) block | Full corpus passes, about 3% faster, but individual errors increase and code grows | Revert |
| Choose numerical orientation by spectral clustering | Worst full-run time falls from about 18 to 8 ms, but one case is declined and individual errors increase | Revert |
| Scale numerical Jacobian columns | No useful sample or worst-case improvement | Revert |
| Joint Jacobi diagonalization of the complex symmetric matrix | Sample passes; no speed gain from either trigonometric or algebraic rotations | Revert |
| Refine only eigenphases | Sample time and worst latency increase | Revert |
| Compress the unitary tangent model and alternate row/column rotations | All 2,744 baseline numerical cases pass; time falls 0.933 → 0.808 s and maximum 17.5 → 8.3 ms, but high-precision checks confirm individual error increases | Revert |
| Direct diagonal/two-block spectral decomposition | Sample passes, but adds slow refinement paths and changes individual errors | Revert |
| Tighten the edge search's routed-root distance from `1e-2` to `1e-12` | Full corpus passes in 7.425 versus 8.057 s; 80-digit checks confirm individual error increases up to `7.70e-14` | Revert |

The smaller sample contains every 53rd corpus row. The numerical subset
contains all rows that reached numerical recovery in the frozen baseline;
neither substitutes for full-corpus validation. Tightening the edge filter
removed useful refinement starts, even though the unrefined sparse frames
could not meet the spectral tolerance. A stricter search filter is therefore
not automatically an accuracy improvement.

## Polishing at clustered spectra

Baseline: `f8c3374` with the radical-filter deletion above. This pass keeps
every search budget and acceptance limit. It changes final polishing, one
face-search loop, and the reuse of computed spectral states.

The face search tries both orders in each of its two blocks. Reversing an
order gives a sign-conjugate frame with the same spectrum, so the four
variants differ only by roundoff. That roundoff matters only when a
candidate is near acceptance, where refinement treats the variants as
restarts. The loop now stops at the first residual above `16 * ACCEPT`.
Every row keeps its status and spectral error, and one paired run measured
7.430 → 7.073 s.

At a vertex, edge, or face frame, the spectrum is stationary in some
rotation directions. A root error `e` there needs rotations of order
`sqrt(e / sigma_max)`, and Levenberg-Marquardt stops without progress. When
polishing stalls, it restarts along each null direction of the root rows of
the Jacobian, with that step size. The eigenbasis rows are excluded from this
test; they remain nonzero at a critical point.

Most remaining polishing failures were failures of the estimate rather than
the iteration. The real eigenbasis comes from one real combination of the
real and imaginary parts. Where projected roots nearly coincide, its residual
stays near `1e-14`, and the old replacement rule then rejected a frame that
was already better. Some constructed frames also carried orthogonality drift
of the same size, which enlarges the residual. Polishing now measures frames
with a joint eigenbasis: if the residual exceeds `1e-15`, one sweep of real
Jacobi rotations of the complex matrix is kept when it lowers the residual.
Polishing first runs the existing iteration, then repeats it with the joint
estimate if that estimate has not converged, and keeps the better frame. It
is attempted whenever the joint root estimate exceeds `1e-14`. A polished
frame replaces the original if its joint error bound is below
`max(root - 4 * residual, (root + 4 * residual) / 2)`. The first term is the
previous rule. The second applies when the residual dominates and requires
the bound to halve.

Verification's state is reused as the plain estimate, and the joint sweep
runs only when its residual exceeds `1e-15`. These reuses change no row and
reduce total time by about 2.4%.

All 1,093,691 rows pass, including the same four infeasible rejections.
Relative to the baseline, 15,155 spectral errors are smaller, 1,078,466 are
equal, and 66 are larger. An 80-digit eigensolve of all 66 increases found a
largest actual increase of `2.15e-15`, at row 101,581 (`7.61e-15` →
`9.76e-15`). The 40 largest reported decreases are actual decreases.

| Metric | Baseline | Revised solver |
|---|---|---|
| Spectral p99 | `1.224e-14` | `8.648e-15` |
| Spectral p99.9 | `3.418e-14` | `1.045e-14` |
| Spectral maximum | `1.216e-13` | `1.216e-13` |
| Orthogonality p99.9 | `2.487e-14` | `1.199e-14` |
| Orthogonality maximum | `1.033e-13` | `9.658e-14` |
| Determinant maximum | `1.910e-14` | `7.883e-15` |

The two readings above `1e-13` are the Schur artifacts of rows 1,071,777 and
1,071,780 described earlier. Three paired runs used release mode, fat LTO,
one codegen unit, CPU 2 affinity, and no diagnostics or concurrent builds,
tests, or benchmarks:

| Metric | Baseline | Revised solver |
|---|---|---|
| Total solver time (s) | 7.432 to 7.467 | 7.865 to 7.877 |
| Median (µs) | 2.520 to 2.540 | 2.560 |
| p95 (µs) | 18.680 to 18.880 | 17.420 to 17.540 |
| p99 (µs) | 40.170 to 40.530 | 40.640 to 40.918 |
| p99.9 (µs) | 393.700 to 396.329 | 599.529 to 602.089 |
| Maximum (ms) | 17.218 to 17.522 | 17.341 to 17.633 |

The accuracy gain costs 5.5 to 5.8% total time, mostly in the p99.9 tail where
polishing restarts run. Row 1,075,377 remains the slowest.

About 180 rows still have actual errors above `2e-14`, most near spectra
perturbed by about `1e-12` from a degenerate point. There the correction
needed along a well-conditioned direction changes the coupling inside a
root cluster by about the cluster gap, so the first-order root model fails.
The iteration makes no progress, or only partial progress.

| Experiment | Evidence | Decision |
|---|---|---|
| Keep only one face order | 10.8% faster; five rows increase to about `9e-14` (80 digits) | Revert |
| Face angle from `atan2` of sine-product `cos²` and `sin²` | 106,616 smaller and 86,313 larger errors; actual increases to `9.5e-14` | Revert |
| Skip refinement of constructed vertex frames | 8.4% faster; five actual increases to `8e-14` | Revert |
| Restart constructed-frame refinement along null directions | 15% slower; 778 larger errors, three above `1e-13` | Revert |
| Accept vertex frames by direct root matching | 1.9% faster; four actual increases to `1.7e-14` | Revert |
| Jacobi-finished eigenbasis in every verification | Laxer acceptance admits worse witnesses; 9,666 larger errors, 732 above `1e-14` | Revert |
| Projection direction chosen from target root differences | 4% slower; 32,029 larger errors | Revert |
| Jacobi angle from the real part with larger anisotropy | Loses polishing on about 100 rows | Revert |
| Polar step on every constructed frame | Orthogonality maximum `1.3e-15`, but spectral errors near `3e-16` increase to `1e-14` | Revert |
| Cap polishing restarts at 30 iterations | 1% faster; an actual increase to `7e-14` | Revert |
| Joint Newton on `Vᵀ M(O) V = diag(t)` in frame and basis | Fixes 11 of the 179 remaining rows | Revert |
| Smaller eigenbasis penalty and damping during polishing | Helps some clustered rows; does not remove the stall | Revert |

The polar-step result shows that a frame can match the spectrum through `Oᵀ`
more accurately than its projection onto SO(4) does. Orthogonality and
spectral error therefore trade against each other near the tolerance.

## Validation

The full corpus test checks spectral matching, orthogonality, determinant,
and endpoint reconstruction. The independent comparison also checks every
returned frame against the original baseline. Passing these finite tests is
not a general completeness proof.

The `solve` and `solve_with_factors` signatures are unchanged. The optional
experimental diagnostic interface changed; see the [source guide](architecture.md).
This experiment measures the standalone solver. GULPS synthesis performance
still depends on how the returned endpoint factors affect later decompositions.
