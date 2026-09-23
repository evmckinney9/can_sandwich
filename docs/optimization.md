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

## Validation

The full corpus test checks spectral matching, orthogonality, determinant,
and endpoint reconstruction. The independent comparison also checks every
returned frame against the original baseline. Passing these finite tests is
not a general completeness proof.

The `solve` and `solve_with_factors` signatures are unchanged. The optional
experimental diagnostic interface changed; see the [source guide](architecture.md).
This experiment measures the standalone solver. GULPS synthesis performance
still depends on how the returned endpoint factors affect later decompositions.
