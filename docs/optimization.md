# Corpus-driven simplification, 2026-09-23

Baseline: `b3dbb7fed701257fab7a7b0a3e8b9da9a6247ed4`.
Corpus: 1,093,691 rows, SHA-256
`75508dd43fb7e462c9af3e59e070fb6f2defb06d8a344f5a1a7f1691850df5cd`.
Compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`.

## Experiments

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

## Implementation

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

## Final measurements

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

## Validation

The full corpus test checks spectral matching, orthogonality, determinant,
and endpoint reconstruction. The independent comparison also checks every
returned frame against the original baseline. Passing these finite tests is
not a general completeness proof.

The `solve` and `solve_with_factors` signatures are unchanged. The optional
experimental diagnostic interface changed; see the [source guide](architecture.md).
This experiment measures the standalone solver. GULPS synthesis performance
still depends on how the returned endpoint factors affect later decompositions.
