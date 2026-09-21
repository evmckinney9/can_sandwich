# Production baselines

## Combined audited corpus

The combined-file run on 2026-09-21 verified **1,075,790 / 1,075,950** cases
(99.9851%). It reported 156 incorrect results, four declines, and no timeouts
or protocol errors. Corpus SHA-256:
`5f1a6689a1a5347d901194555a533897f961f309bf9db61709092fc0cac24bae`.

The original 1,074,495 cases reproduced the previous coverage exactly.
Of the 1,455 independently witnessed additions, 1,451 passed. Four new
failures involve split 3+1 target spectra:

| Combined row | Added-source row | Result |
| ---: | ---: | --- |
| 1,075,398 | 903 | Spectral error `1.0000000064e-8` |
| 1,075,403 | 908 | Spectral error `1.0000000005e-8` |
| 1,075,408 | 913 | Declined; actual target minimum gap about `1.1e-7` |
| 1,075,409 | 914 | Declined; actual target minimum gap about `1.1e-7` |

The first two are floating-point tolerance-edge failures; their precise
classification can depend on numerical platform. The fixed acceptance limit
was not changed. All four planted witnesses have spectral error below
`1.4e-15`. Construction details are stored in `feasible_targeted.json`.

Local timing: median 83.4 microseconds, p95 129.2 microseconds, p99 192.9
microseconds, maximum 178 milliseconds; summed requests 99.62 seconds and
wall time 262.97 seconds. This run does not qualify for full-coverage ranking.

After the production adapter build below, reproduce with:

```sh
python benchmark/run.py --label production \
  --report production.json --command target/release/can_sandwich_server
```

## Complete default suite

The full independent grader run on 2026-09-21 verified **1,074,339 / 1,074,495**
cases (99.9855%). It reported 154 incorrect results, two declines, and no
timeouts or protocol errors. This production version does not have full
coverage under the independent checker.

The local run took 257 seconds wall time. Per-request median was 80.0
microseconds, p95 126.9 microseconds, p99 187.5 microseconds, and maximum
174 milliseconds. Summed request time was 96.24 seconds. These measurements
include transport and exclude verification from the per-request times.

Use the build command below, then run with the default corpus to test the current
complete suite, which now also includes the audited additions. The smaller historical baseline below is retained for
comparison; its five failures do not describe the complete suite.

## Independent submission grader

The public submission profile on 2026-09-21 verified **13,182 / 13,187** cases:
all six correctness fixtures and 13,176 of 13,181 stratified corpus cases.
There were two declines (corpus rows 2889 and 7172), three spectral failures
(rows 12375, 12586, and 12791), and no timeouts. Full coverage is false, so
this production version does not qualify for a full-coverage speed ranking.

The three spectral errors were `1.172e-8`, `1.179e-8`, and `1.028e-8`, above
the fixed `1e-8` limit. The independent matrix eigenspectrum check differs
from the production solver's internal certificate; its coverage is reported
separately from the legacy results below.

One local run measured a median of 86.8 microseconds, p95 of 356 microseconds,
p99 of 1.20 milliseconds, and maximum of 176 milliseconds per request.
These include transport and exclude verification and successful startup.
They are local observations, not portable speed claims.

Reproduce from the repository root:

```sh
cargo build --manifest-path Cargo.toml -p can_sandwich --release \
  --features benchmark --bin can_sandwich_server
python benchmark/run.py --label production \
  --corpus benchmark/unit_cases.npy \
  --corpus corpus/feasible_stratified.npy \
  --report production.json --command target/release/can_sandwich_server
```

The expected exit status is 1 because five cases do not pass. The JSON report
records environment, corpus and grader hashes, and per-case results.

## Legacy native and circuit checks

Measured on 2026-09-21 with the current workspace. The corpus scripts now use
`LocalEquivalenceClass` and `GulpsDecomposer(...).decompose(...)`.
The locked corpus passed its digest check. No locked fixture changed.

The full public replay passed 10,927 of 10,935 rows at the existing `1e-8`
matrix tolerance. All rows completed within the 0.5-second limit.
These are production results, separate from experimental algorithms.

The standard `bench_both.sh` run completed all stages:

| Atomic corpus | Solved | Maximum reported residual |
| --- | --- | --- |
| Stratified | 13,179 / 13,181 | `9.981e-9` |
| Linspace | 761,308 / 761,308 | `9.996e-9` |
| Haar | 300,000 / 300,000 | `9.036e-9` |

The two unsolved atomic rows are 2889 and 7172. Both stable tail rows passed
1,001 repetitions. The public stage reproduced the eight failures below.
The non-iterative guard now scans `../core/src`, which replaces the removed
`../core/src/realization` directory. Its pattern check passed.

| Public row | Atomic row | Public failure | Current atomic result |
| --- | --- | --- | --- |
| 2876 | 2889 | Outside atlas | Unsolved |
| 3913 | 3926 | Edge frame error `1.02e-8` | Edge, residual `5.210e-9` |
| 5338 | 5351 | Outside atlas | Vertex, residual `7.948e-16` |
| 6211 | 6224 | Chart frame error `9.38e-9` | Chart, residual `9.618e-9` |
| 6494 | 6507 | Outside atlas | RankOne31, residual `1.299e-15` |
| 9189 | 9202 | OnePlusThree frame error `1.06e-8` | OnePlusThree, residual `9.572e-9` |
| 9346 | 9359 | Vertex frame error `9.10e-9` | Vertex, residual `9.098e-9` |
| 10344 | 10357 | Outside atlas | Vertex, residual `5.494e-16` |

Atomic rows equal public rows plus 13 because the atomic corpus starts with
13 additional regressions. A fresh release build reproduced the atomic results.
The public compiler can select a different sentence from the supplied witness.
Thus an atomic pass does not establish that the selected public sentence works.
The crate README already records public failures 2876, 5338, and 10344.

The certificate and recovery stages use different error measures and limits.
`../src/certificate.rs` accepts a diagonal spectral error below `1e-8`.
Other certificate branches use the same limit.
`../../core/src/recovery.rs` sets `RECOVERY_ACCEPT` to `9e-9`.
Its `factor` function checks the matrix error at line 189.
An accepted spectral certificate therefore does not guarantee successful frame
recovery. Row 9346 shows the difference directly. A repair needs a compatible
certificate or another candidate, not a larger recovery tolerance.

A fresh small corpus also passed: 843 public rows, worst matrix residual
`4.412e-9`. The generator produced 1,651 atomic triples using
`--samples-per-pair 1 --target-rounds 1`.

Reproduce the public baseline from the repository root:

```sh
.venv/bin/python scripts/generate_realization_edge_corpus.py \
  --output corpus/feasible_stratified.npy --check-existing
.venv/bin/python scripts/validate_realization_pipeline_corpus.py \
  corpus/feasible_stratified.pipeline.npz --max-case-seconds 0.5
```

Use repeated `--corpus-row` options to select the eight public failures.
These numbers use the legacy native and public circuit benchmarks. The
competition runner uses its own independent numerical checker and reports
its results separately.
