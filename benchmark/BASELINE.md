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
