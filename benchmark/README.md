# Submit one solver file

Submit a Python `.py` file or a Rust `.rs` file. We run the fixed correctness
cases and corpus and return the coverage and timing report. Any method is
allowed, including numerical iteration. Your file only implements `solve`.
The harness handles compilation, process isolation, time limits, and reports.

## Function interface

Python:

```python
def solve(c, g, t):
    # c, g, t are lists of three original binary64 monodromy coordinates.
    # Return a real 4x4 frame as nested lists or a NumPy array.
    # Return None if your algorithm cannot find a frame.
    return None
```

Rust:

```rust
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<[[f64; 4]; 4]> {
    None
}
```

[example_solver.py](example_solver.py) and [example_solver.rs](example_solver.rs)
are minimal templates. They solve only identity inputs and decline other
cases. They are interface examples, not competitive algorithms.

Rust submissions compile in release mode in a separate temporary crate.
The standard library, `nalgebra` 0.35, and `serde_json` 1 are available.
Python submissions run with the runner's interpreter, where NumPy is
available. Additional dependencies must be declared with the submission.
Do not add dependencies or lookup tables to the production solver.

## Run and get feedback

From the repository root, with Python 3.10+ and NumPy installed:

```sh
python benchmark/run.py my_solver.py --report result.json
python benchmark/run.py my_solver.rs --report result.json
```

The same command accepts either language. The runner writes `result.json`
and `result.cases.jsonl`, and prints a compact coverage and timing summary
to the terminal. The JSON summary contains:

- Passed, failed, declined, timed-out, invalid, and error counts.
- Median, p95, p99, maximum, and summed per-case time.
- Worst verification errors and the path to the per-case results.
- Source and corpus hashes, environment, and timing conditions.
- Whether the run has verified full coverage and is eligible for comparison.

Every case record names its corpus and zero-based row index. It includes
elapsed time and either verification errors or a failure reason. The process
exits nonzero when any attempted case does not pass.

For a quick development run:

```sh
python benchmark/run.py my_solver.py \
  --step 100 --report smoke.json
```

`--step 100` checks rows 0, 100, 200, and so on across the complete file:
10,760 cases in the current corpus. `--stride` is an alias. The default step
is 1, which checks everything. A fixed step is a repeatable development
sample, not a guarantee that every rare construction family is represented.

Partial runs and custom corpora produce useful feedback but never qualify
for a full-coverage ranking. Use `--row N` to reproduce a failed case from
the combined corpus. Row IDs refer to that file; the provenance metadata
records the source file and original row range.
The original inputs are not rationalized, snapped to spectral strata, or
modified before they reach your function.

## The problem and verification

Each input is a triple `[C, G, T]` of monodromy coordinates. For a coordinate
vector `m`, define

```text
w = [m0 + m1, m0 + m2, m1 + m2]
s(m) = exp(i*pi * [w0-w1+w2, w0+w1-w2, -w0-w1-w2, -w0+w1+w2])
A = diag(s(C))
B = diag(s(G))
```

Return a real matrix `O` in `SO(4)` for which the eigenvalue multiset of
`A @ O @ B @ O.T` matches either `s(T)` or `-s(T)`. The two central target
lifts describe the gate-equivalence problem used by this corpus.
Multiplicity matters. Input ordering must follow the definition above.

The stored corpus remains in monodromy coordinates so that existing boundary
and near-degenerate cases retain their original binary64 values. A solver can
work entirely with eigenvalues. In Python, use this conversion inside your
submitted file:

```python
import numpy as np


def eigvals(m):
    w0, w1, w2 = m[0] + m[1], m[0] + m[2], m[1] + m[2]
    return np.exp(
        1j
        * np.pi
        * np.array(
            [
                w0 - w1 + w2,
                w0 + w1 - w2,
                -w0 - w1 - w2,
                -w0 + w1 + w2,
            ]
        )
    )
```

This conversion fixes the basis ordering of the returned frame. Do not sort
the eigenvalues unless you also transform the frame back to this ordering.
No inverse phase extraction or change to the corpus is needed.

The coordinate tests cross-check this convention against `gulps` canonical
matrices. The grader does not pass inputs through `LocalEquivalenceClass`:
that constructor quantizes coordinates and can erase near-boundary offsets.

The independent NumPy checker requires all three errors to be at most
`1e-8`:

- Maximum entry of `abs(O.T @ O - I)`.
- `abs(det(O) - 1)`.
- Smallest maximum eigenvalue distance over all 24 permutations and both
  central target lifts.

All entries must be finite real numbers. The checker computes the actual
matrix spectrum. A solver's own residual or success claim is not accepted
as evidence. This is a numerical benchmark, not an exact-arithmetic proof
or a test of complete gate-circuit reconstruction.

## Coverage and speed

Every default run reads all **1,075,950 cases** from one file,
`../corpus/feasible_all.npy`. It contains
all original basic, stratified, linspace, and Haar cases, followed by the
independently witnessed cases added after the [coverage audit](COVERAGE.md).
The terminal reports one total. [corpora.json](corpora.json) locks the file's
row count and hash; `../corpus/feasible_all.json` records source hashes and
row ranges. Missing or changed data stops the run.

For development, `--max-cases N` limits the run and `--row N` selects a case.
These runs do not qualify for ranking. There is no smaller default profile.

Rebuild the combined artifact from the retained source files:

```sh
python benchmark/build_corpus.py \
  --additional corpus/feasible_targeted.npy
```

The builder verifies the original source hashes and every copied slice bit
for bit. It preserves order, duplicates, and all previously failing inputs.
It refuses to replace different output unless `--force` is supplied.
The new construction inputs have a separate generator, witness file, and
metadata; see the coverage audit for their scope and limitations.

First compare verified coverage. Only entries that pass every case in the
same complete corpus qualify for speed comparison. A timeout, decline,
incorrect matrix, or malformed response prevents qualification. The default
limit is 0.5 seconds per case and ten seconds for process startup.

Compare summed per-case time for fully passing entries under the same
machine, corpus hash, limits, order, dependencies, and thread settings. The
median and tail statistics show latency distribution. Repeat complete runs
to assess timing variance before publishing a ranking. Reports are evidence
for organizer reruns, not a self-authenticating leaderboard.

Measured time includes the function call and transport overhead, from the
request through the complete response. Successful import/startup and Rust
compilation are recorded separately and excluded. Verification runs outside
the candidate's timed interval. No solver-reported timer determines the score.
A persistent worker serves successive cases. A request timeout kills that
worker before the next case. Internal transport is shared across submissions
and does not have to be implemented in the submitted file.

Submit an algorithm, not a table of corpus answers. Fresh cases from
`../scripts/generate_realization_edge_corpus.py` can be used as additional
checks. Corpus coverage is a measured result, not a claim of coverage for
all feasible mathematical inputs. The runner executes submitted code as the
current user and is not a security sandbox.

## Maintainer checks and production reference

```sh
python -m unittest discover -s benchmark -p 'test_*.py'
cargo build --manifest-path Cargo.toml --release \
  -p can_sandwich --features benchmark --bin can_sandwich_server
python benchmark/run.py --label production \
  --report production.json --command target/release/can_sandwich_server
```

The production adapter calls only the normal `solve` API. It does not enable
diagnostic or experimental routes. The legacy diagnostic binary remains
available through the separate `diagnostics` feature, with source in
[diagnostics.rs](diagnostics.rs). Research prototypes do not participate in
this benchmark.
