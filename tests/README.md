# Solver regression tests

Run `python -m pytest tests` with NumPy, pytest, and Rust installed. The tests
build the production JSON-lines adapter in release mode. The witness checker
in `benchmark/run.py` verifies real SO(4) frames and the full eigenvalue
multiset independently of the solver certificate. No GULPS or Qiskit import
is needed. A known decline is a strict expected failure; an invalid frame,
protocol error, or unexpected success fails the test.

`fixtures/gulps_segments.json` preserves 1,873 solver calls from GULPS commit
`38014b5`, with solver commit `af4e5ea`. Inputs were captured directly at
`gulps-core::solver::solve` before removing the original coverage sweeps:

- `strata/`: 38 dressed chamber points under nine native gate sets.
- `wall/`: five near-wall points under those gate sets, including 27 declines.
- `property/`: 12 seeded dressed targets under each gate set.
- `forward/`: 512 physically generated depth-two segments from the Rust test.

Each row preserves binary64 coordinates as round-trippable decimal numbers,
the source case, and its observed solved/declined status. Segment numbers count
calls within a source case. `forward/` numbers follow the original nested
C-class, G-class, local-frame iteration order. Zero- and one-gate decompositions
made no solver call. Failed decompositions contributed calls up to and including
the first decline. These fixtures do not depend on current GULPS planning.
