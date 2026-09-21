# can_sandwich

The production depth-two two-qubit realization solver used by `gulps-core`.
It takes three monodromy coordinate vectors and returns a verified real
`SO(4)` frame, or declines. It uses binary64 arithmetic and numerical
certificates. It does not claim universal coverage.

```rust
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution;
```

The default library exports only `solve`, `Solution`, `Rung`, and `Mat4`.
Production code lives in `src/`. Diagnostic routes require the `diagnostics`
feature. Benchmark adapters and drivers live in `benchmark/`. Research
records live in `docs/inverse-horn/` and do not participate in the solver or
benchmark.

## Build

This repository is a standalone Rust crate. Rust 1.97 or newer is required.

```sh
cargo build --release --locked
cargo test --locked --all-targets --all-features
python -m pip install numpy pytest
python -m pytest tests
```

The Python tests build the production Rust adapter and check its frames with
an independent NumPy eigenvalue calculation. They require neither GULPS nor
Qiskit. See [test provenance](tests/README.md) for the captured inputs and
known near-wall gaps. These are development checks, not a Python package.

For benchmarks on this machine, set `RUSTFLAGS="-C target-cpu=native"`.
Portable builds use the default target settings.

GULPS includes a pinned checkout as a Git submodule. Its build compiles this
library as a path dependency. Solver commits do not update the GULPS pin.
Use a separate checkout for research to keep the GULPS checkout stable.
The previous Python project is preserved on `archive/python-project` and
in Git history. It is no longer part of the build.

## Submit an algorithm

Submit **one Python or Rust file** with a `solve(c, g, t)` function. The
benchmark runs fixed correctness cases and the unchanged corpus, then
reports cases passed, failed case IDs, and timing. Any method is allowed.
Verified full coverage is required before comparing speed.

From the repository root:

```sh
python benchmark/run.py my_solver.py --report result.json
python benchmark/run.py my_solver.rs --report result.json
```

Python 3.10+ and NumPy are required. Rust submissions also require Cargo.
See [the submission instructions](benchmark/README.md) for the function
signature, examples, verification rules, and the reported data.

The benchmark reads one combined file, `corpus/feasible_all.npy`, containing
all existing cases plus independently witnessed additions. The manifest
locks its hash and row count. Missing files stop the run. See the
[coverage audit](benchmark/COVERAGE.md) for the gaps in the old sampling
methods and what the new cases test.

## Production validation

```sh
cargo test -p can_sandwich
cargo clippy -p can_sandwich --all-targets -- -D warnings
cargo test -p can_sandwich --features diagnostics
cargo clippy -p can_sandwich --features diagnostics --all-targets -- -D warnings
```

[Architecture and earlier measurements](docs/architecture.md) describe the
production cascade and its numerical certificates.
