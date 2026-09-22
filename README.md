# can_sandwich

The depth-two two-qubit realization solver used by `gulps-core`.
It takes three monodromy coordinate vectors and returns a verified real
`SO(4)` frame, or declines. It uses binary64 arithmetic and numerical certificates.

```rust
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution;
```

The library exports `solve`, `Solution`, `Rung`, and `Mat4`. Production code
lives in `src/`; Rust tests and one binary fixture live in `tests/`.
Diagnostic routes require the `diagnostics` feature. Research records live
in `docs/inverse-horn/` and do not participate in the build.

## Development

Rust 1.97 or newer is required.

```sh
make build
make test
make lint
```

The tests check solver frames with an independent matrix eigenvalue calculation.
`make test` uses release mode for the million-case fixture.

## Compare a prototype

```sh
cargo test --release --test solver compare_candidate -- --ignored --nocapture
```

Replace the candidate function in `tests/solver.rs` with your prototype.
It runs against production on the same cases and reports correctness and time.
See [tests](tests/README.md) for the fixture format and comparison details.

GULPS uses a pinned Git submodule and compiles this crate as a path dependency.
Solver commits do not update that pin. The former Python project remains on
`archive/python-project` and in Git history.

[Architecture](docs/architecture.md) describes the solver cascade and certificates.
