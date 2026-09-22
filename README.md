# can_sandwich

The depth-two two-qubit realization solver used by `gulps-core`.
It takes three monodromy coordinate vectors and returns a verified real
`SO(4)` frame, or declines. It uses an algebraic cascade followed by a bounded Levenberg–Marquardt fallback.
Every result must pass numerical certification; convergence is not guaranteed.

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

`make test` runs the corpus and three rejection/checker checks in release mode.
Frames must satisfy SO(4) and an independent spectral check at `1e-8`.

Regenerate with `python3 tests/generate.py` (requires NumPy and SciPy).
`tests/cases.bin` stores nine little-endian `f64` values per row:
`[C0,C1,C2,G0,G1,G2,T0,T1,T2]`, without a header.

To compare a prototype, replace `candidate` in `tests/solver.rs`, then run:

```sh
cargo test --release --test solver compare_candidate -- --ignored --nocapture
```

This reports solver time and fixed/regressed rows. Only regressions fail.

Follow [Google’s testing guide](https://abseil.io/resources/swe-book/html/ch12.html#test_via_public_apis):
test public behavior. Put solver regressions in the corpus generator; keep
separate checks for impossible targets, invalid frames, and checker errors.
Do not pin solver branches or duplicate internal formulas.

GULPS uses a pinned Git submodule and compiles this crate as a path dependency.
Solver commits do not update that pin. The former Python project remains on
`archive/python-project` and in Git history.

[Architecture](docs/architecture.md) describes the solver cascade and certificates.
