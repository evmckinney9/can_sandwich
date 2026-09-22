# can_sandwich

GULPS depends on what is, to the best of my knowledge, an open problem in
mathematics. We can solve it numerically. I think current AI models may be close
to an exact construction.

The research changes quickly, so we moved the solver into its own Rust crate
and Git repository. This is a good problem for models because we can check
the matrices they produce.

For now, we let this be extremely vibe-coded. The algorithm needs to be fast
and correct over the corpus. That is sufficient for GULPS. We want to keep
improving it, so this project includes the corpus and tools to compare a
proposed algorithm with the production solver.

## Problem

Given three feasible spectra, construct matrices $A,B,C\in SU(4)$ with those
spectra such that $AB=C$. We want a practical exact construction for dimension
four that also covers boundary points and repeated eigenvalues.

The [formulation](docs/research.md#the-unrestricted-problem) states the problem.
The [research requirements](docs/research.md#exact-input-and-output) specify the
input model, known algebraic methods, and what a new construction must prove.

GULPS needs a local gate between two canonical two-qubit gates. Its solver must
return a real SO(4) matrix in the magic basis, with global phase tracked separately.
The unrestricted research problem allows any SU(4) witness. The
[coordinate conventions](docs/research.md#gulps-coordinates-and-endpoint-factors)
explain the difference and define the API inputs and outputs.

## Solver

```rust
use nalgebra::Matrix4;

pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<Matrix4<f64>>;

pub fn solve_with_factors(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<(Matrix4<f64>, Matrix4<f64>, Matrix4<f64>, f64)>;
```

The inputs `c`, `g`, and `t` are monodromy triples for the left gate, right gate
or prefix, and target. `solve` returns the middle frame `O`.
`solve_with_factors` returns `(O, L, R, phase)`, including both endpoint frames.
It reuses the verification eigenbasis to avoid another eigendecomposition in GULPS.

The solver tries algebraic constructions, then bounded Levenberg–Marquardt
refinement and restarts. It uses binary64 arithmetic and checks results against
the original spectra. `None` means it found no accepted result within its
search budget. It does not prove infeasibility.

The [source guide](docs/architecture.md) describes the current implementation
and planned redesign. Internal diagnostics use the `diagnostics` feature.

## Development

Rust 1.97 or newer is required.

```sh
make build
make test
make lint
```

The corpus contains 1,093,691 cases. All passed the last full run, with independent
spectral checks and endpoint reconstruction at `1e-8`. Passing the corpus is
our numerical acceptance criterion. Universal coverage still needs a proof.

`tests/cases.bin` stores nine little-endian `f64` values per row:
`[c0,c1,c2,g0,g1,g2,t0,t1,t2]`, without a header. To regenerate it, run
`python3 tests/generate.py` with NumPy and SciPy installed. The solver and corpus
tests run in Rust.

To compare a proposed algorithm:

1. Replace `candidate` in `tests/solver.rs` with the proposed algorithm.
2. Run the comparison:

   ```sh
   cargo test --release --test solver compare_candidate -- --ignored --nocapture
   ```

The comparison reports solver time and the cases each algorithm passes or fails.
It checks the returned matrices independently of the algorithm's own residual.

GULPS pins a commit of this repository as a Git submodule.
