# can_sandwich

GULPS needs to construct two-qubit circuits with a specified nonlocal action.
This leads to an inverse multiplicative Horn problem: given feasible spectra
for $A,B,C\in SU(4)$, construct matrices with those spectra such that $AB=C$.

Numerical methods already let us do this for GULPS. We also want a practical
exact construction with a proof that covers every feasible input, including
boundary cases and repeated eigenvalues. To the best of my knowledge, that
remains an open research problem. I think current AI models may be close to
finding such a construction. We can check the matrices they produce directly,
which makes it possible to test their proposals against a large corpus.

We moved the solver into its own Rust crate and Git repository because the
research changes quickly. Much of the implementation is model-generated and
still experimental. For GULPS, we require it to be fast and correct over the
corpus. This project contains the corpus and comparison tools so we can test
proposed algorithms against the production solver as the research develops.

The [research page](docs/research.md) gives the formal problem, the exact input
model, and the proof requirements. It also explains what general algebraic
methods already provide and what we hope to improve for dimension four.

## GULPS interface

GULPS needs a local gate between two canonical two-qubit gates. In the magic
basis, this is a real SO(4) matrix. The unrestricted research problem allows
any SU(4) witness. GULPS also tracks global phase separately and permits either
central-sign branch of the target spectrum. The
[coordinate conventions](docs/research.md#gulps-coordinates-and-endpoint-factors)
define this application and the API inputs and outputs.

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
