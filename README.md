# can_sandwich

GULPS depends on what is, to the best of my knowledge, an open problem in
mathematics: a practical, proved construction for the realization problem below.
We can do it numerically, but it seems like current AI models are somewhat close
to being able to close the problem.

Because this is open research and changing rapidly, we moved it into its own
Rust crate and separate project. It is a well-posed problem for models because
it is easy to check whether a proposed solution is correct.

We are letting this be extremely vibe-coded for the time being. Our contract
is simple: the produced algorithm must be fast and correct over the corpus.
That is sufficient for GULPS. It is not a proof that the mathematical problem
is solved.

We want to keep improving it. That is why this project has a corpus, an
independent checker, and [tools to compare a proposed algorithm against the
production solver](#build-test-and-compare-algorithms). Bring a better algorithm,
run it against the same cases, and show what gets faster or more reliable.

## The problem

Given the spectra of three SU(4) matrices, construct matrices in those conjugacy
classes such that $AB=C$. The multiplicative Horn inequalities tell us whether
such matrices exist. The inverse problem asks us to produce them.

The research goal is a finite construction specialized to dimension four,
with proofs of coverage and termination, including boundary points and repeated
spectra. The unrestricted problem permits any SU(4) conjugating witness.
The current implementation uses algebraic constructions and bounded numerical
optimization. It is not a proved exact constructor for every feasible input.

The [problem formulation](docs/research.md#the-unrestricted-problem) gives the
precise equations and assumptions. The [research contract](docs/research.md#exact-input-and-output)
sets the exact input model, established algebraic baseline, and proof obligations.

## What the crate provides

GULPS needs a local two-qubit gate between two canonical gates. In the magic
basis, that local gate is a real SO(4) matrix. The solver takes three monodromy
triples `c`, `g`, and `t`: the left gate, the right gate or prefix, and the target.
It constructs a real frame that realizes the target class up to global phase.

This application has two distinctions from the unrestricted research problem:
it requires a real witness, and it permits either central-sign branch of the
target spectrum. GULPS tracks the phase separately. The
[coordinate conventions and factorization](docs/research.md#gulps-coordinates-and-endpoint-factors)
make that relationship explicit.

```rust
use nalgebra::Matrix4;

pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<Matrix4<f64>>;

pub fn solve_with_factors(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<(Matrix4<f64>, Matrix4<f64>, Matrix4<f64>, f64)>;
```

`solve` returns the middle frame `O`. `solve_with_factors` returns
`(O, L, R, phase)`, including both endpoint frames. It reuses the verification
eigenbasis so GULPS does not need a second endpoint eigendecomposition.
`None` means the solver declined. It does not prove infeasibility.
The crate does not evaluate the Horn inequalities.

## How it works and what is checked

The solver prepares the spectra, tries inexpensive constructions and algebraic
search, then refines rejected candidates or tries bounded Levenberg–Marquardt
restarts. It checks returned frames against the original spectra and checks
endpoint reconstruction when factors are requested.

All arithmetic is binary64. Acceptance checks are floating-point tests, not
interval certificates or exact proofs. Finite search budgets can expire.
The corpus contains 1,093,691 cases. All passed the last full run, including
independent spectral checks and endpoint reconstruction at `1e-8`.
That is regression evidence, not a completeness theorem.

The [source guide](docs/architecture.md) maps the implementation and explains
the intended redesign. Research diagnostics require the `diagnostics` feature.

## Build, test, and compare algorithms

Rust 1.97 or newer is required. Run these commands in this repository:

```sh
make build
make test
make lint
```

The acceptance corpus is `tests/cases.bin`. Each row stores nine little-endian
`f64` values: `[c0,c1,c2,g0,g1,g2,t0,t1,t2]`. There is no header.
Regenerate it from the construction code with `python3 tests/generate.py`
(requires NumPy and SciPy). The Rust solver and test runner do not require Python.

To compare an algorithm, replace `candidate` in `tests/solver.rs` and run:

```sh
cargo test --release --test solver compare_candidate -- --ignored --nocapture
```

The comparison reports solver time and fixed or regressed corpus rows.
Check correctness independently of the candidate's own residual. Keep regression
cases in the corpus and measure the cost of each retained special case.

GULPS consumes this independent crate through a pinned Git submodule.
Solver changes and the GULPS integration pin have separate lifecycles.
