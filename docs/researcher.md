# Propose an algorithm

Keep your candidate in a separate Rust project. The comparison harness needs
one function and a path to the corpus. It runs the production solver and your
candidate, then checks both results independently.

## First run

Clone `can_sandwich` next to your candidate project, or use an existing checkout:

```sh
git clone https://github.com/evmckinney9/can_sandwich.git
cargo new candidate
cd candidate
```

Add these dependencies to the candidate's `Cargo.toml`:

```toml
[dependencies]
can_sandwich = { path = "../can_sandwich", features = ["corpus"] }
nalgebra = "0.35"

[profile.release]
lto = "fat"
codegen-units = 1
```

The path is relative to your manifest. For a candidate outside the GULPS tree,
you can point it at `/path/to/gulps/crates/can_sandwich` instead. Both solvers
use your candidate project's build profile.

Put this in your candidate's `src/main.rs`:

```rust
use nalgebra::Matrix4;

fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Matrix4<f64>> {
    // Smoke test: this deliberately compares production with itself.
    // Replace this body with your algorithm.
    can_sandwich::solve(c, g, t)
}

fn main() -> std::process::ExitCode {
    can_sandwich::corpus::compare(solve)
}
```

Run the full comparison:

```sh
cargo run --release -- --corpus ../can_sandwich/tests/cases.bin --report results.csv
```

The smoke test should pass every row and report equal spectral errors. Timing
will vary. Then replace `solve` in **your project**. You do not need to change
can_sandwich, its tests, or its production solver.

Record the can_sandwich commit, any local diff, your candidate revision,
`Cargo.lock`, `rustc --version`, and the corpus SHA-256 alongside results.
A local path dependency follows local edits. For reproducible shared work,
use a clean checkout at a recorded commit or a Git dependency with `rev`.

## Mathematical contract

The function takes three finite monodromy triples, in this order:

| Argument | Meaning |
|---|---|
| `c: [f64; 3]` | Left canonical gate |
| `g: [f64; 3]` | Right canonical gate |
| `t: [f64; 3]` | Target canonical gate |

These coordinates are dimensionless. For zero-based indices, define

$$
D(m)=\mathrm{diag}\left(
 e^{i\pi m_1},\ e^{i\pi m_0},\
 e^{-i\pi(m_0+m_1+m_2)},\ e^{i\pi m_2}
\right).
$$

Return `Some(O)` for a finite real matrix satisfying

$$
O^TO=I,\qquad \det O=1,
$$

and, for one global sign $s\in\{-1,+1\}$,

$$
\mathrm{spec}\left(D(c)^2OD(g)^2O^T\right)
=\mathrm{spec}\left(sD(t)^2\right).
$$

Spectra are multisets. The sign applies to all four target roots together.
Repeated roots keep their multiplicities. GULPS allows the two signs because
it tracks global phase separately.

The output is the middle matrix in the magic basis. It is not the product
matrix, an eigenbasis of that product, or a general complex SU(4) witness.
You do not need to compute the endpoint factors `L` and `R`.

`Matrix4[(i, j)]` means row `i`, column `j`. If your algorithm produces 16
values in row order, use `Matrix4::from_row_slice(&values)`. Nalgebra stores
matrices in column order internally, so `from_column_slice` is not equivalent.

Return `None` when your method cannot produce a witness. A method for a
restricted family is welcome: decline other inputs and measure its coverage.
Do not call production as a fallback unless you explicitly describe the
candidate as a hybrid algorithm. `None` does not establish infeasibility.

The corpus includes generic, boundary, and repeated-spectrum inputs. Do not
assume distinct roots, a nonzero denominator, or one fixed target sign. The
corpus evaluates binary64 outputs. Exact or higher-precision internal
arithmetic is allowed, but passing these checks is not an exact proof.

The [research formulation](research.md) gives the existence theorem, an exact
input model, the three-coefficient reduction, and the remaining proof obligations.

## Read the comparison

```sh
# Replay one zero-based row; prints c, g, t before solving.
cargo run --release -- --corpus ../can_sandwich/tests/cases.bin --case 42

# Print the command-line options.
cargo run --release -- --help
```

Both algorithms receive the same inputs. The checker measures:

| Error | Definition | Acceptance |
|---|---|---|
| Spectrum | Smallest maximum complex-root distance over all 24 bijections and both global signs | At most `1e-8` |
| Orthogonality | Largest absolute entry of $O^TO-I$ | At most `1e-8` |
| Determinant | $|\det O-1|$ | At most `1e-8` |

A returned matrix must pass all three checks. The runner reports `None` as
`declined` and a rejected matrix as `invalid`. It also reports p50, p99, and
maximum errors among returned matrices. An unavailable measurement is `inf`.
For example, it skips spectral evaluation when orthogonality already fails.

`fixed` rows fail production and pass the candidate. `regressed` rows pass
production and fail the candidate. The spectral comparison counts every
strict increase or decrease, even when both results pass. Tiny changes can
be roundoff. Inspect their magnitudes in the CSV before drawing conclusions.

The runner warms up both functions on up to 16 selected rows, then alternates
which function runs first. Each selected row has one timed call per solver.
Timing excludes file loading, verification, and reporting, but includes all
work inside `solve`. Candidate state persists across calls, including warmup.
Repeat runs on an idle machine before claiming a speedup. Compare coverage
alongside time: a fast decline is not a successful solve.

`--report` creates a new CSV with two records per case:

```text
case,solver,status,nanoseconds,spectrum_error,orthogonality_error,determinant_error
```

Errors are blank for declines. Existing files are never overwritten. Choose
a new report name for each run. Exit code `0` means the candidate passed all
selected rows, `1` means candidate failures, and `2` means a runner error.
Baseline failures remain visible and do not prevent evaluation of a candidate.

Both solvers run in the same process. A panic or hang can stop the run. Use
your normal sandbox and a process timeout for experimental candidates.
This harness accepts Rust functions, not executables written in other languages.
