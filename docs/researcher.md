# Propose an algorithm

You can develop a candidate in a separate Rust project and pass its solve
function to the corpus runner. Given a path to the corpus, the runner calls
both your candidate and the production solver, then checks their results
independently.

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

The dependency path is relative to your candidate's `Cargo.toml`; if you're
using the solver checkout inside GULPS, set it to
`/path/to/gulps/crates/can_sandwich`. Your candidate project's release profile
applies to both algorithms in the comparison.

Put this in your candidate's `src/main.rs`:

```rust
use nalgebra::Matrix4;

fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Matrix4<f64>> {
    // This deliberately compares production with itself.
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

This first run compares production with itself, so every row should pass
with equal spectral errors, although the measured times will vary. Once that
works, replace the body of `solve` in your candidate project with your algorithm.
The same runner will evaluate it without changes to can_sandwich or its tests.

When sharing results, include the can_sandwich commit and any local edits,
your candidate revision, `Cargo.lock`, `rustc --version`, and the corpus SHA-256.
A path dependency uses the files in your local checkout, so another researcher
will need the same versions to reproduce your run. A clean checkout at a
recorded commit, or a Git dependency with `rev`, makes that easier.

The timing summary reports total solver time, mean time per case, median,
p95, p99, p99.9, and maximum latency with the slowest row's index. These
measure only the solver call; input loading and independent verification are
outside the timer. Each row is timed once, so the percentiles describe the
distribution across corpus inputs. The median averages the two central
measurements for an even number of rows; percentiles use nearest rank.

Repeat comparisons in separate processes and inspect the slowest rows as
well as aggregate time. A maximum can include scheduling delays or one-time
initialization. The CSV's `nanoseconds` column retains every measured call;
use `--case INDEX` to replay an outlier.

## Mathematical contract

The function takes three finite monodromy triples, in this order:

| Argument | Meaning |
|---|---|
| `c: [f64; 3]` | Left canonical gate |
| `g: [f64; 3]` | Right canonical gate |
| `t: [f64; 3]` | Target canonical gate |

The coordinates are dimensionless and determine the diagonal gate through
the following formula, with indices starting at zero:

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

The spectral equality counts repeated roots with their multiplicities, and
the chosen sign applies to all four target roots together. Both signs are
allowed because GULPS tracks global phase separately.

Your function returns the middle matrix `O` in the magic basis, from which
GULPS can recover the endpoint factors `L` and `R`. This real-matrix output
requirement matters if your construction solves the unrestricted SU(4)
problem: a general complex witness needs a conversion before it can be used
by this interface.

`Matrix4[(i, j)]` means row `i`, column `j`. If your algorithm produces 16
values in row order, use `Matrix4::from_row_slice(&values)`. Nalgebra stores
matrices in column order internally, so `from_column_slice` is not equivalent.

If your method handles only a restricted family, return `None` for other
inputs so the report can measure its coverage. A decline says that your
algorithm didn't produce a witness, without making a claim about feasibility.
If you call production as a fallback, describe that combination in your results
so readers know which algorithm they are evaluating.

The corpus includes generic inputs, boundary points, and repeated spectra, so
a construction that divides by an eigenvalue gap will need to handle cases
where that gap vanishes. It also permits either global target sign. You can
use exact or higher-precision arithmetic internally, but the runner evaluates
the binary64 matrix you return; its numerical checks cannot establish an
exact construction theorem.

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
| Spectrum | Smallest maximum complex-root distance over all 24 bijections and both global signs | At most `5e-14` (`SPECTRAL_CHECK`) |
| Orthogonality | Largest absolute entry of $O^TO-I$ | At most `5e-14` (`FRAME_CHECK`) |
| Determinant | $|\det O-1|$ | At most `5e-14` (`FRAME_CHECK`) |

A returned matrix passes only if it meets all three bounds. The spectrum is
computed independently of the solver. $M=D(c)^2OD(g)^2O^T$ is unitary up to
the orthogonality defect of $O$, so it is normal up to rounding. The checker
shifts $M$ by its mean eigenvalue and scales it to unit largest entry. The
eigenvectors $V$ of a Hermitian projection
$(e^{-i\theta}M+e^{i\theta}M^*)/2$ then diagonalize it, except within groups
whose projected values nearly coincide. Of eight angles $\theta$, the one with
the widest gap between projected values is used. Each group's diagonal block
of $V^*MV$ is solved the same way, and a block of two in closed form. A
Hermitian eigensolve always converges, and the couplings dropped between
groups move an eigenvalue only at second order. This replaced a complex Schur
eigensolve on 2026-09-30. That solve overstated errors near repeated roots, up
to `1.64e-13` on the corpus, and did not converge on some stress rows.

On the corpus, the measured spectral maximum is `1.78e-14`. It is within
`1e-15` of an 80-digit evaluation on the worst rows, and agrees with LAPACK
`zgeev` to `3e-15` on every row checked. `SPECTRAL_CHECK` keeps a factor of
2.8 above it. `FRAME_CHECK` keeps a factor of four above the corpus maximum
of `1.27e-14` over the orthogonality, determinant, and endpoint terms. To
check a run at higher precision, write its matrices with `--witnesses PATH`
and run `python3 tests/check80.py tests/cases.bin PATH`. The script recomputes
at 80 digits every row whose binary64 error exceeds `1e-14`.

Every corpus row is feasible up to the rounding of its stored inputs.
`tests/margins.py` verifies this exactly, apart from rows 1,096,187 to
1,096,228: they are planted in non-alcove coordinates and have explicit
witnesses. The
runner still checks one necessary rank-two Horn inequality, for corpora that
include infeasible rows. For ordered alcove coordinates, let
`a = c0 + c2 + g0 + g2`. The inequality requires
`abs(t1 + t2) <= min(a, 1-a)` for both central lifts. A violation exceeding
`1e-12` in phase turns certifies infeasibility at the root-error ceiling.
Such a row passes only when the solver returns `None`; its CSV status is
`rejected_infeasible`. Other declines remain failures. This partial certificate
does not classify every infeasible input, and uses no row indices.

For other rows, the runner labels a rejected matrix `invalid` and a `None`
result `declined`. It reports p50, p99, p99.9, and maximum errors among the
returned matrices. When a measurement is unavailable, its value is `inf`;
for example, the checker skips spectral
evaluation if the matrix already fails the orthogonality check.

The report calls a row `fixed` when production fails and the candidate passes,
and `regressed` when the reverse happens. It also counts every strict increase
or decrease in spectral error, including rows where both algorithms pass.
Since small differences can come from roundoff, use the CSV to inspect their
magnitudes before interpreting them as an improvement or regression.

After warming up both functions on up to 16 selected rows, the runner makes
one timed call to each solver per row and alternates which runs first.
The measurement includes all work inside `solve`, but excludes file loading,
verification, and reporting. Any state your candidate keeps persists across
calls, including the warmup.

For a useful speed comparison, repeat the run on an idle machine and consider
how many cases each algorithm solves. A candidate that returns `None` on hard
inputs may take less time simply because it does less work.

`--report` creates a new CSV with two records per case:

```text
case,solver,status,nanoseconds,spectrum_error,orthogonality_error,determinant_error
```

For declined cases, the error columns are blank. Choose a new report filename
for each run because the runner refuses to overwrite existing files.
Exit code `0` means the candidate passed every selected row, `1` means it
failed at least one, and `2` means the runner encountered an error. Failures
in production appear in the report without preventing candidate evaluation.

The harness calls Rust functions in the same process, so a panic or hang in
either solver can stop the comparison. If your experiment needs isolation
or a time limit, run the candidate executable inside your usual sandbox or
process timeout.
