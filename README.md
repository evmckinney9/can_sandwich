# can_sandwich

GULPS needs to put a local gate between two given two-qubit gates to produce a
target nonlocal action. In the magic basis, the task is to find a real matrix
$O\in SO(4)$ with a prescribed product spectrum. This is a constructive
multiplicative Horn problem.

We have a numerical solver that works over the corpus. We want a practical
exact construction that covers every feasible input, including repeated
eigenvalues and boundary cases. To the best of my knowledge, that practical
construction remains an open problem. Existence is established, and general
real-algebraic algorithms can construct exact witnesses in principle. The
[mathematical account](docs/research.md) explains the distinction and cites the
relevant results.

I think this is a good problem for AI-assisted research: we can check the
matrices an algorithm produces. For now, much of this solver is vibe-coded.
Our requirement for GULPS is that it stays fast and correct over the corpus.
That is enough to use it, though it does not prove that the algorithm always
works. The solver lives in its own crate and repository so we can keep working
on it independently.

## Try a different algorithm

Write a function with this signature in your own Rust project:

```rust
fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<nalgebra::Matrix4<f64>>;
```

Enable the `corpus` feature on your `can_sandwich` dependency and call
`can_sandwich::corpus::compare(solve)` from `main`. Then run:

```sh
cargo run --release -- --corpus /path/to/can_sandwich/tests/cases.bin --report results.csv
```

The runner compares your algorithm with production on the same inputs. It
reports failures, solver time, and spectral, orthogonality, and determinant
errors. You can replay any row. No edits to this repository are needed.

The [researcher guide](docs/researcher.md) gives a complete external-project
example, the input and output contract, and the report format. Start there
if you want to propose an algorithm.

## Use the solver

The inputs `c`, `g`, and `t` are monodromy triples for the left gate, right
gate, and target. They are dimensionless coordinates, not angles in radians.
[`solve`](src/lib.rs) returns the middle matrix `O`.

```rust
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<nalgebra::Matrix4<f64>>;

pub fn solve_with_factors(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<(nalgebra::Matrix4<f64>, nalgebra::Matrix4<f64>,
               nalgebra::Matrix4<f64>, f64)>;
```

`solve_with_factors` returns `(O, L, R, phase)` for
$D(c)OD(g)=e^{i\,\mathrm{phase}}LD(t)R$, with all three matrices in $SO(4)$.
The [coordinate definitions](docs/research.md#gulps-coordinates-and-endpoint-factors)
specify $D$ and the allowed target signs.

The current solver combines algebraic constructions with bounded
Levenberg–Marquardt refinement and restarts. It uses binary64 arithmetic.
`None` means it found no accepted witness within its search budget. It is
not a proof of infeasibility. The [source guide](docs/architecture.md) explains
the implementation. Internal diagnostics use the `diagnostics` feature.

## Build and check

Use Rust 1.97 or newer.

```sh
make build
make test
make lint
```

The corpus contains 1,093,691 cases in `tests/cases.bin`. Tests check the
returned matrices and endpoint reconstruction independently at `1e-8`.
The comparison runner uses the same spectral checker. Acceptance thresholds
are fixed, and the reports also expose error changes below those thresholds.

The corpus stores nine little-endian `f64` values per row, without a header:
`[c0,c1,c2,g0,g1,g2,t0,t1,t2]`. To generate it from the original construction
code, run `python3 tests/generate.py` with NumPy and SciPy installed. Python
is only needed for generation. The solver and corpus checks run in Rust.

GULPS uses a specific commit of this repository as a Git submodule. To use a
newer solver commit, follow the
[Upgrade the solver](https://github.com/evmckinney9/gulps/blob/main/.github/CONTRIBUTING.md#upgrade-the-solver)
instructions in the GULPS contributor guide.
