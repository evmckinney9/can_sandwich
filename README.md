# can_sandwich

can_sandwich is the Rust solver used by
[GULPS](https://github.com/evmckinney9/gulps), a quantum circuit synthesis
package that finds minimum-cost decompositions of two-qubit unitaries using
a supplied set of gates. Once GULPS has selected a gate sequence, it needs
to find the single-qubit gates that make the sequence implement the target.

That construction reduces to the problem solved here: find a local gate
(a tensor product of two single-qubit gates) that, placed between two given
two-qubit gates, produces a target nonlocal action. We can solve this numerically, but to the
best of my knowledge, finding a practical exact construction for every feasible
input remains an open problem. In the magic basis, the task is to construct
a real matrix $O\in SO(4)$ with a prescribed product spectrum, a form of the
inverse multiplicative Horn problem.

The current solver passes the corpus, and we'd like to replace its numerical
search with a construction whose proof covers boundary cases and repeated
eigenvalues as well as generic inputs. The existence of a witness is already
established, and general real-algebraic algorithms can construct one in
principle. The [mathematical account](docs/research.md) explains what those
results provide and what we still want to improve.

I think AI models may help us find that construction because we can check the
matrices their proposed algorithms produce. For now, we're willing to let the
implementation be largely vibe-coded as long as it's fast and correct over
the corpus. That's sufficient for GULPS, even though it isn't a proof that
the solver always works. We moved this work into its own crate and repository
so researchers can keep trying alternatives without changing the rest of GULPS.

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

The runner compares your algorithm with production on the same inputs and
reports failures, solver time, and spectral, orthogonality, and determinant
errors. If a case needs investigation, you can replay that row on its own.
The [researcher guide](docs/researcher.md) walks through a separate candidate
project and defines its inputs, outputs, and reports, so you can try an
algorithm without editing this repository.

## Use the solver

[`solve`](src/lib.rs) takes monodromy triples `c`, `g`, and `t` for the left
gate, right gate, and target, and returns the middle matrix `O`. The triples
use dimensionless coordinates whose conversion to angles is defined below.

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

The current solver uses binary64 arithmetic and combines algebraic constructions
with bounded Levenberg–Marquardt refinement and restarts. Because the search
has a fixed budget, a `None` result means it found no accepted witness within
that budget; the input may still be feasible. The [source guide](docs/architecture.md)
explains how the constructions fit together and where to find the internal
diagnostics, which are available through the `diagnostics` feature.

## Build and check

Use Rust 1.97 or newer.

```sh
make build
make test
make lint
```

The corpus contains 1,093,691 cases in `tests/cases.bin`, which the tests use
to check returned matrices and endpoint reconstruction independently at `1e-8`.
The comparison runner uses the same spectral checker and reports the measured
errors as well as pass counts, so a loss of accuracy remains visible even
when both algorithms pass.

The corpus stores nine little-endian `f64` values per row, without a header:
`[c0,c1,c2,g0,g1,g2,t0,t1,t2]`. To generate it from the original construction
code, run `python3 tests/generate.py` with NumPy and SciPy installed. Once the
data exists, the solver and corpus checks run entirely in Rust.

Because GULPS records a specific solver commit through a Git submodule,
changes here won't affect other GULPS checkouts until they use that commit.
The GULPS contributor guide explains
[how to update and test the solver dependency](https://github.com/evmckinney9/gulps/blob/main/.github/CONTRIBUTING.md#upgrade-the-solver).
