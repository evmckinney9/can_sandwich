# can_sandwich

can_sandwich is the Rust solver used by
[GULPS](https://github.com/evmckinney9/gulps), a quantum circuit synthesis
package that finds minimum-cost decompositions of two-qubit unitaries using
a supplied set of gates. GULPS selects a sequence of two-qubit gates, then
finds the single-qubit gates needed to implement the target unitary. It reduces
this second step to depth-two problems: given `CAN(G)`, `CAN(C)`, and `CAN(T)`,
find the single-qubit gates `A` and `B` in the circuit below.

```text
     ┌─────────┐┌───┐┌─────────┐            ┌─────────┐
q_0: ┤0        ├┤ A ├┤0        ├       q_0: ┤0        ├
     │  CAN(G) │├───┤│  CAN(C) │   ≈        │  CAN(T) │
q_1: ┤1        ├┤ B ├┤1        ├       q_1: ┤1        ├
     └─────────┘└───┘└─────────┘            └─────────┘
```

`CAN` denotes a canonical two-qubit gate, and the unknown local gate `A ⊗ B`
is sandwiched between `CAN(G)` and `CAN(C)`. The circuits run from left to
right, with `≈` meaning equality up to single-qubit gates before and after
the circuit and a global phase.

> [!IMPORTANT]
> The contents of this repository are largely AI-generated. Once there is enough mathematical clarity on the problem, this code can be rewritten and integrated into the gulps package properly. For now, we handle with exceedingly large amount of special case handling, heuristics, and other carefully engineered *slop*. The correctness of AI-generated code is the large corpus of unit tests which are easily verified via multiplying back the factorization, so what remains is some sort of mathematically elegant or intepretable method instead.

Closed-form solutions are known for special choices of `C`, `G`, and `T`.
Numerical methods, including the least-squares approach described in the
[GULPS paper](https://arxiv.org/abs/2505.00543), can solve more general cases,
but don't always converge reliably. We'd like a practical exact construction
that works for every feasible input, including boundary cases and repeated
eigenvalues. To the best of my knowledge, finding one remains an open problem.

In the magic basis, the task is to construct a real matrix $O\in SO(4)$ with
a prescribed product spectrum, a form of the inverse multiplicative Horn
problem. Existence is already established for feasible inputs, and general
real-algebraic algorithms can construct a witness in principle. The
[mathematical account](docs/research.md) gives the formulation, known results,
and requirements for a practical exact construction.

I think AI models may help us find that construction because we can check
the matrices their proposed algorithms produce. For now, we're willing to
let the implementation be largely vibe-coded as long as it's fast and correct
over the test corpus. The current solver meets that requirement, which is
sufficient for GULPS but doesn't prove correctness for every feasible input.
We moved this work into its own crate and repository so we can keep trying
alternatives without changing the rest of GULPS.

## Try a different algorithm

Write a function with this signature in your own Rust project:

```rust
fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3])
    -> Option<nalgebra::Matrix4<f64>>;
```

With the `corpus` feature enabled on your `can_sandwich` dependency, call
`can_sandwich::corpus::compare(solve)` from `main` and run the comparison:

```sh
cargo run --release -- --corpus /path/to/can_sandwich/tests/cases.bin --report results.csv
```

The runner calls your algorithm and the production solver on the same inputs,
then reports failures, total and per-case solver timings, and errors in the spectrum,
orthogonality, and determinant. You can replay individual cases to investigate
failures. The [researcher guide](docs/researcher.md) provides a complete example
project and defines the inputs, outputs, and report format.

## Use the solver

[`solve`](src/lib.rs) takes dimensionless monodromy triples `c`, `g`, and `t`
for the left gate, right gate, and target in the matrix product `D(c) O D(g)`.
It returns the middle local gate as a real matrix `O` in the magic basis;
`solve_with_factors` also returns the endpoint gates and global phase.

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

The solver uses binary64 arithmetic, combining algebraic constructions with
Levenberg–Marquardt refinement and restarts. It returns `None` if the inputs
are nonfinite or it cannot produce a verified result within its search budget;
the input may still be feasible. The [source guide](docs/architecture.md)
describes the implementation and the internal measurements available with
the `diagnostics` feature.

## Build and check

With Rust 1.97 or newer installed, run:

```sh
make build
make test
make lint
```

The tests check returned matrices and endpoint reconstruction against
1,093,691 cases in `tests/cases.bin`, using an independent checker with a
`1e-12` tolerance. Four historical near-feasible inputs violate necessary
Horn inequalities; the checker requires the solver to reject them. The corpus
bytes are unchanged. The comparison runner uses the same spectral checker and
reports measured errors alongside pass counts, including differences between
algorithms that both pass.

Each corpus row stores nine little-endian `f64` values without a header:
`[c0,c1,c2,g0,g1,g2,t0,t1,t2]`. The file is included in the repository, so
running the solver and its tests requires only Rust. To regenerate the cases
from their construction code, run `python3 tests/generate.py` with NumPy and
SciPy installed.

GULPS uses a specific commit of this repository as a Git submodule. To use a
newer solver commit in GULPS, follow its contributor guide's instructions for
[updating and testing the solver dependency](https://github.com/evmckinney9/gulps/blob/main/.github/CONTRIBUTING.md#upgrade-the-solver).
