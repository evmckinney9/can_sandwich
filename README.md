# can_sandwich

Depth-two two-qubit realization for gulps. Given the monodromy coordinates
of a gate `C`, a prefix class `G`, and a target class `T`, the crate returns
a real frame `O` in `SO(4)` (magic basis) with

    weyl(Can(C) · u · Can(G)) = weyl(T),      u = mb⁻¹(O),

or declines. Every returned frame is verified before it leaves the crate. No
step iterates to convergence: the constructions are radical formulas, bounded
polynomial root isolation, and exact branch tests.

`gulps-core` is the only consumer. It transports the frame through the raw
ISA gate and prefix frames and re-verifies the invariant class; nothing in this
crate depends on core.

## API

```rust
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution;
```

`Solution` carries the frame `o`, the rung that produced it, and the
certificate residual. Core declines `Unsolved`.

## Master object and certificate

`M = D_C · O · Λ · Oᵀ · D_C` with `D_C = mb(Can(C))` and `Λ = mb(Can(G))²`,
both diagonal. The target fixes the elementary symmetric functions `e₁, e₂`
of `M` (`e₄` is determined, `e₃ = conj(e₁) e₄`). Matching uses the symmetric
functions, so routing does not degrade at repeated spectra.

Acceptance is rootwise. For a simple target the annihilation bound on the
master's own characteristic polynomial is a complete certificate: `bound < 1e-8`
proves the master spectrum matches the target to that scale, and the frame is
already known to be `SO(4)`, so the row is accepted with no eigendecomposition.
`solution.residual` then carries that bound, a rigorous upper bound on the
spectral error rather than the exact diagonalization residual. Repeated targets
are diagonalized against the matched root blocks by real spectral projectors and
certified at `1e-8`. A frame that fails the certificate is discarded whatever
rung produced it.

## Cascade

The rungs run in order on the input as given and the first certified frame
returns. Exact strata first, then the two one-sided accelerators, then the
chart atlas. Measured on the locked corpora (2026-09-07), the chart atlas
alone solves every simple-spectrum row; `Radical` is load-bearing for 56
rows with a `3 + 1` or `2 + 1 + 1` gate, the resonance, `OnePlusThree` and
`Interior` rungs for one to three near-degenerate rows each, and the support
strata, `Klein` and `Interior` for latency (2x on linspace, 1.5x on Haar).
The `2 + 2` Plücker construction remains a separately attributed `Pair22`
rung because it is reached directly by the production cascade; it shares the
same forward certificate as the other confluent constructions.

1. Support strata from the routed root mask: signed-permutation vertices,
   one-Givens edges linear in `cos 2θ`, two-Givens faces as independent
   `2 × 2` blocks (`Vertex`, `Edge`, `Face`).
2. Klein-circulant one-sided chart: `e₁` linear in the orthostochastic
   diagonal, `e₂` one real quadratic (`Klein`).
3. Double confluence (repeated inner and repeated target value): the
   resonance construction.
4. Confluent radical strata of the genus-3 curve: skeleton and pin-pair theta
   characteristics in all four orientations (`Radical`).
5. Routed `1 + 3` peel: nine zero-entry walls, then the dense Heron-plane
   selector (`OnePlusThree`).
6. Interior three-Givens charts: the sextic eliminant of one chart, isolated
   real roots, linear back-substitution (`Interior`).
7. The row-in-plane chart atlas (`Chart`): one row of `O` in a coordinate
   2-plane, the inner solve a line cut by the Heron quartic; 48 charts × 2
   target lifts × 3 spectral roles. First the second-order wall rule: for
   every record and block end the target root nearest the split-off product
   `λ_a μ_b` gives the phase margin `m`; the block frames of the chart at that
   end on the target projected onto the wall are points of the 3×3 core
   curve, and the fibre leaves them at `t' = m/κ(R₁) − m² κ₂/κ³`, with `κ`
   and `κ₂` the exact first- and second-order phase rates of the split-off
   root (R0224-H2, R0227-H1, both verified) and `R₁` the first-order
   transverse shift of the block frame; the four candidates of smallest `|m|`
   are probed. Then the chord points, the fixed fractions, the wall pass
   through the block frames nearest a target root, and exact selection from
   each chart's discriminant.

## Modules

| file | contents |
|---|---|
| `lib.rs` | the public boundary |
| `problem.rs` | input normalization (`PreparedSandwich`), frame metrics, exact spectrum kinds |
| `cascade.rs` | rung dispatch and the certificate gate |
| `certificate.rs` | the rootwise certificate: annihilation bound, spectral projectors |
| `support_strata.rs` | vertex, edge, face, and the confluence router |
| `radical.rs` | the genus-3 curve: residue law, skeletons, pin pairs |
| `klein.rs` | Klein-circulant chart and its real quartic |
| `two_plus_two.rs` | `2 + 2` spectra in squared Plücker coordinates |
| `resonance.rs` | double-confluence construction |
| `one_plus_three.rs` | routed `1 + 3` walls and the dense selector |
| `three_givens.rs` | the three-Givens chart eliminant |
| `interior.rs` | the interior chart scan and its bracketed root extraction |
| `charts.rs` | the row-in-plane atlas and its leaves |
| `chart_precision.rs` | conditioning of a chart's affine system near clustered spectra |
| `cpoly.rs` | complex polynomial helpers, companion roots |
| `main.rs` | corpus driver (`diagnostics` feature) |

## Build and validation

The crate is a member of the `crates/` workspace.

```sh
cd crates
cargo test -p can_sandwich
cargo run --release -p can_sandwich --features diagnostics -- bench-npy corpus.npy [stride]
cargo run --release -p can_sandwich --features diagnostics -- bench-row corpus.npy ROW [reps]
```

A corpus is a little-endian C-order `(N, 3, 3)` f64 array of `[C, G, T]`
monodromy triples. `PROF=1` prints per-stage timings from the driver.

`bench_both.sh` is the standard change validation: the locked stratified,
linspace, and Haar corpora at stride 1, the public pipeline corpus, two stable
tail rows, and the line count. Corpora live under `.local/corpora` and are
materialized with `make corpora CORPUS_SOURCE=...`. `bench_realization_stress.sh`
replays freshly generated structured corpora through both the atomic solver
and the public pipeline.

Current locked numbers (stride 1, WSL, 2026-09-08; timings vary between runs):

| corpus | solved | p50 | p99 | worst |
|---|---|---|---|---|
| stratified (13,181 exact and near-Weyl strata) | 13,179 | 2.9 µs | 0.96 ms | 152 ms (a decline) |
| linspace (761,308) | all | 2.9 µs | 95 µs | 1.63 ms |
| Haar (300,000) | all | 2.18 µs | 21 µs | 0.39 ms |

The exact `RankOne31` rung now handles the registered exact `3+1` inputs
directly by grouped spectral-mass recovery and a Householder completion. In
the 2026-09-10 replay it owned 1,048 stratified rows and 16,562 linspace rows.
The two remaining stratified declines are rows 2889 and 7172, and public-pipeline row
2876 hands the solver the same triple as stratified row 2889, so the whole
residue of every locked corpus is these two triples. Each has one gate that is
`3 + 1` up to a phase deviation of about 1e-8 (row 2889: the fourth phase of
`G` sits 9e-9 from its repeated triple; row 7172: `C`, 3e-9), a repeated or
conjugate pair on the other gate, and a generic target. That is the band
between the exact-stratum rungs, which fire only on exact repeats, and the
generic charts, whose inner systems lose the row at that conditioning. A
snapped `3 + 1` frame misses the original problem by the deviation itself,
above the certificate, and the stabilizer-orbit closure of a snapped frame is
refuted (registry R0212-H1); a linearized correction is excluded by the
closed-form rule. A decline costs 110 to 160 ms, all in the chart tier's
exhaustive pass. The pipeline surfaces the row as `CompileFailure`, and
`make research-ready` fails on it.

## Known non-closed-form steps

Three bounded root-bracketing steps remain and are marked in the code: the
interior rung refines an isolated sextic root by Illinois regula falsi with a
QZ rescue on accepted candidates; the chart atlas's exact pass isolates sign
changes of a chart discriminant by bisection; and the near-`3+1` leaf locates
sign changes of a cubic along an ellipse by sampling and bisection. All three
are certificate-gated and bounded; none is a Newton or descent step.
