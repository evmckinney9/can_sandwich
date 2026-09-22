# can_sandwich

Depth-two two-qubit realization for gulps. Given the monodromy coordinates
of a gate `C`, a prefix class `G`, and a target class `T`, the crate returns
a real frame `O` in `SO(4)` (magic basis) with

    weyl(Can(C) · u · Can(G)) = weyl(T),      u = mb⁻¹(O),

or declines. Every returned frame is verified before it leaves the crate. The constructions use radical formulas, bounded
polynomial root isolation, and branch checks in binary64 arithmetic, followed
by bounded Levenberg–Marquardt refinement and restarts.

`gulps-core` is the only consumer. It transports the frame through the raw
ISA gate and prefix frames and re-verifies the invariant class; nothing in this
crate depends on core.

## Realization research

[Inverse Horn investigation, September 2026](inverse-horn/README.md)
records the search for a more explanatory realization method: additive
inverse Horn, hives, rank-three updates, signed singular values, quartic
completion, and proposed tridiagonal normal forms. It includes the exact
counterexamples, conditional results, independent reviews, source corrections,
and failed prototype evidence. No generic constructor or improvement to this
crate resulted from that investigation.

The separate exact-model question is settled for n=4. The reviewed R0217
sparse selector enumerates 288 row/column support records, reduces each by its
actual rank and pivots, and selects an algebraic sign cell before reconstructing
an original-role `SO(4)` frame. Its scope includes multiplicities, endpoints,
zero support, singular fibres, both target lifts, and all permitted factor
roles. The reproducible certificate is in
[`dev/research/attempts/2026-09-04-R217-sparse-selector/`](../../../dev/research/attempts/2026-09-04-R217-sparse-selector/)
and returns `PASS`.

That theorem assumes effective exact ordered-field arithmetic and algebraic
sign/root primitives. It does not provide a stable binary64 implementation;
the production cascade remains a separate engineering artifact.

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

## Numerical fallback

The public `solve` screens the algebraic result against the original spectrum.
If that fails, it refines the frame with the spectral LM implementation from
`dev/prototypes/2026-09-21-lm-sandwich/solver.rs`. A decline triggers deterministic
restarts, swapped-factor and inverse-factor retries. Successful numerical frames
still pass the production certificate and spectral screen and use `Rung::Numerical`.
The finite iteration budget can be exhausted; a decline does not prove infeasibility.

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

## Validation

See the [development instructions](../README.md#development) for building,
testing, regenerating the corpus, and comparing a prototype.

Historical measurements (stride 1, WSL, 2026-09-08; timings vary between runs):

| corpus | solved | p50 | p99 | worst |
|---|---|---|---|---|
| stratified (13,181 exact and near-Weyl strata) | 13,179 | 2.9 µs | 0.96 ms | 152 ms (a decline) |
| linspace (761,308) | all | 2.9 µs | 95 µs | 1.63 ms |
| Haar (300,000) | all | 2.18 µs | 21 µs | 0.39 ms |

The exact `RankOne31` rung now handles the registered exact `3+1` inputs
directly by grouped spectral-mass recovery and a Householder completion. In
the 2026-09-10 replay it owned 1,048 stratified rows and 16,562 linspace rows.
The two remaining stratified declines are rows 2889 and 7172. Each has one gate that is
`3 + 1` up to a phase deviation of about 1e-8 (row 2889: the fourth phase of
`G` sits 9e-9 from its repeated triple; row 7172: `C`, 3e-9), a repeated or
conjugate pair on the other gate, and a generic target. That is the band
between the exact-stratum rungs, which fire only on exact repeats, and the
generic charts, whose inner systems lose the row at that conditioning. A
snapped `3 + 1` frame misses the original problem by the deviation itself,
above the certificate, and the stabilizer-orbit closure of a snapped frame is
refuted (registry R0212-H1); a linearized correction is excluded by the
closed-form rule. A decline costs 110 to 160 ms, all in the chart tier's
exhaustive pass.

## Known non-closed-form steps

Three bounded root-bracketing steps remain and are marked in the code: the
interior rung refines an isolated sextic root by Illinois regula falsi with a
QZ rescue on accepted candidates; the chart atlas's exact pass isolates sign
changes of a chart discriminant by bisection; and the near-`3+1` leaf locates
sign changes of a cubic along an ellipse by sampling and bisection. All three
are certificate-gated and bounded; none is a Newton or descent step.
