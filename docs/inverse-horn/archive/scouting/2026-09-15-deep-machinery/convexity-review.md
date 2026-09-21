# Donaldson-functional convexity: source scope

Date: 2026-09-15. Five-minute source check. No new theorem, proof, experiment,
or registry promotion.

## Checked statement

[Mitchell Faulk, arXiv:2202.08885v1](https://arxiv.org/pdf/2202.08885),
Theorem 1, assumes an **indecomposable holomorphic vector bundle over a
compact Kähler orbifold**. It equates stability, existence of a
Hermitian–Einstein metric, and properness of the Donaldson functional for
each reference metric.

Proposition 31 states, for `H_t=K exp(t s)`, with `s` K-self-adjoint and
integral trace zero,

```text
d² M_K(H_t)/dt² = 2 integral_X |dbar s|²_(H_t) omega^n/n! >= 0.
```

Proposition 19 identifies critical points with Hermitian–Einstein metrics
and states that the functional decreases along the heat flow, with derivative
minus the squared curvature residual. Definition 20's properness is a
specific estimate **along that heat-flow solution**:

```text
sup_X |s_t|_K <= C1 + C2 M_K(K exp(s_t)).
```

The constants are existential. This is not an explicit global
strong-convexity constant, discretization bound, or finite stopping estimate.

## Relevance to the proposed construction

The proposed use is legitimate as **geometric motivation with a conditional
source bridge**: once spectral data has selected a stable real parabolic
object, its metric variable has a functional with a nonnegative second
variation along the displayed exponential metric paths. This supplies
structure absent from an arbitrary frame residual objective. Enlarging the
variable from six frame parameters to a metric field can therefore change
the mathematical form of the navigation problem; its value is not measured
by the number of unknowns alone.

This observation does not establish a computational advantage. The original
six-dimensional construction is replaced by solving an analytic metric
problem, recovering holonomy, and certifying the resulting frame. The cost
and error of those operations remain to be bounded.

## Rational weights and the real locus

For rational parabolic weights, a compact orbifold curve with local cyclic
orders divisible by their denominators is the relevant proposed setting.
The input must include the corresponding orbifold bundle, its degree and
stability; converting signed SU(4) markings and retaining the real structure
are the obligations recorded in `real-parabolic-review.md`. The orbifold
interpretation is supported by the references discussed in
Biswas–Schaffhauser Section 3.1; it is not constructed by Faulk's theorem.
No global finite cover or denominator-independent cost is assumed here.

The following is an application inference requiring the specified compatible
real structure: invariant positive Hermitian metrics are preserved by
pointwise exponential interpolation between invariant endpoints. Thus the
displayed convexity inequality restricts to such paths in the real-invariant
metric space. Preservation of the analytic flow and existence of a compatible
metric must still be justified using the real correspondence. Faulk's paper
does not state a theorem about anti-holomorphic involutions or real parabolic
bundles. Biswas–Schaffhauser Theorem 3.6 is the checked real-metric existence
statement.

General semistable boundary data is outside the stable indecomposable
statement as written. A polystable decomposition and its automorphisms need
separate handling. Scalar metric rescaling also explains why nonnegative
second variation should not be called uniform strict convexity.

## Proof-audit limitation

This review checked the source statements and their domains, not an
independent analytic proof. There is a specific reason to retain that
distinction: the printed proof of Proposition 31 simplifies
`H_t^-1 partial_K H_t` to `t partial_K s` for `H_t=K exp(t s)`.
Such an identity for matrix exponentials needs a commutativity qualification
or an integral/commutator argument; it is not valid for arbitrary
noncommuting `s` and its derivative. The second-variation formula is the
stated source result, but this abbreviated proof is not an independently
checked certificate. A theorem-promotion packet should verify it through
the full noncommutative variation formula or another primary treatment.
This observation is not a refutation of the convexity statement.

## Handoff

Safe conclusion: **published metric-functional convexity makes this a
structured analytic route worth distinguishing from unconstrained frame
optimization.** Keep its rational compact-orbifold domain, real-compatibility
transfer, and analytic-evaluation costs explicit. Do not infer a theorem for
punctured irrational weights, a certified numerical discretization, or an
efficient global realization algorithm from this source check.
