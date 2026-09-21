# Independent preimplementation check

Date: 2026-09-15. Scope: a bounded numerical prototype of the real parabolic
metric route, not a theorem or production algorithm. No code or experiments
were authored by this reviewer. The prior Faulk paragraph in the scouting
README matches the reviewed source scope.

## Decision

**Proceed as a finite-domain numerical diagnostic, with explicit holonomy
checks.** The proposed PDE has the correct Chern-flatness sign. The proposed
boundary exponents produce the inverse peripheral phases under the standard
positive-loop parallel-transport convention. A small PDE residual alone
cannot pass this experiment: finite-radius Dirichlet values do not fix the
connection's residues or its peripheral eigenvalues.

## 1. Fix conventions before running

Use column sections, a positive Hermitian matrix `h`, and squared norm
`v* h v`. In a holomorphic frame the Chern connection is

```text
A = h^-1 partial h = h^-1 h_z dz,
F = dbar(h^-1 partial h),
h_z=(h_x-i h_y)/2,    h_zbar=(h_x+i h_y)/2.
```

Thus `F=0` is equivalent to

```text
Delta h - (h_x+i h_y) h^-1 (h_x-i h_y) = 0.
```

The proposed relaxation has this stationary equation and the forward
Laplacian sign. The nonlinear term is Hermitian: its outer factors are
adjoints. Numerical positivity still requires a controlled step or a
positive-metric parameterization; Hermiticity alone does not ensure it.
This algebra checks the equation, not the convergence of its discretization.

Parallel transport in this convention satisfies

```text
dP/dt = - A(gamma'(t)) P,   P(0)=I.
```

For the scalar model `h=|z|^(2 mu)`, one has `A=mu dz/z`; a positive
(counterclockwise) peripheral loop has holonomy `exp(-2 pi i mu)`.
Consequently the proposed positive exponents and the source's local
`exp(+2 pi i beta)` convention must not be mixed silently.

Either use clockwise peripheral loops for the positive-exponent model and
record their ordered product relation, or explicitly change the metric
exponent/weight convention with the seed-data convention. Negating weights
without checking seed semistability is not just a numerical sign repair.
At infinity use `w=1/z`: positive orientation in `w` is clockwise on a large
circle in `z`. This is another independent orientation reversal to track.

The actual unitary matrix at a common basepoint is

```text
U = h0^(1/2) P h0^(-1/2).
```

All loops must share that basepoint and trivialization. Check their ordered
product against identity in the same convention before interpreting the
third class as the inverse target. Check each peripheral spectrum against
the registered original phase data after the declared inversions.

## 2. Concrete truncated domain and boundary data

A suitable declared domain is

```text
Omega_(epsilon,R) = {|z|<R, |z|>epsilon, |z-1|>epsilon},
0<epsilon<1/2,  R>2.
```

Let each real invertible `F_p` have columns forming the chosen weighted
flag basis, with ordering matched to the signed weight convention. Normalize
its determinant magnitude if determinant-one metric data are wanted.
For the positive-exponent convention the candidate cutoff values are

```text
h|_(|z-p|=epsilon) = F_p^-T diag(epsilon^(2 mu_pk)) F_p^-1,
h|_(|z|=R) = F_infty^-T diag(R^(-2 mu_infty,k)) F_infty^-1.
```

Trace-zero weights make the radial factors determinant one. The constant
determinant of `F_p` should not create inconsistent scalar boundary modes.
Repeated weights permit a partial flag and positive block amplitude; an
arbitrary basis inside such a block is not extra spectral input.

Real compatibility is

```text
h(x,-y)=conjugate(h(x,y)).
```

It does **not** mean that `h` is real symmetric throughout the domain.
The PDE's cross terms generally create imaginary Hermitian off-diagonal
entries from noncommuting real data. A real-matrix-only relaxation would
solve a different problem. The circles and real boundary matrices above
are compatible with conjugation.

## 3. What the cutoff fixes beyond the flags

The parabolic data prescribes weighted filtrations and an adapted singularity
class, not a unique full leading metric matrix in an arbitrary flag basis.
The formula above fixes scale factors and splittings in addition to those
flags. In particular a positive amplitude commuting with the diagonal
weights is not determined by the spectral input.

At a finite cutoff this is a stronger Dirichlet problem than finding any
adapted metric. Its solution need not be the restriction of the desired
global adapted flat metric. It is nevertheless a reasonable numerical
reference boundary model if truncation effects are measured and kept
separate from failure of the geometric route.

An elementary scalar annulus explains the issue. If

```text
log h(epsilon)=2 mu log epsilon + c_inner,
log h(R)=2 mu log R + c_outer,
```

then the harmonic stationary `log h` has radial logarithmic slope

```text
a = 2 mu + (c_outer-c_inner)/(log R-log epsilon).
```

The holonomy is determined by `a/2`, not by the weight inserted at one
boundary. The effect can decay only logarithmically as the domain expands.
This shows why arbitrarily accurate finite-domain flatness does not certify
the requested spectrum, and why arbitrary leading amplitudes can visibly
bias a practical small-domain run. It does not prove that the matrix
truncation converges, or that it cannot converge.

## 4. Practical acceptance and failure attribution

For the authorized bounded prototype, report separately:

- metric positivity, Hermiticity, conjugation symmetry, and PDE residual;
- actual transported unitarity, all three original peripheral spectra,
  ordered loop-product closure, and the reconstructed `SO(4)`/product check;
- truncation/discretization parameters, work, and a minimal sensitivity
  comparison if the runtime budget permits it.

A PDE-converged result with wrong spectra is a boundary/truncation failure
of this prototype, not a realization. A correct accepted frame is a valid
observed result even if the metric approximates no independently certified
continuum solution. Failure on one cutoff/boundary choice does not exclude
the real parabolic route; it can still support a practical no-go for that
specific discretization and resource budget.

The existing exact triangle-to-frame map `R0052-H1` supplies the downstream
algebraic interface once a correctly ordered real holonomy triangle has
been obtained. Do not substitute arbitrary spectral representatives for
the common-frame holonomies.

## Source boundary

The conventions use the Chern connection and adapted-metric framework
reviewed in Biswas–Schaffhauser, arXiv:1806.09782, Section 3.2. That source
proves invariant adapted metric/holonomy existence for correctly prepared
polystable real parabolic objects; it does not prescribe the displayed
finite-circle Dirichlet values or a numerical convergence theorem for this
prototype. The PDE and scalar sign checks above are local algebra checks,
not a new global result.
