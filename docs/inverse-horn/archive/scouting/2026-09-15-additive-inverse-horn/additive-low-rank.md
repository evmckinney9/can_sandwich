# How the low-rank additive constructions work

Date: 2026-09-15. Bounded independent source review. Context reuse disclosed:
this reviewer previously reviewed geometric/parabolic sources and the R386
PDE conventions, but did not author either additive construction. No new
hypothesis, computation, or registry promotion. Existing literature IDs:
`GIFT-WOERDEMAN-RANK2` and `CAO-WOERDEMAN-1712.02922`.

## Gift–Woerdeman: build two updates that fit together

[The 2025 paper](https://journals.uwyo.edu/index.php/ela/article/download/8697/7007/24669)
constructs **real symmetric matrices of arbitrary size** when one summand
has rank two. Start with `A=diag(lambda)` and write

```text
B = b1 v v^T + b2 w w^T,     ||v||=||w||=1, v^T w=0.
```

Choose the intermediate spectrum `rho` after the first update. Each update
then has explicit rank-one reconstruction. Orthogonality ensures that `B`
has the requested eigenvalues, not merely the requested trace.

Theorem 4.1 selects `rho` by a trace equation, two interlacings, and

```text
sum_j r_j sqrt(|f(rho_j) h(rho_j)|) / |g'(rho_j)| = 0,
r_j in {-1,+1},
f=product(x-lambda_i), g=product(x-rho_i), h=product(x-nu_i).
```

For example, the first update uses

```text
v_i² = -g(lambda_i)/(b1 f'(lambda_i)).
```

These are exact conditions and reconstruction formulas. Section 4.2 finds
the intermediate spectrum numerically with MATLAB `fmincon`, trying sign
choices. It supplies no direct exact selector.

Sections 5–7 and Appendix Algorithm 2 handle repeated input eigenvalues and
shared input/output eigenvalues through reductions and additional angle
freedom. Both positive and indefinite updates are covered; negative updates
use negation. “General” means the complete rank-two case, including these
degeneracies. Section 9 notes that scalar shifting consequently covers every
`3×3` additive problem.

## Cao–Woerdeman: build a pencil through moment data

[The 2017 paper](https://arxiv.org/pdf/1712.02922) uses

```text
P=(A+B)/2, R=(A-B)/2, p(x,y)=det(xI-P-yR).
```

The three requested spectra prescribe `p` at `y=0,1,-1`. Newton sums of its
roots form a Hankel matrix polynomial `H(y)`. Positive semidefiniteness for
every real `y` expresses real-rootedness; Newton identities impose
compatibility. Theorem 2.1 gives an exact equivalence with Horn feasibility.

Section 3 turns the `n=3` selection into an SDP: a positive semidefinite
Gram matrix satisfies linear constraints. After spectral factorization
`H=Q*Q`, conjugating the companion matrix by `Q` recovers a linear
Hermitian pencil and hence `A,B`. This makes the compatibility conditions
convex in the implemented dimension.

The implementation uses CVX, approximate rank-three factorization, and final
Hermitian symmetrization. Its output is **Hermitian**, although the existence
theorem also includes real symmetric matrices. The displayed reconstruction
requires simple spectra for `A` and `B`; repeated eigenvalues can make the
needed factors singular. The theorem's existence scope is broader than
that implementation guard.

For `n>3`, additional compatibility constraints become quadratic. Thus the
paper gives a general structural formulation and an implemented convex
`3×3` algorithm, not an arbitrary-dimension pure-SDP constructor.

## What these constructions teach

Both approaches identify information that endpoint inequalities leave
implicit. The update method chooses an intermediate spectrum **and** makes
the two update vectors compatible. The pencil method chooses mixed spectral
data **and** enforces identities that make it one matrix pencil.

This is useful constructive progress: after those selections, matrices can
actually be recovered. The next question is consequently about the selector
and its numerical behavior, rather than whether feasible matrices exist.

Rank two describes the perturbation, not the ambient matrix dimension.
For a general `4×4` problem, subtracting one eigenvalue from the second
summand leaves rank at most three; it reaches rank at most two when that
eigenvalue was repeated. Therefore the rank-two method also directly serves
a meaningful four-dimensional stratum. Extending sequential updates further
requires preserving all pairwise orthogonality conditions.

## Reproduction and scope

Checked Gift–Woerdeman: Sections 2.3, 4.1–4.2, 5–7, 9, Algorithm 1, and
Appendix Algorithm 2. Checked Cao–Woerdeman: Theorem 2.1, Corollary 2.2, and
Section 3, including the stated multiplicity limitation. No claim about
multiplicative transport, certified runtime, or exact termination was inferred.
