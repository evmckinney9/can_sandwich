# Rank-two inverse Horn: selecting the intermediate spectrum

2026-09-15. Additive research checkpoint. Working authority:
`dev/research/attempts/2026-09-15-R392-rank2-sign-change/` (attempt record,
proof, certificate, independent review). This page is the portable summary.
No multiplicative claim, no change to `can_sandwich`.

## The construction studied

Gift and Woerdeman build a real solution of the rank-two Horn problem
`A=diag(lambda)`, `B=d1 vv^T+d2 ww^T`, `v` orthogonal to `w`,
`spec(A+B)=nu` in two rank-one updates. Everything after the intermediate
spectrum `rho=spec(A+d1 vv^T)` is closed form (Gift 2025, Theorem 4.8,
Algorithm 4.8). The spectrum `rho` must satisfy the trace equation, two
interlacings, and a signed balance

    sum_j r_j sqrt(|f(rho_j) h(rho_j)|) / |g'(rho_j)| = 0,   r_j in {-1,1},

with `f,g,h` the polynomials with roots `lambda,rho,nu`. The source proves
such `rho` exists by reading it off an existing witness and finds it with
`fmincon` over sign patterns (Section 4.3.2). That is the first unresolved
step of the construction, and the one attacked here.

## What the attempt establishes

Assume `lambda` simple, `d1>=d2>0`, and no `lambda_i` equal to any `nu_k`.
The two interlacings and the trace make the admissible `rho` a box cut by a
hyperplane, `P`. Each term `m_j(rho)` of the balance vanishes exactly when
`rho_j` sits at an endpoint of its interval that is a root of `f` or `h`.
So at a vertex of `P` all terms vanish except possibly the one at the free
coordinate (and the first coordinate when it is pinned at a non-root value).

- **Lemma A (construction).** If two vertices of `P` have disjoint sets of
  nonzero terms, fix the signs to make the balance positive at one and
  negative at the other; bisection along the segment between them finds a
  zero, and Algorithm 4.8 then reconstructs `v,w`. A vertex with no nonzero
  term is a block witness directly.
- **Theorem B (coverage).** Horn feasibility forces `P` to touch every
  upper face: missing a face unfolds into a violated Pieri-type Horn
  inequality (vertical-strip Littlewood-Richardson triple). A vertex on the
  `nu_1` face and a vertex on the face of its free coordinate then have
  disjoint supports. So feasibility is equivalent to Weyl plus `P` nonempty
  plus the face-touching inequalities (independent review: accepted with
  textual corrections, applied in the attempt record).

The bisection's residual is the actual defect: for any `rho` in `P` the sum
spectrum is exact and only `v^T w` is off, which perturbs `spec(B)` by an
explicit two-by-two formula.

## Worked example (Gift's Example 4.10)

`lambda=(8,4,-8)`, `(d1,d2)=(12,6)`, `nu=(16,3+sqrt57,3-sqrt57)`. `P` has
three vertices: `(16,8,-8)` with no nonzero term (the block witness
`v=e_2`, `w` in the 1-3 plane), `(5+sqrt57,8,3-sqrt57)` with support `{1}`,
and `(16,sqrt57-3,3-sqrt57)` with support `{2}`. Bisection between the last
two with signs `(+,-,+)` gives `rho=(15.48938,5.06045,-4.54983)` and an exact
witness. Gift's `fmincon` point `(14.5222,6.8053,-5.3275)` lies on the same
solution curve.

## Evidence and limits

`certificate.py` checks: all 267 unfolding patterns for `n<=6` are
vertical strips (exact); on seeded samples for `n=3..6` the vertex
condition agrees with the hive-LP feasibility oracle in all 480 chains and
480 witness-generated triples reconstruct to 3e-13. These support the
proof; they are not the proof.

    python3 crates/can_sandwich/docs/inverse-horn/rank2-selector/certificate.py

Follow-up R393 (same day) tested the direct continuation: a Miranda cell
with corners at vertices of the fixed-first-update middle slice. Refuted:
the vertices realize all eight sign octants of the three orthogonality
functions in 6 of 120 witness instances. The three-function sign model,
including the finite coincidence limit of the coupled term, was validated
at true witness chains and is reusable.

Two further same-day probes: R394 found that the first-update spectrum is
not free (grid evidence, detector has false negatives), and R395 refuted the
double-first-eigenvalue stratum, where Gift-Woerdeman's angle mechanism
solves one orthogonality equation explicitly but the stratum is empty for
generic inputs (it forces lambda_j into a Gamma window).

Scope: this is the rank-two stratum. For `n=4` that means `beta` with a
repeated smallest value after shifting. The generic `n=4` problem is rank
three: the chain polytope in `(rho,tau)` is not a box slice, the three
orthogonality conditions share their sign parameters, and a one-dimensional
sign change does not decide three equations. The next obligation, not
started, is the sign of the coupled third condition on the faces where the
two adjacent ones vanish, with a Poincare-Miranda argument replacing
bisection and an analogue of the face-touching lemma derived from Horn.
