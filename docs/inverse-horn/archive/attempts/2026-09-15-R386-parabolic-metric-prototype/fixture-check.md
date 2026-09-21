# Independent R386 fixture applicability check

2026-09-15. Reviewer did not author the fixture or metric candidate and did not
import or inspect the candidate implementation. This is a fixture/convention
check, not a seed stability, convergence, or realization theorem.

## Results

**All 72 directed QLR inequalities pass.** The smallest slack is
`0.12497656777829401`. The independent checker reconstructs the row family from
the explicit rank-two table in `crates/core/src/horn.rs` and independently
generates the cyclic rank-one/rank-three rows. Counts: 41 degree-zero,
30 degree-one, one degree-two. In the production bit convention, bit zero
selects the first ordered phase. For row `(I,J,K,d)` it checks exactly

`d - sum(a[I]) - sum(b[J]) + sum(target[K]) >= 0`.

All phase lists are decreasing and have width below one. The decimal a,b
lists sum exactly to zero. The stored target sums to `-2.5e-16`, so it is a
floating-point alcove representative, not an exact trace-zero rational
certificate. Rational slack evaluation uses the exact displayed decimal
values; this roundoff is far smaller than the minimum slack.

## A positive-degree inequality genuinely strengthens degree zero

The degree-one row with masks `(11,11,14)` (binary
`(1011,1011,1110)`, zero-based row 66 in the checker) gives, using trace zero,

`target[0] <= 1 - (a[0]+a[1]+a[3]) - (b[0]+b[1]+b[3]) = 31/50`.

The exact rational target

`x = (143,-29,-57,-57)/200`

has trace zero, decreasing order, and width exactly one. It satisfies every
degree-zero row for this fixed a,b pair, with minimum degree-zero slack zero,
but violates the displayed degree-one bound by `19/200 = 0.095`.
Thus this positive-degree inequality is not implied by the degree-zero rows
plus the alcove for this pair. The actual fixture target satisfies it; x is a
separate separation certificate, not an attempted feasible target.

A small LP supplied x. Its claimed optimality is unnecessary: subsequent
Fraction arithmetic checks all defining inequalities and the violation
exactly. No LP solver correctness assumption enters this separating-point
certificate.

## Existing finite-angle domain and planted witness

For the chosen trace-zero logs `2*pi*diag(a)` and `2*pi*diag(b)`, the
operator-norm sum divided by pi is exactly

`2*(42/100+39/100) = 81/50 = 1.62`.

These particular logs are outside R0242's strict norm-sum-below-pi domain.
This does not establish that every symmetry, lift, alternate representative,
or decomposition is outside it.

The verifier-only O passes a direct numerical check:

* `||O^T O-I||F = 6.06e-16`, determinant `0.9999999999999999`.
* The roots of
  `diag(exp(pi*i*a)) O diag(exp(2*pi*i*b)) O^T diag(exp(pi*i*a))`
  match `exp(2*pi*i*target)` with maximum assigned root error `5.00e-16`.

This is observed planted feasibility evidence; no exact orthogonal witness
or rigorous eigenvalue enclosure is claimed. The checker reads this O only
for the separate verifier test. It makes no claim that the candidate's three
flags are stable or that its connection has the required holonomy.

## Reproduction and scope

```sh
timeout 45s prlimit --as=1073741824 --cpu=30 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R386-parabolic-metric-prototype/independent-fixture-check.py
```

Completed in under one second. Outputs, all 72 exact decimal-data slacks,
the rational separation witness, and hashes of fixture, verifier-only plant,
and Horn source are in `independent-fixture-check.json`.
Only independent review/checker artifacts were added; candidate code and
fixture data are unchanged. No global claim or registry promotion follows.
