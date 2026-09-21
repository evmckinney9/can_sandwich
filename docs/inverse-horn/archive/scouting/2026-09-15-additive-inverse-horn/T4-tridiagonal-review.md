# Independent exact review: the universal T4 ansatz fails

2026-09-15. Bounded independent review of the supplied counterexample.
Existing reviewer context reused. No numerical search, new candidate, or
novelty claim. Claim registration is the root agent's responsibility.

## Statement under review

T4 asks whether every feasible additive spectral triple of size four admits

`C'=diag(permutation(gamma))`, `A'` real symmetric tridiagonal,
`B'=C'-A'`,

with the prescribed spectra alpha,beta,gamma. The supplied example refutes
this statement for every one of the 24 target permutations. The same proof
also refutes the complex Hermitian tridiagonal version of this particular
ansatz.

## Exact feasible triple

Set `C=diag(3,1,-1,-3)`, `A=J4-I4`, where J4 is the all-ones matrix, and
`B=C-A=C+I4-J4`. These matrices are real symmetric, have trace zero, and
sum as required. Their spectra are

```text
alpha = (3,-1,-1,-1)
gamma = (3,1,-1,-3)
beta  = decreasing roots of p(x)=x^4-16*x^2+8*x+16.
```

The polynomial p has respectively positive, negative, positive, negative,
positive values at `-5,-2,0,2,4`. It therefore has one root in each of
`(-5,-2),(-2,0),(0,2),(2,4)`. As its degree is four, all four roots are
real and simple. An exact gcd check also gives `gcd(p,p')=1`.

## Exclusion of every tridiagonal realization

Every real symmetric A' with spectrum alpha has the form

`A'=4*v*v^T-I4`, with `v^T v=1`.

In the complex Hermitian case the same statement holds with `v*v^*` and
`v^*v=1`. For distinct i,j its off-diagonal entry is a single nonzero
constant times `v_i*conjugate(v_j)`. If A' is tridiagonal, any two nonzero
coordinates of v must therefore be adjacent along the path of four matrix
indices. A path has no three-vertex clique, so v has at most two nonzero
coordinates. This step permits zero subdiagonal entries; irreducibility is
not being assumed.

Consequently

`B'=diag(permutation(gamma))+I4-4*v*v^*`

retains at least two diagonal eigenvalues from the set `{4,2,0,-2}`: each
coordinate where v is zero gives the corresponding coordinate eigenvector
of B'. However,

```text
p(4)=48, p(2)=-16, p(0)=16, p(-2)=-48.
```

None of those four numbers is an eigenvalue of the prescribed B. This is a
contradiction for every target permutation and every real or complex v.
The universal T4 ansatz is therefore false.

## Why excluding repeated input spectra does not repair T4

Let S be the set of ordered triples admitting a T4 realization. It is closed.
To see this, take a convergent sequence of triples in S and choose witnesses
`A'_k`, with target diagonal permutations `pi_k`. Pass to a subsequence on
which the permutation is constant. The convergent alpha lists bound
`||A'_k||op`, because A'_k is symmetric/Hermitian. In fixed dimension, pass
to a further subsequence with `A'_k -> A'`. Tridiagonality and symmetry are
closed conditions. Setting `B'=diag(pi(gamma))-A'`, continuity of ordered
eigenvalues gives exactly the limiting alpha and beta. Thus the limiting
triple belongs to S. The proof works for both real and complex variants.

The excluded triple above consequently has an open neighborhood disjoint
from S. There are feasible all-simple triples arbitrarily close to it.
An explicit continuous path of actual witnesses is

`A_epsilon=A+epsilon*C`, `B_epsilon=B-epsilon*C`, with C fixed.

For every epsilon>0,

`A_epsilon=J4+diag(-1+3epsilon,-1+epsilon,-1-epsilon,-1-3epsilon)`.

The four diagonal entries are distinct, and the all-ones rank-one update
has a nonzero component in every diagonal eigendirection. Its secular
equation has one strictly simple root between each consecutive pole and
one above the largest pole, so A_epsilon has simple spectrum. B_epsilon
has simple spectrum for all sufficiently small epsilon because B already
does. C remains simple, and the traces remain zero.

For sufficiently small positive epsilon the resulting feasible all-simple
spectral triple stays in the open neighborhood excluded from S. Hence the
claim that every feasible all-simple triple admits T4 is also false. This
argument gives existence of such a neighborhood and perturbations; it does
not claim an explicit numerical radius or a named rational epsilon bound.

## Reproduction and scope

```sh
timeout 45s prlimit --as=1073741824 --cpu=30 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python crates/can_sandwich/docs/inverse-horn/archive/scouting/2026-09-15-additive-inverse-horn/check-T4-tridiagonal-counterexample.py
```

The independent checker verifies the exact characteristic polynomials,
four excluded eigenvalue values, root brackets, path-support combinatorics,
and all 24 target permutations. All assertions passed in under one second.
The closure and perturbation arguments above are analytic arguments, not
inferences from that finite check.

This refutes the diagonal-sum/tridiagonal-factor ansatz. It does not refute
a simultaneous-unitary-tridiagonalization theorem where the sum is allowed
to remain tridiagonal instead of diagonal, or another family of structured
witnesses. No successor ansatz was investigated after this refutation.
