# R391: T4 is false for all 24 orderings

Claim IDs affected: R0391-H1.

Hypothesis: Every feasible traceless additive n4 triple admits real
tridiagonal A and B=D_(pi gamma)-A for some permutation pi.

Assumptions/domain: Arbitrary feasible ordered spectra, including repeated
eigenvalues; all 24 diagonal arrangements of gamma allowed.

Acceptance test: Universal proof or exact feasible counterexample excluding
all witnesses of the proposed form, not merely the planted matrix pair.

Cheapest falsifier: Rank-one support rigidity forces unwanted eigenvalues.

Budget: One candidate, one symbolic check, one independent review.
Computations <=1GiB address space,30sCPU,45swall. No numerical search/corpus.

## Exact feasible data

Let J be the 4x4 all-ones matrix and set

    A=J-I,
    C=diag(3,1,-1,-3),
    B=C-A=C+I-J.

All matrices are real symmetric and traceless. Their ordered spectra are

    alpha=(3,-1,-1,-1), gamma=(3,1,-1,-3),
    beta=the decreasing roots of p(x)=x^4-16x^2+8x+16.

This specifies beta exactly by its polynomial and isolating intervals:
beta1 in (2,4), beta2 in (0,2), beta3 in (-2,0), beta4 in (-infinity,-2).
The characteristic polynomial computation proves feasibility. Evaluations
p(4)=48, p(2)=-16, p(0)=16, p(-2)=-48 give four distinct real roots in
these intervals by the intermediate value theorem and the quartic degree.

## Exclude every proposed normal-form witness

Any real symmetric A' with spectrum alpha has the form

    A'=4 vv^T-I, ||v||=1.

If A' is tridiagonal, v_i v_j=0 for |i-j|>1. All indices in the support
of v must therefore be mutually adjacent in the four-vertex path. That
support has at most two elements. Thus at least two coordinates of v vanish.

For every permutation pi, put B'=D_(pi gamma)-A'. Whenever v_i=0,

    B'e_i=(gamma_(pi(i))+1)e_i.

So B' must have at least two eigenvalues from the set {4,2,0,-2}.
But p is nonzero at every member of that set. None belongs to beta.
This contradiction excludes every permutation and every tridiagonal A'.

The same argument even excludes complex Hermitian tridiagonal A': replace
vv^T by vv*. A phase cannot make a product of two nonzero coordinates zero.

## The failure is not confined to repeated spectra

Let N be the set of ordered spectral triples admitting the proposed normal
form. N is closed. To see this, take a convergent sequence of triples in N.
Pass to a subsequence with constant pi. The corresponding tridiagonal A_k
have bounded operator norm, since their eigenvalues converge. A further
subsequence converges to a real symmetric tridiagonal A. Eigenvalue
continuity gives the required limiting alpha and beta with limiting
D_(pi gamma). Thus the limiting triple lies in N.

Our exact triple lies outside this closed set, so some neighborhood of it
also lies outside N. Now perturb the displayed A to real symmetric
traceless A_epsilon with simple spectrum, leave C fixed, and take
B_epsilon=C-A_epsilon. Such perturbations exist arbitrarily close: split
the triple eigenvalue on its orthogonal eigenspace by distinct shifts with
zero total trace. B and C already have simple spectra, so they remain
simple for sufficiently small perturbations. These are feasible all-simple
triples outside N. No explicit neighborhood radius is claimed.

## What the source theorem actually says

Davidson–Djokovic, Theorem 1.1, proves that every complex 4x4 matrix can be
unitarily tridiagonalized, equivalently that every pair of Hermitian 4x4
matrices can be simultaneously unitarily tridiagonalized. This allows C to
be tridiagonal too; it does not hold C diagonal. Their introduction also
states that a real orthogonal simultaneous tridiagonalization need not
exist even when both starting matrices are real symmetric.

The T4 counterexample contradicts neither theorem. It excludes the stronger
spectral normal form the user specifically asked to prove or kill.

## Result and handoff

Result: R0391-H1 REFUTED. No successor normal-form search was launched.

Evidence: gate-0.md; certificate.py and certificate.json; independent
proof review and independently runnable checker in the additive scouting
directory. The exact certificate checks the spectra, root intervals, and
168 permutation/support combinations. The proof above supplies coverage
of every possible vector; the finite enumeration is a replay, not a
numerical inference. Preliminary paper calculation before gate creation is
disclosed in gate-0.md. Existing reviewer context reused under AGENTS.md.

Registry update: R0391-H1 refuted. General additive inverse Horn and the
previous quartic slice-selection obligation remain unresolved.

Independent review: [T4-tridiagonal-review.md](../../scouting/2026-09-15-additive-inverse-horn/T4-tridiagonal-review.md).
Both the author certificate and the independently written checker passed
exactly. Research integrity passed at closure.

Reproduce from the repository root:

    timeout 45s bash -c 'ulimit -v 1048576; ulimit -t 30; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R391-tridiagonal-sum/certificate.py'
    make -f crates/can_sandwich/docs/inverse-horn/archive/Makefile research-check

Primary source checked: Davidson–Djokovic, Tridiagonal forms in low
dimensions, author PDF introduction and Theorem 1.1. No historical Horn
proof audit or new claim about the overall literature frontier was needed.
