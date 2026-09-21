# R389: exact obstruction to the direct Hermitian shortcut

Claim IDs affected: R0389-H1. The separate R0389-L1 remains unreviewed.

Hypothesis: Every feasible real 3x3 singular-value sum has a Hermitian
realization preserving its three singular lists after signs and permutations.

Assumptions/domain: Arbitrary feasible three-dimensional real singular-value
sums; all independent signs allowed for the Hermitian conversion.

Acceptance test: Universal proof or exact feasible counterexample.

Cheapest falsifier: Failure of every signed additive trace equality.

Budget: One candidate, one exact certificate, one independent review;
1 GiB address space, 30 s CPU, 45 s wall per calculation; no corpus.

Result: REFUTED for direct sign/permutation conversion.

Evidence: gate-0.md, certificate.py, certificate.json, and the independent
review and script linked in the handoff below.

Registry update: R0389-H1 refuted; general selection remains unresolved.

R0389-H1 is refuted. This is a restricted reduction claim, not the inverse
Horn problem or the SO(4)/tensor correspondence itself.

## Exact counterexample

Let Da=Db=diag(3,2,1), R=I, and

    S^T = [[0,1,0],[-1,0,0],[0,0,1]].
    T = Da+Db S^T = [[3,3,0],[-2,2,0],[0,0,2]].

S is proper orthogonal, TT^T=diag(18,8,4), and det(T)=24.
Thus input singular lists are (3,2,1) and the proper signed output list
is (3sqrt(2),2sqrt(2),2), all distinct and nonzero.

For any 3x3 Hermitian matrix the eigenvalues are its singular values with
signs. Every input trace is therefore an integer, and the sum of two input
traces is an integer. Every possible output trace equals

    sqrt(2)(3 e1+2 e2)+2 e3, with each ei in {-1,1}.

The coefficient 3e1+2e2 is never zero. Every output trace is irrational.
No signs, permutations, eigenbases, or use of complex Hermitian entries
can repair this trace contradiction.

## Original 4x4 realization is explicit

In one magic-basis ordering the real symmetric matrices are

    A = diag(2,0,4,-6),
    B = [[1,-1,0,0],[-1,1,0,0],[0,0,-1,-5],[0,0,-5,-1]].

Both have ordered spectrum (4,2,0,-6). The sum has ordered spectrum

    (5sqrt(2)-2, sqrt(2)+2, 2-sqrt(2), -5sqrt(2)-2).

All input and output eigenvalues are simple. The certificate verifies the
real SO4 conjugator, the magic-basis transformation, both singular lists,
and the output characteristic polynomial exactly. The independent reviewer
uses a different magic-basis ordering and obtains the same polynomial
(t^2-4t+2)(t^2+4t-46).

## Meaning and limits

The proposed left-right formulation is a valid inverse-problem equivalence;
the original difficulty is not removed merely by having 3x3 matrices. A
Hermitian conversion preserving the three singular lists adds a signed
trace equality that the original problem does not require. This example
shows that equality can fail for every sign choice.

The obstruction also has a positive gap: finitely many signed-trace
differences are all nonzero, so sufficiently small perturbations preserve
the exclusion. No numerical perturbation census or genericity claim is
needed for the exact counterexample. Auxiliary-variable reductions,
different spectra, higher dimensions, and the full singular-value inverse
problem are outside the tested hypothesis.

## Handoff and reproduction

Run from the repository root:

    timeout 45s bash -c 'ulimit -v 1048576; ulimit -t 30; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R389-signed-hermitian/certificate.py'
    make -f crates/can_sandwich/docs/inverse-horn/archive/Makefile research-check

Exact certificate: PASS, 392 distinct signed-trace comparisons; no numeric
solver or production/corpus use. The first integrity command was invoked
from the wrong working directory and failed to locate its script; rerunning
from the repository root with the explicit Makefile passed. There was no
certificate failure or subsequent tuning.

Independent review: [signed-singular-3x3-review.md](../../scouting/2026-09-15-additive-inverse-horn/signed-singular-3x3-review.md), with its own
[exact script](../../scouting/2026-09-15-additive-inverse-horn/check-signed-singular-counterexample.py). Existing reviewer context reused under AGENTS.md. R0389-H1
is recorded as refuted; no claim of novelty. Preliminary paper calculation
preceded gate-0.md as disclosed there. FRONTIER_DELTA for general inverse
selection is NONE. No successor exploration follows this refutation.

The preceding user's LR-chain algebra is separately recorded in
scouting/2026-09-15-additive-inverse-horn/rank3-coupling.md. That conditional
fixed-chain construction does not acquire endpoint coverage from R389.
