# Exact check: direct sign conversion to 3-by-3 Hermitian Horn fails

2026-09-15. Independent review of the supplied explicit counterexample.
Existing uninvolved reviewer context reused. This is a bounded identity and
scope check, not a new solver or a novelty claim. The root agent owns any
claim-registry transaction.

## Precisely the reduction excluded

The excluded proposal replaces each of three proper signed-singular-value
lists by an eigenvalue list obtained solely through signs and permutations,
and then solves the ordinary 3-by-3 additive Hermitian Horn problem.
The example defeats every such sign/permutation choice, even if all eight
sign patterns are allowed independently for every list. It therefore also
defeats smaller Weyl/sign chambers restricted by orientation.

It does **not** exclude reductions using auxiliary matrices, extra variables,
different spectral data, shifts or other transformations beyond the stated
sign/permutation conversion. Nor does it exclude a direct construction for
the signed-singular-value sum problem.

## The left-right sum is exactly feasible

Let `Da=Db=diag(3,2,1)`, `R=I`, and

```text
S^T = [[ 0, 1, 0],
       [-1, 0, 0],
       [ 0, 0, 1]].

T = Da + Db*S^T
  = [[ 3, 3, 0],
     [-2, 2, 0],
     [ 0, 0, 2]].
```

Both rotations are proper. Exact multiplication gives
`T*T^T=diag(18,8,4)` and `det(T)=24`. Thus the decreasing proper
signed singular list can be taken as
`c=(3*sqrt(2),2*sqrt(2),2)`, with all entries positive and distinct.
Both input lists are also positive, distinct, and nonsingular.

## Why every Hermitian sign conversion fails

A real symmetric or complex Hermitian matrix with singular values `(3,2,1)`
must have eigenvalues given by signs of those three numbers. Its trace is
therefore one of `{-6,-4,-2,0,2,4,6}`. The trace of a sum of two such
matrices is an integer.

A Hermitian matrix with singular values c must instead have trace

`sqrt(2)*(±3±2) ±2`.

The coefficient of sqrt(2) belongs to `{-5,-1,1,5}` and is never zero.
Every such trace is irrational. The necessary identity
`trace(A+B)=trace(A)+trace(B)` is impossible, regardless of eigenvalue
permutations or choices of real versus complex eigenvectors.

This is an exact obstruction at the trace equality, before any further
Horn inequalities are considered.

## Explicit real 4-by-4 magic-basis lift

Let X,Y,Z be the Pauli matrices and define
`H(T)=sum_ij T_ij sigma_i tensor sigma_j`, with no extra normalization.
Use the unitary Q whose columns, in the computational basis, are
`(i*Psi+, Phi+, i*Phi-, Psi-)`, where the Bell vectors have norm one.
Then

`Q^* H(Da) Q = Dalpha = diag(4,2,0,-6)`.

For the second local factor take `U=(I-iZ)/sqrt(2)`. It sends
`X -> Y`, `Y -> -X`, and `Z -> Z`, giving the stated right rotation on
the coefficient matrix. In this magic basis its actual real frame is

```text
O = (1/sqrt(2))*[[ 1, 0, 0, 1],
                 [ 0, 1, 1, 0],
                 [ 0,-1, 1, 0],
                 [-1, 0, 0, 1]].
```

Direct exact checks give `O^T O=I` and `det(O)=1`. Moreover,

```text
Q^* H(T) Q = Dalpha + O*Dalpha*O^T
           = [[ 3, 0, 0,-5],
              [ 0, 3,-1, 0],
              [ 0,-1, 1, 0],
              [-5, 0, 0,-7]].
```

This is a real symmetric witness for the four-dimensional additive problem,
with input spectra `alpha=beta=(4,2,0,-6)`. Its characteristic polynomial is

`(t²-4t+2)*(t²+4t-46)`.

The decreasing output spectrum is exactly

`gamma=(5*sqrt(2)-2, sqrt(2)+2, 2-sqrt(2), -5*sqrt(2)-2)`.

All four input and output eigenvalues are simple. The displayed Q and O
verify this particular lift directly; no external magic-basis existence
assertion is needed to certify the example.

## Reproduction and closure

```sh
timeout 45s prlimit --as=1073741824 --cpu=30 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python crates/can_sandwich/docs/inverse-horn/archive/scouting/2026-09-15-additive-inverse-horn/check-signed-singular-counterexample.py
```

The independent SymPy script constructs the Pauli tensor map and Q directly,
checks proper rotations, verifies both matrix identities and the full
characteristic polynomial, and enumerates the possible trace coefficients.
It completed in under one second with all exact assertions passing. No
floating-point feasibility or production constructor was used.

Terminal result: the supplied example refutes the restricted direct
sign/permutation reduction. The broader relationship between these inverse
problems is unchanged by this counterexample. No successor mathematical
exploration was performed in this review.
