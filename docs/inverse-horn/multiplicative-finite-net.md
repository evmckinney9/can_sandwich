# A certified finite constructor for multiplicative (n=4)

This note treats the multiplicative problem directly. Let

\[
 M(Q)=D_\lambda QD_\mu Q^T,
\]

where the diagonal entries of (D_\lambda,D_\mu) and the target roots
\(\tau\) have modulus one and determinant one. The input is feasible when
some \(Q_\ast\in SO(4)\) has \(\operatorname{spec}M(Q_\ast)=\tau\).

## Explicit choice mechanism

Use the six-factor Givens chart

\[
 Q=G_{12}(a_1)G_{13}(a_2)G_{14}(a_3)G_{23}(a_4)G_{24}(a_5)G_{34}(a_6)S,
\]

with \(S\) ranging over the eight diagonal sign matrices of determinant one.
This is a fixed Euler chart. The finite constructor must include the finite
sign/permutation charts needed at its coordinate singularities; a single
six-angle chart is only a local coverage claim. The mesh argument below is
valid on each chart, and exhaustive enumeration over the chart list is what
turns it into a global finite net.
For mesh width \(h\), enumerate every angle tuple whose coordinates are
multiples of \(h\), and every \(S\). For each candidate compute the target
root matching error

\[
 E(Q)=\min_{\pi\in S_4}\max_i|\operatorname{root}_i(M(Q))-\tau_{\pi(i)}|,
\]

using the characteristic polynomial and a certified root-isolation routine.
Return the first candidate with \(E(Q)\le\varepsilon\); if none passes, report
that the prescribed net found no certificate.

This is an actual direction choice: the finite tuple and sign matrix are
enumerated in lexicographic order, and acceptance is checked against the
original multiplicative product.

## Coverage bound

Let \(b=\max(\|D_\lambda\|_2,\|D_\mu\|_2)\). A plane rotation is
1-Lipschitz in its angle, so the six-factor product satisfies

\[
 \|Q(a)-Q(a')\|_2\le 6\|a-a'\|_2.
\]

Consequently

\[
 \|M(Q(a))-M(Q(a'))\|_2\le 6b\|a-a'\|_2.
\]

Weyl's perturbation inequality for unitary matrices gives the same bound on
the Hausdorff distance between their multisets of eigenvalues. Every angle
tuple is within \(\sqrt6h/2\) of a mesh point, hence a feasible \(Q_\ast\)
has a mesh neighbour with

\[
 E(Q)\le 3\sqrt6\,b\,h.
\]

Choosing \(h=\varepsilon/(3\sqrt6\,b)\) therefore guarantees that the
exhaustive search contains an \(\varepsilon\)-accurate accepted candidate.
The argument also covers repeated target roots because it uses Hausdorff
matching rather than ordered eigenvalue gaps.

## What this establishes

For every feasible multiplicative input and every \(\varepsilon>0\), the
procedure terminates and returns an explicitly chosen real orthogonal direction
with certified spectral error at most \(\varepsilon\), provided the root
certificate is implemented exactly.

It does **not** give an exact frame for finite \(\varepsilon\), and it is not
practical at high accuracy: the number of angle evaluations is on the order of
\((2\pi/h)^6\). Thus this is a complete approximate constructor and a precise
baseline for any claimed exact or efficient selector; it does not close those
stronger obligations.
