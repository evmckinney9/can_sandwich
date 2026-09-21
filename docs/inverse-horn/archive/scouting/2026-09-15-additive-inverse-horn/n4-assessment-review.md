# Review of the rank-four hive-to-real-frame assessment

2026-09-15. Bounded independent scope review. The evaluator's earlier
source-review and numerical-audit context was reused, not erased. The root
supplied the quoted claims and requested the audit; this reviewer did not
develop them. No new construction, experiment or theorem promotion.

## Verdict

**The proposed problem needs a sharper specification before its dimensional
or novelty claims are defensible.** The rank-four dimension count is correct
under regularity assumptions, but is not special to rank four. The displayed
output requirement asks only for one spectral witness and does not require
that the interior hive coordinates be used. The 2024 source identifies a
failed proof mechanism, not a failed hive conjecture.

## 1. The dimension equality holds in every rank

Write the real inverse-Horn fibre as

```text
F(alpha,beta,gamma) = {
  Q in SO(n): spec(D_alpha + Q D_beta Q^T) = gamma
}.
```

Assume simple factor and target spectra, and that the spectral map has
rank n-1 along the fibre. The trace is already fixed, leaving n-1 independent
spectral conditions. The regular-level dimension is therefore

```text
dim F = n(n-1)/2 - (n-1) = (n-1)(n-2)/2 = d.
```

A size-n hive has (n-1)(n-2)/2 interior lattice entries. Its fixed-boundary
polytope has dimension d when it has full dimension in those entries.
Thus the comparison is 0 versus 0 at n=2, 1 versus 1 at n=3, 3 versus 3
at n=4, and 6 versus 6 at n=5. Boundary degeneracy or rank loss can change
these dimensions; simple eigenvalues alone do not establish regularity.
The hive coordinates and their boundary interpretation are in
[Knutson–Tao, Appendix 2, Proposition 4](https://arxiv.org/html/math/9807160v4).

Consequently, “the first dimension where a literal hive-to-real-matrix
parametrization is dimensionally plausible and nontrivial” is incorrect.
Dimensionally nontrivial agreement already occurs at n=3. Rank four may
be a useful next size for a particular method, but dimension matching
neither defines a map nor proves one is locally invertible or surjective.

## 2. Specify which matrix space is being counted

For simple factors, fixing A=D_alpha and writing B=Q D_beta Q^T leaves
finite diagonal-sign redundancies. Real symmetric pairs modulo simultaneous
orthogonal conjugation therefore have the same generic dimension d as
the Q-fibre. They are not literally identical spaces: eigenbasis signs,
orientation and effective stabilizers must be accounted for.

For comparison, the regular irreducible **complex Hermitian** moduli has
real dimension

```text
2 n(n-1) - (n-1) - (n²-1) = (n-1)(n-2) = 2d.
```

Here the first term counts the two unitary conjugacy orbits, the second
fixes the sum spectrum, and the third quotients simultaneous conjugation
by the effective projective unitary group. At n=4 this is six, not three.
This is a conditional regular dimension count, not a dimension assertion
for every singular or reducible stratum. The additive problem is naturally
stated for real symmetric/Hermitian matrices; “SU(4)” should not obscure
which of these spaces or trace constraints is intended.

## 3. The displayed h-to-Q requirement can ignore h

For fixed feasible boundary data, suppose a witness Q0 is available. The
constant map Q(h)=Q0 satisfies the quoted requirement for every hive h:

```text
spec(D_alpha + Q(h) D_beta Q(h)^T) = gamma.
```

This observation is a logical property of the proposed interface, not a
construction of Q0. If the intended task is to parameterize all realizations,
require surjectivity onto a stated quotient. If it is to invert a particular
forward hive map Phi, require Phi(Q(h))=h. If only one witness is wanted,
neither extra condition is necessary. These are different problems.

## 4. Topology limits a literal global coordinate identification

For a nonempty regular value gamma, F above is a compact smooth manifold
without boundary: SO(n) is compact without boundary and F is its regular
level set. For d>0, a full-dimensional compact hive polytope has boundary.
They therefore cannot be globally homeomorphic merely because both have
dimension d. For a real moduli quotient the same conclusion requires a
free effective finite-sign action; singular quotients need separate analysis.

This conditional observation excludes a boundary-respecting global
homeomorphism in that regular setting. It does not exclude a many-to-one
map, a local chart, a construction with boundary identifications, several
charts, or a selector returning one witness. In particular it gives no
blanket obstruction to an algorithm taking a hive as input.

## 5. What Bercovici–Li actually establish

[Bercovici–Li (2024), Introduction, Propositions 3.1–3.2,
Corollary 4.4 and Example 4.5](https://imar.ro/journals/Revue_Mathematique/pdfs/2024/3-4/4.pdf)
study the Danilov–Koshevoy compression-trace formula. Their explicit
rank-four example defeats the projection-recombination identities used
by an earlier proposed proof. The example has repeated factor eigenvalues.
It is not a generic simple-spectrum counterexample to the conjecture.

They also construct optimizing projections in special cases and establish
the formula for the unique associated hive in their example class. Thus
the paper supplies positive construction tools while invalidating that
particular proof mechanism. It does not refute the compression-trace
conjecture, prove that no forward hive map exists, or establish that no
inverse construction is available elsewhere. Calling the conjecture
unresolved *in this paper* is supported; a claim about its current global
status would require checking subsequent work.

## 6. The “first unsolved rank” claim

The quoted statement “generic SU(4) is exactly the first unsolved rank
beyond the current exact construction” needs a named method and output
model. It is not justified as a statement that additive matrix realization
itself is globally unsolved in rank four. In particular, for algebraic
spectral input, an exact semialgebraic feasibility search on matrix entries is
a different computational task from a practical hive-coordinate inverse.

Safe scope: a particular reviewed low-rank construction may leave its
rank-four extension unestablished. Neither the dimension count nor failure
of one compression-trace proof establishes a literature-wide impossibility
or absence theorem. The source audit should identify the missing operation
in that particular route, rather than infer that an inverse cannot be
contained in another geometric construction.

## Handoff

Primary source scope was checked directly; the dimension and topology
arguments above are elementary conditional audits with assumptions made
explicit, not new registered research results. No current-global-status
search, singular-fibre classification, construction or computation was
performed. The root should add the 2024 source to LITERATURE.toml and avoid
using older numerical criticism as a blanket refutation of the conjecture.
