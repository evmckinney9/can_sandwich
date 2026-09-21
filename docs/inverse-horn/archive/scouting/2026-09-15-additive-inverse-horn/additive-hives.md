# Additive inverse Horn: what hives explain and what an inverse must do

2026-09-15. Independent bounded primary-source review. Context reused from
the earlier source synthesis and R0386 numerical audit; the reviewer did
not author the present synthesis. No experiment, new hypothesis or claim
promotion. The discussion concerns additive Hermitian matrices throughout.

## 1. Why the same boundary data occurs

[Knutson–Tao's survey](https://arxiv.org/html/math/0009048v1), Theorems 1–2,
equates matrix-sum feasibility with honeycomb boundary feasibility, including
degenerate honeycombs. For A+B=C, the three boundary lists are lambda, mu
and minus nu, with the appropriate reversed ordering.

There is a structural explanation. Integral honeycombs count tensor-product
multiplicities (Theorems 6–7). The matrix problem is the classical counterpart
of that representation problem: Theorem 5 relates a matrix realization to
an invariant after integer scaling, and conversely. Saturation removes the
scaling obstruction. This explains why local combinatorial inequalities
can encode a problem involving eigenvectors.

Section 1 also supplies tangible correspondences: a scalar solution is a
Y-shaped honeycomb, and overlaying diagrams matches block direct sums.
Section 2 permits linear-programming selection of honeycombs. Theorem 4
relates fibre volume to a spectral probability density, with normalization
factors. That volume identity is not an algorithm matching individual
matrices to individual honeycombs. These are substantial links between
objects, beyond coincidence of inequality lists.

## 2. What a hive actually stores

[Knutson–Tao's original paper](https://arxiv.org/html/math/9807160v4),
Appendix 2, Proposition 4, gives an explicit integer-linear equivalence
between honeycombs and hives with one additive constant fixed. A hive
assigns numbers to a triangular lattice. Each rhombus requires the sum at
its obtuse vertices to be at least the sum at its acute vertices. Boundary
differences give the spectra; rhombus slacks give nonnegative honeycomb
edge lengths. Thus conversion between these two descriptions is genuinely
constructive. Sections 5–6 additionally select special honeycombs using a
linear functional; this selects a combinatorial lift.

The introduction's geometric explanation uses three weighted flag
manifolds. Their invariant sections encode tensor multiplicities; the
corresponding symplectic quotient consists of Hermitian triples summing
to zero, modulo simultaneous unitary conjugation. The flags carry subspaces
that an eventual matrix construction needs. This relates the combinatorial
model to matrix geometry without identifying hive entries with matrix
entries or asserting that arbitrary flags already solve the matrix equation.

## 3. What the 2026 proof adds

[Moitra–Postnikov–Woodruff](https://arxiv.org/html/2607.06710v1), Definition
5.1 and Proposition 5.2, explain equality through four shared properties:
scalar base case, direct sum, convexity and splitting of extreme points
into smaller problems. Induction identifies the extreme spectra; convexity
fills their convex hull. This is an unusually direct explanation of why
honeycombs model Hermitian sums.

There are actual matrix operations in the proof. Section 7.1 assembles
block-diagonal witnesses. Sections 7.2–7.4 perturb a supplied unitary
eigenframe by exp(S), S skew-Hermitian, and compute spectral derivatives
to identify a block splitting. Section 7.5 treats repeated eigenvalues.
These ingredients must not be dismissed as merely polyhedral reasoning.

But Remark 5.3 explicitly leaves matrix convexity to symplectic geometry;
Section 7.1 cites the relevant convexity theorems. Proposition 5.2 invokes
membership in the convex hull, without constructing a matrix pair for
an arbitrary convex combination. The derivatives start from a supplied
realization. Neither operation supplies a general spectra-only inverse
for a prescribed interior hive. The theorem is about Hermitian triples;
this audit does not promote its proof into a real-orthogonal numerical
constructor or a complexity guarantee.

## 4. A small matrix example makes the distinction visible

Elementary illustration, obtained directly by multiplying 2-by-2 matrices:

```text
A = diag(a,-a),  a>0
B = b [[cos(2 theta), sin(2 theta)],
       [sin(2 theta),-cos(2 theta)]],  b>0
spec(A+B) = (+c,-c)
c² = a²+b²+2ab cos(2 theta).
```

Here the feasibility interval |a-b| <= c <= a+b gives an actual inverse:
choose cos(2 theta)=(c²-a²-b²)/(2ab). This constructs one real witness.
It illustrates the extra operation a higher-rank inverse needs: recover
compatible relative eigenspaces from feasible spectral data.

Simply averaging witnesses cannot implement spectral convexity. For
example, diag(1,-1) and diag(-1,1) have identical spectra, whereas their
average is zero. A convex combination of target spectra therefore cannot
be lifted by averaging factor matrices and expecting their spectra to
remain fixed.

## 5. Four different inverse specifications

These are task definitions, not four interchangeable claims:

| Task | Required output |
|---|---|
| One witness | Any A,B with the prescribed factor spectra and sum spectrum |
| Lift a supplied hive | A witness whose image under a specified matrix-to-hive map is that particular hive; the map must first be defined |
| Describe all solutions | A surjective parameterization, with gauge and exceptional cases stated |
| Preserve measure | A map transporting specified measures, not just matching total volumes |

For construction, the first task can be enough. A method may accept a hive
as a feasibility certificate, discard its interior coordinates, and build
one matrix witness by a separate operation. That would solve the first
task, even though it does not solve the second. Conversely, inability to
produce a measure-preserving map says nothing by itself about the existence
of a practical one-witness constructor.

## Review conclusion and limitations

The useful boundary for this audit is the missing *matrix operation* that
realizes spectral convex interpolation, not a blanket rejection of hives.
The reviewed proofs already offer combinatorial selection, block assembly,
and supplied-frame deformations. This source check does not establish that
those pieces form a complete inverse, or that another inverse cannot exist.
The broader literature on constructive convexity and additive balancing
is outside this bounded packet. No multiplicative inference is made.

Sources were read at the linked primary versions and exact sections above.
The existing literature IDs cover math/0009048 and 2607.06710; proposed
metadata for math/9807160 is in `hive-source-addition.toml` for the root's
single-writer literature update. No registry entry was changed here.
