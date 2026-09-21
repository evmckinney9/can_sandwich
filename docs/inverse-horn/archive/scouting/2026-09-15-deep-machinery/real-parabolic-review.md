# Real parabolic correspondence: bounded independent review

Date: 2026-09-15. New source-review task, separate from the geometric-navigation
review. No theorem promotion, experiments, or registry edits. The proposed
composition was supplied by the parent; the reviewer did not author it.

## Verdict

**The real parabolic-bundle-to-real-unitary-representation bridge is a
published theorem.** The relevant real structure is the ordinary real
structure, not the paper's compact-type real structure. This materially
supports the proposed connection to physical real frames.

The complete composition from arbitrary feasible spectra and generic flags
is an application of multiple results, not a theorem stated in this paper.
It needs an explicit weight/degree conversion, a real polystable object, and
orbifold-generator conventions. Evaluating the compatible Hermitian-Yang-Mills
metric and its holonomy remains a substantial constructive operation; no
finite algorithm or cost bound follows just from the correspondence.

## Published bridge and exact hypotheses

[Biswas–Schaffhauser, arXiv:1806.09782v3](https://arxiv.org/pdf/1806.09782),
Theorem 1.1, applies to a compact Klein surface `(X,sigma)`, a finite
sigma-invariant puncture set `S`, and `Y=X-S` of negative Euler
characteristic. It identifies polystable real parabolic bundles of parabolic
degree zero with real unitary representations of
`Gamma=pi_1^orb(Y/sigma)` into `U(r) semidirect Z/2`, acting by conjugation
and preserving the quotient to `Z/2`.

Definitions 2.1–2.2 require weights in `[0,1)`, weighted partial flags, and
an anti-linear lift preserving them with square `+1`. Section 2.2 defines
`pardeg(E)=deg(E)+sum(weight times multiplicity)`. Theorem 2.8 equates
ordinary and real semistability, and ordinary and real polystability.

Lemma 3.5 and Theorem 3.6 use an invariant adapted Hermitian-Yang-Mills metric;
its Chern connection is flat at parabolic degree zero and its holonomy
extends to Gamma. Remark 3.3 gives local eigenvalues `exp(2 pi i beta)`
and inverse-loop/conjugation compatibility at real punctures. Theorem 3.7
instead concerns compact-type structures and ordinary `U(r)` representations.

## Apply the hypotheses to the proposed sphere

For `X=P^1`, `sigma(z)=conjugate(z)`, and `S={0,1,infinity}`, the
punctures are fixed and `chi(Y)=2-3=-1`. These source hypotheses hold.
Standard conjugation on a trivial holomorphic rank-four bundle preserves
flags defined over the reals and has square `+1`. Thus real flags give the
correct kind of real structure. No quaternionic choice is forced.

Real flags do not by themselves provide polystability or the degree-zero
normalization required above. These are separate checks.

## The Teleman–Woodward input

[Teleman–Woodward, arXiv:math/0012241v2](https://arxiv.org/pdf/math/0012241),
Proposition 4.2(b), says that in genus zero the parabolic moduli is nonempty
if and only if the trivial principal bundle with general parabolic
structures is semistable. Section 2.1 permits signed markings of spread
strictly less than one. In its `SL(r)` vector-bundle convention the ordinary
degree and the weight sum at each puncture are zero. Section 2.5 describes
grade equivalence. The warning immediately before Proposition 4.2 gives an
example where grading changes the underlying trivial bundle to an unstable
ordinary bundle. The small-marking restriction of Remark 4.3 belongs to its
stronger flag-GIT description; it must not be omitted if that description
is used.

**Application inference:** the real points of the relevant complex flag
varieties are Zariski dense. Consequently, a nonempty Zariski-open set of
general complex flags contains real flags. This supports the existence of
general real semistable flag data under the proposition's domain. It does
not specify a deterministic flag triple, prove that one fixed triple works
for all weights, or bound the work to recognize semistability.

The quantifiers are important: **for each fixed feasible marking satisfying
the source guards, there exists a real flag tuple in its general semistable
set**. To see the density input concretely, the real big cell of a split
partial flag variety is a real affine space inside its complex big cell;
a complex polynomial vanishing on that real affine space vanishes
identically. Products retain this density. Therefore a proper complex
algebraic exceptional set cannot contain every real flag tuple. Selecting
and certifying a tuple outside that exceptional set is an additional
constructive genericity task. The argument does not require the exceptional
set itself to be defined over the reals.

## Weight and degree conversion must be explicit

The signed `SL(4)` convention and the normalized vector-bundle convention
are different. Suppose the chosen peripheral spectra are written
`exp(2 pi i mu_jk)` with `sum_k mu_jk=0`. To express the same local
eigenvalues in the normalized convention, one needs

```text
beta_jk = mu_jk - floor(mu_jk),    0 <= beta_jk < 1,
N = sum_jk beta_jk,
deg(E_normalized) = -N.
```

`N` is integral because the sums of the signed weights are integral. This
is a necessary arithmetic consistency check. It does **not** assert that
relabeling weights alone gives the normalized parabolic bundle.

In particular, retaining `E=O^4` while replacing negative weights by their
fractional parts generally gives positive parabolic degree, so Theorem 1.1
cannot be applied to that object. Appropriate local filtered-lattice changes
(elementary modifications), with corresponding flag reindexing, must change
the underlying holomorphic extension. The application must show that these
changes preserve the real structure, stability, and prescribed peripheral
classes. Real punctures and real flags are favorable for doing so, but this
review does not supply that conversion theorem or implementation.

At a repeated eigenvalue use partial flags with weight multiplicities; do
not artificially separate equal normalized weights. Teleman–Woodward's
spread-one boundary needs its own marking/extension treatment, because the
ordinary definition used by Proposition 4.2 has a strict spread guard.
Biswas–Schaffhauser's theorem itself does not require simple weights.

## Semistable versus polystable

If the supplied real parabolic bundle is geometrically stable, it is already
polystable and this gap disappears. If it is merely semistable, the claimed
chain must use a polystable representative of its grade/S-equivalence class
with a real structure. Theorem 2.8 does not say that semistable implies
polystable. An invariant Jordan–Hölder/socle construction is an appropriate
additional operation to justify; at walls it cannot be silently replaced by
an assumption that generic flags are stable.

Nor can the eventual unitary object be assumed to retain a trivial
underlying holomorphic bundle or the initial flag coordinates. The graded
object can change those data while retaining the relevant parabolic class.

## From the orbifold representation to a physical frame

The following is an explicit **application inference**, not a statement
quoted from Biswas–Schaffhauser.

The quotient of this three-punctured sphere by conjugation has three mirror
boundary intervals. With the usual ideal-triangle presentation, its orbifold
group has reflection generators `r1,r2,r3`, each of square one. Choose
peripheral generators consistently as

```text
a=r1 r2,  b=r2 r3,  c=r3 r1;     a b c=1.
```

Their orientations must be matched to the source's positive peripheral
weight convention. In particular the three classes for a two-factor product
problem are `(A,B,T^-1)`, not `(A,B,T)` without an inverse convention.

A real unitary representation sends each `ri` to an antiunitary involution
`Ji`; an arbitrary element of the antiunitary coset need not be involutive,
so the relations `ri²=1` are essential. Choose a unitary basis in which the
shared `J2` is standard complex conjugation `K`. Then `J1 J2` and `J2 J3`
are symmetric unitary matrices. Each has commuting real symmetric real and
imaginary parts, hence a real orthogonal eigenbasis. The relative eigenbasis
gives an orthogonal middle frame with the prescribed two factor spectra and
product spectrum. A commuting column-sign change repairs its determinant
to `+1`.

This finite-dimensional algebra is compatible with the known
Falbel–Wentworth Lagrangian-involution picture. For the actual GULPS adapter,
the ordered eigenvalues, determinant-one lift, diagonal square roots and
original gate representatives still require the existing explicit
normalization/transport conventions. A unitary representation up to global
conjugacy is enough to begin this reconstruction, but it is not by itself
the final stitched gate witness.

The relevant exact triangle-to-frame operation is already registered:
`R0052-H1` verifies the constructive correspondence modulo the specified
matched-parity centralizer gauge and finite witness transport on all
multiplicity strata. This review does not treat that exact inverse as a new
open problem. What must be supplied to it is the correctly ordered triangle
from the real orbifold representation. R0052's registry explicitly leaves
finite-precision, original-representative, and production claims outside its
scope; those distinctions remain in force.

## Why compact-type is the wrong substitute

The intended generator images reverse complex structure on the fibres.
They therefore use conjugation in the target semidirect product. Ordinary
real flags fit that choice. The compact-type correspondence has a different
duality condition and gives ordinary unitary images of the orbifold group;
it does not supply the three antiunitary involutions required by this
composition. The distinction is structural, not terminology.

## Constructive status and handoff

The theorem confirms that a correctly prepared polystable real parabolic
object carries compatible real holonomy. It does not justify recovering
that holonomy by separately choosing three matrices with the desired
eigenvalues, or by averaging arbitrary metrics and assuming the
Hermitian-Yang-Mills equation survives.

The strongest defensible handoff is:

1. Compile signed spectral/flag data into the correctly normalized real
   degree-zero parabolic object, including boundary and grading cases.
2. Evaluate its invariant adapted metric/flat connection with certified
   precision and termination, then compute the orbifold holonomies.
3. Apply the antiunitary-involution reconstruction with checked generators,
   lifts, and original-factor transport.

This is a published bridge plus a plausible, explicitly conditional
composition. No finite constructor, global coverage, or cost decrease is
promoted. The review ran no experiments. Primary theorem locations above
are the reproduction record; proposed source entries are in
`real-parabolic-sources.toml`.
