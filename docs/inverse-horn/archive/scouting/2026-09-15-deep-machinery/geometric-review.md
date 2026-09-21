# Geometric navigation: independent source and applicability review

Date: 2026-09-15. Scope: the frozen R0238 geometry argument and relevant
published local geometry. This is a source review, not a new construction,
experiment, or promotion review. The reviewer did not develop R0238. No
registry or production files were changed. `R0238-H1` remains unreviewed.

## Result

The literature supplies more than spectral existence. It supplies local image
cones on the compatible real locus, concrete real bending deformations, and
an involution-compatible stabilizer model that remains meaningful at current
endpoint collisions. These are positive ingredients for target-directed
navigation on the full frame space.

They do not yet constitute a bounded navigation operation from the supplied
physical frame. The specific remaining task is to realize a target-facing
direction in the local stabilizer model, transport it to the original
`SO(4)` frame, and certify a nonzero step and progress. Merely knowing the
target is in the image, or that an abstract model has the same local image,
does not provide those outputs.

## Exact navigation interface

Use the R0238 notation with fixed determinant-one diagonal unitary `A=D²`
and `B`, and a supplied current `O in SO(4)`:

```text
S(O) = D O B O^T D,
q(O) = ordered SU(4) endpoint phases in a specified closed alcove,
tau  = feasible target in that same lift,
E(O) = ||q(O)-tau||²/2.
```

An effective step would take `(A,B,O,tau,tolerance)` and return either an
accepted target frame or `O_next`, together with an explicit positive decrease
certificate and a domain/radius certificate. For a definitive procedure,
successive steps must have a termination and precision bound for every
feasible input, including repeated input and target roots. The source review
does not establish this interface.

## What the Horn halfspaces say about directions

After fixing factor spectra and a target lift, let their correctly normalized
Horn/alcove polytope be `P={q : Hq <= b}`. This is notation for the existing
feasibility data, not a newly derived fourteen-inequality formula.

The active inequalities at `q` give its spectral tangent cone:

```text
H_i d <= 0 for each active row H_i q=b_i,
```

with the affine trace constraint and any fixed equalities retained. Since
`tau in P`, `d=tau-q` satisfies these conditions. In fact the whole segment
`q+s(tau-q)`, `0<=s<=1`, is spectrally feasible. Inactive rows give finite
slack for other proposed directions. Equivalent paired interval bounds carry
the same information if they describe the same polytope without losing
variables.

This is concrete directional information in **spectral space**. It supplies
neither a six-component frame velocity nor the relationship between spectral
slack and distance traveled in the frame. If the seven intervals arise by
projection/elimination, their endpoints also do not retain the common
eigenframe needed to glue successive realizations. The exact application's
normalization and interpretation of its fourteen rows/seven intervals must
travel with an implementation; this review does not silently identify them
with coordinates of a symplectic slice.

At a simple endpoint where R0238's phase Jacobian `J` has rank three, its
existing least-norm solve `J delta=tau-q` already converts that direction
into a first-order physical step. At a stall, `im J` can be smaller than the
local image cone: a feasible spectral change can begin at second order. Thus
the cone is precisely the missing information a Jacobian alone cannot give.

## Checked source statements

### Schaffhauser: real local images on the torus cross-section

[Theorem 4.1(iii)](https://arxiv.org/abs/0705.0858) gives openness onto the
same local polyhedral cone for a Hamiltonian torus space and its compatible
anti-symplectic fixed locus. Theorem 4.3 identifies that cone with the cone
over the global image when the connected map has local convexity data and
is proper. Proposition 4.6 permits restriction to a closed convex polyhedral
region. Theorem 4.8 constructs a connected Hamiltonian torus cross-section
whose saturation is dense. Section 4.4 establishes compatibility of its real
locus. Definition 5.2 gives the non-componentwise product involution.

These statements explain why a singular differential need not obstruct nearby
real realizations. The density clause in Theorem 4.8 does not, on its own,
give local openness at a point outside the principal cross-section.

### Falbel–Wentworth: an actual family of real deformations

[Section 4.2, Definition 4.2 and equation (10)](https://arxiv.org/abs/math/0506100)
define real bending by applying an orthogonal transformation associated with
one Lagrangian plane to a consecutive family of planes. The induced
representation preserves the other conjugacy classes and changes one class
in a controlled way. Lemma 4.3 computes the tangent image. Proposition 4.3
uses bending and twists to prove local spectral surjectivity at an
irreducible Lagrangian representation on its specified multiplicity space.
Proposition 4.2 is the corresponding unitary statement; the blockwise use in
Section 5 applies within a fixed relatively irreducible block type.

This is useful deformation machinery, not just an existence proof. Its
fixed-multiplicity conclusion must not be read as a theorem about splitting
an endpoint collision. Bending coordinates still have to be aligned with
the supplied original factors and selected in the desired direction.

### AMM: collisions require the full endpoint centralizer

[Proposition 7.1 and Remark 7.1](https://arxiv.org/html/dg-ga/9707021#S7)
construct a local quasi-Hamiltonian cross-section for the centralizer `Z_f`
of the current product `f`. Shifting its moment map to `f^-1 mu` and using
a local logarithm gives an ordinary Hamiltonian `Z_f` model near the fibre
over `f`. No simple-spectrum hypothesis is required for this cross-section.
At a collision, `Z_f` is nonabelian; replacing it by the principal torus
would discard the directions that split repeated eigenvalues.

### O'Shea–Sjamaar: compatible stabilizer model

[Theorem 7.1](https://arxiv.org/pdf/math/9902059) gives an involution-compatible
Hamiltonian normal form near a fixed point at zero moment value. With ambient
group `U`, point stabilizer `H`, and symplectic slice `V`, its model is

```text
U ×_H (h^0 × V),
Phi([u,xi,v]) = Ad*_u(xi + Phi_V(v)),
beta_model([u,xi,v]) = [sigma(u),-sigma(xi),beta_V(v)].
```

Here `h^0` is the annihilator of the stabilizer Lie algebra in `u*`, and
`Phi_V` is quadratic. Definition 8.1 and Theorem 8.2 describe local moment
cones at arbitrary moment values through the centralizer and slice; the
real cone is the ambient cone intersected with the appropriate real Cartan
space. The stated neighborhoods are invariant neighborhoods of an orbit.
Equation (7.1) is an inclusion of a natural real model into the full fixed
locus, not an unconditional equality for every stabilizer.

This last distinction matters when extracting a step near one specified real
frame rather than somewhere near its entire complex-group orbit.

## Review of the frozen R0238 argument

The explicit identification used there is appropriate to the checked source:

```text
M = C_B × C_A,  mu(X,Y)=XY,
beta(X,Y)=(Y^T X^T conjugate(Y),Y^T),
(X,Y)=(D O B O^T D^-1,A).
```

The involution is the two-factor instance of Definition 5.2. The displayed
pair is fixed, and its product is `S(O)`. This avoids the known invalid
componentwise-transpose involution.

For the stated simple `A,B` domain, the local gauge argument is consistent:
a beta-fixed `Y` is symmetric unitary, a local real eigenbasis puts it into
the original `A`, and the resulting `D^-1 X D` is symmetric unitary with
spectrum `B`. Its commuting real symmetric real/imaginary parts give the
local real orthogonal diagonalizer; column signs repair orientation. This
is a local gauge statement, not a canonical global eigenbasis.

At a **simple current product**, its real eigenbasis puts the pair into the
open-alcove cross-section. A path in that connected cross-section to the
simple feasible target has compact image inside the open alcove. R0238's
compact convex restriction can therefore use 4.6 and 4.3 without falsely
claiming that the whole open cross-section is proper. The resulting cone
contains the direction to the target, and 4.1(iii) supplies nearby real
preimages. The physical local gauge then gives nearby decreasing frames.

I found no source mismatch in this scoped qualitative argument. It does not
prove strict negative Hessian curvature, select a point in the local
preimage, or establish the complete R0238-H1 statement at current collisions.
The registered numerical pilot is not used as evidence in this review.

## The finite-dimensional problem at a stall

The checked source chain identifies the following work packet at a supplied
frame; completing it would be new author work and is not done here.

1. Construct the centralizer cross-section at `f=S(O)`, preserve the physical
   involution through shift/logarithm, and track the affine alcove branch.
2. Compute the **point** stabilizer `H=Z(X) intersect Z(Y)`, its real
   involution, and its symplectic slice `V`. Do not confuse `H` with `Z_f`.
3. Find a real slice displacement whose local moment value faces the target,
   and belongs to the fixed-locus component attached to the supplied frame.
4. Convert that displacement through an effective local model inverse and
   the original-factor gauge. Bound its radius, spectral change, and error.

There are only five endpoint eigenvalue multiplicity types in rank four:
`1+1+1+1`, `2+1+1`, `2+2`, `3+1`, and `4`. Their centralizers are the
corresponding determinant-one block-unitary groups. This elementary
classification makes the collision cases finite **by type**. It does not
make the stabilizer group finite, enumerate its embedding, or solve the
slice inverse. Reducible representation types and real fixed components
need their own treatment.

On an abelian slice the familiar squared-weight normal form suggests a
nonnegative weight allocation and square-root amplitudes. That is an
abstract model operation. At a nonabelian collision the target is a
conjugacy class in `Z_f`, and the slice moment map is matrix-valued; reducing
it to the same scalar allocation would require an additional theorem.

## Decision and handoff

**Retain geometric navigation as a source-supported route.** The useful
next object is an effective real local-slice step at a supplied stalled
frame, with the above interface. Do not replace it by another generic
polynomial chart, an assertion of principal-stratum density, or a search for
a global continuous section.

No coverage or complexity component is discharged by this review. No
universal no-trap or termination theorem is promoted. Source applicability
has been narrowed to precise positive statements and explicit transfer
obligations; absence of a packaged constructor is not evidence of exclusion.

Reproduction: read the named theorem/definition locations at the linked
primary versions and the frozen
`attempts/2026-09-09-R238-geometric-endpoint-descent/geometry.md`.
Source additions are in `geometric-sources.toml`; all four paper IDs already
exist in `LITERATURE.toml`. No computation, fixture, or corpus was introduced.
