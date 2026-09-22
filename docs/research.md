# Problem formulation and research contract

The [README](../README.md) gives the project overview and usage. This page
defines the unrestricted problem, the GULPS conventions, and the requirements
for a proved construction.

## The unrestricted problem

Let $\alpha,\beta,\gamma\in\mathbb R^4$ specify conjugacy classes in $SU(4)$.
Choose their eigenphases in the closed Weyl alcove:

$$
\alpha_1\ge\alpha_2\ge\alpha_3\ge\alpha_4,\qquad
\alpha_1-\alpha_4\le 2\pi,\qquad
\sum_j\alpha_j=0,
$$

and impose the same conditions on $\beta$ and $\gamma$. Write

$$
D_\alpha=\operatorname{diag}(e^{i\alpha_1},\ldots,e^{i\alpha_4}),
$$

with analogous definitions for $D_\beta$ and $D_\gamma$.
A triple is **feasible** if some $A,B,C\in SU(4)$ in these conjugacy classes
satisfy $AB=C$.

The multiplicative Horn inequalities characterize feasibility.
The **inverse problem** asks for a witness. After simultaneous conjugation,
we can fix $A=D_\alpha$ and seek

$$
U\in SU(4),\qquad
\operatorname{spec}(D_\alpha U D_\beta U^\dagger)
=\operatorname{spec}(D_\gamma).
$$

Spectra are multisets: repeated eigenvalues retain their multiplicities.
The resulting matrices are

$$
A=D_\alpha,\qquad B=UD_\beta U^\dagger,\qquad C=AB.
$$

Allowing $U\in U(4)$ gives the same problem. A scalar phase can set
$\det U=1$ without changing $UD_\beta U^\dagger$.
**The research problem places no SO(4) restriction on $U$.**

We seek a construction specialized to $n=4$, with explicit branch choices,
a proof of coverage, and a termination bound. It must include boundary points
and repeated spectra. The witness need not be unique or continuous in the input.
Rational operations, radicals, and explicitly bounded polynomial solves are
acceptable ingredients. A numerical search that works on many examples does
not establish coverage.

Feasibility reference: [Agnihotri–Woodward](https://arxiv.org/abs/alg-geom/9712013).

## Exact input and output

For an exact algebraic baseline, supply the real and imaginary parts of the
three spectra as algebraic numbers with exact representations. Algebraic
angles do not in general imply algebraic eigenvalues.

Write the unknown witness as $U=X+iY$, with $X,Y\in\mathbb R^{4\times4}$.
Impose

$$
U^\dagger U=I,\qquad \det U=1,\qquad
\det(zI-D_\alpha U D_\beta U^\dagger)=\det(zI-D_\gamma).
$$

Equating real and imaginary parts, and coefficients in $z$, gives a finite
polynomial system in 32 real variables. Feasibility makes its real solution
set nonempty. General real-algebraic sampling can select an algebraic point.
This is an application of established algorithms, not a dimension-four result
of this project. See [Basu's survey](https://arxiv.org/abs/1409.1534).

A useful new construction must explain what it gains over that baseline:
lower degree, fewer variables, explicit branches, controlled arithmetic cost,
or a practical stable implementation. State the bound and the input model.
An unspecified call to “solve the remaining equations” does not complete the
construction.

Arbitrary real input requires a separate computational model. A bounded
numerical approximation is not an exact finite representation of its solution.
Keep exact algebraic selection, approximate reconstruction, and floating-point
implementation claims separate.

## What a construction must prove

1. **Reduction.** Every introduced normal form covers the stated inputs.
   Fixing one matrix by simultaneous conjugation is valid. Restricting the
   unknown witness to SO(4), a sparse pattern, or a fixed chart needs its own
   argument if used to solve the unrestricted problem.
2. **Selection.** The procedure chooses every intermediate parameter from the
   supplied spectra. Assuming a compatible hidden witness does not select one.
3. **Coverage.** At least one enumerated branch works for every feasible input,
   including repeated roots, zero denominators, and boundary strata.
4. **Termination.** Each branch has a stated finite procedure and bound.
   Optimizer convergence or a numerical sign change is not such a proof.
5. **Reconstruction.** The output satisfies unitarity, determinant, and the
   original spectral equations, with multiplicities and the target class intact.

For numerical experiments, state the input distribution, failures, error metric,
tolerance, and runtime. Match eigenvalues bijectively. Distance between sets of
distinct roots loses multiplicity and is not an adequate spectral check.

## Keep the problems distinct

The multiplicative feasibility problem concerns products of unitary matrices.
[Agnihotri–Woodward](https://arxiv.org/abs/alg-geom/9712013) characterize their
possible spectra using quantum Schubert calculus. The inverse problem asks for
the matrices themselves.

The additive Horn problem concerns sums of Hermitian matrices. Additive hives,
rank-one updates, and matrix-scaling algorithms cannot be transferred here
without a proved multiplicative reduction.

The GULPS application requires a real magic-basis witness and permits a central
sign on the target spectrum. The unrestricted research problem permits any
SU(4) witness and fixes the target conjugacy class. An algorithm for either
contract needs an explicit conversion before it solves the other.

## GULPS coordinates and endpoint factors

GULPS needs a local two-qubit gate between two canonical gates. In the magic
basis, local $SU(2)\otimes SU(2)$ gates act as real matrices in $SO(4)$.
This is the reason for the current witness restriction. See
[Zhang–Vala–Whaley–Sastry, Eqs. (19–20)](https://arxiv.org/pdf/quant-ph/0209120).

The API takes three monodromy triples `c`, `g`, and `t`: the left gate, the
right gate or prefix, and the target. These are GULPS coordinates, not the
four ordered eigenphases in the unrestricted formulation. With zero-based indices, define

$$
\theta(m)=\pi(m_1,m_0,-m_0-m_1-m_2,m_2),\qquad
D(m)=\operatorname{diag}(e^{i\theta_0(m)},\ldots,e^{i\theta_3(m)}).
$$

The solver seeks $O\in SO(4)$ such that

$$
\operatorname{spec}\!\left(D(c)^2 O D(g)^2 O^T\right)
=\operatorname{spec}\!\left(sD(t)^2\right),\qquad s\in\{+1,-1\}.
$$

The two signs are generally different conjugacy classes in $SU(4)$.
GULPS permits both because it tracks global phase separately. This differs
from the fixed target class in the research problem.
A general complex witness also does not directly provide the local gate
that GULPS requires. Any transfer from an unrestricted constructor must explain
how to recover the required real witness.

For endpoint recovery, set $K=D(c)OD(g)$. Its symmetric unitary matrix

$$
M=KK^T=D(c)OD(g)^2O^TD(c)
$$

is similar to the product in the spectral condition. The real and imaginary
parts of $M$ commute and have a common real orthogonal eigenbasis. The solver
reuses that basis to construct

$$
K=e^{i\phi}L D(t)R,\qquad L,R\in SO(4),\qquad
\phi\in\{0,\pi/2\}.
$$

This avoids a second endpoint eigendecomposition in GULPS. The implementation
checks the factors against the original matrix, including phase.

## Previous work

The former documentation contained additive studies, failed numerical pilots,
exact-model proposals, scripts, data, and duplicate status reports. Those records
are preserved in Git at commit
`0a48fdc0d0a0f264a20e5d07e362cd1d082544b2`, under `docs/inverse-horn/`.
The former `docs/architecture.md` is preserved at the same commit.

Inspect a historical file without restoring the archive into the working tree:

```sh
git show 0a48fdc0d0a0f264a20e5d07e362cd1d082544b2:docs/inverse-horn/README.md
```

Historical labels such as “accepted,” “complete,” and “reviewed” describe those
records. They are not substitutes for a checked theorem and its hypotheses.
The README and these two documentation pages define the current account.
