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
D_\alpha=\mathrm{diag}(e^{i\alpha_1},\ldots,e^{i\alpha_4}),
$$

with analogous definitions for $D_\beta$ and $D_\gamma$.
A triple is **feasible** if some $A,B,C\in SU(4)$ in these conjugacy classes
satisfy $AB=C$.

The multiplicative Horn inequalities characterize feasibility.
The **inverse problem** asks for a witness. After simultaneous conjugation,
we can fix $A=D_\alpha$ and seek

$$
U\in SU(4),\qquad
\mathrm{spec}(D_\alpha U D_\beta U^\dagger)
=\mathrm{spec}(D_\gamma).
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
D(m)=\mathrm{diag}(e^{i\theta_0(m)},\ldots,e^{i\theta_3(m)}).
$$

The solver seeks $O\in SO(4)$ such that

$$
\mathrm{spec}\!\left(D(c)^2 O D(g)^2 O^T\right)
=\mathrm{spec}\!\left(sD(t)^2\right),\qquad s\in\{+1,-1\}.
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

## One spectral acceptance test

For an exactly orthogonal frame, the master $M=KK^T$ is symmetric and unitary.
Write $M=X+iY$ with real symmetric $X,Y$. Expanding $MM^*=I$ gives

$$
X^2+Y^2=I,\qquad XY=YX.
$$

Thus $X,Y$ have a common real orthogonal eigenbasis. `spectral.rs` diagonalizes
real combinations $X+wY$ and checks the full complex matrix in the resulting
basis. If a projection merges distinct eigenspaces, its off-diagonal residual
can trigger another projection. The bounded projection list is a numerical
procedure, not a completeness theorem.

There is also a gap-independent justification for the residual. Let
$T=Q^TMQ$, $r_i=T_{ii}$, and let $\eta$ bound the magnitudes of its off-diagonal
entries. In dimension four,

$$
\|T-\mathrm{diag}(r)\|_F\leq\sqrt{12}\,\eta.
$$

Both matrices are normal in exact arithmetic. The Hoffman–Wielandt theorem
therefore supplies a multiplicity-preserving matching of their spectra within
this Frobenius bound; see the theorem recalled by
[Xu](https://arxiv.org/abs/1703.02422).
If the diagonal entries match the requested roots with maximum error $e$,
the realized eigenvalues admit a matching with maximum error at most
$e+\sqrt{12}\eta\leq e+4\eta$. This explains the residual used by the verifier
without dividing by target root gaps.

Floating-point orthogonality and eigensolver errors still matter. The code
checks the frame separately, and the corpus checks the resulting spectrum
independently. This is numerical verification, not an interval certificate.

The accepted eigenbasis is retained for endpoint recovery. Earlier coefficient
bounds, divided projectors, and companion-root fallback checks have been removed
from the production certificate. Repeated-root retargeting remains a candidate
repair and must pass the same verifier as every other candidate.
Charts retain an additional projector gate: removing it preserved corpus pass
counts but degraded some per-row residuals. It needs an accuracy-preserving
replacement before it can be deleted.

## Complementary minors in dimension four

The coefficient evaluator uses a dimension-four reduction. This simplifies
verification work; it does not construct a witness or prove solver completeness.

Put $A=D(c)^2=\mathrm{diag}(a)$ and $B=D(g)^2=\mathrm{diag}(b)$.
The master matrix is similar to $AOB O^T$. For two-element index sets $I,J$,
write $a_I=\prod_{i\in I}a_i$, $b_J=\prod_{j\in J}b_j$, and
$m_{IJ}=\det O[I,J]$. Cauchy–Binet gives

$$
e_1=\sum_{i,j}a_i b_j O_{ij}^2,\qquad
e_2=\sum_{|I|=|J|=2}a_I b_J m_{IJ}^2.
$$

The second sum has 36 terms. Jacobi's complementary-minor identity states

$$
\det O[I,J]
=(-1)^{\sum I+\sum J}\det(O)\det O^{-1}[J^c,I^c].
$$

For orthogonal $O$, substitute $O^{-1}=O^T$ and $(\det O)^2=1$ to obtain
$m_{IJ}^2=m_{I^cJ^c}^2$. This applies to both orientations of a real orthogonal
matrix. See Theorem 1 in
[Bapat and Sivasubramanian](https://www.isid.ac.in/~rbb/dist12.pdf) for the
underlying Jacobi identity.

Pairing each term with its complement therefore gives

$$
e_2=\sum_{I\in\{01,02,03\}}\sum_{|J|=2}
\left(a_Ib_J+a_{I^c}b_{J^c}\right)m_{IJ}^2.
$$

Only 18 minors are needed. No division by an eigenvalue gap occurs, so the
identity also covers repeated spectra and boundary inputs. It holds for
arbitrary diagonal $A,B$; unit determinant is not needed for this reduction.

`compound_residual` implements this formula with compensated summation. A
small independent test compares it with Newton's trace identities on the
explicit master matrix, using dense rotation products and both target signs.
The full corpus checks spectra and endpoint reconstruction independently.

The identity assumes exact orthogonality. Floating-point frames introduce an
error depending on their orthogonality defect, as well as rounding in the
sum. This coefficient test remains a construction filter. The final frame and
rootwise checks are unchanged. In particular, this reduction does not cure
the poor conditioning of coefficient-to-root inversion near repeated roots.

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
