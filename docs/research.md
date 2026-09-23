# Problem formulation and research contract

This page develops the mathematical problem behind can_sandwich, explains
how GULPS represents its inputs and outputs, and sets out what a complete
construction would need to prove. The [README](../README.md) introduces the
project, while the [researcher guide](researcher.md) shows how to evaluate a
candidate against the corpus.

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

The multiplicative Horn inequalities characterize which triples are feasible;
the **inverse problem** asks us to construct matrices that realize one of
those triples. Simultaneous conjugation lets us fix $A=D_\alpha$, so it is
enough to find

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

Allowing $U\in U(4)$ would give the same problem because a scalar phase
can set $\det U=1$ without changing $UD_\beta U^\dagger$. This formulation
allows a general complex witness, although the existence results below show
that a real one can always be chosen.

An SO(4) witness nevertheless exists for every feasible spectral triple.
[Falbel–Wentworth, Theorems 1 and 3](https://arxiv.org/pdf/math/0506100)
give representatives for which the first two matrices are simultaneously
symmetric. Since the real and imaginary parts of a symmetric unitary matrix
are commuting real symmetric matrices, each representative has a real
orthogonal eigenbasis. The relative eigenbasis then gives an orthogonal
witness, whose determinant can be corrected by flipping one column without
changing the conjugated diagonal matrix. The argument also covers repeated
spectra.

[Peterson–Crooks–Smith, Corollary 14](https://arxiv.org/pdf/1904.10541)
states this spectral reduction in the two-qubit setting and identifies
circuit realization as an algorithmic problem in Section 7.1. The existence
theorem guarantees a real witness, but finding it efficiently, or converting
an arbitrary complex witness into one, still requires a construction.

We seek a construction specialized to $n=4$ that selects its branches
explicitly and has a termination bound for every feasible input, including
boundary points and repeated spectra. The selected witness may change
discontinuously with the input and need not be unique. Rational operations,
radicals, and polynomial solves with explicit bounds are acceptable, provided
the proof establishes coverage beyond the examples used to test the algorithm.

Feasibility reference: [Agnihotri–Woodward](https://arxiv.org/abs/alg-geom/9712013).

## Exact input and output

To make exact computation precise, we can supply the real and imaginary
parts of the three spectra as exactly represented algebraic numbers. This
assumption concerns the eigenvalues themselves: algebraic angles do not in
general give algebraic eigenvalues.

Write the unknown witness as $U=X+iY$, with $X,Y\in\mathbb R^{4\times4}$.
Impose

$$
U^\dagger U=I,\qquad \det U=1,\qquad
\det(zI-D_\alpha U D_\beta U^\dagger)=\det(zI-D_\gamma).
$$

Equating real and imaginary parts and coefficients in $z$ gives a finite
polynomial system in 32 real variables. For a feasible input, its real solution
set is nonempty, so general real-algebraic sampling can select an algebraic
point. These established algorithms already provide an exact construction
in principle, as described in [Basu's survey](https://arxiv.org/abs/1409.1534).

A new construction should explain how it improves on that general procedure,
for example through lower degree, fewer variables, explicit branches, or
better arithmetic cost and numerical stability. Its input model and bounds
need to include the method used to solve any remaining equations, since
that step may account for most of the work.

For arbitrary real inputs, a different computational model is needed to say
what information the algorithm receives and what it can return exactly.
A numerical approximation with a proved error bound is useful, but it makes
a different claim from an exact algebraic construction.

## What a construction must prove

1. **Reduction.** Every introduced normal form covers the stated inputs.
   Fixing one matrix by simultaneous conjugation is valid. Restricting the
   unknown witness to SO(4) uses the existence theorem above. Restricting it
   further to a sparse pattern or a fixed chart needs its own coverage proof.
2. **Selection.** The procedure chooses every intermediate parameter from the
   supplied spectra. Assuming a compatible hidden witness does not select one.
3. **Coverage.** At least one enumerated branch works for every feasible input,
   including repeated roots, zero denominators, and boundary strata.
4. **Termination.** Each branch has a stated finite procedure and bound.
   Optimizer convergence or a numerical sign change is not such a proof.
5. **Reconstruction.** The output satisfies unitarity, determinant, and the
   original spectral equations, with multiplicities and the target class intact.

Numerical results should describe the input distribution, failures, error
metric, tolerance, and runtime so readers can assess the evidence. Spectral
checks must match eigenvalues bijectively, because comparing sets of distinct
roots would discard their multiplicities.

## Related problems and results

For products of unitary matrices,
[Agnihotri–Woodward](https://arxiv.org/abs/alg-geom/9712013) characterize the
possible spectra using quantum Schubert calculus. Those feasibility results
provide the premise for the inverse problem, which asks for the matrices
themselves. The additive Horn problem instead concerns sums of Hermitian
matrices, so using its hives, rank-one updates, or matrix-scaling algorithms
here requires a proved reduction to the multiplicative setting.

GULPS requires a real witness in the magic basis and allows a central sign
on the target spectrum. Although real and complex witnesses have the same
feasible spectral triples, the implementation still needs the real matrix
that represents a local gate. The sign allowance is a separate distinction:
a result on the negative target branch does not realize the fixed target
class in the unrestricted problem.

Several recent results may help with a construction, though each leaves a
selection problem for this application.
[François–Tarrago](https://arxiv.org/abs/2405.06723) give polytope-volume formulas
for products of generic unitary conjugacy classes, from which we would still
need a map taking a polytope point to realizing matrices.
[Kenyon–Ovenhouse](https://arxiv.org/abs/2407.10786) parametrize complex matrix
pairs with prescribed product spectra, leaving the choice of parameters that
give a unitary realization.
[Ye](https://arxiv.org/abs/2607.14634) constructs cluster structures on complex
$SL_n/SO_n$ strata, which would need a method to select a point in the required
compact real spectral fibre. These sources were last checked on 2026-09-22.

## GULPS coordinates and endpoint factors

The matrix GULPS needs represents a local two-qubit gate between two canonical
gates. In the magic basis, local $SU(2)\otimes SU(2)$ gates act as real
matrices in $SO(4)$, which explains the output requirement in the solver API.
See [Zhang–Vala–Whaley–Sastry, Eqs. (19–20)](https://arxiv.org/pdf/quant-ph/0209120).

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

The two signs generally give different conjugacy classes in $SU(4)$, both
of which GULPS permits because it tracks global phase separately. A constructor
for the unrestricted problem must therefore account for the chosen target
class as well as provide the real witness needed for the local gate.

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

Reusing this basis avoids a second endpoint eigendecomposition in GULPS;
the implementation then checks that the factors reconstruct the original
matrix, including its phase.

## Three real spectral equations

For $M\in SU(4)$, unit-modulus eigenvalues and determinant one imply

$$
\det(zI-M)=z^4-e_1z^3+e_2z^2-\overline{e_1}z+1,
\qquad e_2\in\mathbb R.
$$

Thus exact spectral equality requires only three real quantities:
$\Re e_1$, $\Im e_1$, and $e_2$. For the negative target, $e_1$ changes
sign and $e_2$ stays the same. The complementary-minor formulas below compute
these coefficients directly from a proposed $O$.

These three quantities simplify the forward equations, but selecting a matrix
that satisfies them still requires a construction and a coverage proof.
For numerical verification, the corpus checks matrices and matched roots
directly because small coefficient residuals can conceal larger root errors
near repeated eigenvalues.

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

Because floating-point orthogonality and eigensolver errors affect this
argument, the code also checks the frame, and the corpus independently checks
the resulting spectrum. These numerical checks do not provide the rigorous
enclosures needed for an interval certificate.

The accepted eigenbasis is retained for endpoint recovery. Candidate
constructions use coefficient residuals for screening, but only the rootwise
verifier can accept a result. A rejected candidate falls through to the next
construction or to numerical recovery. Repeated-root retargeting remains a
candidate repair and must pass the same verifier.

## Complementary minors in dimension four

In dimension four, complementary minors let the coefficient evaluator compute
$e_2$ with half as many minors. This reduces the work for evaluating a proposed
matrix without resolving how to construct one.

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

The reduced sum needs 18 minors and involves no division by an eigenvalue
gap, so it also covers repeated spectra and boundary inputs. This identity
holds for arbitrary diagonal $A,B$, without a unit-determinant assumption.

`compound_residual` implements this formula with compensated summation. A
small independent test compares it with Newton's trace identities on the
explicit master matrix, using dense rotation products and both target signs.
The full corpus checks spectra and endpoint reconstruction independently.

With floating-point frames, the identity acquires errors from the
orthogonality defect and rounding in the sum. The implementation therefore
uses it to filter candidate constructions before the final frame and rootwise
checks, which remain necessary because coefficient-to-root inversion is
poorly conditioned near repeated roots.

## Previous work

Earlier studies, experiments, scripts, and data are preserved under
`docs/inverse-horn/` at Git commit
`0a48fdc0d0a0f264a20e5d07e362cd1d082544b2`, along with the former
`docs/architecture.md`.

Inspect a historical file without restoring the archive into the working tree:

```sh
git show 0a48fdc0d0a0f264a20e5d07e362cd1d082544b2:docs/inverse-horn/README.md
```

When using a result from those records, check its statement, hypotheses, and
evidence rather than relying on labels such as “accepted” or “complete.”
The current problem formulation is on this page, and the
[researcher guide](researcher.md) describes the current evaluation workflow.
