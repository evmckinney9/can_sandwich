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

## Critical points and strata

The constructions in the solver follow a stratification of the fiber by the
stabilizer of the triangle. The measurements below were taken on the
production witnesses of 16,000 corpus rows, 2,000 from each construction.

### Rank of the spectral map

Let $F:SO(4)\to\mathbb R^3$ send $O$ to the three real coefficients
$(\Re e_1,\Im e_1,e_2)$ of $M=AOBO^T$, and let
$\mathfrak s\subset\mathfrak{su}(4)$ be the Lie algebra of the common
stabilizer of $A$ and $OBO^T$. For the product map $\mu(A,B)=AB$ of the
quasi-Hamiltonian space of Alekseev, Malkin, and Meinrenken, the image of
$d\mu$ at a point is $\mathfrak s^\perp$. The characteristic-polynomial map
$\chi$ has $d\chi$ of rank $d_C-1$ at $C=AB$, where $d_C$ is the number of
distinct eigenvalues of $C$, and $d\chi$ only sees the traces against the
spectral projectors $P_i$ of $C$ (Kostant). Composing the two gives

$$
\operatorname{rank} dF = (d_C-1)-\dim\left(\mathfrak s\cap
\operatorname{span}_0\{iP_i\}\right),
$$

where the span is traceless. For a regular target the span is the whole
centralizer of $C$, which contains $\mathfrak s$, so the rank is
$3-\dim\mathfrak s$. The derivation assumes that $dF$ on the real locus has
the rank of its complex counterpart; the corpus supports that assumption.

Central differences with step `1e-5` give the rank, with singular values
below `1e-6` of the largest treated as zero. The stabilizer is the null space
of $X\mapsto([X,A],[X,OBO^T])$. On rows whose spectral gaps are all below
`1e-14` or above `1e-4`, the formula holds for 4,933 of 4,938 regular
witnesses and 8,258 of 8,409 witnesses with repeated spectra. The remaining
repeated cases are inputs with triple or quadruple degeneracy, where the
finite-difference rank is unreliable.

### Constructions as strata

| Construction | Witness structure | Stabilizer | Rank of $dF$ | Observed |
|---|---|---|---|---|
| Vertex | Four 1×1 blocks | 3-torus | 0 | 98.2% rank ≤ 1 |
| Edge | 2+1+1 blocks | 2-torus | 1 | 98.4% rank ≤ 1 |
| Face | 2+2 blocks | circle | 2 | 97.5% rank 2 |
| Klein, Interior | Irreducible | discrete | 3 | 98 to 99% rank 3 |

Vertex, edge, and face frames are reducible triangles: $A$, $OBO^T$, and
their product preserve a common coordinate splitting. They are critical
points of $F$, which explains two observations. Their targets have closed
forms because the problem splits into lower-rank problems. Refinement
starting at them stalls, because a root error $e$ at a critical point needs
a displacement of order $\sqrt e$. The radical and rank-one constructions
cover the second axis of the stratification, non-regular conjugacy classes,
where a repeated class is a low-rank perturbation of a scalar matrix.

The numerical fallback mostly handles perturbations of these strata:
1,820 of the 2,000 sampled rows that reach it have a spectral gap between
`1e-14` and `1e-4`.

### A single-qubit middle gate does not suffice

The one-sided family $O=L_qP$, a middle gate $u\otimes I$, is closed under
the symmetries of the problem. Right multiplication reduces to it through
the diagonal sign $\mathrm{diag}(1,-1,-1,-1)$, conjugation by a permutation
preserves the pair of normal $SU(2)$ factors, and transposition exchanges
$c$ and $g$. The Klein construction searches this whole family in closed
form. An independent search over all 24 permutations, both target signs, and
12 quaternion starts each found a one-sided solution for 59 of 60 Klein rows
but for only 2 of 60 radical rows, 6 of 60 interior rows, 17 of 60 edge rows,
and 14 of 60 face rows. The family $u\otimes u$ fixes a magic-basis vector,
so it reaches only targets containing a product $a_ib_j$. Neither
three-dimensional family covers the feasible region.

### The numerical fallback works on 1+3 walls

Of the 2,744 rows that reach numerical recovery, 2,338 have a routed product
$a_ib_j$ within `1e-10` of a target root, possibly after the central sign.
Such a target lies on the 1+3 wall, where the triangle can keep coordinate
$i$ fixed, and the fixed-root search solves the complementary block. The
block is a $3\times3$ problem with a one-dimensional fiber: a unitary
$3\times3$ matrix with fixed determinant $d$ has $e_2=d\,\overline{e_1}$, so
only $e_1=a^TPb$ remains, with $P=O\circ O$.

A $3\times3$ doubly stochastic $P$ is orthostochastic exactly when the
numbers $u_j=\sqrt{p_{1j}p_{2j}}$ form a degenerate triangle,
$u_j=u_k+u_l$ for some $j$: two real unit rows with these squared entries can
then be orthogonal, and the third row is their cross product. Fixing $e_1$
cuts the four-dimensional Birkhoff polytope to a polygon, and the fiber is
the zero set of $g_j=u_k+u_l-u_j$ on that polygon. A prototype that locates
sign changes of $g_j$ on a grid and bisects along the crossing edge found
candidates for 1,649 of the 2,338 wall rows. The remaining rows have a zero
set that touches the polygon only tangentially: their $e_1$ lies on the
boundary of the $3\times3$ reachable set, a deeper 2+1+1 stratum, where the
square-root behavior of critical values returns. The single chart
$G_{01}G_{12}$ forces $p_{20}=0$ and reaches only 1,137 rows.

### Exact strata and the multiplicity threshold

The radical constructions assume exact repeated roots. Among rows whose merged
roots coincide to `1e-14`, only six that reached them fell through to later
constructions, two of them infeasible rows since removed from the corpus. The earlier
multiplicity test merged roots within `1.5e-8`, so it also sent inputs up to
that distance from a stratum to these formulas. All 2,307 other radical
failures were such inputs. Radical still succeeds on most inputs within
`1e-11` of a stratum, after refinement, and fails on most inputs farther away.
The test now merges roots within `1e-11`. The near-rank-one construction keeps
the looser triple test, because it is designed for spectra near a triple.

### A homotopy over the feasible polytope

For fixed $c$ and $g$ the reachable target classes form a convex polytope in
alcove coordinates, and every point of it has a real witness. This gives one
construction that needs no chart atlas. Choose a random frame $O_0$; its
target $t_0$ is an interior point with a known witness. Move $t$ along the
segment from $t_0$ to the requested target and carry the witness with
minimum-norm Gauss-Newton steps on $(e_1,e_2)$. The segment stays feasible by
convexity, and its open part stays off the alcove walls. Along it, the
spectral map loses rank only at reducible frames, whose images meet the
segment in a curve inside a three-dimensional fiber, so a random start misses
them with probability one. Only the endpoint can be singular. There the
lifted path has an expansion in $\sqrt{1-s}$, so the path is sampled at
$1-s=10^{-3},\,2.5\times10^{-4},\,6.25\times10^{-5}$, extrapolated to $s=1$
by Richardson's rule for that variable, and finished by Newton on
$V^TM(O)V=\mathrm{diag}(\tau)$ jointly in the frame and the eigenbasis.

A prototype with finite-difference Jacobians, at most four random starts and
both central signs, was run on 200 corpus rows from each construction. Without
the endgame, tracking reached the target within `1e-10` on all Klein and
interior rows but on only 83 radical rows and about 150 edge and face rows;
the losses were all within `1e-3` of the endpoint. With the endgame the
counts below `1e-13` were 200 Klein, 200 interior, 200 radical, 199 face, 194
rank-one, 174 vertex, and 127 numerical rows, and 40 of the 46 edge rows
completed before the run was stopped. Every remaining failure is a row whose
target is a corner or a repeated class, where the endpoint expansion has a
larger exponent than the one assumed. The prototype takes 0.5 to 4 ms per
row against 7 µs for the production cascade, so it is evidence about
coverage, not a replacement. It shows that the cascade's many constructions
are accelerators for one mechanism, and that the mechanism's only hard part
is the endpoint singularity that the strata above describe.

### Rows with residual errors near `1e-13`

After the polishing changes of September 2026, 179 corpus rows keep actual
spectral errors above `2e-14`, all between rows 1,068,617 and 1,090,479, and
175 of them have a pair of roots closer than `1e-6` in $c$, $g$, or $t$.

They are feasible. `tests/margins.py` evaluates the complete quantum-Horn
inequality list of `tests/generate.py` in exact rational arithmetic on the
binary64 inputs: over the whole corpus, only four historical regression rows
violated an inequality by more than `1e-15`, and they have since been removed
(see the corpus revision in the [optimization record](optimization.md)). The
other 8,103 negative margins are between `-4e-16` and zero, rounding on rows
generated on the boundary. Of the
179 rows, 118 have tight inequalities of ranks 1, 2, and 3 at once, 15 have
no tight inequality at all, and 42 have exact margins near `-1e-16`.

The residual errors are therefore failures of refinement, not of the inputs.
Newton at 50 digits from the production witness stops at residuals between
`6e-14` and `2e-13` on six of these rows, and random offsets up to `1e-3` do
not help: the witness sits in a basin whose floor is above zero, at a critical
point of the spectral map where a nearly repeated pair makes the fiber
singular. The polytope homotopy above, which starts from a random regular
point instead, reaches errors below `1e-14` on 77 of the 179 rows and below
`1e-13` on 97; its 47 outright failures are rows whose target is itself a
repeated or corner class, where its endgame assumes the wrong exponent.

### The generic problem has degree 16 and is not solvable by radicals

Write $O = L_{q_+} R_{q_-}$ with unit quaternions, so that $SO(4)$ acts on
self-dual and anti-self-dual 2-forms by $(k_+, k_-) \in SO(3)\times SO(3)$.
Numerically, to `1e-14` over thousands of random frames: $e_1$ is affine in
the entries of $k_+\otimes k_-$, and $e_2$ is affine in the entries of
$k_+\otimes k_-$, $k_+\circ k_+$, and $k_-\circ k_-$, and in nothing lower.
Fixing $k_-$ therefore leaves two real equations affine in $k_+$ and one
quadratic in $k_+$. In quaternion coordinates that is the unit sphere, two
quadrics, and a quartic in $\mathbb{C}^4$: Bézout gives 32 points, or 16
rotations after $q_+ \sim -q_+$.

For random data and a random real $k_-$, Newton from 4,000 random complex
starts found exactly 32 complex solutions, 16 rotations, of which 4 were real
in one instance and 2 in another, always including the planted witness. So
the generic slice has degree 16 and Bézout is attained. Tracking the 16
rotations around complex loops in $k_-$ gave permutations that generate a
group of order 16!/2 on one instance and 16! on another: the monodromy is
the full symmetric group $S_{16}$, which is not solvable. Consequently there
is no formula in radicals for $k_+$ in terms of $k_-$ and the data: every
closed-form route in the solver (vertex, edge, face, Klein, the three-Givens
charts) is a locus where this degree collapses, and no single formula can
cover the generic fiber.

### The open question: a finite list of sections

Call a *section* an algebraic family $\Sigma \subset SO(4)$ of dimension 3
on which the three spectral equations reduce to one univariate polynomial
with coefficients rational in the data. The Klein family $\{L_qP\}$ is a
section of degree 4; each three-Givens chart is a section of degree 6; the
generic slice above is a section of degree 16. On a section, a witness is a
real root of that polynomial, and a real root can be isolated and refined to
any precision from exact coefficients. That is what "exact" can mean for
interior targets, and it needs no iteration whose convergence is assumed.
A target close to a wall is not a separate case: its root is ill-conditioned
but still a root, and extended precision on exact coefficients resolves it.

What is not proved is coverage: that some fixed finite list of sections
meets every interior fiber in a real point. Topology gives no help. The fiber
$F_t = \varphi^{-1}(t)$ of the spectral map $\varphi: SO(4) \to P$ is a
compact 3-manifold, but across a facet of the polytope its dimension drops
to 1, since a witness on a rank-1 facet is a fixed coordinate plus a
$3\times3$ problem with two equations. So $\varphi^{-1}$ of a ray from $t$ to
the facet is a 4-chain with boundary $F_t$, the fiber is null-homologous,
and no intersection number can force a real point of $F_t \cap \Sigma$. Near a
facet the fiber has diameter about $\sqrt{\delta}$ around the reducible
curve, which is why the block-structured sections (vertex, edge, face) meet
it there, and why far from the walls the large fibers are met by Klein and
the charts. Empirically, the closed-form sections miss rows only near a wall
or a repeated class; numerical recovery solves those, and the trace-hull depths
below quantify how near.

The object to study is $D(c,g,t) = \{k_- \in SO(3) : \text{the slice over } k_-
\text{ has a real root}\}$, the projection of the fiber. A section is a
choice of $k_-$ (or a curve or surface of them) as a function of the data;
coverage is the statement that the choice lands in $D$. The Klein family is
six points of $SO(3)$ and lies in $D$ for about a fifth of the corpus.
Two questions whose answers would settle coverage for a class of sections:
whether $D$ is the interior of a semialgebraic body with an explicit
description (its $e_1$ part is membership of a point in the image of $SO(3)$
under a linear map to $\mathbb{R}^2$), and whether a section built from the
data itself, such as the frames of the nearest wall's split witness, can be
shown to intersect. Neither is known.

### The slice as a spectrahedral section

For fixed $k_-$, write $O = L_qR_p$ with $p$ fixed. The entries of $O$ are
linear in $q$, so $e_1 = \sum_{ij} a_ib_j O_{ij}^2$ is a quadratic form in
$q$: $e_1 = \langle N(p), Q\rangle$ with $Q = qq^{\mathsf T}$, a rank-one
positive semidefinite matrix of trace one, and $N(p)$ a complex symmetric
$4\times4$ matrix depending on the data and $p$. The rank-one matrices of
trace one are exactly the extreme points of the spectrahedron
$\{Q \succeq 0,\ \mathrm{tr}\,Q = 1\}$, which is $SO(3)$ in the guise of
$\mathbb{RP}^3$. The two real $e_1$ equations are therefore a linear section
of this spectrahedron, and $e_2$ is one quadratic form in $Q$ on it. The
Klein chart is the case $p$ a unit, where $N(p)$ is diagonal: the section
then depends only on the diagonal of $Q$, which ranges over the simplex, and
the rank-one condition costs nothing, which is why its degree is 4 while a
generic $p$ gives degree 16.

Pataki's bound says an extreme point of a spectrahedron cut by two
equations has rank $r$ with $r(r+1)/2 \le 3$, so rank at most 2. The
rank-one points of the section form a curve (the elliptic curve of the
degree count), and the coverage question for a section is whether the
quadric $e_2$ meets that curve in a real point. Stated this way, the open
question is about rank-one points of a spectrahedral section, a question
with a literature of its own, rather than about the fiber of a spectral map.

### Where the Klein chart works, and where nothing closed-form does

Run on its own, the Klein chart with its six coset representatives and both
central signs solves 1,040 of 1,500 Haar rows (69%), far more than its 22%
share when the cascade runs it after the support strata. Its region has an
exact necessary condition: $e_1$ of the target must lie in one of the six
quadrilaterals $\mathrm{conv}\{C_m(P)\}$, and no row is solved without it.
The condition is not sufficient: 225 rows satisfy it and have no admissible
real root of the quartic, and 235 fail it.

The relevant coordinate is the depth of the target's trace inside the
convex hull of the 24 permutation traces $\sum_i a_i b_{\pi(i)}$, measured as
a fraction of the hull's diameter. Klein-solved rows have median depth 0.24;
rows that satisfy the hull condition but fail the quartic have 0.18; rows
that fail the hull condition have 0.13. The 240 corpus rows with regular
spectra that no closed-form route solves have median depth 0.006, and 143 of
them lie within 1% of the boundary. The uncovered region is a thin shell at
the boundary of the trace hull, where the witness is close to an extremal
permutation frame $P^*$, and the nearest $P^*$ ranges over all 24
permutations across those rows.

### Charts at a vertex are chosen by Carathéodory

At a permutation frame $P^*$ the product $A P^* B P^{*\mathsf T}$ is
diagonal, so the first derivative of the spectral map vanishes. Write
$O = P^*\exp(X)$ with $X = \sum_k X_k E_k$ over the six coordinate planes.
With $B' = P^*BP^{*\mathsf T}$ diagonal and $B'(X) = e^{X}B'e^{-X}
= B' + [X,B'] + \tfrac12[X,[X,B']] + \dots$, every first-order term is a
trace of a diagonal matrix against a commutator with zero diagonal and
vanishes, and the second-order terms are
$$e_1:\ -\sum_{i<j} X_{ij}^2\,(a_i - a_j)(b'_i - b'_j),\qquad
\mathrm{tr}(A[X,B']A[X,B']) = 2\sum_{i<j} a_ia_j (b'_i - b'_j)^2 X_{ij}^2,$$
$$\mathrm{tr}(AB'A[X,[X,B']]) = -2\sum_{i<j} X_{ij}^2\,(b'_i - b'_j)(a_i^2b'_i - a_j^2b'_j),$$
using $[X,B']_{ij} = X_{ij}(b'_j - b'_i)$ and
$[X,[X,B']]_{ii} = -2\sum_j X_{ij}^2 (b'_i - b'_j)$. Since
$e_2 = \tfrac12[(\mathrm{tr}\,M)^2 - \mathrm{tr}\,M^2]$, its second-order term
is a combination of these, so all three real spectral coordinates have
Hessians diagonal in the plane coordinates $X_{ij}$. Numerically the
off-diagonal entries are at finite-difference noise, and the $e_1$ diagonal
matches the formula. So the second-order map is
$$X \mapsto \varphi(P^*) + \sum_{k=1}^{6} X_k^2\, h_k,\qquad h_k\in\mathbb{R}^3,$$
whose image is the polyhedral cone generated by the $h_k$: the tangent cone of
the reachable set at the vertex. By Carathéodory's theorem every point of that
cone is a nonnegative combination of three of the $h_k$, and a three-Givens
chart at $P^*$ on those three planes is, in its own coordinates
$v_k = \cos^2\theta_k \approx 1 - X_k^2/2$, a small perturbation of the
linear map $x \mapsto \sum x_k h_k$ on the simplicial cone. For an offset in
the interior of such a cone the inverse function theorem gives a real root of
the chart's degree-6 polynomial. So near every vertex value, the target
selects its chart: the three $h_k$ whose cone contains the offset. The solver's
schedule of 16 charts chosen by measured coverage is replaced by a rule with
a proof, and the shell at the boundary of the trace hull is its domain. On the
240 regular rows that the scheduled charts miss, 221 have their offset inside
the second-order cone of some vertex, and the chart the rule selects has a
real root on 198 of them; a probe of all 384 (permutation, plane triple)
charts on 40 of these rows found roots for 16, so the rule finds charts the
atlas search does not. The 19 rows outside every vertex cone lie near an
edge or face of the trace hull, where the base frame is the reducible family
of that stratum (a one- or two-Givens frame) rather than a permutation.

The same structure holds one stratum up. At a one-Givens frame the spectral
Jacobian has rank 1, its direction the edge angle; the null space has
dimension 5 and the cokernel dimension 2, and the two cokernel Hessians
restricted to the null space commute (commutators at finite-difference
noise in every trial). They therefore share an eigenbasis, the transverse
second-order map is again $x \mapsto \sum_k x_k h_k$ with five vectors
$h_k \in \mathbb{R}^2$, and Carathéodory in the plane selects two of the
five eigen-directions. At a two-Givens frame the rank is 2 and the cokernel
is a line, so one transverse direction suffices. So near every stratum of
the boundary of the reachable set the chart is: the stratum's own angles,
which enter linearly, plus as many transverse directions as the codimension,
chosen by Carathéodory among the eigen-directions of the commuting cokernel
Hessians. At a vertex those eigen-directions are the coordinate planes, which
is why three-Givens charts are the right family there.

### Tight inequalities certify the split

On 20,000 random corpus rows, every tight quantum-Horn inequality (slack
below `1e-11`) came with the signature of an invariant subspace of its rank:
a routed product $a_ib_j$ equal to a target root for rank 1 and 3 (3,760 and
3,723 rows), and a product $a_Ib_J$ over pairs equal to a product of two
target roots for rank 2 (3,528 rows), in every case. This is the practical
form of the theorem that a tight inequality forces a reducible witness. A
prototype that reads the tightest rank-1 signature, fixes that coordinate,
solves the complementary $3\times3$ problem by minimum-norm Gauss-Newton,
recurses when the block has its own tight signature, and finishes with a
full $SO(4)$ polish reaches errors below `1e-15` on rows where the
production cascade stalls at `3e-14`.

### Toward one mechanism: the atlas as the solver

The results above suggest a solver with two exact mechanisms and no
measured schedules: exact Horn margins with recursive splits on tight walls,
and for interior targets a three-Givens chart selected at the nearest
stratum of the trace hull, each chart one univariate polynomial of degree at
most 6 whose real roots are isolated from exact coefficients. Two necessary
conditions for the interior half hold on the corpus: the target lies inside
the corner hull of some chart for 300 of 300 Haar rows and 239 of 240 of the
schedule-missed regular rows (the last on a hull boundary to rounding), and
a dense multistart probe of all 384 charts found a real chart root for 40 of
40 Haar rows, including those the cascade solves by Klein. The same probe on the
schedule-missed regular rows found a chart root for 17 of 24, and on
repeated-spectrum interior rows (Radical route, no tight inequality) for 5
of 12; two-Givens frames solve 1 of those 12. For every uncovered row the
production witness was fitted against all 384 charts and lies in none (best
entrywise fit 0.09 to 0.39), so the atlas is genuinely incomplete: it is the
right family near the vertices of the trace hull, not a universal one. The
radical formulas therefore remain the section for repeated spectra, and the
interior needs a family beyond three-Givens charts; the generic slice at a
fixed $k_-$, one degree-16 polynomial with exact coefficients, is the
candidate, and its coverage is the question of how a fixed finite list of
$k_-$ meets the projection $D$ of the fiber.

Twelve fixed random $k_-$ answer that question. On 28 Haar rows the number
of the twelve slices with a real root is 9 to 12 for most rows, but 0, 1, 3
and 3 for four of them; on ten of the uncovered rows above it is 0 for six
and at most 3 for the rest. So $D$ fills most of $SO(3)$ for a target deep in
the reachable set and shrinks toward the boundary, as the fiber's diameter
$\sqrt{\delta}$ does, and no fixed finite list of $k_-$ is universal either.
Coverage near the boundary must come from $k_-$ adapted to the data, which
is what the stratum charts provide.

### What the cascade is, and what replaces it

Three universal families were tested and each fails in one regime: formulas
in radicals (monodromy $S_{16}$), the three-Givens atlas (incomplete away
from the vertices and for repeated spectra), and fixed slices (empty near the
boundary). What holds is a two-regime structure governed by one number, the
distance $\delta$ of the target from the boundary of the reachable set,
measured by the smallest quantum-Horn slack or the trace-hull depth:

- $\delta = 0$: a tight inequality certifies an invariant subspace, and the
  problem splits into exact lower-dimensional ones.
- $\delta$ small: the fiber is a small hypersurface around the reducible
  frame of the nearest stratum, and Carathéodory on the commuting cokernel
  Hessians selects the chart, of degree at most 6.
- $\delta$ large: the fiber is large, $D$ is most of $SO(3)$, and Klein
  (degree 4) or any fixed slice (degree 16) has a real root.

Each regime is one exact univariate root isolation from exact coefficients.
The current cascade of eight routes ordered by measured coverage is this
rule discovered empirically, with search where the rule should decide. The
simplification available is to replace the schedule by the rule; what a
proof of completeness still needs is the radius of the small-$\delta$
regime, where the second-order model of the fiber stops being accurate
before the fixed slices begin to have real roots.

### Where the precision floor is

Measured on 2026-09-24, with GULPS's tolerances consolidated to three scales
(`INPUT_ATOL`, `CLASS_TOL`, `MEMBERSHIP_TOL`; see gulps
`crates/core/src/lib.rs`). On the corpus, accepted witnesses have spectral
error `1e-15` at the median and `1.05e-14` at the 99.9th percentile. GULPS
factors and stitches them into circuits whose entrywise error against the
target is below `1e-14` on its fixed benchmark workloads; that is about ten
roundings of a 4×4 product chain, the floor for binary64.

Two things stand between this and a solver that is exact to working
precision, and both meet at the acceptance ceiling `SPECTRAL_TOLERANCE = 1e-13`:

1. The 179 rows above are accepted with actual errors between `2e-14` and
   the ceiling. They are
   feasible, so the ceiling can drop only when the solver reaches `1e-14` on
   them. That needs refinement that does not stall at a singular fiber: a
   restart from a regular point (the homotopy), or a construction that reads
   the tight inequality and solves the split problem it certifies, since a
   tight inequality of rank $r$ forces an invariant $r$-dimensional subspace.
2. GULPS derives its reachability slack from the ceiling
   (`MEMBERSHIP_TOL = SPECTRAL_TOLERANCE / 2π`), so the ceiling also sets how
   far outside a region a target may lie before GULPS reports it unreachable.

The grader's infeasibility certificate uses four rank-two inequalities. The
complete list is already in `tests/generate.py`; moving it into the grader
and the solver's precheck, with a threshold derived from the ceiling (a
violation $\delta$ in turns forces a root error of at least about $0.23\,\delta$,
from the coefficient bounds of the inequalities), makes the certificate
exact for every input, not only the four infeasible rows since removed
from the corpus.

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
