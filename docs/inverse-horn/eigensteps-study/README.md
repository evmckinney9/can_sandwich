# Eigensteps: what extends, and what Horn still requires

2026-09-15. This is an additive research checkpoint. It supplies a proof-level
account of an established selector and a derived error-transfer proposition.
It does **not** supply the endpoint selector for inverse Horn, an iterative
convergence theorem, or a multiplicative realization algorithm. No novelty
or independent-review claim is made.

## 1. Why this construction

The obligation is to choose directions while preserving the possibility of
finishing. The successful construction studied here is Fickus–Mixon–Poteet–
Strawn's **Top Kill**, Theorem 3.2 of
[Constructing all self-adjoint matrices with prescribed spectrum and diagonal](https://arxiv.org/pdf/1107.2173).
Its input is a desired spectrum and prescribed squared vector lengths. It
actually selects intermediate spectra and proves that the remaining lengths
can still be realized. The proof and reconstruction are worked out below.

The rank-two Horn construction was checked first because it is closer to
our problem. In [Gift–Woerdeman](https://journals.uwyo.edu/index.php/ela/article/download/8697/7007/24669),
Theorem 4.1 characterizes a compatible intermediate spectrum by two
interlacings, a trace, and a signed scalar balance. The necessity proof starts
with an existing Horn witness and takes its intermediate spectrum. Section
4.2 then uses `fmincon` and sign trials to find that spectrum. Thus this
theorem does not supply the guaranteed selector sought here. This is a
reading of its proof, not a refutation or a new discovery about rank two.
The previous conditional rank-three reconstruction is not being reproved.

## 2. The successful extension argument

### Input and first choice

Let `lambda=(lambda_1,...,lambda_N)` and `mu=(mu_1,...,mu_N)` be
nonnegative decreasing lists. Require equal totals and

    sum_{i<=j} lambda_i >= sum_{i<=j} mu_i,  j<N.

These inequalities say that `lambda` majorizes `mu`. The output is a real
positive semidefinite Gram matrix G with spectrum lambda and diagonal mu.
Equivalently, G=F^T F gives vectors of squared lengths mu_i, with FF^T
having the nonzero eigenvalues of lambda.

Work backwards. Suppose the current spectrum is b of length n and the
remaining lengths are mu_1,...,mu_n. Choose an index k with

    b_{k+1} <= mu_n <= b_k.

Such an index exists: majorization gives b_1>=mu_1>=mu_n and, by subtracting
the (n-1)-term inequality from the equal totals, b_n<=mu_n. Define a list
a of length n-1 by

    a_i = b_i                         (i<k),
    a_k = b_k+b_{k+1}-mu_n,
    a_i = b_{i+1}                     (i>k).

This is a concrete arithmetic choice. With ties, any eligible k produces
the same shorter list. Repeat until the list has length one.

### Why this choice can be completed

First, b_i>=a_i>=b_{i+1}, including at k. This is exactly the interlacing
needed for bordering a symmetric matrix with one more row and column.

Second, a majorizes the remaining lengths. For j<k, its first j terms equal
the first j terms of b, so the old inequality applies. For k<=j<=n-1,

    sum_{i<=j} a_i
      = sum_{i<=j+1} b_i - mu_n
      >= sum_{i<=j+1} mu_i - mu_n
      >= sum_{i<=j} mu_i.

The last inequality uses **mu_{j+1}>=mu_n**. At j=n-1 the totals agree.
Thus the shorter problem satisfies precisely the same hypotheses. Induction
supplies all earlier steps. There is no appeal to an unknown frame in this
extension proof and no search for a point in an unspecified region.

This is the mechanism of the source's Theorem 3.2. Theorem 4.1 additionally
allows intervals of choices; those intervals are not needed for this single
selector. We do not import an interval theorem for the extra Horn constraints.

### Recovering entries, including multiplicities

Reverse the selected list of spectra. Start with G_1=[mu_1]. Given a real
G_{n-1}=V D_a V^T, construct

    G_n = [[G_{n-1}, V z], [(V z)^T, mu_n]].

For simple a, let p_a and p_b be the monic characteristic polynomials. Set

    z_i^2 = -p_b(a_i)/p_a'(a_i).

Interlacing makes these numbers nonnegative. The Schur complement identity

    det(xI-G_n)/p_a(x) = x-mu_n-sum_i z_i^2/(x-a_i)

proves the full characteristic polynomial: the residues match p_b/p_a,
and the trace fixes the polynomial part at infinity. Either sign of each
nonzero z_i works. We may choose positive square roots.

For a repeated eigenvalue t, cancel common factors in p_b/p_a first.
Interlacing leaves at most a simple pole. The negative residue gives the
required squared norm of z's projection onto the t-eigenspace. Put that
norm along any one unit vector in that eigenspace. All other components
there can be zero. A zero residue requires a zero projection. This proves
existence and gives an executable rule without dividing by p_a'(t)=0.

After constructing G_N, diagonalize it and factor it as F^T F, retaining
the nonzero eigenvalues and their square roots. All operations can be real.
The construction uses no genericity. The number of available physical
coordinates only enters when embedding this factor: rank(G_N) must be at
most that number. Exact algebraic inputs permit algebraic eigenvalue
isolation and linear algebra. For arbitrary reals this is an exact-real
arithmetic description, not a finite-bit input model.

### A worked example

Take lambda=(8,5,3,0), mu=(6,4,4,2). The chosen spectra, backwards, are

    (8,5,3,0) -> (8,5,1) -> (8,2) -> (6).

Bordering gives the explicit matrix

    G = [[6,       2 sqrt(2), 1,       -sqrt(2)/2],
         [2 sqrt(2), 4,     -sqrt(2), 1],
         [1,      -sqrt(2),  4,        sqrt(2)/2],
         [-sqrt(2)/2, 1,     sqrt(2)/2, 2]].

Its leading principal matrices have exactly the selected spectra, and its
diagonal is mu. Factoring G produces four vectors in R^3. Their directions
are recovered by the borders above; they were not input to the procedure.
The accompanying exact certificate verifies every characteristic polynomial.

## 3. Applying this mechanism to Horn without hiding the extra constraint

There is a direct way to use the successful construction without first
assuming compatible rho and tau. Set

    a=(alpha_1-alpha_4, alpha_2-alpha_4, alpha_3-alpha_4),
    b=(beta_1-beta_4, beta_2-beta_4, beta_3-beta_4),
    c=gamma-(alpha_4+beta_4)*(1,1,1,1).

Keep the labels A1,A2,A3,B1,B2,B3 while sorting the six numbers (a,b) into
mu. Set lambda=(c_1,c_2,c_3,c_4,0,0).

Horn feasibility implies that c is nonnegative and lambda majorizes mu:
an existing real Horn witness supplies a 4x6 matrix whose columns are the
three weighted eigenvectors of each shifted summand. Its Gram matrix has
diagonal (a,b) and spectrum lambda. For completeness, the needed inequality
follows by projecting this Gram matrix onto the j coordinate positions of
its largest diagonal entries: their sum is at most the sum of its largest
j eigenvalues. Trace gives equality of totals.

Only this input check uses Horn existence. Once it holds, Top Kill constructs
a Gram matrix, hence real blocks X,Y in R^{4x3}, with

    ||X_i||^2=a_i,  ||Y_i||^2=b_i,
    spec(XX^T+YY^T)=c.

Undo the sorting by labels. Zero columns and repeated lengths cause no
problem. This is a fully specified **relaxed construction** from endpoints.

For it to be a Horn realization one also needs

    X^T X = D_a,    Y^T Y = D_b.                 (H)

The diagonal of each identity already holds. Its off-diagonal entries are
the inner products between columns belonging to the same summand.

**This is the first extension step Top Kill does not justify.** Its proof
preserves partial sums of squared lengths. Those sums contain no information
about the inner products in (H). Choosing a residue direction freely in an
eigenspace preserves the Gram spectrum and diagonal; it need not preserve
the possibility of the labeled groups being orthogonal. The problem occurs
before any proposed transfer from a rank-two theorem to rank three.

For a concrete warning, consider the Horn-feasible commuting data

    alpha=(3,1,0,0), beta=(2,1,0,0), gamma=(4,2,1,0).

A witness is A=diag(3,1,0,0), B=diag(1,0,2,0). Drop zero columns and sort
the lengths in the labeled order (A1,B1,A2,B2), giving mu=(3,2,1,1).
One exact Top Kill output is

    G = [[3, sqrt(2), -1/sqrt(3), 0],
         [sqrt(2), 2, sqrt(2/3), 0],
         [-1/sqrt(3), sqrt(2/3), 1, 0],
         [0, 0, 0, 1]].

Its spectrum is (4,2,1,0), but the A1,A2 inner product is -1/sqrt(3).
Thus this valid output of the relaxed construction is not a Horn witness.
This checks a particular choice and explains its failure; it is not an
exclusion of all eigenstep choices or all their lifts.

## 4. A bounded proposition: turn overlap into original spectral error

The next obligation addressed here is quantitative: if a search within these
frames reduces the missing inner products, can we recover an actual Q with
an error bound for the original sum? A small auxiliary residual alone would
not answer this question.

**Proposition (derived here).** For any such real blocks X,Y, define

    E_A=X^T X-D_a,  E_B=Y^T Y-D_b,
    R_A=||E_A||_F^2, R_B=||E_B||_F^2.

One can explicitly construct Q in SO(4) such that B=Q D_beta Q^T has the
exact prescribed spectrum and, with A=D_alpha,

    max_i |lambda_i(A+B)-gamma_i|
        <= ||E_A||_op+||E_B||_op
        <= sqrt(2 R_A/3)+sqrt(2 R_B/3).          (1)

If the relaxed sum only matches c to ordered eigenvalue error e, add e
to the right side. No spectral gap, simple spectrum, or full column rank
is required. In particular R_A=R_B=0 gives an exact Horn witness.

### Proof and actual direction choices

Write G_A=X^T X and H_A=G_A^{1/2}. Construct an isometry U_A from R^3
into R^4 with X=U_A H_A. To do so, diagonalize G_A. For a positive
eigenpair (g,z), map z to Xz/sqrt(g). These images are orthonormal. For
zero eigenvalues, complete the images to three orthonormal vectors in R^4,
for example by projecting standard coordinate vectors and applying
Gram–Schmidt. There is room because 4>=3. This handles rank deficiency
without applying a nonexistent inverse square root. Construct U_B likewise.

Define

    Ahat=alpha_4 I+U_A D_a U_A^T,
    Bhat=beta_4 I+U_B D_b U_B^T.

These have the required spectra. Also

    XX^T=U_A G_A U_A^T,
    Ahat-(alpha_4 I+XX^T)=-U_A E_A U_A^T.

Since U_A is an isometry, the difference has exactly the nonzero singular
values of E_A; the same holds for B. The triangle inequality and the
min–max eigenvalue bound prove the first inequality of (1).

Each E is symmetric 3x3 with trace zero. If t is an eigenvalue of largest
absolute value, the other two sum to -t. Their squared sum is at least
t^2/2. Thus ||E||_F^2>=3t^2/2, proving the second inequality.

Complete U_A and U_B to proper orthogonal matrices T_A,T_B by adding
one column each and fixing its sign. Then

    Q=T_A^T T_B.

Conjugating Ahat+Bhat by T_A^T fixes A=D_alpha and preserves the sum
spectrum. This is the requested original-coordinate construction. With
fewer than three active columns, the same argument gives the sharper
factor sqrt((r-1)/r) for a block with r>=2 columns; a one-column block
has zero defect.

The useful feature is that there is no division by a smallest positive
strength in the error bound. Numerical implementations still need certified
linear algebra and outward error bounds; floating-point SVD alone is not
such a certification.

## 5. What selection remains

The missing equations have a nonnegative scalar measure:

    R_A=2 sum_{i<j} <X_i,X_j>^2,
    R_B=2 sum_{i<j} <Y_i,Y_j>^2.

Equivalently R_A=tr((XX^T)^2)-sum_i a_i^2, and similarly for B.
These are elementary Gram identities (related to the usual frame
potential), not a new feasibility theorem.

Let F be the set of all labeled real 4x6 frames with these column norms
and total spectrum c. It is closed and bounded, hence compact. It is
nonempty by the constructive procedure above. Horn feasibility implies
that the minimum of R_A+R_B on F is zero, since a Horn witness belongs to
F. This existence argument does not locate a minimizer. In particular it
does not show that local minimization, alternating corrections, or a
particular path through eigensteps can find one. Correcting the two blocks
as above generally changes the total spectrum, so repeating the correction
has no proved feasibility invariant.

The concrete remaining lemma for an iterative route is: construct from the
endpoints, for every epsilon>0, a frame in F with

    sqrt(2 R_A/3)+sqrt(2 R_B/3) <= epsilon,

with proved termination and a stated arithmetic/error model. For an exact
limiting Q, further control ensuring convergence of the directions is
needed; unrelated epsilon-witnesses do not by themselves specify that limit.
An exact zero-defect selector would instead finish the task directly.

The successful eigenstep parametrization can supply coordinates for F,
but its extension inequalities do not control this objective. Merely
instructing an optimizer to minimize it would repeat the rank-two selection
gap. No descent or global-selection theorem is established here.

## 6. Evidence and checkpoint

- **Source result studied:** Top Kill's extension invariant, its proof, and
  real reconstruction through bordering; rank-two Theorem 4.1 and its
  Section 4.2 selection limitation. No exhaustive literature survey.
- **Derived here:** the labeled two-block relaxation and the explicit
  polar correction with original-spectrum bound (1). These are elementary
  deductions; no novelty claim.
- **Exact checks:** `certificate.py` verifies the worked Gram matrices,
  their intermediate spectra, the labeled failure, and full-rank and
  rank-deficient polar identities. It does not certify universal coverage;
  the stated bounds rest on the proof above.
- **Independent review:** none for this checkpoint. The author checks must
  not be described as independent review. Archived reviews retain their
  original scope; no claim-registry promotion is made.
- **Construction now specified:** an endpoint-to-relaxed-frame procedure,
  followed by an explicit proper-orthogonal correction and a computable
  error bound. The bound need not meet a requested tolerance.
- **Remaining gap:** selecting a zero- or sufficiently small-defect frame
  with guaranteed extension or termination. The general inverse Horn
  construction remains unresolved in this investigation.
- **Why the next step follows:** a selection theorem reducing this same
  defect would transfer immediately to the original spectral target by
  (1). A smaller residual from an unproved numerical search would not
  establish that theorem.

Reproduce the exact checks from the repository root:

    python3 crates/can_sandwich/docs/inverse-horn/eigensteps-study/certificate.py

This checkpoint changes no solver and establishes no additive-to-
multiplicative transfer for `can_sandwich`.
