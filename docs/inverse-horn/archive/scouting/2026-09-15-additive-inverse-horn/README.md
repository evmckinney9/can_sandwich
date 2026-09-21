# Additive inverse Horn: a first course and source map

2026-09-15. User-directed literature study, before multiplicative Horn.
This is an explanatory account of existing mathematics, not a new candidate
solver or an exhaustive literature review. Source-specific details and review
limits are in the companion notes. No project claim is promoted.

## 1. What are we trying to construct?

Given decreasing real lists alpha, beta, gamma of length n, find matrices

\[
A=A^*,\quad B=B^*,\quad C=A+B,
\qquad \lambda(A)=\alpha,\ \lambda(B)=\beta,\ \lambda(C)=\gamma.
\]

The spectra prescribe the strengths of the matrices along their principal
directions. The unknown is the relative arrangement of those directions.
By a common change of orthonormal coordinates, one can fix A=diag(alpha)
and seek B=U diag(beta) U*. For real symmetric witnesses U is orthogonal.
An exact witness can always be put in this form by the spectral theorem.

Distinguish these questions throughout the course:

| Question | Required output |
|---|---|
| Feasibility | Whether any witness exists |
| One approximate witness | Matrices with a specified, checked error contract |
| One exact witness | A finite exact representation, with the permitted operations specified |
| All witnesses | A description that retains the available continuous choices |
| A lift of a supplied hive | A matrix object corresponding to that particular combinatorial object under a specified map |
| Sampling | A witness drawn according to a specified distribution |

The last three are not consequences of possessing one witness algorithm.
Nor does lack of a simple formula imply lack of an algorithm.

## 2. Why the inequalities refer to directions

For a Hermitian matrix, its largest eigenvalue is the largest quadratic form
value on a unit vector. Thus

\[
\gamma_1=\max_{\|v\|=1}v^*(A+B)v\leq\alpha_1+\beta_1.
\]

Equality forces a maximizing vector to maximize both terms: A and B share
a top eigenvector. The equation then splits into a one-dimensional piece
and its orthogonal complement. This elementary example explains why a
spectral inequality can carry information about eigenvectors.

The higher Horn inequalities use selected eigenvalue sums and intersections
of subspaces, organized by flags (nested subspaces). The Littlewood–Richardson
rules determine which incidence conditions must meet. Fulton's Theorem 5 says
that equality in a Horn inequality gives a common invariant subspace in the
Hermitian problem. His following example explicitly qualifies the real version:
it holds for n<=5 but can fail for larger n. Theorem 3/20 says the feasible
spectral triples themselves are the same over real symmetric and complex
Hermitian matrices. Equality of feasible spectra does not say their individual
solution spaces are identical. [Fulton, Sections 1, 8, 10.7](https://arxiv.org/pdf/math/9908012)

This is our first geometric lesson: inequalities are not detached numerical
restrictions. Some describe how spectral mass can fit into common subspaces;
saturation can expose a smaller matrix problem.

## 3. First complete inverse: one rank-one update

Take D=diag(alpha), initially with distinct entries. Seek D+xx^T with
eigenvalues gamma. The determinant identity gives

\[
\frac{\prod_j(t-\gamma_j)}{\prod_i(t-\alpha_i)}
=1-\sum_i\frac{x_i^2}{t-\alpha_i}.
\]

Comparing residues supplies the actual vector coordinates:

\[
x_i^2=-\frac{\prod_j(\alpha_i-\gamma_j)}
                  {\prod_{k\ne i}(\alpha_i-\alpha_k)}.
\]

For a positive update, interlacing
gamma1 >= alpha1 >= gamma2 >= ... >= gamman >= alphan makes these
numbers nonnegative. Choose square roots. Trace fixes ||x||², the one
nonzero eigenvalue of xx^T. This is the precise point at which inequalities
become a construction. Queiró–Santana obtain the same rank-one inverse
through elementary symmetric functions and their Jacobian.
[Queiró–Santana, Section 3](https://journals.uwyo.edu/index.php/ela/article/download/7539/6115/15125)

For repeated alpha values, use the squared norm of the projection of x onto
each eigenspace, rather than dividing by zero. The residue-limit formulation
in the eigensteps source below expresses exactly these projection norms.

## 4. Where interval-by-interval navigation is actually proved

Schur–Horn prescribes one matrix's spectrum and diagonal. In frame language,
one constructs vectors with prescribed lengths whose outer products have a
prescribed sum spectrum. Partial sums give a sequence of interlacing spectra,
called eigensteps.

Fickus–Mixon–Poteet–Strawn do both jobs: select valid intermediate spectra,
then reconstruct vectors. Their Theorem 1.4 gives explicit intervals for
successive choices; Theorem 3.2 gives the Top Kill construction; Theorem 2.1
recovers all matrices with the given spectrum and diagonal. Their formulation
also permits repeated eigenvalues through eigenspace projection norms.
[Primary paper, Theorems 1.3–1.4, 2.1, 3.2](https://arxiv.org/pdf/1107.2173)

This is a useful model of a complete inverse theory: admissible intermediate
choices have a proved extension rule and a proved matrix reconstruction.
Its contract differs from prescribing the spectra of two full summands.

## 5. What changes when a summand has rank two?

For a positive rank-two summand, write B=xx^T+yy^T. Prescribing ||x||² and
||y||² gives B's desired nonzero eigenvalues only if x and y are orthogonal.
Two independent rank-one constructions need not meet that condition.

Gift–Woerdeman introduce the spectrum rho after the first update, enforce
two interlacings, and enforce orthogonality through a signed square-root
equation. Their construction applies in arbitrary matrix size. Appendix
Algorithm 2 includes reductions for repeated and shared eigenvalues; signs
of the nonzero eigenvalues are also handled. The implemented intermediate
selection uses sign enumeration and numerical constrained optimization, which
must be distinguished from an explicit interval selector with a convergence
bound. [Gift–Woerdeman, Sections 4–7 and Appendix](https://journals.uwyo.edu/index.php/ela/article/download/8697/7007/24669)

The lesson is joint compatibility: one must preserve the desired spectrum of
the accumulated summand while constructing the desired total. Study the actual
rank-two mechanism before attempting to extend the rank-one recipe.

## 6. A general inverse already exists: scaling

Franks's thesis explicitly constructs real symmetric witnesses to arbitrary
positive tolerance. Shift and rescale the lists so that the task becomes
finding positive-spectrum H1,H2,H3 with sum approximately I.

Write Hi=Ui diag(pi) Ui^T. Orthogonal Ui give the right individual spectra.
For S=sum(Hi)=LL^T, replacing every Ui by L^-1 Ui makes the sum I, but
generally destroys orthogonality. QR factorization of each Ui restores its
orthogonality and hence its spectrum. Alternate these operations, testing
the sum residual at the orthogonal checkpoints. The theorem includes repeated
spectra and feasible boundaries. Its continuous-random version succeeds almost
surely; a discrete-random version gives a bounded failure probability.
[Franks thesis, Chapter 4, Theorems 4.2 and 4.78](https://sites.math.rutgers.edu/~wcf17/images/ThesisPhD_website.pdf)

Why does alternation work? Franks's proof establishes a positive quantity,
capacity, for suitable initialization on feasible data. A sufficiently
unbalanced iteration increases it by a controlled amount; normalization bounds
it above. Consequently imbalance cannot persist indefinitely. The relevant
subspace obstruction is

\[
\sum_{i,j}(p_{ij}-p_{i,j+1})\dim(R\cap F_{ij})\leq\dim R
\quad\hbox{for every subspace }R,
\]

where each p(i) is positive and decreasing, the total sum of all p(i,j) is n,
p(i,n+1)=0, and F(i,j) are generic nested j-dimensional subspaces of C^n.
Here R ranges over complex subspaces of C^n. Theorem 70 identifies these
conditions, including the total-trace equality, with additive
feasibility; Proposition 72 connects that feasibility to scaling. Thus the
existence criterion and the navigation guarantee are parts of the same
theory. This does not assert monotonic decrease of the visible sum residual.
[Franks paper, Sections 4–6.2](https://arxiv.org/html/1801.01412v2)

The output is approximate in the stated norm. An error return is not an
infeasibility certificate. Nor does the theorem prove every fixed seed works.
See [the scaling review](additive-inverse-scaling.md) for computational models,
probability, rounding, and source transcription details.

## 7. Other constructions we must understand

Cao–Woerdeman use a real-zero polynomial with three specified restrictions
encoding the spectra. In dimension three they obtain a semidefinite
feasibility problem, followed by matrix reconstruction. The explicit algorithm
produces Hermitian matrices; “real-zero” describes the polynomial property
and must not be mistaken for “real symmetric output.” Gift's thesis develops
a real symmetric dimension-three construction using real polynomial
factorization. These are substantial constructive results to learn, even
though they do not make every dimension-four coefficient-selection problem
convex. [Cao–Woerdeman](https://arxiv.org/abs/1712.02922),
[Gift thesis](https://researchdiscovery.drexel.edu/esploro/outputs/doctoral/Constructive-solutions-to-A-Horns-problem/991022058837704721)

Hives and honeycombs provide a combinatorial model with the same feasible
boundary spectra as matrix triples. Their internal coordinates and local
moves deserve explanation in their own right. The equivalence of feasible
boundaries does not specify a unique correspondence between individual
matrices and individual hives. The 2026 Moitra–Postnikov–Woodruff proof has
constructive ingredients, including block direct sums and matrix
perturbations, but retains matrix convexity as a geometric input. See the
[hive review](additive-hives.md) for the exact distinction between those
operations and a full inverse. [Primary 2026 paper](https://arxiv.org/abs/2607.06710)

## 8. What “open” must mean before we repeat it

The 2023 inverse-Horn note calls the problem open and names only the n=3
construction known to its authors. It omits Franks. The 2025 rank-two paper
explicitly cites Franks. We cannot infer the reason for the omission.
We can say the unqualified literature summary “no general constructor” is
wrong for arbitrary-accuracy randomized reconstruction.

“General algebraic construction is unknown” also needs an operation model.
For algebraic input spectra, symmetric matrix entries and prescribed
characteristic polynomials define a real algebraic feasibility problem.
General real algebraic sample-point methods supply an exact fallback in
principle. That observation does not supply a practical or explanatory Horn
construction. The interesting question must specify which efficiency,
structure, formula, or parameterization is sought. This is a distinction in
problem formulation, not a proposal to resume polynomial elimination.
Source background: the already-indexed
[Collins CAD construction](https://doi.org/10.1007/3-540-07407-4_17).

## 9. Study order and what remains unread

1. Define the output contracts; work the rank-one reconstruction by hand.
2. Learn the Schur–Horn eigenstep selector and its extension proof.
3. Learn how flags explain the Horn inequalities and their equality cases.
4. Follow Franks from that subspace criterion through capacity to a constructor.
5. Work through rank-two and dimension-three reconstruction, including the
   intermediate selectors and actual output fields.
6. Study hive geometry and matrix convexity, keeping the several inverse tasks
   distinct.

This first study has not audited every convergence estimate, every numerical
implementation, or all later literature on the topology and sampling of
solution spaces. Those topics remain part of the additive curriculum. It is
premature to claim an exhaustive account or to start a multiplicative transfer.

## Handoff

Terminal scope: additive source reconciliation and first teaching account.
Companion independent source reviews cover scaling, rank-two/polynomial
constructions, and hives; existing uninvolved reviewer contexts were reused
under AGENTS.md. No prototype or corpus computation was run. No new hypothesis,
complexity reduction, or project realization result is asserted; CLAIMS.yaml
is unchanged. The next obligation is understanding the documented additive
constructions and their proofs, not testing an invented substitute for them.

Independent synthesis review: [review.md](review.md). Its required total-trace
and complex-subspace clarification is incorporated in Section 6. Literature
metadata now records this study's reading scope. `make research-check` passes.
