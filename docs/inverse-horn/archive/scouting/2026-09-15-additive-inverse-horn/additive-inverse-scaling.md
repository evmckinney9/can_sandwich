# What scaling already solves in additive inverse Horn

2026-09-15. Independent bounded literature review, reusing the earlier Franks
source audit. This note concerns additive inverse Horn only. It introduces no
algorithm, experiment, or mathematical claim for promotion.

## The apparent literature discrepancy

There is a published general approximate construction. It is inaccurate to
say without qualification that only dimensions two and three have constructive
algorithms.

[Queiró–Santana (2023), Sections 2–3](https://journals.uwyo.edu/index.php/ela/article/download/7539/6115/15125)
asks for Hermitian A,B with prescribed spectra and prescribed sum spectrum,
calls this inverse problem open, and identifies Cao–Woerdeman's 2018
dimension-three SDP construction as the one construction paper known to them.
It also gives an exact rank-one construction. Its complete 21-item
bibliography does not contain Franks. The text does not explicitly restrict
its general construction question to algebraic formulas or deterministic
algorithms. Accordingly, the omission cannot be explained from this source
as a deliberate exclusion of approximation or randomization.

[Gift–Woerdeman (2025), introduction, p. 304](https://journals.uwyo.edu/index.php/ela/article/download/8697/7007/24669)
explicitly cites Franks's 2018 Algorithm 1 and 2019 thesis Algorithm 5 as an
iterative numerical construction, recording failure probability at most 1/3.
It then addresses rank-two construction through two orthogonal rank-one
updates. Thus its own construction claim does not assert absence of a general
iterative method.

Gift's [locally archived thesis, Section 1.3, p. 4](https://researchdiscovery.drexel.edu/esploro/outputs/doctoral/Constructive-solutions-to-A-Horns-problem/991022058837704721)
makes the distinction especially clear: it discusses Franks, then states that
general algebraic reconstruction remains unknown. Its Appendix, pp. 136 onward,
contains MATLAB code testing a multi-summand adaptation of Franks and reports
success in its attempted random trials. Those experiments are observations,
not a stronger probability or exact-construction theorem.

**Documented conclusion:** the task formulations differ in what the output
guarantee specifies. Franks supplies a randomized approximate answer in all
dimensions. The 2023 note's literature inventory is incomplete for that task.
No explanation of the authors' awareness or reasons for omitting a citation is
established, and none is inferred here.

## Three meanings of construction that must be kept separate

Given decreasing lists alpha,beta,gamma, an existence theorem decides whether
the three spectra are compatible. A numerical inverse constructor must then
produce actual matrices for compatible data. An exact algebraic constructor
would express a solution through a specified finite symbolic procedure or
representation. Knowing one of these results does not identify the other two.

For an approximate output there are several possible contracts. One may keep
the individual spectra exact in ideal arithmetic and allow a small matrix
sum error, or preserve the sum exactly while allowing spectral error. These
are related, but a source stating the former must not be summarized as a
finite exact realization of all three spectra.

Franks's normalized contract keeps three spectra fixed and seeks

`H1 + H2 + H3 ≈ I`.

Here H1,H2,H3 are real symmetric and the approximation is measured in
Frobenius norm. Shifts and a common positive scale turn the usual
`A+B=C` data into this positive-spectrum problem; the third list corresponds
to a shifted negative of gamma. No solution matrices are supplied as input.

## The mechanism, in plain linear algebra

Write each candidate as `Hi = Ui diag(lambdai) Ui^T`. The columns of Ui describe
the directions in which the prescribed eigenvalues act. If Ui is orthogonal,
the eigenvalues are correct, but the sum generally is not the identity.

A common change of coordinates can make the sum exactly the identity: factor
the positive definite sum as `S=L L^T` and replace all three Ui by `L^-1 Ui`.
This balances the total matrix. It usually destroys the orthogonality of the
three frames, so the individual spectra are temporarily wrong.

Orthogonalize each frame by a QR factorization. This restores its prescribed
spectrum but generally perturbs the sum. Repeat these two explicitly specified
operations. The factorization choices are triangular; replacing them by a
different intuitive normalization is a different algorithm whose convergence
needs its own source or proof. No Horn facet is converted one at a time into
a rotation: the spectral lists enter all three weighted frames together.

The reason to trust this alternation is not that each visible residual must
decrease. The scaling proof uses a positive quantity called capacity. Suitable
initialization makes it positive on feasible data; an iteration whose
imbalance is still large forces a controlled capacity increase, while
normalization bounds capacity above. These bounds limit the number of such
iterations. See Franks 2018, Theorem 37 and Lemmas 38–41. Random initialization
avoids an exceptional algebraic set; it is part of the coverage argument,
not an optional convenience.

## What the Franks statements actually guarantee

| Source | Input and output scope | Probability, cost, and limitation |
|---|---|---|
| [2018 paper, Algorithm 1 and Proposition 4](https://arxiv.org/html/1801.01412v2) | Normalized rational positive spectral lists; prescribed individual spectra and sum residual at most epsilon | Stated cost O(b m²/epsilon²), failure at most 1/3 on feasible data |
| Same paper, Theorem 21; Proposition 56; Remark 57 | Operator scaling with specified marginals and finite-precision discussion | Polynomial dependence on inverse tolerance, smallest marginal inverses, and input bit size; the iteration count alone is not a count of bit operations |
| [2019 thesis, Theorem 4.2](https://sites.math.rutgers.edu/~wcf17/images/ThesisPhD_website.pdf) | Positive real lists, explicitly real symmetric outputs | Almost-sure epsilon-approximation with ideal continuous random initialization and exact arithmetic |
| Thesis Theorem 4.78, Algorithm 5, Remark 4.77 | Discrete random initialization, normalized marginals | O(epsilon^-2 m² log m) scaling steps under equation (4.18), failure at most 1/3; finite-precision analysis is sketched |

Neither theorem requires distinct eigenvalues or strict interior Horn
inequalities. General input lists can be shifted away from zero before the
positive-spectrum formulation. A feasible boundary input is therefore not
excluded merely because of repeated eigenvalues or facet equality.

The almost-sure theorem and the finite-grid theorem use different computational
models. Their probability statements are compatible. Neither says that every
fixed initialization succeeds. The error output on a randomized run also does
not, by itself, certify infeasibility. Approximation to any requested positive
tolerance is different from stopping with exact sum equality. The bounded
review did not establish a deterministic, exact, finite arithmetic construction.

The paper and thesis contain several displayed transcription inconsistencies.
The earlier [source audit](../2026-09-15-deep-machinery/scaling-review.md#source-transcription-cautions)
lists them, including the identity term in stopping tests and a sign in undoing
the normalization. This is why reproducing a theorem-backed implementation
requires reconciling formulas with the proof, rather than copying a box verbatim.

## The earlier algorithms in Franks's introduction

The earlier work does not amount to an unnoticed general inverse-Horn solver:

* **Gurvits (2004)** is the operator-scaling predecessor, used by Franks for
  doubly stochastic scaling. This review checked Franks's attribution and
  bibliography, not Gurvits's complete proof. It must not be cited here as
  directly solving arbitrary prescribed additive spectra.
* **Georgiou–Pavon (2015)**, [Theorems 5–6 and Conjecture 1](https://arxiv.org/pdf/1405.6650),
  proves a contractive construction for a positivity-improving Kraus map with
  uniform marginals. The nonuniform quantum-marginal extension is presented as
  a conjecture with numerical evidence. Theorems 5–6 and the conjecture were
  checked in primary full text.
* **Friedland (2016)**, [Theorem 4.2 and Section 5](https://arxiv.org/html/1608.05862v2),
  establishes existence for positive quantum channels and prescribed positive
  definite densities using a fixed-point theorem, with contraction/uniqueness
  results in special regimes. This is not the general additive reconstruction
  theorem. Franks's thesis explicitly says the cited convergence special cases
  do not include his general constructive Horn problem.
* **Mulmuley–Narayanan–Sohoni (2012)** concerns deciding nonvanishing of
  Littlewood–Richardson coefficients. In Franks's introduction it supports
  polynomial-time feasibility, not witness matrices. Its bibliography/title
  and Franks's use were checked; the primary paper was not independently
  audited in this bounded review.

The useful distinction is between proving convergence for the uniform
operator-balancing problem and proving it for the nonuniform marginal problem
that carries all three spectral lists. Franks's stated contribution supplies
the latter and its additive inverse-Horn application.

## Reading order and handoff

Read the thesis's Problem 4.1, Theorem 4.2, Remark 4.3, and Algorithm 1 first;
these say what is computed. Then read Examples 4.20/4.22 and the capacity
analysis for why the matrix problem becomes scaling. Finally read Section 4.5
for computational guarantees and the paper's Algorithm 1/Proposition 4 for
comparison. This order avoids mistaking a feasibility reduction for the
actual constructor or an ideal-arithmetic algorithm for tested numerical code.

No new construction is proposed here. No corpus or prototype was run. CLAIMS
is unchanged because this is source reconciliation and explanatory review.
Supplemental bibliography entries are supplied separately for the root agent's
single-writer literature update. Reading the complete rounding proof and
auditing the practical implementation remain further source/implementation
work, not claimed results of this bounded review.
