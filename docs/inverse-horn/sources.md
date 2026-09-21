# Sources and reading limits

These links identify the primary literature used in the investigation.
Reading scope is recorded in the archived reviews; this is not a claim of
an exhaustive literature survey or proof that no later construction exists.

| Source | What it supports; what it does not supply here |
|---|---|
| [Horn, Eigenvalues of sums of Hermitian matrices (1962)](https://msp.org/pjm/1962/12-1/pjm-v12-n1-p21-s.pdf) | The conclusion after Theorem 13 establishes sufficiency for n<=4. The existence proof is not the sought reconstruction recipe. |
| [Fulton, Eigenvalues, invariant factors, highest weights, and Schubert calculus](https://arxiv.org/abs/math/9908012) | Feasibility, real/complex existence, and equality/common-subspace results. Real equality-case assertions have dimension qualifications; see the source review. |
| [Queiro–Santana, The inverse problem for the sum of Hermitian matrices (2023)](https://journals.uwyo.edu/index.php/ela/article/view/7539) | Poses reconstruction and gives rank-one treatment. Its construction literature inventory omits Franks, so it cannot support a claim that general approximate construction is unknown. |
| [Franks, Operator Scaling with Specified Marginals](https://arxiv.org/abs/1801.01412) | General randomized approximate additive reconstruction with stated arithmetic/probability/tolerance contracts. Does not give finite exact sum equality. |
| [Franks, doctoral thesis (2019)](https://sites.math.rutgers.edu/~wcf17/images/ThesisPhD_website.pdf) | Explicit real symmetric formulation and algorithmic details; ideal continuous initialization and finite-grid probability statements are different models. |
| [Gift–Woerdeman, rank-two construction (2025)](https://journals.uwyo.edu/index.php/ela/article/view/8697) | Rank-two reconstruction and orthogonality balancing; Section 9 names rank three as a next step. Acknowledges Franks. Exact formulas and numerical intermediate selection are distinct. |
| [Gift, Constructive Solutions to A. Horn's Problem, Determinantal Representations, and Real Fejer–Riesz Factorization (2025)](https://researchdiscovery.drexel.edu/esploro/outputs/doctoral/Constructive-solutions-to-A-Horns-problem/991022058837704721) | Dimension-three and rank-two constructions, Franks discussion and numerical implementation. Not evidence of absence of every general algorithm. |
| [Cao–Woerdeman, Real zero polynomials and A. Horn's problem](https://arxiv.org/abs/1712.02922) | Interpolation/Hankel equivalence and Hermitian determinantal construction; implemented dimension-three method. Real symmetric existence is a separate representation-theorem implication. |
| [Pak–Vallejo, Combinatorics and geometry of Littlewood–Richardson cones](https://www.math.ucla.edu/~pak/papers/liri91.pdf) | Sections 3–4 and Theorem 4.1: LR boundary conventions and linear hive correspondence. No matrix inverse. |
| [Knutson–Tao, The honeycomb model of GL_n(C) tensor products I](https://arxiv.org/abs/math/9807160) | Hive/honeycomb correspondence and feasibility geometry; not a direct inverse to real matrix pairs. |
| [Bercovici–Li, On the Calculation of Hives Associated to Sums of Selfadjoint Matrices (2024)](https://imar.ro/journals/Revue_Mathematique/pdfs/2024/3-4/4.pdf) | A proposed compression-trace hive proof has a failed projection-recombination step. That is not a counterexample to the whole hive conjecture or every inverse map. |
| [Fickus–Mixon–Poteet–Strawn, Constructing finite frames of a given spectrum and set of lengths](https://arxiv.org/abs/1107.2173) | Constructive eigensteps for the Schur–Horn/frame problem. Its vector constraints differ from mutually orthogonal rank-three update directions. |
| [Davidson–Djokovic, Tridiagonal forms in low dimensions](https://www.math.uwaterloo.ca/~krdavids/Preprints/DavDjoktrid.pdf) | Theorem 1.1 gives simultaneous unitary tridiagonalization. It does not keep the sum diagonal; real orthogonal simultaneous tridiagonalization can fail. |
| [Fasino, Orthogonal Cauchy-like matrices](https://air.uniud.it/handle/11390/1234449) | Repository abstract and metadata located; no theorem from the full proof was imported. The coupling used here was derived and reviewed directly. |
| [Collins, Quantifier elimination for real closed fields by cylindrical algebraic decomposition](https://doi.org/10.1007/3-540-07407-4_17) | General exact algebraic fallback, not a new Horn-specific contribution. |
| [Helly, convex intersection theorem (1923)](https://eudml.org/doc/145659) | Finite planar Helly used in R390; original retrieval was blocked. The attempt includes an elementary Radon-based proof of the needed finite version. |

The earlier parabolic, symplectic and geometric literature is documented in
the [machinery review](archive/scouting/2026-09-15-deep-machinery/README.md),
its four source inventories, and independent applicability reviews. No
general multiplicative constructor was obtained from that material.

## Corrections that must survive future summaries

The [eigensteps study](eigensteps-study/README.md) additionally checks the
Top Kill extension proof in Fickus–Mixon–Poteet–Strawn, Theorem 3.2, and
Gift–Woerdeman, Theorem 4.1 and Section 4.2. In the latter, existence of
compatible intermediate spectra is extracted from a Horn witness; the
implementation finds them with `fmincon` and sign trials. It is not a
guaranteed endpoint selector. The arXiv PDF linked for eigensteps is titled
*Constructing all self-adjoint matrices with prescribed spectrum and diagonal*.

1. Rank-two perturbation is not the same as matrix dimension two. A generic
   n4 summand becomes rank three after subtracting its smallest eigenvalue.
2. A three-dimensional hive/real-fiber match is not unique to n4.
3. The entrywise-square map Q to Q∘Q loses finite sign choices locally on
   nonzero-entry sheets, not an additional continuous angle.
4. The compression-trace proof failure is not a general hive-map refutation.
5. General approximate construction exists. General exact computability for
   algebraic inputs also exists. The desired new contribution needs a more
   precise contract than "constructive inverse Horn is open."
