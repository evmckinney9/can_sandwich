# Constructive machinery for the rank-four atomic problem

Prototype follow-up: [R0386](../../attempts/2026-09-15-R386-parabolic-metric-prototype/README.md)
returns a practical NO-GO for the tested direct finite-domain metric method.
The source-study recommendation below preceded that experiment. Its
published correspondences remain valid at their stated scopes; the pilot
does not establish a practical constructor.

2026-09-15. Source-study synthesis; no new universal theorem or working
constructor is claimed. The study follows [research-brief.md](research-brief.md).

## Finding

The strongest connection found is **stability and balancing**. The spectral
inequalities can certify that a geometric configuration admits a balanced
representative. A balancing process then supplies matrices. For the full
compact unitary problem the configuration includes weighted flags over a
punctured sphere, and balancing means finding a compatible flat connection.

This provides a concrete explanation of how feasibility could guide
construction. My recommendation is to investigate a rank-four **real
parabolic balancing constructor**. This is a research direction supported
by published machinery, not a claim that the remaining computation is easy
or that it will meet production cost requirements.

## What the fourteen bounds retain

The repository stores four trace-zero phases through three coordinates.
There are fourteen nonempty proper subsets of four labels. Complementary
subsets give opposite sums, leaving seven forms:

```text
x, y, z, x+y, x+z, y+z, x+y+z.
```

The QLR rules combine bounds on these forms. This compresses spectral
feasibility; it does not attach seven independent frame rotations to the
seven intervals. The constructor needs a state carrying compatible
subspaces as well as their spectral bounds.

For a fixed pair and fixed target lift, the segment from a current feasible spectrum to the
target stays feasible in its Horn polytope. The immediate question is how
to lift that segment to actual frames. A Jacobian can lose rank even when
the spectral direction remains locally reachable. The independent
[geometric review](geometric-review.md) identifies the additional local
data needed there.

## The source chain

### 1. Additive Horn already has a balancing algorithm

[Franks's thesis](https://sites.math.rutgers.edu/~wcf17/images/ThesisPhD_website.pdf),
Theorem 4.2, explicitly gives real symmetric approximate additive
realizations from spectra. Its Algorithm 1 uses repeated orthogonalization
and common Cholesky normalization. The related
[operator-scaling paper](https://arxiv.org/abs/1801.01412) supplies a
randomized polynomial guarantee in its stated input model. This is a
useful concrete instance of the mechanism: the iteration
moves frames while maintaining prescribed spectral information, and
feasibility controls whether balancing is possible. See
[the independent scaling review](scaling-review.md) for exact formulas,
version issues, real-arithmetic applicability, and guarantees.

Our registered R0242 transport converts a supplied real additive pair to
the symmetric unitary product when the operator-norm sum is below pi.
R0243 already has a floating evaluator with sound independent endpoint
acceptance. Composing these would require an additive-error budget and
spectra-only endpoint acceptance; R0243 currently starts from a supplied
additive witness. This is a useful calibration route. It does not cover
arbitrary compact phase data merely by taking logarithms.

### 2. The quantum inequalities belong to parabolic stability

[Teleman–Woodward](https://arxiv.org/abs/math/0012241), Proposition 4.2(b)
and equation (13), connect genus-zero feasibility to semistability of
general weighted flags and inequalities indexed by degrees of reductions.
Sections 3.1–3.2 use Yang–Mills flow in the correspondence to flat
connections. The integer quantum degree records a nonconstant geometric
reduction; it is not an arbitrary correction to an additive calculation.
Remark 4.3 explains the guarded small-weight reduction to ordinary flag
GIT. Retain the strict marking-spread guard and the warning that grading
can change the underlying holomorphic bundle.

The constructive interpretation is to put the target spectrum into the
boundary weights before solving. Balancing then seeks compatibility of
that whole object. This differs operationally from repeatedly fitting
endpoint coefficients on restricted frame charts.

### 3. There is a published real correspondence

[Biswas–Schaffhauser](https://arxiv.org/abs/1806.09782), Theorems 1.1 and
3.6, provide the ordinary-real parabolic correspondence with unitary
orbifold representations. The sphere with conjugation and punctures
0, 1, infinity satisfies the surface hypotheses. The source requires a
polystable real object of parabolic degree zero; it does not turn arbitrary
real flags into one. Its normalized weights differ from signed SU(4)
markings. See the [independent applicability review](real-parabolic-review.md).

For the proposed application, reflection generators give three
antiunitary involutions. Their adjacent products give the peripheral
unitaries. With orientations fixed, use factor classes A, B and the
inverse target class. This is an application inference. The registered
R0052 ordered-Lagrangian-triangle adapter supplies the existing algebraic
route back to an SO(4) middle frame, including multiplicities.

### 4. The analytic unknown has an explicit equation

[Meneses–Takhtajan](https://www.math.stonybrook.edu/~leontak/Meneses-Takhtajan2021.pdf),
Sections 2.3 and 4, describe the adapted positive Hermitian metric by

\[
\bar\partial(h^{-1}\partial h)=0.
\]

The connection is `nabla = d + h^-1 partial(h)`. Singular asymptotics
encode the weights and flags. Section 4.3 gives a regularized variational
functional on its stated regular locus. This supplies an explicit analytic
problem, not a globally certified numerical method or all-wall result.

[Keller–Seyyedali](https://arxiv.org/abs/1411.2716), Theorem 7.4, approximate
a Donaldson-type heat flow using repeated `FS_k o Hilb_k` operations on
bundle metrics over smooth projective manifolds. The approximation is at
fixed finite time as the section-space size grows. It motivates a numerical
approach; it does not establish the parabolic/orbifold extension or a
terminal holonomy-error bound needed here. Passing to rational ramified
covers can also introduce denominator-dependent size, despite fixed rank.

### Why a larger metric problem could help

[Faulk](https://arxiv.org/abs/2202.08885), Proposition 31, states convexity
of the Donaldson functional along exponential metric paths. Theorem 1
relates stability, Hermitian–Einstein existence and a heat-flow properness
condition for indecomposable bundles over compact Kähler orbifolds. This
motivates changing the optimization landscape by changing the variable.
It supplies no effective conditioning or stopping constant. Applying it
to rational parabolic data and restricting to compatible real metrics
requires the transfers described in the [convexity review](convexity-review.md).
That review also flags an abbreviated noncommutative step in the printed
proof: the source statement is recorded, not independently certified here.
No irrational-puncture or semistable-boundary theorem is inferred.

## A concrete constructor specification

The proposed interface is:

```text
input: A and B diagonal unitary spectra, target spectrum T, tolerance
  -> original convention/lift and peripheral classes (A, B, inverse(T))
  -> real parabolic seed with matching weights and degree
  -> compatible balanced metric / flat connection
  -> reflection holonomies / ordered Lagrangian triangle
  -> O in SO(4)
  -> independent check of spec(D O B O^T D) against T, D^2=A
```

No input frame is supplied. Only one representative is needed. The seed's
moduli coordinates can be freely chosen wherever the stability theorem
allows them. We do not need to invert a volume-preserving honeycomb map or
parameterize all realizations.

Three interfaces require new work:

1. **Seed compilation.** Choose and recognize sufficiently general real
   flags, translate signed phases to the correct filtered bundle, and
   preserve its real structure. With normalized weights beta in [0,1),
   the degree must be minus their total sum with multiplicity. Merely
   wrapping negative weights while retaining the trivial bundle is wrong.
   General-real-seed existence is supported by real Zariski density; a
   deterministic seed or one fixed seed for all weights is not established.
2. **Certified analytic evaluation.** Build a symmetry-preserving metric
   iteration, control puncture truncation and discretization, and derive
   an error certificate that controls extracted holonomy. A small sampled
   curvature residual alone is insufficient. The strongest practical
   question is whether this can be done with small rank-four computations
   on the punctured sphere, avoiding a large ramified cover.
3. **Limits and boundary strata.** Semistable inputs can require a graded
   polystable representative. Repeated weights use partial flags. Alcove
   walls require the correct extension convention. These must be part of
   the construction; a generic-interior prototype cannot establish them.

These are substantial obligations, but they concern a specified geometric
process with a stability theorem behind it. They are more informative than
an unspecified search for useful coordinates.

## First experiment worth doing

The next mathematical attempt should first freeze and review the seed and
metric equations. Its first numerical test should use a generic feasible
rank-four triple selected from spectra alone. Identify a positive-degree
QLR inequality and demonstrate that it contributes a bound beyond the
degree-zero subsystem. Also check failure of the R0242 operator-norm guard
explicitly for the chosen logarithm convention. This is not a claim that
all symmetry-related logarithm choices fail. This tests a compact feature
outside the stated additive calibration guard.

Require one original-factor frame, an independent spectral check, and
documented improvement under mesh/precision refinement. Record seed work,
section-space or mesh size, iteration count, conditioning, and total cost.
Use one interior input initially to establish the operation, then require
a separate coverage and boundary study before treating it as a solution.
An isolated success is numerical evidence only. A failure should identify
whether the seed, metric solve, holonomy extraction, or adapter failed.

This is a proposed next acceptance test, not an experiment run in this
source study. Implementing an arbitrary discretization before the input
and boundary conditions are correct would not test the intended machinery.

## Alternative finite-dimensional navigation

The geometric review also found useful collision machinery in AMM's
centralizer cross-sections and O'Shea–Sjamaar's compatible symplectic
slices. These identify the local degrees of freedom that can move a
stalled spectrum. The missing implementation is a certified slice
displacement transported to the supplied original frame, including its
real fixed component. This is a credible second route; qualitative local
openness alone is not a bounded descent algorithm.

## Handoff and limits

The study supports prioritizing real parabolic balancing. It has not
established that this route is faster, globally effective, or a new result
in mathematics. Earlier radical-only assessments such as R0233 do not
exclude it under the current iterative contract. Sparse-chart failures
also do not refute it.

No mathematical claim status is changed and no new universal claim is
registered. Source reading records are updated in LITERATURE.toml. No
production code or corpus was used or changed. This directory is local
research under the repository's intentionally ignored `dev/` tree.

Closure validation and the independent synthesis review are recorded in
[handoff.md](handoff.md).
