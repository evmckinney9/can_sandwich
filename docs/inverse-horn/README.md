# Inverse Horn investigation: results and limits

Snapshot: 2026-09-15. This documents the investigation requested to find
better mathematical machinery for depth-two atomic realization. The user
asked to understand additive inverse Horn before attempting a multiplicative
transfer, and explicitly rejected presenting another representation or a
special-case numerical success as a solution.

**A practical Horn-specific exact formula was not obtained.** The finite-net
note now gives a universal certified epsilon-constructor, and the canonical
selector gives an exact terminating construction for algebraic inputs. The
arbitrary-real exact finite-output problem remains a representation question;
no production change to `can_sandwich` resulted.

The work did produce two exact exclusions, a checked convex-slice lemma,
and a conditional spectral-chain reconstruction. Publication novelty has
not been established for these results.

## Reading order

1. This status assessment and [research lessons](research-lessons.md).
2. [Mathematical formulations](formulations.md): the problem, hive chains,
   the exceptional tensor correspondence, quartic completion, and exact
   algebraic computability.
3. [Primary sources and their actual contracts](sources.md).
4. The proof or experiment linked in the table below.
5. [Reproduction and archive provenance](reproduction.md).

## What is already known

The existence of additive witnesses is characterized by Horn inequalities.
Horn's 1962 proof establishes sufficiency in dimension four. This does not
itself supply the sought matrix reconstruction procedure.

General approximate construction is also known: Franks supplies a randomized
operator-scaling algorithm. Gift's thesis and the rank-two paper acknowledge
it. An unqualified statement that there is no general constructive algorithm
is therefore misleading. Exact algebraic sampling is available in principle
through real algebraic geometry for algebraic inputs. Neither is a result
of this investigation.

The sought contribution is a proved, explanatory, effective construction
from arbitrary feasible endpoint spectra, with a precise output contract.
"A Horn-specific construction is open" needs this qualification: approximate
Horn-specific construction already exists. A direct rank-three construction
is one formulation of the remaining objective, not the only permissible
method.

## Results, with their actual scope

| Item | Status | Evidence and remaining obligation |
|---|---|---|
| Hive/real-fiber dimension match | Correct, not exceptional to n=4 | The generic count agrees for every n. It supplies no map between individual hives and matrices. [Assessment](archive/scouting/2026-09-15-additive-inverse-horn/n4-assessment.md) |
| Natural LR spectral chain | Useful interpretation; not an automatic orthogonal lift | Interlacing does not impose mutual orthogonality of update directions. [Derivation](archive/scouting/2026-09-15-additive-inverse-horn/rank3-coupling.md) |
| Rank-three Cauchy coupling | Independently checked algebra | Explicit nonadjacent overlap; a strict supplied-chain lifting test and reconstruction. Choosing a chain remains open. The scoped claim R0389-L1 remains unreviewed at the full project-certificate level. [Review](archive/scouting/2026-09-15-additive-inverse-horn/rank3-chain-algebra-review.md) |
| Rank-two intermediate selector | Accepted restricted result | Vertex-support sign change and certified bisection select the intermediate spectrum for the simple positive rank-two stratum. This does not cover generic rank-three beta. [Study](rank2-selector/README.md) |
| SO(4) to signed 3x3 singular-value sum | Valid equivalence | No proof that the resulting inverse is easier. [Formulation](formulations.md#signed-singular-value-formulation) |
| Direct conversion to 3x3 Hermitian Horn by signs | Refuted, R0389-H1 | Exact trace obstruction, including a real 4x4 lift. Does not exclude auxiliary-variable reductions. [Proof](archive/attempts/2026-09-15-R389-signed-hermitian/attempt.md) |
| Three-parameter quartic completion | Valid equivalent selection problem | Three unknown coefficients still require global real-rootedness. [Formulation](formulations.md#quartic-completion) |
| Fixed-s completion slices | Verified scoped lemma, R0390-L1 | Compact convex planar slices and at most three scalar constraints certifying an empty slice. Neither a global s selector nor a terminating cutting-plane method is proved. [Proof](archive/attempts/2026-09-15-R390-quartic-slices/attempt.md) |
| Horn-facet block construction | Established source mechanism | A saturated Horn inequality exposes smaller blocks. No interior-extension procedure follows. Ordered-chamber walls must not all be treated as saturated Horn inequalities. |
| Diagonal sum plus tridiagonal summand, T4 | Refuted, R0391-H1 | Rank-one support obstruction excludes all 24 permutations. Failure persists near the example, including feasible all-simple triples. [Proof](archive/attempts/2026-09-15-R391-tridiagonal-sum/attempt.md) |
| Three-Givens/arrowhead/tree coverage claims from the other thread | Unestablished here | Numerical search reports were supplied in conversation, without imported universal coverage proofs or exact exclusion certificates. A numerical miss is not a proof of impossibility. |

## Failed or parked experiments

These are included so later work cannot silently reuse their optimistic
interpretations after forgetting the outcome.

| Attempt | Actual outcome |
|---|---|
| [R386: direct parabolic metric pilot](archive/attempts/2026-09-15-R386-parabolic-metric-prototype/README.md) | Small discrete PDE residuals did not produce the required monodromy. Original-target errors were about 1.03, 0.81, 0.65 against a 1e-4 gate. Practical NO-GO for this frozen prototype, not a theorem against the broader geometry. |
| [R387: additive scaling and block assembly](archive/attempts/2026-09-15-R387-additive-constructor/README.md) | Plain scaling returned 5/12; block wrapper 7/12. Five wrapped cases remained UNKNOWN. Known machinery, fixed finite budgets, no new constructor theorem; parked after the user rejected it as progress toward the objective. |
| [R388: compression-trace hive image](archive/attempts/2026-09-15-R388-hive-image/README.md) | The attempted non-surjectivity counterexample failed. Independent review supplied an exact preimage. General surjectivity remained UNKNOWN; no exclusion was established. |

The earlier [machinery review](archive/scouting/2026-09-15-deep-machinery/README.md)
and its method discussion are retained as historical records. Their proposed
next steps are superseded by the experiment outcomes above; they are not
current endorsements.

## What remains open in this investigation

The subsequent [eigensteps study](eigensteps-study/README.md) works through
Top Kill's successful extension proof and the point where the labeled
orthogonality constraints of Horn escape that proof. It derives an explicit
polar correction with an original-spectrum error bound. This supplies a
quantitative contract for a possible selector, not the selector itself.
The derivation has author checks only and does not change the status above.

The [rank-two selector study](rank2-selector/README.md) (R392; proof,
certificate and independent review in the working attempt directory) makes
the Gift-Woerdeman intermediate-spectrum selection constructive on the
simple-spectrum, no-shared-value rank-two stratum: a vertex sign change on
the interlacing box slice plus bisection, with coverage from Pieri-type Horn
inequalities. For n=4 this is the repeated-beta stratum only; the generic
rank-three selection is untouched.

The [finite-net constructor](finite_net_constructor.md) gives a separate,
fully covered epsilon-level algorithm. It enumerates a finite Givens net in
SO(4), with a Weyl bound proving that a feasible input must produce a passing
candidate. This is an effective certified approximate construction; its cost
is exponential in the requested precision. It provides the certified
approximate construction for arbitrary real data, with an adaptive
branch-and-bound variant; a canonical exact output
still requires the algebraic or real-oracle selector described below.

This closes the formal certified-iterative contract, but the finite-net
baseline is computationally impractical: its worst-case work is proportional
to epsilon^(-6), with roughly 10^32 evaluations at tolerance 10^(-3) after
normalizing max|beta_i| to one. It is a coverage certificate, not the
practical constructor sought by the research objective. The algebraic
selector is exact but has similarly uncontrolled quantifier-elimination cost.

For algebraic endpoint data, [canonical_selector.md](canonical_selector.md)
adds an explicit exact choice: the lexicographically least feasible (Q),
selected by a stated sequence of semialgebraic existence queries. Compactness
proves every minimum exists, and real-closed-field quantifier elimination
terminates on these concrete polynomial constraints. This closes the exact
selection obligation in the algebraic input model while keeping the
arbitrary-real case in the certified finite-net model.

The remaining finite-description issue is separate from this convergent
oracle construction. In the quartic formulation this means selecting s for
which K_s is nonempty, then a point in K_s. The interval allowed by the
derivative conditions is not proved equal to the projection of the full
completion set. That projection has not been proved connected, so bisection
has no established justification.

In the rank-three formulation the missing step is selecting intermediate
spectra and compatible signs, rather than reconstructing vectors after the
necessary compatible data are supplied. In the hive formulation, the map
between individual hives and realizations still needs a precise definition
and a lifting theorem. Nothing here proves an additive-to-multiplicative
transfer to the production depth-two problem.

The convex-slice lemma is the clearest surviving structure. It is a reason
to formulate a bounded selection question, not evidence that this approach
will eventually succeed. No new route is authorized or implied by this
documentation.

The standalone additive implementation is
[additive-generic-selector.py](additive-generic-selector.py); its empirical
results and comparison measurements are in
[generic-selector-benchmark.md](generic-selector-benchmark.md). It is a
benchmark baseline, not a universal coverage proof.
Machine-readable measurements are collected in
[benchmark-results.json](benchmark-results.json).

## Provenance and authority

This is a portable documentation snapshot, not a second live claim registry.
The working research authority remains `dev/research/CLAIMS.yaml` when that
local workspace is available. [claims-snapshot.json](claims-snapshot.json)
preserves the seven relevant entries as they stood at import. Historical
notes stating "no claim promoted" describe their own scope and date.

[archive-manifest.json](archive-manifest.json) identifies every imported
source and its source/archive SHA-256. Executable code and evidence data
are copied unchanged. Markdown reproduction paths point into this archive;
local thesis-PDF links point to the primary repository instead. The archive
includes independent reviews, failed evidence, and original scope limits.

This documentation changes no solver, API, corpus, or production behavior.
