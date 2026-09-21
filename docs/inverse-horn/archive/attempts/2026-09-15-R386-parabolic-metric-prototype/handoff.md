# R0386 closure

2026-09-15. Terminal result: **observed / practical NO-GO** for the frozen
direct parabolic metric pilot. No further candidate runs or successor
hypotheses followed the failed three-level acceptance test.

The executable is prototype.py; README.md records outputs, numerical scope,
costs, reproduction commands and the reason for the decision. All three
levels fail the original-factor 1e-4 root gate. A small discrete residual
does not imply the correct peripheral monodromy or even accurate loop
closure for the interpolated connection.

Independent verification used separate code for QLR/fixture checks and for
matrix/metric/holonomy checks. The latter exhausts all24 root assignments,
integrates a third large loop and doubles its RK4 transport resolution.
No candidate code is imported by the independent verifiers. Reviewers
disclose context reuse and their roles; no new theorem is promoted.

R0386-M1 is added to CLAIMS.yaml as observed. The five diagnostic input/metric
files are indexed by hash and byte count in evidence/corpora.toml. Existing
primary-source entries already cover the Chern metric/real correspondence;
this attempt uses no new paper or stronger source theorem. STATE.md is
unchanged because no accepted mathematical fact or coverage result changed.

The broader parabolic route remains mathematically open. The exact
reopening obligation is a concrete singular-boundary/monodromy prescription
and compatible discretization with a residual-to-holonomy error argument.
The current results do not isolate which of truncation, spatial error and
uncertified seed applicability dominates. No asymptotic rate or impossibility
is inferred.

All computational runs stayed below the1GiB/30CPU-second/45wall-second caps,
with one BLAS thread. The50-minute author budget was not exhausted. No
production source, public pipeline or locked corpus was changed or evaluated.
Research files remain local under the intentionally ignored dev/ tree.

Closure: `make research-check` passes, recorded in validation.txt. Existing
user modifications outside research remain unchanged.
