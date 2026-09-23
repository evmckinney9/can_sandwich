# Working on the solver

This repository is an experimental solver with an independent corpus grader.
For cleanup and optimization, question entire algorithms, stages, and modules.
Broad redesign and deletion are welcome; cosmetic edits alone do not satisfy
that task. Read the relevant flow, form a hypothesis, measure it, and iterate.

Use `tests/cases.bin` to compare coverage, spectral accuracy, endpoint
reconstruction, and release-mode runtime against the current implementation.
Keep the checker independent. Do not loosen its tolerances, remove difficult
cases, or recognize corpus rows in the solver. Passing the corpus is evidence
of progress, not a proof for every feasible input.

Keep changes that reduce complexity or improve measured performance without
losing corpus coverage. Record substantial tradeoffs and failed experiments
briefly so the next iteration can build on them. Use the existing corpus tests
and comparison runner instead of adding a second testing framework.

Before finishing solver changes, run `make test` and `make lint`. Keep the
public `solve` and `solve_with_factors` contracts unless the task changes them.