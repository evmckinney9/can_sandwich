# Working on the solver

## Reported failures come first

Before other solver work, check
[issue #1](https://github.com/evmckinney9/can_sandwich/issues/1) for unresolved
reports. GULPS links every decline message to it.

```sh
gh api repos/evmckinney9/can_sandwich/issues/1/comments \
  --jq '.[] | {id, user: .user.login, fixed: .reactions["+1"], body}'
```

A report comment is resolved when it has a 👍 reaction; GitHub offers no
checkmark reaction. Handle every unresolved report before anything else:

1. Reproduce the decline with `can_sandwich::solve` on the reported `c`, `g`,
   `t`, using the digits as printed. If it does not reproduce, check the
   reported `can_sandwich` version against the current commit.
2. Decide whether a witness can exist within the acceptance ceiling. If the
   input is provably farther from feasibility than `1e-13`, the defect is in
   the caller's coordinates: report that on the issue instead of changing the
   solver, and do not loosen any tolerance.
3. Append the failing inputs to `tests/cases.bin` as new rows. Do not edit or
   remove existing rows. Note the new row indices and the corpus SHA-256.
4. Fix the solver, not the corpus. The appended rows must pass `make test`,
   and the full comparison must show no coverage loss or accuracy regression.
5. After the fix is committed and pushed, add a 👍 reaction to the report:

   ```sh
   gh api -X POST repos/evmckinney9/can_sandwich/issues/comments/<id>/reactions -f content=+1
   ```

Confirm with the user before pushing or commenting.

## General

This repository is an experimental solver with an independent corpus grader.
For cleanup and optimization, question entire algorithms, stages, and modules.
Broad redesign and deletion are welcome; cosmetic edits alone do not satisfy
that task. Read the relevant flow, form a hypothesis, measure it, and iterate.

Use `tests/cases.bin` to compare coverage, spectral accuracy, endpoint
reconstruction, and release-mode runtime against the current implementation.
Keep the checker independent. Do not loosen its tolerances, remove difficult
cases, or recognize corpus rows in the solver. Passing the corpus is evidence
of progress, not a proof for every feasible input. Before calling any row
infeasible, run `tests/margins.py`: it evaluates the complete quantum-Horn
inequalities exactly on the binary64 inputs.

Accuracy takes priority over speed and source size. The solver addresses an
exact mathematical problem in floating-point arithmetic; acceptance tolerance
is a failure ceiling, not a precision target. Do not accept higher spectral
error merely because every corpus row still passes. Compare error distributions
and individual regressions as well as coverage and runtime before removing
accuracy mechanisms.

Keep changes that reduce complexity or improve measured performance without
losing accuracy or corpus coverage. Record substantial tradeoffs and failed
experiments briefly so the next iteration can build on them. Use the existing
corpus tests and comparison runner instead of adding a second testing framework.

Before finishing solver changes, run `make test` and `make lint`. Keep the
public `solve` and `solve_with_factors` contracts unless the task changes them.
