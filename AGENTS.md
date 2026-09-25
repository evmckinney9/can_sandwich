# Working on the solver

Agents work here unattended. When told "go" with no other task, run the
session loop below. It picks the work, keeps the docs true, and commits and
pushes without asking.

## Session loop

1. Orient: `git status`, `git log --oneline -10`, `docs/next.md`, and the CI
   result of `HEAD` (`gh run list -L 1`). A red CI run is the first task.
2. A dirty tree is an earlier session's unfinished work. Validate and commit
   it, or, if it cannot pass, `git stash push -m "<why>"` and add a line to
   `docs/next.md`.
3. Handle unresolved reports on issue #1 (below).
4. Take the top item of `docs/next.md`. Read the relevant flow, form a
   hypothesis, measure it on the corpus, and keep or reject the change.
5. Close the item: record the result where the documentation table says,
   remove or rewrite the item in `docs/next.md`, run the finishing checks,
   and commit.
6. Repeat from step 4 while budget remains. Push once at the end.

Stop and leave a note in `docs/next.md` instead of guessing when a change
needs the public API, the GULPS repository, or a tolerance to move.

## Reported failures come first

[Issue #1](https://github.com/evmckinney9/can_sandwich/issues/1) collects
decline reports; GULPS links every decline message to it.

```sh
gh api repos/evmckinney9/can_sandwich/issues/1/comments \
  --jq '.[] | {id, user: .user.login, fixed: .reactions["+1"], body}'
```

A report comment is resolved when it has a 👍 reaction; GitHub offers no
checkmark reaction. For each unresolved report:

1. Reproduce the decline with `can_sandwich::solve` on the reported `c`, `g`,
   `t`, using the digits as printed. If it does not reproduce, check the
   reported `can_sandwich` version against the current commit.
2. Decide whether a witness can exist within the acceptance ceiling. If the
   input is provably farther from feasibility than `1e-13`, the defect is in
   the caller's coordinates: say so on the issue instead of changing the
   solver, and do not loosen any tolerance.
3. Append the failing inputs to `tests/cases.bin` as new rows. Do not edit or
   remove existing rows. Note the new row indices and the corpus SHA-256.
4. Fix the solver, not the corpus. The appended rows must pass `make test`,
   and the full comparison must show no coverage loss or accuracy regression.
5. After the fix is pushed, add a 👍 reaction to the report:

   ```sh
   gh api -X POST repos/evmckinney9/can_sandwich/issues/comments/<id>/reactions -f content=+1
   ```

## Solver rules

This repository is an experimental solver with an independent corpus grader.
Question entire algorithms, stages, and modules; broad redesign and deletion
are welcome, and cosmetic edits alone are not progress.

Use `tests/cases.bin` to compare coverage, spectral accuracy, endpoint
reconstruction, and release-mode runtime against the current implementation.
Keep the checker independent. Do not loosen its tolerances, remove difficult
cases, or recognize corpus rows in the solver. Passing the corpus is evidence
of progress, not a proof for every feasible input. Before calling any row
infeasible, run `tests/margins.py`: it evaluates the complete quantum-Horn
inequalities exactly on the binary64 inputs.

Accuracy takes priority over speed and source size. The acceptance tolerance
is a failure ceiling, not a precision target. Do not accept higher spectral
error because every row still passes. Compare error distributions and
individual regressions as well as coverage and runtime before removing an
accuracy mechanism. Keep the public `solve` and `solve_with_factors`
contracts. Use the existing corpus tests and comparison runner instead of
adding a second testing framework.

## Documentation

Docs are part of the change. A commit that changes behavior, a constant, a
public type, a module, the corpus, or the runner updates the doc that owns
that fact in the same commit. Each fact has one owner; other files link to it.

| Doc | Owns |
|---|---|
| `README.md` | Public API, decline semantics, build commands, corpus format |
| `docs/architecture.md` | The code as it is now: layout, search order, tolerances, diagnostics; no history |
| `docs/researcher.md` | Comparison runner contract, report format, checker definitions |
| `docs/research.md` | Mathematical formulation and current findings |
| `docs/optimization.md` | Dated experiment log with commit and corpus SHA-256, including rejected approaches |
| `docs/next.md` | Ranked work queue, at most ten items |

When a finding is superseded, edit or delete it; do not append a section that
contradicts an earlier one. When an experiment-log entry is about code that no
longer exists, shrink it to one line under Retired interfaces.

Stale docs here have come from updating one figure and leaving the sentence
beside it. When editing a number or name, reread its paragraph, and before
committing grep every doc for each identifier, constant, and count the diff
touches.

Write plain declarative sentences. Give numbers with the measurement behind
them. Cut any sentence that does not change what a reader would do. No
adjectives of praise, no dashes as punctuation, no summaries of the section
just read.

## Commits and pushes

Commit each finished unit once `make test`, `make lint`, and the documentation
check pass. Follow the prefixes in `git log` (`feat`, `fix`, `perf`,
`refactor`, `docs`, `ci`). The body gives the measured effect, the corpus
SHA-256, and the validation run. No AI attribution trailers. A rejected
experiment is committed only as its log entry, never as code.

Every push to `main` runs the full corpus in CI, which takes minutes. Run the
checks locally, and push once per session, not per commit. Docs-only pushes
skip CI. Push only `main` to `origin`; never force-push, and never commit or
push in the parent GULPS repository.

## Changing these rules

Edit this file when a session hits a failure these rules would have
prevented, or when a rule wastes work. Name the incident in the commit
message. Replace or delete a rule rather than adding one beside it, and keep
the file under 130 lines. Rules about how to work go here, pending work goes
in `docs/next.md`, and findings go in the docs.
