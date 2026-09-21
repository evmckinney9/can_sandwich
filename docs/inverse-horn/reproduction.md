# Reproduction and archive provenance

Run commands from the repository root. The exact checks below need Python 3
and SymPy. The historical numerical prototypes additionally use NumPy and
SciPy; they are evidence of failed or limited attempts, not production
dependencies or endorsed constructors.

## Exact checks

Each command below can be bounded on Linux with
`timeout 45s` and `ulimit -v 1048576; ulimit -t 30`. The documentation import
reran the six checks under these memory and time bounds with one BLAS thread.

```sh
python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R388-hive-image/certificate.py
python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R389-signed-hermitian/certificate.py
python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R390-quartic-slices/certificate.py
python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R391-tridiagonal-sum/certificate.py
python3 crates/can_sandwich/docs/inverse-horn/archive/scouting/2026-09-15-additive-inverse-horn/check-signed-singular-counterexample.py
python3 crates/can_sandwich/docs/inverse-horn/archive/scouting/2026-09-15-additive-inverse-horn/check-T4-tridiagonal-counterexample.py
```

R388 rewrites its adjacent certificate.json with deterministic output; the
other checks print their results. No benchmark or production corpus is run.
The archived prototypes have their own reproduction commands and frozen
inputs, separate from these exact checks.

| Check | What a pass establishes |
|---|---|
| R388 | One exact hive preimage, defeating the proposed exclusion; not general surjectivity. |
| R389 | Exact feasible signed-singular counterexample and original real 4x4 lift. |
| R390 | Polynomial identities, repeated-critical-point factorization, Hermite minors and anchor invertibility supporting the independently reviewed proof. Not an automated proof of global selection. |
| R391 | Exact feasible T4 counterexample and all permutation/support cases. The written proof covers all possible vectors. |
| Independent R389/R391 checks | Separate exact replays and proof checks; no numerical-search assumptions. |

## Original evidence and adaptations

The archive contains both the additive study and the preceding machinery
study, plus attempts R386–R391. It includes original code, input fixtures,
outputs, metric arrays, source inventories, review notes and stopping records.
It excludes Python bytecode caches. Primary papers are linked rather than
copied into the crate.

Code and evidence bytes are unchanged. Markdown command paths that referred
to `dev/research/` now refer to this archive; local Gift thesis links now
point to its primary repository. The source and destination hashes in
[archive-manifest.json](archive-manifest.json) distinguish these adaptations.
Archived narratives retain their original claims, timing and limitations;
the main README states the final assessment. Mentions of `STATE.md`,
`CLAIMS.yaml`, `research-check`, or other research governance refer to the
original local workspace, not prerequisites for running the exact checks here.

The source registry remains authoritative in that workspace. The imported
[claims snapshot](claims-snapshot.json) records statuses, not new acceptance
decisions. The documentation import does not promote the conditional chain
lemma, reopen failed attempts, or claim independent novelty validation.

## Validation performed for this documentation change

- Resolve local Markdown file links throughout the imported documentation.
- Verify each archived file against its recorded destination SHA-256, and
  its original source against the recorded source SHA-256.
- Run the six exact certificate scripts above under their resource caps.
- Check the tracked diff for whitespace errors.

No Rust implementation changed, so no solver benchmark or production test
was required for this documentation import.
