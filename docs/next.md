# Work queue

Ranked; take the top item. Each item says what decides it. Delete an item
when it is done or rejected; its result goes in the doc that owns it.

1. **The 42 rows still above `2e-14`.** The rank-1 split restart
   ([log](optimization.md#split-restart-on-rank-1-walls-2026-09-25)) left 42
   rows, 13 above `3e-14`. Candidates: a split on a rank-2 signature (two
   2×2 blocks), or the polytope homotopy restart
   ([research](research.md#a-homotopy-over-the-feasible-polytope)). Measure
   errors at 80 digits; the corpus Schur checker misranks errors near
   `1e-14`. The 80-digit check so far was a scratch script (mpmath
   eigenvalues of $D(c)^2OD(g)^2O^T$, best of 24 bijections and both signs,
   on dumped witnesses); first give the runner a way to write witnesses and
   add the script under `tests/`. Decided by: rows above `2e-14`, no row
   worse, and runtime.
2. **Lower the spectral ceiling to `1e-14`.** Blocked on item 1
   ([research](research.md#where-the-precision-floor-is)). Decided by: the
   whole corpus passing at the lower ceiling. GULPS derives its reachability
   slack from this constant, so leave a note for the GULPS side.
3. **Select three-Givens charts by Carathéodory instead of the measured
   schedule.** The vertex rule found real roots on 198 of the 240 regular rows
   the 16 scheduled charts miss
   ([research](research.md#charts-at-a-vertex-are-chosen-by-carathéodory)).
   Decided by: coverage and errors with the rule replacing `INTERIOR_PLANES`,
   and runtime.
4. **Consolidate `research.md` findings.** The sections from "Critical points
   and strata" onward are a chronological notebook; "What the cascade is, and
   what replaces it" supersedes parts of the earlier ones. Rewrite them as one
   current account and cut what no longer informs an item here.
