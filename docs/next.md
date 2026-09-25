# Work queue

Ranked; take the top item. Each item says what decides it. Delete an item
when it is done or rejected; its result goes in the doc that owns it.

1. **Split on tight inequalities.** A prototype that fixes the coordinate of
   the tightest rank-1 Horn signature and solves the complementary 3×3
   problem reached errors below `1e-15` on rows where the cascade stalls near
   `3e-14` ([research](research.md#tight-inequalities-certify-the-split)).
   Decided by: the 179 residual rows' spectral errors, full-corpus coverage,
   and runtime against `HEAD`.
2. **Lower the spectral ceiling to `1e-14`.** Blocked on item 1; the 179
   residual rows must first reach `1e-14`
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
