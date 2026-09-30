# Work queue

Ranked; take the top item. Each item says what decides it. Delete an item
when it is done or rejected; its result goes in the doc that owns it.

1. **Confirm the tail at 80 digits.** By the binary64 measure, the corpus
   maximum is `1.78e-14`
   ([log](optimization.md#held-out-hillclimb-on-speed-and-accuracy-2026-09-30)).
   The 80-digit check is still a scratch script (mpmath eigenvalues of
   $D(c)^2OD(g)^2O^T$, best of 24 bijections and both signs, on dumped
   witnesses). First give the runner a way to write witnesses, and add the
   script under `tests/`. Decided by: the 80-digit maximum and the count
   above `2e-14`.
2. **Lower the spectral ceiling.** Needs the owner's decision. The corpus
   passes at `2e-14` by the binary64 measure, and every harness row passes
   at `3e-14`. GULPS derives `MEMBERSHIP_TOL` from `SPECTRAL_TOLERANCE`
   ([research](research.md#where-the-precision-floor-is)), so the change
   needs a matching GULPS commit. Decided by: item 1, then the whole corpus
   at the lower ceiling.
3. **Strengthen the corpus spectral check.** Matching characteristic
   polynomial coefficients at `1e-12` accepted root errors up to `1.5e-6` at
   repeated roots in the external harness. That harness adds a
   Hermitian-projection test at the same tolerance. The check is a checker
   definition ([researcher guide](researcher.md)), so it needs the owner's
   decision. Decided by: no false accept in a mutation test, and the corpus
   still passing.
4. **Remove the band-row gauge cost.** About 40 train rows sit near
   `2.2e-14` after polish. The gauge search on them costs about 9 ms per
   train pass, which is the train-total gap to the build before it. Candidates:
   an analytic Jacobian of the root phases along the gauge, or a direct
   construction. Decided by: train total and the band rows' errors.
5. **Cheaper Horn-wall split in polish.** On 364 train rows the polish
   takes more than 150 µs, mostly in the split, which usually leaves the
   error unchanged. Refining before the split costs 14% to 31% in total. A
   cheap exact wall test would let non-wall rows refine first. Decided by:
   runtime, and no row worse.
6. **Merge the tail symmetry searches.** The transpose dual, sheet
   reflections, and gauge search are overlapping searches over exact
   symmetries ([research](research.md#symmetries-that-move-a-stalled-witness)).
   The source grew from 48,630 to 58,761 tokens. Decided by: tokens, with
   errors and runtime unchanged.
7. **Select three-Givens charts by Carathéodory instead of the measured
   schedule.** The vertex rule found real roots on 198 of the 240 regular rows
   the 16 scheduled charts miss
   ([research](research.md#charts-at-a-vertex-are-chosen-by-carathéodory)).
   Decided by: coverage and errors with the rule replacing `INTERIOR_PLANES`,
   and runtime.
8. **Consolidate `research.md` findings.** The sections from "Critical points
   and strata" onward are a chronological notebook; "What the cascade is, and
   what replaces it" supersedes parts of the earlier ones. Rewrite them as one
   current account and cut what no longer informs an item here.
