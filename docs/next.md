# Work queue

Ranked; take the top item. Each item says what decides it. Delete an item
when it is done or rejected; its result goes in the doc that owns it.

1. **Lower the solver ceiling `SPECTRAL_TOLERANCE`.** Needs the owner's
   decision. GULPS derives `MEMBERSHIP_TOL` from it
   ([research](research.md#where-the-precision-floor-is)), so lowering it
   also shrinks GULPS's reachability slack. Targets on region boundaries
   need that slack. Test in the GULPS tree first, with the submodule
   ceiling at the candidate value: `make test`, and the benchmark compared
   against the current solver, with identical costs and depths. Decided by:
   that GULPS run, then the whole corpus at the lower ceiling.
2. **Remove the band-row gauge cost.** About 40 train rows sit near
   `2.2e-14` after polish. The gauge search on them costs about 9 ms per
   train pass, which is the train-total gap to the build before it. Candidates:
   an analytic Jacobian of the root phases along the gauge, or a direct
   construction. Decided by: train total and the band rows' errors.
3. **Cheaper Horn-wall split in polish.** On 364 train rows the polish
   takes more than 150 µs, mostly in the split, which usually leaves the
   error unchanged. Refining before the split costs 14% to 31% in total. A
   cheap exact wall test would let non-wall rows refine first. Decided by:
   runtime, and no row worse.
4. **Merge the tail symmetry searches.** The transpose dual, sheet
   reflections, and gauge search are overlapping searches over exact
   symmetries ([research](research.md#symmetries-that-move-a-stalled-witness)).
   The source grew from 48,630 to 58,761 tokens. Decided by: tokens, with
   errors and runtime unchanged.
5. **Select three-Givens charts by Carathéodory instead of the measured
   schedule.** The vertex rule found real roots on 198 of the 240 regular rows
   the 16 scheduled charts miss
   ([research](research.md#charts-at-a-vertex-are-chosen-by-carathéodory)).
   Decided by: coverage and errors with the rule replacing `INTERIOR_PLANES`,
   and runtime.
6. **Consolidate `research.md` findings.** The sections from "Critical points
   and strata" onward are a chronological notebook; "What the cascade is, and
   what replaces it" supersedes parts of the earlier ones. Rewrite them as one
   current account and cut what no longer informs an item here.
