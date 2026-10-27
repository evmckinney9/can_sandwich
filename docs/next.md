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
3. **Cheaper refinement in polish on slow rows.** Since the route hull gate
   (`149f4c3`) the split is 5.9% of samples on the 1,446 corpus rows over
   150 µs; `Problem::polish` is 47% and the numerical rung 23%, nearly all in
   `refine_fixed`. On every fourth row a polish refine that is discarded
   costs 564 µs (80 calls), and `escape` 330 µs (232 calls). Capping damping
   retries was rejected (`aacdd1f`), and so was factoring the Jacobian once
   per step (`e9bcf65`). Decided by: runtime, and no row worse.
4. **Merge the tail symmetry searches.** The transpose dual, sheet
   reflections, and gauge search are overlapping searches over exact
   symmetries ([research](research.md#symmetries-that-move-a-stalled-witness)).
   The source grew from 48,630 to 58,761 tokens. Decided by: tokens, with
   errors and runtime unchanged.
   Which search supplies the final frame (`ee83b91`, counted under `PROF`):
   on `cases.bin` 40 rows reach the tail, won by the gauge search 37, the
   transposed primary 2 and a sheet reflection 1. On held-out seeds 9101 and
   9202 (98k rows each) 89 and 62 rows reach it; the dual witness's polish
   wins 1 and 3 and the stratum candidates 3 and 0, so no search can be
   dropped on this evidence.
5. **Consolidate `research.md` findings.** The sections from "Critical points
   and strata" onward are a chronological notebook; "What the cascade is, and
   what replaces it" supersedes parts of the earlier ones. Rewrite them as one
   current account and cut what no longer informs an item here.
