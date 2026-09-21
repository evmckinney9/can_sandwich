# Independent review of the additive teaching README

2026-09-15. Existing uninvolved reviewer context reused: this reviewer authored
the independent scaling-source note, but did not author the README. No new
claim promotion or numerical experiment.

## Verdict

The teaching and output-contract distinctions are sound in the reviewed scope.
One clarification is required before treating the displayed flag condition as
a complete feasibility statement:

**Section 6 must include the trace condition `sum(i,j) p_ij = n`, positive
decreasing p(i), and complex ambient subspaces `R,F_ij ⊂ C^n`.** Franks's
Theorem 70 explicitly requires `Tr P = n` in addition to the inequalities
for every R and a generic flag tuple. Taking R=C^n in the inequality only
gives a less-than-or-equal trace bound. The preceding sum-I normalization
suggests the intended premise but should not replace stating it.

## Checks completed

* The rank-one determinant identity and minus sign in the residue formula
  are correct. Positive-update interlacing gives nonnegative squared
  coordinates; the trace supplies the nonzero update eigenvalue. The
  repeated-spectrum paragraph correctly changes to eigenspace projection
  norms rather than evaluating a zero denominator.
* The flag inequality's direction and decreasing-weight differences match
  Franks Theorem 70, equation (22), at the primary full text. Its generic
  quantifier is distinct from claiming arbitrary supplied flags work.
* Common Cholesky whitening and separate QR normalization describe the
  published scaling mechanism. Capacity, rather than monotone visible
  residual, supports the cited convergence reasoning.
* The almost-sure ideal-random statement is distinguished from bounded-error
  discrete randomization; a failed run is not called an infeasibility proof.
* The 2023/2025 source reconciliation does not invent an explanation for
  the citation omission. The exact-algebraic fallback paragraph correctly
  restricts its claim to algebraic input data and avoids equating it with a
  practical or explanatory inverse.
* Exact witness, approximate witness, parameterization, hive lift, and
  sampling are distinguished. The README does not declare lack of one
  stronger contract an obstruction to obtaining a single witness.

## Limits

Fulton, eigensteps, rank-two, and hive statements were read in the synthesis
and companion source reports, not independently re-proved during this
three-minute review. The source-specific reviewers retain responsibility for
those detailed attributions. No claim of exhaustive literature coverage or
universal numerical certification is justified or made.

The trace/ambient-field clarification was sent to the root author for the
single-writer edit. This review does not silently change the README.

## Completion

The root author added the positive decreasing weights, total-trace equality,
and complex ambient subspaces. This reviewer re-read that paragraph and
confirmed the required clarification. The reviewed explanatory scope passes;
the root reports `research-check` passed. This is not a new claim promotion.
