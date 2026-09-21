# R0387 independent verification

Date: 2026-09-15. Reviewer context reuse disclosed: prior unrelated reviews
covered low-rank additive literature, real parabolic geometry, and the R386
PDE conventions. This reviewer did not author or tune R387's constructor,
fixtures, or paired run. The frozen packet was read before verification.

## Verdict

**All returned matrices pass independent numerical checks; the full
12-case acceptance hypothesis fails.** Plain scaling accepts 5/12 inputs,
and the block wrapper accepts 7/12. All other outcomes remain `UNKNOWN`.
This is a finite observed comparison, not a refutation of Franks scaling
or a claim of general coverage, novel construction, or certified arithmetic.

The root README's finite-result summary and qualifications match the
checked outputs. Its failed-row residuals, except the separately replayed
`wall22` row, are reported solver diagnostics rather than independently
reconstructed failed matrices, because those rows save no candidate frame.

## Source transcription and assembly checks

The primary Franks thesis Chapter 4, Algorithm 1, prescribes common
lower-triangular whitening and separate upper-triangular orthogonalization.
The following implementation checks pass:

1. `scale=12*max(1,max|inputs|)` and the three spectral lists
   `alpha/scale+1/3`, `beta/scale+1/3`, and sorted
   `-gamma/scale+1/3` are strictly positive. Trace compatibility makes their
   total trace `n`; the intended common marginal is `I_n`.
2. For NumPy's lower Cholesky factor `L`, `total=L L^T`.
   Solving `L V=U` whitens the total by `L^-1`. The direction is correct.
3. QR factorization `V=Q R` means `V R^-1=Q`, the required right
   upper-triangular change. QR column signs do not change the represented
   spectral matrices.
4. Undoing the normalization gives
   `A=scale*(H1-I/3)`, `B=scale*(H2-I/3)`, and the target-spectrum
   matrix `D=scale*(I/3-H3)`. Thus the relative frame is
   `U1^T U2`, as implemented. The third list's reversed ordering is correct.
5. Flipping a column of the returned frame changes determinant sign and
   commutes with diagonal `beta`, preserving the represented second summand.
6. Child solutions are inserted in the row/column index sets belonging to
   the selected alpha/beta subsets. Those index sets preserve descending
   spectral order. Complementary blocks cover all rows/columns once.
   Final eigenvalue checking correctly determines acceptance even if a
   numerical trace-compatible partition is not exactly feasible.
7. The two-dimensional angle formula is the standard gap identity
   `z²=x²+y²+2xy cos(2 theta)`; the original sum-spectrum check also guards
   trace compatibility and clipping.

The fixed Gaussian seed, initial QR, finite sweep budget, and stopping on
the original sum spectrum are implementation choices. They are not the
source's discrete randomized guarantee or an invocation of its iteration
bound. The direct spectral stopping rule suffices for the returned additive
witness; it does not assert a particular normalized three-marginal residual.

## Independent numerical checks

`verifier.py` imports NumPy, JSON, hashing and path utilities only. It does
not import `constructor.py` or its helpers. It independently forms

```text
A = diag(alpha), B = Q diag(beta) Q^T,
eigvalsh(A+B), eigvalsh(B), Q^T Q, det(Q).
```

It checks finiteness, dimensions, packet completeness, all 24 unique
input/method rows, and preservation of every `UNKNOWN` row.

| Quantity | Result |
|---|---:|
| Plain accepted/attempted | 5/12 |
| Wrapped accepted/attempted | 7/12 |
| Returned frames independently accepted | 12/12 |
| Largest accepted sum-spectrum error | 4.989716140268996e-9 |
| Largest orthogonality infinity norm | 2.606700770010323e-15 |
| Largest determinant error from +1 | 9.992007221626409e-16 |

The verifier also checks each separately stored plant against its input
spectra. This supports numerical fixture feasibility; it is not an exact
Horn certificate for the rounded input arrays. Plant determinant `-1` is
permitted in that feasibility check because it leaves the symmetric
conjugate unchanged; accepted constructor frames all have determinant `+1`.

Static inspection confirms that `constructor.py` reads no files and receives
only spectra and numerical options. `run.py` reads `inputs.json` and never
reads `plants.json`. The generator writes plant evidence; only the verifier
reads it. No plant is supplied to the solver.

## Independent failed-case replay

The verifier separately implements the source sweeps with seed 387 on
`wall22`, without importing constructor code. Its marginal is assembled
with explicit diagonal matrices, independently of the constructor's
columnwise multiplication expression. After exactly 6000 sweeps:

```text
independent final error = 0.007344222724209537
recorded final error    = 0.007344222724209537
```

No earlier replayed iterate reaches the implementation's acceptance gate.
This reproduces the reported bounded failure. The accepted block result
for the same input is independently verified above. No seed changes,
longer runs, method repairs, or tuning occurred.

## Reproduction and handoff

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 45s prlimit --as=1073741824 --cpu=30 python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R387-additive-constructor/verifier.py
```

This command passed under the specified limits. `independent.json` stores
per-frame measurements, all statuses, the failed-case replay, plant checks,
and SHA256 hashes of the frozen packet and verifier.

The exact remaining implementation result is 5 unresolved wrapped inputs:
`generic0`, `wall13`, and all three `nearwall` cases. The run does not prove
those inputs infeasible or establish a convergence-rate obstruction. The
attempt should close with the finite measurement and unmet full-packet
acceptance test; no further tuning is authorized by this review.
