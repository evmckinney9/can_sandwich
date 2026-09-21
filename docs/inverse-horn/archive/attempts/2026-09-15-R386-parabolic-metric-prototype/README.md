# R0386 result: NO-GO for the direct metric prototype

The finite-domain metric solver converges, but its holonomy does not realize
the prescribed spectra. None of the three predeclared truncation/grid levels
passes the 1e-4 original-factor root-error gate. This supports parking this
direct implementation. It does not refute real parabolic correspondence or
exclude other metric formulations.

## What was implemented

[prototype.py](prototype.py) is a 190-line Python prototype. It solves

```text
Delta h - (h_x+i h_y) h^-1 (h_x-i h_y) = 0
```

for a complex Hermitian positive metric on a rectangular plane domain with
square holes around 0 and 1. Spectral powers and three fixed real flag bases
set inner/outer Dirichlet data. A bounded Newton/GMRES solve uses a sparse
Laplacian preconditioner and a positivity-preserving line search.

The program interpolates the Chern connection, integrates two common-base
peripheral loops, extracts real eigenframes, assigns the original factor
spectra, and checks the actual original-factor sandwich. It does not fit
endpoint coefficients, read a planted frame, or call another realization
solver. It is an unvalidated discretization; acceptance rests on the final
root check, not on any claim of continuum convergence.

## Input and control

One simple-spectrum rank-four input was frozen in [fixture.json](fixture.json).
The independently checked feasible product has minimum QLR slack 0.12498;
it is not a near-wall stress case. The chosen logarithms have norm sum
1.62 pi. An exact rational separation witness confirms that a degree-one
quantum inequality strengthens the degree-zero subsystem for this factor
pair. See [fixture-check.md](fixture-check.md).

The feasibility generator's frame is stored separately and is read only by
verification. The rounded target trace differs from zero by 2.5e-16; no exact
normalized-input theorem is claimed.

The manufactured noncommuting flat metric `h=g* g`, with `g=I+zN` and fixed
real strictly upper-triangular N, converges in three Newton steps. The
reported maximum metric-entry error is 2.89e-12. This checks the smooth
solver core; it does not validate singular boundary limits.

## Frozen results

| Level | Inner cutoff | Outer parameter | Metric nodes | Real unknowns | Author time | Original target root error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1/8 | 2 | 897 | 12,144 | 0.65 s | 1.0264312 |
| 1 | 1/16 | 4 | 1,705 | 24,304 | 1.09 s | 0.8073789 |
| 2 | 1/32 | 8 | 2,769 | 40,560 | 1.85 s | 0.6500450 |

Times include mesh setup, solve and extraction, but exclude interpreter/import
startup and output serialization. They are observations, not benchmarks
against production. The three runs completed in four or five Newton steps.
Their row-scaled discrete PDE residuals are below 5.4e-12. The finest unscaled
PDE RMS residual is 8.55e-8; scaled residuals must not be called certified
continuum curvature errors.

All extracted O matrices satisfy SO(4) checks to about 1.4e-15. Their target
root errors remain thousands of times above the deliberately loose pilot
threshold. Independent exhaustive 24-permutation matching reproduces those
errors exactly to the shown precision.

## What the independent check found

[independent-verifier.py](independent-verifier.py) imports no prototype code.
It reconstructs finite differences, verifies saved metric positivity and
conjugation symmetry, independently integrates holonomy, and checks all
output frames against the original factor spectra.

The failure already appears before frame extraction:

- At level 2 the two actual peripheral root errors are approximately 0.684
  and 0.727. The raw holonomy product has target error 0.629.
- An independently integrated large loop has target error 0.599. Its inverse
  also fails the required infinity spectrum.
- Large-loop versus composed-small-loop matrix discrepancies are 0.140,
  0.149 and 0.172 across the levels. Thus the interpolated connection also
  fails the expected loop closure appreciably.
- Doubling independent ODE resolution changes the peripheral transports by
  only about 2.4e-6 to 3.9e-6. ODE integration error at that resolution does
  not explain the observed order-one spectral errors.

See [independent-review.md](independent-review.md) and its JSON output for
the complete checks and limitations. Reviewer context reuse is disclosed.

## Interpretation

Solving the finite Dirichlet problem is computationally easy at these sizes.
It does not establish the required singular asymptotics or monodromy. Fixed
boundary values prescribe more than weighted flags and do not prescribe the
connection's radial derivatives. The source precheck identified this risk
before the runs. The observed failure is consistent with that risk, but the
experiment does not separate boundary truncation, spatial/interpolation
error and unverified seed applicability.

The levels change cutoffs, outer domain and logarithmic grid together.
Although node counts increase, spacing does not uniformly shrink on every
fixed annulus. This is not a controlled mesh-convergence study, and no
continuum-limit or convergence-rate inference follows.

The practical decision is **NO-GO for further tuning of this direct pilot**.
My earlier recommendation to prioritize the metric route was ahead of the
computational evidence. Reopening it should require a concrete method for
imposing/controlling the peripheral monodromy and a compatible discretization,
with a residual-to-holonomy error argument. A smaller Newton residual or a
larger grid by itself would not meet that requirement.

## Reproduction and scope

The fixture is already stored; do not regenerate it for a replay. From the
repository root, run the following with `MODE` replaced by `control` or by
`target --level 0`, `target --level 1`, `target --level 2`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 45s prlimit --as=1073741824 --cpu=30 python crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R386-parabolic-metric-prototype/prototype.py MODE
```

This overwrites the corresponding numerical output and timings. Independent
checks use the same resource prefix with `independent-verifier.py` or
`independent-fixture-check.py` instead. Source/data hashes are retained in the
independent outputs; metric fixtures are indexed in evidence/corpora.toml.

R0386-M1 records a finite numerical failure, not a mathematical refutation.
FRONTIER_DELTA remains NONE. No production implementation or corpus was
changed. The research directory remains local under the ignored dev/ tree.
