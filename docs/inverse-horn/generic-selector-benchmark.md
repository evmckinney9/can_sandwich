# Generic six-Givens selector: benchmark record

This is an independent implementation in
[`multiplicative-generic-selector.py`](multiplicative-generic-selector.py). It
does not call the Rust `can_sandwich` solver. It parameterizes an `SO(4)`
matrix by six Givens angles and minimizes three real characteristic
coefficients, then checks the resulting eigenvalues against the requested
target. The method is an empirical baseline; it is not an exact selector
theorem.

## Results

| input | starts | result |
|---|---:|---|
| 1,000 synthetic feasible witnesses | 32 | 0 failures; worst root error `1.12e-9` |
| 100 Haar-random `SO(4)` witnesses | 16 | 0 failures; worst root error below `3e-11` |
| stratified corpus, first 20 rows | 2 | 6 failures (`30%`); worst `1.85e-4`; 13.0 s |
| stratified corpus, first 100 rows | 2 | 15 failures (`15%`); worst `1.61e-1`; 117 s |

Rows 12 and 14 remain above the `1e-7` eigenvalue certificate even after 128
random starts. Their best errors are on the order of `1e-5` and `1e-4`.
Increasing starts therefore reduces some local-minimum misses but does not
establish coverage.

For comparison, the existing Rust production benchmark (a separate prior
implementation) reports 13,179/13,181 solved on the full corpus, at 118.40
microseconds per triple. Two rows remain unsolved there (rows 2889 and 7172).
This result is included only as a performance/reference point, not as evidence
for either new selector.

As a targeted cross-check, the Givens selector solves reference failure row
2889 (coefficient residual `9.2e-16`, root error `6.4e-15` in eight starts),
while row 7172 remains ill-conditioned (best root error `3.5e-4`). Thus the
reference solver's two misses are not a universal obstruction: at least one is
an implementation/routing failure in that separate solver.

Using a three-point finite-difference Jacobian with `x_scale="jac"` on row 7172
reduced its best coefficient residual to `6.4e-10` in 100 starts, but did not
produce a rootwise certificate. This separates optimizer conditioning from the
remaining spectral-clustering issue.

That configuration was tested on 100 rows and produced 15 failures with two
starts (117 s), so it is retained only as a diagnostic refinement, not the
default search policy.

The target-root separations explain why the rootwise number is unstable: row 12
has a minimum separation of about `6.3e-11`, row 14 about `1.3e-4`, and row 7172
about `1.8e-3`. Coefficient residuals are therefore the more reliable progress
measure on the near-collision strata, while exact certification still requires
validated polynomial root isolation.

## Reproduction

```bash
PYTHONPATH=crates/can_sandwich/docs/inverse-horn \
  python crates/can_sandwich/docs/inverse-horn/multiplicative-generic-selector.py \
  --corpus --rows 100 --starts 2

cargo test --release --manifest-path crates/can_sandwich/Cargo.toml --test solver production_cases
```

The result answers a narrow implementation question: generic six-angle
optimization is adequate on random interior witnesses, but its extension
mechanism does not cover the corpus. The unresolved construction obligation is
still an explicit, guaranteed way to choose and extend the matrix directions,
especially on the singular/boundary strata.

## Global finite-net prototype

[`global-quaternion-net.py`](global-quaternion-net.py) removes the Euler-chart
coverage issue by enumerating normalized integer vectors on each of the two
unit 3-spheres in the double cover (S^3\\times S^3\\to SO(4)). For lattice
radius `K`, it checks (((2K+1)^4-1)^2) candidates before duplicate removal,
so termination and the selected direction are explicit. The nearest-lattice
bound gives a covering radius at most `2/(K-1)` on each sphere for `K>1`;
quaternion multiplication converts this to a computable spectral error bound.

Measured exhaustive runs on Haar witnesses were:

| K | candidates | time | best root error |
|---:|---:|---:|---:|
| 1 | 6,400 | 3.9 s (10 cases) | 0.255 (worst) |
| 2 | 389,376 | 15.6 s (1 case, batched) | 0.0816 |

On the first 10 rows of `feasible_stratified.npy`, `K=1` evaluates both target
lifts (12,800 products per row) in 7.3 s; the worst best-lift error is 0.0952.
On the first 100 rows the same exhaustive search takes 43.0 s; the worst
best-lift error is 0.4373.

The enumeration/map self-check plants three targets from net directions and
recovers all three with root error exactly `0.0` at `K=1`.

This is a globally covering selector for a prescribed tolerance, but its coarse
accuracy and exponential cost make it a certified approximation baseline, not
a practical exact construction.

## Additive inverse-Horn baseline

The original additive problem is implemented separately in
[`additive-generic-selector.py`](additive-generic-selector.py). It receives
real ordered `alpha`, `beta`, and `gamma`, minimizes the three nonconstant
characteristic coefficients plus the trace of `diag(alpha)+Q diag(beta) Q^T`, and certifies
the sorted eigenvalues directly. On 1,000 feasible instances generated from
Haar-random witnesses, eight starts per instance gave **0 failures**; the
worst eigenvalue error was `6.2e-10` (about 81 s total). Every returned matrix
also passed `||Q^TQ-I||_∞<1e-12` and `|det(Q)-1|<1e-12`. These are planted
feasible cases, so they test reconstruction, not universal coverage.

The trace is now part of the solve residual: a deliberately trace-infeasible
CLI input returns coefficient residual `6.96` and eigenvalue error `1.48`,
rather than being silently treated as feasible.

For an explicit input, the same program accepts comma-separated spectra:

```bash
python additive-generic-selector.py \
  --alpha 3,1,-1,-3 --beta 2,1,-1,-2 --gamma 5,2,-2,-5 --starts 32
```
The `--tol` option (default `1e-8`) adds a machine-readable `success` field
requiring both coefficient and eigenvalue residuals to be below that threshold.
`--seed` controls the deterministic benchmark stream (default `20260919`).

Three structured multiplicity families were also tested (100 Haar witnesses
each), including 3+1 and 2+2 repeated spectra. All 300 cases passed; the
largest eigenvalue error was `1.9e-12`.

On a fixed 100-case seed, restart sensitivity was measurable: 1 start gave
91/100 successes (10.2 s), while 2 starts gave 97/100 (11.2 s). The 8-start
setting used for the 1,000-case run eliminates these observed local-minimum
misses on planted data.
On the full 1,000-case workload, 2 starts gives 985/1,000 successes (1.5%)
with worst eigenvalue error `0.294`.
Four starts gives 999/1,000 (0.1%) with worst error `0.0285`; eight starts
gives 1,000/1,000 on this seed. The remaining single four-start miss is a
local-minimum event, not an (SO(4)) constraint failure.

The selector accepts `--tol ε` and stops at the first candidate whose direct
eigenvalue matching error is at most `ε`; with `--tol 0` it exhausts the net and
returns the best candidate. Thus the stopping rule is operational rather than
an unspecified “search for a point.”

`certified_k(ε,b)=1+ceil(8b/ε)` in the script is a conservative sufficient
radius from the lattice normalization and the two quaternion factors (with
`b=max(||D_λ||,||D_μ||)=1` for unitary inputs). It is intentionally enormous;
the benchmark uses smaller `K` only to measure practical behavior.
