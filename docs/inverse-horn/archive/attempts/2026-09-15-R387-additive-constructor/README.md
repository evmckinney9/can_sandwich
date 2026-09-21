# R0387 results

The full 12-case acceptance test did not pass. Plain real scaling accepts 5/12;
spectral block assembly plus scaling accepts 7/12. These are finite numerical
observations, not a new mathematical constructor theorem or a hive inverse.

| Input | Plain scaling | Block wrapper | Wrapper sweeps |
|---|---:|---:|---:|
| generic0 | UNKNOWN (4.16e-07) | UNKNOWN (4.16e-07) | 6000 |
| generic1 | PASS (4.99e-09) | PASS (4.99e-09) | 4329 |
| generic2 | PASS (4.99e-09) | PASS (4.99e-09) | 3193 |
| generic3 | PASS (4.98e-09) | PASS (4.98e-09) | 2343 |
| repeated0 | PASS (4.98e-09) | PASS (4.98e-09) | 3246 |
| repeated1 | UNKNOWN (1.89e-05) | PASS (8.88e-16) | 0 |
| wall22 | UNKNOWN (0.00734) | PASS (4.44e-16) | 0 |
| wall13 | UNKNOWN (0.0109) | UNKNOWN (0.0109) | 6000 |
| scalar | PASS (1.33e-15) | PASS (0) | 0 |
| nearwall0.1 | UNKNOWN (0.00439) | UNKNOWN (0.00439) | 6000 |
| nearwall0.01 | UNKNOWN (0.00731) | UNKNOWN (0.00731) | 6000 |
| nearwall0.001 | UNKNOWN (0.00734) | UNKNOWN (0.00734) | 6000 |

Errors are maximum absolute errors in the original sum eigenvalues. No
input spectrum was snapped to a boundary. Scalar normalizations are internal.
No planted frame was passed to the constructor. The block trace comparison
is numerical, not a certificate of an exact Horn equality; full-matrix output
checks determine acceptance.

## What the comparison establishes

On the selected 2+2 case and paired repeated-spectrum case, block assembly
constructs a witness to about machine precision, whereas 6000 plain scaling
sweeps miss the requested tolerance. The 1+3 case fails because its 1000-sweep
three-dimensional child does not reach tolerance; it then falls through to
the unchanged full scaling routine. This is not evidence that the split is
invalid. All three near-wall inputs fail the bounded sweep gate, as does one
generic input. Genericity here describes the fixture generation, not a proved
lower bound on its distance from every Horn facet.

The source theorem permits much larger tolerance-dependent iteration bounds
than this practical pilot. These failures neither refute the theorem nor prove
a convergence rate or asymptotic obstruction. The implementation returns
UNKNOWN at its limit, not infeasible. A fixed seed does not establish coverage.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 45s bash -c 'ulimit -v 1048576; ulimit -t 30; python crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R387-additive-constructor/run.py'
```

The frozen inputs and separate plant evidence are included. Independent
verification is in verifier.py, independent.json and review.md. No post-packet
method changes or tuning were performed. Numerical verification is not interval
certification. No claim of novelty or universal boundary handling follows.
