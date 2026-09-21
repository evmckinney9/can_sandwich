# R0386 independent implementation and numerical review

Date: 2026-09-15. Reviewer: delegated agent, uninvolved in authoring this
prototype. Context was reused from the source-synthesis and unrelated R0385
audits; it was not erased. Read the frozen attempt, Gate 0, precheck,
prototype source, fixture and results. No candidate tuning was performed.

## Terminal verdict

**NO-GO for this fixed finite-domain implementation at its declared 1e-4
original-target root tolerance.** All three returned matrices are numerical
SO(4) frames, but their original fixed-factor products miss the target by
large margins. The decreasing frame error does not pass the acceptance
test. This supports parking this bounded prototype; it does not refute
the parabolic correspondence or every metric constructor.

The observations below are independently reproduced floating-point results.
They are not interval certificates, a continuum convergence proof, a
universal failure result, or a theorem promotion. FRONTIER_DELTA=NONE.

## Equation and implementation scope

For the stated column-section convention, A=h^-1 h_z dz. Differentiating
gives flatness equivalent to

```text
Delta h - (h_x+i h_y) h^-1 (h_x-i h_y) = 0.
```

The prototype implements this equation, with the correct order of the
noncommuting factors. Its nonlinear term is Hermitian, and its sixteen
real components span full Hermitian matrices. The analytic Newton
derivative has all three differentiated nonlinear factors with the correct
signs. The line search checks positive eigenvalues and discrete residual
decrease. These checks support the finite discretization, not its global
convergence or asymptotic accuracy near punctures.

The manufactured metric h=g* g, g=I+zN, is valid: N is strictly upper
triangular, so g is everywhere invertible, and h^-1 h_z=g^-1 g' is
holomorphic. Independent reconstruction of this metric gives a maximum
interior finite-difference residual 2.06e-14 and minimum eigenvalue 0.46668.
The author records solver error 2.89e-12. The solved control metric itself
was not saved, so that last error was inspected in the log and source,
not independently recomputed from a separate control output array.

The scalar sign check is consistent: h=|z|^(2 mu) yields CCW holonomy
exp(-2 pi i mu). The code explicitly negates the two factor weights and
uses the target weights at infinity. Its loops share a real basepoint;
the metric square-root conjugation there has the correct direction.
This checks numerical conventions, not semistability of the chosen flags
or correctness of the global signed-weight extension.

## Input and absence of a supplied-frame shortcut

The target-mode code reads only `fixture.json`. It does not read
`fixture-feasibility.json`, invoke the fixture generator, reconstruct the
planted random frame from its seed, or optimize an endpoint residual.
The fixture was generated from a frame, which is retained separately for
verification. That makes it a planted feasible test input, but the planted
realization does not enter the solve. Fixed flag bases are the supplied
seed data; no stability certificate is present.

Independent fixture checking gives an original-product root error
4.997e-16, orthogonality error 6.06e-16 and determinant 1. The three root
lists are simple, with minimum pairwise chord gaps 1.0181, 0.7943 and
0.4810. Their phase spreads are below one. The chosen factor logarithms
have operator-norm sum 1.62 pi, outside R0242's guard in this convention.
This does not show that every alternative lift/logarithm fails, and no
positive-degree QLR contribution was independently certified here.

## Independent results

The verifier imports no prototype functions. It reconstructs derivative
stencils by solving local polynomial interpolation systems, reads the
saved metric arrays, and checks the original-factor sandwich directly.
Root matching exhausts all 24 assignments, including the optimal
bottleneck assignment, so the failure is not an artifact of minimizing
sum distance instead of worst-root distance.

| Level | Original-frame target root error | Normalized discrete PDE residual | Minimum metric eigenvalue | Large CCW loop target root error |
|---|---:|---:|---:|---:|
| 0 | 1.0264312 | 5.8771e-13 | 0.15177 | 0.89659 |
| 1 | 0.8073789 | 6.5242e-15 | 0.09473 | 0.71808 |
| 2 | 0.6500450 | 5.3341e-12 | 0.05913 | 0.59946 |

All frame orthogonality errors are below 1.4e-15, with determinant within
7e-16 of one. The metrics satisfy Hermiticity within 1.8e-15, conjugation
symmetry h(x,-y)=conjugate(h(x,y)) within 6.8e-14, and the specified
Dirichlet values within 3.6e-15. Independently computed normalized
residuals agree with the author logs. These are scaled discrete residuals;
unscaled RMS residuals are also retained in the verifier output.

Independent bilinear interpolation and fixed-step RK4 transport reproduce
the two factor-loop errors to about 3e-6. Those errors stay around
0.64–0.73, despite improving frame target error. At level 2, the raw product
of the two transported holonomies misses target roots by 0.62910, so the
failure already appears before replacement by prescribed factor spectra
in the extracted orthogonal frames.

## Third loop and closure

The prototype itself records only two peripheral loops. The independent
verifier additionally integrates the large CCW loop specified during
review, from the same basepoint. Its inverse is the infinity-oriented
loop, compared to the inverse target class. Both comparisons fail by
roughly the final table column.

For column transport, the chosen large loop traverses the right puncture
before the left, giving the flat-connection order U0 U1. Its Frobenius
differences from that product are 0.13982, 0.14923 and 0.17241. Differences
from the opposite order are 2.39129, 2.24803 and 2.13810. Thus neither
order restores closure. This is a direct loop comparison, not the
tautological definition of a third matrix as the inverse product.

Doubling the independent transport steps from 128 to 256 per segment
changes the loop matrices by at most 3.95e-6. This numerical sensitivity
check is far smaller than the observed errors, but is not an ODE error
certificate. It does not certify the interpolated connection as flat.

## Attribution and refinement limits

The three levels change puncture cutoff, outer domain and grid together.
The geometric-grid ratio does not uniformly improve on fixed annuli as
the node count increases. They are **combined truncation/grid levels**,
not a controlled mesh convergence experiment. No convergence rate or
asymptotic error conclusion follows from the decreasing frame error.

The first independently failed acceptance gate is the original-target
root check. Metric positivity and the small discrete PDE residual pass;
factor classes and continuum loop closure do not. The evidence does not
isolate finite-cutoff amplitudes, seed/degree preparation, spatial
discretization or holonomy interpolation as the sole cause. In particular,
calling this only a boundary/truncation failure would exceed the evidence.
No further candidate runs or successor construction are justified by this
review's completion.

## Reproduction and evidence

From the repository root:

```sh
timeout 45s prlimit --as=1073741824 --cpu=30 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R386-parabolic-metric-prototype/independent-verifier.py
```

The verifier commands exited successfully within these caps. The final
command includes the independent transport-step sensitivity check. Durable
evidence is `independent-verifier.py` and
`independent-verifier-output.json`; the latter records SHA-256 hashes of
the reviewed source, fixture, run outputs and saved metrics. No registry
or production files were edited by the reviewer.
