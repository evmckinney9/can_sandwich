# R0388: attempted obstruction to a prescribed-hive inverse

Result: no general theorem or counterexample obtained. The surjectivity
question remains unresolved. One proposed nonattainment candidate is exactly
attainable. This result is not presented as progress on the general inverse.

## Candidate

alpha=gamma=(1,0,0,-1), beta=(1,1/2,-1/2,-1). The three internal entries
h(1,1),h(1,2),h(2,1) are all3/2. Linear hive feasibility suggested this
lower endpoint as a place to test simultaneous variational equality cases.
The exploratory LP also checked three nearby spectral choices and an
involution-spectrum control; those computations supply no exclusion evidence.

The intended argument was to force incompatible eigenspaces from equality
in lower variational bounds. No such incompatibility was proved. The
independent reviewer instead supplied an exact real matrix triple attaining
all three requested maxima. Merely showing one different matrix triple missed
the endpoint would not have been a valid refutation of surjectivity.

## Exact resolution of this candidate

See [independent-candidate-review.md](independent-candidate-review.md).
The reviewer gives A,C as two2x2 blocks and B=C-A. The proof bounds each
compression objective above by3/2 using Hermitian matrix majorants, and
exhibits real orthonormal vectors attaining that value. It therefore identifies
the prescribed hive itself, rather than just matching boundary spectra.

`certificate.py` independently replays the displayed algebra in exact SymPy
arithmetic. It checks characteristic polynomials,105 principal minors for the
positive-semidefinite majorants, orthonormal maximizers, and all18 hive rhombi.
It passes. One initial certificate assertion compared expanded and factored
polynomials structurally; replacing it with an expanded zero difference fixed
that coding error without changing the mathematical argument or candidate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 45s bash -c 'ulimit -v 1048576; ulimit -t 30; python crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R388-hive-image/certificate.py'
```

## Handoff

This attempt ends UNKNOWN for R0388-H1. The exact candidate fails as a
counterexample. No successor was started after that independent correction.
The remaining obligation is unchanged: prove real coverage and construct a
section of a specified hive map, or give a valid hive excluded for every
matrix realization. Forward hive validity is a different conjecture.

FRONTIER_DELTA=NONE. The complexity vector and generic inverse obligation
are unchanged. R0388-H1 is recorded as unreviewed, not verified or refuted;
its note states the terminal unknown outcome. The exact candidate certificate
does not promote the general claim. No production or corpus evidence used.

An existing uninvolved reviewer context was reused under AGENTS.md. The
preliminary hand exploration preceded the written gate, as disclosed in
attempt.md; no fully prospective preregistration claim is made. No novelty is
claimed for the elementary candidate lift or the majorant argument.
