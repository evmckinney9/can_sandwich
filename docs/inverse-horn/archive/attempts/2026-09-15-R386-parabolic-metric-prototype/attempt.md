# R0386: bounded real parabolic metric prototype

Claim IDs affected: R0386-M1 (proposed numerical observation only).

Hypothesis: A direct Chern-flat metric discretization with spectral weighted
flag boundary data can produce one original-factor SO(4) witness on a fixed
generic finite-angle rank-four input, without reading an input realization,
at rootwise error below 1e-4 within the stated pilot budget.

Assumptions/domain: One frozen feasible spectral triple; simple spectra,
signed trace-zero weights of spread below one, fixed generic real flag
bases. This is a finite-domain numerical approximation to the proposed
parabolic metric construction, not a certified stable-seed compiler or a
universal solver. Full Hermitian metrics retain conjugation symmetry.

Acceptance test: Exact-form smooth flat-metric control first. Then at most
three predeclared spatial/truncation levels, 30 Newton steps each, with
positive-definite line search. Accept only a returned SO(4) frame whose
original fixed-factor product independently matches target roots to 1e-4.
Report refinement, actual peripheral holonomy errors and total time. A
research GO additionally requires improvement with refinement and no hidden
input frame or endpoint optimizer. Production readiness is not tested.

Cheapest falsifier: A solved finite-domain Dirichlet metric can have the
wrong peripheral holonomy. Measure that directly; small PDE residual alone
is never success. Failure of the smooth manufactured control stops this
implementation before interpreting a target miss.

Budget: At most 50 minutes author wall time beginning 16:07 UTC; three
substantive cycles (control, coarse target, frozen refinement). Computational
commands capped at 1 GiB address space, 30 seconds CPU and 45 seconds wall;
one BLAS thread. No corpus/background runs or budget escalation. Numerical
implementation repairs within these cycles must be logged; no successor
mathematical hypothesis after a failed acceptance test.

Result: NO-GO for this direct numerical prototype. All three fixed levels
converge to small discrete residual but miss the original target root gate:
1.0264312, 0.8073789, 0.6500450 versus 1e-4. This is an observed finite
implementation failure, not a refutation of parabolic machinery. The smooth
manufactured control passes. No successor hypothesis or post-failure tuning.

Evidence: prototype-precheck.md; prototype.py; fixture.json and separately
stored fixture feasibility evidence; run outputs and independent review.

Registry update: R0386-M1 records the observed finite result; no theorem
promotion. README.md states the exact remaining boundary/discretization and
holonomy-control obligations. Independent review and closure are in the
handoff. Author time was within the 50-minute cap; all computations obeyed
the resource limits. No implementation repairs were made between runs.
