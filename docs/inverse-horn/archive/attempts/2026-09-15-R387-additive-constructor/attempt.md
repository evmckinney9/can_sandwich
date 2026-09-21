# R0387: executable additive n=4 construction

User instruction: make progress on the open problem using any technique.
This attempt delivers and evaluates a concrete additive witness constructor;
it does not claim to solve a prescribed-hive inverse or invent Franks scaling.

Claim IDs affected: R0387-M1.

Hypothesis: a spectra-only implementation of Franks's real scaling,
with elementary scalar/2x2 and supplied-spectra block assembly, returns directly
checked SO(4) additive witnesses on a frozen 12-case diagnostic packet at
maximum eigenvalue error <=1e-8 and orthogonality error <=1e-12.

Assumptions/domain: The packet is generated once with seed 387: four generic simple inputs; two
repeated-factor inputs; two exact block-diagonal boundary inputs; one scalar
factor; and three near-block inputs at mixing angles 0.1,0.01,0.001. Planted
frames are stored in a separate feasibility file and never passed to solve.
This finite-domain observation is not a generic coverage theorem.

Acceptance test: independent reconstruction of diag(alpha)+Q diag(beta) Q^T,
its eigenvalues, Q^T Q and determinant; input-only solver signature; all
UNKNOWN outcomes retained. Compare unchanged plain scaling with the block
wrapper on the same cases. No optimizer or planted-frame fallback.

Cheapest falsifier: slow convergence near or on a feasible wall despite small
normalized residual; test original unnormalized eigenvalues. Block candidates
must be accepted by the original full-matrix check, never merely a trace sum.

Budget: 25 minutes, at most three cycles (implementation control, frozen paired
packet, independent verification), 6000 scaling sweeps per call with one fixed
documented seed. At most 8 block candidates per input; child scaling capped
at 1000 sweeps. No tuning after the paired packet result; no successor method.
Each computation <=1 GiB address space,30s CPU,45s wall,one BLAS thread.

Sources: FRANKS-THESIS-2019-HORN; FRANKS-1801.01412;
FULTON-MATH-9908012. Existing source audits supply exact theorem scope.
The fixed seed is an implementation choice, not a universal success proof.
Numerical checks are not interval certification.

Result: the full finite acceptance test fails. Plain scaling accepts
5/12; the block wrapper accepts7/12. No post-packet tuning or successor method.
The numerical observation is recorded as R0387-M1; it is not a new theorem or
progress on a prescribed-hive inverse. See README.md and independent review.

Evidence: inputs.json, plants.json, results.json, independent.json, verifier.py,
review.md, constructor.py, gate-0.md and README.md.

Registry update: R0387-M1 records the finite comparison as observed; no theorem
promotion. Independent verification confirms returned matrices and one failed
scaling replay. Five inputs remain unresolved under the wrapper's fixed budget.
