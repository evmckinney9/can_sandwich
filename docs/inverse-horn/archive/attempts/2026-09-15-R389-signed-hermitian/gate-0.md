# R389: direct signed-Hermitian reduction

2026-09-15. New user steering explicitly asks whether the three-dimensional
singular-value-sum inverse can be reduced to Hermitian Horn by sign/Weyl
choices. This is one bounded comparison, following closure of R388.

Hypothesis R0389-H1: Every feasible real 3x3 singular-value sum can be
realized by two 3x3 Hermitian matrices with the same input singular lists
and whose sum has the same output singular list, after arbitrary signs
and permutations. This allows more sign choices than proper signed SVD.

Acceptance: a proof covering all three lists. Cheapest falsifier: a feasible
real matrix sum for which no signed traces add. Candidate identified on
paper before this record: a=b=(3,2,1), c=(3sqrt(2),2sqrt(2),2). This
is a retrospective statement of that short calculation, not preregistration
of its discovery. The certificate and independent review remain pending
at creation. Research integrity passed before the certificate work.

Forward obligation: explicit rotations and exact singular spectra; inverse
obligation being tested: direct signs/permutations into Hermitian spectra;
coverage: universal over feasible triples. This does not test arbitrary
auxiliary-variable reductions or different singular lists.

FRONTIER_DELTA = NONE for general endpoint selection. A successful
counterexample refutes only this proposed shortcut. Exact 4x4 lift will
confirm it is also a feasible simple-spectrum original Horn instance.
One certificate, one independent review; no successor route after closure.
Compute cap: 1 GiB address space, 30 s CPU, 45 s wall; no corpus.
