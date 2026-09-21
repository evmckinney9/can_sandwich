# R0388: image of the compression-trace hive map

Claim IDs affected: R0388-H1, initially unreviewed.

Hypothesis: Every valid fixed-boundary additive n4 hive has a Hermitian
preimage under the Danilov-Koshevoy compression-trace prescription. The real
symmetric restriction is the user's preferred setting; complex exclusion
would imply real exclusion. This tests a proposed inverse-map premise, not
the separate conjecture that every forward output is a hive.

Assumptions/domain: n4, C=A+B Hermitian, arbitrary real spectral boundary
lists with trace compatibility; all three internal hive entries prescribed.
Strict/simple extension is desirable but not assumed. Preliminary hand
exploration preceded this written record; it is not represented as a fully
prospective preregistration. Claim now frozen before certificate construction.

Acceptance test: A proof of coverage/construction, or one exactly specified
valid hive with feasible spectral boundary data and a proof that no matrix
triple attains it. A failed numerical optimization is not exclusion evidence.

Cheapest falsifier: A hive endpoint saturates a variational bound that forces
incompatible common eigenspaces. Every possible equality case, including
coincident top eigenvectors and repeated eigenspaces, must be included.

Budget: 35 minutes, one image/surjectivity hypothesis, at most three cycles
(identify exact candidate, complete certificate, independent review). No
prototype benchmark, corpus or production use. Computations <=1GiB address
space,30sCPU,45swall,one BLAS thread. No successor after a final falsification.

Sources: BERCOVICI-LI-HIVES-2024; KNUTSON-TAO-MATH-9807160;
FULTON-MATH-9908012. No novelty presumed without source comparison.

Result: UNKNOWN for general surjectivity. The tested candidate endpoint has
an exact real preimage supplied by independent review, so it is not a
counterexample. No general construction or non-surjectivity theorem obtained.
Evidence: gate-0.md; independent-candidate-review.md; certificate.py;
certificate.json; README.md. LP values were exploratory only; all final hive
inequalities and matrix certificates were checked exactly.
Registry update: R0388-H1 remains unreviewed/unproved; the attempted exclusion
failed. No frontier reduction or claimed mathematical advance.
