# What went wrong, and what must change

The recurring failure was treating a clearer representation of an unknown
as evidence that we could select it. The user's objection was not that
algebraic formulations are illegitimate. It was that reductions, special
cases, and references repeatedly arrived with claims of significance that
their proofs did not support.

## Distinctions the investigation repeatedly blurred

- Matching dimensions does not define a correspondence. The generic real
  Horn fiber and fixed-boundary hive have dimension (n-1)(n-2)/2 for every
  n, not only n=4.
- Three remaining parameters do not imply an easy inverse. Their admissible
  region may still encode the entire selection problem.
- An inverse given compatible internal data does not select those data from
  endpoints. This is the gap in the rank-three chain construction.
- An equivalent problem is not an easier problem until a solver or a useful
  structural theorem transfers back with a proved contract.
- A theorem about simultaneous unitary tridiagonalization does not imply a
  real tridiagonal summand while the sum stays diagonal. T4 is false.
- Numerical successes do not prove coverage. Numerical failures do not
  prove nonexistence. R387's UNKNOWN cases and the supplied Givens reports
  must retain that distinction.
- A small residual in an auxiliary equation need not control the requested
  matrix or monodromy error. R386 demonstrates this failure in a prototype.
- A plausible obstruction can miss valid equality cases. R388's candidate
  had an exact preimage and did not refute the hive map.
- Correctness, novelty, and practical usefulness are different claims. The
  local lemmas have not been shown to constitute a publishable advance.

## Required standard for future work

State the endpoint input, requested output, and permitted operations before
claiming a construction. Name the exact remaining choice. Separate the proof
that a choice exists from the procedure that finds it and the bound that
makes the procedure useful. Test any proposed universal restriction against
exact, structured cases before using numerical coverage as encouragement.

For a normal form, quantify over all its permitted permutations and gauges.
For a conditional inverse, retain every guard and explain how the input will
satisfy them. For a geometric method, explain how its abstract coordinates
produce the original real frame. For an iterative procedure, state its error
contract, failure behavior, termination conditions, and computational model.

Independent review must be capable of rejecting the central claim. Record
failed candidates, corrections, and stopping decisions alongside successful
identities. Do not replace a failed hypothesis with another exciting story
without closing the failed attempt.

The honest outcome of this investigation is partial structural understanding
and two exact shortcut exclusions. It does not justify saying the desired
construction is close to completion.
