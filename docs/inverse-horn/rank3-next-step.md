# Proposed next step: degree on the full rank-three chain polytope

This is a proposal, not an established result. It follows R392's accepted
rank-two selector and R393's refutation of a fixed-rho vertex Miranda cell.

## Obligation

For generic positive strengths d1 >= d2 >= d3 and n=4, select intermediate
spectra rho,tau from the full LR chain polytope and make the three update
directions mutually orthogonal. The existing residue reconstruction already
does this once the chain and compatible signs are supplied.

## Mechanism

The unrestricted sequential-interlacing domain in (rho,tau) has six free
coordinates after the two trace constraints. Its expected zero set has
dimension three after imposing three orthogonality equations; the domain
itself is not three-dimensional. A more useful nesting is to apply the
verified rank-two selector to the remaining updates d2,d3 for each admissible
rho. This supplies tau and directions w,u with w perpendicular to u. The
remaining residual is the two-vector

    G(rho) = (v dot w, v dot u)

on the three-dimensional admissible rho domain.

On each collision stratum, retain the finite signed cover used
by the Cauchy coupling and define

    Phi = (v dot w, w dot u, v dot u).

The first two components are the adjacent residue balances; the third is the
nonadjacent Cauchy sum. Multiply by the known nonnegative residue factors so
that Phi extends continuously to a collision face. The extension must be
checked stratum by stratum; a formal 0/0 substitution is invalid.

The target proposition is:

> Every generic feasible endpoint triple has a sign chart and a compact
> two-cell S in the admissible rho domain, together with a continuous choice
> of the rank-two (tau,w,u) branch, on which the Brouwer degree of G at zero
> is nonzero.

Nonzero degree forces a zero. An interval-degree subdivision of S would then
give a terminating certified selector. The rank-three residue formulas would
recover v,w,u and hence Q. Rank and multiplicity boundaries would be handled
recursively by the same block reductions used in R392 and Gift--Woerdeman.

## Why this is the next step

R392 shows that Horn face-touching can force disjoint supports and a scalar
sign change in a box slice. R393 shows that taking vertices of a fixed-rho
tau-slice does not provide all three signs. The rank-two selector avoids
solving the w-perpendicular-u condition again; the new difficulty is proving
that its selected branch can be made continuous while rho varies, then
finding the remaining two zeros.

## Falsifiable first experiment

For exact planted rank-three witnesses, construct the admissible rho domain,
run the rank-two selector for the remaining pair, and evaluate the corrected
G on candidate two-cells. Compute the signed boundary degree, not a vertex
octant census. A stable nonzero degree across structured examples supports
the proposition; a zero degree identifies the precise boundary stratum where
the proposed mechanism fails. Numerical zeros alone do not establish
coverage.

No implementation or theorem is claimed by this note. If the degree vanishes,
the failure should be recorded before considering another mechanism.

## Preliminary continuity probe

An exploratory probe on eight planted rank-three witnesses sampled segments
from the true rho to vertices of its first-update interlacing polytope. Along
these segments the rank-two selector was unavailable at many sampled rho
values, and its selected support/sign label changed 1--3 times per segment.
This is numerical diagnostic evidence only, but it falsifies the stronger
idea of using one global continuous selector over the whole rho-domain.
The degree experiment must first enumerate the rank-two feasibility cells and
work on a cell with a fixed branch; proving that an appropriate cell contains
the required zero is the revised obligation.
