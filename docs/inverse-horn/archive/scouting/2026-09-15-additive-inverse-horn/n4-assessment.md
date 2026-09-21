# Assessment of the proposed n=4 hive inverse

2026-09-15. User requested progress on another agent's proposed additive n=4
hive-to-real-matrix parameterization. This is a bounded mathematical and
primary-source assessment of that proposal, not a new solver attempt.

## What is correct

For real symmetric existence, fix A=diag(alpha) and B=Q diag(beta) Q^T.
Q can be in SO(4): multiplying one column of an orthogonal Q by -1 changes
its determinant and leaves B unchanged. This does not parameterize all
complex Hermitian equivalence classes, only the real realization problem.

With trace already fixed, equality of power sums of orders 2,3,4 is exactly
equivalent to equality of the four eigenvalues as a multiset. Newton identities
recover the characteristic polynomial. This statement itself needs no generic
assumption. Genericity enters the rank and dimension assertion.

The expansions for tr(C²), tr(C³), tr(C⁴) in the assessment are correct. In
particular, the first two nontrivial equations are linear in P_ij=q_ij².
The quaternion representation covers SO(4), but it is a representation of
the unknown eigenframe, not an input-only selection of that frame.

## The dimension coincidence is not special to n=4

For a regular value of the spectral-invariant map, with simple factor spectra,

    dim SO(n) - (n-1) = n(n-1)/2 - (n-1) = (n-1)(n-2)/2.

This equals the number of internal hive coordinates for every n. Full
dimensional fixed-boundary hive polytopes have this dimension; special boundary
data can lower it. Thus n=3 already has a nontrivial one-dimensional match.

For comparison, the generic complex Hermitian moduli space has real dimension
(n-1)(n-2): fix A, start with the n²-n dimensional orbit of B, impose n-1
sum spectral conditions, and quotient the effective (n-1)-dimensional diagonal
unitary stabilizer of A. For real symmetric matrices the remaining sign gauge
is finite. Consequently the real dimension count is half the complex count.
This explains the count without furnishing a map between individual objects.

## The orthostochastic observation needs one qualification

At a Q with all entries nonzero, the map Q -> P=Q circ Q is locally injective
on each fixed sign sheet: recover q_ij=sign(q_ij)*sqrt(P_ij). Its derivative
sends delta q_ij to 2q_ij delta q_ij and is injective there.

Globally, P can have more than one inequivalent sign lift. But these are
finitely many choices, not a missing continuous angle. On any chosen sign
sheet, tr(ABAB) is already a function of P. Its quartic occurrence does not
by itself reveal a separate continuous coordinate to recover last.

Conversely, selecting an arbitrary doubly stochastic P satisfying the first
two linear equations is not sufficient. Orthostochasticity requires compatible
signs making all columns orthogonal simultaneously. The proposal does not
provide this selection. This is a scope observation, not a new polynomial
method. Relevant existing sources: CHEN-DEY-2001.10691, DEY-1708.09559.

## The rank-two paper does not establish the advertised cutoff

Subtracting beta4 I leaves a generic rank-three summand, as claimed. The
target must also be shifted:

    spec(D_alpha + sum_{j=1}^3 (beta_j-beta4) u_j u_j^T)
        = gamma - beta4*(1,1,1,1).

The three vectors are orthonormal. This identifies the next rank for that
construction program, not the first dimension with no general constructor.
Gift–Woerdeman's introduction acknowledges Franks's approximate constructor.
Its own rank-two intermediate-selection implementation uses numerical
optimization and sign enumeration; exact reconstruction formulas do not
turn that implementation into a proved finite exact selector.

Primary: https://journals.uwyo.edu/index.php/ela/article/download/8697/7007/24669
See additive-low-rank.md and additive-inverse-scaling.md for audited sections.

## What the 2024 hive paper establishes

Bercovici–Li study the candidate forward map

    h(p,q)=max Tr(A E + (A+B) F),

over mutually orthogonal projections E,F of ranks p,q. At n=4 its internal
coordinates are h(1,1), h(1,2), h(2,1). Their counterexample defeats a
projection-recombination argument intended to prove the hive inequalities;
it is not a refutation of the proposed hive function itself. Their studied
examples also supply positive construction results in a unique-hive setting.

Even proving this function always gives a hive would not establish that it
attains every fixed-boundary hive on the real symmetric fiber. Forward
validity, real surjectivity, and inverse reconstruction are different claims.

Primary: https://imar.ro/journals/Revue_Mathematique/pdfs/2024/3-4/4.pdf
Independently checked introduction and Example 4.5; see n4-assessment-review.md.

## A precise version of the actual research problem

The displayed condition in the quoted proposal only requires

    spec(D_alpha+Q(h) D_beta Q(h)^T)=gamma.

For fixed boundary data a constant Q(h)=Q0 satisfies that condition for every
hive, provided Q0 is any exact witness. Therefore the condition does not yet
express a hive inverse or a parameterization.

One meaningful stronger problem is to specify a forward map Phi on the real
matrix fiber modulo the relevant sign gauges and seek a computable section:

    Phi(Q(h)) = h,
    spec(D_alpha+Q(h) D_beta Q(h)^T)=gamma.

The compression-trace map above is one precisely defined candidate, not an
established surjective map. Alternatively, a parameterization of all witnesses
must explicitly require coverage and specify how multiple sheets and boundary
identifications work; it need not use this particular forward map.

For any chosen forward map, the concrete obligations are:

1. Every output lies in the correct hive polytope.
2. Every required hive has a real preimage, not merely a complex one.
3. Those preimages can be constructed from the input data.

Equal dimensions prove none of these. Even a nonsingular derivative at one
point would give only local coordinates, not global coverage. In particular,
a nonempty regular Q fiber is a compact manifold without boundary. A full
dimensional hive polytope has boundary, so they cannot be globally identified
by a homeomorphism without changing the domain or introducing identifications.
This does not rule out a surjection, multiple charts, or a section for some map.

## Result and handoff

The assessment's basic invariant formulation survives. Its n=4-only dimension
claim does not; its literature cutoff is overstated; its proposed inverse
condition omits the relationship to the interior hive coordinates. These are
corrections to the proposed justification, not evidence that hive construction
is impossible or that scaling solves a supplied-hive lift.

No hive-to-matrix constructor has been obtained in this review. No new
prototype, polynomial search, or corpus test was run. CLAIMS.yaml is unchanged:
this is an external-proposal/source audit, not a new accepted project result.
Independent reviewer context reused under AGENTS.md. The remaining research
obligation is a specified map with real coverage and an effective inverse.
