# R390: convex planar slices of quartic completion

Claim IDs affected: R0390-L1.

Hypothesis: Fixing s in the supplied quartic completion leaves a compact
convex feasible set K_s in (t,u). If the derivative is globally real-rooted
but K_s is empty, at most three scalar halfplane constraints already have
empty intersection. Derivative-admissible s values form a compact interval.

Assumptions/domain: a,b0,c0 real polynomials of degree at most two;
phi=y(y^2-1), b_s=b0+s phi, c_tu=c0+phi(t+uy). All finite real y;
repeated roots included. Empty intervals and empty compact sets allowed.

Acceptance test: Proof of the exact critical-value characterization with
repeated roots, explicit boundedness anchors, finite-certificate argument,
and independent review.

Cheapest falsifier: A multiple critical point invalidates the interval
criterion, or the infinite halfplane family escapes finite certification.

Budget: One proof and independent review; symbolic checks <=1GiB/30sCPU/
45swall. No corpus, matrix prototype, or selection search.

## 1. Exact single-quartic lemma

For f(x)=x^4+a x^2+b x+c, its derivative has three real roots counting
multiplicity if and only if

    -8 a^3 - 27 b^2 >= 0.

Assume this and order the roots r1<=r2<=r3. Put

    K(r)=3r^4+a r^2,
    L=K(r2), U=min(K(r1),K(r3)).

Then f has four real roots counting multiplicity if and only if L<=c<=U.
Indeed f(r_i)=c-K(r_i). With distinct critical points the quartic decreases,
increases, decreases, then increases, with positive tails. Four real roots
are equivalent to both minima being nonpositive and the maximum being
nonnegative. This gives exactly the three displayed bounds, with tangencies
allowed by counting multiplicity. Conversely Rolle's theorem gives the
derivative condition and these signs from four ordered real roots.

If two critical roots coincide at r, then a=-6r² and b=8r³. The common
minimum/maximum bounds force c=-3r^4. The quartic is exactly
(x-r)^3(x+3r), which has four real roots. The third critical point is -2r;
its upper bound is 24r^4, so the other bound holds. If all critical roots
coincide then a=b=0 and c=0, giving x^4. Thus all degeneracies are covered.

## 2. Apply the lemma at every y

First retain only s such that

    27 (b0(y)+s phi(y))^2 <= -8 a(y)^3, for every real y.

If a(y)>0 somewhere this set is empty. Otherwise each y condition is a
closed interval constraint on s (possibly all R or empty when phi=0).
Their intersection I is closed and convex. At y=2, phi=6 gives a finite
bound on s. Consequently I is a compact interval, possibly empty or a point.

For fixed s in I, compute the ordered roots of the derivative cubic at y,
and their L_s(y), U_s(y) from Section 1. The exact completion criterion is

    L_s(y)-c0(y) <= phi(y)(t+uy) <= U_s(y)-c0(y)
    for every real y.                                      (1)

Each side is a closed halfplane in the (t,u) plane, possibly an empty or
vacuous constraint when phi=0. Their intersection is exactly K_s. Therefore
K_s is closed and convex. The cubic is independent of t,u; no branch of
the original quartic is being assumed known.

At y=2 and y=3, (1) bounds respectively 6t+12u and 24t+72u in finite
intervals. The coefficient matrix has determinant 144. These four anchor
halfplanes have a bounded intersection B. If B is empty, K_s is empty;
otherwise B is compact and contains K_s. Hence K_s is compact.

## 3. Finite infeasibility certificate

If K_s is empty, intersect the entire closed-halfplane family with compact
B. The finite-intersection property implies that some finite family,
together with the four anchors, already has empty intersection. Planar
Helly then supplies at most three of those original scalar halfplanes with
empty intersection. If B was empty to begin with, apply planar Helly to
the four anchors directly. Thus three constraints suffice in either case.

These are three scalar inequalities, not necessarily three distinct y
values. Their locations are not prescribed and may depend on s and the
endpoint data. No fixed grid, three universal samples, bound on finding
the certificate, or corresponding finite feasibility certificate follows.

For completeness, the finite planar Helly fact can be obtained from
Radon's elementary observation: any four planar points admit disjoint
subsets with intersecting convex hulls (an affine dependence in R^3).
For four convex sets with every three intersecting, choose a point in each
triple intersection and apply that observation. The resulting convex-hull
intersection lies in all four sets. Induction, replacing the final two
sets by their intersection, proves the finite-family statement. This proof
also accommodates halfplanes and degenerate intersections.

## 4. Relation to the supplied proposal

The interpolation coefficients and Hermite leading minors are correct.
The rational p=p_x=p_y=0 parametrization describes candidate singular
completions at ordinary finite y away from the three interpolation slices.
It does not label which candidates satisfy global real-rootedness. The
critical-value inequalities (1) do provide the exact feasible-side test.

Cao–Woerdeman Theorem 2.1 supplies the existence equivalence. Its displayed
linear-algebra/spectral-factorization route produces Hermitian matrices;
the real symmetric existence conclusion uses a separate representation
theorem. A direct real symmetric implementation must not be silently
inferred from Hermitian factorization. Their Section 3 also distinguishes
simple-spectrum implementation assumptions from the existence theorem.

Actual saturated Horn inequalities give block constructions, but that does
not itself provide a continuation from the boundary to an interior target.
Boundary in an ordered chamber must also be distinguished from a saturated
Horn inequality. Neither claim is needed for this slice lemma.

All finite-y feasibility includes the limiting leading polynomial: divide
p(xy,y) by y^4 and let y tend to infinity. Monic real-rooted polynomials are
closed under coefficient limits, so no extra *weak* feasibility condition
at infinity is missing. Strict projective hyperbolicity is a different
condition and is not asserted here.

## 5. Result and exact remaining obligation

Result: The scoped convex-slice and finite-infeasibility assertions hold.
This reduces the geometry to a one-parameter family of convex planar
problems. It does not establish that {s:K_s nonempty} is an interval,
give an s selector, or show that an exchange/cutting-plane algorithm
terminates. An interval of derivative-admissible values is not the same
as the projection of the completion set.

Evidence: certificate.py checks the polynomial identities, all repeated
critical-point factorization cases symbolically, Hermite minors, and the
anchor determinant. Independent review checks the proof and quantifiers.
The finite Helly argument is given above; it is not a numerical inference.

Independent proof review: [quartic-slice-review.md](../../scouting/2026-09-15-additive-inverse-horn/quartic-slice-review.md).
Existing reviewer context reused under AGENTS.md; the author was not the
sole evaluator. Symbolic certificate result: PASS.

Registry update: R0390-L1 records the scoped structural lemma; endpoint
selection remains unresolved. No novelty assertion or practical complexity
reduction claimed. FRONTIER_DELTA for general inverse selection: NONE.

## Handoff

Reproduce from the repository root:

    timeout 45s bash -c 'ulimit -v 1048576; ulimit -t 30; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 crates/can_sandwich/docs/inverse-horn/archive/attempts/2026-09-15-R390-quartic-slices/certificate.py'
    make -f crates/can_sandwich/docs/inverse-horn/archive/Makefile research-check

The precise remaining task is selecting s with K_s nonempty and then a
point in K_s, with coverage and termination justified. No additional
representation or automatic successor attempt was launched.
