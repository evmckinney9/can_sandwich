# R390: convex slices of the supplied quartic completion problem

User steering: assess the supplied three-coefficient real-rooted quartic
formulation and make concrete progress on its remaining obstruction.

One scoped hypothesis R0390-L1: fixing the cubic coefficient parameter s
leaves a compact convex feasible set in (t,u), characterized by explicit
critical-value halfplanes. If empty, at most three scalar halfplanes witness
infeasibility, provided the derivative passes the global cubic-root test.
The derivative-admissible s set is a compact interval (possibly empty).

Assumptions: monic depressed quartic with a(y) fixed quadratic,
b_s=b0+s phi, c=c0+phi(t+uy), phi=y(y^2-1), and b0,c0 fixed polynomials
of degree at most two. Real coefficients. Quantifier: all finite real y.
Repeated roots included; no Horn strictness assumption needed.

Acceptance: exact equivalence proof for single quartics including repeated
critical points, compactness argument, and independent review. Cheapest
falsifiers: critical-point multiplicities or an infinite halfplane family
with empty intersection but no finite empty subfamily. The anchors y=2,3
must explicitly remove the latter possibility.

Paper derivation preceded this written record; no claim of prospective
registration of discovery. Freeze this statement before symbolic checks.
Research integrity already passed. Budget: one proof/review cycle; symbolic
checks <=1GiB address space,30sCPU,45swall; no production or corpus.

Forward obligation: every true completion lies in the stated slice.
Inverse obligation: satisfying all its halfplanes gives global quartic
real-rootedness. Coverage: all coefficients under the displayed assumptions.
This is not a coverage claim for a finite sample of y values or an s selector.
FRONTIER_DELTA = NONE for endpoint-to-completion selection. The local
structural assertion is separately testable; no efficiency claim follows.
