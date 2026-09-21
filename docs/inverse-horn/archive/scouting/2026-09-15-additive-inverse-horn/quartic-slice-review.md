# Independent audit: convex planar slices of a quartic completion

2026-09-15. Existing uninvolved reviewer context reused. This reviews the
supplied structural statement; it does not produce a global selector,
implementation, or novelty claim. No computation was needed.

## Statement and assumptions checked

For each real y, consider the depressed monic quartic

```text
f(x;y) = x^4 + a(y)*x^2 + b_s(y)*x + c(y)
b_s(y) = b0(y) + s*phi(y)
c(y) = c0(y) + phi(y)*(t+u*y)
phi(y) = y*(y^2-1).
```

All coefficients and parameters are real and finite at every finite real y.
For the homogeneous-at-infinity observation below, assume additionally that
the full bivariate polynomial has total degree at most four. The conclusions
concern real-rootedness with multiplicities, not distinct-root requirements.

**Verdict:** the supplied real-rootedness criterion, compact convex planar
slices, at-most-three-halfplane infeasibility certificates, and closed bounded
derivative-admissible parameter interval are correct under these assumptions.

## 1. A single quartic, including repeated critical roots

Suppress y. The derivative is `4*x^3+2*a*x+b_s`, independent of t,u.
It has three real roots, counted with multiplicity, if and only if

`-8*a^3-27*b_s^2 >= 0`.

Indeed, dividing the derivative by four gives a depressed cubic with
discriminant `(-8*a^3-27*b_s^2)/16`. The displayed inequality also forces
`a<=0`.

Write the ordered derivative roots as `r1<=r2<=r3`, and define
`K(r)=3*r^4+a*r^2`. At any derivative root,
`b_s=-4*r^3-2*a*r`, and therefore

`f(r)=c-K(r)`.

The monic quartic is real-rooted if and only if

`K(r2) <= c <= min(K(r1),K(r3))`.

For distinct critical roots these are precisely the conditions that the two
local minima be nonpositive and the intervening maximum nonnegative. They
give a real root in each monotonicity interval, counting endpoint contacts
with their multiplicities.

If r1=r2 or r2=r3, the two inequalities at the coincident root force f=0
there. Since f' has a double zero there, f has a root of multiplicity at
least three; its remaining root is real. Conversely, a real-rooted quartic
has the required critical-value signs by interlacing and continuity. If all
three critical roots coincide, the depressed derivative has a=b_s=0 and the
interval condition forces c=0, giving x^4. Thus no strictness assumption or
excluded repeated-root case is hidden in this criterion.

## 2. Fixed s gives a closed convex compact set, if nonempty

First reject the slice if the derivative condition fails at any y. Otherwise
define finite real numbers

`L_s(y)=K(r2(y))`, `U_s(y)=min(K(r1(y)),K(r3(y)))`.

The slice is exactly the intersection, over all real y, of the scalar
halfplanes

```text
L_s(y) <= c0(y)+phi(y)*(t+u*y)
c0(y)+phi(y)*(t+u*y) <= U_s(y).
```

They are closed and convex, including the possibility of a constant
constraint at phi(y)=0. Therefore their intersection is closed and convex.

The actual phi gives `phi(2)=6` and `phi(3)=24`. The two samples bound
`v2=t+2u` and `v3=t+3u` to finite closed intervals. Since

`u=v3-v2`, `t=3*v2-2*v3`,

their four scalar halfplanes have bounded intersection: a closed
parallelogram, possibly degenerate or empty. Every feasible slice is a closed
subset of this bounded intersection and hence compact. No bound uniform in s
or uniform in the original input data is asserted.

## 3. Why infinite-y infeasibility has a finite certificate

Let K be the intersection of those four anchoring halfplanes. If K is empty,
ordinary finite planar Helly already yields an empty subintersection of at
most three of them.

Otherwise K is nonempty compact. If the full slice is empty, the complements
of all its halfplanes cover K. These complements are open, so compactness
supplies a finite subcover. The corresponding finitely many halfplanes,
together with the four anchors, have empty intersection. Apply the finite
planar Helly theorem to this family: at most three scalar halfplanes already
have empty intersection.

The selected constraints can be upper or lower bounds, and two may come
from the same y. Thus the conclusion is **at most three scalar halfplanes**,
not necessarily three distinct y samples or three two-sided strips. A
derivative failure is a separate one-sample certificate that no t,u works.

This argument establishes existence of a small infeasibility certificate.
It does not specify an algorithm to find its y values, bound their algebraic
complexity, or terminate a separation oracle. In particular, an infinite
family cannot be fed directly to finite Helly without the compactness step.

## 4. The derivative-admissible s values

For any y with a(y)>0, no s satisfies the derivative condition. Otherwise
that condition is

`|b0(y)+s*phi(y)| <= sqrt(-8*a(y)^3/27)`.

For phi(y) nonzero this is a finite closed interval of s. For phi(y)=0 it
is either no restriction or an immediate contradiction. The intersection
over all y is therefore a closed interval, possibly empty or a singleton.
The single sample y=2, with phi(2)=6, bounds it whenever it is nonempty.
Thus for the specified phi it is a **closed bounded interval or empty**.

These are the s values permitted by the derivative conditions alone.
Existence of suitable t,u for every such s does not follow. Nor does convexity
of every fixed-s planar slice imply that the full feasible set, or its
projection onto s, is convex or connected.

## 5. Infinity and the exact quantifier

The compactness and Helly proof above handles the full index set of finite
real y, even though that index set is unbounded. It does not require an
extra sampled point at infinity.

For a polynomial of total degree at most four, real-rootedness for every
finite y also implies real-rootedness of its leading homogeneous restriction.
For nonzero y, the polynomial `y^-4*f(y*z;y)` is monic of degree four in z
and has real roots. As y tends to infinity its coefficients tend to those
of the leading homogeneous polynomial evaluated at `(z,1)`. The set of
monic real-rooted quartics is closed, so that limit is real-rooted too.
Repeated limiting roots are permitted; no strict projective hyperbolicity
claim follows.

## Scope at closure

The supplied reasoning exposes a one-parameter family of compact convex
planar feasibility problems and small fixed-s infeasibility certificates.
It does not solve the global choice of s, supply a finite set of y tests for
all inputs, or construct a determinantal representation. Those are separate
obligations and were not explored in this bounded review.
