# Independent check of the supplied rank-three chain identities

2026-09-15. Existing reviewer context reused. This is a check of the supplied
identities and their stated overlap specialization, not a new solver, global
selector, or novelty claim. No numerical experiment was needed.

## Verdict and assumptions

The factors and signs in the supplied formulas are correct. The final
numerator has **degree at most one**, rather than necessarily degree one.

Use monic characteristic polynomials `p_s(t)=prod_i(t-s_i)`. Assume
`d1*d2*d3 != 0`, real unit update directions, and the actual symmetric chain

`A1=A0+d1 vv^T`, `A2=A1+d2 ww^T`, `A3=A2+d3 uu^T`.

Let U1 be an orthogonal eigenbasis of A1, and U2 an orthogonal eigenbasis of
`U1^T A2 U1`. Thus `U1^T A1 U1=D_rho` and
`U2^T(D_rho+d2 zz^T)U2=D_tau`, where `z=U1^T w`.
Set `V=U1^T v`, `W=U2^T z`, `y=U2^T U1^T u`,
`a_j=V_j z_j`, and `b_k=W_k y_k`. U2 is a relative eigenbasis, not a second
eigenbasis expressed in the original coordinates.

The residue formulas below require rho and tau separately simple. Their
disjointness is additionally needed for the unrestricted Cauchy formula.
Simplicity of alpha and Gamma is not needed for these residue evaluations.
Positivity of the d_i is not needed for the identities, although it matters
for the particular interlacing/feasibility contract being imposed.

## Factors and signs from four residue equations

In the A1 eigenbasis, `U1^T A0 U1=D_rho-d1 VV^T`. In the A2 eigenbasis,
`U2^T D_rho U2=D_tau-d2 WW^T`. The determinant lemma therefore gives

```text
d1 V_j² =  p_alpha(rho_j) / p_rho'(rho_j)
d2 z_j² = -p_tau(rho_j)   / p_rho'(rho_j)
d2 W_k² =  p_rho(tau_k)   / p_tau'(tau_k)
d3 y_k² = -p_Gamma(tau_k) / p_tau'(tau_k).
```

Multiplication proves exactly

```text
a_j² = -p_alpha(rho_j)*p_tau(rho_j)
        / (d1*d2*p_rho'(rho_j)²)
b_k² = -p_rho(tau_k)*p_Gamma(tau_k)
        / (d2*d3*p_tau'(tau_k)²).
```

The minus signs come from pairing a backward negative update with a forward
positive coefficient in the determinant lemma; they do not depend on the
numerical signs of the nonzero d_i.

## The Cauchy expression and all three overlaps

The eigenvector equation reads

`(tau_k-rho_j)*(U2)_jk = d2*z_j*W_k`.

For disjoint spectra this permits division in every entry. Substituting into
`v^T u = V^T U2 y` gives

`v^T u = d2*sum_jk a_j*b_k/(tau_k-rho_j)`.

Directly from orthogonal changes of basis,
`v^T w=sum_j a_j` and `w^T u=sum_k b_k`.
These are identities for the actual signed coordinates; replacing them by
their absolute values would change the orthogonality tests.

## The stated four-dimensional overlap specialization

Assume rho and tau are separately simple,
`rho1=tau1=Gamma1`, `tau2=Gamma2`, and no other overlap of rho with tau.
The residue equations imply

`z1=0`, `W1=0`, `y1=y2=0`.

Consequently `a1=0` and `b1=b2=0`. All k=1,2 terms in `V^T U2 y` vanish
before any division is made. For k=3,4, the j=1 eigenvector equation has
nonzero denominator and zero right side, so `(U2)_1k=0`. The only remaining
terms are j=2,3,4 and k=3,4, all with nonzero denominators:

`v^T u = d2*sum_{j=2..4,k=3..4} a_j*b_k/(tau_k-rho_j)`.

This reasoning is needed; the singular overlap entry must not be assigned a
value by formally substituting into a `0/0` expression.

If `E23 := w^T u=0`, then `b4=-b3`. With `b3 != 0` and `d2 != 0`,

`E13 := v^T u=0` if and only if `R(tau3)=R(tau4)`,

where `R(t)=sum_{j=2..4} a_j/(t-rho_j)`.
If b3=0, both remaining b's vanish and E13=0 automatically; the proposed
equivalence is then not justified.

Write the numerator over `D(t)=prod_{j=2..4}(t-rho_j)` as

`N(t)=sum_{j=2..4} a_j*prod_{ell!=j,ell=2..4}(t-rho_ell)`.

Its quadratic coefficient is `a2+a3+a4=E12`, since a1=0. Thus E12=0
removes the quadratic term. The remaining coefficient of t can also vanish,
so the exact conclusion is `degree(N)<=1` (including the zero polynomial).

## Gauge and scope

Changing the sign of an A1 eigenvector changes both V_j and z_j, leaving
a_j unchanged. Changing the sign of an A2 eigenvector changes both W_k and
y_k, leaving b_k unchanged. The displayed identities are therefore invariant
under eigenvector sign gauges. Eigenvalue order/permutations must be carried
consistently through all coordinates and polynomials.

The review certifies the displayed algebra conditional on an actual chain
and the stated nonzero/simple/overlap guards. Magnitude identities alone do
not solve E12,E23,E13 or select rho,tau. No claim of global sign selection,
coverage of an LR family, or completeness outside these guards follows.

## Extension: exact compatibility test for the supplied strict LR chain

The subsequently supplied equivalence and backward/forward reconstruction
also check out, with the following precise input assumptions. Write the
three update strengths as `s1,s2,s3` to distinguish them from the rational
coefficients below. They are nonzero. The three adjacent spectral steps
`alpha -> rho -> tau -> Gamma` must each be feasible rank-one updates with
these strengths: their determinant-lemma residue weights must be
nonnegative, and their trace increments must equal `s1,s2,s3`, respectively.
These conditions ensure the update directions are real and unit.

Retain separately simple rho,tau, exactly the stated overlaps, and the strict
ordering

`rho2 > tau3 > rho3 > tau4 > rho4`.

Assume the residue products `c2,c3,c4,e3,e4` are strictly positive. Strict
ordering of rho and tau alone does not establish feasibility of the alpha/rho
and tau/Gamma steps, so those adjacent-step assumptions cannot be dropped.

Define

```text
q_j = 1/((tau3-rho_j)*(tau4-rho_j)), j=2,3,4
n = (q4-q3, q2-q4, q3-q2)
c_j = a_j², e_k = b_k², using the residue formulas above.
```

Then this fixed chain admits mutually orthogonal real unit update directions
if and only if

```text
e3 = e4
(c2,c3,c4) = lambda*(n2²,n3²,n4²) for some lambda>0.
```

This is a conditional compatibility test for supplied intermediate spectra,
not a rule for selecting them.

### Necessity and the one-dimensional signed solution space

With `b4=-b3`, the earlier E13 condition becomes

`0=R(tau3)-R(tau4)=(tau4-tau3)*sum_j q_j*a_j`.

Thus E12=E13=0 is exactly the linear system
`sum_j a_j=0`, `sum_j q_j*a_j=0`. Strict ordering gives
`q2,q4>0` and `q3<0`. The two coefficient rows are independent, and their
kernel is the line spanned by n. In particular `n2>0` and `n4<0`, so the
outer signed products have opposite signs. `n3` can vanish; in that case
the assumed `c3>0` makes the test fail automatically.

For nonzero b3,b4, E23=0 gives equality of their squares. Taking squares of
the kernel condition gives the displayed c proportionality with a positive
factor. Conversely, the two conditions permit the choices
`a_j=sqrt(lambda)*n_j`, `b3=sqrt(e3)`, `b4=-sqrt(e3)`.

### Why the signed products can be realized together

1. Work in the D_rho frame. Choose z with the forward rho/tau residue
   squares and arbitrary signs on its nonzero components. Here z1=0 and
   z2,z3,z4 are nonzero. Form `M2=D_rho+s2 zz^T`, diagonalize it by U2,
   and set `W=U2^T z`. The determinant lemma gives spectrum tau.
2. For j=2,3,4 set `V_j=a_j/z_j`; choose V1 with the backward alpha/rho
   residue square. The c identities ensure every V_j has precisely its
   required square. Hence `M0=D_rho-s1 VV^T` has spectrum alpha.
3. For k=3,4 set `y_k=b_k/W_k`; these denominators are nonzero under the
   residue-product assumptions. Set y1=y2=0. The e identities ensure its
   remaining squares equal the forward tau/Gamma residues. Thus
   `M3=M2+s3 U2*y*y^T*U2^T` has spectrum Gamma.
4. Trace normalization gives `||V||=||z||=||y||=1`. More explicitly, the
   sum of the backward residue squares is
   `(sum(rho)-sum(alpha))/s1`; the two forward sums are
   `(sum(tau)-sum(rho))/s2` and `(sum(Gamma)-sum(tau))/s3`.
   Each equals one by the assumed trace increments.
5. The constructed directions `V,z,U2*y` have pairwise inner products
   E12,E23,E13, all zero by the selected a,b. Thus they realize the desired
   orthogonal updates simultaneously. The sign products do not carry an
   additional coherence obstruction on this chart: the construction supplies
   a common frame chain realizing them.

Finally diagonalize M0 by a real orthogonal Q with
`Q^T M0 Q=D_alpha`. Conjugate every matrix and update direction by the same
Q. This gives the requested chain starting at D_alpha, preserving every
spectrum, norm, and orthogonality equation. Its total update is the sum of
three mutually orthogonal rank-one terms, so its nonzero eigenvalues are
the prescribed strengths (in their sorted order).

The determinant-lemma reconstruction uses full characteristic polynomials,
not only selected coefficients. With simple denominator roots, the residue
identities and the common value one at infinity identify each rational
characteristic-polynomial ratio. No hidden eigenvector condition remains
after this explicit construction. The only unresolved broader task is
whether and how to select admissible rho,tau satisfying this conditional test;
that is outside this algebra review.
