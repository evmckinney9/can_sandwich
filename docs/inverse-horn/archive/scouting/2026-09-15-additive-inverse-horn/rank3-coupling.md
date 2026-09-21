# Rank-three chains: the nonadjacent overlap and a fixed-chain inverse

2026-09-15. Algebraic response to the user's supplied LR-chain proposal.
This derives a test and reconstruction for a supplied chain, not a selector
of intermediate spectra from endpoints. No novelty claim or general Horn
coverage claim is made. Existing independent reviewer context reused;
see [algebra review](rank3-chain-algebra-review.md).

## 1. Input and conventions

Write the positive strengths as d1,d2,d3, with d1>=d2>=d3>0.
Let alpha,rho,tau,Gamma be decreasing real four-tuples. Require the trace
increments d1,d2,d3 and the three positive rank-one interlacings. Put
p_s(t)=prod_i(t-s_i). The intermediate spectra rho and tau must be simple.

An unrestricted chain has unit real directions v,w,u and

    A1=A0+d1 vv^T, A2=A1+d2 ww^T, A3=A2+d3 uu^T.

It realizes the prescribed rank-three summand precisely when the three
directions are mutually orthogonal. U1 diagonalizes A1; U2 diagonalizes
U1^T A2 U1, so U2 is expressed relative to the first eigenbasis. Define

    V=U1^T v, z=U1^T w, W=U2^T z, y=U2^T U1^T u,
    a_j=V_j z_j, b_k=W_k y_k.

The a_j here are overlap products, not the LR-triangle entries a_ij.

## 2. The nonadjacent equation without unknown eigenbases

The determinant lemma gives four sets of nonnegative weights:

    V_j^2 =  p_alpha(rho_j)/(d1 p_rho'(rho_j)),
    z_j^2 = -p_tau(rho_j)/(d2 p_rho'(rho_j)),
    W_k^2 =  p_rho(tau_k)/(d2 p_tau'(tau_k)),
    y_k^2 = -p_Gamma(tau_k)/(d3 p_tau'(tau_k)).

Each set sums to one by the corresponding trace increment. Thus define

    c_j = a_j^2 = -p_alpha(rho_j)p_tau(rho_j)
                  /(d1 d2 p_rho'(rho_j)^2),
    e_k = b_k^2 = -p_rho(tau_k)p_Gamma(tau_k)
                  /(d2 d3 p_tau'(tau_k)^2).

The middle-update eigenvector equation is

    (tau_k-rho_j)(U2)_jk = d2 z_j W_k.

For disjoint rho and tau it proves the exact identities

    v^T w = sum_j a_j,
    w^T u = sum_k b_k,
    v^T u = d2 sum_jk a_j b_k/(tau_k-rho_j).

The last expression is the missing E13. It couples the signed terms from
the two adjacent equations through a Cauchy matrix. It is not justified at
common roots by assigning arbitrary values to 0/0 terms.

## 3. The LR specialization, with its common roots retained

Pak–Vallejo Section 3 and Theorem 4.1 give the boundary convention and the
linear LR/hive correspondence used in the user's proposal. Reading letters
successively does give the asserted interlacings. The LR inequalities also
contain lattice-word inequalities; interlacing alone is not the whole hive
condition. For delta4=0 the diagonal LR entries are nonnegative, since the
LR inequalities imply a11>=a22>=a33>=a44=0.

The natural LR chain satisfies

    rho1=tau1=Gamma1, tau2=Gamma2.

Assume these are the only consecutive overlaps relevant here, and in
particular the remaining middle-update gaps are strict:

    tau2 > rho2 > tau3 > rho3 > tau4 > rho4.

The residue equations force z1=W1=y1=y2=0. Consequently a1=b1=b2=0.
Only j=2,3,4 and k=3,4 remain in E13; every denominator in that restricted
sum is nonzero. If E23=0 and b3 is nonzero, b4=-b3. Put

    R(t)=sum_{j=2}^4 a_j/(t-rho_j).

Then E13=0 iff R(tau3)=R(tau4). If E12=0, the numerator of R over
prod_{j=2}^4(t-rho_j) has degree at most one.

## 4. Eliminate all sign choices on this strict chart

Assume c2,c3,c4,e3,e4 are positive. These guards exclude additional zero
residues, beyond the common roots already handled. Define

    D_j = 1/((tau3-rho_j)(tau4-rho_j)), j=2,3,4,
    n = (D4-D3, D2-D4, D3-D2),

where the components of n are indexed 2,3,4. Strict interlacing gives
D2,D4>0 and D3<0, hence n2>0 and n4<0.

Since R(tau3)-R(tau4)=(tau4-tau3) sum_j D_j a_j,
the first and third orthogonality conditions are exactly

    (1,1,1) dot a = 0, (D2,D3,D4) dot a = 0.

Their common kernel is the line spanned by n. Therefore a fixed chain in
this domain admits mutually orthogonal updates if and only if

    e3 = e4,
    c3 n2^2 = c2 n3^2,
    c4 n2^2 = c2 n4^2.                         (*)

There are no square-root sign choices in (*). If n3=0 then c3>0 makes
the second equality fail. These are rational spectral tests, not a claim
that the LR inequalities imply them.

In particular, the branch with a2 and a4 of the same sign cannot work.
This excludes the adjacent-balance branch |a3|=|a2|+|a4| on this strict
chart, whenever its three magnitudes are positive.

## 5. Sufficiency supplies an actual reconstruction

This also shows why freely selecting the required signs here is legitimate.
Work in the rho eigenbasis, reconstructing backwards and forwards from it:

1. Choose z_j as nonnegative square roots of their residue weights, with
   z1=0. Form M2=D_rho+d2 zz^T and take an orthogonal eigenbasis U2 with
   ordered eigenvalues tau. Set W=U2^T z.
2. Set a=(sqrt(c2)/n2)n. For j=2,3,4 put V_j=a_j/z_j. Choose either
   permitted sign of V1 from its residue weight.
3. Form M0=D_rho-d1 VV^T. Its characteristic polynomial is p_alpha:
   the determinant-lemma ratio has the prescribed residues and value at
   infinity. The trace condition gives ||V||=1.
4. Choose b3=sqrt(e3), b4=-b3, set y3=b3/W3, y4=b4/W4 and y1=y2=0.
   Their squared coordinates equal the final-update residue weights.
   Set u=U2 y. The matrix M3=M2+d3 uu^T has spectrum Gamma, and
   ||u||=||z||=1 by the trace increments.
5. The three equations in (*) and Section 4 give V perpendicular to z,
   z perpendicular to u, and V perpendicular to u. Thus
   M3-M0=d1 VV^T+d2 zz^T+d3 uu^T has spectrum (d1,d2,d3,0).
6. Diagonalize M0=T D_alpha T^T. Transform the three directions by T^T
   and complete them to an orthonormal four-frame. Choose the last
   column's sign so its determinant is +1. This is the required Q.

Every division in this construction is covered by the positivity guards.
No independent consistency condition on the signs remains: V determines
the backwards update and y the forwards update around the fixed middle
update. The magnitude-only tests became sufficient because this explicit
reconstruction supplies the signs and bases together.

## Scope and handoff

Outcome: explicit E13 and an exact necessary-and-sufficient test with
reconstruction for the stated strict, supplied LR chain. The result does
not show that a feasible endpoint triple has any chain passing (*), that
each hive should keep its natural LR chain, or that boundary chains are
covered. It neither establishes nor excludes a nonlinear hive-to-matrix
map. Its usefulness is deciding fixed-chain lifting without a sign search.

This is a derivation audit of the user-supplied proposal, not a preregistered
generic construction attempt. FRONTIER_DELTA for endpoint selection: NONE.
The registry records the scoped lemma as unreviewed at project-certificate
level; the independent mathematical review is linked above. No numerical
benchmark was run. The exact remaining obligation is selection of rho,tau
from alpha,delta,Gamma with proved coverage and controlled construction
cost. No implicit authorization for another speculative route is inferred.

Primary source checked: [Pak–Vallejo, Sections 3–4 and Theorem 4.1](https://www.math.ucla.edu/~pak/papers/liri91.pdf). The calculations in
Sections 2–5 are derived here, not attributed to that source.
