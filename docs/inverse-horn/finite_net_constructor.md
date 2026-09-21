# A certified finite-net constructor for n=4

This note gives a covered but deliberately impractical epsilon-level baseline.
Given Horn-feasible
endpoint spectra and a tolerance epsilon > 0, it returns a specified Q in
SO(4) whose sum spectrum is within epsilon of gamma. It is exhaustive and
can be expensive. It is not a radicals-only formula.

## Algorithm

Let b = max_i |beta_i|. If b = 0, return Q = I. Otherwise choose

    h = epsilon / (24 b).

Use the standard six-factor QR Givens factorization of SO(4). Enumerate the eight
diagonal sign matrices D = diag(d1,d2,d3,d4), where each di is plus or minus
one and their product is one. Enumerate all six-tuples of angles on the mesh
h Z intersected with [-pi,pi]. For each tuple form

    Q = G34 G23 G12 G34 G23 G34 D.

Here each Gij is the identity except for a two-dimensional rotation by the
chosen angle in coordinates i,j. The displayed order is the reverse of the
QR elimination order; it uses the six planes 34, 23, 12, 34, 23, 34.
Evaluate the ordered eigenvalues of
D_alpha + Q D_beta Q^T, using interval eigenvalue bounds when the input is
numerical. Return the first candidate whose certified residual is at most
epsilon/2. The mesh guarantee below gives residual at most epsilon/4, leaving
a strict margin for interval certification. If input uncertainty consumes
that margin, refine the intervals and halve h.

The QR/Givens factorization represents every Q_star in SO(4) with one of the
eight sign matrices and six angles. Rounding each angle to the nearest mesh
point changes it by at most h/2. Each Givens factor is 1-Lipschitz in its
angle in operator norm, so telescoping gives

    ||Q - Q_star||_op <= ||Q - Q_star||_F <= 6(h/2) = 3h.

For M(Q) = D_alpha + Q D_beta Q^T,

    ||M(Q) - M(Q_star)||_op <= 2 b ||Q - Q_star||_op <= 6 b h = epsilon/4.

Weyl's inequality therefore gives

    max_i |lambda_i(M(Q)) - gamma_i| <= epsilon/4 < epsilon/2.

Horn feasibility supplies Q_star, so the exhaustive search must contain a
passing candidate at the first mesh in exact arithmetic. The refinement loop
only handles numerical certification slack; it is not a heuristic test.

The returned columns are the actual matrix directions. No intermediate
spectra, sign balances, or unproved hyperbolicity point is selected. The cost
is at most

    8 (1 + ceil(2 pi / h))^6

eigenvalue evaluations. This is effective in the real-oracle model, but the
bound is not a practical runtime guarantee. For
algebraic or rational input, all comparisons can be certified by algebraic
root isolation or validated intervals.

## Adaptive branch-and-bound form

The same proof gives a less wasteful certified search. Represent each
six-angle mesh cell by a centre c and an L1 angular radius r (the sum of its
six half-widths), and let Q(c) be its
Givens product. Compute the exact or interval-enclosed residual f(Q(c)). The
Lipschitz estimate gives the lower bound

    LB(cell) = max(0, f(Q(c)) - 6 b r).

Discard a cell when LB(cell) exceeds the requested tolerance. Otherwise
bisect its longest angular side and continue with the cell having smallest
lower bound. Keep the best certified candidate seen so far. A cell containing
an exact Horn witness is never discarded below its true residual zero. Once
its radius is small enough, its centre is a passing candidate. Therefore the
branch-and-bound procedure terminates for every positive tolerance, with the
same coverage and error contract as the full mesh but typically many fewer
eigenvalue evaluations.

## Verification obligations

The residual must be certified in the original 4 by 4 sum. A small residual
in a Givens parameter, Gram defect, or auxiliary tensor invariant is not
sufficient. Interval eigenvalue enclosures must include input uncertainty.
Exact algebraic inputs can use characteristic-polynomial root isolation.

## Runtime warning

For b = 1, the full-mesh count is approximately 8 (48 pi / epsilon)^6:
about 10^20 evaluations at epsilon = 10^(-1), 10^32 at epsilon = 10^(-3),
and 10^50 at epsilon = 10^(-6). Adaptive pruning may help on favorable
instances, but no pruning-rate theorem or benchmark is claimed. This baseline
proves coverage; it is not a practical constructor.
