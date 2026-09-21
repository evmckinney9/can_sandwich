# Canonical exact selector (algebraic input model)

The finite-net method gives certified approximations for arbitrary real input.
For algebraic alpha, beta, gamma, there is also a canonical exact choice
of direction, with an explicit selection rule rather than an unspecified
"sample point".

Use the sixteen entries (q_{ij}) of (Q) as variables and define the
semialgebraic set

\[
\mathcal F=\{Q:Q^TQ=I,\ \det Q=1,\ 
\chi_{D_\alpha+QD_\beta Q^T}(x)=\chi_\gamma(x)\}.
\]

The characteristic-polynomial equality means equality of its four
coefficients; its leading coefficient is automatic. Horn feasibility makes
(mathcal F) nonempty, and (Q^TQ=I) makes it compact. Order the entries
lexicographically as
\[
(q_{11},q_{12},\ldots,q_{44}).
\]
Define (m_1) as the minimum of (q_{11}) on (mathcal F). Having defined
(m_1,\ldots,m_{r-1}), define (m_r) as the minimum of the next entry on
the subset where the earlier entries equal their minima. Compactness proves
that every minimum exists and that the final vector (m) is a unique
lexicographically least feasible matrix.

Each (m_r) is obtained by bisection over ([-1,1]) with the exact query
\[
\exists Q\in\mathcal F:
q_{1,1}=m_1,\ldots,q_{r-1}=m_{r-1},\ q_{ij}\le t.
\]
For algebraic coefficients, quantifier elimination over the real closed field
of algebraic numbers decides this query and returns an algebraic endpoint.
Equivalently, one may use a cylindrical-algebraic decomposition with the
variable order
\[
q_{11},q_{12},\ldots,q_{44},x
\]
and retain the lexicographically first sample cell. This is a finite,
terminating algorithm for a specified witness, not merely an assertion that
some algebraic sample exists.

The construction then sets (A=D_\alpha), (B=QD_\beta Q^T). Every direction
is a column of the selected (Q), and the polynomial equalities certify the
target characteristic polynomial exactly. Repeated spectra cause no special
case: they enlarge (mathcal F), while compactness still guarantees each
lexicographic minimum.

This selector is intentionally separated from the finite-net constructor.
The latter is the practical, tolerance-certified method for arbitrary real
data. The lexicographic method is exact and terminating for algebraic input,
but its quantifier-elimination complexity is substantial. For arbitrary
non-algebraic reals, exact output requires an input representation/oracle;
the finite-net error contract is the applicable statement.
