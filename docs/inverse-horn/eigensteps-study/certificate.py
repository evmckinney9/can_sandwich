"""Exact checks for the eigensteps study; not a coverage experiment.

Possible outcomes: failure identifies an error in a displayed example or
identity; success checks those finite calculations only. The general
extension and error bounds require the proofs in README.md.
"""

import json

import sympy as s


def zero(matrix):
    assert all(s.simplify(entry) == 0 for entry in matrix)


def polynomial(roots, x):
    return s.prod(x - root for root in roots).expand()


def spectrum(matrix, roots):
    x = s.Symbol("x")
    assert s.expand(matrix.charpoly(x).as_expr() - polynomial(roots, x)) == 0


def top_kill(lam, mu):
    """The source's single-choice selector, exact arithmetic only."""
    assert len(lam) == len(mu)
    assert all(v >= 0 for v in (*lam, *mu))
    assert list(lam) == sorted(lam, reverse=True)
    assert list(mu) == sorted(mu, reverse=True)
    assert sum(lam) == sum(mu)
    assert all(sum(lam[:j]) >= sum(mu[:j]) for j in range(1, len(mu)))
    result = [list(lam)]
    while len(lam) > 1:
        k = next(k for k in range(len(lam) - 1) if lam[k + 1] <= mu[-1] <= lam[k])
        shorter = list(lam[:k]) + [lam[k] + lam[k + 1] - mu[-1]] + list(lam[k + 2 :])
        assert all(lam[i] >= shorter[i] >= lam[i + 1] for i in range(len(shorter)))
        mu = mu[:-1]
        assert sum(shorter) == sum(mu)
        assert all(sum(shorter[:j]) >= sum(mu[:j]) for j in range(1, len(mu)))
        lam = shorter
        result.append(lam)
    return result


rt2 = s.sqrt(2)
worked = s.Matrix([
    [6, 2 * rt2, 1, -rt2 / 2],
    [2 * rt2, 4, -rt2, 1],
    [1, -rt2, 4, rt2 / 2],
    [-rt2 / 2, 1, rt2 / 2, 2],
])
chain = top_kill([8, 5, 3, 0], [6, 4, 4, 2])
assert chain == [[8, 5, 3, 0], [8, 5, 1], [8, 2], [6]]
for n, roots in enumerate(reversed(chain), 1):
    spectrum(worked[:n, :n], roots)
assert list(worked.diagonal()) == [6, 4, 4, 2]

# This second fixture has an exact commuting Horn witness. Its displayed
# relaxed Top Kill output has a nonzero within-A inner product.
alpha = s.diag(3, 1, 0, 0)
beta_witness = s.diag(1, 0, 2, 0)
spectrum(beta_witness, [2, 1, 0, 0])
spectrum(alpha + beta_witness, [4, 2, 1, 0])
relaxed = s.Matrix([
    [3, rt2, -1 / s.sqrt(3), 0],
    [rt2, 2, s.sqrt(s.Rational(2, 3)), 0],
    [-1 / s.sqrt(3), s.sqrt(s.Rational(2, 3)), 1, 0],
    [0, 0, 0, 1],
])
chain2 = top_kill([4, 2, 1, 0], [3, 2, 1, 1])
for n, roots in enumerate(reversed(chain2), 1):
    spectrum(relaxed[:n, :n], roots)
ea = relaxed.extract([0, 2], [0, 2]) - s.diag(3, 1)
eb = relaxed.extract([1, 3], [1, 3]) - s.diag(2, 1)
assert s.trace(ea**2) == s.Rational(2, 3)
zero(eb)
spectrum(ea, [1 / s.sqrt(3), -1 / s.sqrt(3)])

# Both polar factors below are supplied exactly, so this tests the
# identities and the deficient-rank extension, not a floating-point SVD.
ta = s.eye(4)
ta[0, 0] = ta[3, 3] = s.Rational(3, 5)
ta[0, 3], ta[3, 0] = -s.Rational(4, 5), s.Rational(4, 5)
tb = s.eye(4)
tb[1, 1] = tb[2, 2] = s.Rational(5, 13)
tb[1, 2], tb[2, 1] = -s.Rational(12, 13), s.Rational(12, 13)
ha = s.Matrix([[2, 1, 0], [1, 1, 0], [0, 0, 1]])
hb = s.Matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
spectrum(ha, [(3 + s.sqrt(5)) / 2, (3 - s.sqrt(5)) / 2, 1])
spectrum(hb, [2, 1, 0])
ua, ub = ta[:, :3], tb[:, :3]
x, y = ua * ha, ub * hb
assert x.rank() == 3 and y.rank() == 2
da, db = s.diag(5, 2, 1), s.diag(2, 2, 1)
errors = []
for frame, iso, h, d in [(x, ua, ha, da), (y, ub, hb, db)]:
    zero(iso.T * iso - s.eye(3))
    zero(frame.T * frame - h**2)
    assert list((frame.T * frame).diagonal()) == list(d.diagonal())
    defect = frame.T * frame - d
    change = iso * d * iso.T - frame * frame.T
    zero(change + iso * defect * iso.T)
    assert s.trace(change**2) == s.trace(defect**2)
    errors.append(int(s.trace(defect**2)))

q = ta.T * tb
assert ta.det() == tb.det() == q.det() == 1
zero(q.T * q - s.eye(4))
ahat, bhat = ua * da * ua.T, ub * db * ub.T
zero(ta.T * (ahat + bhat) * ta - (s.diag(5, 2, 1, 0) + q * s.diag(2, 2, 1, 0) * q.T))
spectrum(ahat, [5, 2, 1, 0])
spectrum(bhat, [2, 2, 1, 0])

p, qv, r = s.symbols("p q r", real=True)
e = s.Matrix([[0, p, qv], [p, 0, r], [qv, r, 0]])
assert s.expand(s.trace(e**2) - 2 * (p**2 + qv**2 + r**2)) == 0

print(json.dumps({
    "status": "PASS",
    "worked_intermediate_spectra": chain,
    "feasible_warning_intermediate_spectra": chain2,
    "feasible_warning_defect_A": "2/3",
    "feasible_warning_defect_B": "0",
    "polar_fixture_defects": errors,
    "polar_fixture_ranks": [3, 2],
    "scope": "Exact finite identity checks; no coverage or convergence claim; author verification.",
}, indent=2))
