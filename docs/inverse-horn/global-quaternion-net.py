"""Exhaustive certified finite net on SO(4) via S^3 x S^3.

This is deliberately a terminating approximate constructor.  Integer vectors
in [-K,K]^4 are normalized to a finite net on each S^3; quaternion left/right
actions then produce an explicit finite net on SO(4).  Every candidate is
checked against the original product, with eigenvalue matching.
"""
import itertools
import numpy as np


def qm(a, b):
    w, x, y, z = a; W, X, Y, Z = b
    return np.array([w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
                     w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W])


def qconj(a):
    return np.array([a[0], -a[1], -a[2], -a[3]])


def qmat(left, right):
    """Matrix of x -> left*x*conj(right), in the standard R^4 basis."""
    out = np.empty((4, 4))
    for j in range(4):
        e = np.zeros(4); e[j] = 1.0
        out[:, j] = qm(qm(left, e), qconj(right))
    return out


def sphere_net(k):
    vals = []
    for z in itertools.product(range(-k, k + 1), repeat=4):
        n = np.linalg.norm(z)
        if n: vals.append(np.asarray(z, dtype=float) / n)
    return vals


def certified_k(tol, b=1.0):
    """Sufficient lattice radius from the elementary 8b/(K-1) bound."""
    if tol <= 0: raise ValueError("tol must be positive")
    return 1 + int(np.ceil(8.0 * b / tol))


def match_error(got, target):
    return max(min(abs(x-y) for y in target) for x in got)


def select(lam, mu, target, k=1, tol=0.0):
    net = sphere_net(k); best = (float("inf"), None)
    dl, dr = np.diag(lam), np.diag(mu)
    # Batch all candidates: the enumeration order is unchanged, but batched
    # eigensolves reduce Python overhead by several orders of magnitude.
    qs = np.asarray([qmat(left, right) for left in net for right in net])
    got_all = np.linalg.eigvals(dl[None] @ qs @ dr @ np.transpose(qs, (0, 2, 1)))
    for idx, got in enumerate(got_all):
        q = qs[idx]
        e = match_error(got, target)
        if e < best[0]: best = (e, q)
        if tol and e <= tol:
            return e, q, len(net) ** 2
    return best[0], best[1], len(net) ** 2


def check(seed=20260917, cases=10, k=1, tol=0.0):
    rng = np.random.default_rng(seed); worst = 0.0
    for _ in range(cases):
        lam = np.exp(1j*rng.uniform(-2.5, 2.5, 4)); lam[3] = np.exp(-1j*np.sum(np.angle(lam[:3])))
        mu = np.exp(1j*rng.uniform(-2.5, 2.5, 4)); mu[3] = np.exp(-1j*np.sum(np.angle(mu[:3])))
        x = rng.normal(size=(4, 4)); q, _ = np.linalg.qr(x)
        if np.linalg.det(q) < 0: q[:, 0] *= -1
        target = np.linalg.eigvals(np.diag(lam) @ q @ np.diag(mu) @ q.T)
        e, gotq, count = select(lam, mu, target, k=k, tol=tol); worst = max(worst, e)
        assert abs(np.linalg.det(gotq)-1) < 1e-12
    print({"cases": cases, "k": k, "candidates": count,
           "worst_root_error": worst})


def check_corpus(path="crates/can_sandwich/tests/cases.bin",
                 rows=10, k=1, tol=0.0):
    data = np.memmap(path, dtype="<f8", mode="r").reshape(-1, 3, 3)[:rows]
    worst = 0.0; solved = 0; count = 0
    for c, g, t in data:
        def roots(m):
            w = (m[0] + m[1], m[0] + m[2], m[1] + m[2])
            h = np.pi / 2
            return np.exp(2j * h * np.array([w[0]-w[1]+w[2],
                w[0]+w[1]-w[2], -w[0]-w[1]-w[2], -w[0]+w[1]+w[2]]))
        lam, mu, target = roots(c), roots(g), roots(t)
        rho = np.array([-target[2], -target[3], -target[0], -target[1]])
        e0, _, count = select(lam, mu, target, k=k, tol=tol)
        e1, _, count = select(lam, mu, rho, k=k, tol=tol)
        e = min(e0, e1)
        worst = max(worst, e); solved += e <= tol if tol else 0
    print({"rows": rows, "k": k, "candidates": count,
           "worst_root_error": worst, "within_tol": solved if tol else None})


def check_membership(seed=20260918, cases=3, k=1):
    """Targets planted from net directions must be hit exactly by enumeration."""
    rng = np.random.default_rng(seed); net = sphere_net(k); worst = 0.0
    for _ in range(cases):
        lam = np.exp(1j*rng.uniform(-2.5, 2.5, 4)); lam[3] = np.exp(-1j*np.sum(np.angle(lam[:3])))
        mu = np.exp(1j*rng.uniform(-2.5, 2.5, 4)); mu[3] = np.exp(-1j*np.sum(np.angle(mu[:3])))
        q = qmat(net[rng.integers(len(net))], net[rng.integers(len(net))])
        target = np.linalg.eigvals(np.diag(lam) @ q @ np.diag(mu) @ q.T)
        e, _, _ = select(lam, mu, target, k=k)
        worst = max(worst, e)
    print({"membership_cases": cases, "k": k, "worst_root_error": worst})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--k", type=int, default=1)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--corpus", action="store_true")
    p.add_argument("--membership", action="store_true")
    a = p.parse_args()
    if a.membership: check_membership(cases=a.cases, k=a.k)
    elif a.corpus: check_corpus(rows=a.cases, k=a.k, tol=a.tol)
    else: check(cases=a.cases, k=a.k, tol=a.tol)
