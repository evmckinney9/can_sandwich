"""Standalone empirical additive inverse-Horn selector for n=4.

Given real spectra alpha, beta, gamma, minimize the three nontrivial
characteristic-coefficient residuals of diag(alpha)+Q diag(beta) Q^T over six
Givens angles. A final eigvalsh certificate checks the requested spectrum.
This is an implementation/benchmark baseline, not a coverage theorem.
"""
import numpy as np
import time
from scipy.optimize import least_squares

PLANES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def givens(i, j, x):
    q = np.eye(4); c, s = np.cos(x), np.sin(x)
    q[i, i] = q[j, j] = c; q[i, j] = -s; q[j, i] = s
    return q


def q_of(a):
    q = np.eye(4)
    for p, x in zip(PLANES, a): q = q @ givens(*p, x)
    return q


def coeffs(x):
    e1 = np.trace(x); e2 = (e1*e1 - np.trace(x@x))/2
    e3 = (e1**3 - 3*e1*np.trace(x@x) + 2*np.trace(x@x@x))/6
    return np.array([e1, e2, e3, np.linalg.det(x)])


def select(alpha, beta, gamma, starts=16, seed=0):
    a0 = np.diag(alpha); b0 = np.diag(beta); goal = coeffs(np.diag(gamma))
    rng = np.random.default_rng(seed); best = None
    def fun(a): return coeffs(a0 + q_of(a) @ b0 @ q_of(a).T) - goal
    for _ in range(starts):
        r = least_squares(fun, rng.uniform(-np.pi, np.pi, 6),
                          bounds=(-np.pi, np.pi), max_nfev=1200,
                          xtol=1e-13, ftol=1e-13, gtol=1e-13)
        z = np.linalg.norm(r.fun)
        if best is None or z < best[0]: best = (z, r.x)
        if z < 1e-11: break
    x = a0 + q_of(best[1]) @ b0 @ q_of(best[1]).T
    got = np.linalg.eigvalsh(x)
    return best[0], float(np.max(np.abs(got - np.sort(gamma)))), q_of(best[1])


def check(cases=1000, starts=16, seed=20260919):
    began = time.perf_counter(); rng = np.random.default_rng(seed); failures = 0; worst = 0.0
    for n in range(cases):
        alpha = np.sort(rng.normal(size=4))[::-1]
        beta = np.sort(rng.normal(size=4))[::-1]
        q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        if np.linalg.det(q) < 0: q[:, 0] *= -1
        gamma = np.linalg.eigvalsh(np.diag(alpha) + q @ np.diag(beta) @ q.T)
        _, e, qgot = select(alpha, beta, gamma, starts=starts, seed=seed+n)
        worst = max(worst, e); failures += e > 1e-8
        assert np.linalg.norm(qgot.T @ qgot - np.eye(4), ord=np.inf) < 1e-12
        assert abs(np.linalg.det(qgot) - 1.0) < 1e-12
    print({"cases": cases, "starts": starts, "failures": failures,
           "failure_rate": failures/cases, "worst_eigenvalue_error": worst,
           "elapsed_seconds": time.perf_counter() - began})


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser(); p.add_argument("--cases", type=int, default=1000)
    p.add_argument("--starts", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--alpha", type=str); p.add_argument("--beta", type=str)
    p.add_argument("--gamma", type=str)
    p.add_argument("--tol", type=float, default=1e-8)
    a = p.parse_args()
    if a.alpha and a.beta and a.gamma:
        parse = lambda s: np.sort(np.fromstring(s, sep=','))[::-1]
        alpha, beta, gamma = parse(a.alpha), parse(a.beta), parse(a.gamma)
        if any(x.size != 4 for x in (alpha, beta, gamma)):
            p.error("alpha, beta, gamma must each contain four comma-separated values")
        c, e, q = select(alpha, beta, gamma, starts=a.starts)
        print(json.dumps({"success": bool(c <= a.tol and e <= a.tol), "tolerance": a.tol,
               "coefficient_residual": float(c), "eigenvalue_error": float(e),
               "Q": q.tolist(), "orthogonality_error": float(np.linalg.norm(q.T@q-np.eye(4), ord=np.inf)),
               "determinant": float(np.linalg.det(q))}))
    else:
        check(cases=a.cases, starts=a.starts, seed=a.seed)
