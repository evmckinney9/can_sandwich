"""Standalone empirical generic n=4 selector (does not call can_sandwich).

Inputs are unit-modulus determinant-one spectra.  A feasible target is made by
sampling a random SO(4) witness.  The selector minimizes the three independent
characteristic-coefficient residuals over six Givens angles and accepts only a
final rootwise certificate.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import expm

# One factor for each coordinate plane.  This is the usual six-angle Euler
# chart for SO(4); the previous repeated-plane chart was only a restricted
# family and could not support a coverage claim.
PLANES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def givens(i, j, x):
    g = np.eye(4); c, s = np.cos(x), np.sin(x)
    g[i, i] = g[j, j] = c; g[i, j] = -s; g[j, i] = s
    return g


def q_of(a):
    q = np.eye(4)
    for p, x in zip(PLANES, a): q = q @ givens(*p, x)
    return q


def q_exp(a):
    k = 0; x = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            x[i, j] = a[k]; x[j, i] = -a[k]; k += 1
    return expm(x)


def coeffs(m):
    e1 = np.trace(m)
    e2 = (e1 * e1 - np.trace(m @ m)) / 2
    return np.array([e1.real, e1.imag, e2.real])


def coeffs6(m):
    e1 = np.trace(m)
    e2 = (e1 * e1 - np.trace(m @ m)) / 2
    e3 = (np.trace(m)**3 - 3*np.trace(m)*np.trace(m @ m) + 2*np.trace(m @ m @ m)) / 6
    return np.array([e1.real, e1.imag, e2.real, e2.imag, e3.real, e3.imag])


def roots_from_mono(m):
    """Canonical magic-basis roots from monodromy coordinates."""
    c1, c2, c3 = m[0] + m[1], m[0] + m[2], m[1] + m[2]
    h = np.pi / 2
    phases = h * np.array([c1-c2+c3, c1+c2-c3, -c1-c2-c3, -c1+c2+c3])
    return np.exp(2j * phases)


def rho_roots(m):
    w = roots_from_mono(m)
    return np.array([-w[2], -w[3], -w[0], -w[1]])


def select(lam, mu, target, starts=32, seed=0):
    rng = np.random.default_rng(seed)
    goal = coeffs(np.diag(target))
    def fun(a): return coeffs(np.diag(lam) @ q_of(a) @ np.diag(mu) @ q_of(a).T) - goal
    best = None
    for k in range(starts):
        x0 = rng.uniform(-np.pi, np.pi, 6)
        r = least_squares(fun, x0, bounds=(-np.pi, np.pi), max_nfev=1200,
                          xtol=1e-13, ftol=1e-13, gtol=1e-13)
        if best is None or np.linalg.norm(r.fun) < best[0]: best = (np.linalg.norm(r.fun), r.x)
        if best[0] < 1e-10: break
    err, a = best
    m = np.diag(lam) @ q_of(a) @ np.diag(mu) @ q_of(a).T
    got = np.linalg.eigvals(m)
    root_err = max(min(abs(x-y) for y in target) for x in got)
    return err, root_err


def select_exp(lam, mu, target, starts=12, seed=0):
    rng = np.random.default_rng(seed); goal = coeffs(np.diag(target))
    def fun(a): return coeffs(np.diag(lam) @ q_exp(a) @ np.diag(mu) @ q_exp(a).T) - goal
    best = None
    for _ in range(starts):
        r = least_squares(fun, rng.uniform(-2.0, 2.0, 6), max_nfev=1000,
                          xtol=1e-12, ftol=1e-12, gtol=1e-12)
        z = np.linalg.norm(r.fun)
        if best is None or z < best[0]: best = (z, r.x)
        if z < 1e-10: break
    m = np.diag(lam) @ q_exp(best[1]) @ np.diag(mu) @ q_exp(best[1]).T
    got = np.linalg.eigvals(m)
    return best[0], max(min(abs(x-y) for y in target) for x in got)


def check(trials=1000, seed=20260916, starts=32):
    rng = np.random.default_rng(seed); failures = 0; worst = 0.0; worst_coeff = 0.0; failed_rows = []
    for n in range(trials):
        lam = np.exp(1j * rng.uniform(-2.5, 2.5, 4)); lam[3] = np.exp(-1j*np.sum(np.angle(lam[:3])))
        mu = np.exp(1j * rng.uniform(-2.5, 2.5, 4)); mu[3] = np.exp(-1j*np.sum(np.angle(mu[:3])))
        a = rng.uniform(-np.pi, np.pi, 6)
        target = np.linalg.eigvals(np.diag(lam) @ q_of(a) @ np.diag(mu) @ q_of(a).T)
        c, e = select(lam, mu, target, starts=starts, seed=seed+n)
        worst_coeff = max(worst_coeff, c)
        worst = max(worst, e)
        if e > 1e-7:
            failures += 1; failed_rows.append(n)
    print({"trials":trials, "failures":failures, "failure_rate":failures/trials,
           "worst_root_error":worst, "worst_coeff_residual":worst_coeff,
           "failed_rows": failed_rows})


def check_monodromy_corpus(path="crates/can_sandwich/corpus/feasible_stratified.npy",
                           rows=1000, starts=8, exponential=False):
    data = np.load(path, mmap_mode="r")[:rows]
    failures = 0; worst = 0.0; worst_coeff = 0.0; failed_rows = []
    for n, (c, g, t) in enumerate(data):
        lam, mu = roots_from_mono(c), roots_from_mono(g)
        fn = select_exp if exponential else select
        attempts = [fn(lam, mu, roots_from_mono(t), starts=starts, seed=n),
                    fn(lam, mu, rho_roots(t), starts=starts, seed=100000+n)]
        # Keep the coefficient residual for the better of the two target lifts;
        # the other lift is a deliberately rejected branch.
        chosen = attempts[int(np.argmin([x[1] for x in attempts]))]
        worst_coeff = max(worst_coeff, chosen[0])
        errs = [x[1] for x in attempts]
        e = min(errs); worst = max(worst, e)
        if e > 1e-7:
            failures += 1; failed_rows.append((n, float(e)))
    print({"rows": rows, "failures": failures,
           "failure_rate": failures/rows, "worst_root_error": worst,
           "worst_coeff_residual": worst_coeff,
           "failed_rows": failed_rows})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", action="store_true")
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--starts", type=int, default=32)
    args = p.parse_args()
    if args.corpus:
        check_monodromy_corpus(rows=args.rows, starts=args.starts)
    else:
        check(args.rows, starts=args.starts)
