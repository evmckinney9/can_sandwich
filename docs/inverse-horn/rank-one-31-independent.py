"""Independent rank-one (3+1) multiplicative selector check.

For A=diag(a,a,a,c), O diag(mu) O^T = a I + (c-a) u u^T.  The four
weights |u_i|^2 are recovered from e1, e2 and sum(weights)=1; a Householder
map then supplies O.  This script does not call can_sandwich.
"""
import numpy as np


def selector(lam, mu, target):
    # The second factor is the 3+1 gate: mu=(a,a,a,c).
    a, c = mu[0], mu[3]
    d = c - a
    e1m = np.sum(lam * a)
    # coefficient of w_i in e1 after the rank-one determinant lemma
    e1coef = lam * d
    e1rhs = np.sum(target) - e1m
    # e2(M)=((tr M)^2-tr(M^2))/2; evaluate each affine basis response
    def e2(z):
        return (np.sum(z) ** 2 - np.sum(z * z)) / 2
    base = lam * a
    e2base = e2(base)
    e2coef = np.empty(4, complex)
    for i in range(4):
        z = base.copy()
        z[i] += d * lam[i]
        e2coef[i] = e2(z) - e2base
    A = np.vstack([e1coef.real, e1coef.imag, e2coef.real, np.ones(4)])
    b = np.array([e1rhs.real, e1rhs.imag, (e2(target)-e2base).real, 1.0])
    w = np.linalg.solve(A, b)
    if np.min(w) < -1e-9:
        raise ValueError("target is outside this rank-one fibre")
    w = np.maximum(w, 0.0)
    w /= np.sum(w)
    u = np.sqrt(w)
    # Householder H e4 = u (with a sign repair for det +1).
    e = np.zeros(4); e[3] = 1.0
    v = e - u
    if np.linalg.norm(v) < 1e-14:
        H = np.eye(4)
    else:
        H = np.eye(4) - 2.0 * np.outer(v, v) / np.dot(v, v)
    if np.linalg.det(H) < 0:
        H[:, 0] *= -1
    return H, w


def selector_left(lam, mu, target):
    """Left 3+1 case, reduced by spec(M)=spec(M.T) and O -> O.T."""
    h, w = selector(mu, lam, target)
    return h.T, w


def check(seed=20260916, trials=200):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        lam = np.exp(1j * rng.uniform(-2.5, 2.5, 4))
        mu = np.array([1.0, 1.0, 1.0, np.exp(1j * 0.7)])
        q, _ = selector(lam, mu, np.array([])) if False else (None, None)
        x = rng.normal(size=4); x /= np.linalg.norm(x)
        v = np.eye(4)[:, 3] - x
        q = np.eye(4) if np.linalg.norm(v) < 1e-14 else np.eye(4)-2*np.outer(v,v)/np.dot(v,v)
        if np.linalg.det(q) < 0: q[:, 0] *= -1
        m = np.diag(lam) @ q @ np.diag(mu) @ q.T
        t = np.linalg.eigvals(m)
        q2, w = selector(lam, mu, t)
        m2 = np.diag(lam) @ q2 @ np.diag(mu) @ q2.T
        err = np.max(np.abs(np.sort_complex(np.linalg.eigvals(m2))-np.sort_complex(t)))
        worst = max(worst, err)
        assert err < 2e-8 and np.min(w) >= -1e-12
    print({"result":"PASS", "trials":trials, "worst_eigenvalue_error":worst})


def check_left(seed=20260917, trials=200):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        lam = np.array([1.0, 1.0, 1.0, np.exp(1j * 0.7)])
        mu = np.exp(1j * rng.uniform(-2.5, 2.5, 4))
        x = rng.normal(size=4); x /= np.linalg.norm(x)
        v = np.eye(4)[:, 3] - x
        q = np.eye(4) if np.linalg.norm(v) < 1e-14 else np.eye(4)-2*np.outer(v,v)/np.dot(v,v)
        if np.linalg.det(q) < 0: q[:, 0] *= -1
        m = np.diag(lam) @ q @ np.diag(mu) @ q.T
        t = np.linalg.eigvals(m)
        q2, w = selector_left(lam, mu, t)
        m2 = np.diag(lam) @ q2 @ np.diag(mu) @ q2.T
        err = np.max(np.abs(np.sort_complex(np.linalg.eigvals(m2))-np.sort_complex(t)))
        worst = max(worst, err)
        assert err < 2e-8 and np.min(w) >= -1e-12
    print({"result":"PASS", "left_trials":trials, "left_worst_eigenvalue_error":worst})


if __name__ == "__main__":
    check()
    check_left()
