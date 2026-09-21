"""Independent checks for multiplicative-finite-net.md.

This checks only the Lipschitz estimate used by the constructor.  It is not a
coverage test for the Horn problem and does not call can_sandwich.
"""
import numpy as np


def givens(i, j, x):
    g = np.eye(4)
    c, s = np.cos(x), np.sin(x)
    g[i, i] = g[j, j] = c
    g[i, j] = -s
    g[j, i] = s
    return g


def q_of(a):
    planes = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    q = np.eye(4)
    for (i, j), x in zip(planes, a):
        q = q @ givens(i, j, x)
    return q


def check(seed=20260916, trials=2000):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        a = rng.uniform(-np.pi, np.pi, 6)
        da = rng.uniform(-1e-3, 1e-3, 6)
        lhs = np.linalg.norm(q_of(a + da) - q_of(a), 2)
        rhs = 6.0 * np.linalg.norm(da, 2)
        assert lhs <= rhs + 2e-14
        worst = max(worst, lhs / np.linalg.norm(da, 2))
    print({"result": "PASS", "trials": trials, "max_ratio": worst})


if __name__ == "__main__":
    check()
