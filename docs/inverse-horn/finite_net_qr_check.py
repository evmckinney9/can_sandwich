"""Replay the six-factor SO(4) QR coverage check.

This tests only the stated Givens factorization and its numerical residual.
It is not evidence of Horn feasibility or universal spectral coverage.
"""

import json

import numpy as np


def givens(i, j, c, s):
    g = np.eye(4)
    g[i, i] = g[j, j] = c
    g[i, j] = -s
    g[j, i] = s
    return g


def factor(q):
    h = q.copy()
    factors = []
    for col in range(3):
        for row in range(3, col, -1):
            a, b = h[row - 1, col], h[row, col]
            radius = np.hypot(a, b)
            c, s = (1.0, 0.0) if radius == 0 else (a / radius, -b / radius)
            g = givens(row - 1, row, c, s)
            h = g @ h
            factors.append(g)
    d = np.diag(np.diag(h))
    # The input is special orthogonal, so the diagonal signs have product 1.
    return factors, d


rng = np.random.default_rng(20260915)
residuals = []
for _ in range(32):
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    factors, d = factor(q)
    reconstructed = np.eye(4)
    for g in factors:
        reconstructed = reconstructed @ g.T
    reconstructed = reconstructed @ d
    residuals.append(float(np.max(np.abs(reconstructed - q))))
    assert np.linalg.det(d) > 0
    assert np.max(np.abs(np.abs(np.diag(d)) - 1)) < 1e-12
    assert residuals[-1] < 1e-12

print(json.dumps({
    "status": "PASS",
    "trials": len(residuals),
    "max_reconstruction_residual": max(residuals),
    "scope": "QR/Givens factorization only; no Horn coverage claim.",
}, indent=2))
