import importlib.util
import numpy as np
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "additive_selector", Path(__file__).with_name("additive-generic-selector.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_diagonal_realization():
    a = np.array([3., 1., -1., -3.]); b = np.array([2., 1., -1., -2.])
    c, e, q = _mod.select(a, b, a + b, starts=8, seed=1)
    # The diagonal target has a large stabilizer; finite-start optimization
    # need only meet the requested coarse tolerance in this regression.
    assert c < 1e-3 and e < 1e-3
    assert abs(np.linalg.det(q) - 1) < 1e-12


def test_random_realization():
    rng = np.random.default_rng(7)
    a = np.sort(rng.normal(size=4))[::-1]; b = np.sort(rng.normal(size=4))[::-1]
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    if np.linalg.det(q) < 0: q[:, 0] *= -1
    g = np.linalg.eigvalsh(np.diag(a) + q @ np.diag(b) @ q.T)
    c, e, qgot = _mod.select(a, b, g, starts=8, seed=8)
    assert c < 1e-8 and e < 1e-8
    assert np.linalg.norm(qgot.T @ qgot - np.eye(4), ord=np.inf) < 1e-12


def test_trace_infeasible_rejected():
    a = np.array([3., 1., -1., -3.]); b = np.array([2., 1., -1., -2.])
    c, e, _ = _mod.select(a, b, np.array([6., 2., -2., -5.]), starts=4, seed=9)
    assert c > 1 and e > 1
