#!/usr/bin/env python3
"""Recompute spectral errors of runner witnesses at 80 significant digits.

    cargo run --release -- --corpus tests/cases.bin --witnesses frames.bin
    python tests/check80.py tests/cases.bin frames.bin [--start INDEX] [--above 1e-14]

The witness file holds 16 little-endian f64 per tested row (column-major O, NaN for a
decline); --start is the corpus index of its first row (the --case index, if one was
given). The error of O is the smallest maximum root distance between the eigenvalues of
D(c)^2 O D(g)^2 O^T and +-spec(t) over all 24 bijections, as in the corpus checker. Rows
whose binary64 (LAPACK) error exceeds --above are recomputed with mpmath, and each is printed
with both values.
"""
import argparse, itertools
import numpy as np
import mpmath as mp

mp.mp.dps = 80


def spectrum(m, exp, pi):
    x, y, z = m[0] + m[1], m[0] + m[2], m[1] + m[2]
    return [exp(1j * pi * v) for v in (x - y + z, x + y - z, -x - y - z, -x + y + z)]


def error(roots, target):
    return min(max(abs(roots[n] - s * target[p[n]]) for n in range(4))
               for p in itertools.permutations(range(4)) for s in (1, -1))


def exact(case, o):
    c, g, t = ([mp.mpf(float(v)) for v in m] for m in case)
    a, b = (mp.diag(spectrum(m, mp.exp, mp.pi)) for m in (c, g))
    frame = mp.matrix([[mp.mpf(float(v)) for v in row] for row in o])
    roots = mp.eig(a * frame * b * frame.T, left=False, right=False)
    return float(error(roots, spectrum(t, mp.exp, mp.pi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("witnesses")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--above", type=float, default=1e-14)
    a = ap.parse_args()
    frames = np.fromfile(a.witnesses, "<f8").reshape(-1, 4, 4).transpose(0, 2, 1)
    cases = np.fromfile(a.corpus, "<f8").reshape(-1, 3, 3)[a.start:a.start + len(frames)]
    assert len(cases) == len(frames), "witness file is longer than the corpus"
    found = np.isfinite(frames).all(axis=(1, 2))
    fast = np.full(len(frames), np.inf)
    for i in np.flatnonzero(found):
        c, g, t = cases[i]
        m = np.diag(spectrum(c, np.exp, np.pi)) @ frames[i] @ np.diag(spectrum(g, np.exp, np.pi)) @ frames[i].T
        fast[i] = error(np.linalg.eigvals(m), spectrum(t, np.exp, np.pi))
    worst = 0.0
    print("row binary64 80-digit")
    for i in np.flatnonzero(found & (fast > a.above)):
        e = exact(cases[i], frames[i])
        worst = max(worst, e)
        print(a.start + i, "%.3e" % fast[i], "%.3e" % e)
    print("witnesses %d; declined %d; binary64 max %.3e; recomputed %d; 80-digit max of those %.3e"
          % (len(frames), (~found).sum(), fast[found].max(initial=0.0), (found & (fast > a.above)).sum(), worst))


if __name__ == "__main__":
    main()
