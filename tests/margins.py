#!/usr/bin/env python3
"""Exact quantum-Horn margins of corpus rows. Requires NumPy.

Uses the QLR inequality table from generate.py. A row is feasible when some
central lift of its target satisfies every inequality; the margin is the
smallest slack over inequalities, maximized over the two lifts. Rows with a
float margin below 1e-9 are re-evaluated in exact rational arithmetic on the
binary64 inputs.

    python tests/margins.py [--rows 1 2 3] [--summary]
"""

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np

from generate import QLR

CASES = Path(__file__).with_name("cases.bin")


def _phi(rank, width, partition):
    counts = [0] * 5
    for i in range(rank):
        counts[width + i + 1 - partition[i]] += 1
    return [counts[1] - counts[4], counts[2] - counts[4], counts[3] - counts[4]]


def inequalities():
    """Rows (left, right, target, degree) with integer coefficient vectors."""
    rows = []
    for rank, width, a, b, c, degree in QLR:
        for left, right in ((a, b), (b, a)):
            rows.append(
                (
                    _phi(rank, width, left),
                    _phi(rank, width, right),
                    [-v for v in _phi(rank, width, c)],
                    int(degree),
                )
            )
            if a == b:
                break
    return rows


def float_margins(cases):
    rows = inequalities()
    ci, gi, ti, bound = (np.array([r[k] for r in rows], float) for k in range(4))
    c, g, t = cases[:, 0], cases[:, 1], cases[:, 2]
    reflected = np.column_stack([t[:, 2] + 0.5, -t.sum(axis=1) + 0.5, t[:, 0] - 0.5])
    slack = bound - c @ ci.T - g @ gi.T
    return np.maximum((slack - t @ ti.T).min(axis=1), (slack - reflected @ ti.T).min(axis=1))


def exact_margin(case):
    """Exact margin of one row, as a Fraction of the binary64 inputs."""
    c, g, t = ([Fraction(v) for v in m] for m in case)
    half = Fraction(1, 2)
    lifts = (t, [t[2] + half, -(t[0] + t[1] + t[2]) + half, t[0] - half])
    return max(
        min(
            Fraction(d)
            - sum(Fraction(a) * v for a, v in zip(left, c, strict=True))
            - sum(Fraction(b) * v for b, v in zip(right, g, strict=True))
            - sum(Fraction(k) * v for k, v in zip(target, lift, strict=True))
            for left, right, target, d in inequalities()
        )
        for lift in lifts
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="*", help="zero-based rows to print")
    parser.add_argument("--summary", action="store_true", help="corpus-wide counts")
    args = parser.parse_args()
    cases = np.fromfile(CASES, dtype="<f8").reshape(-1, 3, 3)
    margins = float_margins(cases)
    for row in args.rows or ():
        exact = exact_margin(cases[row])
        print(f"row {row}: float {margins[row]:.3e} exact {float(exact):.3e}")
    if args.summary or not args.rows:
        near = np.flatnonzero(margins < 1e-9)
        exact = {int(i): exact_margin(cases[i]) for i in near}
        violated = sorted((float(-m), i) for i, m in exact.items() if m < 0)
        print(f"rows {len(cases):,}; float margin < 1e-9: {len(near):,}")
        print(f"exactly infeasible: {len(violated):,}")
        for level in (1e-16, 1e-15, 1e-13, 1e-10):
            print(f"  violation > {level:g}: {sum(v > level for v, _ in violated)}")
        print("worst:", [(i, f"{v:.2e}") for v, i in violated[-5:][::-1]])


if __name__ == "__main__":
    main()
