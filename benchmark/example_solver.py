#!/usr/bin/env python3
"""One-file submission template; replace solve() with your algorithm.

This example handles only identity inputs. It demonstrates the interface,
not a competitive realization algorithm. The benchmark owns all transport.
"""

from __future__ import annotations


def solve(c: list[float], g: list[float], t: list[float]) -> list[list[float]] | None:
    """Return a real SO(4) frame, or None when no frame was found."""
    if all(value == 0.0 for point in (c, g, t) for value in point):
        return [[float(i == j) for j in range(4)] for i in range(4)]
    return None
