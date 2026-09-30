#!/usr/bin/env python3
"""Seeded held-out rows from the constructions of tests/generate.py.

    python tests/heldout.py --seed S --out DIR   (writes DIR/test.bin, DIR/test_labels.txt, DIR/test_meta.json)

A change tuned on cases.bin can be checked on rows it has never seen: run the corpus runner
on DIR/test.bin. Use a new seed for each such check. Rows identical to a cases.bin row are
dropped.

Families:
  haar            generate.haar() construction with rng = default_rng(seed)
  grid_d{4,8,16,32}  exact dyadic c, g, t: 30% of points from generate.grid()'s 120-point set,
                  70% new dyadic points (barycentric weights with denominator 8 or 16);
                  at least one new point per row; labelled by the finest dyadic level of the row
  bnd{V,E,F,I}_<gapbin>, bndrep_<gapbin>, bndtgt_<gapbin>
                  generate.boundaries() construction with rng = default_rng(seed + 1), gap scales
                  TEST_SCALES (off the SCALES ladder: 5e-14, 3e-13, 5e-9, 3e-3, plus ladder values 0,
                  1e-12, 1e-8, 1e-4); the repeated-input block also uses a seeded generic point and
                  off-ladder local angles (0, 3e-11, 5e-7, U(0.1, 1.2)) instead of (0, 1e-12, 1e-8, 0.31).
  (grid_d4 does not occur and grid_d8 is rare (a handful of rows per seed, e.g. 5 at seed 777):
                  generate.grid()'s 120-point set covers most low-level rows, and the corpus-duplicate
                  filter removes any repeat; cases.bin covers both families in full.)
  regressions     the 24 fixed rows of generate.REGRESSIONS (identical in cases.bin: fixed data
                  cannot be re-seeded; included so every family is covered).
Gap bins: g0 = 0; tiny <= 1e-13; small <= 1e-9; mid <= 1e-5; large > 1e-5.

Feasibility: every row is checked with margins.py's quantum-Horn test (float margins, then
exact rational margins on the binary64 inputs for rows with float margin < 1e-9). Rows with
exact margin < FEAS_MIN are dropped and counted in test_meta.json.
"""
import argparse, hashlib, itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate as G
import margins as MG

TEST_SCALES = (0, 5e-14, 3e-13, 1e-12, 5e-9, 1e-8, 1e-4, 3e-3)
FEAS_MIN = -1e-15  # exact margin floor. The corpus itself has 8,103 rows with exact margin in
# [-3.9e-16, 0) (rounding of planted targets); -1e-15 admits that rounding level and nothing
# the corpus would not contain. It is 1000x below the checker's 1e-12 root allowance.
SUP = "VEFI"


def gap_bin(gap):
    if gap == 0:
        return "g0"
    if gap <= 1e-13:
        return "tiny"
    if gap <= 1e-9:
        return "small"
    if gap <= 1e-5:
        return "mid"
    return "large"


def dyadic_label(row):
    v = np.asarray(row, float).ravel()
    for k in range(0, 40):
        if np.all(v * 2.0**k == np.round(v * 2.0**k)):
            break
    return "grid_d4" if k <= 2 else "grid_d8" if k == 3 else "grid_d16" if k == 4 else "grid_d32"


def haar_rows(rng, count, batch=20_000):
    from scipy.stats import unitary_group
    rows, total = [], 0
    while total < count:
        coords = []
        for _ in range(3):
            m = unitary_group.rvs(4, size=batch, random_state=rng)
            m /= np.linalg.det(m)[:, None, None] ** 0.25
            roots = np.linalg.eigvals(m.swapaxes(-1, -2) @ m)
            coords.append(np.array([G.fold(r) for r in roots]))
        b = G.feasible(*coords)
        rows.append(b)
        total += len(b)
    return np.concatenate(rows)[:count]


def grid_rows(rng, count):
    # generate.grid()'s point list (first 120 points), rebuilt exactly as in grid().
    points = list(G.VERTICES) + [G.VERTICES.mean(axis=0)]
    den = 2
    while len(points) < 120:
        w = [(a, b, c, den - a - b - c) for a in range(den + 1) for b in range(den - a + 1)
             for c in range(den - a - b + 1) if any(v % 2 for v in (a, b, c, den - a - b - c))]
        w.sort(key=lambda x: tuple(-v for v in sorted(x)))
        points.extend(np.array(x) / den @ G.VERTICES for x in w)
        den *= 2
    old = G.monodromy(points[:120])
    old_keys = {tuple(p) for p in old}

    def new_point():
        d = int(rng.choice([8, 16]))
        cuts = np.sort(rng.integers(0, d + 1, size=3))
        w = np.diff(np.concatenate([[0], cuts, [d]])) / d
        return G.monodromy((w @ G.VERTICES)[None, :])[0]

    out = []
    while sum(len(x) for x in out) < count:
        n = 20_000
        trip = np.empty((n, 3, 3))
        isnew = rng.random((n, 3)) < 0.7
        isnew[~isnew.any(axis=1), rng.integers(0, 3)] = True
        for i in range(n):
            for j in range(3):
                trip[i, j] = new_point() if isnew[i, j] else old[rng.integers(0, 120)]
        keep = np.array([not all(tuple(trip[i, j]) in old_keys for j in range(3)) for i in range(n)])
        trip = trip[keep]
        out.append(G.feasible(trip[:, 0], trip[:, 1], trip[:, 2]))
    rows = np.concatenate(out)
    rows = np.unique(rows.reshape(-1, 9), axis=0).reshape(-1, 3, 3)
    rng.shuffle(rows)
    return rows[:count]


def frame_kinds():
    yield "identity"
    yield "dense"
    for _ in G.PAIRS:
        yield "plane1"
    for _ in range(3):
        yield "plane2"
    for _ in range(4):
        yield "plane3"


def boundary_rows(rng, scales, generic=None, angles=(0, 1e-12, 1e-8, 0.31)):
    """generate.boundaries() with labels; generic/angles default to generate.py's values."""
    supports = [s for size in range(1, 5) for s in itertools.combinations(range(4), size)]
    rows, labels = [], []
    for gap in scales:
        pts = []
        for support in supports:
            w = np.zeros(4)
            w[list(support)] = rng.dirichlet(np.ones(len(support)))
            pts.append((len(support), G.monodromy(((1 - gap) * w + gap / 4) @ G.VERTICES)))
        for (sc, c), (sg, g) in itertools.product(pts, repeat=2):
            for _kind, o in zip(frame_kinds(), G.frames(rng)):
                rows.append(G.plant(c, g, o))
                labels.append("bnd" + SUP[min(sc, sg) - 1] + "_" + gap_bin(gap))
    generic = G.monodromy([0.41, 0.23, 0.07]) if generic is None else generic
    for w, gap, angle, pair in itertools.product(
        ([0.5, 0.2, 0.2], [0.5, 0, 0], [0.2, 0.2, 0.2], [0, 0, 0]), scales, angles, G.PAIRS
    ):
        rep = G.monodromy(np.array(w) + gap * np.array([3, 2, 1]))
        for c, g in ((generic, rep), (rep, generic)):
            rows.append(G.plant(c, g, G.rotation(pair, angle)))
            labels.append("bndrep_" + gap_bin(gap))
    d = np.diag(np.exp(0.5j * np.angle(G.spectrum(generic))))
    for w, gap in itertools.product(([0.5, 0.2, 0.2], [0.5, 0, 0], [0.2, 0.2, 0.2], [0, 0, 0]), scales):
        t = G.monodromy(np.array(w) + gap * np.array([3, 2, 1]))
        for r in [G.dense_frame(rng), *(G.rotation(p, 0.31) for p in G.PAIRS)]:
            b = d.conj() @ r @ np.diag(G.spectrum(t)) @ r.T @ d.conj()
            cands = []
            for weight in (0, 1, -1, np.sqrt(2), np.pi):
                _, o = np.linalg.eigh(b.real + weight * b.imag)
                diag = o.T @ b @ o
                cands.append((np.max(abs(diag - np.diag(np.diag(diag)))), np.diag(diag), o))
            err, roots, o = min(cands, key=lambda x: x[0])
            if err > 1e-12:
                continue  # generate.py raises; the test generator skips the row instead
            g = G.fold(roots)
            order = min(G.PERMUTATIONS, key=lambda p: min(np.max(abs(G.spectrum(g) - s * roots[p])) for s in (-1, 1)))
            o = o[:, order]
            if np.linalg.det(o) < 0:
                o[:, 0] *= -1
            try:
                rows.append(G.plant(generic, g, o, G.fold(G.spectrum(t))))
            except ValueError:
                continue
            labels.append("bndtgt_" + gap_bin(gap))
    return np.array(rows), labels


def safe_plant_boundaries(rng, scales, generic, angles):
    # plant() raises on an invalid witness; retry-free: wrap plant to skip failures
    orig = G.plant
    skipped = [0]

    def plant(c, g, o, t=None):
        try:
            return orig(c, g, o, t)
        except ValueError:
            skipped[0] += 1
            return None

    G.plant = plant
    try:
        rows, labels = boundary_rows(rng, scales, generic, angles)
    finally:
        G.plant = orig
    keep = [i for i, r in enumerate(rows) if r is not None]
    rows = np.array([rows[i] for i in keep])
    return rows, [labels[i] for i in keep], skipped[0]


SCALE_BITS = 1100


def _toint(x):
    m, e = float(x).as_integer_ratio()  # x = m / e, e a power of two <= 2^1074
    return m * ((1 << SCALE_BITS) // e)


_INEQ = None


def fast_exact_margin(case):
    """Exact margins.exact_margin(case) * 2^SCALE_BITS as a Python int (binary64 inputs are
    dyadic, so every quantity is an integer at this scale). Sign/zero agree exactly with
    margins.exact_margin; divide by 2^SCALE_BITS for the value."""
    global _INEQ
    if _INEQ is None:
        _INEQ = MG.inequalities()
    c, g, t = ([_toint(v) for v in m] for m in case)
    half = 1 << (SCALE_BITS - 1)
    lifts = (t, [t[2] + half, -(t[0] + t[1] + t[2]) + half, t[0] - half])
    one = 1 << SCALE_BITS
    best = None
    for lift in lifts:
        worst = None
        for left, right, target, d in _INEQ:
            s = d * one - sum(a * v for a, v in zip(left, c)) - sum(b * v for b, v in zip(right, g)) - sum(k * v for k, v in zip(target, lift))
            worst = s if worst is None or s < worst else worst
        best = worst if best is None or worst > best else best
    return best


def exact_feasible(rows):
    from fractions import Fraction
    global FEAS_MIN_INT
    FEAS_MIN_INT = int(Fraction(FEAS_MIN) * (1 << SCALE_BITS))
    fm = MG.float_margins(rows)
    ok = fm >= FEAS_MIN
    near = np.flatnonzero(fm < 1e-9)
    for i in near:
        ok[i] = fast_exact_margin(rows[i]) >= FEAS_MIN_INT
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--haar", type=int, default=40_000)
    ap.add_argument("--grid", type=int, default=30_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng_h = np.random.default_rng(a.seed)
    rng_b = np.random.default_rng(a.seed + 1)
    rng_g = np.random.default_rng(a.seed + 2)
    parts, labs = [], []
    parts.append(np.array(G.REGRESSIONS, float)); labs += ["regressions"] * len(G.REGRESSIONS)
    h = haar_rows(rng_h, a.haar); parts.append(h); labs += ["haar"] * len(h)
    gr = grid_rows(rng_g, a.grid); parts.append(gr); labs += [dyadic_label(r) for r in gr]
    generic = G.monodromy(rng_b.dirichlet(np.ones(4)) @ G.VERTICES)
    angles = (0, 3e-11, 5e-7, float(rng_b.uniform(0.1, 1.2)))
    b, bl, skipped = safe_plant_boundaries(rng_b, TEST_SCALES, generic, angles)
    parts.append(b); labs += bl
    rows = np.concatenate(parts).reshape(-1, 3, 3)
    ok = exact_feasible(rows)
    ok &= np.isfinite(rows).all(axis=(1, 2))
    # Drop rows byte-identical to a corpus row (vertex x vertex rows at ladder scales with the
    # deterministic frames reproduce corpus rows exactly). Regressions are kept on purpose.
    corpus = open(os.path.join(HERE, "cases.bin"), "rb").read()
    seen = {corpus[i:i + 72] for i in range(0, len(corpus), 72)}
    flat = np.ascontiguousarray(rows.reshape(-1, 9).astype("<f8"))
    dup = np.array([flat[i].tobytes() in seen for i in range(len(flat))]) & (np.array(labs) != "regressions")
    ok &= ~dup
    rows = rows[ok]
    labs = [l for l, k in zip(labs, ok) if k]
    rows.astype("<f8").tofile(os.path.join(a.out, "test.bin"))
    open(os.path.join(a.out, "test_labels.txt"), "w").write("\n".join(labs) + "\n")
    fam, cnt = np.unique(labs, return_counts=True)
    src = open(__file__, "rb").read() + open(G.__file__, "rb").read() + open(MG.__file__, "rb").read()
    meta = {"seed": a.seed, "rows_total": int(len(rows)), "dropped_infeasible": int((~ok).sum() - dup.sum()), "dropped_corpus_duplicates": int(dup.sum()),
            "plant_skipped": skipped, "family_counts": {str(k): int(v) for k, v in zip(fam, cnt)},
            "scales": TEST_SCALES, "generator_sha256": hashlib.sha256(src).hexdigest()}
    json.dump(meta, open(os.path.join(a.out, "test_meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("seed", "rows_total", "dropped_infeasible", "plant_skipped")}))


if __name__ == "__main__":
    main()
