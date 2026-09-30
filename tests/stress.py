#!/usr/bin/env python3
"""Stress rows of cases.bin: rows that are hard for a stated mathematical reason.

    python tests/stress.py --append     (after tests/generate.py; appends the 6,292 rows)
    python tests/stress.py --out DIR    (writes DIR/stress.bin, planted witnesses, and metadata)

Every planted row is (c, g, fold(spec(D(c)^2 O D(g)^2 O^T))) for an explicit O (generate.plant,
which also verifies the witness at 1e-12), so it is feasible up to the rounding of t. Rows are
also given their exact quantum-Horn margin (margins.py, exact rational arithmetic).

Families and why each is hard:
  nearwall_a   planted near-reducible O = expm(eps Z) O0 P (O0 in SO(3)xSO(1), SO(2)xSO(2) or I, P a
               permutation), exact Horn margin in [1e-12, 1e-9]. Target within ~eps^2 of a Horn wall:
               the spectral map is critical at reducible frames, so the witness sits in an
               O(sqrt(slack)) neighbourhood where the Jacobian is nearly rank deficient.
  nearwall_b   same construction, exact margin in [1e-14, 1e-12): closer to the wall than the
               solver's refinement target resolves linearly (sqrt(1e-14) = 1e-7 rotations).
  onwall       exact block-reducible O (margin exactly at, or rounding-level below, the wall):
               the witness set is a lower-dimensional stratum; first-order methods stall.
  nearwall_hist42  the 42 planted near-wall rows the incumbent declined in the 2026-09-27 benchmark
               (stress_declines.csv, full-precision c, g, t, non-canonical coordinates as planted;
               feasible by construction, PGT Horn slack 1e-9..1e-14).
  reptarget    generic c, target with an exactly repeated spectrum (generate.py's prescribed-target
               construction with gap 0): multiplicity-m roots are only eps^(1/m)-resolvable rootwise and
               the splitting direction of the Jacobian vanishes.
  nearreptarget  same with target gaps log-uniform in [1e-12, 1e-5]: near-multiplicity, ill-conditioned
               eigenbasis of the target.
  near31       c near a (3,1) spectrum (c = (x,x,x) + delta*u, delta log-uniform [1e-12, 1e-3]),
               g generic, Haar O: a near-triple data cluster; the rank-one selector is exact only at delta=0.
  nearident    c near the identity gate (c = sigma*u, sigma log-uniform [1e-12, 1e-3]), g generic,
               Haar O: D(c)^2 ~ scalar, so the target is within O(sigma) of spec(D(g)^2) and O is
               determined only through a 1/sigma-conditioned pinched (additive-Horn) limit.
  nearident2   both c and g near identity (sigma_c, sigma_g log-uniform [1e-10, 1e-3]): the doubly
               near-scalar stratum (row 1081606 type); every datum is clustered.
"""
import argparse, ast, json, os, sys
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate as G
import margins as MG
from heldout import fast_exact_margin, SCALE_BITS

CASES = os.path.join(HERE, "cases.bin")


def haar_o(rng):
    q, r = np.linalg.qr(rng.standard_normal((4, 4)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def generic_point(rng):
    return G.monodromy((rng.dirichlet(np.ones(4)) @ G.VERTICES)[None, :])[0]


def reducible(rng, kind):
    o0 = np.eye(4)
    if kind == 0:
        o0[:3, :3] = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    elif kind == 1:
        a, b = rng.uniform(0, 2 * np.pi, 2)
        o0[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        o0[2:, 2:] = [[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]]
    p = np.eye(4)[rng.permutation(4)]
    o = o0 @ p
    if np.linalg.det(o) < 0:
        o[:, 0] *= -1
    return o


def margin(row):
    return fast_exact_margin(row) / 2**SCALE_BITS  # exact int / 2^1100, rounded once to float


def tgap(row):
    s = G.spectrum(row[2])
    return float(min(abs(s[i] - s[j]) for i in range(4) for j in range(i + 1, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="directory for stress.bin and its metadata")
    ap.add_argument("--append", action="store_true", help="append the rows to tests/cases.bin")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--hist", default=os.path.join(HERE, "stress_declines.csv"), help="the 42 historical declines")
    ap.add_argument("--per", type=int, default=1000)
    a = ap.parse_args()
    if not (a.out or a.append):
        ap.error("give --out, --append, or both")
    rng = np.random.default_rng(a.seed)
    rows, meta = [], []

    wit = []

    def add(row, fam, param, reason, o=None):
        rows.append(np.asarray(row, float))
        wit.append(np.full((4, 4), np.nan) if o is None else np.asarray(o, float))
        meta.append({"family": fam, "param": param, "reason": reason})

    def plant(c, g, o, t=None):
        try:
            return G.plant(c, g, o, t)
        except ValueError:
            return None

    # near-wall: sample until both margin windows are filled
    want = {"nearwall_a": a.per, "nearwall_b": a.per}
    have = {k: 0 for k in want}
    tries = 0
    while any(have[k] < want[k] for k in want) and tries < 400 * a.per:
        tries += 1
        eps = 10 ** rng.uniform(-8, -4)
        z = rng.standard_normal((4, 4)); z = z - z.T
        o = expm(eps * z) @ reducible(rng, int(rng.integers(3)))
        r = plant(generic_point(rng), generic_point(rng), o)
        if r is None:
            continue
        m = margin(r)
        fam = "nearwall_a" if 1e-12 <= m <= 1e-9 else "nearwall_b" if 1e-14 <= m < 1e-12 else None
        if fam and have[fam] < want[fam]:
            have[fam] += 1
            add(r, fam, eps, "planted near-reducible frame; Horn margin %.1e" % m, o)
    # on the wall: exact reducible frames
    n = 0
    while n < a.per // 2:
        o = reducible(rng, int(rng.integers(3)))
        r = plant(generic_point(rng), generic_point(rng), o)
        if r is not None:
            add(r, "onwall", 0.0, "exact block-reducible frame (target on a Horn wall)", o); n += 1
    # historical declines
    if a.hist:
        import pandas as pd
        h = pd.read_csv(a.hist)
        for _, x in h.iterrows():
            add([ast.literal_eval(x["c"]), ast.literal_eval(x["g"]), ast.literal_eval(x["t"])], "nearwall_hist42",
                int(x["row"]), "incumbent decline 2026-09-27 (planted near-wall, PGT slack %.1e)" % x["slack_p"])
    # repeated / near-repeated targets: generate.py's prescribed-target construction
    ws = ([0.5, 0.2, 0.2], [0.5, 0, 0], [0.2, 0.2, 0.2], [0, 0, 0], [0.25, 0.25, 0.25], [0.3, 0.3, 0.1])
    for fam, count in (("reptarget", a.per // 2), ("nearreptarget", a.per)):
        n = 0
        while n < count:
            generic = generic_point(rng)
            gap = 0.0 if fam == "reptarget" else 10 ** rng.uniform(-12, -5)
            t = G.monodromy(np.array(ws[rng.integers(len(ws))]) + gap * rng.uniform(0.5, 3, 3))
            d = np.diag(np.exp(0.5j * np.angle(G.spectrum(generic))))
            rr = haar_o(rng)
            b = d.conj() @ rr @ np.diag(G.spectrum(t)) @ rr.T @ d.conj()
            cands = []
            for weight in (0, 1, -1, np.sqrt(2), np.pi):
                _, o = np.linalg.eigh(b.real + weight * b.imag)
                diag = o.T @ b @ o
                cands.append((np.max(abs(diag - np.diag(np.diag(diag)))), np.diag(diag), o))
            err, roots, o = min(cands, key=lambda x: x[0])
            if err > 1e-12:
                continue
            g = G.fold(roots)
            order = min(G.PERMUTATIONS, key=lambda p: min(np.max(abs(G.spectrum(g) - s * roots[p])) for s in (-1, 1)))
            o = o[:, order]
            if np.linalg.det(o) < 0:
                o[:, 0] *= -1
            r = plant(generic, g, o, G.fold(G.spectrum(t)))
            if r is None:
                continue
            add(r, fam, gap, "target spectrum with multiplicity (gap %.1e)" % gap, o); n += 1
    # clustered data
    n = 0
    while n < a.per:
        x = rng.uniform(0.02, 0.24)
        delta = 10 ** rng.uniform(-12, -3)
        c = np.array([x, x, x]) + delta * rng.uniform(-1, 1, 3)
        c = np.sort(c)[::-1]
        o = haar_o(rng)
        r = plant(c, generic_point(rng), o)
        if r is not None:
            add(r, "near31", delta, "c within %.1e of a (3,1) spectrum" % delta, o); n += 1
    n = 0
    while n < a.per:
        sigma = 10 ** rng.uniform(-12, -3)
        c = G.monodromy((((1 - sigma) * np.array([1.0, 0, 0, 0]) + sigma * rng.dirichlet(np.ones(4))) @ G.VERTICES)[None, :])[0]
        o = haar_o(rng)
        r = plant(c, generic_point(rng), o)
        if r is not None:
            add(r, "nearident", sigma, "c within %.1e of the identity gate" % sigma, o); n += 1
    n = 0
    while n < a.per // 4:
        sc, sg = 10 ** rng.uniform(-10, -3, 2)
        mk = lambda s: G.monodromy((((1 - s) * np.array([1.0, 0, 0, 0]) + s * rng.dirichlet(np.ones(4))) @ G.VERTICES)[None, :])[0]
        o = haar_o(rng)
        r = plant(mk(sc), mk(sg), o)
        if r is not None:
            add(r, "nearident2", float(min(sc, sg)), "c and g within %.1e / %.1e of identity" % (sc, sg), o); n += 1

    rows = np.array(rows).reshape(-1, 3, 3)
    for i, m in enumerate(meta):
        canonical = m["family"] != "nearwall_hist42"
        m["exact_margin"] = margin(rows[i]) if canonical else None
        m["target_min_gap"] = tgap(rows[i])
        m["row"] = i
    data = rows.astype("<f8").tobytes()
    if a.append:
        with open(CASES, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - len(data), 0))
            if f.read() == data:
                sys.exit("tests/cases.bin already ends with the stress rows")
        with open(CASES, "ab") as f:
            f.write(data)
    if not a.out:
        return
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "stress.bin"), "wb").write(data)
    # planted witnesses (row-major 16 f64, NaN = none) for the passability check; not a solver input
    np.array(wit).reshape(-1, 16).astype("<f8").tofile(os.path.join(a.out, "stress_planted_witness.bin"))
    open(os.path.join(a.out, "stress_labels.txt"), "w").write("\n".join(m["family"] for m in meta) + "\n")
    import pandas as pd
    pd.DataFrame(meta)[["row", "family", "param", "exact_margin", "target_min_gap", "reason"]].to_csv(
        os.path.join(a.out, "stress_meta.csv"), index=False)
    fam, cnt = np.unique([m["family"] for m in meta], return_counts=True)
    print(json.dumps({"rows": len(rows), "families": dict(zip(map(str, fam), map(int, cnt))), "nearwall_tries": tries}))


if __name__ == "__main__":
    main()
