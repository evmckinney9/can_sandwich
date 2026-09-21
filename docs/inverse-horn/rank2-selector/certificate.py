"""R392 certificate: rank-two intermediate spectrum selected by sign change.

Checks, in order:
  C1  combinatorial: every unfolded face-miss inequality is a Pieri (vertical
      strip) Horn triple, for n=3..6, all k and all max-attaining patterns.
  C2  seeded samples: feasible triples (from witnesses) touch every upper root
      face, satisfy the disjoint-support vertex criterion, and reconstruct.
  C3  seeded interlacing-feasible random chains: criterion <=> hive LP, and each
      infeasible case exhibits a violated unfolded Pieri inequality.
Numerical parts support the proof; they are not the proof. Exit code 1 on any failure.
"""
import itertools, json, sys
import numpy as np
from scipy.optimize import linprog

OUT = {}
FAIL = []


def check(name, cond, info=None):
    OUT[name] = bool(cond) if info is None else {"ok": bool(cond), **info}
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- C1: Pieri unfolding
def partition_of(I):
    I = sorted(I)
    return [I[s] - (s + 1) for s in range(len(I))][::-1]  # Fulton: (i_r - r, ..., i_1 - 1)


def vertical_strip(iota, kappa):
    return all(0 <= k - i <= 1 for i, k in zip(iota, kappa)) and all(
        kappa[s] >= kappa[s + 1] for s in range(len(kappa) - 1))


def unfold(n, k, choice):
    """choice[j] for j in 1..n-1, j!=k: 'L' (lambda_j), 'N' (nu_{j+1}), or 'D' (nu_1-d2, j=1 only).
    Returns (I, K, mu_pattern) of the implied violated inequality, per attempt.md section 3."""
    S = {j for j, c in choice.items() if c in ("N", "D")}
    e1 = choice.get(1) == "D"
    if k >= 2 and (k - 1) in S and choice[k - 1] == "N":  # case B: use u_k <= lambda_{k-1}
        S.discard(k - 1)
    I = {k} | S
    K = {k} | {j + 1 for j in S if choice[j] == "N"} | ({1} if e1 else set())
    r = len(I)
    if len(K) != r or max(K) > n:
        return None
    return sorted(I), sorted(K), ("d1+d2" if e1 else "d1"), r


c1_total = 0
c1_bad = []
for n in range(3, 7):
    for k in range(1, n + 1):
        js = [j for j in range(1, n) if j != k]
        opts = [("L", "N", "D") if j == 1 else ("L", "N") for j in js]
        for combo in itertools.product(*opts):
            choice = dict(zip(js, combo))
            res = unfold(n, k, choice)
            c1_total += 1
            if res is None:
                c1_bad.append((n, k, combo, "malformed"))
                continue
            I, K, mu, r = res
            iota, kappa = partition_of(I), partition_of(K)
            size = r - 1 if mu == "d1" else r - 2
            ok = vertical_strip(iota, kappa) and (sum(kappa) - sum(iota) == size)
            if not ok:
                c1_bad.append((n, k, combo, I, K, mu))
check("C1_pieri_vertical_strip_all_patterns", not c1_bad, {"patterns": c1_total, "bad": c1_bad[:5]})


# ---------------------------------------------------------------- shared numerics
def pe(roots, x):
    return float(np.prod([x - r for r in roots]))


def dpe(roots, x):
    return float(sum(np.prod([x - r for kk, r in enumerate(roots) if kk != j]) for j in range(len(roots))))


def intervals(lam, d1, d2, nu):
    n = len(lam)
    lo, hi = np.empty(n), np.empty(n)
    for j in range(n):
        l1, u1 = lam[j], (lam[j - 1] if j > 0 else lam[0] + d1)
        if j == 0:
            l2, u2 = max(nu[1], nu[0] - d2), nu[0]
        elif j < n - 1:
            l2, u2 = nu[j + 1], nu[j]
        else:
            l2, u2 = -np.inf, nu[n - 1]
        lo[j], hi[j] = max(l1, l2), min(u1, u2)
    return lo, hi


def vertices(lo, hi, T):
    n = len(lo)
    out = []
    for k in range(n):
        others = [j for j in range(n) if j != k]
        for pins in itertools.product([0, 1], repeat=n - 1):
            rho = np.empty(n)
            for j, p in zip(others, pins):
                rho[j] = hi[j] if p else lo[j]
            rho[k] = T - rho[others].sum()
            if lo[k] - 1e-12 <= rho[k] <= hi[k] + 1e-12:
                out.append(rho)
    return out


def mvec(lam, nu, rho):
    return np.array([np.sqrt(abs(pe(lam, r) * pe(nu, r))) / abs(dpe(rho, r)) for r in rho])


def hive_feasible(lam, d1, d2, nu):
    n = len(lam)
    T = lam.sum() + d1
    lo, hi = intervals(lam, d1, d2, nu)
    if np.any(lo > hi + 1e-12):
        return False
    A_ub, b_ub = [], []
    for j in range(1, n):  # lattice word at 1-based row j+1: sum_{j'<=j}(rho-lam) >= sum_{j'<=j+1}(nu-rho)
        row = np.zeros(n); row[:j] -= 1.0; row[:j + 1] -= 1.0
        A_ub.append(row); b_ub.append(-(lam[:j].sum()) - nu[:j + 1].sum())
    res = linprog(np.zeros(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array([np.ones(n), np.eye(n)[0]]), b_eq=[T, nu[0]],
                  bounds=list(zip(lo, hi)), method="highs")
    return res.status == 0


def reconstruct(lam, d1, d2, nu, rho, eps):
    """Gift Alg. 4.8 steps 3-4; U_rho from Lemma 4.6 in closed form."""
    n = len(lam)
    v = np.array([np.sqrt(max(0.0, -pe(rho, lam[i]) / (d1 * dpe(lam, lam[i])))) for i in range(n)])
    U = np.zeros((n, n))
    for j in range(n):
        hit = [i for i in range(n) if abs(lam[i] - rho[j]) < 1e-11 * (1 + abs(lam[i]))]
        if hit:
            U[hit[0], j] = 1.0
            continue
        s = np.sqrt(max(0.0, d1 * pe(lam, rho[j]) / dpe(rho, rho[j])))
        for i in range(n):
            U[i, j] = v[i] / (lam[i] - rho[j]) * s
    z = np.array([eps[j] * np.sqrt(max(0.0, -pe(nu, rho[j]) / (d2 * dpe(rho, rho[j])))) for j in range(n)])
    return v, U @ z


def bisect(lam, nu, ra, rb, eps, iters=200):
    Fa, Fb = eps @ mvec(lam, nu, ra), eps @ mvec(lam, nu, rb)
    assert Fa > 0 > Fb, (Fa, Fb)
    for _ in range(iters):
        rm = 0.5 * (ra + rb)
        Fm = eps @ mvec(lam, nu, rm)
        if Fm == 0:
            return rm
        if Fm > 0:
            ra = rm
        else:
            rb = rm
    return 0.5 * (ra + rb)


def criterion(lam, d1, d2, nu):
    """(ra, rb, eps) with F_eps(ra)>0>F_eps(rb); (ra, None, ones) for an empty-support vertex; None."""
    n = len(lam)
    T = lam.sum() + d1
    lo, hi = intervals(lam, d1, d2, nu)
    if np.any(lo > hi + 1e-12):
        return None
    scale = max(1.0, np.abs(lam).max(), np.abs(nu).max())
    S = []
    for rho in vertices(lo, hi, T):
        m = mvec(lam, nu, rho)
        S.append((frozenset(np.nonzero(m > 1e-7 * scale)[0].tolist()), rho))
    for s, rho in S:
        if not s:
            return rho, None, np.ones(n)
    for sa, ra in S:
        for sb, rb in S:
            if sa and sb and not (sa & sb):
                eps = np.ones(n); eps[list(sb)] = -1.0
                return ra, rb, eps
    return None


def face_touch_all(lam, d1, d2, nu):
    """T >= u_k + sum_{j!=k} l_j for every k (P touches every upper face)."""
    lo, hi = intervals(lam, d1, d2, nu)
    T = lam.sum() + d1
    return all(T >= hi[k] + lo.sum() - lo[k] - 1e-9 for k in range(len(lam)))


def violated_pieri(lam, d1, d2, nu):
    """For an infeasible triple with P nonempty: find k with a missed upper face and return the
    unfolded inequality (I,K,mu) and its numerical slack (must be > 0 = violated)."""
    n = len(lam)
    if nu[0] > lam[0] + d1 + 1e-9:  # Weyl, the r=1 Pieri triple ({1},{1},{1}); u_1 is then not a root
        return dict(k=1, I=[1], K=[1], mu="d1", violation=float(nu[0] - lam[0] - d1))
    lo, hi = intervals(lam, d1, d2, nu)
    T = lam.sum() + d1
    for k in range(n):
        if T < hi[k] + lo.sum() - lo[k] - 1e-9:
            choice = {}
            for j in range(1, n):
                if j == k + 1:
                    continue
                idx = j - 1
                cands = {"L": lam[idx], "N": nu[j] if j < n else -np.inf}
                if j == 1:
                    cands["D"] = nu[0] - d2
                choice[j] = max(cands, key=cands.get)
            res = unfold(n, k + 1, choice)
            if res is None:
                return None
            I, K, mu, r = res
            lhs = sum(nu[kk - 1] for kk in K)
            rhs = sum(lam[ii - 1] for ii in I) + d1 + (d2 if mu == "d1+d2" else 0.0)
            return dict(k=k + 1, I=I, K=K, mu=mu, violation=float(lhs - rhs))
    return None


def witness_error(lam, d1, d2, nu, v, w):
    B = d1 * np.outer(v, v) + d2 * np.outer(w, w)
    nuh = np.sort(np.linalg.eigvalsh(np.diag(lam) + B))[::-1]
    bs = np.sort(np.linalg.eigvalsh(B))[::-1]
    return max(np.abs(nuh - nu).max(), abs(bs[0] - d1), abs(bs[1] - d2), np.abs(bs[2:]).max(),
               abs(v @ w), abs(v @ v - 1), abs(w @ w - 1))


rng = np.random.default_rng(20260915)


def sample_feasible(n):
    lam = np.sort(rng.normal(size=n))[::-1] * 2
    d1, d2 = np.sort(rng.uniform(0.2, 3.0, size=2))[::-1]
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    B = d1 * np.outer(Q[:, 0], Q[:, 0]) + d2 * np.outer(Q[:, 1], Q[:, 1])
    return lam, d1, d2, np.sort(np.linalg.eigvalsh(np.diag(lam) + B))[::-1]


def sample_chain(n):
    while True:
        lam = np.sort(rng.normal(size=n))[::-1] * 2
        d1, d2 = np.sort(rng.uniform(0.2, 3.0, size=2))[::-1]
        v = rng.normal(size=n); v /= np.linalg.norm(v)
        rho = np.sort(np.linalg.eigvalsh(np.diag(lam) + d1 * np.outer(v, v)))[::-1]
        w = rng.normal(size=n); w /= np.linalg.norm(w)
        nu = np.sort(np.linalg.eigvalsh(np.diag(rho) + d2 * np.outer(w, w)))[::-1]
        if np.min(np.abs(lam[:, None] - nu[None, :])) > 1e-6:
            return lam, d1, d2, nu


# ---------------------------------------------------------------- C2, C3
for n in range(3, 7):
    N = 120
    touch = crit = recon = 0; worst = 0.0
    for _ in range(N):
        lam, d1, d2, nu = sample_feasible(n)
        touch += face_touch_all(lam, d1, d2, nu)
        c = criterion(lam, d1, d2, nu)
        if c is None:
            continue
        crit += 1
        ra, rb, eps = c
        rho = ra if rb is None else bisect(lam, nu, ra, rb, eps)
        v, w = reconstruct(lam, d1, d2, nu, rho, eps)
        err = witness_error(lam, d1, d2, nu, v, w)
        worst = max(worst, err); recon += err < 1e-8
    check(f"C2_n{n}_feasible_touch_all_faces", touch == N, {"count": touch, "N": N})
    check(f"C2_n{n}_feasible_criterion", crit == N, {"count": crit, "N": N})
    check(f"C2_n{n}_reconstruction_1e-8", recon == N, {"count": recon, "N": N, "worst_err": worst})

    M = 120
    tab = {"feas_crit": 0, "feas_nocrit": 0, "infeas_crit": 0, "infeas_nocrit": 0}
    pieri_found = pieri_needed = 0
    for _ in range(M):
        lam, d1, d2, nu = sample_chain(n)
        hf = hive_feasible(lam, d1, d2, nu)
        c = criterion(lam, d1, d2, nu) is not None
        tab[("feas" if hf else "infeas") + ("_crit" if c else "_nocrit")] += 1
        if not hf:
            pieri_needed += 1
            vp = violated_pieri(lam, d1, d2, nu)
            pieri_found += (vp is not None and vp["violation"] > 1e-9)
    check(f"C3_n{n}_criterion_iff_hiveLP", tab["feas_nocrit"] == 0 and tab["infeas_crit"] == 0, tab)
    check(f"C3_n{n}_infeasible_has_violated_pieri", pieri_found == pieri_needed,
          {"found": pieri_found, "needed": pieri_needed})

OUT["result"] = "PASS" if not FAIL else "FAIL"
OUT["failures"] = FAIL
print(json.dumps(OUT, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
sys.exit(1 if FAIL else 0)
