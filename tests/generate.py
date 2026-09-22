#!/usr/bin/env python3
"""Generate grid, Haar, and witnessed boundary cases. Requires NumPy and SciPy."""

import itertools
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]])
PAIRS = tuple(itertools.combinations(range(4), 2))
PERMUTATIONS = np.array(list(itertools.permutations(range(4))))
SCALES = (0, 2e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-4, 1e-2)

# Quantum Littlewood--Richardson products: (rank, width, left, right, target, degree).
# fmt: off
QLR = [
    (1, 3, [0], [0], [0], 0.0),
    (1, 3, [0], [1], [1], 0.0),
    (1, 3, [0], [2], [2], 0.0),
    (1, 3, [0], [3], [3], 0.0),
    (1, 3, [1], [1], [2], 0.0),
    (1, 3, [1], [2], [3], 0.0),
    (1, 3, [1], [3], [0], 1.0),
    (1, 3, [2], [2], [0], 1.0),
    (1, 3, [2], [3], [1], 1.0),
    (1, 3, [3], [3], [2], 1.0),
    (2, 2, [0, 0], [0, 0], [0, 0], 0.0),
    (2, 2, [0, 0], [1, 0], [1, 0], 0.0),
    (2, 2, [0, 0], [1, 1], [1, 1], 0.0),
    (2, 2, [0, 0], [2, 0], [2, 0], 0.0),
    (2, 2, [0, 0], [2, 1], [2, 1], 0.0),
    (2, 2, [0, 0], [2, 2], [2, 2], 0.0),
    (2, 2, [1, 0], [1, 0], [1, 1], 0.0),
    (2, 2, [1, 0], [1, 0], [2, 0], 0.0),
    (2, 2, [1, 0], [1, 1], [2, 1], 0.0),
    (2, 2, [1, 0], [2, 0], [2, 1], 0.0),
    (2, 2, [1, 0], [2, 1], [2, 2], 0.0),
    (2, 2, [1, 0], [2, 1], [0, 0], 1.0),
    (2, 2, [1, 0], [2, 2], [1, 0], 1.0),
    (2, 2, [1, 1], [1, 1], [2, 2], 0.0),
    (2, 2, [1, 1], [2, 0], [0, 0], 1.0),
    (2, 2, [1, 1], [2, 1], [1, 0], 1.0),
    (2, 2, [1, 1], [2, 2], [2, 0], 1.0),
    (2, 2, [2, 0], [2, 0], [2, 2], 0.0),
    (2, 2, [2, 0], [2, 1], [1, 0], 1.0),
    (2, 2, [2, 0], [2, 2], [1, 1], 1.0),
    (2, 2, [2, 1], [2, 1], [2, 0], 1.0),
    (2, 2, [2, 1], [2, 1], [1, 1], 1.0),
    (2, 2, [2, 1], [2, 2], [2, 1], 1.0),
    (2, 2, [2, 2], [2, 2], [0, 0], 2.0),
    (3, 1, [0, 0, 0], [0, 0, 0], [0, 0, 0], 0.0),
    (3, 1, [0, 0, 0], [1, 0, 0], [1, 0, 0], 0.0),
    (3, 1, [0, 0, 0], [1, 1, 0], [1, 1, 0], 0.0),
    (3, 1, [0, 0, 0], [1, 1, 1], [1, 1, 1], 0.0),
    (3, 1, [1, 0, 0], [1, 0, 0], [1, 1, 0], 0.0),
    (3, 1, [1, 0, 0], [1, 1, 0], [1, 1, 1], 0.0),
    (3, 1, [1, 0, 0], [1, 1, 1], [0, 0, 0], 1.0),
    (3, 1, [1, 1, 0], [1, 1, 0], [0, 0, 0], 1.0),
    (3, 1, [1, 1, 0], [1, 1, 1], [1, 0, 0], 1.0),
    (3, 1, [1, 1, 1], [1, 1, 1], [1, 1, 0], 1.0),
]

# Fixed regressions for clustered spectra and near-identity gates.
REGRESSIONS = [
    [[0.25156717590725, 0.24843282409275, 0.24690239120876], [0.03363566241836, 0.02691404826411, 0.02691404826411], [0.27897727016045, 0.27741771973249, 0.16056554217426]],  # near-swap repeated-gate witness
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.29546099931489134, 0.20453900018510873, -0.20453900061744035]],  # exact-double target 4101
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.484847219872809, 0.015152779130006122, -0.015152779314408282]],  # exact-double target 4102
    [[0.2555803919905, 0.25558039198104, -0.2555803919405], [0.2555803919905, 0.25558039198104, -0.2555803919405], [0.2444196080691815, 0.07613277864424217, -0.0761327790245726]],  # near-pair target 4660
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.47976171336231027, 3.350705823912392e-10, -4.450235180364359e-10]],  # quantized face target 6363
    [[0.22382655216654007, 4.5365999717547037e-10, -3.6097999717547044e-10], [0.37673542185379, 0.0, 0.0], [0.39943802588698385, 4.5366116106166384e-10, -3.609834684704048e-10]],  # clustered proxy certificate 4673
    [[4.920400000000001e-09, 1.38676e-09, -1.22755e-09], [0.5, 0.30909146235419005, -0.30909146235419005], [0.19090853874762126, 3.421808059123066e-09, -3.4846467933391523e-09]],  # near-identity edge target 8497
    [[0.49999999751607005, 0.499999996086, -0.4999999939139901], [0.45073929780807004, 0.21470014625631004, -0.11617874287243998], [0.38382125933803596, -0.04926070113759151, -0.04926070436221991]],  # near-swap edge target 9082
    [[0.49999999924211, 0.49999999906905, -0.49999999793094996], [0.5, 0.0, 0.0], [0.4999999991841258, 3.225370526216409e-10, -7.092456066892794e-10]],  # repeated-root multiplicity 9380
    [[0.49999999927363004, 0.4999999989163201, -0.49999999808368], [0.49999999995969, 9.689980276617723e-12, 3.390019723382273e-12], [0.4999999992595101, 3.593121356004758e-10, -5.216825860188123e-10]],  # repeated-root multiplicity 9393
    [[1.4011299999999997e-09, 1.34878e-09, -1.15103e-09], [0.49999999999956, 5.999978085130805e-14, -9.999780851308059e-15], [0.4999999992117509, 5.635630504262929e-10, -4.81460850321273e-10]],  # near-identity vertex target 8639
    [[0.5, 0.0, 0.0], [0.33556062825582006, 0.16437603912132998, 0.16437603911132997], [0.1643760391213343, 0.16437603911133436, -0.164312706488488]],  # public alternate witness 9482
    [[0.33556062825582006, 0.16437603912132998, 0.16437603911132997], [0.33556062825582006, 0.16437603912132998, 0.16437603911132997], [0.1643760391213343, 0.16437603911133436, -0.164312706488488]],  # repeated-input caustic 9482
    [[0.22275108585431175, 0.034783033435165096, -0.034783033434665094], [3.940164785237606e-05, 3.719643390931126e-05, -1.5999729614063347e-05], [0.465164848419121, 0.27726681794701724, -0.2772149763792984]],  # clustered spectrum
    [[0.2500000203855839, 0.24999997790254228, 0.24999997209745775], [3.162039039450218e-05, 2.875089443661313e-05, 8.008324774382537e-06], [0.25002876924039463, 0.25000799499260795, 0.24993161651888982]],  # clustered spectrum
    [[0.2500000125082198, 0.24999999305716747, 0.24999995694283256], [0.2500061412329326, 0.2500036050596355, 0.24994639494036455], [4.387932868452071e-05, 6.155433609517091e-06, 3.6078516658932802e-06]],  # clustered spectrum
    [[0.46986120830703, 0.46986120819175, -0.40958362757526], [0.24787741830247, 0.24762903238808998, 0.24737096761191002], [0.226980135397151, 0.21769777836797044, 0.2174181051570272]],  # clustered spectrum
    [[0.03125, 0.03125, -0.03125], [0.34375, -0.09375, -0.09375], [0.375, -0.125, -0.125]],  # dyadic vertex
    [[0.07630185155126362, 0.07630185155126362, 0.07630185155126362], [4.8535488656452666e-08, 8.22735189781043e-09, -5.298329210715759e-09], [0.07630190001380294, 0.07630185003208301, 0.0763018035305153]],  # near-scalar triple collision
    [[0.05, 0.02, -0.01], [0.1, 0.1, 0.1], [0.15, 0.12, 0.09]],  # triple root with identity witness
    [[0.3203878570275, 0.26514617218837, -0.06600635979049], [0.29362951834432, 0.23674475538673, -0.1514932153219], [0.45501976862371, 0.04946836621953, 0.02972299753699]],  # dense rational counterexample
    [[0.42350331556457, 0.2906704187091, -0.13767704983824], [3.8e-13, 1.5e-13, 1e-13], [0.42350331556427, 0.29067041870924, -0.13767704983822]],  # near identity gate roundoff
    [[0.2060568634486984, 0.19094682530673945, 0.19094682530673945], [0.5, 0.0, 0.0], [0.19094682530673945, 0.19094682530673945, -0.08795051406217724]],  # paired boundary
    [[0.21020884189861286, 0.19865945585858416, -0.19865945585858416], [0.25616112958989956, 0.2561611291642858, 0.23151660250714257], [0.45471293355896775, 0.04838484696533113, 0.03572592036113248]],  # external sparse witness
    [[0.4698612083070275, 0.46986120819175425, -0.4095836275752627], [0.24787741830246718, 0.2476290323880901, 0.2473709676119099], [0.226980135397151, 0.21769777836797044, 0.2174181051570272]],  # external sparse witness
    [[0.46590084118024, 0.24851454111699, -0.24623936783874], [0.39153582986042, 0.21148216642904, -0.18074076641], [0.30480818031164, 0.27052704777688, -0.04410941746934]],  # external sparse witness
    [[0.17, 0.04, -0.09], [0, 0, 0], [0.17, 0.04, -0.09]],  # identity right
    [[0, 0, 0], [0.17, 0.04, -0.09], [0.17, 0.04, -0.09]],  # identity left
]
# fmt: on


def monodromy(w):
    a, b, c = np.asarray(w).T
    return np.array([(a + b - c) / 2, (a - b + c) / 2, (-a + b + c) / 2]).T


def spectrum(m):
    x, y, z = m[0] + m[1], m[0] + m[2], m[1] + m[2]
    return np.exp(1j * np.pi * np.array([x - y + z, x + y - z, -x - y - z, -x + y + z]))


def fold(roots):
    # Lift the four phases to sum zero, then fold their pair sums into the chamber.
    turns = np.sort(np.angle(roots) / (2 * np.pi))[::-1].copy()
    winding = int(round(turns.sum()))
    if abs(turns.sum() - winding) > 2e-13:
        raise ValueError("spectrum determinant is not one")
    if winding > 0:
        turns[:winding] -= 1
    elif winding < 0:
        turns[winding:] += 1
    a, b, c, _ = np.sort(turns)[::-1]
    x, y, z = a + b, a + c, b + c
    return monodromy([1 - x, y, -z] if z < 0 else [x, y, z])


def rotation(pair, angle):
    i, j = pair
    o = np.eye(4)
    o[i, i] = o[j, j] = np.cos(angle)
    o[i, j], o[j, i] = -np.sin(angle), np.sin(angle)
    return o


def dense_frame(rng):
    q, r = np.linalg.qr(rng.normal(size=(4, 4)))
    q *= np.sign(np.diag(r))
    q[:, 0] *= np.linalg.det(q)
    return q


def plant(c, g, o, t=None):
    product = np.diag(spectrum(c)) @ o @ np.diag(spectrum(g)) @ o.T
    actual = np.linalg.eigvals(product)
    if t is None:
        t = fold(actual)
    error = min(
        np.min(np.max(abs(actual - sign * spectrum(t)[PERMUTATIONS]), axis=1))
        for sign in (-1, 1)
    )
    if (
        error > 1e-12
        or np.max(abs(o.T @ o - np.eye(4))) > 1e-12
        or abs(np.linalg.det(o) - 1) > 1e-12
    ):
        raise ValueError("invalid planted witness")
    return np.array([c, g, t])


def qlr_blocks():
    def phi(r, k, p):
        g = np.zeros(5)
        for i in range(r):
            g[k + i + 1 - p[i]] += 1
        return g[1:4] - g[4]

    rows = []
    for r, k, a, b, c, d in QLR:
        for left, right in ((a, b), (b, a)):
            rows.append((phi(r, k, left), phi(r, k, right), -phi(r, k, c), d))
            if a == b:
                break
    return tuple(np.array([row[i] for row in rows]) for i in range(4))


CI, GI, TI, BOUND = qlr_blocks()


def feasible(c, g, t):
    reflected = np.column_stack([t[:, 2] + 0.5, -t.sum(axis=1) + 0.5, t[:, 0] - 0.5])
    bound = BOUND - c @ CI.T - g @ GI.T
    plain = (bound - t @ TI.T).min(axis=1) >= -1e-7
    reflected_ok = (bound - reflected @ TI.T).min(axis=1) >= -1e-7
    keep = plain | reflected_ok
    return np.stack(
        [c[keep], g[keep], np.where(plain[:, None], t, reflected)[keep]], axis=1
    )


def grid(count=120):
    points = list(VERTICES) + [VERTICES.mean(axis=0)]
    denominator = 2
    while len(points) < count:
        weights = [
            (a, b, c, denominator - a - b - c)
            for a in range(denominator + 1)
            for b in range(denominator - a + 1)
            for c in range(denominator - a - b + 1)
            if any(v % 2 for v in (a, b, c, denominator - a - b - c))
        ]
        weights.sort(key=lambda w: tuple(-v for v in sorted(w)))
        points.extend(np.array(w) / denominator @ VERTICES for w in weights)
        denominator *= 2
    m = monodromy(points[:count])
    gates = m[np.any(m != 0, axis=1)]
    g = np.repeat(gates, len(m), axis=0)
    t = np.tile(m, (len(gates), 1))
    rows = np.concatenate([feasible(np.broadcast_to(c, g.shape), g, t) for c in m])
    return np.unique(rows.reshape(-1, 9), axis=0).reshape(-1, 3, 3)


def haar(count=300_000):
    rng = np.random.default_rng(0)
    rows, total = [], 0
    while total < count:
        coordinates = []
        for _ in range(3):
            matrices = unitary_group.rvs(4, size=50_000, random_state=rng)
            # Haar sampling is basis invariant; only remove the global phase.
            matrices /= np.linalg.det(matrices)[:, None, None] ** 0.25
            roots = np.linalg.eigvals(matrices.swapaxes(-1, -2) @ matrices)
            coordinates.append(np.array([fold(root) for root in roots]))
        batch = feasible(*coordinates)
        rows.append(batch)
        total += len(batch)
    return np.concatenate(rows)[:count]


def frames(rng):
    yield np.eye(4)
    yield dense_frame(rng)
    for pair in PAIRS:
        yield rotation(pair, 0.31)
    for i, j, k, l in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        yield rotation((i, j), 0.17) @ rotation((k, l), 0.39)
    for fixed in range(4):
        o = np.eye(4)
        for pair in PAIRS:
            if fixed not in pair:
                o = o @ rotation(pair, 0.31)
        yield o


def boundaries():
    rng = np.random.default_rng(12648430)
    supports = [
        s for size in range(1, 5) for s in itertools.combinations(range(4), size)
    ]
    rows = []
    # Cross every vertex, edge, face, and interior input family at each gap scale.
    for gap in SCALES:
        points = []
        for support in supports:
            weights = np.zeros(4)
            weights[list(support)] = rng.dirichlet(np.ones(len(support)))
            points.append(monodromy(((1 - gap) * weights + gap / 4) @ VERTICES))
        for c, g in itertools.product(points, repeat=2):
            for o in frames(rng):
                rows.append(plant(c, g, o))
    # Input gap and local angle vary independently, in both factor positions.
    generic = monodromy([0.41, 0.23, 0.07])
    for w, gap, angle, pair in itertools.product(
        ([0.5, 0.2, 0.2], [0.5, 0, 0], [0.2, 0.2, 0.2], [0, 0, 0]),
        SCALES,
        (0, 1e-12, 1e-8, 0.31),
        PAIRS,
    ):
        repeated = monodromy(np.array(w) + gap * np.array([3, 2, 1]))
        for c, g in ((generic, repeated), (repeated, generic)):
            rows.append(plant(c, g, rotation(pair, angle)))
    # Prescribe repeated/near-repeated targets: D^-1 R T R^T D^-1 is symmetric unitary.
    d = np.diag(np.exp(0.5j * np.angle(spectrum(generic))))
    for w, gap in itertools.product(
        ([0.5, 0.2, 0.2], [0.5, 0, 0], [0.2, 0.2, 0.2], [0, 0, 0]), SCALES
    ):
        t = monodromy(np.array(w) + gap * np.array([3, 2, 1]))
        for r in [dense_frame(rng), *(rotation(p, 0.31) for p in PAIRS)]:
            b = d.conj() @ r @ np.diag(spectrum(t)) @ r.T @ d.conj()
            candidates = []
            for weight in (0, 1, -1, np.sqrt(2), np.pi):
                _, o = np.linalg.eigh(b.real + weight * b.imag)
                diagonal = o.T @ b @ o
                candidates.append(
                    (
                        np.max(abs(diagonal - np.diag(np.diag(diagonal)))),
                        np.diag(diagonal),
                        o,
                    )
                )
            error, roots, o = min(candidates, key=lambda item: item[0])
            if error > 1e-12:
                raise ValueError("failed to construct prescribed-target case")
            g = fold(roots)
            order = min(
                PERMUTATIONS,
                key=lambda p: min(
                    np.max(abs(spectrum(g) - sign * roots[p])) for sign in (-1, 1)
                ),
            )
            o = o[:, order]
            if np.linalg.det(o) < 0:
                o[:, 0] *= -1
            rows.append(plant(generic, g, o, fold(spectrum(t))))
    return np.array(rows)


def generate(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        with temporary.open("wb") as stream:
            for name, construct in (
                ("regressions", lambda: np.array(REGRESSIONS)),
                ("grid", grid),
                ("haar", haar),
                ("boundaries", boundaries),
            ):
                rows = construct()
                if rows.shape[1:] != (3, 3) or not np.isfinite(rows).all():
                    raise ValueError(f"invalid {name} cases")
                np.asarray(rows, dtype="<f8").tofile(stream)
                print(f"{name}: {len(rows):,}", flush=True)
        temporary.replace(output)


if __name__ == "__main__":
    generate(Path(__file__).with_name("cases.bin"))
