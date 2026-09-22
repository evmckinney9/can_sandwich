#!/usr/bin/env python3
"""Generate the complete solver corpus from the original construction recipes.

Requires NumPy, SciPy and GULPS for canonical coordinates and Horn facets.
The solver under test is never used to accept or reject a generated case.
"""

from __future__ import annotations

import argparse
import itertools
import tempfile
from collections.abc import Generator
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import unitary_group

TOLERANCE = 1e-8
TOL = 1e-7

# Stratified recipe: former scripts/generate_realization_edge_corpus.py.
VERTICES = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.5),
    ],
    dtype=np.float64,
)
# These scales cross machine precision, coefficient cutoffs, the public
# realization tolerance, root-clustering thresholds, and ordinary near-wall
# geometry. Exact strata are separate families and always use scale zero.
NEAR_SCALES = np.array(
    (
        2e-15,
        1e-14,
        1e-13,
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        3e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-4,
        1e-2,
    )
)
FAMILIES = [
    "interior",
    *(f"face:{i}:{mode}" for i in range(4) for mode in ("exact", "near")),
    *(
        f"edge:{i},{j}:{mode}"
        for i, j in combinations(range(4), 2)
        for mode in ("exact", "near")
    ),
    *(f"vertex:{i}:{mode}" for i in range(4) for mode in ("exact", "near")),
]
LOCAL_MODES = (
    "identity",
    "left_haar",
    "right_haar",
    "both_haar",
    "correlated_haar",
    "near_identity",
)
DEFAULT_SAMPLES_PER_PAIR = len(NEAR_SCALES)
DEFAULT_TARGET_ROUNDS = 3
HORN_TOL = 2e-12

# Fixed raw-coordinate regressions found by the public replay. They remain at
# the head of every generated corpus, independently of random seed or corpus
# size, so the regular atomic bench cannot silently lose a repaired boundary.
REGRESSION_TRIPLES = np.array(
    [
        [
            [0.25156717590725, 0.24843282409275, 0.24690239120876],
            [0.03363566241836, 0.02691404826411, 0.02691404826411],
            [0.27897727016045, 0.27741771973249, 0.16056554217426],
        ],
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.29546099931489134, 0.20453900018510873, -0.20453900061744035],
        ],
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.484847219872809, 0.015152779130006122, -0.015152779314408282],
        ],
        [
            [0.2555803919905, 0.25558039198104, -0.2555803919405],
            [0.2555803919905, 0.25558039198104, -0.2555803919405],
            [0.2444196080691815, 0.07613277864424217, -0.0761327790245726],
        ],
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.47976171336231027, 3.350705823912392e-10, -4.450235180364359e-10],
        ],
        [
            [0.22382655216654007, 4.5365999717547037e-10, -3.6097999717547044e-10],
            [0.37673542185379, 0.0, 0.0],
            [0.39943802588698385, 4.5366116106166384e-10, -3.609834684704048e-10],
        ],
        [
            [4.920400000000001e-9, 1.38676e-9, -1.22755e-9],
            [0.5, 0.30909146235419005, -0.30909146235419005],
            [0.19090853874762126, 3.421808059123066e-9, -3.4846467933391523e-9],
        ],
        [
            [0.49999999751607005, 0.499999996086, -0.4999999939139901],
            [0.45073929780807004, 0.21470014625631004, -0.11617874287243998],
            [0.38382125933803596, -0.04926070113759151, -0.04926070436221991],
        ],
        [
            [0.49999999924211, 0.49999999906905, -0.49999999793094996],
            [0.5, 0.0, 0.0],
            [0.4999999991841258, 3.225370526216409e-10, -7.092456066892794e-10],
        ],
        [
            [0.49999999927363004, 0.4999999989163201, -0.49999999808368],
            [0.49999999995969, 9.689980276617723e-12, 3.390019723382273e-12],
            [0.4999999992595101, 3.593121356004758e-10, -5.216825860188123e-10],
        ],
        [
            [1.4011299999999997e-9, 1.34878e-9, -1.15103e-9],
            [0.49999999999956, 5.999978085130805e-14, -9.999780851308059e-15],
            [0.4999999992117509, 5.635630504262929e-10, -4.81460850321273e-10],
        ],
        [
            [0.5, 0.0, 0.0],
            [0.33556062825582006, 0.16437603912132998, 0.16437603911132997],
            [0.1643760391213343, 0.16437603911133436, -0.164312706488488],
        ],
        [
            [0.33556062825582006, 0.16437603912132998, 0.16437603911132997],
            [0.33556062825582006, 0.16437603912132998, 0.16437603911132997],
            [0.1643760391213343, 0.16437603911133436, -0.164312706488488],
        ],
    ],
    dtype=np.float64,
)

# Full-matrix regressions found only after changing the structured RNG seed.
# They are also converted to atomic triples below, but retaining the raw target
# is essential: the bugs lived in planner routing and endpoint recovery, which
# an invariant-only replay cannot exercise.
PIPELINE_REGRESSION_C_WEYL = np.array(
    [
        [0.6516483993422042, 0.3483516006577958, 0.0],
        [0.5000068692130752, 0.4999645953992824, 1.4595399282402184e-5],
    ]
)
PIPELINE_REGRESSION_G_WEYL = np.array(
    [
        [0.5093412847017496, 0.49065871529825034, 0.0],
        [0.5000261608572555, 0.4999528081467351, 2.8081467351152355e-6],
    ]
)
PIPELINE_REGRESSION_TARGETS = np.array(
    [
        [
            [
                0.874804581572709 + 1.6645030508924787e-5j,
                -8.46310384617607e-5 + 2.1572658010733333e-5j,
                1.113291128891849e-5 - 4.367518566994611e-5j,
                7.980421304697761e-6 + 0.4844759375442682j,
            ],
            [
                -9.519519718771955e-5 - 2.4265487827676243e-5j,
                -0.999999995 + 1.845925689479062e-5j,
                7.595738183130482e-19 + 1.2297963200405266e-16j,
                7.123119031889404e-7 - 2.7944491978970205e-6j,
            ],
            [
                -7.123119207480409e-7 - 2.7944491915636074e-6j,
                7.595738183130482e-19 + 1.2194972660077118e-16j,
                -0.999999995 - 1.845925689479062e-5j,
                -9.519519718790546e-5 + 2.42654878271608e-5j,
            ],
            [
                -7.980421304699359e-6 + 0.4844759375442682j,
                -1.1132911304528785e-5 - 4.367518566431497e-5j,
                -8.463103845885465e-5 - 2.1572658018789274e-5j,
                0.874804581572709 - 1.6645030508925675e-5j,
            ],
        ],
        [
            [
                0.9999999782882343 + 2.163635820188819e-5j,
                9.851160092000385e-5 + 1.6216111376301047e-5j,
                1.6122781834379098e-9 + 3.2871916385557364e-9j,
                -4.686861424369736e-9 + 0.00018162571770432228j,
            ],
            [
                9.851160073073624e-5 - 1.621611124879904e-5j,
                -0.9999999917511971 + 2.163635855071297e-5j,
                2.195659714304544e-9 + 7.785869287397015e-5j,
                2.595158457675651e-9 + 6.935082924263271e-9j,
            ],
            [
                -2.59488740096875e-9 + 6.935173297096767e-9j,
                2.0612528226790463e-9 + 7.785869287764415e-5j,
                -0.999999991439492 + 3.303849331842851e-5j,
                9.851220115789059e-5 + 1.6212463279616936e-5j,
            ],
            [
                -5.243497871096257e-9 + 0.00018162571768910604j,
                -1.6123856655805926e-9 + 3.287126210062619e-9j,
                9.851220134716147e-5 - 1.621246340711045e-5j,
                0.9999999779765292 + 3.3038492931167785e-5j,
            ],
        ],
    ],
    dtype=np.complex128,
)


def _simplex(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.dirichlet(np.ones(size))


def _weights(
    family: str,
    sample: int,
    rng: np.random.Generator,
    near_scale: float | None = None,
) -> np.ndarray:
    if family == "interior":
        # A floor keeps this family separate from deliberately near strata.
        raw = _simplex(rng, 4)
        return 0.04 + 0.84 * raw

    kind, indices, mode = family.split(":")
    chosen = tuple(map(int, indices.split(",")))
    scale = (
        0.0
        if mode == "exact"
        else float(
            NEAR_SCALES[sample % len(NEAR_SCALES)] if near_scale is None else near_scale
        )
    )
    weights = np.zeros(4)

    if kind == "face":
        omitted = chosen[0]
        support = [i for i in range(4) if i != omitted]
        weights[support] = (1.0 - scale) * _simplex(rng, 3)
        weights[omitted] = scale
    elif kind == "edge":
        support = list(chosen)
        complement = [i for i in range(4) if i not in support]
        weights[support] = (1.0 - scale) * _simplex(rng, 2)
        if scale:
            weights[complement] = scale * _simplex(rng, 2)
    elif kind == "vertex":
        vertex = chosen[0]
        complement = [i for i in range(4) if i != vertex]
        weights[vertex] = 1.0 - scale
        if scale:
            weights[complement] = scale * _simplex(rng, 3)
    else:  # pragma: no cover - FAMILIES is fixed above
        raise ValueError(f"unknown family: {family}")
    return weights


def _quaternion_su2(q: np.ndarray) -> np.ndarray:
    a, b, c, d = np.asarray(q).T
    out = np.empty((len(q), 2, 2), dtype=np.complex128)
    out[:, 0, 0] = a + 1j * b
    out[:, 0, 1] = c + 1j * d
    out[:, 1, 0] = -c + 1j * d
    out[:, 1, 1] = a - 1j * b
    return out


def _haar_su2(rng: np.random.Generator, count: int) -> np.ndarray:
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return _quaternion_su2(q)


def _near_identity_su2(
    rng: np.random.Generator, sample_indices: np.ndarray
) -> np.ndarray:
    axes = rng.normal(size=(len(sample_indices), 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.array([NEAR_SCALES[int(i) % len(NEAR_SCALES)] for i in sample_indices])
    q = np.empty((len(sample_indices), 4))
    q[:, 0] = np.cos(angles)
    q[:, 1:] = axes * np.sin(angles)[:, None]
    return _quaternion_su2(q)


def _local_layers(
    rng: np.random.Generator, sample_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(sample_indices)
    identity = np.broadcast_to(np.eye(2, dtype=np.complex128), (count, 2, 2)).copy()
    left = _haar_su2(rng, count)
    right = _haar_su2(rng, count)
    near_left = _near_identity_su2(rng, sample_indices)
    near_right = _near_identity_su2(rng, sample_indices[::-1])
    modes = sample_indices % len(LOCAL_MODES)
    a = left.copy()
    b = right.copy()
    a[modes == 0] = identity[modes == 0]
    b[modes == 0] = identity[modes == 0]
    b[modes == 1] = identity[modes == 1]
    a[modes == 2] = identity[modes == 2]
    b[modes == 4] = left[modes == 4]
    a[modes == 5] = near_left[modes == 5]
    b[modes == 5] = near_right[modes == 5]
    return a, b, modes.astype(np.int16)


def _monodromy_from_weyl(points: np.ndarray) -> np.ndarray:
    c1, c2, c3 = points.T
    return 0.5 * np.column_stack((c1 + c2 - c3, c1 - c2 + c3, -c1 + c2 + c3))


def _monodromy_from_unitaries(unitaries: np.ndarray) -> np.ndarray:
    from gulps import LocalEquivalenceClass

    classes = LocalEquivalenceClass.from_unitaries(list(unitaries))
    return np.asarray([cls._monodromy for cls in classes], dtype=float)


def _canonical_matrices(points: np.ndarray) -> np.ndarray:
    from gulps import LocalEquivalenceClass

    return np.asarray([LocalEquivalenceClass(list(point)).matrix for point in points])


def _witness_section(
    samples_per_pair: int,
    rng: np.random.Generator,
    batch_size: int,
    near_scale: float | None,
) -> np.ndarray:
    pair_codes = np.array(
        [
            (c, g, sample)
            for c in range(len(FAMILIES))
            for g in range(len(FAMILIES))
            for sample in range(samples_per_pair)
        ],
        dtype=np.int16,
    )
    count = len(pair_codes)
    c_weyl = np.empty((count, 3))
    g_weyl = np.empty((count, 3))
    for row, (c_code, g_code, sample) in enumerate(pair_codes):
        c_weyl[row] = _weights(FAMILIES[c_code], sample, rng, near_scale) @ VERTICES
        # Offset the gate scale so near C/G pairs do not move in lockstep.
        g_weyl[row] = (
            _weights(FAMILIES[g_code], sample + c_code, rng, near_scale) @ VERTICES
        )

    sample_indices = pair_codes[:, 2].astype(int)
    a, b, _ = _local_layers(rng, sample_indices)
    local = np.einsum("nij,nkl->nikjl", a, b).reshape(count, 4, 4)
    c_mono = _monodromy_from_weyl(c_weyl)
    g_mono = _monodromy_from_weyl(g_weyl)
    targets = np.empty((count, 3))
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        unitary = (
            _canonical_matrices(c_weyl[start:stop])
            @ local[start:stop]
            @ _canonical_matrices(g_weyl[start:stop])
        )
        targets[start:stop] = _monodromy_from_unitaries(unitary)

    return np.stack((c_mono, g_mono, targets), axis=1)


def _contains(facets: np.ndarray, point: np.ndarray) -> bool:
    return bool(np.all(facets[:, :3] @ point >= facets[:, 3] - HORN_TOL))


def _target_section(
    rounds: int,
    rng: np.random.Generator,
    near_scale: float | None,
) -> np.ndarray:
    from gulps import LocalEquivalenceClass
    from gulps._accelerate import invariants

    triples: list[np.ndarray] = []
    for c_code, c_family in enumerate(FAMILIES):
        for g_code, g_family in enumerate(FAMILIES):
            for round_index in range(rounds):
                key = round_index + 3 * c_code + 5 * g_code
                c_weyl = _weights(c_family, key, rng, near_scale) @ VERTICES
                g_weyl = _weights(g_family, key + c_code, rng, near_scale) @ VERTICES
                c_mono, g_mono = _monodromy_from_weyl(np.asarray([c_weyl, g_weyl]))
                flat = np.asarray(
                    invariants.sentence_facets([c_mono.tolist(), g_mono.tolist()]),
                    dtype=float,
                )
                facets = flat.reshape(-1, 4)
                for t_code, t_family in enumerate(FAMILIES):
                    t_weyl = (
                        _weights(t_family, key + 7 * t_code, rng, near_scale) @ VERTICES
                    )
                    t_mono = _monodromy_from_weyl(t_weyl[None])[0]
                    if not _contains(facets, t_mono):
                        weyl = invariants.weyl_from_monodromy(t_mono[None])[0]
                        reflected = np.asarray(
                            LocalEquivalenceClass(
                                weyl.tolist()
                            )._rho_reflect._monodromy,
                            dtype=float,
                        )
                        if not _contains(facets, reflected):
                            continue
                        t_mono = reflected
                    triples.append(np.stack((c_mono, g_mono, t_mono)))
    if not triples:
        raise RuntimeError("the target-stratum generator found no feasible rows")
    return np.asarray(triples)


def generate_stratified(
    samples_per_pair: int,
    target_rounds: int,
    seed: int,
    batch_size: int,
    near_scale: float | None = None,
) -> np.ndarray:
    if not 1 <= samples_per_pair <= np.iinfo(np.int16).max + 1:
        raise ValueError("samples_per_pair must be in [1, 32768]")
    if target_rounds < 1:
        raise ValueError("target_rounds must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if near_scale is not None and (
        not np.isfinite(near_scale) or not 0 <= near_scale <= 1
    ):
        raise ValueError("near_scale must be finite and in [0, 1]")
    rng = np.random.default_rng(seed)
    witness = _witness_section(samples_per_pair, rng, batch_size, near_scale)
    target = _target_section(target_rounds, rng, near_scale)
    pipeline_regression_triples = np.stack(
        (
            _monodromy_from_weyl(PIPELINE_REGRESSION_C_WEYL),
            _monodromy_from_weyl(PIPELINE_REGRESSION_G_WEYL),
            _monodromy_from_unitaries(PIPELINE_REGRESSION_TARGETS),
        ),
        axis=1,
    )
    triples = np.concatenate(
        (REGRESSION_TRIPLES, pipeline_regression_triples, witness, target)
    )
    return triples


# Targeted recipe: former benchmark/generate_targeted_cases.py.
PERMUTATIONS = np.array(list(itertools.permutations(range(4))))
SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=4)))
PAIRS = tuple(itertools.combinations(range(4), 2))
BLOCKS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
SCALES = (
    0.0,
    2e-15,
    1e-14,
    1e-13,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    3e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-4,
    1e-2,
)


def monodromy(weyl: np.ndarray) -> np.ndarray:
    a, b, c = weyl
    return np.array([(a + b - c) / 2, (a - b + c) / 2, (-a + b + c) / 2])


def fold_spectrum(roots: np.ndarray) -> np.ndarray:
    for sign in (1, -1):
        turns = np.angle(sign * roots) / (2 * np.pi)
        candidates = turns[PERMUTATIONS][:, None, :] + SHIFTS[None, :, :]
        m = candidates[..., [1, 0, 3]]
        w0 = m[..., 0] + m[..., 1]
        w1 = m[..., 0] + m[..., 2]
        w2 = m[..., 1] + m[..., 2]
        margins = np.stack((w0 - w1, w1 - w2, w2, 1 - w0 - w1), axis=-1)
        valid = (np.abs(candidates.sum(axis=-1)) < 2e-13) & (
            margins.min(axis=-1) >= -2e-13
        )
        positions = np.argwhere(valid)
        if len(positions):
            # Prefer the smallest numerical wall violation before deterministic
            # lexicographic tie-breaking; never project onto that wall.
            violations = np.maximum(-margins[valid], 0).max(axis=-1)
            permutation, shift = positions[np.argmin(violations)]
            return m[permutation, shift].copy(), PERMUTATIONS[permutation].copy(), sign
    raise ValueError("unit-determinant spectrum has no chamber representative")


def givens(pair: tuple[int, int], angle: float) -> np.ndarray:
    i, j = pair
    result = np.eye(4)
    cosine, sine = np.cos(angle), np.sin(angle)
    result[i, i] = result[j, j] = cosine
    result[i, j], result[j, i] = -sine, sine
    return result


def dense_frame(rng: np.random.Generator) -> np.ndarray:
    result = np.eye(4)
    for pair, angle in zip(PAIRS, rng.uniform(-np.pi, np.pi, 6)):
        result = result @ givens(pair, float(angle))
    return result


def diagonalize_symmetric_unitary(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best = None
    for weight in (0.0, 1.0, -1.0, np.sqrt(2), np.pi):
        _, orthogonal = np.linalg.eigh(matrix.real + weight * matrix.imag)
        transformed = orthogonal.T @ matrix @ orthogonal
        residual = float(np.max(np.abs(transformed - np.diag(np.diag(transformed)))))
        if best is None or residual < best[0]:
            best = residual, np.diag(transformed).copy(), orthogonal
    assert best is not None
    if best[0] > 1e-13:
        raise RuntimeError(f"joint spectral decomposition failed: {best[0]:.3g}")
    return best[1], best[2]


def generate_targeted(
    seed: int = 20260921,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []

    def append(row: np.ndarray, frame: np.ndarray, family: str, **details: Any) -> None:
        frame = frame.copy()
        if np.linalg.det(frame) < 0:
            frame[:, 0] *= -1
        valid, metrics = check_witness(row, frame.tolist())
        if not valid or max(metrics.values()) > 1e-12:
            raise RuntimeError(
                f"planted witness failed for {family}: {details}: {metrics}"
            )
        rows.append(row)

    def plant(
        a: np.ndarray, b: np.ndarray, frame: np.ndarray, family: str, **details: Any
    ) -> None:
        master = np.diag(spectrum(a)) @ frame @ np.diag(spectrum(b)) @ frame.T
        target, _, sign = fold_spectrum(np.linalg.eigvals(master))
        append(
            np.array([a, b, target]), frame, family, target_central_sign=sign, **details
        )

    generic = [
        monodromy(np.array(w))
        for w in ((0.41, 0.23, 0.07), (0.62, 0.24, 0.13), (0.47, 0.31, 0.19))
    ]
    inputs = [
        (generic[0], generic[1]),
        (generic[1], generic[2]),
        (generic[2], generic[0]),
    ]
    for input_index, (a, b) in enumerate(inputs):
        for permutation in PERMUTATIONS:
            frame = np.eye(4)[:, permutation]
            if np.linalg.det(frame) < 0:
                frame[:, 0] *= -1
            plant(
                a,
                b,
                frame,
                "permutation_vertex",
                input=input_index,
                permutation=permutation.tolist(),
            )
        for pair in PAIRS:
            for angle in (0.17, np.pi / 4, np.pi / 2, np.pi / 2 - 1e-10):
                plant(
                    a,
                    b,
                    givens(pair, angle),
                    "one_plane",
                    input=input_index,
                    plane=list(pair),
                    angle=angle,
                )
        for block_index, (first, second) in enumerate(BLOCKS):
            for angles in (
                (0.13, 0.39),
                (np.pi / 4, np.pi / 4),
                (0.0, 0.27),
                (1e-12, 0.3),
                (0.3, np.pi / 2 - 1e-12),
            ):
                frame = givens(first, angles[0]) @ givens(second, angles[1])
                plant(
                    a,
                    b,
                    frame,
                    "two_plus_two_support",
                    input=input_index,
                    block=block_index,
                    angles=list(angles),
                )
        for fixed in range(4):
            pairs = list(itertools.combinations([i for i in range(4) if i != fixed], 2))
            for sample in range(4):
                frame = np.eye(4)
                for pair, angle in zip(pairs, rng.uniform(-np.pi, np.pi, 3)):
                    frame = frame @ givens(pair, float(angle))
                plant(
                    a,
                    b,
                    frame,
                    "one_plus_three_support",
                    input=input_index,
                    fixed=fixed,
                    sample=sample,
                )
        for scale in SCALES:
            for pair in PAIRS:
                plant(
                    a,
                    b,
                    givens(pair, scale),
                    "near_local_scale",
                    input=input_index,
                    plane=list(pair),
                    scale=scale,
                )
            frame = np.eye(4)
            for pair in PAIRS:
                frame = frame @ givens(pair, scale)
            plant(a, b, frame, "near_local_dense", input=input_index, scale=scale)

    # Identical spectra force a_i b_j = a_j b_i; inverse spectra force four
    # diagonal routed products to agree, despite individually simple spectra.
    for input_index, a in enumerate(generic):
        for kind in ("equal", "inverse"):
            b = a.copy() if kind == "equal" else fold_spectrum(np.conj(spectrum(a)))[0]
            for sample in range(16):
                plant(
                    a,
                    b,
                    dense_frame(rng),
                    "routed_collision",
                    input=input_index,
                    kind=kind,
                    sample=sample,
                )

    # A prescribed target is realized by Btilde = D^-1 R target R^T D^-1.
    # It is symmetric unitary, so a real eigenbasis gives a planted witness.
    partitions = {
        "pair211": np.array([0.17, 0.17, -0.11, -0.23]),
        "pair22": np.array([0.19, 0.19, -0.19, -0.19]),
        "triple31": np.array([0.11, 0.11, 0.11, -0.33]),
        "scalar4": np.zeros(4),
    }
    root_thresholds = (
        64 * np.finfo(float).eps,
        np.sqrt(np.finfo(float).eps),
        1e-8,
        1e-7,
        1e-3,
    )
    perturbations = [("scale", scale) for scale in SCALES]
    for threshold in root_thresholds:
        phase = float(np.arcsin(threshold / 2) / np.pi)
        for side, value in (
            ("below", phase * 0.9),
            ("above", phase * 1.1),
            ("ulp_below", np.nextafter(phase, 0)),
            ("ulp_above", np.nextafter(phase, np.inf)),
        ):
            perturbations.append((f"threshold_{threshold:.17g}_{side}", float(value)))
    for partition, base_turns in partitions.items():
        for perturbation, scale in perturbations:
            turns = base_turns + scale * np.array([-1.5, -0.5, 0.5, 1.5])
            turns[3] = -sum(turns[:3])
            target_roots = np.exp(2j * np.pi * turns)
            target, _, target_sign = fold_spectrum(target_roots)
            for sample, a in enumerate(generic):
                a_turns = np.array([a[1], a[0], -sum(a), a[2]])
                d_inverse = np.diag(np.exp(-1j * np.pi * a_turns))
                rotation = dense_frame(rng)
                matrix = (
                    d_inverse
                    @ rotation
                    @ np.diag(target_roots)
                    @ rotation.T
                    @ d_inverse
                )
                b_roots, frame = diagonalize_symmetric_unitary(matrix)
                b, ordering, b_sign = fold_spectrum(b_roots)
                gaps = np.abs(target_roots[:, None] - target_roots[None, :])[
                    np.triu_indices(4, 1)
                ]
                append(
                    np.array([a, b, target]),
                    frame[:, ordering],
                    "target_multiplicity",
                    partition=partition,
                    perturbation=perturbation,
                    scale=scale,
                    sample=sample,
                    target_central_sign=b_sign * target_sign,
                    actual_target_min_gap=float(gaps.min()),
                )

    # Independently control both input gaps, including highly unequal scales.
    for first in (0.0, 1e-14, 1e-10, 1e-7, 1e-4):
        for second in (0.0, 1e-14, 1e-10, 1e-7, 1e-4):
            for sample in range(4):
                a = monodromy(np.array([0.5, first, 0.0]))
                b = monodromy(np.array([0.5, second, 0.0]))
                plant(
                    a,
                    b,
                    dense_frame(rng),
                    "anisotropic_pair22_inputs",
                    left_scale=first,
                    right_scale=second,
                    sample=sample,
                )

    # Cross the input gap and local angle independently, on both factor roles.
    # These raw Weyl coordinates already lie in the chamber; no spectral fold
    # or class constructor is allowed to quantize either input.
    input_partitions = {
        "pair211": np.array([0.5, 0.2, 0.2]),
        "pair22": np.array([0.5, 0.0, 0.0]),
        "triple31": np.array([0.2, 0.2, 0.2]),
        "scalar4": np.zeros(3),
    }
    for partition, base in input_partitions.items():
        for gap in (0.0, 1e-14, 1e-10, 1e-7, 1e-4):
            repeated = monodromy(base + gap * np.array([3.0, 2.0, 1.0]))
            for angle in (0.0, 1e-12, 1e-8, 0.31):
                for role in ("left", "right"):
                    a, b = (
                        (repeated, generic[0])
                        if role == "left"
                        else (generic[0], repeated)
                    )
                    for mode in ("one_plane", "dense"):
                        frame = givens((0, 2), angle)
                        if mode == "dense":
                            frame = np.eye(4)
                            for index, pair in enumerate(PAIRS):
                                frame = frame @ givens(pair, angle * (1 + index / 7))
                        plant(
                            a,
                            b,
                            frame,
                            "input_gap_local_angle_product",
                            partition=partition,
                            input_gap=gap,
                            local_angle=angle,
                            role=role,
                            mode=mode,
                        )

    return np.asarray(rows)


def spectrum(monodromy: np.ndarray) -> np.ndarray:
    w0, w1, w2 = (
        monodromy[0] + monodromy[1],
        monodromy[0] + monodromy[2],
        monodromy[1] + monodromy[2],
    )
    return np.exp(
        1j
        * np.pi
        * np.array([w0 - w1 + w2, w0 + w1 - w2, -w0 - w1 - w2, -w0 + w1 + w2])
    )


def check_witness(row: np.ndarray, witness: Any) -> tuple[bool, dict[str, float]]:
    if (
        not isinstance(witness, list)
        or len(witness) != 4
        or any(
            not isinstance(line, list)
            or len(line) != 4
            or any(type(value) not in (int, float) for value in line)
            for line in witness
        )
    ):
        raise ValueError("o must be a real numeric 4x4 JSON array")
    try:
        orthogonal = np.asarray(witness, dtype=float)
    except (ValueError, OverflowError) as exc:
        raise ValueError("o must contain finite real numbers") from exc
    if not np.all(np.isfinite(orthogonal)):
        raise ValueError("o must contain finite real numbers")
    gram = float(np.max(np.abs(orthogonal.T @ orthogonal - np.eye(4))))
    determinant = float(abs(np.linalg.det(orthogonal) - 1))
    metrics = {"orthogonality_max": gram, "determinant_error": determinant}
    if not np.isfinite(gram) or not np.isfinite(determinant):
        raise ValueError("o produces nonfinite matrix metrics")
    a, b, target = (spectrum(value) for value in row)
    actual = np.linalg.eigvals(np.diag(a) @ orthogonal @ np.diag(b) @ orthogonal.T)
    spectral = float(
        min(
            np.min(
                np.max(np.abs(actual[None, :] - sign * target[PERMUTATIONS]), axis=1)
            )
            for sign in (-1, 1)
        )
    )
    if not np.isfinite(spectral):
        raise ValueError("o produces nonfinite spectral metrics")
    metrics["spectral_bottleneck"] = spectral
    return all(value <= TOLERANCE for value in metrics.values()), metrics


# Original large-corpus QLR recipe (archived s01_generate.py).
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


def qlr_blocks():
    def phi(r, k, part):
        g = np.zeros(5)
        for i in range(r):
            g[k + i + 1 - part[i]] += 1
        return g[1:4] - g[4]

    blocks = []
    for r, k, a, b, c, d in QLR:
        for left, right in ((a, b), (b, a)):
            blocks.append((phi(r, k, left), phi(r, k, right), -phi(r, k, c), d))
            if a == b:
                break
    return tuple(np.array([row[i] for row in blocks]) for i in range(4))


ci, gi, cip, bi = qlr_blocks()


def weyl_linspace(N: int) -> Generator[tuple[float, ...], None, None]:
    N = int(N)
    if N <= 0:
        return iter(())

    V = VERTICES

    def gen() -> Generator[tuple[float, ...], None, None]:
        out = 0

        # 1) vertices
        for p in map(tuple, V):
            if out >= N:
                return
            yield p
            out += 1

        # 2) centroid (dead center)
        if out < N:
            yield tuple(V.mean(axis=0))
            out += 1

        # 3) refinement levels: denom = 2^L. Emit only NEW points at each level:
        # skip tuples where all (a,b,c,d) are even (those existed at denom/2).
        level = 1
        while out < N:
            denom = 2**level
            new_tuples = []
            for a in range(denom + 1):
                for b in range(denom - a + 1):
                    for c in range(denom - a - b + 1):
                        d = denom - a - b - c
                        if a % 2 == 0 and b % 2 == 0 and c % 2 == 0 and d % 2 == 0:
                            continue
                        # interior first: maximize the smallest barycentric
                        # weight, then the next-smallest, and so on
                        s = sorted((a, b, c, d))
                        new_tuples.append(((-s[0], -s[1], -s[2], -s[3]), a, b, c, d))
            new_tuples.sort(key=lambda t: t[0])
            for _, a, b, c, d in new_tuples:
                if out >= N:
                    return
                yield tuple(np.array([a, b, c, d], float) / denom @ V)
                out += 1
            level += 1

    return gen()


def rho_mono(m):
    d = -m.sum(axis=1)
    return np.stack([m[:, 2] + 0.5, d + 0.5, m[:, 0] - 0.5], axis=1)


def gen_haar(target, batch=50_000, seed=0):
    rng = np.random.default_rng(seed)

    def hm(n):
        U = unitary_group.rvs(4, size=n, random_state=rng)
        return _monodromy_from_unitaries(U)

    rows, got = [], 0
    while got < target:
        Cm, Gm, Tm = hm(batch), hm(batch), hm(batch)
        mar = (bi - Cm @ ci.T - Gm @ gi.T - Tm @ cip.T).min(1)
        marr = (bi - Cm @ ci.T - Gm @ gi.T - rho_mono(Tm) @ cip.T).min(1)
        plain = mar >= -TOL
        use_rho = ~plain & (marr >= -TOL)
        keep = plain | use_rho
        T_out = Tm.copy()
        T_out[use_rho] = rho_mono(Tm[use_rho])
        rows.append(np.stack([Cm[keep], Gm[keep], T_out[keep]], axis=1))
        got += int(keep.sum())
    return np.concatenate(rows, 0)[:target]


def gen_linspace(N):
    pts = np.array(list(weyl_linspace(N)), float)
    mono = _monodromy_from_weyl(pts)
    g_is_O = np.all(np.abs(pts) < 1e-12, axis=1)  # drop the no-op gate G = identity
    pc, pg = mono @ ci.T, mono @ gi.T
    pt, ptr = mono @ cip.T, rho_mono(mono) @ cip.T
    out = []
    for i in range(N):
        A = bi - pc[i]
        m_plain = (A - pg[:, None, :] - pt[None, :, :]).min(2)  # (g, t)
        m_rho = (A - pg[:, None, :] - ptr[None, :, :]).min(2)
        plain = m_plain >= -TOL
        use_rho = ~plain & (m_rho >= -TOL)
        keep = plain | use_rho
        keep[g_is_O, :] = False
        g_idx, t_idx = np.where(keep)
        T = mono[t_idx].copy()
        r = use_rho[g_idx, t_idx]
        T[r] = rho_mono(mono[t_idx[r]])
        C = np.broadcast_to(mono[i], (len(g_idx), 3))
        out.append(np.stack([C, mono[g_idx], T], axis=1))
    res = np.concatenate(out, 0)
    res = np.unique(res.reshape(-1, 9), axis=0).reshape(-1, 3, 3)
    return res


BASIC_CASES = [
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.173, 0.071, -0.019], [0.173, 0.071, -0.019]],
    [[0.173, 0.071, -0.019], [0.0, 0.0, 0.0], [0.173, 0.071, -0.019]],
    [
        [0.173, 0.071, -0.019],
        [0.109, 0.038, -0.011],
        [0.282, 0.10899999999999999, -0.03],
    ],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.25, -0.25]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
]

# Fixed solver calls captured from GULPS 38014b5 with can_sandwich af4e5ea.
# fmt: off
CAPTURED_CASES = [
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.25, 0.25, 0.25]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.375, 0.125, -0.125]],
    [[0.25, 0.25, -0.25], [0.375, 0.125, -0.125], [0.125, 0.125, 0.125]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.375, 0.125, -0.125]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.1875, 0.0625, -0.0625]],
    [[0.25, 0.25, -0.25], [0.1875, 0.0625, -0.0625], [0.3125, 0.1875, -0.0625]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.3, 0.10000000000000002, -0.10000000000000002]],
    [[0.25, 0.25, -0.25], [0.3, 0.10000000000000002, -0.10000000000000002], [0.35, 0.05, 0.05]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 0.04999999999999999, -0.04999999999999999]],
    [[0.25, 0.25, -0.25], [0.25, 0.04999999999999999, -0.04999999999999999], [0.20000000000000004, 0.2, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.2, 0.10000000000000002, -0.10000000000000002]],
    [[0.25, 0.25, -0.25], [0.2, 0.10000000000000002, -0.10000000000000002], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 0.04999999999999999, -0.04999999999999999]],
    [[0.25, 0.25, -0.25], [0.25, 0.04999999999999999, -0.04999999999999999], [0.20000000000000004, 0.2, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.005, 0.005, -0.005]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.255, 0.245, -0.245]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.255, 0.245, -0.245]],
    [[0.25, 0.25, -0.25], [0.255, 0.245, -0.245], [0.495, 0.005, 0.005]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.495, 0.0050000000000000044, -0.0050000000000000044]],
    [[0.25, 0.25, -0.25], [0.495, 0.0050000000000000044, -0.0050000000000000044], [0.255, 0.245, 0.245]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [5e-05, 5e-05, -5e-05]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25005, 0.24995, -0.24995]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25005, 0.24995, -0.24995]],
    [[0.25, 0.25, -0.25], [0.25005, 0.24995, -0.24995], [0.49995, 5e-05, 5e-05]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.49995, 4.999999999999449e-05, -4.999999999999449e-05]],
    [[0.25, 0.25, -0.25], [0.49995, 4.999999999999449e-05, -4.999999999999449e-05], [0.25005, 0.24995, 0.24995]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [5e-07, 5e-07, -5e-07]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.25, 0.25, -0.25], [0.2500005, 0.24999949999999999, -0.24999949999999999], [0.4999995, 5e-07, 5e-07]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.4999995, 5.000000000143778e-07, -5.000000000143778e-07]],
    [[0.25, 0.25, -0.25], [0.4999995, 5.000000000143778e-07, -5.000000000143778e-07], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [5e-11, 5e-11, -5e-11]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.49999999995, 5e-11, 5e-11]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.49999999995, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.49999999995, 0.0, 0.0], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 1.000000082740371e-10, -1.000000082740371e-10]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.4999999999995, 5e-13, 5e-13]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.4999999999995, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.4999999999995, 0.0, 0.0], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 9.999917560676863e-13, -9.999917560676863e-13]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.250000005, 0.24999999499999997, -0.24999999499999997]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 0.25, -0.25]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.499999995, 5e-09, 5e-09]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.499999995, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.499999995, 0.0, 0.0], [0.25000000499999997, 0.249999995, 0.249999995]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.03571693876035, 0.03571408163382, -0.03571408163382]],
    [[0.25, 0.25, -0.25], [0.03571693876035, 0.03571408163382, -0.03571408163382], [0.21428591836618, 0.21428591836618, -0.21428306123964996]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.11439045103094, 0.08457391710796, -0.08457391710796]],
    [[0.25, 0.25, -0.25], [0.11439045103094, 0.08457391710796, -0.08457391710796], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.24266232562440998, 0.028813266264189993, -0.028813266264189993]],
    [[0.25, 0.25, -0.25], [0.24266232562440998, 0.028813266264189993, -0.028813266264189993], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.15624797814723, 0.05420297540007999, -0.05420297540007999]],
    [[0.25, 0.25, -0.25], [0.15624797814723, 0.05420297540007999, -0.05420297540007999], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.19471724673529, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.30016103400692, 0.19983896599306, -0.19983896599306]],
    [[0.25, 0.25, -0.25], [0.30016103400692, 0.19983896599306, -0.19983896599306], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.26389674666574, 0.035266619199510005, -0.035266619199510005]],
    [[0.25, 0.25, -0.25], [0.26389674666574, 0.035266619199510005, -0.035266619199510005], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.2700866686565, 0.0019422880197300096, -0.0019422880197300096]],
    [[0.25, 0.25, -0.25], [0.2700866686565, 0.0019422880197300096, -0.0019422880197300096], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.26929977582863, 0.0164910443704, -0.0164910443704]],
    [[0.25, 0.25, -0.25], [0.26929977582863, 0.0164910443704, -0.0164910443704], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.25, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.25, 0.0, 0.0], [0.375, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.375, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, 0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, 0.125], [0.25, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.125, 0.125], [0.25, 0.25, 0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.25, 0.125], [0.25, 0.25, 0.25]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, 0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.25, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.125, -0.125], [0.375, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.1875, 0.0625, -0.0625]],
    [[0.125, 0.0, 0.0], [0.1875, 0.0625, -0.0625], [0.1875, 0.1875, -0.0625]],
    [[0.125, 0.0, 0.0], [0.1875, 0.1875, -0.0625], [0.3125, 0.1875, -0.0625]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15, 0.05000000000000001, -6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.15, 0.05000000000000001, -6.938893903907228e-18], [0.22500000000000003, 0.05, 0.05]],
    [[0.125, 0.0, 0.0], [0.22500000000000003, 0.05, 0.05], [0.35, 0.05, 0.05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0]],
    [[0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18], [0.20000000000000004, 0.2, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15, 0.07499999999999998, -0.07499999999999998]],
    [[0.125, 0.0, 0.0], [0.15, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15, 0.1, -0.05000000000000001]],
    [[0.125, 0.0, 0.0], [0.15, 0.1, -0.05000000000000001], [0.22500000000000003, 0.15, -0.05000000000000001]],
    [[0.125, 0.0, 0.0], [0.22500000000000003, 0.15, -0.05000000000000001], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0]],
    [[0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18], [0.20000000000000004, 0.2, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.005, 0.005, -0.005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.13, 0.12, -0.12]],
    [[0.125, 0.0, 0.0], [0.13, 0.12, -0.12], [0.245, 0.13, -0.13]],
    [[0.125, 0.0, 0.0], [0.245, 0.13, -0.13], [0.255, 0.245, -0.245]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.005000000000000003, -2.168404344971009e-18]],
    [[0.125, 0.0, 0.0], [0.125, 0.005000000000000003, -2.168404344971009e-18], [0.245, 0.005, 0.005]],
    [[0.125, 0.0, 0.0], [0.245, 0.005, 0.005], [0.37, 0.005, 0.005]],
    [[0.125, 0.0, 0.0], [0.37, 0.005, 0.005], [0.495, 0.005, 0.005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.12, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.12, 0.0], [0.13, 0.12, 0.12]],
    [[0.125, 0.0, 0.0], [0.13, 0.12, 0.12], [0.245, 0.13, 0.12]],
    [[0.125, 0.0, 0.0], [0.245, 0.13, 0.12], [0.245, 0.245, 0.13]],
    [[0.125, 0.0, 0.0], [0.245, 0.245, 0.13], [0.255, 0.245, 0.245]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.010000000000000002, -0.010000000000000002]],
    [[0.125, 0.0, 0.0], [0.125, 0.010000000000000002, -0.010000000000000002], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [5e-05, 5e-05, -5e-05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12505, 0.12495, -0.12495]],
    [[0.125, 0.0, 0.0], [0.12505, 0.12495, -0.12495], [0.24995, 0.12505, -0.12505]],
    [[0.125, 0.0, 0.0], [0.24995, 0.12505, -0.12505], [0.25005, 0.24995, -0.24995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 4.9999999999997244e-05, 2.7545511444709847e-18]],
    [[0.125, 0.0, 0.0], [0.125, 4.9999999999997244e-05, 2.7545511444709847e-18], [0.24995, 5e-05, 5e-05]],
    [[0.125, 0.0, 0.0], [0.24995, 5e-05, 5e-05], [0.37495, 5e-05, 5e-05]],
    [[0.125, 0.0, 0.0], [0.37495, 5e-05, 5e-05], [0.49995, 5e-05, 5e-05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.12495, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.12495, 0.0], [0.12505, 0.12495, 0.12495]],
    [[0.125, 0.0, 0.0], [0.12505, 0.12495, 0.12495], [0.24995, 0.12505, 0.12495]],
    [[0.125, 0.0, 0.0], [0.24995, 0.12505, 0.12495], [0.24995, 0.24995, 0.12505]],
    [[0.125, 0.0, 0.0], [0.24995, 0.24995, 0.12505], [0.25005, 0.24995, 0.24995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 9.999999999999593e-05, -9.999999999999593e-05]],
    [[0.125, 0.0, 0.0], [0.125, 9.999999999999593e-05, -9.999999999999593e-05], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [5e-07, 5e-07, -5e-07]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.1250005, 0.12499950000000001, -0.12499950000000001]],
    [[0.125, 0.0, 0.0], [0.1250005, 0.12499950000000001, -0.12499950000000001], [0.2499995, 0.1250005, -0.1250005]],
    [[0.125, 0.0, 0.0], [0.2499995, 0.1250005, -0.1250005], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999999999999, 4.999999999933111e-07, 6.6888603657895996e-18]],
    [[0.125, 0.0, 0.0], [0.12499999999999999, 4.999999999933111e-07, 6.6888603657895996e-18], [0.2499995, 5e-07, 5e-07]],
    [[0.125, 0.0, 0.0], [0.2499995, 5e-07, 5e-07], [0.3749995, 5e-07, 5e-07]],
    [[0.125, 0.0, 0.0], [0.3749995, 5e-07, 5e-07], [0.4999995, 5e-07, 5e-07]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.12499949999999999, 6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.125, 0.12499949999999999, 6.938893903907228e-18], [0.12500050000000001, 0.1249995, 0.1249995]],
    [[0.125, 0.0, 0.0], [0.12500050000000001, 0.1249995, 0.1249995], [0.24999950000000004, 0.1250005, 0.12499950000000001]],
    [[0.125, 0.0, 0.0], [0.24999950000000004, 0.1250005, 0.12499950000000001], [0.24999950000000004, 0.2499995, 0.1250005]],
    [[0.125, 0.0, 0.0], [0.24999950000000004, 0.2499995, 0.1250005], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 1.000000000001e-06, -1.000000000001e-06]],
    [[0.125, 0.0, 0.0], [0.125, 1.000000000001e-06, -1.000000000001e-06], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [5e-11, 5e-11, -5e-11]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.24999999995, 0.12500000005, -0.12500000005]],
    [[0.125, 0.0, 0.0], [0.24999999995, 0.12500000005, -0.12500000005], [0.25000000005, 0.24999999995, -0.24999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.24999999995, 5e-11, 5e-11]],
    [[0.125, 0.0, 0.0], [0.24999999995, 5e-11, 5e-11], [0.37499999995, 5e-11, 5e-11]],
    [[0.125, 0.0, 0.0], [0.37499999995, 5e-11, 5e-11], [0.49999999995, 5e-11, 5e-11]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999998333, 0.12499999998333, -1.6669998714746725e-11]],
    [[0.125, 0.0, 0.0], [0.12499999998333, 0.12499999998333, -1.6669998714746725e-11], [0.12499999998333, 0.12499999998333, 0.12499999998333]],
    [[0.125, 0.0, 0.0], [0.12499999998333, 0.12499999998333, 0.12499999998333], [0.24999999995, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.24999999995, 0.125, 0.125], [0.24999999995, 0.24999999995, 0.12500000005]],
    [[0.125, 0.0, 0.0], [0.24999999995, 0.24999999995, 0.12500000005], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 1.000000082740371e-10, -1.000000082740371e-10]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.2499999999995, 0.1250000000005, -0.1250000000005]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 0.1250000000005, -0.1250000000005], [0.2500000000005, 0.2499999999995, -0.2499999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.2499999999995, 5e-13, 5e-13]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 5e-13, 5e-13], [0.3749999999995, 5e-13, 5e-13]],
    [[0.125, 0.0, 0.0], [0.3749999999995, 5e-13, 5e-13], [0.4999999999995, 5e-13, 5e-13]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999999983, 0.12499999999983, -1.700029006457271e-13]],
    [[0.125, 0.0, 0.0], [0.12499999999983, 0.12499999999983, -1.700029006457271e-13], [0.12499999999983, 0.12499999999983, 0.12499999999983]],
    [[0.125, 0.0, 0.0], [0.12499999999983, 0.12499999999983, 0.12499999999983], [0.2499999999995, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 0.125, 0.125], [0.2499999999995, 0.2499999999995, 0.1250000000005]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 0.2499999999995, 0.1250000000005], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 9.999917560676863e-13, -9.999917560676863e-13]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.249999995, 0.125000005, -0.125000005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.124999995, 5e-09, 5e-09]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999833333, 0.12499999833333, -1.6666699964584808e-09]],
    [[0.125, 0.0, 0.0], [0.12499999833333, 0.12499999833333, -1.6666699964584808e-09], [0.12499999833333, 0.12499999833333, 0.12499999833333]],
    [[0.125, 0.0, 0.0], [0.12499999833333, 0.12499999833333, 0.12499999833333], [0.24999999500000003, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.08928591836618, 0.08928591836618, -0.08928306123965002]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.14417975537864, 0.04042608289204, -0.048996289301620005]],
    [[0.125, 0.0, 0.0], [0.14417975537864, 0.04042608289204, -0.048996289301620005], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15682556066818, 0.0073376743755899955, -0.06797650130796]],
    [[0.125, 0.0, 0.0], [0.15682556066818, 0.0073376743755899955, -0.06797650130796], [0.28182556066818, 0.007337674375590006, -0.09618673373581]],
    [[0.125, 0.0, 0.0], [0.28182556066818, 0.007337674375590006, -0.09618673373581], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.1085391923974, 0.1085391923974, -0.1085391923974]],
    [[0.125, 0.0, 0.0], [0.1085391923974, 0.1085391923974, -0.1085391923974], [0.2335391923974, 0.10853919239740001, -0.10853919239740001]],
    [[0.125, 0.0, 0.0], [0.2335391923974, 0.10853919239740001, -0.10853919239740001], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.10984543262718, 0.054202975400080004, -0.07029638617449]],
    [[0.125, 0.0, 0.0], [0.10984543262718, 0.054202975400080004, -0.07029638617449], [0.17920297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.125, 0.0, 0.0], [0.17920297540008, 0.10984543262718, -0.09375202185276998], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.19471724673529, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.13725317642186, 0.03482075476810001, -0.03482075476810001]],
    [[0.125, 0.0, 0.0], [0.13725317642186, 0.03482075476810001, -0.03482075476810001], [0.1598207547681, 0.13725317642186, -0.13725317642186]],
    [[0.125, 0.0, 0.0], [0.1598207547681, 0.13725317642186, -0.13725317642186], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.07483896599306, 0.025483102020800008, -0.050161034006930005]],
    [[0.125, 0.0, 0.0], [0.07483896599306, 0.025483102020800008, -0.050161034006930005], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12755519361579, 3.469446951953614e-18, -0.037821812815300004]],
    [[0.125, 0.0, 0.0], [0.12755519361579, 3.469446951953614e-18, -0.037821812815300004], [0.25255519361579, -0.013896746665740006, -0.08973338080048998]],
    [[0.125, 0.0, 0.0], [0.25255519361579, -0.013896746665740006, -0.08973338080048998], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12694228801973, 0.02819367505855, 0.0]],
    [[0.125, 0.0, 0.0], [0.12694228801973, 0.02819367505855, 0.0], [0.13310700640205, 0.12694228801973, 0.02008666865649998]],
    [[0.125, 0.0, 0.0], [0.13310700640205, 0.12694228801973, 0.02008666865649998], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12780873145823, 0.019299775828629996, 5.204170427930421e-18]],
    [[0.125, 0.0, 0.0], [0.12780873145823, 0.019299775828629996, 5.204170427930421e-18], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, -0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, -0.16666666666667], [0.25, 0.25, -0.25]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666668, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666668, 0.0, 0.0], [0.25000000000001, 0.08333333333332998, -0.08333333333332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.25000000000001, 0.08333333333332998, -0.08333333333332998], [0.33333333333334, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333333333334, 0.0, 0.0], [0.41666666666667, 0.08333333333332998, -0.08333333333332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.41666666666667, 0.08333333333332998, -0.08333333333332998], [0.5, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 1e-14, 1e-14]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 1e-14, 1e-14], [0.08333333333334, 0.08333333333334, 0.08333333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333334, 0.08333333333334, 0.08333333333334], [0.16666666666667002, 0.16666666666667, 9.992007221626409e-15]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667002, 0.16666666666667, 9.992007221626409e-15], [0.25, 0.08333333333334, 0.08333333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.25, 0.08333333333334, 0.08333333333334], [0.16666666666667, 0.16666666666667, 0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, 0.16666666666667], [0.25, 0.25, 0.08333333333334003]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.25, 0.25, 0.08333333333334003], [0.33333333333333004, 0.16666666666667, 0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333333333333004, 0.16666666666667, 0.16666666666667], [0.25, 0.25, 0.25]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.12500000000001, 0.04166666666667001, -0.04166666666667001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.12500000000001, 0.04166666666667001, -0.04166666666667001], [0.125, 0.125, -0.041666666666659996]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.125, 0.125, -0.041666666666659996], [0.20833333333332998, 0.04166666666667, 0.04166666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.20833333333332998, 0.04166666666667, 0.04166666666667], [0.125, 0.125, 0.125]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.08333333333333001, -0.08333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.08333333333333001, -0.08333333333333001], [0.25, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.12500000000001, 0.04166666666667001, -0.04166666666667001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.12500000000001, 0.04166666666667001, -0.04166666666667001], [0.20833333333334, 0.041666666666659996, -0.041666666666659996]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.20833333333334, 0.041666666666659996, -0.041666666666659996], [0.29166666666667, 0.041666666666670016, -0.041666666666670016]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.29166666666667, 0.041666666666670016, -0.041666666666670016], [0.375, 0.125, -0.125]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.10416666666666, 0.020833333333340004, -0.020833333333340004]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.10416666666666, 0.020833333333340004, -0.020833333333340004], [0.10416666666667, 0.06250000000001, 0.020833333333329998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.10416666666667, 0.06250000000001, 0.020833333333329998], [0.14583333333334, 0.10416666666666, 0.020833333333340004]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.14583333333334, 0.10416666666666, 0.020833333333340004], [0.22916666666666996, 0.10416666666666999, 0.020833333333330012]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.22916666666666996, 0.10416666666666999, 0.020833333333330012], [0.3125, 0.1875, -0.0625]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11666666666668, 0.01666666666668, -0.01666666666668]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11666666666668, 0.01666666666668, -0.01666666666668], [0.13333333333333003, 0.10000000000001, -0.033333333333329995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.13333333333333003, 0.10000000000001, -0.033333333333329995], [0.18333333333334, 0.05, 0.05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.18333333333334, 0.05, 0.05], [0.26666666666667, 0.13333333333333003, -0.03333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.26666666666667, 0.13333333333333003, -0.03333333333333001], [0.35, 0.05, 0.05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15000000000001, 0.04999999999999, -0.04999999999999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15000000000001, 0.04999999999999, -0.04999999999999], [0.16666666666666002, 0.03333333333334, 0.03333333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666666002, 0.03333333333334, 0.03333333333334], [0.11666666666667, 0.11666666666667, 0.08333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11666666666667, 0.11666666666667, 0.08333333333333001], [0.20000000000000004, 0.2, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.10833333333334, 0.07500000000000001, -0.07500000000000001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.10833333333334, 0.07500000000000001, -0.07500000000000001], [0.19166666666667, 0.00833333333333, -0.00833333333333]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.19166666666667, 0.00833333333333, -0.00833333333333], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11666666666666, 0.01666666666668, -0.01666666666668]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11666666666666, 0.01666666666668, -0.01666666666668], [0.10000000000001001, 0.06666666666667001, 0.033333333333329995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.10000000000001001, 0.06666666666667001, 0.033333333333329995], [0.18333333333334, 0.11666666666665998, -0.016666666666659995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.18333333333334, 0.11666666666665998, -0.016666666666659995], [0.26666666666667, 0.06666666666667, 0.03333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.26666666666667, 0.06666666666667, 0.03333333333333001], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15000000000001, 0.04999999999999, -0.04999999999999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15000000000001, 0.04999999999999, -0.04999999999999], [0.16666666666666002, 0.03333333333334, 0.03333333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666666002, 0.03333333333334, 0.03333333333334], [0.11666666666667, 0.11666666666667, 0.08333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11666666666667, 0.11666666666667, 0.08333333333333001], [0.20000000000000004, 0.2, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.005, 0.005, -0.005]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08833333333334001, 0.07833333333334, -0.07833333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08833333333334001, 0.07833333333334, -0.07833333333334], [0.17166666666667002, 0.16166666666667, -0.16166666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.17166666666667002, 0.16166666666667, -0.16166666666667], [0.255, 0.245, -0.245]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08833333333334001, 0.07833333333334, -0.07833333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08833333333334001, 0.07833333333334, -0.07833333333334], [0.16166666666668, 0.005, 0.005]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16166666666668, 0.005, 0.005], [0.24500000000001, 0.08833333333332999, -0.07833333333332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24500000000001, 0.08833333333332999, -0.07833333333332998], [0.32833333333334, 0.005, 0.005]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.32833333333334, 0.005, 0.005], [0.41166666666667, 0.08833333333332999, -0.07833333333332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.41166666666667, 0.08833333333332999, -0.07833333333332998], [0.495, 0.005, 0.005]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16166666666667, 0.0050000000000099964, -0.0050000000000099964]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16166666666667, 0.0050000000000099964, -0.0050000000000099964], [0.08833333333334001, 0.07833333333334, 0.07833333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08833333333334001, 0.07833333333334, 0.07833333333334], [0.16166666666667, 0.16166666666667, 0.0050000000000099964]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16166666666667, 0.16166666666667, 0.0050000000000099964], [0.24500000000000002, 0.08833333333333998, 0.07833333333334003]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24500000000000002, 0.08833333333333998, 0.07833333333334003], [0.17166666666667002, 0.16166666666667, 0.16166666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.17166666666667002, 0.16166666666667, 0.16166666666667], [0.24499999999999997, 0.245, 0.08833333333333998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24499999999999997, 0.245, 0.08833333333333998], [0.32833333333333, 0.17166666666667002, 0.16166666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.32833333333333, 0.17166666666667002, 0.16166666666667], [0.255, 0.245, 0.245]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.07333333333333, -0.07333333333333]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.07333333333333, -0.07333333333333], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [5e-05, 5e-05, -5e-05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08338333333334, 0.08328333333334001, -0.08328333333334001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08338333333334, 0.08328333333334001, -0.08328333333334001], [0.16671666666667, 0.16661666666667002, -0.16661666666667002]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16671666666667, 0.16661666666667002, -0.16661666666667002], [0.25005, 0.24995, -0.24995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08338333333334, 0.08328333333334001, -0.08328333333334001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08338333333334, 0.08328333333334001, -0.08328333333334001], [0.16661666666668, 5e-05, 5e-05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16661666666668, 5e-05, 5e-05], [0.24995000000001, 0.08338333333333, -0.08328333333333002]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24995000000001, 0.08338333333333, -0.08328333333333002], [0.33328333333334, 5e-05, 5e-05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33328333333334, 5e-05, 5e-05], [0.41661666666667, 0.08338333333332998, -0.08328333333332999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.41661666666667, 0.08338333333332998, -0.08328333333332999], [0.49995, 5e-05, 5e-05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16661666666667, 5.00000000099865e-05, -5.00000000099865e-05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16661666666667, 5.00000000099865e-05, -5.00000000099865e-05], [0.08338333333334001, 0.08328333333334, 0.08328333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08338333333334001, 0.08328333333334, 0.08328333333334], [0.16661666666667002, 0.16661666666667, 5.00000000099865e-05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16661666666667002, 0.16661666666667, 5.00000000099865e-05], [0.24995000000000003, 0.08338333333334003, 0.08328333333333998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24995000000000003, 0.08338333333334003, 0.08328333333333998], [0.16671666666667004, 0.16661666666667, 0.16661666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16671666666667004, 0.16661666666667, 0.16661666666667], [0.24995, 0.24995, 0.08338333333334003]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24995, 0.24995, 0.08338333333334003], [0.33328333333333, 0.16671666666667, 0.16661666666667002]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33328333333333, 0.16671666666667, 0.16661666666667002], [0.25005, 0.24995, 0.24995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.08323333333333, -0.08323333333333]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.08323333333333, -0.08323333333333], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [5e-07, 5e-07, -5e-07]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333383333334, 0.08333283333334, -0.08333283333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333383333334, 0.08333283333334, -0.08333283333334], [0.16666716666667003, 0.16666616666667, -0.16666616666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666716666667003, 0.16666616666667, -0.16666616666667], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333383333334, 0.08333283333334, -0.08333283333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333383333334, 0.08333283333334, -0.08333283333334], [0.16666616666668, 5e-07, 5e-07]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666616666668, 5e-07, 5e-07], [0.24999950000000998, 0.08333383333333001, -0.08333283333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24999950000000998, 0.08333383333333001, -0.08333283333333001], [0.33333283333334, 5e-07, 5e-07]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333283333334, 5e-07, 5e-07], [0.41666616666667, 0.08333383333332998, -0.08333283333332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.41666616666667, 0.08333383333332998, -0.08333283333332998], [0.4999995, 5e-07, 5e-07]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666616666667, 5.000000100063851e-07, -5.000000100063851e-07]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666616666667, 5.000000100063851e-07, -5.000000100063851e-07], [0.08333383333334, 0.08333283333334, 0.08333283333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333383333334, 0.08333283333334, 0.08333283333334], [0.16666616666667, 0.16666616666667, 5.000000100063851e-07]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666616666667, 0.16666616666667, 5.000000100063851e-07], [0.2499995, 0.08333383333334002, 0.08333283333333999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.2499995, 0.08333383333334002, 0.08333283333333999], [0.16666716666667003, 0.16666616666667, 0.16666616666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666716666667003, 0.16666616666667, 0.16666616666667], [0.24999949999999999, 0.2499995, 0.08333383333334002]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24999949999999999, 0.2499995, 0.08333383333334002], [0.33333283333333, 0.16666716666667, 0.16666616666667003]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333283333333, 0.16666716666667, 0.16666616666667003], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.08333233333332998, -0.08333233333332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.08333233333332998, -0.08333233333332998], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [5e-11, 5e-11, -5e-11]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, -0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, -0.16666666666667], [0.25000000005, 0.24999999995, -0.24999999995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666661668, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666661668, 0.0, 0.0], [0.24999999995001, 0.08333333338332999, -0.08333333328332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24999999995001, 0.08333333338332999, -0.08333333328332998], [0.33333333328333997, 5e-11, 5e-11]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333333328333997, 5e-11, 5e-11], [0.41666666661667, 0.08333333338332999, -0.08333333328332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.41666666661667, 0.08333333338332999, -0.08333333328332998], [0.49999999995, 5e-11, 5e-11]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666665, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666665, 0.0, 0.0], [0.08333333331667, 0.08333333331667, 0.08333333331667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333331667, 0.08333333331667, 0.08333333331667], [0.16666666665000002, 0.16666666665, -1.66600067075251e-11]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666665000002, 0.16666666665, -1.66600067075251e-11], [0.24999999998333003, 0.08333333331667, 0.08333333331667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24999999998333003, 0.08333333331667, 0.08333333331667], [0.16666666665, 0.16666666665, 0.16666666665]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666665, 0.16666666665, 0.16666666665], [0.249999999975, 0.249999999975, 0.08333333333334003]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.249999999975, 0.249999999975, 0.08333333333334003], [0.33333333328333004, 0.16666666666667, 0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333333328333004, 0.16666666666667, 0.16666666666667], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666656667, 0.08333333333332998, -0.08333333333332998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666656667, 0.08333333333332998, -0.08333333333332998], [0.25, 1.000000082740371e-10, -1.000000082740371e-10]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, -0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, -0.16666666666667], [0.2500000000005, 0.2499999999995, -0.2499999999995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666618, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666618, 0.0, 0.0], [0.24999999999951, 0.08333333333383001, -0.08333333333283001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24999999999951, 0.08333333333383001, -0.08333333333283001], [0.33333333333284, 5e-13, 5e-13]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333333333284, 5e-13, 5e-13], [0.41666666666617, 0.08333333333382999, -0.08333333333282998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.41666666666617, 0.08333333333382999, -0.08333333333282998], [0.4999999999995, 5e-13, 5e-13]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1666666666665, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1666666666665, 0.0, 0.0], [0.08333333333317, 0.08333333333317, 0.08333333333317]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333317, 0.08333333333317, 0.08333333333317], [0.1666666666665, 0.1666666666665, -1.6001089342410069e-13]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1666666666665, 0.1666666666665, -1.6001089342410069e-13], [0.24999999999982997, 0.08333333333317, 0.08333333333317]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24999999999982997, 0.08333333333317, 0.08333333333317], [0.1666666666665, 0.1666666666665, 0.1666666666665]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1666666666665, 0.1666666666665, 0.1666666666665], [0.24999999999975003, 0.24999999999975, 0.08333333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24999999999975003, 0.24999999999975, 0.08333333333334], [0.33333333333283, 0.16666666666667, 0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.33333333333283, 0.16666666666667, 0.16666666666667], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666567, 0.08333333333333001, -0.08333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666567, 0.08333333333333001, -0.08333333333333001], [0.25, 9.999917560676863e-13, -9.999917560676863e-13]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333334, 0.08333333333334, -0.08333333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333334, 0.08333333333334, -0.08333333333334], [0.16666666666667, 0.16666666666667, -0.16666666666667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666666666667, 0.16666666666667, -0.16666666666667], [0.250000005, 0.24999999499999997, -0.24999999499999997]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333334, 0.08333333333334, -0.08333333333334]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333334, 0.08333333333334, -0.08333333333334], [0.16666666166668, 5e-09, 5e-09]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.166666665, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.166666665, 0.0, 0.0], [0.08333333166667, 0.08333333166667, 0.08333333166667]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666665666667, 0.08333333333333001, -0.08333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16666665666667, 0.08333333333333001, -0.08333333333333001], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.13095544215938, 0.13095258503285, -0.13095258503285]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.13095544215938, 0.13095258503285, -0.13095258503285], [0.21428591836618, 0.21428591836618, -0.21428306123964996]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.09066295596829, 0.060846422045309996, -0.060846422045309996]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.09066295596829, 0.060846422045309996, -0.060846422045309996], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15682556066819, 0.028813266264179987, -0.028813266264179987]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15682556066819, 0.028813266264179987, -0.028813266264179987], [0.24015889400152002, -0.026309834641299996, -0.054520067069150004]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.24015889400152002, -0.026309834641299996, -0.054520067069150004], [0.32349222733485, -0.07599565895774, -0.10964316797462999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.32349222733485, -0.07599565895774, -0.10964316797462999], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15020585906407, 0.15020585906407, -0.15020585906407]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15020585906407, 0.15020585906407, -0.15020585906407], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15362971950783, 0.05682123403948, -0.05682123403948]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15362971950783, 0.05682123403948, -0.05682123403948], [0.22086964206675, 0.026512099293850004, -0.010418688519440005]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.22086964206675, 0.026512099293850004, -0.010418688519440005], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11138391340196, 0.08333333333333001, -0.08333333333333001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11138391340196, 0.08333333333333001, -0.08333333333333001], [0.19471724673529, 0.0, 0.0]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11815408810144001, 0.029413490244800007, -0.029413490244800007]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.11815408810144001, 0.029413490244800007, -0.029413490244800007], [0.20148742143477, 0.053919843088529984, -0.053919843088529984]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.20148742143477, 0.053919843088529984, -0.053919843088529984], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.13349436734026, 0.033172299326389994, -0.033172299326389994]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.13349436734026, 0.033172299326389994, -0.033172299326389994], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1275551936158, 0.0352666191995, -0.0352666191995]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1275551936158, 0.0352666191995, -0.0352666191995], [0.21088852694913, -0.01389674666574001, -0.048066714133829985]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.21088852694913, -0.01389674666574001, -0.048066714133829985], [0.29422186028246, -0.06559173281622999, -0.09723007999907002]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.29422186028246, -0.06559173281622999, -0.09723007999907002], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15513596307829, 0.0019422880197400016, -0.0019422880197400016]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15513596307829, 0.0019422880197400016, -0.0019422880197400016], [0.13310700640205003, 0.08527562135307001, 0.02008666865649999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.13310700640205003, 0.08527562135307001, 0.02008666865649999], [0.16860895468640003, 0.10342000198983001, 0.04977367306871999]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.16860895468640003, 0.10342000198983001, 0.04977367306871999], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1054418406202, 0.019299775828630003, -0.019299775828630003]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.1054418406202, 0.019299775828630003, -0.019299775828630003], [0.15017562229627002, 0.10263310916196001, -0.06403355750470001]],
    [[0.08333333333333, 0.08333333333333, -0.08333333333333], [0.15017562229627002, 0.10263310916196001, -0.06403355750470001], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625], [0.09375, 0.09375, -0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, -0.09375], [0.125, 0.125, -0.125]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.125, -0.125], [0.15625, 0.15625, -0.15625]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.15625, -0.15625], [0.1875, 0.1875, -0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.1875, -0.1875], [0.21875, 0.21875, -0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.21875, -0.21875], [0.25, 0.25, -0.25]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.09375, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, -0.03125], [0.125, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0, 0.0], [0.15625, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.03125, -0.03125], [0.1875, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.0, 0.0], [0.21875, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.03125, -0.03125], [0.25, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.25, 0.0, 0.0], [0.28125, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.28125, 0.03125, -0.03125], [0.3125, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.3125, 0.0, 0.0], [0.34375, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.34375, 0.03125, -0.03125], [0.375, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.375, 0.0, 0.0], [0.40625, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.40625, 0.03125, -0.03125], [0.4375, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.4375, 0.0, 0.0], [0.46875, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.46875, 0.03125, -0.03125], [0.5, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.03125, 0.03125, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, 0.03125], [0.0625, 0.0625, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, 0.0], [0.09375, 0.03125, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, 0.03125], [0.0625, 0.0625, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, 0.0625], [0.09375, 0.09375, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, 0.03125], [0.125, 0.0625, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0625, 0.0625], [0.09375, 0.09375, 0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, 0.09375], [0.125, 0.125, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.125, 0.0625], [0.15625, 0.09375, 0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.09375, 0.09375], [0.125, 0.125, 0.125]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.125, 0.125], [0.15625, 0.15625, 0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.15625, 0.09375], [0.1875, 0.125, 0.125]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.125, 0.125], [0.15625, 0.15625, 0.15625]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.15625, 0.15625], [0.1875, 0.1875, 0.125]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.1875, 0.125], [0.21875, 0.15625, 0.15625]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.15625, 0.15625], [0.1875, 0.1875, 0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.1875, 0.1875], [0.21875, 0.21875, 0.15625]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.21875, 0.15625], [0.25, 0.1875, 0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.25, 0.1875, 0.1875], [0.21875, 0.21875, 0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.21875, 0.21875], [0.25, 0.25, 0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.25, 0.25, 0.1875], [0.28125, 0.21875, 0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.28125, 0.21875, 0.21875], [0.25, 0.25, 0.25]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.03125, 0.03125, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, 0.03125], [0.0625, 0.0625, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, 0.0], [0.09375, 0.03125, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, 0.03125], [0.0625, 0.0625, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, 0.0625], [0.09375, 0.09375, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, 0.03125], [0.125, 0.0625, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0625, 0.0625], [0.09375, 0.09375, 0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, 0.09375], [0.125, 0.125, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.125, 0.0625], [0.15625, 0.09375, 0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.09375, 0.09375], [0.125, 0.125, 0.125]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.09375, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, -0.03125], [0.125, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0, 0.0], [0.15625, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.03125, -0.03125], [0.1875, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.0, 0.0], [0.21875, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.03125, -0.03125], [0.25, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.09375, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, -0.03125], [0.125, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0, 0.0], [0.15625, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.03125, -0.03125], [0.1875, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.0, 0.0], [0.21875, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.03125, -0.03125], [0.25, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.25, 0.0, 0.0], [0.28125, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.28125, 0.03125, -0.03125], [0.3125, 0.0625, -0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.3125, 0.0625, -0.0625], [0.34375, 0.09375, -0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.34375, 0.09375, -0.09375], [0.375, 0.125, -0.125]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.03125, 0.03125, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, 0.03125], [0.0625, 0.0625, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, 0.0], [0.09375, 0.03125, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, 0.03125], [0.0625, 0.0625, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, 0.0625], [0.09375, 0.09375, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, 0.03125], [0.125, 0.0625, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0625, 0.0625], [0.15625, 0.09375, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.09375, 0.03125], [0.1875, 0.0625, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.0625, 0.0625], [0.21875, 0.09375, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.09375, 0.03125], [0.25, 0.125, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.25, 0.125, 0.0], [0.28125, 0.15625, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.28125, 0.15625, -0.03125], [0.3125, 0.1875, -0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.04375, 0.006249999999999999, -0.006249999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.04375, 0.006249999999999999, -0.006249999999999999], [0.05, 0.03749999999999999, -0.012499999999999995]],
    [[0.03125, 0.03125, -0.03125], [0.05, 0.03749999999999999, -0.012499999999999995], [0.06875, 0.01875, 0.01875]],
    [[0.03125, 0.03125, -0.03125], [0.06875, 0.01875, 0.01875], [0.05, 0.05, 0.03749999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.05, 0.05, 0.03749999999999999], [0.08124999999999999, 0.06875, 0.018749999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.08124999999999999, 0.06875, 0.018749999999999996], [0.10000000000000002, 0.05, 0.05]],
    [[0.03125, 0.03125, -0.03125], [0.10000000000000002, 0.05, 0.05], [0.13125000000000003, 0.08125000000000002, 0.01874999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.13125000000000003, 0.08125000000000002, 0.01874999999999999], [0.16250000000000003, 0.05, 0.05]],
    [[0.03125, 0.03125, -0.03125], [0.16250000000000003, 0.05, 0.05], [0.19375000000000003, 0.08125000000000002, 0.01874999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.19375000000000003, 0.08125000000000002, 0.01874999999999999], [0.22500000000000003, 0.05, 0.05]],
    [[0.03125, 0.03125, -0.03125], [0.22500000000000003, 0.05, 0.05], [0.25625, 0.08125, 0.018750000000000003]],
    [[0.03125, 0.03125, -0.03125], [0.25625, 0.08125, 0.018750000000000003], [0.2875, 0.05, 0.05]],
    [[0.03125, 0.03125, -0.03125], [0.2875, 0.05, 0.05], [0.31875, 0.08125, 0.018750000000000003]],
    [[0.03125, 0.03125, -0.03125], [0.31875, 0.08125, 0.018750000000000003], [0.35, 0.05, 0.05]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.05625, 0.018749999999999996, -0.018749999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.05625, 0.018749999999999996, -0.018749999999999996], [0.0625, 0.0125, 0.0125]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0125, 0.0125], [0.04374999999999999, 0.04375, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.04374999999999999, 0.04375, 0.03125], [0.07500000000000001, 0.0625, 0.01249999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.07500000000000001, 0.0625, 0.01249999999999999], [0.09375000000000001, 0.04375, 0.04375]],
    [[0.03125, 0.03125, -0.03125], [0.09375000000000001, 0.04375, 0.04375], [0.07499999999999998, 0.075, 0.06250000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.07499999999999998, 0.075, 0.06250000000000001], [0.10624999999999998, 0.09375000000000001, 0.04375]],
    [[0.03125, 0.03125, -0.03125], [0.10624999999999998, 0.09375000000000001, 0.04375], [0.125, 0.075, 0.075]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.075, 0.075], [0.10624999999999998, 0.10625, 0.09375000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.10624999999999998, 0.10625, 0.09375000000000001], [0.1375, 0.1375, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.1375, 0.1375, 0.0625], [0.16875000000000004, 0.16875, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.16875000000000004, 0.16875, 0.03125], [0.20000000000000004, 0.2, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.05625, 0.018749999999999996, -0.018749999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.05625, 0.018749999999999996, -0.018749999999999996], [0.0875, 0.012499999999999997, -0.012499999999999997]],
    [[0.03125, 0.03125, -0.03125], [0.0875, 0.012499999999999997, -0.012499999999999997], [0.11875, 0.018749999999999996, -0.018749999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.11875, 0.018749999999999996, -0.018749999999999996], [0.15, 0.012500000000000011, -0.012500000000000011]],
    [[0.03125, 0.03125, -0.03125], [0.15, 0.012500000000000011, -0.012500000000000011], [0.18125, 0.01874999999999999, -0.01874999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.18125, 0.01874999999999999, -0.01874999999999999], [0.2125, 0.012500000000000011, -0.012500000000000011]],
    [[0.03125, 0.03125, -0.03125], [0.2125, 0.012500000000000011, -0.012500000000000011], [0.24375, 0.04374999999999998, -0.04374999999999998]],
    [[0.03125, 0.03125, -0.03125], [0.24375, 0.04374999999999998, -0.04374999999999998], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.04375, 0.006249999999999999, -0.006249999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.04375, 0.006249999999999999, -0.006249999999999999], [0.0375, 0.025, 0.012500000000000004]],
    [[0.03125, 0.03125, -0.03125], [0.0375, 0.025, 0.012500000000000004], [0.05625000000000001, 0.04375, 0.006249999999999995]],
    [[0.03125, 0.03125, -0.03125], [0.05625000000000001, 0.04375, 0.006249999999999995], [0.075, 0.03749999999999999, 0.02500000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.075, 0.03749999999999999, 0.02500000000000001], [0.06874999999999999, 0.05625, 0.043750000000000004]],
    [[0.03125, 0.03125, -0.03125], [0.06874999999999999, 0.05625, 0.043750000000000004], [0.09999999999999999, 0.075, 0.02500000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.09999999999999999, 0.075, 0.02500000000000001], [0.13124999999999998, 0.05625000000000001, 0.04375]],
    [[0.03125, 0.03125, -0.03125], [0.13124999999999998, 0.05625000000000001, 0.04375], [0.16249999999999998, 0.075, 0.02500000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.16249999999999998, 0.075, 0.02500000000000001], [0.19374999999999998, 0.05625000000000001, 0.04375]],
    [[0.03125, 0.03125, -0.03125], [0.19374999999999998, 0.05625000000000001, 0.04375], [0.22500000000000003, 0.075, 0.02500000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.22500000000000003, 0.075, 0.02500000000000001], [0.25625000000000003, 0.05625000000000001, 0.04375]],
    [[0.03125, 0.03125, -0.03125], [0.25625000000000003, 0.05625000000000001, 0.04375], [0.28750000000000003, 0.0875, 0.01249999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.28750000000000003, 0.0875, 0.01249999999999999], [0.31875000000000003, 0.11875, -0.01875000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.31875000000000003, 0.11875, -0.01875000000000001], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.05625, 0.018749999999999996, -0.018749999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.05625, 0.018749999999999996, -0.018749999999999996], [0.0625, 0.0125, 0.0125]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0125, 0.0125], [0.04374999999999999, 0.04375, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.04374999999999999, 0.04375, 0.03125], [0.07500000000000001, 0.0625, 0.01249999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.07500000000000001, 0.0625, 0.01249999999999999], [0.09375000000000001, 0.04375, 0.04375]],
    [[0.03125, 0.03125, -0.03125], [0.09375000000000001, 0.04375, 0.04375], [0.07499999999999998, 0.075, 0.06250000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.07499999999999998, 0.075, 0.06250000000000001], [0.10624999999999998, 0.09375000000000001, 0.04375]],
    [[0.03125, 0.03125, -0.03125], [0.10624999999999998, 0.09375000000000001, 0.04375], [0.125, 0.075, 0.075]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.075, 0.075], [0.10624999999999998, 0.10625, 0.09375000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.10624999999999998, 0.10625, 0.09375000000000001], [0.1375, 0.1375, 0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.1375, 0.1375, 0.0625], [0.16875000000000004, 0.16875, 0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.16875000000000004, 0.16875, 0.03125], [0.20000000000000004, 0.2, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.005, 0.005, -0.005]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.03625, 0.026250000000000002, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.03625, 0.026250000000000002, -0.026250000000000002], [0.0675, 0.057499999999999996, -0.057499999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.0675, 0.057499999999999996, -0.057499999999999996], [0.09875, 0.08875, -0.08875]],
    [[0.03125, 0.03125, -0.03125], [0.09875, 0.08875, -0.08875], [0.13, 0.12, -0.12]],
    [[0.03125, 0.03125, -0.03125], [0.13, 0.12, -0.12], [0.16125, 0.15125, -0.15125]],
    [[0.03125, 0.03125, -0.03125], [0.16125, 0.15125, -0.15125], [0.1925, 0.1825, -0.1825]],
    [[0.03125, 0.03125, -0.03125], [0.1925, 0.1825, -0.1825], [0.22375, 0.21375, -0.21375]],
    [[0.03125, 0.03125, -0.03125], [0.22375, 0.21375, -0.21375], [0.255, 0.245, -0.245]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.03625, 0.026250000000000002, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.03625, 0.026250000000000002, -0.026250000000000002], [0.0575, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.0575, 0.005, 0.005], [0.08875, 0.03625, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.08875, 0.03625, -0.026250000000000002], [0.12, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.12, 0.005, 0.005], [0.15125, 0.03625, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.15125, 0.03625, -0.026250000000000002], [0.1825, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.1825, 0.005, 0.005], [0.21375, 0.03625, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.21375, 0.03625, -0.026250000000000002], [0.245, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.245, 0.005, 0.005], [0.27625, 0.03625, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.27625, 0.03625, -0.026250000000000002], [0.3075, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.3075, 0.005, 0.005], [0.33875, 0.03625, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.33875, 0.03625, -0.026250000000000002], [0.37, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.37, 0.005, 0.005], [0.40125, 0.03625, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.40125, 0.03625, -0.026250000000000002], [0.4325, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.4325, 0.005, 0.005], [0.46375, 0.03625, -0.026250000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.46375, 0.03625, -0.026250000000000002], [0.495, 0.005, 0.005]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0575, 0.0049999999999999975, -0.0049999999999999975]],
    [[0.03125, 0.03125, -0.03125], [0.0575, 0.0049999999999999975, -0.0049999999999999975], [0.036250000000000004, 0.02625, 0.02625]],
    [[0.03125, 0.03125, -0.03125], [0.036250000000000004, 0.02625, 0.02625], [0.057499999999999996, 0.0575, 0.0049999999999999975]],
    [[0.03125, 0.03125, -0.03125], [0.057499999999999996, 0.0575, 0.0049999999999999975], [0.08875, 0.036250000000000004, 0.026249999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.08875, 0.036250000000000004, 0.026249999999999996], [0.0675, 0.0575, 0.0575]],
    [[0.03125, 0.03125, -0.03125], [0.0675, 0.0575, 0.0575], [0.08875, 0.08875, 0.036250000000000004]],
    [[0.03125, 0.03125, -0.03125], [0.08875, 0.08875, 0.036250000000000004], [0.12, 0.0675, 0.057499999999999996]],
    [[0.03125, 0.03125, -0.03125], [0.12, 0.0675, 0.057499999999999996], [0.09875, 0.08875, 0.08875]],
    [[0.03125, 0.03125, -0.03125], [0.09875, 0.08875, 0.08875], [0.12, 0.12, 0.0675]],
    [[0.03125, 0.03125, -0.03125], [0.12, 0.12, 0.0675], [0.15125, 0.09875, 0.08875]],
    [[0.03125, 0.03125, -0.03125], [0.15125, 0.09875, 0.08875], [0.13, 0.12, 0.12]],
    [[0.03125, 0.03125, -0.03125], [0.13, 0.12, 0.12], [0.15125, 0.15125, 0.09875]],
    [[0.03125, 0.03125, -0.03125], [0.15125, 0.15125, 0.09875], [0.1825, 0.13, 0.12]],
    [[0.03125, 0.03125, -0.03125], [0.1825, 0.13, 0.12], [0.16125, 0.15125, 0.15125]],
    [[0.03125, 0.03125, -0.03125], [0.16125, 0.15125, 0.15125], [0.1825, 0.1825, 0.13]],
    [[0.03125, 0.03125, -0.03125], [0.1825, 0.1825, 0.13], [0.21375, 0.16125, 0.15125]],
    [[0.03125, 0.03125, -0.03125], [0.21375, 0.16125, 0.15125], [0.1925, 0.1825, 0.1825]],
    [[0.03125, 0.03125, -0.03125], [0.1925, 0.1825, 0.1825], [0.21375, 0.21375, 0.16125]],
    [[0.03125, 0.03125, -0.03125], [0.21375, 0.21375, 0.16125], [0.245, 0.1925, 0.1825]],
    [[0.03125, 0.03125, -0.03125], [0.245, 0.1925, 0.1825], [0.22375, 0.21375, 0.21375]],
    [[0.03125, 0.03125, -0.03125], [0.22375, 0.21375, 0.21375], [0.245, 0.245, 0.1925]],
    [[0.03125, 0.03125, -0.03125], [0.245, 0.245, 0.1925], [0.27625, 0.22375, 0.21375]],
    [[0.03125, 0.03125, -0.03125], [0.27625, 0.22375, 0.21375], [0.255, 0.245, 0.245]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.009999999999999998, -0.009999999999999998]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.009999999999999998, -0.009999999999999998], [0.09375, 0.021250000000000005, -0.021250000000000005]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.021250000000000005, -0.021250000000000005], [0.125, 0.010000000000000002, -0.010000000000000002]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.010000000000000002, -0.010000000000000002], [0.15625, 0.02124999999999999, -0.02124999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.02124999999999999, -0.02124999999999999], [0.1875, 0.010000000000000009, -0.010000000000000009]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.010000000000000009, -0.010000000000000009], [0.21875, 0.02124999999999999, -0.02124999999999999]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.02124999999999999, -0.02124999999999999], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [5e-05, 5e-05, -5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0313, 0.0312, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.0313, 0.0312, -0.0312], [0.06255, 0.062450000000000006, -0.062450000000000006]],
    [[0.03125, 0.03125, -0.03125], [0.06255, 0.062450000000000006, -0.062450000000000006], [0.0938, 0.0937, -0.0937]],
    [[0.03125, 0.03125, -0.03125], [0.0938, 0.0937, -0.0937], [0.12505, 0.12495, -0.12495]],
    [[0.03125, 0.03125, -0.03125], [0.12505, 0.12495, -0.12495], [0.1563, 0.1562, -0.1562]],
    [[0.03125, 0.03125, -0.03125], [0.1563, 0.1562, -0.1562], [0.18755, 0.18745, -0.18745]],
    [[0.03125, 0.03125, -0.03125], [0.18755, 0.18745, -0.18745], [0.2188, 0.2187, -0.2187]],
    [[0.03125, 0.03125, -0.03125], [0.2188, 0.2187, -0.2187], [0.25005, 0.24995, -0.24995]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0313, 0.0312, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.0313, 0.0312, -0.0312], [0.06245, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.06245, 5e-05, 5e-05], [0.0937, 0.0313, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.0937, 0.0313, -0.0312], [0.12495, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.12495, 5e-05, 5e-05], [0.1562, 0.0313, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.1562, 0.0313, -0.0312], [0.18745, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.18745, 5e-05, 5e-05], [0.2187, 0.0313, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.2187, 0.0313, -0.0312], [0.24995, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.24995, 5e-05, 5e-05], [0.2812, 0.0313, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.2812, 0.0313, -0.0312], [0.31245, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.31245, 5e-05, 5e-05], [0.3437, 0.0313, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.3437, 0.0313, -0.0312], [0.37495, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.37495, 5e-05, 5e-05], [0.4062, 0.0313, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.4062, 0.0313, -0.0312], [0.43745, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.43745, 5e-05, 5e-05], [0.4687, 0.0313, -0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.4687, 0.0313, -0.0312], [0.49995, 5e-05, 5e-05]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.06245, 5.000000000000143e-05, -5.000000000000143e-05]],
    [[0.03125, 0.03125, -0.03125], [0.06245, 5.000000000000143e-05, -5.000000000000143e-05], [0.0313, 0.0312, 0.0312]],
    [[0.03125, 0.03125, -0.03125], [0.0313, 0.0312, 0.0312], [0.062450000000000006, 0.06245, 5.000000000000143e-05]],
    [[0.03125, 0.03125, -0.03125], [0.062450000000000006, 0.06245, 5.000000000000143e-05], [0.0937, 0.031299999999999994, 0.031200000000000006]],
    [[0.03125, 0.03125, -0.03125], [0.0937, 0.031299999999999994, 0.031200000000000006], [0.06255, 0.06245, 0.06245]],
    [[0.03125, 0.03125, -0.03125], [0.06255, 0.06245, 0.06245], [0.0937, 0.0937, 0.031299999999999994]],
    [[0.03125, 0.03125, -0.03125], [0.0937, 0.0937, 0.031299999999999994], [0.12495, 0.06255, 0.062450000000000006]],
    [[0.03125, 0.03125, -0.03125], [0.12495, 0.06255, 0.062450000000000006], [0.0938, 0.0937, 0.0937]],
    [[0.03125, 0.03125, -0.03125], [0.0938, 0.0937, 0.0937], [0.12495, 0.12495, 0.06255]],
    [[0.03125, 0.03125, -0.03125], [0.12495, 0.12495, 0.06255], [0.1562, 0.0938, 0.0937]],
    [[0.03125, 0.03125, -0.03125], [0.1562, 0.0938, 0.0937], [0.12505, 0.12495, 0.12495]],
    [[0.03125, 0.03125, -0.03125], [0.12505, 0.12495, 0.12495], [0.1562, 0.1562, 0.0938]],
    [[0.03125, 0.03125, -0.03125], [0.1562, 0.1562, 0.0938], [0.18745, 0.12505, 0.12495]],
    [[0.03125, 0.03125, -0.03125], [0.18745, 0.12505, 0.12495], [0.1563, 0.1562, 0.1562]],
    [[0.03125, 0.03125, -0.03125], [0.1563, 0.1562, 0.1562], [0.18745, 0.18745, 0.12505]],
    [[0.03125, 0.03125, -0.03125], [0.18745, 0.18745, 0.12505], [0.2187, 0.1563, 0.1562]],
    [[0.03125, 0.03125, -0.03125], [0.2187, 0.1563, 0.1562], [0.18755, 0.18745, 0.18745]],
    [[0.03125, 0.03125, -0.03125], [0.18755, 0.18745, 0.18745], [0.2187, 0.2187, 0.1563]],
    [[0.03125, 0.03125, -0.03125], [0.2187, 0.2187, 0.1563], [0.24995, 0.18755, 0.18745]],
    [[0.03125, 0.03125, -0.03125], [0.24995, 0.18755, 0.18745], [0.2188, 0.2187, 0.2187]],
    [[0.03125, 0.03125, -0.03125], [0.2188, 0.2187, 0.2187], [0.24995, 0.24995, 0.18755]],
    [[0.03125, 0.03125, -0.03125], [0.24995, 0.24995, 0.18755], [0.2812, 0.2188, 0.2187]],
    [[0.03125, 0.03125, -0.03125], [0.2812, 0.2188, 0.2187], [0.25005, 0.24995, 0.24995]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.00010000000000000286, -0.00010000000000000286]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.00010000000000000286, -0.00010000000000000286], [0.09375, 0.031149999999999997, -0.031149999999999997]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.031149999999999997, -0.031149999999999997], [0.125, 9.999999999999593e-05, -9.999999999999593e-05]],
    [[0.03125, 0.03125, -0.03125], [0.125, 9.999999999999593e-05, -9.999999999999593e-05], [0.15625, 0.03115000000000001, -0.03115000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.03115000000000001, -0.03115000000000001], [0.1875, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 9.999999999998899e-05, -9.999999999998899e-05], [0.21875, 0.03115000000000001, -0.03115000000000001]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.03115000000000001, -0.03115000000000001], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [5e-07, 5e-07, -5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0312505, 0.0312495, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.0312505, 0.0312495, -0.0312495], [0.0625005, 0.0624995, -0.0624995]],
    [[0.03125, 0.03125, -0.03125], [0.0625005, 0.0624995, -0.0624995], [0.0937505, 0.0937495, -0.0937495]],
    [[0.03125, 0.03125, -0.03125], [0.0937505, 0.0937495, -0.0937495], [0.1250005, 0.12499950000000001, -0.12499950000000001]],
    [[0.03125, 0.03125, -0.03125], [0.1250005, 0.12499950000000001, -0.12499950000000001], [0.1562505, 0.1562495, -0.1562495]],
    [[0.03125, 0.03125, -0.03125], [0.1562505, 0.1562495, -0.1562495], [0.1875005, 0.1874995, -0.1874995]],
    [[0.03125, 0.03125, -0.03125], [0.1875005, 0.1874995, -0.1874995], [0.2187505, 0.2187495, -0.2187495]],
    [[0.03125, 0.03125, -0.03125], [0.2187505, 0.2187495, -0.2187495], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0312505, 0.0312495, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.0312505, 0.0312495, -0.0312495], [0.0624995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.0624995, 5e-07, 5e-07], [0.0937495, 0.0312505, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.0937495, 0.0312505, -0.0312495], [0.1249995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.1249995, 5e-07, 5e-07], [0.15624949999999999, 0.0312505, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.15624949999999999, 0.0312505, -0.0312495], [0.1874995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.1874995, 5e-07, 5e-07], [0.21874949999999999, 0.0312505, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.21874949999999999, 0.0312505, -0.0312495], [0.2499995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.2499995, 5e-07, 5e-07], [0.2812495, 0.0312505, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.2812495, 0.0312505, -0.0312495], [0.3124995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.3124995, 5e-07, 5e-07], [0.3437495, 0.0312505, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.3437495, 0.0312505, -0.0312495], [0.3749995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.3749995, 5e-07, 5e-07], [0.4062495, 0.0312505, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.4062495, 0.0312505, -0.0312495], [0.4374995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.4374995, 5e-07, 5e-07], [0.4687495, 0.0312505, -0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.4687495, 0.0312505, -0.0312495], [0.4999995, 5e-07, 5e-07]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0624995, 5.000000000005e-07, -5.000000000005e-07]],
    [[0.03125, 0.03125, -0.03125], [0.0624995, 5.000000000005e-07, -5.000000000005e-07], [0.0312505, 0.0312495, 0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.0312505, 0.0312495, 0.0312495], [0.0624995, 0.0624995, 5.000000000005e-07]],
    [[0.03125, 0.03125, -0.03125], [0.0624995, 0.0624995, 5.000000000005e-07], [0.0937495, 0.0312505, 0.0312495]],
    [[0.03125, 0.03125, -0.03125], [0.0937495, 0.0312505, 0.0312495], [0.0625005, 0.0624995, 0.0624995]],
    [[0.03125, 0.03125, -0.03125], [0.0625005, 0.0624995, 0.0624995], [0.09374949999999999, 0.0937495, 0.0312505]],
    [[0.03125, 0.03125, -0.03125], [0.09374949999999999, 0.0937495, 0.0312505], [0.12499949999999999, 0.0625005, 0.0624995]],
    [[0.03125, 0.03125, -0.03125], [0.12499949999999999, 0.0625005, 0.0624995], [0.0937505, 0.0937495, 0.0937495]],
    [[0.03125, 0.03125, -0.03125], [0.0937505, 0.0937495, 0.0937495], [0.12499949999999999, 0.1249995, 0.0625005]],
    [[0.03125, 0.03125, -0.03125], [0.12499949999999999, 0.1249995, 0.0625005], [0.1562495, 0.09375049999999999, 0.09374950000000001]],
    [[0.03125, 0.03125, -0.03125], [0.1562495, 0.09375049999999999, 0.09374950000000001], [0.12500050000000001, 0.1249995, 0.1249995]],
    [[0.03125, 0.03125, -0.03125], [0.12500050000000001, 0.1249995, 0.1249995], [0.15624950000000004, 0.1562495, 0.09375049999999999]],
    [[0.03125, 0.03125, -0.03125], [0.15624950000000004, 0.1562495, 0.09375049999999999], [0.18749950000000004, 0.1250005, 0.12499950000000001]],
    [[0.03125, 0.03125, -0.03125], [0.18749950000000004, 0.1250005, 0.12499950000000001], [0.1562505, 0.1562495, 0.1562495]],
    [[0.03125, 0.03125, -0.03125], [0.1562505, 0.1562495, 0.1562495], [0.18749950000000004, 0.1874995, 0.1250005]],
    [[0.03125, 0.03125, -0.03125], [0.18749950000000004, 0.1874995, 0.1250005], [0.21874950000000004, 0.1562505, 0.1562495]],
    [[0.03125, 0.03125, -0.03125], [0.21874950000000004, 0.1562505, 0.1562495], [0.1875005, 0.1874995, 0.1874995]],
    [[0.03125, 0.03125, -0.03125], [0.1875005, 0.1874995, 0.1874995], [0.21874950000000004, 0.2187495, 0.1562505]],
    [[0.03125, 0.03125, -0.03125], [0.21874950000000004, 0.2187495, 0.1562505], [0.24999950000000004, 0.1875005, 0.1874995]],
    [[0.03125, 0.03125, -0.03125], [0.24999950000000004, 0.1875005, 0.1874995], [0.2187505, 0.2187495, 0.2187495]],
    [[0.03125, 0.03125, -0.03125], [0.2187505, 0.2187495, 0.2187495], [0.24999950000000004, 0.2499995, 0.1875005]],
    [[0.03125, 0.03125, -0.03125], [0.24999950000000004, 0.2499995, 0.1875005], [0.2812495, 0.21875050000000001, 0.21874949999999999]],
    [[0.03125, 0.03125, -0.03125], [0.2812495, 0.21875050000000001, 0.21874949999999999], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 1.000000000001e-06, -1.000000000001e-06]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 1.000000000001e-06, -1.000000000001e-06], [0.09375, 0.031249, -0.031249]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.031249, -0.031249], [0.125, 1.000000000001e-06, -1.000000000001e-06]],
    [[0.03125, 0.03125, -0.03125], [0.125, 1.000000000001e-06, -1.000000000001e-06], [0.15625, 0.031249, -0.031249]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.031249, -0.031249], [0.1875, 1.000000000001e-06, -1.000000000001e-06]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 1.000000000001e-06, -1.000000000001e-06], [0.21875, 0.031249, -0.031249]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.031249, -0.031249], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [5e-11, 5e-11, -5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625], [0.09375, 0.09375, -0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, -0.09375], [0.125, 0.125, -0.125]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.125, -0.125], [0.15625, 0.15625, -0.15625]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.15625, -0.15625], [0.1875, 0.1875, -0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.1875, -0.1875], [0.21875, 0.21875, -0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.21875, -0.21875], [0.25000000005, 0.24999999995, -0.24999999995]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.06249999995, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.06249999995, 0.0, 0.0], [0.09374999995, 0.03125000005, -0.031249999950000003]],
    [[0.03125, 0.03125, -0.03125], [0.09374999995, 0.03125000005, -0.031249999950000003], [0.12499999995, 5e-11, 5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.12499999995, 5e-11, 5e-11], [0.15624999995, 0.03125000005, -0.031249999950000003]],
    [[0.03125, 0.03125, -0.03125], [0.15624999995, 0.03125000005, -0.031249999950000003], [0.18749999995, 5e-11, 5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.18749999995, 5e-11, 5e-11], [0.21874999995, 0.03125000005, -0.031249999950000003]],
    [[0.03125, 0.03125, -0.03125], [0.21874999995, 0.03125000005, -0.031249999950000003], [0.24999999995, 5e-11, 5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.24999999995, 5e-11, 5e-11], [0.28124999995, 0.03125000005, -0.031249999950000003]],
    [[0.03125, 0.03125, -0.03125], [0.28124999995, 0.03125000005, -0.031249999950000003], [0.31249999995, 5e-11, 5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.31249999995, 5e-11, 5e-11], [0.34374999995, 0.03125000005, -0.031249999950000003]],
    [[0.03125, 0.03125, -0.03125], [0.34374999995, 0.03125000005, -0.031249999950000003], [0.37499999995, 5e-11, 5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.37499999995, 5e-11, 5e-11], [0.40624999995, 0.03125000005, -0.031249999950000003]],
    [[0.03125, 0.03125, -0.03125], [0.40624999995, 0.03125000005, -0.031249999950000003], [0.43749999995, 5e-11, 5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.43749999995, 5e-11, 5e-11], [0.46874999995, 0.03125000005, -0.031249999950000003]],
    [[0.03125, 0.03125, -0.03125], [0.46874999995, 0.03125000005, -0.031249999950000003], [0.49999999995, 5e-11, 5e-11]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.06249999998333, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.06249999998333, 0.0, 0.0], [0.03124999998333, 0.03124999998333, 0.03124999998333]],
    [[0.03125, 0.03125, -0.03125], [0.03124999998333, 0.03124999998333, 0.03124999998333], [0.06249999998333, 0.06249999998333, -1.6669998714746725e-11]],
    [[0.03125, 0.03125, -0.03125], [0.06249999998333, 0.06249999998333, -1.6669998714746725e-11], [0.09374999998333, 0.03124999998333, 0.03124999998333]],
    [[0.03125, 0.03125, -0.03125], [0.09374999998333, 0.03124999998333, 0.03124999998333], [0.06249999998333, 0.06249999998333, 0.06249999998333]],
    [[0.03125, 0.03125, -0.03125], [0.06249999998333, 0.06249999998333, 0.06249999998333], [0.09374999998333, 0.09374999998333, 0.03124999998333]],
    [[0.03125, 0.03125, -0.03125], [0.09374999998333, 0.09374999998333, 0.03124999998333], [0.12499999998333, 0.06249999998333, 0.06249999998333]],
    [[0.03125, 0.03125, -0.03125], [0.12499999998333, 0.06249999998333, 0.06249999998333], [0.09374999998333, 0.09374999998333, 0.09374999998333]],
    [[0.03125, 0.03125, -0.03125], [0.09374999998333, 0.09374999998333, 0.09374999998333], [0.12499999998333, 0.12499999998333, 0.06249999998333]],
    [[0.03125, 0.03125, -0.03125], [0.12499999998333, 0.12499999998333, 0.06249999998333], [0.15624999998333, 0.09374999998333, 0.09374999998333]],
    [[0.03125, 0.03125, -0.03125], [0.15624999998333, 0.09374999998333, 0.09374999998333], [0.12499999998333, 0.12499999998333, 0.12499999998333]],
    [[0.03125, 0.03125, -0.03125], [0.12499999998333, 0.12499999998333, 0.12499999998333], [0.15624999998333, 0.15624999998333, 0.09374999998333]],
    [[0.03125, 0.03125, -0.03125], [0.15624999998333, 0.15624999998333, 0.09374999998333], [0.18749999998333, 0.12499999998333, 0.12499999998333]],
    [[0.03125, 0.03125, -0.03125], [0.18749999998333, 0.12499999998333, 0.12499999998333], [0.15624999998333, 0.15624999998333, 0.15624999998333]],
    [[0.03125, 0.03125, -0.03125], [0.15624999998333, 0.15624999998333, 0.15624999998333], [0.18749999998333, 0.18749999998333, 0.12499999998333]],
    [[0.03125, 0.03125, -0.03125], [0.18749999998333, 0.18749999998333, 0.12499999998333], [0.21874999998333, 0.15624999998333, 0.15624999998333]],
    [[0.03125, 0.03125, -0.03125], [0.21874999998333, 0.15624999998333, 0.15624999998333], [0.18749999998333, 0.18749999998333, 0.18749999998333]],
    [[0.03125, 0.03125, -0.03125], [0.18749999998333, 0.18749999998333, 0.18749999998333], [0.21874999998333, 0.21874999998333, 0.15624999998333]],
    [[0.03125, 0.03125, -0.03125], [0.21874999998333, 0.21874999998333, 0.15624999998333], [0.24999999998333, 0.18749999998333, 0.18749999998333]],
    [[0.03125, 0.03125, -0.03125], [0.24999999998333, 0.18749999998333, 0.18749999998333], [0.21874999998333, 0.21874999998333, 0.21874999998333]],
    [[0.03125, 0.03125, -0.03125], [0.21874999998333, 0.21874999998333, 0.21874999998333], [0.249999999975, 0.249999999975, 0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.249999999975, 0.249999999975, 0.1875], [0.28124999995, 0.21875, 0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.28124999995, 0.21875, 0.21875], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.09375, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, -0.03125], [0.125, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0, 0.0], [0.15625, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.03125, -0.03125], [0.1875, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.0, 0.0], [0.21875, 0.03124999989999999, -0.03124999989999999]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.03124999989999999, -0.03124999989999999], [0.25, 1.000000082740371e-10, -1.000000082740371e-10]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625], [0.09375, 0.09375, -0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, -0.09375], [0.125, 0.125, -0.125]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.125, -0.125], [0.15625, 0.15625, -0.15625]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.15625, -0.15625], [0.1875, 0.1875, -0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.1875, -0.1875], [0.21875, 0.21875, -0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.21875, -0.21875], [0.2500000000005, 0.2499999999995, -0.2499999999995]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0624999999995, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0624999999995, 0.0, 0.0], [0.0937499999995, 0.0312500000005, -0.031249999999499997]],
    [[0.03125, 0.03125, -0.03125], [0.0937499999995, 0.0312500000005, -0.031249999999499997], [0.1249999999995, 5e-13, 5e-13]],
    [[0.03125, 0.03125, -0.03125], [0.1249999999995, 5e-13, 5e-13], [0.1562499999995, 0.0312500000005, -0.031249999999499997]],
    [[0.03125, 0.03125, -0.03125], [0.1562499999995, 0.0312500000005, -0.031249999999499997], [0.1874999999995, 5e-13, 5e-13]],
    [[0.03125, 0.03125, -0.03125], [0.1874999999995, 5e-13, 5e-13], [0.2187499999995, 0.0312500000005, -0.031249999999499997]],
    [[0.03125, 0.03125, -0.03125], [0.2187499999995, 0.0312500000005, -0.031249999999499997], [0.2499999999995, 5e-13, 5e-13]],
    [[0.03125, 0.03125, -0.03125], [0.2499999999995, 5e-13, 5e-13], [0.2812499999995, 0.0312500000005, -0.031249999999499997]],
    [[0.03125, 0.03125, -0.03125], [0.2812499999995, 0.0312500000005, -0.031249999999499997], [0.3124999999995, 5e-13, 5e-13]],
    [[0.03125, 0.03125, -0.03125], [0.3124999999995, 5e-13, 5e-13], [0.3437499999995, 0.0312500000005, -0.031249999999499997]],
    [[0.03125, 0.03125, -0.03125], [0.3437499999995, 0.0312500000005, -0.031249999999499997], [0.3749999999995, 5e-13, 5e-13]],
    [[0.03125, 0.03125, -0.03125], [0.3749999999995, 5e-13, 5e-13], [0.4062499999995, 0.0312500000005, -0.031249999999499997]],
    [[0.03125, 0.03125, -0.03125], [0.4062499999995, 0.0312500000005, -0.031249999999499997], [0.4374999999995, 5e-13, 5e-13]],
    [[0.03125, 0.03125, -0.03125], [0.4374999999995, 5e-13, 5e-13], [0.4687499999995, 0.0312500000005, -0.031249999999499997]],
    [[0.03125, 0.03125, -0.03125], [0.4687499999995, 0.0312500000005, -0.031249999999499997], [0.4999999999995, 5e-13, 5e-13]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.06249999999983, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.06249999999983, 0.0, 0.0], [0.03124999999983, 0.03124999999983, 0.03124999999983]],
    [[0.03125, 0.03125, -0.03125], [0.03124999999983, 0.03124999999983, 0.03124999999983], [0.06249999999983, 0.06249999999983, -1.700029006457271e-13]],
    [[0.03125, 0.03125, -0.03125], [0.06249999999983, 0.06249999999983, -1.700029006457271e-13], [0.09374999999983, 0.03124999999983, 0.03124999999983]],
    [[0.03125, 0.03125, -0.03125], [0.09374999999983, 0.03124999999983, 0.03124999999983], [0.06249999999983, 0.06249999999983, 0.06249999999983]],
    [[0.03125, 0.03125, -0.03125], [0.06249999999983, 0.06249999999983, 0.06249999999983], [0.09374999999983, 0.09374999999983, 0.031249999999829997]],
    [[0.03125, 0.03125, -0.03125], [0.09374999999983, 0.09374999999983, 0.031249999999829997], [0.12499999999983, 0.06249999999983, 0.06249999999983]],
    [[0.03125, 0.03125, -0.03125], [0.12499999999983, 0.06249999999983, 0.06249999999983], [0.09374999999983, 0.09374999999983, 0.09374999999983]],
    [[0.03125, 0.03125, -0.03125], [0.09374999999983, 0.09374999999983, 0.09374999999983], [0.12499999999983, 0.12499999999983, 0.06249999999983]],
    [[0.03125, 0.03125, -0.03125], [0.12499999999983, 0.12499999999983, 0.06249999999983], [0.15624999999983, 0.09374999999983, 0.09374999999983]],
    [[0.03125, 0.03125, -0.03125], [0.15624999999983, 0.09374999999983, 0.09374999999983], [0.12499999999983, 0.12499999999983, 0.12499999999983]],
    [[0.03125, 0.03125, -0.03125], [0.12499999999983, 0.12499999999983, 0.12499999999983], [0.15624999999983, 0.15624999999983, 0.09374999999983]],
    [[0.03125, 0.03125, -0.03125], [0.15624999999983, 0.15624999999983, 0.09374999999983], [0.18749999999983, 0.12499999999983, 0.12499999999983]],
    [[0.03125, 0.03125, -0.03125], [0.18749999999983, 0.12499999999983, 0.12499999999983], [0.15624999999983, 0.15624999999983, 0.15624999999983]],
    [[0.03125, 0.03125, -0.03125], [0.15624999999983, 0.15624999999983, 0.15624999999983], [0.18749999999983, 0.18749999999983, 0.12499999999983]],
    [[0.03125, 0.03125, -0.03125], [0.18749999999983, 0.18749999999983, 0.12499999999983], [0.21874999999983, 0.15624999999983, 0.15624999999983]],
    [[0.03125, 0.03125, -0.03125], [0.21874999999983, 0.15624999999983, 0.15624999999983], [0.18749999999983, 0.18749999999983, 0.18749999999983]],
    [[0.03125, 0.03125, -0.03125], [0.18749999999983, 0.18749999999983, 0.18749999999983], [0.21874999999983, 0.21874999999983, 0.15624999999983]],
    [[0.03125, 0.03125, -0.03125], [0.21874999999983, 0.21874999999983, 0.15624999999983], [0.24999999999983, 0.18749999999983, 0.18749999999983]],
    [[0.03125, 0.03125, -0.03125], [0.24999999999983, 0.18749999999983, 0.18749999999983], [0.21874999999983, 0.21874999999983, 0.21874999999983]],
    [[0.03125, 0.03125, -0.03125], [0.21874999999983, 0.21874999999983, 0.21874999999983], [0.24999999999974998, 0.24999999999975, 0.18749999999999997]],
    [[0.03125, 0.03125, -0.03125], [0.24999999999974998, 0.24999999999975, 0.18749999999999997], [0.2812499999995, 0.21875, 0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.2812499999995, 0.21875, 0.21875], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.09375, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, -0.03125], [0.125, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0, 0.0], [0.15625, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.03125, -0.03125], [0.1875, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.0, 0.0], [0.21875, 0.031249999998999994, -0.031249999998999994]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.031249999998999994, -0.031249999998999994], [0.25, 9.999917560676863e-13, -9.999917560676863e-13]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0625, -0.0625], [0.09375, 0.09375, -0.09375]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.09375, -0.09375], [0.125, 0.125, -0.125]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.125, -0.125], [0.15625, 0.15625, -0.15625]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.15625, -0.15625], [0.1875, 0.1875, -0.1875]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.1875, -0.1875], [0.21875, 0.21875, -0.21875]],
    [[0.03125, 0.03125, -0.03125], [0.21875, 0.21875, -0.21875], [0.250000005, 0.24999999499999997, -0.24999999499999997]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.062499995, 5e-09, 5e-09]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.06249999833333, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.06249999833333, 0.0, 0.0], [0.03124999833333, 0.03124999833333, 0.03124999833333]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.0625, 0.0, 0.0], [0.09375, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.09375, 0.03125, -0.03125], [0.125, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.125, 0.0, 0.0], [0.15625, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.15625, 0.03125, -0.03125], [0.1875, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.1875, 0.0, 0.0], [0.21875, 0.031249990000000005, -0.031249990000000005]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.05803877549271, 0.058035918366179995, -0.058035918366179995]],
    [[0.03125, 0.03125, -0.03125], [0.05803877549271, 0.058035918366179995, -0.058035918366179995], [0.08928591836618, 0.08928591836618, -0.08928306123965002]],
    [[0.03125, 0.03125, -0.03125], [0.08928591836618, 0.08928591836618, -0.08928306123965002], [0.12053591836618, 0.12053591836618, -0.12053306123965002]],
    [[0.03125, 0.03125, -0.03125], [0.12053591836618, 0.12053591836618, -0.12053306123965002], [0.15178591836618, 0.15178591836618, -0.15178306123964996]],
    [[0.03125, 0.03125, -0.03125], [0.15178591836618, 0.15178591836618, -0.15178306123964996], [0.18303591836618, 0.18303591836618, -0.18303306123964996]],
    [[0.03125, 0.03125, -0.03125], [0.18303591836618, 0.18303591836618, -0.18303306123964996], [0.21428591836618, 0.21428591836618, -0.21428306123964996]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.048996289301620005, 0.019179755378640004, -0.019179755378640004]],
    [[0.03125, 0.03125, -0.03125], [0.048996289301620005, 0.019179755378640004, -0.019179755378640004], [0.07167608289204, 0.05042975537864, -0.041859548969059995]],
    [[0.03125, 0.03125, -0.03125], [0.07167608289204, 0.05042975537864, -0.041859548969059995], [0.10292608289204, 0.08167975537864, -0.07310954896906]],
    [[0.03125, 0.03125, -0.03125], [0.10292608289204, 0.08167975537864, -0.07310954896906], [0.13417608289204, 0.11292975537864, -0.10435954896906]],
    [[0.03125, 0.03125, -0.03125], [0.13417608289204, 0.11292975537864, -0.10435954896906], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.03368673373581, 0.0054765013079599995, -0.0054765013079599995]],
    [[0.03125, 0.03125, -0.03125], [0.03368673373581, 0.0054765013079599995, -0.0054765013079599995], [0.06307556066818, -0.0024367337358099997, -0.02391232562441]],
    [[0.03125, 0.03125, -0.03125], [0.06307556066818, -0.0024367337358099997, -0.02391232562441], [0.09432556066818, -0.0054765013079599995, -0.03368673373581]],
    [[0.03125, 0.03125, -0.03125], [0.09432556066818, -0.0054765013079599995, -0.03368673373581], [0.12557556066818, -0.023912325624410005, -0.03672650130796]],
    [[0.03125, 0.03125, -0.03125], [0.12557556066818, -0.023912325624410005, -0.03672650130796], [0.15682556066818, -0.03368673373580999, -0.05516232562441001]],
    [[0.03125, 0.03125, -0.03125], [0.15682556066818, -0.03368673373580999, -0.05516232562441001], [0.18807556066818, -0.036726501307960006, -0.06493673373580999]],
    [[0.03125, 0.03125, -0.03125], [0.18807556066818, -0.036726501307960006, -0.06493673373580999], [0.21932556066818, -0.055162325624409994, -0.06797650130796]],
    [[0.03125, 0.03125, -0.03125], [0.21932556066818, -0.055162325624409994, -0.06797650130796], [0.25057556066818, -0.06493673373581, -0.08641232562441]],
    [[0.03125, 0.03125, -0.03125], [0.25057556066818, -0.06493673373581, -0.08641232562441], [0.28182556066818, -0.06797650130796001, -0.09618673373580999]],
    [[0.03125, 0.03125, -0.03125], [0.28182556066818, -0.06797650130796001, -0.09618673373580999], [0.31307556066818, -0.08641232562440998, -0.09922650130796001]],
    [[0.03125, 0.03125, -0.03125], [0.31307556066818, -0.08641232562440998, -0.09922650130796001], [0.34432556066818, -0.055162325624409994, -0.13047650130796]],
    [[0.03125, 0.03125, -0.03125], [0.34432556066818, -0.055162325624409994, -0.13047650130796], [0.37557556066818, -0.023912325624409994, -0.16172650130796]],
    [[0.03125, 0.03125, -0.03125], [0.37557556066818, -0.023912325624409994, -0.16172650130796], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0460391923974, 0.0460391923974, -0.0460391923974]],
    [[0.03125, 0.03125, -0.03125], [0.0460391923974, 0.0460391923974, -0.0460391923974], [0.0772891923974, 0.0772891923974, -0.0772891923974]],
    [[0.03125, 0.03125, -0.03125], [0.0772891923974, 0.0772891923974, -0.0772891923974], [0.1085391923974, 0.1085391923974, -0.1085391923974]],
    [[0.03125, 0.03125, -0.03125], [0.1085391923974, 0.1085391923974, -0.1085391923974], [0.1397891923974, 0.1397891923974, -0.1397891923974]],
    [[0.03125, 0.03125, -0.03125], [0.1397891923974, 0.1397891923974, -0.1397891923974], [0.1710391923974, 0.1710391923974, -0.1710391923974]],
    [[0.03125, 0.03125, -0.03125], [0.1710391923974, 0.1710391923974, -0.1710391923974], [0.2022891923974, 0.2022891923974, -0.2022891923974]],
    [[0.03125, 0.03125, -0.03125], [0.2022891923974, 0.2022891923974, -0.2022891923974], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.03904638617449, 2.0218527699983335e-06, -2.0218527699983335e-06]],
    [[0.03125, 0.03125, -0.03125], [0.03904638617449, 2.0218527699983335e-06, -2.0218527699983335e-06], [0.05420297540008, 0.03124797814723, -0.015154567372819998]],
    [[0.03125, 0.03125, -0.03125], [0.05420297540008, 0.03124797814723, -0.015154567372819998], [0.08545297540007998, 0.016095432627179995, -2.021852769996599e-06]],
    [[0.03125, 0.03125, -0.03125], [0.08545297540007998, 0.016095432627179995, -2.021852769996599e-06], [0.11670297540007998, 0.031247978147230005, -0.015154567372820001]],
    [[0.03125, 0.03125, -0.03125], [0.11670297540007998, 0.031247978147230005, -0.015154567372820001], [0.14795297540007998, 0.016095432627180002, -2.0218527700035377e-06]],
    [[0.03125, 0.03125, -0.03125], [0.14795297540007998, 0.016095432627180002, -2.0218527700035377e-06], [0.17920297540008, 0.031247978147230012, -0.015154567372820008]],
    [[0.03125, 0.03125, -0.03125], [0.17920297540008, 0.031247978147230012, -0.015154567372820008], [0.21045297540007998, 0.016095432627180002, -2.0218527700035377e-06]],
    [[0.03125, 0.03125, -0.03125], [0.21045297540007998, 0.016095432627180002, -2.0218527700035377e-06], [0.24170297540008, 0.047345432627179995, -0.031252021852769984]],
    [[0.03125, 0.03125, -0.03125], [0.24170297540008, 0.047345432627179995, -0.031252021852769984], [0.27295297540008, 0.07859543262718, -0.06250202185276998]],
    [[0.03125, 0.03125, -0.03125], [0.27295297540008, 0.07859543262718, -0.06250202185276998], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.038467246735290006, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.038467246735290006, 0.03125, -0.03125], [0.06971724673529, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.06971724673529, 0.0, 0.0], [0.10096724673529, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.10096724673529, 0.03125, -0.03125], [0.13221724673529, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.13221724673529, 0.0, 0.0], [0.16346724673529, 0.03125, -0.03125]],
    [[0.03125, 0.03125, -0.03125], [0.16346724673529, 0.03125, -0.03125], [0.19471724673529, 0.0, 0.0]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0348207547681, 0.012253176421860001, -0.012253176421860001]],
    [[0.03125, 0.03125, -0.03125], [0.0348207547681, 0.012253176421860001, -0.012253176421860001], [0.0660707547681, 0.018996823578140004, -0.018996823578140004]],
    [[0.03125, 0.03125, -0.03125], [0.0660707547681, 0.018996823578140004, -0.018996823578140004], [0.0973207547681, 0.012253176421859996, -0.012253176421859996]],
    [[0.03125, 0.03125, -0.03125], [0.0973207547681, 0.012253176421859996, -0.012253176421859996], [0.1285707547681, 0.018996823578140004, -0.018996823578140004]],
    [[0.03125, 0.03125, -0.03125], [0.1285707547681, 0.018996823578140004, -0.018996823578140004], [0.1598207547681, 0.012253176421859996, -0.012253176421859996]],
    [[0.03125, 0.03125, -0.03125], [0.1598207547681, 0.012253176421859996, -0.012253176421859996], [0.1910707547681, 0.043503176421859996, -0.043503176421859996]],
    [[0.03125, 0.03125, -0.03125], [0.1910707547681, 0.043503176421859996, -0.043503176421859996], [0.2223207547681, 0.07475317642186, -0.07475317642186]],
    [[0.03125, 0.03125, -0.03125], [0.2223207547681, 0.07475317642186, -0.07475317642186], [0.2535707547681, 0.10600317642185998, -0.10600317642185998]],
    [[0.03125, 0.03125, -0.03125], [0.2535707547681, 0.10600317642185998, -0.10600317642185998], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.0567331020208, 0.01891103400693, -0.01891103400693]],
    [[0.03125, 0.03125, -0.03125], [0.0567331020208, 0.01891103400693, -0.01891103400693], [0.08798310202079998, 0.012338965993069998, -0.05016103400692999]],
    [[0.03125, 0.03125, -0.03125], [0.08798310202079998, 0.012338965993069998, -0.05016103400692999], [0.1192331020208, -0.01891103400694, -0.01891103400694]],
    [[0.03125, 0.03125, -0.03125], [0.1192331020208, -0.01891103400694, -0.01891103400694], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.04514674666574, 0.004016619199509998, -0.004016619199509998]],
    [[0.03125, 0.03125, -0.03125], [0.04514674666574, 0.004016619199509998, -0.004016619199509998], [0.06505519361579, -0.013896746665740003, -0.02392506614956]],
    [[0.03125, 0.03125, -0.03125], [0.06505519361579, -0.013896746665740003, -0.02392506614956], [0.09630519361579001, 0.004016619199510001, -0.045146746665740006]],
    [[0.03125, 0.03125, -0.03125], [0.09630519361579001, 0.004016619199510001, -0.045146746665740006], [0.12755519361579, -0.02392506614956, -0.02723338080049]],
    [[0.03125, 0.03125, -0.03125], [0.12755519361579, -0.02392506614956, -0.02723338080049], [0.15880519361579, -0.045146746665740006, -0.05517506614956]],
    [[0.03125, 0.03125, -0.03125], [0.15880519361579, -0.045146746665740006, -0.05517506614956], [0.19005519361579, -0.027233380800490002, -0.07639674666574]],
    [[0.03125, 0.03125, -0.03125], [0.19005519361579, -0.027233380800490002, -0.07639674666574], [0.22130519361579, -0.05517506614956, -0.05848338080049001]],
    [[0.03125, 0.03125, -0.03125], [0.22130519361579, -0.05517506614956, -0.05848338080049001], [0.25255519361579, -0.07639674666574, -0.08642506614956]],
    [[0.03125, 0.03125, -0.03125], [0.25255519361579, -0.07639674666574, -0.08642506614956], [0.28380519361579, -0.05517506614956, -0.10764674666574]],
    [[0.03125, 0.03125, -0.03125], [0.28380519361579, -0.05517506614956, -0.10764674666574], [0.31505519361579, -0.07639674666574, -0.08642506614956]],
    [[0.03125, 0.03125, -0.03125], [0.31505519361579, -0.07639674666574, -0.08642506614956], [0.34630519361579004, -0.04514674666573999, -0.11767506614956001]],
    [[0.03125, 0.03125, -0.03125], [0.34630519361579004, -0.04514674666573999, -0.11767506614956001], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.06138596307828, 0.011163331343499998, -0.011163331343499998]],
    [[0.03125, 0.03125, -0.03125], [0.06138596307828, 0.011163331343499998, -0.011163331343499998], [0.06444228801973, 0.020086668656499995, 0.008107006402050004]],
    [[0.03125, 0.03125, -0.03125], [0.06444228801973, 0.020086668656499995, 0.008107006402050004], [0.0513366686565, 0.03935700640205, 0.03319228801973]],
    [[0.03125, 0.03125, -0.03125], [0.0513366686565, 0.03935700640205, 0.03319228801973], [0.07060700640205, 0.06444228801972998, 0.020086668656500002]],
    [[0.03125, 0.03125, -0.03125], [0.07060700640205, 0.06444228801972998, 0.020086668656500002], [0.09569228801973, 0.05133666865650001, 0.03935700640204999]],
    [[0.03125, 0.03125, -0.03125], [0.09569228801973, 0.05133666865650001, 0.03935700640204999], [0.08258666865650002, 0.07060700640204999, 0.06444228801973]],
    [[0.03125, 0.03125, -0.03125], [0.08258666865650002, 0.07060700640204999, 0.06444228801973], [0.10185700640205, 0.09569228801973, 0.05133666865650001]],
    [[0.03125, 0.03125, -0.03125], [0.10185700640205, 0.09569228801973, 0.05133666865650001], [0.12694228801973, 0.0825866686565, 0.07060700640205]],
    [[0.03125, 0.03125, -0.03125], [0.12694228801973, 0.0825866686565, 0.07060700640205], [0.15819228801973, 0.10185700640205002, 0.05133666865649998]],
    [[0.03125, 0.03125, -0.03125], [0.15819228801973, 0.10185700640205002, 0.05133666865649998], [0.18944228801973, 0.08258666865649998, 0.07060700640205002]],
    [[0.03125, 0.03125, -0.03125], [0.18944228801973, 0.08258666865649998, 0.07060700640205002], [0.22069228801973, 0.10185700640205002, 0.05133666865649998]],
    [[0.03125, 0.03125, -0.03125], [0.22069228801973, 0.10185700640205002, 0.05133666865649998], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.03125, 0.03125, -0.03125], [0.03125, 0.03125, -0.03125], [0.05335850728686, 0.011950224171370004, -0.011950224171370004]],
    [[0.03125, 0.03125, -0.03125], [0.05335850728686, 0.011950224171370004, -0.011950224171370004], [0.0460089556296, 0.01929977582863, 0.01929977582863]],
    [[0.03125, 0.03125, -0.03125], [0.0460089556296, 0.01929977582863, 0.01929977582863], [0.07725895562960002, 0.05054977582863001, -0.011950224171370004]],
    [[0.03125, 0.03125, -0.03125], [0.07725895562960002, 0.05054977582863001, -0.011950224171370004], [0.10850895562959999, 0.01929977582863, 0.01929977582863]],
    [[0.03125, 0.03125, -0.03125], [0.10850895562959999, 0.01929977582863, 0.01929977582863], [0.1397589556296, 0.05054977582863002, -0.01195022417137001]],
    [[0.03125, 0.03125, -0.03125], [0.1397589556296, 0.05054977582863002, -0.01195022417137001], [0.1710089556296, 0.01929977582863, 0.01929977582863]],
    [[0.03125, 0.03125, -0.03125], [0.1710089556296, 0.01929977582863, 0.01929977582863], [0.2022589556296, 0.05054977582863002, -0.01195022417137001]],
    [[0.03125, 0.03125, -0.03125], [0.2022589556296, 0.05054977582863002, -0.01195022417137001], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.33333333333333, 0.0, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.33333333333333, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666666666666, 0.08333333333333001, -6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666666, 0.08333333333333001, -6.938893903907228e-18], [0.25, 0.08333333333333, 0.08333333333333]],
    [[0.16666666666667, 0.0, 0.0], [0.25, 0.08333333333333, 0.08333333333333], [0.24999999999999997, 0.25, 0.08333333333332998]],
    [[0.16666666666667, 0.0, 0.0], [0.24999999999999997, 0.25, 0.08333333333332998], [0.25, 0.25, 0.25]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.125, 0.125, -0.04166666666667001]],
    [[0.16666666666667, 0.0, 0.0], [0.125, 0.125, -0.04166666666667001], [0.125, 0.125, 0.125]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 0.0, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.20833333333333004, 0.125, -0.125]],
    [[0.16666666666667, 0.0, 0.0], [0.20833333333333004, 0.125, -0.125], [0.375, 0.125, -0.125]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.1875, 0.14583333333333004, -0.06250000000000001]],
    [[0.16666666666667, 0.0, 0.0], [0.1875, 0.14583333333333004, -0.06250000000000001], [0.3125, 0.1875, -0.0625]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.23333333333332998, 0.049999999999999996, 6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.23333333333332998, 0.049999999999999996, 6.938893903907228e-18], [0.35, 0.05, 0.05]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.2, 0.033333333333329995, 3.469446951953614e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.2, 0.033333333333329995, 3.469446951953614e-18], [0.20000000000000004, 0.2, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.10833333333333, 0.07500000000000001, -0.07500000000000001]],
    [[0.16666666666667, 0.0, 0.0], [0.10833333333333, 0.07500000000000001, -0.07500000000000001], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.18333333333333, 0.15, -0.05000000000000001]],
    [[0.16666666666667, 0.0, 0.0], [0.18333333333333, 0.15, -0.05000000000000001], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.2, 0.033333333333329995, 3.469446951953614e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.2, 0.033333333333329995, 3.469446951953614e-18], [0.20000000000000004, 0.2, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.005, 0.005, -0.005]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.245, 0.08833333333332999, -0.08833333333332999]],
    [[0.16666666666667, 0.0, 0.0], [0.245, 0.08833333333332999, -0.08833333333332999], [0.255, 0.245, -0.245]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666666666666, 0.005000000000000003, -2.168404344971009e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666666, 0.005000000000000003, -2.168404344971009e-18], [0.32833333333333, 0.005, 0.005]],
    [[0.16666666666667, 0.0, 0.0], [0.32833333333333, 0.005, 0.005], [0.495, 0.005, 0.005]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.15666666666666, 0.08833333333333002, -6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.15666666666666, 0.08833333333333002, -6.938893903907228e-18], [0.245, 0.08833333333333, 0.07833333333333]],
    [[0.16666666666667, 0.0, 0.0], [0.245, 0.08833333333333, 0.07833333333333], [0.24499999999999997, 0.245, 0.08833333333332999]],
    [[0.16666666666667, 0.0, 0.0], [0.24499999999999997, 0.245, 0.08833333333332999], [0.255, 0.245, 0.245]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [5e-05, 5e-05, -5e-05]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.24995, 0.08338333333333002, -0.08338333333333002]],
    [[0.16666666666667, 0.0, 0.0], [0.24995, 0.08338333333333002, -0.08338333333333002], [0.25005, 0.24995, -0.24995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666666666666, 4.9999999999997244e-05, 2.7545511444709847e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666666, 4.9999999999997244e-05, 2.7545511444709847e-18], [0.33328333333333, 5e-05, 5e-05]],
    [[0.16666666666667, 0.0, 0.0], [0.33328333333333, 5e-05, 5e-05], [0.49995, 5e-05, 5e-05]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16656666666665998, 0.08338333333333, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16656666666665998, 0.08338333333333, 0.0], [0.24995000000000003, 0.08338333333333003, 0.08328333333332999]],
    [[0.16666666666667, 0.0, 0.0], [0.24995000000000003, 0.08338333333333003, 0.08328333333332999], [0.24995, 0.24995, 0.08338333333333003]],
    [[0.16666666666667, 0.0, 0.0], [0.24995, 0.24995, 0.08338333333333003], [0.25005, 0.24995, 0.24995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [5e-07, 5e-07, -5e-07]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.24999950000000004, 0.08333383333333001, -0.08333383333333001]],
    [[0.16666666666667, 0.0, 0.0], [0.24999950000000004, 0.08333383333333001, -0.08333383333333001], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666666666666, 4.999999999933111e-07, 6.6888603657895996e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666666, 4.999999999933111e-07, 6.6888603657895996e-18], [0.33333283333333, 5e-07, 5e-07]],
    [[0.16666666666667, 0.0, 0.0], [0.33333283333333, 5e-07, 5e-07], [0.4999995, 5e-07, 5e-07]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666566666666, 0.08333383333333, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666566666666, 0.08333383333333, 0.0], [0.24999950000000004, 0.08333383333333001, 0.08333283333332998]],
    [[0.16666666666667, 0.0, 0.0], [0.24999950000000004, 0.08333383333333001, 0.08333283333332998], [0.24999949999999999, 0.2499995, 0.08333383333333003]],
    [[0.16666666666667, 0.0, 0.0], [0.24999949999999999, 0.2499995, 0.08333383333333003], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [5e-11, 5e-11, -5e-11]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25000000005, 0.24999999995, -0.24999999995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.33333333328333, 5e-11, 5e-11]],
    [[0.16666666666667, 0.0, 0.0], [0.33333333328333, 5e-11, 5e-11], [0.49999999995, 5e-11, 5e-11]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666666656666, 0.08333333328333001, -6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666656666, 0.08333333328333001, -6.938893903907228e-18], [0.24999999995, 0.08333333328333, 0.08333333328333]],
    [[0.16666666666667, 0.0, 0.0], [0.24999999995, 0.08333333328333, 0.08333333328333], [0.24999999994999997, 0.24999999995, 0.08333333328332998]],
    [[0.16666666666667, 0.0, 0.0], [0.24999999994999997, 0.24999999995, 0.08333333328332998], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 1.000000082740371e-10, -1.000000082740371e-10]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.2500000000005, 0.2499999999995, -0.2499999999995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.33333333333283, 5e-13, 5e-13]],
    [[0.16666666666667, 0.0, 0.0], [0.33333333333283, 5e-13, 5e-13], [0.4999999999995, 5e-13, 5e-13]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666666666566, 0.08333333333283, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666566, 0.08333333333283, 0.0], [0.2499999999995, 0.08333333333283, 0.08333333333283]],
    [[0.16666666666667, 0.0, 0.0], [0.2499999999995, 0.08333333333283, 0.08333333333283], [0.24999999999949998, 0.2499999999995, 0.08333333333283]],
    [[0.16666666666667, 0.0, 0.0], [0.24999999999949998, 0.2499999999995, 0.08333333333283], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 9.999917560676863e-13, -9.999917560676863e-13]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.249999995, 0.08333332833332999, -0.08333332833332999]],
    [[0.16666666666667, 0.0, 0.0], [0.249999995, 0.08333332833332999, -0.08333332833332999], [0.250000005, 0.24999999499999997, -0.24999999499999997]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666666166666, 5e-09, 5e-09]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16666665666666, 0.08333332833332999, 6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.16666665666666, 0.08333332833332999, 6.938893903907228e-18], [0.24999999499999997, 0.08333332833333, 0.08333332833333]],
    [[0.16666666666667, 0.0, 0.0], [0.24999999499999997, 0.08333332833333, 0.08333332833333], [0.24999999500000003, 0.249999995, 0.08333332833332999]],
    [[0.16666666666667, 0.0, 0.0], [0.24999999500000003, 0.249999995, 0.08333332833332999], [0.25000000499999997, 0.249999995, 0.249999995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.21428591836617997, 0.04761925169950999, -0.04762210882604]],
    [[0.16666666666667, 0.0, 0.0], [0.21428591836617997, 0.04761925169950999, -0.04762210882604], [0.21428591836618, 0.21428591836618, -0.21428306123964996]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.24015889400151, 0.007337674375589999, -0.05452006706914]],
    [[0.16666666666667, 0.0, 0.0], [0.24015889400151, 0.007337674375589999, -0.05452006706914], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.13753630873341, 0.10984543262718, -0.09375202185276998]],
    [[0.16666666666667, 0.0, 0.0], [0.13753630873341, 0.10984543262718, -0.09375202185276998], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.19471724673529, 0.0, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.13725317642186, 0.11815408810143, -0.11815408810143]],
    [[0.16666666666667, 0.0, 0.0], [0.13725317642186, 0.11815408810143, -0.11815408810143], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.11650563265973, 0.0, -0.05016103400694]],
    [[0.16666666666667, 0.0, 0.0], [0.11650563265973, 0.0, -0.05016103400694], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.21088852694912, 0.0, -0.06196346079956]],
    [[0.16666666666667, 0.0, 0.0], [0.21088852694912, 0.0, -0.06196346079956], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.13310700640205003, 0.10536229000956002, -6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.13310700640205003, 0.10536229000956002, -6.938893903907228e-18], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.08614206479156002, 0.019299775828630003, -1.734723475976807e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.08614206479156002, 0.019299775828630003, -1.734723475976807e-18], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.25, -0.25]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.25, 0.25, 0.25]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.125, 0.125, -0.04166666666667001]],
    [[0.16666666666667, 0.0, 0.0], [0.125, 0.125, -0.04166666666667001], [0.125, 0.125, 0.125]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.375, 0.125, -0.125]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.3125, 0.1875, -0.0625]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.23333333333332998, 0.049999999999999996, 6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.23333333333332998, 0.049999999999999996, 6.938893903907228e-18], [0.35, 0.05, 0.05]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.20000000000000004, 0.2, 0.0]],
    [[0.25, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.20000000000000004, 0.2, 0.0]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.005, 0.005, -0.005]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.255, 0.245, -0.245]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 0.005000000000000003, -2.168404344971009e-18]],
    [[0.25, 0.0, 0.0], [0.25, 0.005000000000000003, -2.168404344971009e-18], [0.495, 0.005, 0.005]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.245, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.245, 0.0], [0.255, 0.245, 0.245]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [5e-05, 5e-05, -5e-05]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25005, 0.24995, -0.24995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 4.9999999999997244e-05, 2.7545511444709847e-18]],
    [[0.25, 0.0, 0.0], [0.25, 4.9999999999997244e-05, 2.7545511444709847e-18], [0.49995, 5e-05, 5e-05]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.24995, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.24995, 0.0], [0.25005, 0.24995, 0.24995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [5e-07, 5e-07, -5e-07]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25000000000000006, 5.000000000071889e-07, -7.188927442024857e-18]],
    [[0.25, 0.0, 0.0], [0.25000000000000006, 5.000000000071889e-07, -7.188927442024857e-18], [0.4999995, 5e-07, 5e-07]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.24999950000000004, -1.3877787807814457e-17]],
    [[0.25, 0.0, 0.0], [0.25, 0.24999950000000004, -1.3877787807814457e-17], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [5e-11, 5e-11, -5e-11]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25000000005, 0.24999999995, -0.24999999995]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.24999999995, 0.24999999995, -5.000000413701855e-11]],
    [[0.25, 0.0, 0.0], [0.24999999995, 0.24999999995, -5.000000413701855e-11], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2500000000005, 0.2499999999995, -0.2499999999995]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2499999999995, 0.2499999999995, -4.999889391399392e-13]],
    [[0.25, 0.0, 0.0], [0.2499999999995, 0.2499999999995, -4.999889391399392e-13], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.250000005, 0.24999999499999997, -0.24999999499999997]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.249999995, 5e-09, 5e-09]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.249999995, 0.249999995, -4.999999997368221e-09]],
    [[0.25, 0.0, 0.0], [0.249999995, 0.249999995, -4.999999997368221e-09], [0.25000000499999997, 0.249999995, 0.249999995]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.21428591836618, 0.21428591836618, -0.21428306123964996]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.25, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.19471724673529, 0.0, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.11650563265973, 0.0, -0.05016103400694]],
    [[0.16666666666667, 0.0, 0.0], [0.11650563265973, 0.0, -0.05016103400694], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.21088852694912, 0.0, -0.06196346079956]],
    [[0.16666666666667, 0.0, 0.0], [0.21088852694912, 0.0, -0.06196346079956], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.13310700640205003, 0.10536229000956002, -6.938893903907228e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.13310700640205003, 0.10536229000956002, -6.938893903907228e-18], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.16666666666667, 0.0, 0.0], [0.16666666666667, 0.0, 0.0], [0.08614206479156002, 0.019299775828630003, -1.734723475976807e-18]],
    [[0.16666666666667, 0.0, 0.0], [0.08614206479156002, 0.019299775828630003, -1.734723475976807e-18], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.25, 0.25, 0.25]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 0.0, 0.0]],
    [[0.125, 0.125, -0.125], [0.25, 0.0, 0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [0.375, 0.125, -0.125]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.1875, 0.0625, -0.0625]],
    [[0.25, 0.25, -0.25], [0.1875, 0.0625, -0.0625], [0.3125, 0.1875, -0.0625]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.2, 0.04999999999999999, -0.04999999999999999]],
    [[0.125, 0.125, -0.125], [0.2, 0.04999999999999999, -0.04999999999999999], [0.22500000000000003, 0.175, -0.07500000000000001]],
    [[0.125, 0.125, -0.125], [0.22500000000000003, 0.175, -0.07500000000000001], [0.35, 0.05, 0.05]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.1875, 0.0625, -0.0625]],
    [[0.125, 0.125, -0.125], [0.1875, 0.0625, -0.0625], [0.15, 0.07499999999999998, 0.05000000000000002]],
    [[0.125, 0.125, -0.125], [0.15, 0.07499999999999998, 0.05000000000000002], [0.20000000000000004, 0.2, 0.0]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.15, 0.0625, -0.0625]],
    [[0.125, 0.125, -0.125], [0.15, 0.0625, -0.0625], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.2, 0.10000000000000002, -0.10000000000000002]],
    [[0.25, 0.25, -0.25], [0.2, 0.10000000000000002, -0.10000000000000002], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.1875, 0.0625, -0.0625]],
    [[0.125, 0.125, -0.125], [0.1875, 0.0625, -0.0625], [0.15, 0.07499999999999998, 0.05000000000000002]],
    [[0.125, 0.125, -0.125], [0.15, 0.07499999999999998, 0.05000000000000002], [0.20000000000000004, 0.2, 0.0]],
    [[0.0625, 0.0625, -0.0625], [0.0625, 0.0625, -0.0625], [0.005, 0.005, -0.005]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.255, 0.245, -0.245]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.255, 0.245, -0.245]],
    [[0.25, 0.25, -0.25], [0.255, 0.245, -0.245], [0.495, 0.005, 0.005]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.495, 0.0050000000000000044, -0.0050000000000000044]],
    [[0.25, 0.25, -0.25], [0.495, 0.0050000000000000044, -0.0050000000000000044], [0.255, 0.245, 0.245]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.0625, 0.0625, -0.0625], [0.0625, 0.0625, -0.0625], [5e-05, 5e-05, -5e-05]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.25005, 0.24995, -0.24995]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.25005, 0.24995, -0.24995]],
    [[0.25, 0.25, -0.25], [0.25005, 0.24995, -0.24995], [0.49995, 5e-05, 5e-05]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.49995, 4.999999999999449e-05, -4.999999999999449e-05]],
    [[0.25, 0.25, -0.25], [0.49995, 4.999999999999449e-05, -4.999999999999449e-05], [0.25005, 0.24995, 0.24995]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.0625, 0.0625, -0.0625], [0.0625, 0.0625, -0.0625], [5e-07, 5e-07, -5e-07]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.25, 0.25, -0.25], [0.2500005, 0.24999949999999999, -0.24999949999999999], [0.4999995, 5e-07, 5e-07]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.4999995, 5.000000000143778e-07, -5.000000000143778e-07]],
    [[0.25, 0.25, -0.25], [0.4999995, 5.000000000143778e-07, -5.000000000143778e-07], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.0625, 0.0625, -0.0625], [0.0625, 0.0625, -0.0625], [5e-11, 5e-11, -5e-11]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.49999999995, 5e-11, 5e-11]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.49999999995, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.49999999995, 0.0, 0.0], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 1.000000082740371e-10, -1.000000082740371e-10]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.4999999999995, 5e-13, 5e-13]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.4999999999995, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.4999999999995, 0.0, 0.0], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 9.999917560676863e-13, -9.999917560676863e-13]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.250000005, 0.24999999499999997, -0.24999999499999997]],
    [[0.25, 0.25, -0.25], [0.0625, 0.0625, -0.0625], [0.25, 0.25, -0.25]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.499999995, 5e-09, 5e-09]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.499999995, 0.0, 0.0]],
    [[0.25, 0.25, -0.25], [0.499999995, 0.0, 0.0], [0.25000000499999997, 0.249999995, 0.249999995]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.0625, 0.0625, -0.0625], [0.0625, 0.0625, -0.0625], [0.08928877549271, 0.08928591836618, -0.08928591836618]],
    [[0.125, 0.125, -0.125], [0.08928877549271, 0.08928591836618, -0.08928591836618], [0.21428591836618, 0.21428591836618, -0.21428306123964996]],
    [[0.0625, 0.0625, -0.0625], [0.0625, 0.0625, -0.0625], [0.11149628930162, 0.08167975537864, -0.08167975537864]],
    [[0.0625, 0.0625, -0.0625], [0.11149628930162, 0.08167975537864, -0.08167975537864], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.1875, 0.0625, -0.0625]],
    [[0.125, 0.125, -0.125], [0.1875, 0.0625, -0.0625], [0.28182556066818, -0.05702349869203999, -0.06797650130796001]],
    [[0.125, 0.125, -0.125], [0.28182556066818, -0.05702349869203999, -0.06797650130796001], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.19529638617449, 0.015154567372820005, -0.015154567372820005]],
    [[0.125, 0.125, -0.125], [0.19529638617449, 0.015154567372820005, -0.015154567372820005], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.125, 0.125, -0.125], [0.125, 0.125, -0.125], [0.19471724673529, 0.0, 0.0]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.1598207547681, 0.0625, -0.0625]],
    [[0.125, 0.125, -0.125], [0.1598207547681, 0.0625, -0.0625], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.0625, 0.0625, -0.0625], [0.0625, 0.0625, -0.0625], [0.11266103400693, 0.012338965993060003, -0.012338965993060003]],
    [[0.0625, 0.0625, -0.0625], [0.11266103400693, 0.012338965993060003, -0.012338965993060003], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.17615844695005, 0.0625, -0.0625]],
    [[0.125, 0.125, -0.125], [0.17615844695005, 0.0625, -0.0625], [0.25255519361579, -0.023925066149560005, -0.08973338080048998]],
    [[0.125, 0.125, -0.125], [0.25255519361579, -0.023925066149560005, -0.08973338080048998], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.1875, 0.0625, -0.0625]],
    [[0.125, 0.125, -0.125], [0.1875, 0.0625, -0.0625], [0.15513596307828, 0.11689299359795, 0.008107006402050004]],
    [[0.125, 0.125, -0.125], [0.15513596307828, 0.11689299359795, 0.008107006402050004], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.125, 0.125, -0.125], [0.0625, 0.0625, -0.0625], [0.14710850728686, 0.10570022417137001, -0.10570022417137001]],
    [[0.125, 0.125, -0.125], [0.14710850728686, 0.10570022417137001, -0.10570022417137001], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.25, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.25, 0.0, 0.0], [0.375, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.375, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, 0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, 0.125], [0.25, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.125, 0.125], [0.25, 0.25, 0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.25, 0.125], [0.25, 0.25, 0.25]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, 0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.25, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.25, 0.125, -0.125], [0.375, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.1875, 0.0625, -0.0625]],
    [[0.125, 0.0, 0.0], [0.1875, 0.0625, -0.0625], [0.1875, 0.1875, -0.0625]],
    [[0.125, 0.0, 0.0], [0.1875, 0.1875, -0.0625], [0.3125, 0.1875, -0.0625]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15, 0.05000000000000001, -6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.15, 0.05000000000000001, -6.938893903907228e-18], [0.22500000000000003, 0.05, 0.05]],
    [[0.125, 0.0, 0.0], [0.22500000000000003, 0.05, 0.05], [0.35, 0.05, 0.05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0]],
    [[0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18], [0.20000000000000004, 0.2, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15, 0.07499999999999998, -0.07499999999999998]],
    [[0.125, 0.0, 0.0], [0.15, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15, 0.1, -0.05000000000000001]],
    [[0.125, 0.0, 0.0], [0.15, 0.1, -0.05000000000000001], [0.22500000000000003, 0.15, -0.05000000000000001]],
    [[0.125, 0.0, 0.0], [0.22500000000000003, 0.15, -0.05000000000000001], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0]],
    [[0.125, 0.0, 0.0], [0.07499999999999998, 0.075, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.2, 0.07500000000000001, -6.938893903907228e-18], [0.20000000000000004, 0.2, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.005, 0.005, -0.005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.13, 0.12, -0.12]],
    [[0.125, 0.0, 0.0], [0.13, 0.12, -0.12], [0.245, 0.13, -0.13]],
    [[0.125, 0.0, 0.0], [0.245, 0.13, -0.13], [0.255, 0.245, -0.245]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.005000000000000003, -2.168404344971009e-18]],
    [[0.125, 0.0, 0.0], [0.125, 0.005000000000000003, -2.168404344971009e-18], [0.245, 0.005, 0.005]],
    [[0.125, 0.0, 0.0], [0.245, 0.005, 0.005], [0.37, 0.005, 0.005]],
    [[0.125, 0.0, 0.0], [0.37, 0.005, 0.005], [0.495, 0.005, 0.005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.12, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.12, 0.0], [0.13, 0.12, 0.12]],
    [[0.125, 0.0, 0.0], [0.13, 0.12, 0.12], [0.245, 0.13, 0.12]],
    [[0.125, 0.0, 0.0], [0.245, 0.13, 0.12], [0.245, 0.245, 0.13]],
    [[0.125, 0.0, 0.0], [0.245, 0.245, 0.13], [0.255, 0.245, 0.245]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.010000000000000002, -0.010000000000000002]],
    [[0.125, 0.0, 0.0], [0.125, 0.010000000000000002, -0.010000000000000002], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [5e-05, 5e-05, -5e-05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12505, 0.12495, -0.12495]],
    [[0.125, 0.0, 0.0], [0.12505, 0.12495, -0.12495], [0.24995, 0.12505, -0.12505]],
    [[0.125, 0.0, 0.0], [0.24995, 0.12505, -0.12505], [0.25005, 0.24995, -0.24995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 4.9999999999997244e-05, 2.7545511444709847e-18]],
    [[0.125, 0.0, 0.0], [0.125, 4.9999999999997244e-05, 2.7545511444709847e-18], [0.24995, 5e-05, 5e-05]],
    [[0.125, 0.0, 0.0], [0.24995, 5e-05, 5e-05], [0.37495, 5e-05, 5e-05]],
    [[0.125, 0.0, 0.0], [0.37495, 5e-05, 5e-05], [0.49995, 5e-05, 5e-05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.12495, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.12495, 0.0], [0.12505, 0.12495, 0.12495]],
    [[0.125, 0.0, 0.0], [0.12505, 0.12495, 0.12495], [0.24995, 0.12505, 0.12495]],
    [[0.125, 0.0, 0.0], [0.24995, 0.12505, 0.12495], [0.24995, 0.24995, 0.12505]],
    [[0.125, 0.0, 0.0], [0.24995, 0.24995, 0.12505], [0.25005, 0.24995, 0.24995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 9.999999999999593e-05, -9.999999999999593e-05]],
    [[0.125, 0.0, 0.0], [0.125, 9.999999999999593e-05, -9.999999999999593e-05], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [5e-07, 5e-07, -5e-07]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.1250005, 0.12499950000000001, -0.12499950000000001]],
    [[0.125, 0.0, 0.0], [0.1250005, 0.12499950000000001, -0.12499950000000001], [0.2499995, 0.1250005, -0.1250005]],
    [[0.125, 0.0, 0.0], [0.2499995, 0.1250005, -0.1250005], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999999999999, 4.999999999933111e-07, 6.6888603657895996e-18]],
    [[0.125, 0.0, 0.0], [0.12499999999999999, 4.999999999933111e-07, 6.6888603657895996e-18], [0.2499995, 5e-07, 5e-07]],
    [[0.125, 0.0, 0.0], [0.2499995, 5e-07, 5e-07], [0.3749995, 5e-07, 5e-07]],
    [[0.125, 0.0, 0.0], [0.3749995, 5e-07, 5e-07], [0.4999995, 5e-07, 5e-07]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.12499949999999999, 6.938893903907228e-18]],
    [[0.125, 0.0, 0.0], [0.125, 0.12499949999999999, 6.938893903907228e-18], [0.12500050000000001, 0.1249995, 0.1249995]],
    [[0.125, 0.0, 0.0], [0.12500050000000001, 0.1249995, 0.1249995], [0.24999950000000004, 0.1250005, 0.12499950000000001]],
    [[0.125, 0.0, 0.0], [0.24999950000000004, 0.1250005, 0.12499950000000001], [0.24999950000000004, 0.2499995, 0.1250005]],
    [[0.125, 0.0, 0.0], [0.24999950000000004, 0.2499995, 0.1250005], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 1.000000000001e-06, -1.000000000001e-06]],
    [[0.125, 0.0, 0.0], [0.125, 1.000000000001e-06, -1.000000000001e-06], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [5e-11, 5e-11, -5e-11]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.24999999995, 0.12500000005, -0.12500000005]],
    [[0.125, 0.0, 0.0], [0.24999999995, 0.12500000005, -0.12500000005], [0.25000000005, 0.24999999995, -0.24999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.24999999995, 5e-11, 5e-11]],
    [[0.125, 0.0, 0.0], [0.24999999995, 5e-11, 5e-11], [0.37499999995, 5e-11, 5e-11]],
    [[0.125, 0.0, 0.0], [0.37499999995, 5e-11, 5e-11], [0.49999999995, 5e-11, 5e-11]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999998333, 0.12499999998333, -1.6669998714746725e-11]],
    [[0.125, 0.0, 0.0], [0.12499999998333, 0.12499999998333, -1.6669998714746725e-11], [0.12499999998333, 0.12499999998333, 0.12499999998333]],
    [[0.125, 0.0, 0.0], [0.12499999998333, 0.12499999998333, 0.12499999998333], [0.24999999995, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.24999999995, 0.125, 0.125], [0.24999999995, 0.24999999995, 0.12500000005]],
    [[0.125, 0.0, 0.0], [0.24999999995, 0.24999999995, 0.12500000005], [0.25000000005, 0.24999999995, 0.24999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 1.000000082740371e-10, -1.000000082740371e-10]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.2499999999995, 0.1250000000005, -0.1250000000005]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 0.1250000000005, -0.1250000000005], [0.2500000000005, 0.2499999999995, -0.2499999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.2499999999995, 5e-13, 5e-13]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 5e-13, 5e-13], [0.3749999999995, 5e-13, 5e-13]],
    [[0.125, 0.0, 0.0], [0.3749999999995, 5e-13, 5e-13], [0.4999999999995, 5e-13, 5e-13]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999999983, 0.12499999999983, -1.700029006457271e-13]],
    [[0.125, 0.0, 0.0], [0.12499999999983, 0.12499999999983, -1.700029006457271e-13], [0.12499999999983, 0.12499999999983, 0.12499999999983]],
    [[0.125, 0.0, 0.0], [0.12499999999983, 0.12499999999983, 0.12499999999983], [0.2499999999995, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 0.125, 0.125], [0.2499999999995, 0.2499999999995, 0.1250000000005]],
    [[0.125, 0.0, 0.0], [0.2499999999995, 0.2499999999995, 0.1250000000005], [0.2500000000005, 0.2499999999995, 0.2499999999995]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 9.999917560676863e-13, -9.999917560676863e-13]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.125, -0.125], [0.249999995, 0.125000005, -0.125000005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.124999995, 5e-09, 5e-09]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12499999833333, 0.12499999833333, -1.6666699964584808e-09]],
    [[0.125, 0.0, 0.0], [0.12499999833333, 0.12499999833333, -1.6666699964584808e-09], [0.12499999833333, 0.12499999833333, 0.12499999833333]],
    [[0.125, 0.0, 0.0], [0.12499999833333, 0.12499999833333, 0.12499999833333], [0.24999999500000003, 0.125, 0.125]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.125, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.08928591836618, 0.08928591836618, -0.08928306123965002]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.14417975537864, 0.04042608289204, -0.048996289301620005]],
    [[0.125, 0.0, 0.0], [0.14417975537864, 0.04042608289204, -0.048996289301620005], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.15682556066818, 0.0073376743755899955, -0.06797650130796]],
    [[0.125, 0.0, 0.0], [0.15682556066818, 0.0073376743755899955, -0.06797650130796], [0.28182556066818, 0.007337674375590006, -0.09618673373581]],
    [[0.125, 0.0, 0.0], [0.28182556066818, 0.007337674375590006, -0.09618673373581], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.1085391923974, 0.1085391923974, -0.1085391923974]],
    [[0.125, 0.0, 0.0], [0.1085391923974, 0.1085391923974, -0.1085391923974], [0.2335391923974, 0.10853919239740001, -0.10853919239740001]],
    [[0.125, 0.0, 0.0], [0.2335391923974, 0.10853919239740001, -0.10853919239740001], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.10984543262718, 0.054202975400080004, -0.07029638617449]],
    [[0.125, 0.0, 0.0], [0.10984543262718, 0.054202975400080004, -0.07029638617449], [0.17920297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.125, 0.0, 0.0], [0.17920297540008, 0.10984543262718, -0.09375202185276998], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.19471724673529, 0.0, 0.0]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.13725317642186, 0.03482075476810001, -0.03482075476810001]],
    [[0.125, 0.0, 0.0], [0.13725317642186, 0.03482075476810001, -0.03482075476810001], [0.1598207547681, 0.13725317642186, -0.13725317642186]],
    [[0.125, 0.0, 0.0], [0.1598207547681, 0.13725317642186, -0.13725317642186], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.07483896599306, 0.025483102020800008, -0.050161034006930005]],
    [[0.125, 0.0, 0.0], [0.07483896599306, 0.025483102020800008, -0.050161034006930005], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12755519361579, 3.469446951953614e-18, -0.037821812815300004]],
    [[0.125, 0.0, 0.0], [0.12755519361579, 3.469446951953614e-18, -0.037821812815300004], [0.25255519361579, -0.013896746665740006, -0.08973338080048998]],
    [[0.125, 0.0, 0.0], [0.25255519361579, -0.013896746665740006, -0.08973338080048998], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12694228801973, 0.02819367505855, 0.0]],
    [[0.125, 0.0, 0.0], [0.12694228801973, 0.02819367505855, 0.0], [0.13310700640205, 0.12694228801973, 0.02008666865649998]],
    [[0.125, 0.0, 0.0], [0.13310700640205, 0.12694228801973, 0.02008666865649998], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0], [0.12780873145823, 0.019299775828629996, 5.204170427930421e-18]],
    [[0.125, 0.0, 0.0], [0.12780873145823, 0.019299775828629996, 5.204170427930421e-18], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, 0.25], [0.5, 0.0, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.125, 0.125, -0.125]],
    [[0.25, 0.0, 0.0], [0.125, 0.125, -0.125], [0.125, 0.125, 0.125]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.375, 0.125, -0.125]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.3125, 0.1875, -0.0625]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.35, 0.05, 0.05]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.20000000000000004, 0.2, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.275, 0.07500000000000001, -0.07500000000000001]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.35000000000000003, 0.15, -0.05000000000000001]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.20000000000000004, 0.2, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.005, 0.005, -0.005]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.255, 0.245, -0.245]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.495, 0.005, 0.005]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.255, 0.245, 0.245]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.010000000000000009, -0.010000000000000009]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [5e-05, 5e-05, -5e-05]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25005, 0.24995, -0.24995]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.49995, 5e-05, 5e-05]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.25005, 0.24995, 0.24995]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 9.999999999998899e-05, -9.999999999998899e-05]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [5e-07, 5e-07, -5e-07]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2500005, 0.24999949999999999, -0.24999949999999999]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.4999995, 5e-07, 5e-07]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.25000049999999996, 0.2499995, 0.2499995]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 9.999999999871223e-07, -9.999999999871223e-07]],
    [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [5e-11, 5e-11, -5e-11]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, 0.25], [0.49999999995, 5e-11, 5e-11]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, 0.25], [0.4999999999995, 5e-13, 5e-13]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.250000005, 0.24999999499999997, -0.24999999499999997]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.499999995, 5e-09, 5e-09]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.25000000499999997, 0.249999995, 0.249999995]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 9.999999994736442e-09, -9.999999994736442e-09]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.21428591836618, 0.21428591836618, -0.21428306123964996]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.16542608289204, 0.14417975537864, -0.13560954896906]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.30702349869204, 0.27881326626419, -0.09317443933181999]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2335391923974, 0.2335391923974, -0.2335391923974]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.30420297540008, 0.10984543262718, -0.09375202185276998]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.19471724673529, 0.0, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.2848207547681, 0.13725317642185997, -0.13725317642185997]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.19983896599306, 0.0, -0.05016103400694]],
    [[0.25, 0.0, 0.0], [0.19983896599306, 0.0, -0.05016103400694], [0.1504831020208, -0.05016103400693006, -0.050161034006930005]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.28526661919950996, 0.10107493385044, -0.12244480638420999]],
    [[0.25, 0.0, 0.0], [0.28526661919950996, 0.10107493385044, -0.12244480638420999], [0.37755519361579, -0.013896746665740034, -0.14892506614956003]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.0]],
    [[0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.25194228801973, 0.13310700640205003, 0.02008666865649998]],
    [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.019299775828629996, 0.019299775828630003, -0.0164910443704]],
    [[0.25, 0.0, 0.0], [0.019299775828629996, 0.019299775828630003, -0.0164910443704], [0.23350895562960003, 0.01929977582863, 0.01929977582863]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.3340180318849028, 0.2628992499814247, -0.08026241667832462]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.2137227876497769, 0.093093664714089, -0.05974862059816807]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.27552886942034693, 0.11690556058675651, -0.10318129621874124]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.39151773252170513, 0.21381995493164835, -0.12776341180635514]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.40866906211566806, 0.16183436433230652, -0.1613018036242459]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.22003916099471688, 0.20501660151112777, -0.09800760249479253]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.24045118715523867, 0.10702260984114352, -0.0776786404971453]],
    [[0.325, 0.175, -0.075], [0.325, 0.175, -0.075], [0.2504242892942908, 0.019389578083142928, -0.019368112608480842]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.295013852498609, 0.2508204227741194, -0.04577659902897381]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.24242525410227775, 0.06731871300873452, -0.03305872603257662]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.24046208249868806, 0.17990921955569128, -0.11372498103311701]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.3882711546085389, 0.22395535982198989, -0.1369335503279827]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.3844989503031868, 0.1886946121862806, -0.16295993098124692]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.2779461273406579, 0.17481324315642394, -0.13072958297641307]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.39612227824410334, 0.1967733022627433, -0.17188555700845254]],
    [[0.325, 0.175, -0.075], [0.275, 0.125, -0.07500000000000001], [0.4437911987943085, 0.20001347607593456, -0.15008794222606148]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.28153562089802436, 0.22029032842067947, -0.0630533989375725]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.22323449865962852, 0.11524491536183655, -0.024288339464344735]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.25238137565549157, 0.14846847957792697, -0.07600003553999336]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.36965942656945555, 0.2769654085930047, -0.14677312120466685]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.376155466209785, 0.265472969941267, -0.1898837620659733]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.2717448848975106, 0.12074676080370272, -0.09585527622092507]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.3097938502432877, 0.1207271527804505, -0.11368945627301073]],
    [[0.325, 0.175, -0.075], [0.25, 0.05, 0.05], [0.37438174622746684, 0.17537635028715332, -0.1751119836561042]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.38942623041255087, 0.1668908013440321, -0.05985959766099158]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.3241981985743515, 0.09881933020509061, -0.056004256206049416]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.4325540171568208, 0.1225196992441386, -0.07502671364818396]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.31917788812836334, 0.22346387944656418, -0.10465738066258609]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.2528327970649438, 0.11843362275174081, -0.06346412494889069]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.1934074281999878, 0.06485525950634964, 0.032095426256244744]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.1843019954111068, 0.09269401596975724, -0.07067855487539437]],
    [[0.325, 0.175, -0.075], [0.5, 0.0, 0.0], [0.4235020978285542, 0.3241793734432341, -0.32415191707701396]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.30018600986169774, 0.22318501304242291, -0.12976270869643178]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.23424970699489833, 0.12292076466875447, -0.10086320044224688]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.20337187171641105, 0.1445403612415572, -0.07629799012928276]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.4165429327630585, 0.2753566177576484, -0.1494898953629182]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.34372129437108523, 0.2349821665191594, -0.16467190340988666]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.2584107242444468, 0.18970889879212194, -0.09234462470606337]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.294769815848312, 0.17529137712479287, -0.050149933280235356]],
    [[0.325, 0.175, -0.075], [0.125, 0.125, 0.125], [0.2998173420237973, 0.19984872613671412, -0.05074906793540013]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3779270583631744, 0.22088283967739247, -0.08943619195119923]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.20876372430171464, 0.0707007989111148, -0.06636985921828792]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.20400792681600835, 0.16200823487383617, -0.08670905575752122]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.4346451763952708, 0.20677659062622866, -0.12502650965296214]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3398335725941768, 0.17483842608648772, -0.11873321283515062]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.26514080740908247, 0.23120970436273408, -0.1615937975570199]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2809586570396744, 0.14352425232093402, -0.08121291874497402]],
    [[0.325, 0.175, -0.075], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3001364965294611, 0.056820009661976464, -0.007125350210267903]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.25900784477131744, 0.2501312517086803, -0.022374283099603876]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.27066857986336507, 0.05909162115201409, -0.019744990686160716]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2455675574975922, 0.20744223479867482, -0.12752550650011754]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3632733158951638, 0.23068713592971796, -0.14221237054949648]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.38987190896681234, 0.20150005456736597, -0.16636911528282128]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3154194837385081, 0.13095552365943303, -0.1305406191982989]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3746902784543945, 0.19774511818900242, -0.1266950622384393]],
    [[0.325, 0.175, -0.075], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3970598550281102, 0.2001228542167533, -0.10031494885606128]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.4009257823635958, 0.1518744720570625, -0.02419999835767285]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.43340286041143256, 0.2259497231380883, -0.21100238285982692]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.27345148560157573, 0.17342687426589953, -0.07980978323710265]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.3661906151020522, 0.04837369844207751, -0.03702099934091862]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.39073304316259505, 0.1479416871176485, -0.13064018491454565]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.33289690472431094, 0.2432956503776273, -0.18809492558761426]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.3091465862952982, 0.16537289390776916, -0.11212132928977436]],
    [[0.325, 0.175, -0.075], [0.25, 0.25, -0.25], [0.3241860839618918, 0.17451557913283355, -0.0753331584666797]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.29694332415672775, 0.2566276466489984, -0.06963097658231752]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.193643330648122, 0.12775649566616287, -0.04567424714683982]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.38774103316349373, 0.20004406993085572, -0.18178664078176043]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.37468359296687537, 0.21775277414679528, -0.08460664048436944]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.38572175102786577, 0.1289508451917784, -0.09494431042386443]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.30456273516234816, 0.16716280454647942, -0.12463261154764617]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.2168628334931728, 0.14992253887593965, -0.10085794265337723]],
    [[0.275, 0.125, -0.07500000000000001], [0.325, 0.175, -0.075], [0.20061250147118156, 0.10385235170071519, -0.054289813250445546]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.252057500984569, 0.2328731608649095, -0.04105116041197121]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.22742087718767512, 0.11686000925626085, -0.031830608529047885]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.2909509854263653, 0.16926726864602315, -0.13193284085040113]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.3577764265088395, 0.21140703883838113, -0.10233663078624926]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.3535162468362013, 0.14986485478493705, -0.12109190763714196]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.2648788788150056, 0.24669964236128328, -0.16263567706916682]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.28754551062958417, 0.15376262949529798, -0.13665504126036535]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.125, -0.07500000000000001], [0.39709171501160884, 0.2000859754857208, -0.19988256173245908]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.25539840930671587, 0.17641858980774888, -0.03647924399406009]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.2517575666931923, 0.15467908426541244, -0.05091823431349678]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.33245540826779585, 0.13846294714374005, -0.10993341493945596]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.3255766476520289, 0.26540868534967915, -0.10035986402700053]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.3352958728848033, 0.23915435158975057, -0.16415816273179368]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.25806774698262674, 0.2019340862850882, -0.14196689765437315]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.2860453403608344, 0.19796909857120848, -0.16140031303507474]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.05, 0.05], [0.2752534582927718, 0.22508059241876355, -0.17561047545549757]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.4042660998386862, 0.12142512442200985, -0.03762251880344637]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.27502119539350556, 0.18611938437543604, -0.05462646052163547]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.4529532685397286, 0.0964692886561195, -0.05398798250274331]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.27296579051987196, 0.2693335530791934, -0.08275620347167542]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.27247922586528345, 0.11120755135609524, -0.052957474947716285]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.19634717882493036, 0.11165712686056196, 0.013803049566931694]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.200025614934763, 0.11467792944139144, -0.06826485483514699]],
    [[0.275, 0.125, -0.07500000000000001], [0.5, 0.0, 0.0], [0.17597945526733225, 0.12557999594688188, -0.07576096566720389]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.250295982790165, 0.20975909500181444, -0.09363798711477084]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.35567593026203825, 0.20794489218937695, -0.18543692782502308]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.3010402165000878, 0.1275910038060108, -0.12320360355514563]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.39501805753252783, 0.23530262847297778, -0.09509690261515785]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.3047667523826255, 0.20756348202142796, -0.11650387104709878]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.30690861994998697, 0.20920632283557739, -0.14097915950042728]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.32446647470215073, 0.20286670709776966, -0.10014693996326264]],
    [[0.275, 0.125, -0.07500000000000001], [0.125, 0.125, 0.125], [0.3003079260486868, 0.24965289492149617, -0.10074878917906756]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3421443335996295, 0.20015920940249313, -0.07796282091780779]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2083011616331787, 0.10379472931768241, -0.07525784768336852]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.26311388012034115, 0.14317596472341293, -0.11081768634600836]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.4084134372766692, 0.19102442313842366, -0.08315826193192516]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3050329925729527, 0.14108855898140066, -0.0510509340078612]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.340623993887179, 0.20392390179537823, -0.19624965052601673]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25020279575822935, 0.18217298996853737, -0.08923110865525342]],
    [[0.275, 0.125, -0.07500000000000001], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2503757060729662, 0.10591413806023697, -0.006464740534468633]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2397360702444454, 0.2043279677976979, -0.020933925015943433]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2551839913924603, 0.11537492765223635, -0.019823029763953184]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2914346968300868, 0.20620845423101064, -0.14386899716739077]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.333352668132588, 0.21190080941947342, -0.11478424610948278]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.35760920448204225, 0.15903536101218751, -0.14429412033232047]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.28807762918155955, 0.2235461224616867, -0.16294750843017697]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3352051582203532, 0.19241872592121337, -0.1681207891433422]],
    [[0.275, 0.125, -0.07500000000000001], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3479981075667055, 0.20014043488433567, -0.15007308986775705]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.3708518541175624, 0.1626863512789652, -0.04482523251506586]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.4466888600285535, 0.24776298671009406, -0.23127546926728987]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.25495275500984993, 0.19530612942812567, -0.10436474388417413]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.35839667194890285, 0.08233791780013716, -0.07944362176507286]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.425422122797554, 0.16510827409224413, -0.16420605974374028]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.35068866390924786, 0.2536908570823358, -0.211926705622231]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.3367436517795682, 0.17634749483138823, -0.15272909601876983]],
    [[0.275, 0.125, -0.07500000000000001], [0.25, 0.25, -0.25], [0.41879229429619885, 0.17499134487894902, -0.12503199589456998]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.3162506700663971, 0.22518890577203649, -0.08873699589297886]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.198588994217611, 0.10362388812666333, 0.006185328348350602]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.29480925546610864, 0.1332848955452542, -0.10864358937160916]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.3713032356475955, 0.17608293414371634, -0.06349864812437361]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.3789266860488182, 0.11657566857238466, -0.08680037739178242]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.28043027652069585, 0.16239975493267456, -0.11574876379652975]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.1893838098121745, 0.09742238464114894, -0.08712101015344681]],
    [[0.25, 0.05, 0.05], [0.325, 0.175, -0.075], [0.4205171502077082, 0.37353625162499793, -0.37310229329250677]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.2845070179518021, 0.17947681933348614, -0.04159000385997759]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.2708256891805429, 0.09550504551111431, -0.015225798539501859]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.28436705969897147, 0.21446427070499186, -0.14219151650322193]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.3288043810806397, 0.16446523384908246, -0.05483802593381526]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.33075165990944944, 0.12753015999420145, -0.08645456465354759]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.25713774131685063, 0.23767992863902776, -0.162346509008076]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.36743834706680084, 0.26623064648844247, -0.25801161551018065]],
    [[0.25, 0.05, 0.05], [0.275, 0.125, -0.07500000000000001], [0.3720093167680988, 0.3238691679592889, -0.27393268294984696]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.30895145413556435, 0.10974632751650436, -0.03330404074277628]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.2735506414493288, 0.16275807621990773, -0.03711305992788032]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.31217953295132095, 0.2066617103699678, -0.13633838473330892]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.30251082224790377, 0.2053961700047437, -0.04997073084548592]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.310171398157679, 0.23023808172478646, -0.12476941544353523]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.2504828899311511, 0.1882298009636585, -0.1733399423818911]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.33332147461544553, 0.2611905097483603, -0.24918698363731812]],
    [[0.25, 0.05, 0.05], [0.25, 0.05, 0.05], [0.30156667658848246, 0.2971124603593358, -0.29557502650600753]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.4333797609179842, 0.07207311434609735, -0.06843791545752888]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.24951320721384285, 0.17658076249717758, 0.022432479598419208]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.4268347847429085, 0.09501612571927152, -0.08521234609825935]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.28605699317996575, 0.2412474605613135, -0.08237742681676483]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.28157230673012895, 0.060833789501402655, -0.019299992385728285]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.16320557027526772, 0.10113124045657672, 0.03914253945422051]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.1891339280509956, 0.05317235517669672, 0.01658987686866964]],
    [[0.25, 0.05, 0.05], [0.5, 0.0, 0.0], [0.1524949672366354, 0.05109210857074495, 0.047509085460687855]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.260562303040743, 0.17500000000000004, -0.06538685585512313]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.3070236033943674, 0.1717419507574209, -0.15376555415178836]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.27758362419912774, 0.18834713728510155, -0.14093076148422934]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.3731745370699318, 0.175, -0.056185835606259185]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.3059217167718478, 0.17499999999999996, -0.09346527482547703]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.285697054573123, 0.20742144198017343, -0.16811849655329636]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.2954983589312588, 0.1546955921289927, -0.1251939510602515]],
    [[0.25, 0.05, 0.05], [0.125, 0.125, 0.125], [0.2751528060015763, 0.17591092635341743, -0.12606373235499369]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3006370430891311, 0.22896811289667196, -0.07773552770151164]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22826206481957523, 0.10137728914571299, -0.06360952870201407]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2497196871775953, 0.1748001587810203, -0.09655770316509728]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3838201141033228, 0.137714121830898, -0.05082628908877848]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2817155440958499, 0.12129897529379134, -0.04024292782721124]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.29044923481668716, 0.18253228953883022, -0.18204670998154804]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.21821705065656216, 0.13882167300770054, -0.10789977219465191]],
    [[0.25, 0.05, 0.05], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.1766469310657781, 0.08098178834120517, -0.03185221831402829]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.28154923312044433, 0.1359090339289616, -0.013596430852547192]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.31377551937040926, 0.09145600645507956, -0.01095410011495878]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3092679096499802, 0.23825690057723323, -0.16190796431864388]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.29729824098524066, 0.16751418161714232, -0.05291634183183329]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3311487285734721, 0.13364932879485825, -0.08978632273662496]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3016318137336176, 0.1987002562460974, -0.16182390666531013]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.35222833213810645, 0.24808394230937802, -0.21533078616992263]],
    [[0.25, 0.05, 0.05], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3265353862951772, 0.3203025438443141, -0.22415507124811185]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.3163301006162561, 0.18839468139711169, -0.036374648671737994]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.45561813422932923, 0.21927007009595573, -0.17348620135912024]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.32156337362721793, 0.15747473500910564, -0.09746270725299228]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.3833209054401592, 0.11856194752058327, -0.11296983850515149]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.4244894765228524, 0.20688018210632494, -0.1948630156852014]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.33405117831772935, 0.2670788851778667, -0.19081325045485353]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.35423508643728474, 0.22840040287389882, -0.19916705713838936]],
    [[0.25, 0.05, 0.05], [0.25, 0.25, -0.25], [0.3954693319602912, 0.2974761023691209, -0.19890027241737387]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.3666731773891582, 0.16845742395970037, -0.018108723997266096]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.3605426233262123, 0.06909669595841787, -0.06377027187874502]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.322558283249717, 0.08095050554877423, -0.016709325690744257]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.446758521923156, 0.14673148216553697, -0.14119438095334194]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.34090885985662245, 0.25677112473821445, -0.19259875021468464]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.20957757909761998, 0.13256600900065996, -0.11890571698719066]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.178802211125294, 0.08834867681084771, -0.04890764787186477]],
    [[0.5, 0.0, 0.0], [0.325, 0.175, -0.075], [0.42740251451601585, 0.3225612642564628, -0.32055369248027854]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.37971865480178824, 0.12527992052655723, 0.0165581025478619]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.3493758250626005, 0.11506591714113618, -0.05927927744281352]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.29957578905060894, 0.14014635100889122, -0.02001990157444982]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.4697251001119402, 0.12707054112619937, -0.12236079864238358]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.3405294896386446, 0.2510250598494679, -0.1644661824395923]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.24412190500422004, 0.11671532976531453, -0.1073479161873147]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.18673932052078482, 0.12252078469262237, -0.05233752697049418]],
    [[0.5, 0.0, 0.0], [0.275, 0.125, -0.07500000000000001], [0.1831834125079194, 0.11683948769946859, -0.07344329767394883]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.37980552540568874, 0.07358255435995739, -0.018460389320709127]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.32743298935402404, 0.11269430394664899, 0.008120799284108318]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.26852463687873784, 0.13554712886345263, 0.022881816219325196]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.35694214386194406, 0.060549780395067365, -0.060454138585835315]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.36095322223313453, 0.2413768583658127, -0.1562269406997828]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.2598856306591381, 0.05352077729798102, -0.046808674217943824]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.17231026286937245, 0.0577062453757653, 0.03467847040094063]],
    [[0.5, 0.0, 0.0], [0.25, 0.05, 0.05], [0.1544747780190564, 0.05059771320673502, 0.04552681645917368]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4334084025928526, 0.05092958178940657, -0.050929581789406514]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4397746003165283, 0.025464790894703215, -0.025464790894703215]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.44614079804020423, 0.101859163578813, -0.101859163578813]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.45250699576388, 0.1782535362629227, -0.1782535362629227]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.45887319348755584, 0.2546479089470326, -0.2546479089470326]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4652393912112316, 0.33104228163114224, -0.33104228163114224]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.47160558893490756, 0.4074366543152521, -0.4074366543152521]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.48383102699936187, 0.4779717866585833, -0.4779717866585833]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.3171926940683434, 0.125, -0.06719269406834344]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.2565069846165048, 0.125, -0.006506984616504785]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.21233338611550612, 0.12499999999999999, 0.03766661388449387]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.34308529291113177, 0.125, -0.09308529291113177]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.3673780747292119, 0.12499999999999994, -0.11737807472921191]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.2759014615235507, 0.12500000000000003, -0.025901461523550738]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.1763378138875329, 0.1250000000000001, 0.073662186112467]],
    [[0.5, 0.0, 0.0], [0.125, 0.125, 0.125], [0.1469892576142661, 0.12500000000000008, 0.1030107423857339]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.35692009262151064, 0.20187636311378537, -0.023236151261740792]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.30082860617581186, 0.07625909090709805, -0.04873974749795355]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.26810738547369617, 0.0946561303044503, 0.008933743250283177]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.43148853081056027, 0.11880923173047472, -0.11802969071289252]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.31658558648724405, 0.2080460132717707, -0.15072623359680565]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2500321215717287, 0.17045359467308596, -0.1279511677606575]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22768855097286705, 0.13291223090954551, -0.062194429723361824]],
    [[0.5, 0.0, 0.0], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2294375480292492, 0.12058100744500568, -0.07391178203601799]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.39212716571510237, 0.09521877567359716, 0.026728586815082722]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3677621184607648, 0.1408821192971085, -0.06320679729393422]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.31114272632986595, 0.16877171510014619, -0.033631368549949886]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.48665921052840766, 0.1253853422611139, -0.11598817643853618]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3570798324377258, 0.25456926222804893, -0.1522543629764287]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.40398315681960256, 0.24622738147826456, -0.2244727199271122]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.23210254191040544, 0.07570317881177394, -0.05064914868521417]],
    [[0.5, 0.0, 0.0], [0.275, 0.07499999999999998, -0.07499999999999998], [0.42656023250159303, 0.27341655063851533, -0.2720916294215263]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.32331884577107006, 0.17668115422893005, 0.048449092078569345]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.3625789851545691, 0.1374210148454309, -0.1281805858656324]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.32819600204740995, 0.17180399795258988, -0.0681695104950951]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.43047244382476285, 0.0695275561752371, -0.06247356464320168]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.2954911441584108, 0.2045088558415892, -0.1243897521811321]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.2904654518736746, 0.2095345481263255, -0.1928940065885978]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.2662334639376571, 0.23376653606234288, -0.17335174929230035]],
    [[0.5, 0.0, 0.0], [0.25, 0.25, -0.25], [0.2660431888425958, 0.2339568111574042, -0.2281075632608004]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.30277571741728104, 0.2862572127464075, -0.16631353218634315]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.2237819408208814, 0.0700882024789298, -0.05102720267932384]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.25949217146586573, 0.19743290044667056, -0.1844222508882279]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.34901091512646554, 0.13495225334267227, -0.05587199246904148]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.4091863318106685, 0.09100748886114335, -0.05025029020721522]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.26070494217198836, 0.1885373314033092, -0.05117244839304287]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.22319118420637762, 0.10903832110478374, -0.06159638230460483]],
    [[0.125, 0.125, 0.125], [0.325, 0.175, -0.075], [0.20009857398917028, 0.05459138023594126, -0.05226688557026751]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.2739568008259755, 0.2486988551516315, -0.11592812396737914]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.39922941006710444, 0.21762069792193095, -0.18795850494118854]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.2686907542950664, 0.2003566155668084, -0.19766127475715906]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.30614479247968723, 0.12830421075232734, -0.02933033909170113]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.4301950460637808, 0.11998640609688258, -0.10022554629478966]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.3072767753071623, 0.20385322616950455, -0.1011452938502487]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.39018177271047516, 0.1953814796535107, -0.1899220869959614]],
    [[0.125, 0.125, 0.125], [0.275, 0.125, -0.07500000000000001], [0.39851324856716985, 0.2482356115730413, -0.1999502620475364]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.3040887527473891, 0.175, -0.08627962343778446]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.30199569400760023, 0.14840986702967146, -0.12540556103727168]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.289777134601938, 0.2605449754654352, -0.22532211006737315]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.26995289506803144, 0.175, -0.018107405397135415]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.37467194245352703, 0.175, -0.11580028599331288]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.2821411058405471, 0.16926611218867707, -0.12640721802922408]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.29630253343285445, 0.1601932965395328, -0.13149582997238723]],
    [[0.125, 0.125, 0.125], [0.25, 0.05, 0.05], [0.27515214571148555, 0.175414573527298, -0.12556671923878354]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.37349598762126135, 0.125, -0.12349598762126134]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.18592546207115657, 0.12499999999999999, 0.06407453792884342]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.37408245347784264, 0.12500000000000006, -0.12408245347784262]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.30409087108787863, 0.125, -0.05409087108787863]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.27920471976630146, 0.12500000000000003, -0.029204719766301436]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.1624099959706824, 0.125, 0.0875900040293176]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.19866574954559213, 0.125, 0.05133425045440787]],
    [[0.125, 0.125, 0.125], [0.5, 0.0, 0.0], [0.14126264440667047, 0.1250000000000001, 0.10873735559332942]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, -0.08934669645474003]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25000000000000006, 0.24999999999999994, -0.19052301924393966]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.24999999999999997, -0.2346321121637948]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, -0.033236625117717064]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.24999999999999997, 0.25, -0.09619773187941977]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.24999999999999997, -0.22202764280922485]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, -0.19948505097242697]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.24999999999999997, -0.24603989212204572]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.35, 0.24152638919547942, -0.14798123860825185]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.35, 0.29513552860953496, -0.23252897044705836]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25412690431571006, 0.20507981605350153, -0.15]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.35, 0.13162083368762245, -0.04664611173745924]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.35, 0.05009205630854052, -0.04095743276265973]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2984212833849751, 0.23634095893065335, -0.15]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3499999999999999, 0.2584514330464684, -0.23394429621976837]],
    [[0.125, 0.125, 0.125], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.35, 0.34566905021595096, -0.24993121039283384]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2768252113788367, 0.19960323670753, -0.08499155785627706]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.39913016126054257, 0.1766662068166972, -0.14137220828151242]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.24230551217065702, 0.1780303157868085, -0.1487284268339823]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2839732529333215, 0.11730401308181027, -0.013879827924025707]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.4405441471363983, 0.10974203731073334, -0.1003168494878835]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.35361103580224423, 0.16882696957883986, -0.10159129293499941]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3882953346932103, 0.16024236890464552, -0.1430399053888712]],
    [[0.125, 0.125, 0.125], [0.275, 0.07499999999999998, -0.07499999999999998], [0.39850052338939085, 0.19876228746512697, -0.1499639380187649]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.375, 0.20381535582441546, -0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.37499999999999994, 0.25011127175117603, -0.12500000000000006]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.375, 0.13602080906214373, -0.12500000000000003]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.37499999999999994, 0.1288408433763184, -0.12499999999999994]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.375, 0.2211515036800228, -0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.375, 0.23067588328259991, -0.12499999999999999]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.37499999999999994, 0.20071641320439398, -0.12500000000000006]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, -0.25], [0.37499999999999994, 0.34300858984143456, -0.12499999999999992]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.38177512750135423, 0.23837776846702904, -0.11322954759457973]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.18401111398227712, 0.08083873875622435, -0.05121697853913318]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.34914709447221376, 0.24524421668069513, -0.21088766904273876]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.35548468128809874, 0.20658440485844462, -0.10252550869914964]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.3745011923066929, 0.10322845926723201, -0.07336545354085755]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.26193912851506845, 0.22422581495236665, -0.1063712242665847]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.3599817174730114, 0.22681841265657227, -0.214094707304477]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.325, 0.175, -0.075], [0.396575138041145, 0.24983480517324316, -0.20059084278176195]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.3440504534552241, 0.2091470125706495, -0.07801916000048015]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.2245397546368855, 0.07045327119184594, -0.057646002135976177]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.2566208812362611, 0.18677628074186398, -0.17629893282812598]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.36033704773498043, 0.20575895666104516, -0.11758154272530118]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.36125363066267563, 0.1102790277589219, -0.08191842632037111]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.301547467617896, 0.21353715998923636, -0.1341913205339003]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.3220429449069745, 0.19678180228358333, -0.12907163358715068]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.125, -0.07500000000000001], [0.34777604578113, 0.20036521392449766, -0.10101240795564428]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.2990539667518176, 0.20824595269581142, -0.07285001964010629]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.22064185768673972, 0.12722506442064319, -0.07335783934055747]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.2859900305288818, 0.15526084812279264, -0.1352399904163181]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.3205496990036343, 0.27443810150245895, -0.12111400422054841]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.4097483342485066, 0.20409360423116835, -0.18086006774817215]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.2889318681284246, 0.16794388403360133, -0.12552054298358734]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.2936045529219853, 0.18262062671714985, -0.1467603383900749]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.05, 0.05], [0.27503074705555397, 0.17563024104905983, -0.12597357050748814]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.3945294282403158, 0.20412525922499916, -0.08118278635167522]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.22634797313431176, 0.1413762674396617, -0.03798935409326999]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.42314311885710987, 0.12620773438092656, -0.07456912486825476]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.25491857282691577, 0.22356499670351793, -0.06458742529666078]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.2734911529595689, 0.15489557657086556, -0.07958955646842475]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.2370602929136083, 0.11295069131632496, 0.0013560856579999392]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.23389564675968533, 0.13518084610649528, -0.07835455519071777]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.5, 0.0, 0.0], [0.22508933708855317, 0.12600869142572002, -0.07607056009864811]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.3499999999999999, 0.18361896747630552, -0.12459557632557433]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.35, 0.25698019499597236, -0.2284649261794791]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.35, 0.25105800395148636, -0.24820352830386336]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.35, 0.31515429831761177, -0.12205009453385954]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.34999999999999987, 0.22171628174971333, -0.14664910020340574]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.27391397463347933, 0.22360824994798661, -0.1499999999999999]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.3368220585191234, 0.2240232178818802, -0.14999999999999988]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.125, 0.125, 0.125], [0.3486714060409625, 0.24984691847454385, -0.14999999999999994]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.4228234962086613, 0.1821976228813535, -0.1170949916802146]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.4086137130212095, 0.3136813958094826, -0.2857990131978876]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22833323342601575, 0.16488815323868367, -0.14319344686690177]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.39905084897784765, 0.20625295560506565, -0.11075680549217606]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3745843912696249, 0.10216361919896007, -0.08284815980708231]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2913838169805212, 0.25458803489208925, -0.17352910741137406]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.323183246725657, 0.14886051510229842, -0.1361637765939483]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3489973937805171, 0.05617233728924245, -0.055651334533023705]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.30325507867034784, 0.2061406803616619, -0.05307742873477289]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2512885285244144, 0.06758838296204428, -0.051639288163817076]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.25570871006024837, 0.22034513151943907, -0.1881902455701741]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.34857724317384386, 0.19789333026128755, -0.12428322672337586]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3497207260727666, 0.10642808261411052, -0.07412266712093274]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.33851629120497, 0.17362605007014803, -0.13042994729712049]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2986477146336451, 0.1922915229643133, -0.08555816651537307]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.275, 0.07499999999999998, -0.07499999999999998], [0.29859505323839625, 0.20054412352173584, -0.05121438263546145]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.4275544416261017, 0.13755196170012352, -0.079237106830244]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.4113420222123495, 0.271401091114978, -0.21681708008512413]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.2725850052959568, 0.17676401978446016, -0.11078652385849269]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.3421436374387711, 0.0813304033685692, -0.05445018144291516]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.39726705195383155, 0.17239032865165188, -0.09830370841322218]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.3537416691071988, 0.25187552370210337, -0.16982443898650745]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.3004356741930112, 0.18176030207250327, -0.07342331751685188]],
    [[0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25, 0.25, -0.25], [0.36911787957146314, 0.1754012630278565, -0.026514794103698863]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.26463011788014656, 0.24909995642503857, -0.052268628770587675]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.20947232635342905, 0.1453088457371865, -0.04635689900645225]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.3956068734135826, 0.1818997723012711, -0.1578278500704727]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.37458681710995395, 0.2195379828080835, -0.06941570295122573]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.3573912817008314, 0.12954211105191038, -0.06112967287337219]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.35712182747700083, 0.1572090839518955, -0.15165186368395786]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.23474677906419072, 0.12669662065191944, -0.09744263004078475]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.325, 0.175, -0.075], [0.15665752605364824, 0.14760240561460614, -0.05412869037313289]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.23620201128640755, 0.206226440417233, -0.028552524180745054]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.23069822352234964, 0.1508763863046467, -0.03147658563483055]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.3201622615844156, 0.15854625683878298, -0.12439812144918301]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.3474744160505784, 0.20192465069571075, -0.08857186645132897]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.3189334843340585, 0.14938174347209782, -0.09595559522666501]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.3105319201155529, 0.22065357430989235, -0.1824473325111466]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.25497523210357576, 0.18614110643911175, -0.13468902718459064]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.125, -0.07500000000000001], [0.25061176166984683, 0.15223677262503849, -0.10287481691066232]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.2500438656693429, 0.13491601393304345, -0.014698404788721205]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.2902914297090464, 0.16103148765476447, -0.05378090618375774]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.3752986414702448, 0.13014119211444444, -0.11186352176380773]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.32701202587366957, 0.2362782441686148, -0.07939888831703892]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.2952182968591686, 0.23960221216269634, -0.14375089584856787]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.28472890430266445, 0.2006012349489546, -0.16153815274224786]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.2973333490892134, 0.19093604598868796, -0.161652531767712]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.05, 0.05], [0.27551184451316, 0.22516122400711247, -0.17582750633103628]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.41037393731442495, 0.08061193778215736, -0.018957672681192923]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.27582070993811886, 0.2313994309431654, -0.060243042050231746]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.45652585391026496, 0.0851967988874345, -0.048443025522021704]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.29971648851540167, 0.26741745154912133, -0.08311704375772744]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.2954455325161495, 0.0834455532957627, -0.04554257458662486]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.23359554415508876, 0.08045595528776127, 0.008999914222757463]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.24008441297376928, 0.07099465979512773, -0.06480152113107232]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.5, 0.0, 0.0], [0.22563791922364998, 0.07602589207314422, -0.0755432974672102]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.2152176590821146, 0.19921224295914514, -0.06731525886824602]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.3495890649083117, 0.17102972954158796, -0.1398697911980918]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.37247423858807177, 0.16006908684126525, -0.1493051908255129]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.3956361490343377, 0.1902824710829017, -0.06699659263280783]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.29322427312487037, 0.17546672081454823, -0.08615385474777842]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.3536874363277844, 0.19217853657948603, -0.15310536572496586]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.3624469016135022, 0.1713763746167682, -0.10020665010590217]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.125, 0.125, 0.125], [0.3500966553843017, 0.20041594690163841, -0.10106599026975835]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.29960748480209676, 0.19824068172646764, -0.0607582374267923]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.22046229065605016, 0.1230232345535679, -0.0773302187186164]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.29743077482444547, 0.1279915754615258, -0.10830227981871322]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.39900429905088103, 0.1744175487318938, -0.0631794950151468]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2691241720575924, 0.14319636349048545, -0.017929529420821405]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2865669930385849, 0.1533668512456669, -0.11803426701474273]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.2546621306603556, 0.17128321392538945, -0.08044983620736582]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.20135028885500897, 0.1539840150996072, -0.005428844828251039]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.24049341628378526, 0.16244095632372535, -0.01248811295794823]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2540926486152494, 0.15699634683526947, -0.019148804072585915]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3203010417726545, 0.19808149135729003, -0.13527100539752335]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.32166989768655807, 0.2017244667698072, -0.10429697710965102]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3220562082711918, 0.15746790797884752, -0.12821226308377326]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2731875489250675, 0.26067151235352776, -0.1833350282849881]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.29176397877151033, 0.18339114155365394, -0.16471888290891384]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.275, 0.07499999999999998, -0.07499999999999998], [0.34794428943432293, 0.20004667058389985, -0.19966693031742327]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.32960866759800767, 0.1737790264146012, -0.04296313592900268]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.2589105902500626, 0.048586145579952805, -0.0423069271800032]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.25157850099002266, 0.19961303327387217, -0.10807855009461245]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.39673960729749863, 0.14579634123059987, -0.14105571195614247]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.300515021148911, 0.09046398622208263, -0.057739316428282804]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.36847764631913543, 0.23916635993259666, -0.2324431657521117]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.3053746374621116, 0.1603853225911016, -0.14185638019968883]],
    [[0.275, 0.07499999999999998, -0.07499999999999998], [0.25, 0.25, -0.25], [0.46311560813808367, 0.17497344921278035, -0.17446378781518318]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.4173891344458804, 0.12676914259205121, -0.05721306647030847]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.2379137638907931, 0.11677337632892783, -0.05924098947812638]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.3821028783407197, 0.18020711008983237, -0.12117660756835197]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.3076031207690436, 0.25677657580810653, -0.09111286204430646]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.3563291074803129, 0.10360231377259388, -0.08605922169965868]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.36473647910882867, 0.22073354787702515, -0.16219660888371745]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.2971341372424676, 0.22781551962795127, -0.12803107783893986]],
    [[0.25, 0.25, -0.25], [0.325, 0.175, -0.075], [0.32395728582993427, 0.17733294267597505, -0.07738009079147083]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.37426430249139075, 0.13240518180400024, -0.06109552869172218]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.22693795536302452, 0.11354926984736534, -0.06911661291794598]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.3588626707004545, 0.20575310311272738, -0.1457094789754979]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.3745552130320196, 0.24000590748120035, -0.12433142271487094]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.3691853822017206, 0.12744244769409094, -0.12522954669957823]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.3308032603100663, 0.24972465398346302, -0.17553352527801827]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.35432577713994046, 0.21910171755345678, -0.16442277974918149]],
    [[0.25, 0.25, -0.25], [0.275, 0.125, -0.07500000000000001], [0.42173584788670854, 0.17671364291952424, -0.1267324891280498]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.3134682863724784, 0.14002403506340527, -0.046208500448844794]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.30034356935301737, 0.09639483821244894, -0.09519776614065141]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.3995088533677285, 0.1608990792246531, -0.10002844365506644]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.3475940348951373, 0.2991042283094417, -0.12768366938841594]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.3120748454939747, 0.10665023217445711, -0.09977978301583672]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.31023952719413495, 0.22418449452572559, -0.13143945380433797]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.34212429066328265, 0.284911171329187, -0.1905196047423201]],
    [[0.25, 0.25, -0.25], [0.25, 0.05, 0.05], [0.3967967178269121, 0.29990975457517066, -0.1999854898162217]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.32017594786317116, 0.17982405213682878, -0.011724252520586065]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.2610848372499826, 0.23891516275001728, -0.1253322932724789]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.43500012713105174, 0.06499987286894837, -0.0007260038091997378]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.2577482809475834, 0.2422517190524166, -0.053208855396606344]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.3152641879129895, 0.1847358120870104, -0.11341194747743953]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.28738797883110945, 0.21261202116889039, -0.09300846932408313]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.27945855381096196, 0.22054144618903804, -0.1892441795421662]],
    [[0.25, 0.25, -0.25], [0.5, 0.0, 0.0], [0.25251599729514596, 0.24748400270485402, -0.2469713538609897]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.375, 0.14280366328582483, -0.125]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.375, 0.19982821263626116, -0.12499999999999999]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.375, 0.188969450604641, -0.12499999999999999]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.375, 0.3057313023094761, -0.125]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.375, 0.1798953769231767, -0.125]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.37500000000000006, 0.18262740932500288, -0.12500000000000006]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.37500000000000006, 0.260269973930621, -0.12500000000000003]],
    [[0.25, 0.25, -0.25], [0.125, 0.125, 0.125], [0.375, 0.35600381104400286, -0.125]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.449794014953915, 0.09992820836914426, -0.0951726853767724]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.25445831102505545, 0.07332036846079745, -0.060658842478770625]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3399349692365304, 0.22239805475934865, -0.16837001055514306]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.34658777776146527, 0.2536074606408431, -0.10722680546821178]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.33877572545493073, 0.12554989053075571, -0.05132063092330706]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.39259477515569224, 0.20667400183088264, -0.19141173821332708]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.30608189960268395, 0.24605954989718157, -0.09946026705956493]],
    [[0.25, 0.25, -0.25], [0.22499999999999998, 0.22500000000000003, -0.07500000000000002], [0.3717467309356389, 0.1782285205274817, -0.028268555245380217]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.33083597150946564, 0.14744971424361872, -0.05506892830760904]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.21520010063510664, 0.13172766996081642, -0.06975440007150902]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.3566436942535958, 0.2009636721083356, -0.1464614037854562]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.39054526233166104, 0.22602721383682617, -0.14036920240395617]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.32670131974164307, 0.14460906861456088, -0.10958370171123885]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.2959903396998185, 0.2692833396243885, -0.1728486922828723]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.38401004382908593, 0.2073957268717683, -0.19871586889606005]],
    [[0.25, 0.25, -0.25], [0.275, 0.07499999999999998, -0.07499999999999998], [0.32378993865895794, 0.03119454792657473, -0.03118799856821746]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.45173355300020496, 0.10668325125857198, -0.10668325125857198]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.42166867362563676, 0.2837451867831311, -0.2837451867831311]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.33522656865403083, 0.23048900125734623, -0.23048900125734623]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.4343215116182747, 0.16425418178648266, -0.16425418178648266]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.3631627854197298, 0.07217862690345897, -0.07217862690345897]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.26476567620236935, 0.12286528101841675, -0.12286528101841672]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.34586346192243295, 0.11319820712922424, -0.11319820712922424]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.472280390743829, 0.018754719216254656, -0.018754719216254656]],
]
# fmt: on


def generate(output):
    sections = [
        ("basic", lambda: np.asarray(BASIC_CASES)),
        ("stratified", lambda: generate_stratified(13, 3, 12648430, 4096)),
        ("linspace", lambda: gen_linspace(120)),
        ("haar", lambda: gen_haar(300_000)),
        ("targeted", lambda: generate_targeted()),
        ("captured", lambda: np.asarray(CAPTURED_CASES)),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        total = 0
        with temporary.open("wb") as stream:
            for name, construct in sections:
                rows = np.asarray(construct(), dtype="<f8")
                if (
                    rows.ndim != 3
                    or rows.shape[1:] != (3, 3)
                    or not np.isfinite(rows).all()
                ):
                    raise ValueError(f"invalid {name} cases")
                rows.tofile(stream)
                total += len(rows)
                print(f"{name}: {len(rows):,}", flush=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{output}: {total:,} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", type=Path, default=Path(__file__).with_name("cases.bin")
    )
    generate(parser.parse_args().output)
