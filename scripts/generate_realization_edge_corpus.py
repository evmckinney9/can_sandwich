#!/usr/bin/env python3
"""Generate deterministic realization cases concentrated on singular geometry.

The corpus has two independently generated sections.

The witness section forms each target as ``Can(C) @ (A x B) @ Can(G)``. This
gives an explicit realization witness and does not use the reachability or
realization code under test. It crosses every exact and near Weyl stratum for
``C`` and ``G`` over a scale ladder and several structured local layers.

The target-stratum section samples ``C``, ``G``, and ``T`` independently from
the same strata. It retains targets accepted by the live Horn region for the
two-gate sentence. This section deliberately puts the target, not only the two
inputs, on exact and near spectral strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from gulps import GateInvariants
from gulps._accelerate import invariants


CORPUS_VERSION = 5
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
REGRESSION_LABELS = (
    "near-swap repeated-gate witness",
    "exact-double target 4101",
    "exact-double target 4102",
    "near-pair target 4660",
    "quantized face target 6363",
    "clustered proxy certificate 4673",
    "near-identity edge target 8497",
    "near-swap edge target 9082",
    "repeated-root multiplicity 9380",
    "repeated-root multiplicity 9393",
    "near-identity vertex target 8639",
    "public alternate witness 9482",
    "repeated-input caustic 9482",
    "seeded near-identity recovery 5861",
    "seeded repeated-word latency 10151",
)
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
PIPELINE_REGRESSION_PAIR_CODES = np.array([[15, 15], [26, 26]], dtype=np.int16)
PIPELINE_REGRESSION_SAMPLE_INDICES = np.array([11, 11], dtype=np.int16)
PIPELINE_REGRESSION_LOCAL_MODES = np.array([5, 5], dtype=np.int16)


def _simplex(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.dirichlet(np.ones(size))


def _weights(
    family: str,
    sample: int,
    rng: np.random.Generator,
    near_scale: float | None = None,
) -> np.ndarray:
    """Return exact or scale-controlled barycentric stratum coordinates."""
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
    """Return structured SU(2) pairs and their mode codes."""
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
    classes = GateInvariants.from_unitaries(list(unitaries))
    return np.asarray([cls._monodromy for cls in classes], dtype=float)


def _canonical_matrices(points: np.ndarray) -> np.ndarray:
    return np.asarray([GateInvariants(list(point)).matrix for point in points])


def _witness_section(
    samples_per_pair: int,
    rng: np.random.Generator,
    batch_size: int,
    near_scale: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
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
    a, b, local_modes = _local_layers(rng, sample_indices)
    local = np.einsum("nij,nkl->nikjl", a, b).reshape(count, 4, 4)
    c_mono = _monodromy_from_weyl(c_weyl)
    g_mono = _monodromy_from_weyl(g_weyl)
    targets = np.empty((count, 3))
    target_unitaries = np.empty((count, 4, 4), dtype=np.complex128)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        unitary = (
            _canonical_matrices(c_weyl[start:stop])
            @ local[start:stop]
            @ _canonical_matrices(g_weyl[start:stop])
        )
        target_unitaries[start:stop] = unitary
        targets[start:stop] = _monodromy_from_unitaries(unitary)

    triples = np.stack((c_mono, g_mono, targets), axis=1)
    # Target code -1 denotes an explicit-witness target. The fourth column
    # identifies the section: 0 = witness, 1 = target stratum.
    codes = np.column_stack(
        (
            pair_codes[:, :2],
            np.full(count, -1, dtype=np.int16),
            np.zeros(count, dtype=np.int16),
        )
    )
    pipeline = {
        "c_weyl": c_weyl,
        "g_weyl": g_weyl,
        "target_unitaries": target_unitaries,
        "pair_codes": pair_codes[:, :2],
        "sample_indices": pair_codes[:, 2],
        "local_modes": local_modes,
    }
    return triples, codes, pipeline


def _contains(facets: np.ndarray, point: np.ndarray) -> bool:
    return bool(np.all(facets[:, :3] @ point >= facets[:, 3] - HORN_TOL))


def _target_section(
    rounds: int,
    rng: np.random.Generator,
    near_scale: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    triples: list[np.ndarray] = []
    codes: list[tuple[int, int, int, int]] = []
    candidates = 0
    for c_code, c_family in enumerate(FAMILIES):
        for g_code, g_family in enumerate(FAMILIES):
            for round_index in range(rounds):
                key = round_index + 3 * c_code + 5 * g_code
                c_weyl = _weights(c_family, key, rng, near_scale) @ VERTICES
                g_weyl = _weights(g_family, key + c_code, rng, near_scale) @ VERTICES
                c_mono, g_mono = _monodromy_from_weyl(np.asarray([c_weyl, g_weyl]))
                flat = np.asarray(invariants.sentence_facets([c_mono.tolist(), g_mono.tolist()]), dtype=float)
                facets = flat.reshape(-1, 4)
                for t_code, t_family in enumerate(FAMILIES):
                    candidates += 1
                    t_weyl = (
                        _weights(t_family, key + 7 * t_code, rng, near_scale) @ VERTICES
                    )
                    t_mono = _monodromy_from_weyl(t_weyl[None])[0]
                    if not _contains(facets, t_mono):
                        weyl = invariants.weyl_from_monodromy(t_mono[None])[0]
                        reflected = np.asarray(
                            GateInvariants(weyl.tolist())._rho_reflect._monodromy,
                            dtype=float,
                        )
                        if not _contains(facets, reflected):
                            continue
                        t_mono = reflected
                    triples.append(np.stack((c_mono, g_mono, t_mono)))
                    codes.append((c_code, g_code, t_code, 1))
    if not triples:
        raise RuntimeError("the target-stratum generator found no feasible rows")
    return np.asarray(triples), np.asarray(codes, dtype=np.int16), candidates


def generate(
    samples_per_pair: int,
    target_rounds: int,
    seed: int,
    batch_size: int,
    near_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Return the corpus, row codes, public-pipeline cases, and section sizes."""
    if samples_per_pair < 1:
        raise ValueError("samples_per_pair must be positive")
    if target_rounds < 1:
        raise ValueError("target_rounds must be positive")
    rng = np.random.default_rng(seed)
    witness, witness_codes, pipeline = _witness_section(
        samples_per_pair, rng, batch_size, near_scale
    )
    target, target_codes, target_candidates = _target_section(
        target_rounds, rng, near_scale
    )
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
    known_count = len(REGRESSION_TRIPLES) + len(pipeline_regression_triples)
    known_code = np.full((known_count, 4), -1, dtype=np.int16)
    codes = np.concatenate((known_code, witness_codes, target_codes))
    pipeline_regressions = {
        "c_weyl": PIPELINE_REGRESSION_C_WEYL,
        "g_weyl": PIPELINE_REGRESSION_G_WEYL,
        "target_unitaries": PIPELINE_REGRESSION_TARGETS,
        "pair_codes": PIPELINE_REGRESSION_PAIR_CODES,
        "sample_indices": PIPELINE_REGRESSION_SAMPLE_INDICES,
        "local_modes": PIPELINE_REGRESSION_LOCAL_MODES,
    }
    pipeline = {
        key: np.concatenate((pipeline_regressions[key], values))
        for key, values in pipeline.items()
    }
    counts = {
        "known_regressions": known_count,
        "explicit_witness_rows": len(pipeline["target_unitaries"]),
        "generated_witness_rows": len(witness),
        "target_stratum_rows": len(target),
        "target_stratum_candidates": target_candidates,
    }
    return triples, codes, pipeline, counts


def _sidecar_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output.with_name(f"{output.stem}.strata.npy"),
        output.with_suffix(".json"),
        output.with_name(f"{output.stem}.pipeline.npz"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _existing_is_current(
    output: Path,
    samples_per_pair: int,
    target_rounds: int,
    seed: int,
    near_scale: float | None,
) -> bool:
    code_path, metadata_path, pipeline_path = _sidecar_paths(output)
    if not all(
        path.is_file() for path in (output, code_path, metadata_path, pipeline_path)
    ):
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        triples = np.load(output, mmap_mode="r")
        codes = np.load(code_path, mmap_mode="r")
        with np.load(pipeline_path) as pipeline:
            pipeline_rows = len(pipeline["target_unitaries"])
    except (KeyError, OSError, ValueError):
        return False
    configuration_matches = (
        metadata.get("corpus_version") == CORPUS_VERSION
        and metadata.get("samples_per_pair") == samples_per_pair
        and metadata.get("target_rounds") == target_rounds
        and metadata.get("seed") == seed
        and metadata.get("near_scale_override") == near_scale
    )
    rows = metadata.get("rows")
    witness_rows = metadata.get("explicit_witness_rows")
    artifacts_match = (
        triples.ndim == 3
        and triples.shape[1:] == (3, 3)
        and len(triples) == rows
        and codes.shape == (rows, 4)
        and pipeline_rows == witness_rows
    )
    expected_hashes = metadata.get("artifact_sha256")
    hashes_match = expected_hashes == {
        output.name: _sha256(output),
        code_path.name: _sha256(code_path),
        pipeline_path.name: _sha256(pipeline_path),
    }
    if not (configuration_matches and artifacts_match and hashes_match):
        return False
    family_codes = set(range(len(FAMILIES)))
    return all(
        family_codes <= set(map(int, codes[codes[:, 3] >= 0, column]))
        for column in range(3)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--samples-per-pair", type=int, default=DEFAULT_SAMPLES_PER_PAIR
    )
    parser.add_argument("--target-rounds", type=int, default=DEFAULT_TARGET_ROUNDS)
    parser.add_argument("--seed", type=int, default=0xC0FFEE)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--near-scale",
        type=float,
        help="override the near-stratum scale for focused diagnostic sweeps",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="exit successfully only when all sidecars match the current defaults",
    )
    args = parser.parse_args()

    if args.check_existing:
        if _existing_is_current(
            args.output,
            args.samples_per_pair,
            args.target_rounds,
            args.seed,
            args.near_scale,
        ):
            print(f"current realization corpus: {args.output}")
            return
        print(f"missing or stale realization corpus: {args.output}", file=sys.stderr)
        raise SystemExit(1)

    triples, codes, pipeline, counts = generate(
        args.samples_per_pair,
        args.target_rounds,
        args.seed,
        args.batch_size,
        args.near_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, triples)
    code_path, metadata_path, pipeline_path = _sidecar_paths(args.output)
    np.save(code_path, codes)
    np.savez(pipeline_path, **pipeline)
    artifact_sha256 = {
        args.output.name: _sha256(args.output),
        code_path.name: _sha256(code_path),
        pipeline_path.name: _sha256(pipeline_path),
    }
    metadata = {
        "corpus_version": CORPUS_VERSION,
        "seed": args.seed,
        "samples_per_pair": args.samples_per_pair,
        "target_rounds": args.target_rounds,
        "rows": len(triples),
        **counts,
        "families": FAMILIES,
        "local_modes": LOCAL_MODES,
        "near_scales": NEAR_SCALES.tolist(),
        "near_scale_override": args.near_scale,
        "horn_tolerance": HORN_TOL,
        "strata_file": code_path.name,
        "pipeline_file": pipeline_path.name,
        "artifact_sha256": artifact_sha256,
        "code_columns": ["c_family", "g_family", "t_family", "section"],
        "sections": {
            "-1": "known regression",
            "0": "explicit witness",
            "1": "target stratum",
        },
        "regression_labels": REGRESSION_LABELS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"wrote {len(triples):,} reachable triples to {args.output} "
        f"({counts['explicit_witness_rows']:,} explicit witnesses, "
        f"{counts['target_stratum_rows']:,} target-stratum rows)"
    )


if __name__ == "__main__":
    main()
