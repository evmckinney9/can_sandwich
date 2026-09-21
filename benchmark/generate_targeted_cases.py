#!/usr/bin/env python3
"""Generate independent planted witnesses for support and spectral degeneracies.

No production solver, Horn feasibility filter, or coordinate quantizer is used.
Every stored row must pass the public witness checker at 1e-12, stricter than
the submission acceptance tolerance. This is a targeted finite sample, not a
proof that all Horn facets or singular fibers have been covered.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from run import check_witness, spectrum


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
    """Invert the Weyl pairwise-sum chart without decimal rounding."""
    a, b, c = weyl
    return np.array([(a + b - c) / 2, (a - b + c) / 2, (-a + b + c) / 2])


def fold_spectrum(roots: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Select a chamber representative by phase lifts, permutations, and sign.

    Return monodromy coordinates, original-root ordering, and central sign.
    Values are not snapped to chamber walls: roundoff-sized excursions remain
    in the raw output and are checked through the independent matrix witness.
    """
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
    """Construct a real determinant-one coordinate-plane rotation."""
    i, j = pair
    result = np.eye(4)
    cosine, sine = np.cos(angle), np.sin(angle)
    result[i, i] = result[j, j] = cosine
    result[i, j], result[j, i] = -sine, sine
    return result


def dense_frame(rng: np.random.Generator) -> np.ndarray:
    """Construct a dense SO(4) frame as six independent plane rotations."""
    result = np.eye(4)
    for pair, angle in zip(PAIRS, rng.uniform(-np.pi, np.pi, 6)):
        result = result @ givens(pair, float(angle))
    return result


def diagonalize_symmetric_unitary(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a real joint eigenbasis of its commuting real and imaginary parts."""
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


def generate(
    seed: int = 20260921,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, float]]:
    """Return raw triples, planted frames, row provenance, and worst metrics."""
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    witnesses: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    maximum: dict[str, float] = {}

    def append(row: np.ndarray, frame: np.ndarray, family: str, **details: Any) -> None:
        frame = frame.copy()
        if np.linalg.det(frame) < 0:
            frame[:, 0] *= -1
        valid, metrics = check_witness(row, frame.tolist())
        if not valid or max(metrics.values()) > 1e-12:
            raise RuntimeError(
                f"planted witness failed for {family}: {details}: {metrics}"
            )
        for key, value in metrics.items():
            maximum[key] = max(maximum.get(key, 0.0), value)
        rows.append(row)
        witnesses.append(frame)
        records.append({"family": family, **details, "witness_metrics": metrics})

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

    return np.asarray(rows), np.asarray(witnesses), records, maximum


def write_corpus(
    output: Path, seed: int = 20260921, *, force: bool = False
) -> dict[str, Any]:
    """Write triples, independent witnesses, and deterministic provenance."""
    if output.suffix != ".npy":
        raise ValueError("The corpus output must have a .npy suffix.")
    witness_path = output.with_suffix(".witnesses.npz")
    metadata_path = output.with_suffix(".json")
    if not force:
        for path in (output, witness_path, metadata_path):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to replace {path}; use --force explicitly."
                )
    rows, witnesses, records, maximum = generate(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, rows, allow_pickle=False)
    np.savez_compressed(witness_path, o=witnesses)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "rows": len(rows),
        "construction": "independently planted SO(4) witnesses; no production solver or Horn filter",
        "coordinate_model": "unquantized binary64 phases; permutation and central-sign chamber folding",
        "witness_tolerance": 1e-12,
        "maximum_witness_metrics": maximum,
        "family_counts": dict(Counter(row["family"] for row in records)),
        "artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (output, witness_path)
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "cases": records,
        "limitations": [
            "Finite targeted coverage, not enumeration of all Horn facets or singular fibers.",
            "Binary64 spectral extraction may merge sub-ulp perturbations; actual target gaps are recorded.",
            "A strict independent numeric witness certificate is not an exact algebraic feasibility proof.",
        ],
    }
    metadata_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Generate a reproducible corpus at the requested path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "corpus/feasible_targeted.npy",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace existing corpus artifacts",
    )
    args = parser.parse_args()
    manifest = write_corpus(args.output, args.seed, force=args.force)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("rows", "family_counts", "maximum_witness_metrics")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
