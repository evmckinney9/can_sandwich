#!/usr/bin/env python3
"""Audit immutable source corpus coverage using NumPy, without calling a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
THRESHOLDS = (1e-14, 1e-12, 1e-8, 1e-4)
SOURCE_NAMES = ("feasible_stratified.npy", "feasible_linspace.npy", "feasible_haar.npy")


def sha256(path: Path) -> str:
    """Hash an artifact without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_stats(
    path: Path, expected: dict[str, Any], batch_size: int
) -> dict[str, Any]:
    """Measure floating spectral gaps after checking the registered source digest."""
    digest = sha256(path)
    if digest != expected["sha256"]:
        raise ValueError(f"source digest mismatch: {path}")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.shape != (expected["rows"], 3, 3) or data.dtype != np.float64:
        raise ValueError(f"source shape or dtype mismatch: {path}")
    counts = np.zeros((3, len(THRESHOLDS)), dtype=np.int64)
    minima = np.full(3, np.inf)
    low, high = np.inf, -np.inf
    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        if not np.isfinite(rows).all():
            raise ValueError(f"nonfinite source coordinate: {path}")
        low, high = min(low, rows.min()), max(high, rows.max())
        turns = np.stack(
            [rows[:, :, 1], rows[:, :, 0], -rows.sum(axis=2), rows[:, :, 2]], axis=-1
        )
        roots = np.exp(2j * np.pi * turns)
        gaps = np.stack(
            [
                abs(roots[:, :, i] - roots[:, :, j])
                for i in range(4)
                for j in range(i + 1, 4)
            ],
            axis=-1,
        ).min(axis=-1)
        minima = np.minimum(minima, gaps.min(axis=0))
        counts += (gaps[:, :, None] <= np.array(THRESHOLDS)).sum(axis=0)
    result = {
        "path": expected["path"],
        "rows": len(data),
        "sha256": digest,
        "registered_hash_verified": True,
        "finite": True,
        "coordinate_min_max": [float(low), float(high)],
        "spectral_gap_counts": {
            label: {
                **{str(t): int(counts[c, i]) for i, t in enumerate(THRESHOLDS)},
                "minimum": float(minima[c]),
            }
            for c, label in enumerate(("C", "G", "T"))
        },
    }
    if path.name == "feasible_linspace.npy":
        values = np.unique(data)
        result.update(
            unique_scalar_values=len(values),
            scalar_values=values.tolist(),
            all_coordinates_multiples_of_one_over_32=bool(
                np.all(values * 32 == np.rint(values * 32))
            ),
            unique_C_G_T=[len(np.unique(data[:, c], axis=0)) for c in range(3)],
        )
    return result


def strata_stats(directory: Path) -> dict[str, Any]:
    """Read source labels and check the original metadata's artifact digests."""
    metadata_path = directory / "feasible_stratified.json"
    metadata = json.loads(metadata_path.read_text())
    for name, expected in metadata["artifact_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"stratified artifact digest mismatch: {name}")
    codes = np.load(directory / metadata["strata_file"], allow_pickle=False)
    sections = {}
    for section in (-1, 0, 1):
        rows = codes[codes[:, 3] == section]
        _, frequencies = np.unique(rows[:, :2], axis=0, return_counts=True)
        sections[str(section)] = {
            "rows": len(rows),
            "families_per_C_G_T": [sorted(set(map(int, rows[:, c]))) for c in range(3)],
            "distinct_C_G_pairs": len(frequencies),
            "rows_per_C_G_pair_min_max": [
                int(frequencies.min()),
                int(frequencies.max()),
            ],
            "distinct_C_G_T_family_triples": len(np.unique(rows[:, :3], axis=0)),
        }
    with np.load(directory / metadata["pipeline_file"], allow_pickle=False) as pipeline:
        modes, counts = np.unique(pipeline["local_modes"], return_counts=True)
        pipeline_stats = {
            "rows": len(pipeline["sample_indices"]),
            "local_mode_counts": {
                metadata["local_modes"][int(m)]: int(n)
                for m, n in zip(modes, counts, strict=True)
            },
            "sample_modes": {
                str(i): sorted(
                    set(
                        map(
                            int,
                            pipeline["local_modes"][pipeline["sample_indices"] == i],
                        )
                    )
                )
                for i in sorted(set(pipeline["sample_indices"]))
            },
        }
    return {
        "metadata_sha256": sha256(metadata_path),
        "artifact_sha256": metadata["artifact_sha256"],
        "generation": {
            k: metadata[k]
            for k in (
                "corpus_version",
                "seed",
                "samples_per_pair",
                "target_rounds",
                "target_stratum_candidates",
                "horn_tolerance",
            )
        },
        "families": metadata["families"],
        "near_scales": metadata["near_scales"],
        "sections": sections,
        "pipeline": pipeline_stats,
    }


def audit(manifest_path: Path, batch_size: int) -> dict[str, Any]:
    """Validate and scan the three historical source arrays without modifying them."""
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["profiles"]["extended"]["corpora"]
    selected = {
        Path(entry["path"]).name: entry
        for entry in entries
        if Path(entry["path"]).name in SOURCE_NAMES
    }
    if set(selected) != set(SOURCE_NAMES):
        raise ValueError("manifest must register all three original source arrays")
    sources = {
        name: source_stats(
            manifest_path.parent / selected[name]["path"], selected[name], batch_size
        )
        for name in SOURCE_NAMES
    }
    directory = (manifest_path.parent / selected[SOURCE_NAMES[0]]["path"]).parent
    return {
        "schema_version": 1,
        "provenance": {
            "audit_script": "audit_corpus.py",
            "audit_script_sha256": sha256(Path(__file__)),
            "source_manifest": manifest_path.name,
            "source_manifest_sha256": sha256(manifest_path),
        },
        "method": "NumPy binary64 unit-circle chord gaps between all six eigenvalue pairs; thresholds are empirical, not exact multiplicity certificates. No feasibility solver is called.",
        "total_original_rows": sum(value["rows"] for value in sources.values()),
        "sources": sources,
        "stratified": strata_stats(directory),
        "interpretation_limits": [
            "Haar and linspace names do not establish their generation distribution; their manifest entries contain no generator or seed.",
            "The stratified witness section crosses all C/G families; each sample index fixes both a near scale and a local mode, not their full Cartesian product.",
            "The target-stratum section is filtered by live Horn inequalities at the recorded tolerance; it supplies no independent planted witness and does not establish coverage of every Horn facet.",
            "Public LocalEquivalenceClass construction and matrix extraction quantize coordinates in the stratified generator; near-scale labels do not prove that every intended gap survives target extraction.",
            "Finite corpus coverage is not a mathematical completeness claim. Source rows and hashes are preserved.",
        ],
    }


def main() -> None:
    """Write the reproducible source audit to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    preferred = HERE / "source_corpora.json"
    parser.add_argument(
        "--manifest",
        type=Path,
        default=preferred if preferred.exists() else HERE / "corpora.json",
    )
    parser.add_argument("--output", type=Path, default=HERE / "coverage_sources.json")
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    report = audit(args.manifest, args.batch_size)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"audited {report['total_original_rows']:,} immutable source rows: {args.output}"
    )


if __name__ == "__main__":
    main()
