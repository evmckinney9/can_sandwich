#!/usr/bin/env python3
"""Combine audited corpus sources without changing any binary64 coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
BLOCK_ROWS = 65536


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        array.dtype != np.dtype("<f8")
        or array.ndim != 3
        or array.shape[1:] != (3, 3)
        or not len(array)
    ):
        raise ValueError(
            f"source must be a nonempty little-endian binary64 (N,3,3) array: {path}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"source contains nonfinite coordinates: {path}")
    return array


def _text(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()


def _atomic_text(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_corpus(
    source_manifest: Path,
    output: Path,
    additional: Sequence[Path] = (),
    *,
    force: bool = False,
    manifest_output: Path | None = None,
) -> dict[str, Any]:
    """Validate source hashes, concatenate rows, and audit every output slice."""
    source_manifest = source_manifest.resolve(strict=True)
    output = output.resolve()
    provenance_path = output.with_suffix(".json")
    if output.suffix != ".npy":
        raise ValueError("the output must have a .npy extension")
    source_document = json.loads(source_manifest.read_text())
    if source_document.get("schema_version") != 1:
        raise ValueError("the source manifest must use legacy schema version 1")
    entries = source_document["profiles"]["extended"]["corpora"]
    if not entries:
        raise ValueError("the source manifest must list at least one corpus")
    sources = []
    start = 0
    for entry, kind in [(entry, "legacy") for entry in entries] + [
        ({"path": str(path.resolve())}, "additional") for path in additional
    ]:
        path = (source_manifest.parent / entry["path"]).resolve(strict=True)
        digest = sha256(path)
        if kind == "legacy" and ("sha256" not in entry or digest != entry["sha256"]):
            raise ValueError(f"source digest mismatch: {path}")
        array = _load(path)
        if kind == "legacy" and entry.get("rows") != len(array):
            raise ValueError(f"source row count mismatch: {path}")
        stop = start + len(array)
        sources.append(
            {
                "path": path,
                "sha256": digest,
                "rows": len(array),
                "start": start,
                "stop": stop,
                "kind": kind,
                "array": array,
            }
        )
        start = stop
    manifest_path = manifest_output.resolve() if manifest_output is not None else None
    targets = [output, provenance_path] + ([manifest_path] if manifest_path else [])
    protected = {source_manifest} | {source["path"] for source in sources}
    if len(set(targets)) != len(targets) or protected.intersection(targets):
        raise ValueError(
            "output paths must be distinct and must not overwrite a source"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        destination = np.lib.format.open_memmap(
            temporary, mode="w+", dtype="<f8", shape=(start, 3, 3)
        )
        for source in sources:
            for offset in range(0, source["rows"], BLOCK_ROWS):
                count = min(BLOCK_ROWS, source["rows"] - offset)
                destination[
                    source["start"] + offset : source["start"] + offset + count
                ] = source["array"][offset : offset + count]
        destination.flush()
        del destination
        checked = _load(temporary)
        for source in sources:
            for offset in range(0, source["rows"], BLOCK_ROWS):
                count = min(BLOCK_ROWS, source["rows"] - offset)
                actual = checked[
                    source["start"] + offset : source["start"] + offset + count
                ].view(np.uint64)
                expected = source["array"][offset : offset + count].view(np.uint64)
                if not np.array_equal(actual, expected):
                    raise RuntimeError(
                        f"bitwise slice verification failed: {source['path']} at row {offset}"
                    )
        del checked
        digest = sha256(temporary)
        provenance = {
            "schema_version": 1,
            "array": output.name,
            "rows": start,
            "shape": [start, 3, 3],
            "dtype": "<f8",
            "sha256": digest,
            "bitwise_slices_verified": True,
            "range_convention": "start inclusive, stop exclusive",
            "source_manifest": os.path.relpath(source_manifest, provenance_path.parent),
            "source_manifest_sha256": sha256(source_manifest),
            "sources": [
                {
                    key: os.path.relpath(value, provenance_path.parent)
                    if key == "path"
                    else value
                    for key, value in source.items()
                    if key != "array"
                }
                for source in sources
            ],
            "claim": "All source coordinates are preserved bitwise. Additional inputs must be audited separately; concatenation does not establish feasibility.",
        }
        manifest_base = manifest_path.parent if manifest_path else HERE
        manifest = {
            "schema_version": 2,
            "corpus": {
                "path": os.path.relpath(output, manifest_base),
                "rows": start,
                "sha256": digest,
                "provenance": os.path.relpath(provenance_path, manifest_base),
            },
        }
        documents = [(provenance_path, _text(provenance))]
        if manifest_path:
            documents.append((manifest_path, _text(manifest)))
        if not force:
            if output.exists() and sha256(output) != digest:
                raise FileExistsError(
                    f"refusing to replace a different corpus: {output}; use --force"
                )
            for path, contents in documents:
                if path.exists() and path.read_bytes() != contents:
                    raise FileExistsError(
                        f"refusing to replace different metadata: {path}; use --force"
                    )
        if not output.exists() or sha256(output) != digest:
            temporary.replace(output)
        for path, contents in documents:
            if not path.exists() or path.read_bytes() != contents:
                _atomic_text(path, contents)
        return {
            "output": str(output),
            "provenance": str(provenance_path),
            "rows": start,
            "sha256": digest,
            "source_count": len(sources),
            "bitwise_slices_verified": True,
            "manifest": manifest,
        }
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """Build one combined corpus and optionally write its runner manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path, default=HERE / "source_corpora.json"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE.parent / "corpus" / "feasible_all.npy"
    )
    parser.add_argument("--additional", type=Path, action="append", default=[])
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace different existing output artifacts",
    )
    args = parser.parse_args()
    try:
        result = build_corpus(
            args.source_manifest,
            args.output,
            args.additional,
            force=args.force,
            manifest_output=args.manifest_output,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
