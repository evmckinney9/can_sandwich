#!/usr/bin/env python3
"""Replay explicit-witness realization cases through the public GULPS pipeline."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

LOCAL_MODE_NAMES = (
    "identity",
    "left_haar",
    "right_haar",
    "both_haar",
    "correlated_haar",
    "near_identity",
)


def _phase_aligned_residual(actual: np.ndarray, target: np.ndarray) -> float:
    overlap = np.vdot(target, actual)
    if abs(overlap) == 0.0:
        return float("inf")
    aligned = actual * np.conj(overlap / abs(overlap))
    return float(np.max(np.abs(aligned - target)))


def _compile_target(c_point, g_point, target):
    """Run optional host integration only when a replay is requested."""
    from qiskit.quantum_info import Operator

    from gulps import GulpsDecomposer, LocalEquivalenceClass

    c_gate = LocalEquivalenceClass(list(c_point))
    g_gate = LocalEquivalenceClass(list(g_point))
    if c_gate == g_gate:
        decomposer = GulpsDecomposer([g_gate], [1.0])
    else:
        decomposer = GulpsDecomposer([g_gate, c_gate], [1.0, 1.0])
    circuit = decomposer.decompose(target)
    return _phase_aligned_residual(Operator(circuit).data, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument(
        "--sample-index",
        type=int,
        action="append",
        default=None,
        help="witness sample to replay; repeat the option to select more",
    )
    parser.add_argument(
        "--corpus-row",
        type=int,
        action="append",
        default=None,
        help="exact pipeline row to replay; repeat the option to select more",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--max-case-seconds",
        type=float,
        help="fail when any single public compilation exceeds this wall time",
    )
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive")
    if args.max_case_seconds is not None and (
        not np.isfinite(args.max_case_seconds) or args.max_case_seconds <= 0
    ):
        parser.error("--max-case-seconds must be finite and positive")
    if args.sample_index is not None and args.corpus_row is not None:
        parser.error("choose either --sample-index or --corpus-row")

    selected_samples = None if args.sample_index is None else set(args.sample_index)
    with np.load(args.corpus) as data:
        sample_indices = data["sample_indices"]
        if selected_samples is not None:
            missing = selected_samples - set(map(int, sample_indices))
            if missing:
                parser.error(f"unknown --sample-index values: {sorted(missing)}")
        if args.corpus_row is not None:
            selected = np.asarray(list(dict.fromkeys(args.corpus_row)), dtype=np.int64)
            if np.any(selected < 0) or np.any(selected >= len(sample_indices)):
                parser.error(f"--corpus-row must be in [0, {len(sample_indices)})")
        else:
            selected = (
                np.arange(len(sample_indices))
                if selected_samples is None
                else np.flatnonzero(np.isin(sample_indices, list(selected_samples)))
            )
        if args.max_cases is not None:
            selected = selected[: args.max_cases]
        if not len(selected):
            parser.error("selection contains no corpus rows")
        c_weyl = data["c_weyl"][selected]
        g_weyl = data["g_weyl"][selected]
        targets = data["target_unitaries"][selected]
        pair_codes = data["pair_codes"][selected]
        selected_sample_indices = sample_indices[selected]
        local_modes = data["local_modes"][selected]

    failure_examples: list[str] = []
    failed_rows: set[int] = set()
    failure_reasons: Counter[str] = Counter()
    failure_pairs: Counter[tuple[int, int]] = Counter()
    failure_samples: Counter[int] = Counter()
    failure_local_modes: Counter[str] = Counter()
    worst = 0.0
    timings: list[tuple[float, int, tuple[int, int], int, int]] = []
    attempted = 0
    for corpus_row, (c_point, g_point, target, pair, sample, local_mode) in zip(
        selected,
        zip(
            c_weyl,
            g_weyl,
            targets,
            pair_codes,
            selected_sample_indices,
            local_modes,
            strict=True,
        ),
        strict=True,
    ):
        attempted += 1
        reason: str | None = None
        detail: str | None = None
        started = time.perf_counter()
        try:
            residual = _compile_target(c_point, g_point, target)
            worst = max(worst, residual)
            if not np.isfinite(residual) or residual > args.tolerance:
                reason = "matrix residual"
                detail = f"residual {residual:.3e}"
        except Exception as exc:  # noqa: BLE001 - report every public failure together
            message = str(exc)
            if "outside the can_sandwich atlas" in message:
                reason = "outside atlas"
            elif "gate-lifted representative" in message:
                reason = "gate-lifted only"
            else:
                reason = type(exc).__name__
            detail = f"{type(exc).__name__}: {message}"
        timings.append(
            (
                time.perf_counter() - started,
                int(corpus_row),
                tuple(map(int, pair)),
                int(sample),
                int(local_mode),
            )
        )
        if reason is not None:
            failed_rows.add(int(corpus_row))
            failure_reasons[reason] += 1
            failure_pairs[tuple(map(int, pair))] += 1
            failure_samples[int(sample)] += 1
            failure_local_modes[LOCAL_MODE_NAMES[int(local_mode)]] += 1
            if len(failure_examples) < 20:
                failure_examples.append(f"pipeline row {corpus_row}: {detail}")

    timings.sort(key=lambda timing: timing[0], reverse=True)
    slowest_seconds, slowest_row, *_ = timings[0] if timings else (0.0, -1)
    if args.max_case_seconds is not None:
        for seconds, row, pair, sample, local_mode in timings:
            if seconds <= args.max_case_seconds:
                break
            failed_rows.add(row)
            failure_reasons["latency"] += 1
            failure_pairs[pair] += 1
            failure_samples[sample] += 1
            failure_local_modes[LOCAL_MODE_NAMES[local_mode]] += 1
            if len(failure_examples) < 20:
                failure_examples.append(
                    f"pipeline row {row}: latency {seconds:.3f}s exceeds "
                    f"{args.max_case_seconds:.3f}s"
                )
    if failed_rows:
        detail = "\n".join(failure_examples)
        slowest = "\n".join(
            f"pipeline row {row}: {seconds:.3f}s, pair {pair}, "
            f"sample {sample}, local mode {LOCAL_MODE_NAMES[local_mode]}"
            for seconds, row, pair, sample, local_mode in timings[:10]
        )
        summaries = (
            f"reasons: {dict(failure_reasons)}\n"
            f"local modes: {dict(failure_local_modes)}\n"
            f"sample indices: {dict(sorted(failure_samples.items()))}\n"
            f"most affected C/G family-code pairs: {failure_pairs.most_common(10)}\n"
            f"slowest cases:\n{slowest}"
        )
        raise AssertionError(
            f"public realization corpus found {len(failed_rows):,} failed rows in "
            f"{attempted:,} attempted rows (first {len(failure_examples)} shown):\n"
            f"{summaries}\n{detail}"
        )
    print(
        f"public realization corpus: {len(selected):,}/{len(selected):,} passed; "
        f"worst phase-aligned matrix residual {worst:.3e}; "
        f"slowest row {slowest_row} {slowest_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
