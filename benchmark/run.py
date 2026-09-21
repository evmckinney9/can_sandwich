#!/usr/bin/env python3
"""Check and time a single Python or Rust can-sandwich solver submission."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import select
import signal
import subprocess
import time
from typing import Any

import numpy as np

PROTOCOL = "can-sandwich-v1"
TOLERANCE = 1e-8
MAX_LINE_BYTES = 65536
PERMUTATIONS = np.asarray(list(itertools.permutations(range(4))))


class ProtocolError(Exception):
    """The solver violated the JSONL protocol."""


def _json(line: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        return json.loads(line, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc


class SolverProcess:
    """Keep a solver alive between cases and enforce bounded line reads."""

    def __init__(self, command: list[str], startup_timeout: float = 10) -> None:
        """Start in the caller's working directory, with inherited stderr."""
        self.buffer = bytearray()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            start_new_session=True,
            bufsize=0,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            ready = _json(self._read_line(time.perf_counter() + startup_timeout))
            if (
                not isinstance(ready, dict)
                or ready.get("protocol") != PROTOCOL
                or ready.get("ready") is not True
            ):
                raise ProtocolError("expected protocol ready handshake")
        except BaseException:
            self.close()
            raise

    def _read_line(self, deadline: float) -> bytes:
        assert self.process.stdout is not None
        descriptor = self.process.stdout.fileno()
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                if newline + 1 > MAX_LINE_BYTES:
                    raise ProtocolError("response exceeds 64 KiB")
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                return line
            if len(self.buffer) >= MAX_LINE_BYTES:
                raise ProtocolError("response exceeds 64 KiB")
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
                raise TimeoutError("solver response deadline exceeded")
            chunk = os.read(descriptor, MAX_LINE_BYTES)
            if not chunk:
                raise ProtocolError("solver closed stdout before a complete response")
            self.buffer.extend(chunk)

    def request(
        self, request: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], float]:
        """Time request transmission through receipt of the complete response."""
        assert self.process.stdin is not None
        started = time.perf_counter()
        deadline = started + timeout
        payload = memoryview((json.dumps(request, allow_nan=False) + "\n").encode())
        descriptor = self.process.stdin.fileno()
        while payload:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
                raise TimeoutError("solver request deadline exceeded")
            try:
                sent = os.write(descriptor, payload)
            except BrokenPipeError as exc:
                raise ProtocolError("solver closed stdin") from exc
            payload = payload[sent:]
        response = _json(self._read_line(deadline))
        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            raise TimeoutError("solver response deadline exceeded")
        if not isinstance(response, dict):
            raise ProtocolError("response must be an object")
        if type(response.get("id")) is not int or response["id"] != request["id"]:
            raise ProtocolError("response id does not match request")
        if response.get("status") not in ("solved", "declined", "error"):
            raise ProtocolError("response status must be solved, declined, or error")
        return response, elapsed

    def close(self) -> None:
        """Kill the complete process group and reap the solver."""
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()


def spectrum(monodromy: np.ndarray) -> np.ndarray:
    """Convert the original floating coordinates to magic-basis eigenvalues."""
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
    """Check SO(4) and the full eigenvalue multiset independently of the solver."""
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
        raise ProtocolError("o must be a real numeric 4x4 JSON array")
    try:
        orthogonal = np.asarray(witness, dtype=float)
    except (ValueError, OverflowError) as exc:
        raise ProtocolError("o must contain finite real numbers") from exc
    if not np.all(np.isfinite(orthogonal)):
        raise ProtocolError("o must contain finite real numbers")
    gram = float(np.max(np.abs(orthogonal.T @ orthogonal - np.eye(4))))
    determinant = float(abs(np.linalg.det(orthogonal) - 1))
    metrics = {"orthogonality_max": gram, "determinant_error": determinant}
    if not np.isfinite(gram) or not np.isfinite(determinant):
        raise ProtocolError("o produces nonfinite matrix metrics")
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
        raise ProtocolError("o produces nonfinite spectral metrics")
    metrics["spectral_bottleneck"] = spectral
    return all(value <= TOLERANCE for value in metrics.values()), metrics


def print_summary(report: dict[str, Any], report_path: Path) -> None:
    """Show coverage and timing while leaving full details in the report files."""

    def duration(seconds: float) -> str:
        if seconds < 0.001:
            return f"{seconds * 1e6:.1f} us"
        if seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        return f"{seconds:.2f} s"

    label = report["label"] or (
        Path(report["submission"]["source_path"]).name
        if report["submission"]
        else "solver"
    )
    counts = report["counts"]
    total = sum(corpus["rows"] for corpus in report["corpora"])
    checked = report["attempted_cases"]
    print(f"\n{label}")
    print(
        f"Passed {counts['passed']:,} / {checked:,} checked cases ({counts['passed'] / checked:.4%})"
    )
    if checked != total:
        print(f"Partial run: checked {checked:,} of {total:,} corpus cases")
    if report.get("stride", 1) != 1:
        print(
            f"Sampling step: {report['stride']:,} (rows 0, {report['stride']:,}, ...)"
        )
    outcome_labels = {
        "failed": "Incorrect result",
        "declined": "No solution returned",
        "timeout": "Timed out",
        "invalid": "Invalid response",
        "error": "Error",
    }
    print(
        "  ".join(
            f"{label}: {counts[name]:,}" for name, label in outcome_labels.items()
        )
    )
    if report["unattempted_cases"]:
        print(f"Not attempted: {report['unattempted_cases']:,}")
    print(f"Full coverage: {'yes' if report['full_coverage'] else 'no'}")
    print(f"Speed ranking eligible: {'yes' if report['ranking_eligible'] else 'no'}")
    print("\nPer-case time (includes transport)")
    timing = report["timing_seconds"]
    print(
        "  ".join(
            f"{key}: {duration(timing[key])}" for key in ("median", "p95", "p99", "max")
        )
    )
    print(
        f"Total case time: {duration(timing['sum'])}  |  Wall time: {duration(report['total_wall_seconds'])}"
    )
    print(f"\nReport: {report_path.resolve()}")
    print(f"Cases:  {report['case_results']}")


def sha256(path: Path) -> str:
    """Hash an artifact without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpora(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    """Validate all corpus files and registered digests before starting a solver."""
    if args.corpus:
        entries = [{"path": str(path.resolve())} for path in args.corpus]
        profile = "custom"
        base = Path.cwd()
    else:
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("schema_version") != 2:
            raise ValueError("unsupported corpus manifest schema")
        entries = [manifest["corpus"]]
        profile = "full"
        base = args.manifest.resolve().parent
    if not entries:
        raise ValueError("at least one corpus artifact is required")
    corpora = []
    for entry in entries:
        path = (base / entry["path"]).resolve()
        if not path.is_file():
            raise ValueError(
                f"missing corpus {path}; install feasible_all.npy or rebuild it with benchmark/build_corpus.py from the registered sources"
            )
        digest = sha256(path)
        if "sha256" in entry and digest != entry["sha256"]:
            raise ValueError(f"corpus digest mismatch: {path}")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            data.dtype != np.dtype("float64")
            or data.ndim != 3
            or data.shape[1:] != (3, 3)
            or not len(data)
        ):
            raise ValueError(
                f"corpus must be a nonempty binary64 (N,3,3) array: {path}"
            )
        if "rows" in entry and len(data) != entry["rows"]:
            raise ValueError(f"corpus row count mismatch: {path}")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"corpus contains nonfinite coordinates: {path}")
        corpora.append(
            {"path": str(path), "sha256": digest, "rows": len(data), "data": data}
        )
    return corpora, profile


def main() -> int:
    """Stream case results and publish coverage and timing for the combined corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("corpora.json")
    )
    parser.add_argument("--corpus", type=Path, action="append")
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=10)
    parser.add_argument("--row", type=int, action="append")
    parser.add_argument(
        "--step",
        "--stride",
        dest="stride",
        type=int,
        default=1,
        help="check every Nth row starting at row 0 (default: 1, all cases)",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label", help="solver version or revision label for comparison"
    )
    parser.add_argument("--report", type=Path, default=Path("benchmark-report.json"))
    parser.add_argument(
        "--cases", type=Path, help="case JSONL path, default REPORT.cases.jsonl"
    )
    parser.add_argument(
        "submission",
        type=Path,
        nargs="?",
        help="one .py or .rs file defining solve(c, g, t)",
    )
    parser.add_argument("--command", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    command = args.command or []
    if command[:1] == ["--"]:
        command = command[1:]
    if bool(args.submission) == bool(command):
        parser.error("provide one submission file (or an internal --command, not both)")
    if (
        not np.isfinite(args.timeout)
        or not np.isfinite(args.startup_timeout)
        or args.timeout <= 0
        or args.startup_timeout <= 0
        or args.stride <= 0
        or args.seed < 0
        or (args.max_cases is not None and args.max_cases <= 0)
    ):
        parser.error(
            "timeouts, stride, and max-cases must be positive and finite; seed must be nonnegative"
        )
    try:
        corpora, profile = load_corpora(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    selected = []
    for corpus_index, corpus in enumerate(corpora):
        rows = (
            list(dict.fromkeys(args.row))
            if args.row is not None
            else range(0, corpus["rows"], args.stride)
        )
        if any(row < 0 or row >= corpus["rows"] for row in rows):
            parser.error("selected row is outside a corpus")
        selected.extend((corpus_index, row) for row in rows)
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(selected)
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    registered_cases = sum(corpus["rows"] for corpus in corpora)
    full_run = len(selected) == registered_cases
    selected_partial = (
        args.row is not None or args.stride != 1 or args.max_cases is not None
    )
    cases_path = args.cases or args.report.with_suffix(".cases.jsonl")
    if cases_path.resolve() == args.report.resolve():
        parser.error("report and case output paths must differ")
    protected = {Path(corpus["path"]) for corpus in corpora} | {args.manifest.resolve()}
    if args.submission is not None:
        protected.add(args.submission.resolve())
    if args.report.resolve() in protected or cases_path.resolve() in protected:
        parser.error("output must not overwrite a corpus, manifest, or submission")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_submission = None
    submission_metadata = None
    if args.submission is not None:
        from submission import prepare_submission

        try:
            prepared_submission = prepare_submission(args.submission)
        except (
            OSError,
            ValueError,
            RuntimeError,
            SyntaxError,
            subprocess.SubprocessError,
        ) as exc:
            parser.error(f"could not prepare submission: {exc}")
        command = prepared_submission.command
        submission_metadata = prepared_submission.metadata
    counts: Counter[str] = Counter()
    corpus_counts: list[Counter[str]] = [Counter() for _ in corpora]
    corpus_planned = Counter(index for index, _ in selected)
    timings = []
    worst: dict[str, float] = {}
    startup_seconds = 0.0
    process = None
    runner_digest = sha256(Path(__file__))
    helper_digests = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("submission.py", "rust_submission.py")
        if Path(__file__).with_name(name).is_file()
    }
    manifest_digest = None if profile == "custom" else sha256(args.manifest)
    total_started = time.perf_counter()
    try:
        with cases_path.open("w") as stream:
            for identifier, (corpus_index, row_index) in enumerate(selected):
                corpus = corpora[corpus_index]
                row = corpus["data"][row_index]
                record: dict[str, Any] = {
                    "id": identifier,
                    "corpus": corpus["path"],
                    "row": row_index,
                }
                started = time.perf_counter()
                request_started = None
                try:
                    if process is None:
                        startup_started = time.perf_counter()
                        try:
                            process = SolverProcess(command, args.startup_timeout)
                        finally:
                            startup_seconds += time.perf_counter() - startup_started
                    request_started = time.perf_counter()
                    response, elapsed = process.request(
                        {
                            "id": identifier,
                            "c": row[0].tolist(),
                            "g": row[1].tolist(),
                            "t": row[2].tolist(),
                        },
                        args.timeout,
                    )
                    record["seconds"] = elapsed
                    if response["status"] == "declined":
                        record["outcome"] = "declined"
                    elif response["status"] == "error":
                        record.update(
                            outcome="error",
                            reason=str(
                                response.get("message", "solver reported an error")
                            ),
                        )
                    else:
                        valid, metrics = check_witness(row, response.get("o"))
                        record.update(
                            outcome="passed" if valid else "failed", metrics=metrics
                        )
                        for name, value in metrics.items():
                            worst[name] = max(worst.get(name, 0.0), value)
                except (
                    TimeoutError,
                    ProtocolError,
                    OSError,
                    np.linalg.LinAlgError,
                ) as exc:
                    record["outcome"] = (
                        "timeout"
                        if isinstance(exc, TimeoutError)
                        else "invalid"
                        if isinstance(exc, ProtocolError)
                        else "error"
                    )
                    record["reason"] = str(exc)
                    record["stage"] = (
                        "startup" if request_started is None else "request_or_check"
                    )
                    record.setdefault(
                        "seconds",
                        time.perf_counter()
                        - (request_started if request_started is not None else started),
                    )
                    if process is not None:
                        process.close()
                        process = None
                counts[record["outcome"]] += 1
                corpus_counts[corpus_index][record["outcome"]] += 1
                timings.append(record["seconds"])
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if request_started is None:
                    # A broken import, executable, or handshake would fail
                    # identically on every row. Preserve one failure and stop.
                    break
    finally:
        if process is not None:
            process.close()
        if prepared_submission is not None:
            prepared_submission.close()
    attempted = len(timings)
    full_run = full_run and attempted == len(selected)
    all_passed = attempted == len(selected) and counts["passed"] == attempted
    full_coverage = full_run and all_passed
    report = {
        "protocol": PROTOCOL,
        "profile": profile,
        "manifest": None if profile == "custom" else str(args.manifest.resolve()),
        "manifest_sha256": manifest_digest,
        "corpora": [
            {
                **{key: value for key, value in corpus.items() if key != "data"},
                "planned_cases": corpus_planned[index],
                "attempted_cases": sum(corpus_counts[index].values()),
                "full_run": sum(corpus_counts[index].values()) == corpus["rows"],
                "full_coverage": corpus_counts[index]["passed"] == corpus["rows"],
                "verified_cases": corpus_counts[index]["passed"],
                "coverage_fraction": corpus_counts[index]["passed"] / corpus["rows"],
                "counts": {
                    name: corpus_counts[index][name]
                    for name in (
                        "passed",
                        "failed",
                        "declined",
                        "timeout",
                        "invalid",
                        "error",
                    )
                },
            }
            for index, corpus in enumerate(corpora)
        ],
        "command": command,
        "submission": submission_metadata,
        "label": args.label,
        "runner_sha256": runner_digest,
        "helper_sha256": helper_digests,
        "working_directory": str(Path.cwd()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timeout_seconds": args.timeout,
        "startup_timeout_seconds": args.startup_timeout,
        "tolerance": TOLERANCE,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "stride": args.stride,
        "planned_cases": len(selected),
        "attempted_cases": attempted,
        "unattempted_cases": len(selected) - attempted,
        "counts": {
            name: counts[name]
            for name in ("passed", "failed", "declined", "timeout", "invalid", "error")
        },
        "full_run": full_run,
        "full_coverage": full_coverage,
        "verified_cases": counts["passed"],
        "coverage_fraction": counts["passed"] / registered_cases,
        "coverage_scope": "selected corpus files",
        "ranking_eligible": profile == "full"
        and full_coverage
        and not selected_partial
        and all_passed,
        "all_passed": all_passed,
        "timing_seconds": {
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "p99": float(np.percentile(timings, 99)),
            "max": max(timings),
            "sum": sum(timings),
        },
        "timing_scope": "request send through complete response; failed startup durations included for startup failures",
        "startup_seconds": startup_seconds,
        "total_wall_seconds": time.perf_counter() - total_started,
        "worst_metrics": worst,
        "case_results": str(cases_path.resolve()),
        "claim": "Numerical SO(4) and spectral checks on original stored corpus values; no exact or general-completeness claim.",
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print_summary(report, args.report)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
