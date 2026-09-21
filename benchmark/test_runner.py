"""Protocol, independent witness checks, and report qualification regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import build_corpus as builder
import run as benchmark

SOLVER = r"""
import json, sys, time
mode = sys.argv[1]
if mode == 'badready':
    print('{}', flush=True)
else:
    print(json.dumps({'protocol':'can-sandwich-v1','ready':True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if mode == 'timeout':
        time.sleep(5)
    if mode == 'oversized':
        print('x'*65537, flush=True)
        continue
    if mode == 'nested':
        print('['*2000+']'*2000, flush=True)
        continue
    if mode == 'partial':
        print('{', end='', flush=True)
        time.sleep(5)
    result = {'id':request['id'], 'status':'solved', 'o':[[int(i==j) for j in range(4)] for i in range(4)]}
    if mode == 'echo':
        result['request'] = request
    if mode == 'declined':
        result = {'id':request['id'], 'status':'declined'}
    if mode == 'badid':
        result['id'] += 1
    if mode == 'boolid':
        result['id'] = False
    if mode == 'nonorthogonal':
        result['o'][0][0] = 2
    print(json.dumps(result), flush=True)
"""


class CheckerTests(unittest.TestCase):
    """Reject plausible but incorrect matrices independently of protocol behavior."""

    def test_identity_and_permuted_target(self) -> None:
        """Check a nontrivial diagonal product in the original float coordinates."""
        c = np.array([0.173, 0.071, -0.019])
        g = np.array([0.109, 0.038, -0.011])
        row = np.array([c, g, c + g])
        passed, metrics = benchmark.check_witness(row, np.eye(4).tolist())
        self.assertTrue(passed, metrics)
        self.assertLess(metrics["spectral_bottleneck"], 1e-14)

    def test_signed_permutation_witness(self) -> None:
        """Match the spectrum after a nontrivial orthogonal conjugation."""
        c = np.array([0.173, 0.071, -0.019])
        g = np.array([0.109, 0.038, -0.011])
        turns = lambda m: np.array([m[1], m[0], -sum(m), m[2]])
        permutation = [1, 0, 2, 3]
        target = turns(c) + turns(g)[permutation]
        row = np.array([c, g, target[[1, 0, 3]]])
        q = np.eye(4)[permutation]
        q[:, 0] *= -1
        passed, metrics = benchmark.check_witness(row, q.tolist())
        self.assertTrue(passed, metrics)

    def test_negative_central_target_lift(self) -> None:
        """Accept the central negative lift while retaining determinant one."""
        row = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]])
        passed, metrics = benchmark.check_witness(row, np.eye(4).tolist())
        self.assertTrue(passed, metrics)

    def test_nonorthogonal_and_orientation_reversal(self) -> None:
        """Require both the Gram condition and positive orientation."""
        row = np.zeros((3, 3))
        for matrix in (2 * np.eye(4), np.diag([-1, 1, 1, 1])):
            passed, _ = benchmark.check_witness(row, matrix.tolist())
            self.assertFalse(passed)

    def test_wrong_spectrum(self) -> None:
        """Reject an orthogonal witness with the wrong product spectrum."""
        row = np.zeros((3, 3))
        row[2] = [0.25, 0, 0]
        passed, metrics = benchmark.check_witness(row, np.eye(4).tolist())
        self.assertFalse(passed)
        self.assertGreater(metrics["spectral_bottleneck"], 1)

    def test_malformed_and_nonfinite(self) -> None:
        """Reject complex/string, malformed, boolean, and nonfinite responses."""
        for matrix in (
            [[]],
            [["1"] * 4] * 4,
            [[False] * 4] * 4,
            [[float("nan")] * 4] * 4,
        ):
            with (
                self.subTest(matrix=matrix),
                self.assertRaises(benchmark.ProtocolError),
            ):
                benchmark.check_witness(np.zeros((3, 3)), matrix)


class ProcessTests(unittest.TestCase):
    """Exercise real subprocess transport, timeouts, and restart accounting."""

    def setUp(self) -> None:
        """Create an isolated solver and original binary64 corpus."""
        self.directory = tempfile.TemporaryDirectory(prefix="can-sandwich-protocol-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.solver = self.root / "solver.py"
        self.solver.write_text(SOLVER)
        self.corpus = self.root / "cases.npy"
        np.save(self.corpus, np.zeros((2, 3, 3)))
        self.request = {"id": 0, "c": [0, 0, 0], "g": [0, 0, 0], "t": [0, 0, 0]}

    def start(self, mode: str) -> benchmark.SolverProcess:
        """Start the selected test solver and guarantee cleanup."""
        process = benchmark.SolverProcess([sys.executable, str(self.solver), mode], 2)
        self.addCleanup(process.close)
        return process

    def test_success_and_decline(self) -> None:
        """Keep a persistent process alive for multiple matching responses."""
        process = self.start("good")
        for identifier in range(2):
            response, seconds = process.request(dict(self.request, id=identifier), 1)
            self.assertEqual(response["id"], identifier)
            self.assertLess(seconds, 1)
        response, _ = self.start("declined").request(self.request, 1)
        self.assertEqual(response["status"], "declined")

    def test_json_transport_preserves_binary64_bits(self) -> None:
        """Transport adjacent and subnormal binary64 values without decimal rounding."""
        values = [
            float(np.nextafter(0.5, 1)),
            float(np.nextafter(0.0, 1)),
            float(np.nextafter(-0.5, -1)),
        ]
        request = {"id": 0, "c": values, "g": values[::-1], "t": [-v for v in values]}
        response, _ = self.start("echo").request(request, 1)
        for key in ("c", "g", "t"):
            self.assertEqual(
                [v.hex() for v in response["request"][key]],
                [v.hex() for v in request[key]],
            )

    def test_invalid_corpus_arrays(self) -> None:
        """Reject nonbinary64, empty, malformed, and nonfinite fixture arrays."""
        arrays = [
            np.zeros((2, 3, 3), dtype=np.float32),
            np.zeros((0, 3, 3)),
            np.zeros((2, 4, 4)),
            np.full((1, 3, 3), np.nan),
        ]
        for index, array in enumerate(arrays):
            with self.subTest(index=index):
                path = self.root / f"invalid{index}.npy"
                np.save(path, array)
                args = argparse.Namespace(corpus=[path])
                with self.assertRaises(ValueError):
                    benchmark.load_corpora(args)

    def test_protocol_failures(self) -> None:
        """Reject bad identifiers and oversized stdout lines."""
        for mode in ("badid", "boolid", "oversized", "nested"):
            with self.subTest(mode=mode), self.assertRaises(benchmark.ProtocolError):
                self.start(mode).request(self.request, 1)
        with self.assertRaises(benchmark.ProtocolError):
            self.start("badready")

    def test_timeout_and_incomplete_line(self) -> None:
        """Bound both silent solvers and partially written responses."""
        for mode in ("timeout", "partial"):
            with self.subTest(mode=mode), self.assertRaises(TimeoutError):
                self.start(mode).request(self.request, 0.05)

    def run_cli(
        self, mode: str, *options: str
    ) -> tuple[subprocess.CompletedProcess, dict, list]:
        """Run the CLI and read the complete report and streaming case log."""
        report = self.root / "report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(benchmark.__file__)),
                "--corpus",
                str(self.corpus),
                "--timeout",
                "0.05",
                "--startup-timeout",
                "2",
                "--report",
                str(report),
                *options,
                "--command",
                sys.executable,
                str(self.solver),
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertTrue(report.is_file(), completed.stderr)
        summary = json.loads(report.read_text())
        cases = [
            json.loads(line)
            for line in Path(summary["case_results"]).read_text().splitlines()
        ]
        return completed, summary, cases

    def test_timeout_restarts_and_records_each_case(self) -> None:
        """A timed-out child must not prevent the next case from running."""
        completed, report, cases = self.run_cli("timeout")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(report["counts"]["timeout"], 2)
        self.assertEqual(len(cases), 2)
        self.assertTrue(report["full_run"])
        self.assertFalse(report["full_coverage"])
        self.assertEqual(report["coverage_fraction"], 0)
        self.assertFalse(report["ranking_eligible"])

    def test_single_python_file_submission(self) -> None:
        """Accept NumPy frames, redirect prints, and preserve explicit solve errors."""
        submission = self.root / "candidate.py"
        for body, outcome in [
            (
                "import numpy as np\nprint('import diagnostic')\ndef solve(c, g, t):\n    print('solve diagnostic')\n    return np.eye(4)\n",
                "passed",
            ),
            ("def solve(c, g, t):\n    return None\n", "declined"),
            (
                "def solve(c, g, t):\n    raise RuntimeError('deliberate failure')\n",
                "error",
            ),
        ]:
            with self.subTest(outcome=outcome):
                submission.write_text(body)
                report = self.root / "file-report.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(benchmark.__file__)),
                        str(submission),
                        "--corpus",
                        str(self.corpus),
                        "--report",
                        str(report),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                summary = json.loads(report.read_text())
                self.assertEqual(summary["counts"][outcome], 2, completed.stderr)
                self.assertEqual(
                    summary["submission"]["source_sha256"], benchmark.sha256(submission)
                )
                if outcome == "passed":
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("solve diagnostic", completed.stderr)
                else:
                    self.assertEqual(completed.returncode, 1)
                if outcome == "error":
                    records = [
                        json.loads(line)
                        for line in Path(summary["case_results"])
                        .read_text()
                        .splitlines()
                    ]
                    self.assertIn(
                        "RuntimeError: deliberate failure", records[0]["reason"]
                    )

    def test_startup_failure_stops_after_one_case(self) -> None:
        """Do not restart a broken submission for every corpus row."""
        completed, report, cases = self.run_cli("badready")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(report["attempted_cases"], 1)
        self.assertEqual(report["unattempted_cases"], 1)
        self.assertFalse(report["full_coverage"])
        self.assertFalse(report["ranking_eligible"])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["stage"], "startup")

    def test_failed_witness_report(self) -> None:
        """Report mathematically invalid witnesses as failures, not declines."""
        completed, report, cases = self.run_cli("nonorthogonal")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(report["counts"]["failed"], 2)
        self.assertTrue(report["full_run"])
        self.assertFalse(report["full_coverage"])
        self.assertEqual(report["verified_cases"], 0)
        self.assertEqual(report["coverage_fraction"], 0)
        self.assertTrue(report["corpora"][0]["full_run"])
        self.assertFalse(report["corpora"][0]["full_coverage"])
        self.assertEqual(report["corpora"][0]["coverage_fraction"], 0)
        self.assertTrue(all("metrics" in case for case in cases))

    def test_registered_corpus_eligibility_and_hash_validation(self) -> None:
        """Only full passing corpus runs with matching artifact hashes can rank."""
        manifest = self.root / "manifest.json"
        entry = {
            "path": self.corpus.name,
            "rows": 2,
            "sha256": benchmark.sha256(self.corpus),
        }
        contents = {"schema_version": 2, "corpus": entry}
        manifest.write_text(json.dumps(contents))
        report = self.root / "registered.json"
        base = [
            sys.executable,
            str(Path(benchmark.__file__)),
            "--manifest",
            str(manifest),
            "--report",
            str(report),
        ]
        tail = ["--command", sys.executable, str(self.solver), "good"]
        completed = subprocess.run(
            base + tail, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(report.read_text())["ranking_eligible"])
        completed = subprocess.run(
            base + ["--max-cases", "2"] + tail,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(json.loads(report.read_text())["ranking_eligible"])
        entry["sha256"] = "0" * 64
        manifest.write_text(json.dumps(contents))
        completed = subprocess.run(
            base + tail, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("digest mismatch", completed.stderr)

    def test_custom_and_partial_reports_never_rank(self) -> None:
        """Distinguish complete custom runs from partial case selections."""
        completed, report, _ = self.run_cli("good")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(report["all_passed"])
        self.assertTrue(report["full_run"])
        self.assertEqual(report["coverage_fraction"], 1)
        self.assertTrue(report["corpora"][0]["full_coverage"])
        self.assertEqual(report["corpora"][0]["coverage_fraction"], 1)
        for helper in ("submission.py", "rust_submission.py"):
            self.assertEqual(
                report["helper_sha256"][helper],
                benchmark.sha256(Path(benchmark.__file__).with_name(helper)),
            )
        self.assertTrue(report["full_coverage"])
        self.assertFalse(report["ranking_eligible"])
        _, partial, cases = self.run_cli("good", "--max-cases", "1")
        self.assertFalse(partial["full_run"])
        self.assertFalse(partial["full_coverage"])
        self.assertEqual(partial["coverage_fraction"], 0.5)
        self.assertFalse(partial["ranking_eligible"])
        self.assertEqual(len(cases), 1)


class ConsolidationTests(unittest.TestCase):
    """Preserve original float bits and guard corpus replacement."""

    def setUp(self) -> None:
        """Create two registered sources and one optional audited source."""
        self.directory = tempfile.TemporaryDirectory(prefix="can-sandwich-combine-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        first = np.zeros((2, 3, 3))
        first[0, 0] = [float(np.nextafter(0.0, 1)), -0.0, float(np.nextafter(0.5, 1))]
        self.arrays = [first, np.full((1, 3, 3), 0.125), np.full((1, 3, 3), 0.25)]
        self.paths = []
        for index, array in enumerate(self.arrays):
            path = self.root / f"source{index}.npy"
            np.save(path, array)
            self.paths.append(path)
        entries = [
            {"path": path.name, "rows": len(array), "sha256": builder.sha256(path)}
            for path, array in zip(self.paths[:2], self.arrays[:2], strict=True)
        ]
        self.manifest = self.root / "sources.json"
        self.manifest.write_text(
            json.dumps(
                {"schema_version": 1, "profiles": {"extended": {"corpora": entries}}}
            )
        )
        self.output = self.root / "feasible_all.npy"

    def test_bitwise_combination_ranges_and_runner_manifest(self) -> None:
        """Retain signed zeros and adjacent floats in source order."""
        manifest = self.root / "combined.json"
        result = builder.build_corpus(
            self.manifest, self.output, self.paths[2:], manifest_output=manifest
        )
        self.assertEqual(result["rows"], 4)
        actual = np.load(self.output)
        expected = np.concatenate(self.arrays)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        provenance = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(
            [(source["start"], source["stop"]) for source in provenance["sources"]],
            [(0, 2), (2, 3), (3, 4)],
        )
        self.assertTrue(provenance["bitwise_slices_verified"])
        args = argparse.Namespace(corpus=None, manifest=manifest)
        corpora, profile = benchmark.load_corpora(args)
        self.assertEqual(profile, "full")
        self.assertEqual(len(corpora), 1)
        self.assertEqual(corpora[0]["rows"], 4)
        previous = self.output.stat().st_mtime_ns
        builder.build_corpus(
            self.manifest, self.output, self.paths[2:], manifest_output=manifest
        )
        self.assertEqual(self.output.stat().st_mtime_ns, previous)

    def test_different_output_requires_explicit_force(self) -> None:
        """Refuse changes until the caller explicitly requests replacement."""
        builder.build_corpus(self.manifest, self.output)
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            builder.build_corpus(self.manifest, self.output, self.paths[2:])
        self.assertEqual(self.output.read_bytes(), original)
        result = builder.build_corpus(
            self.manifest, self.output, self.paths[2:], force=True
        )
        self.assertEqual(result["rows"], 4)

    def test_source_hash_and_output_collision(self) -> None:
        """Reject changed registered inputs and writes over source artifacts."""
        with self.assertRaises(ValueError):
            builder.build_corpus(self.manifest, self.paths[0], force=True)
        np.save(self.paths[0], np.ones_like(self.arrays[0]))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            builder.build_corpus(self.manifest, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
