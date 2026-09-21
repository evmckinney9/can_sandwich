"""Compile and exercise the single-file Rust submission adapter."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from rust_submission import prepare_rust


class RustSubmissionTests(unittest.TestCase):
    """Test real builds and protocol responses without a production dependency."""

    def test_example_build_and_protocol(self) -> None:
        """Compile the example and check solved, declined, and invalid requests."""
        source = Path(__file__).with_name("example_solver.rs")
        command, metadata, temporary = prepare_rust(source)
        with temporary:
            self.assertEqual(
                metadata["source_sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(metadata["language"], "rust")
            self.assertEqual(Path(command[0]).parent, Path(temporary.name))
            manifest = (Path(temporary.name) / "Cargo.toml").read_text()
            self.assertNotIn("can_sandwich =", manifest)
            requests = [
                {"id": 0, "c": [0, 0, 0], "g": [0, 0, 0], "t": [0, 0, 0]},
                {"id": 1, "c": [0.125, 0, 0], "g": [0, 0, 0], "t": [0, 0, 0]},
                {"id": 2, "c": [0, 0], "g": [0, 0, 0], "t": [0, 0, 0]},
            ]
            completed = subprocess.run(
                command,
                input="".join(json.dumps(row) + "\n" for row in requests),
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            )
            ready, solved, declined, malformed = map(
                json.loads, completed.stdout.splitlines()
            )
            self.assertEqual(ready, {"protocol": "can-sandwich-v1", "ready": True})
            self.assertEqual(solved["id"], 0)
            self.assertEqual(solved["status"], "solved")
            self.assertEqual(
                solved["o"], [[float(i == j) for j in range(4)] for i in range(4)]
            )
            self.assertEqual(declined, {"id": 1, "status": "declined"})
            self.assertEqual(malformed["id"], 2)
            self.assertEqual(malformed["status"], "error")
        self.assertFalse(Path(command[0]).exists())

    def test_invalid_source_reports_compiler_error(self) -> None:
        """Reject compilation failure rather than returning a cached executable."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.rs"
            source.write_text("pub fn solve( this is not Rust\n")
            with self.assertRaisesRegex(RuntimeError, "failed to compile") as raised:
                prepare_rust(source)
            self.assertIn("error", str(raised.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
