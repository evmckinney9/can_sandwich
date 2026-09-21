#!/usr/bin/env python3
"""Load a one-file solver and hide the benchmark transport from its author."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any


@dataclass
class PreparedSubmission:
    """A runnable file submission and its reproducible build metadata."""

    command: list[str]
    metadata: dict[str, Any]
    temporary: TemporaryDirectory | None = None

    def close(self) -> None:
        """Remove private build files after the candidate process has exited."""
        if self.temporary is not None:
            self.temporary.cleanup()


def prepare_submission(path: Path) -> PreparedSubmission:
    """Prepare Python or Rust source without changing the submitted file."""
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("The submission must be one Python or Rust source file.")
    if path.suffix == ".rs":
        from rust_submission import prepare_rust

        command, metadata, temporary = prepare_rust(path)
        return PreparedSubmission(command, metadata, temporary)
    if path.suffix != ".py":
        raise ValueError("The submission must have a .py or .rs extension.")
    source = path.read_bytes()
    compile(source, str(path), "exec")
    return PreparedSubmission(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(path)],
        {
            "language": "python",
            "source_path": str(path),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "build_seconds": 0.0,
        },
    )


def serve_python(path: Path) -> None:
    """Import a candidate and translate its solve function into internal JSONL."""
    spec = importlib.util.spec_from_file_location("can_sandwich_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import the submission.")
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    sys.path.insert(0, str(path.parent))
    with redirect_stdout(sys.stderr):
        spec.loader.exec_module(candidate)
    solve = getattr(candidate, "solve", None)
    if not callable(solve):
        raise ValueError("The submission must define solve(c, g, t).")
    print(json.dumps({"protocol": "can-sandwich-v1", "ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with redirect_stdout(sys.stderr):
                frame = solve(request["c"], request["g"], request["t"])
            if frame is None:
                response = {"id": request["id"], "status": "declined"}
            else:
                # NumPy arrays are convenient, but NumPy is not required by
                # this transport wrapper or by a pure-Python submission.
                if hasattr(frame, "tolist"):
                    frame = frame.tolist()
                response = {"id": request["id"], "status": "solved", "o": frame}
            encoded = json.dumps(response, allow_nan=False)
        except Exception as exc:
            encoded = json.dumps(
                {
                    "id": request["id"],
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        print(encoded, flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Run benchmark/run.py with your submission file.")
    serve_python(Path(sys.argv[2]).resolve())
