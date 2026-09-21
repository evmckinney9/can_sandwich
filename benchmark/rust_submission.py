"""Build one Rust solver file behind the public benchmark protocol."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


_SERVER = r"""mod candidate;

use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn coordinates(request: &Value, key: &str) -> Result<[f64; 3], String> {
    let values = request[key]
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| format!("{key} must contain three finite numbers"))?;
    let mut result = [0.0; 3];
    for (slot, value) in result.iter_mut().zip(values) {
        *slot = value.as_f64().filter(|value| value.is_finite())
            .ok_or_else(|| format!("{key} must contain three finite numbers"))?;
    }
    Ok(result)
}

fn respond(request: &Value) -> Result<Value, String> {
    let id = request["id"].as_u64().ok_or("id must be a nonnegative integer")?;
    let c = coordinates(request, "c")?;
    let g = coordinates(request, "g")?;
    let t = coordinates(request, "t")?;
    match candidate::solve(c, g, t) {
        None => Ok(json!({"id": id, "status": "declined"})),
        Some(o) => {
            if o.iter().flatten().any(|value| !value.is_finite()) {
                return Err("solver returned a nonfinite frame".into());
            }
            Ok(json!({"id": id, "status": "solved", "o": o}))
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    writeln!(stdout, "{}", json!({"protocol": "can-sandwich-v1", "ready": true}))?;
    stdout.flush()?;
    for line in stdin.lock().lines() {
        let request: Value = serde_json::from_str(&line?)?;
        let response = match respond(&request) {
            Ok(response) => response,
            Err(message) => json!({"id": request["id"], "status": "error", "message": message}),
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}
"""


def prepare_rust(
    path: Path,
) -> tuple[list[str], dict[str, object], tempfile.TemporaryDirectory[str]]:
    """Compile a source snapshot and return its command, metadata, and owner.

    The caller must keep the returned temporary directory alive until the
    command exits. Dependencies and build artifacts share an ignored cache;
    the executable used for this run is copied into the private directory.
    Compilation is excluded from benchmark latency. Build failures raise
    RuntimeError and never return an executable from a previous build.
    """
    source_path = path.resolve(strict=True)
    source = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    package = "can_sandwich_candidate_" + source_sha256[:20]
    temporary = tempfile.TemporaryDirectory(prefix="can-sandwich-submission-")
    directory = Path(temporary.name)
    cache = (
        Path(__file__).resolve().parents[3] / ".local/can-sandwich-submissions/target"
    )
    started = time.perf_counter()
    try:
        (directory / "src").mkdir()
        (directory / "src/candidate.rs").write_bytes(source)
        (directory / "src/main.rs").write_text(_SERVER)
        (directory / "Cargo.toml").write_text(
            f'[package]\nname = "{package}"\nversion = "0.0.0"\nedition = "2021"\n'
            '[workspace]\n\n[dependencies]\nnalgebra = "0.35"\n'
            'serde_json = { version = "1", features = ["float_roundtrip"] }\n'
            '\n[profile.release]\nlto = "fat"\ncodegen-units = 1\n'
        )
        environment = os.environ.copy()
        environment["CARGO_TARGET_DIR"] = str(cache)
        completed = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(directory / "Cargo.toml"),
            ],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Rust submission failed to compile: {source_path}\n{completed.stderr}"
            )
        suffix = ".exe" if os.name == "nt" else ""
        executable = directory / ("solver" + suffix)
        shutil.copy2(cache / "release" / (package + suffix), executable)
        metadata: dict[str, object] = {
            "language": "rust",
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "build_seconds": time.perf_counter() - started,
            "build_profile": "release",
        }
        return [str(executable)], metadata, temporary
    except BaseException:
        temporary.cleanup()
        raise
