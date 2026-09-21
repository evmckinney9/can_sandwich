"""Solver regressions extracted at the GULPS adapter, without a host dependency."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/gulps_segments.json").read_text())["cases"]
spec = importlib.util.spec_from_file_location(
    "witness_checker", ROOT / "benchmark/run.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class Declined(Exception):
    """The production solver did not construct a witness."""


@pytest.fixture(scope="module")
def solver():
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--features",
            "benchmark",
            "--bin",
            "can_sandwich_server",
        ],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT
        )
    )
    process = checker.SolverProcess(
        [str(Path(metadata["target_directory"]) / "release/can_sandwich_server")]
    )
    try:
        yield process
    finally:
        process.close()


@pytest.mark.parametrize(
    "index,case",
    [
        pytest.param(
            index,
            case,
            id=case["case"],
            marks=()
            if case["solved"]
            else pytest.mark.xfail(
                strict=True, raises=Declined, reason="open near-wall realization gap"
            ),
        )
        for index, case in enumerate(CASES)
    ],
)
def test_captured_segment_has_a_valid_frame(solver, index, case):
    response, _ = solver.request(
        {"id": index, **{k: case[k] for k in ("c", "g", "t")}}, 30
    )
    if response["status"] == "declined":
        raise Declined(case["case"])
    assert response["status"] == "solved", response
    valid, metrics = checker.check_witness(
        np.asarray([case[k] for k in ("c", "g", "t")]), response["o"]
    )
    assert valid, metrics
