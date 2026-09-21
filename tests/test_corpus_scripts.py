"""Regression checks for corpus CLI safeguards and replay accounting."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_script("generate_realization_edge_corpus")


@pytest.fixture(scope="module")
def validator():
    return _load_script("validate_realization_pipeline_corpus")


@pytest.mark.parametrize("artifact", ["output", "strata", "metadata", "pipeline"])
def test_generator_refuses_existing_artifact(
    generator, monkeypatch, tmp_path, artifact
):
    output = tmp_path / "fresh.npy"
    paths = dict(
        zip(
            ("output", "strata", "metadata", "pipeline"),
            (output, *generator._sidecar_paths(output)),
            strict=True,
        )
    )
    paths[artifact].write_bytes(b"do not replace")
    monkeypatch.setattr(sys, "argv", ["generator", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        generator.main()
    assert error.value.code == 2
    assert paths[artifact].read_bytes() == b"do not replace"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "option,value",
    [
        ("--batch-size", "0"),
        ("--target-rounds", "0"),
        ("--samples-per-pair", "32769"),
        ("--seed", "-1"),
        ("--near-scale", "nan"),
        ("--near-scale", "1.01"),
    ],
)
def test_generator_rejects_invalid_arguments(
    generator, monkeypatch, tmp_path, option, value
):
    monkeypatch.setattr(
        sys, "argv", ["generator", "--output", str(tmp_path / "new.npy"), option, value]
    )
    with pytest.raises(SystemExit) as error:
        generator.main()
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def _pipeline_fixture(path, count=1):
    np.savez(
        path,
        sample_indices=np.full(count, 7),
        c_weyl=np.zeros((count, 3)),
        g_weyl=np.zeros((count, 3)),
        target_unitaries=np.tile(np.eye(4), (count, 1, 1)),
        pair_codes=np.zeros((count, 2), dtype=int),
        local_modes=np.zeros(count, dtype=int),
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--max-cases", "0"],
        ["--max-cases", "-1"],
        ["--tolerance", "nan"],
        ["--tolerance", "0"],
        ["--max-case-seconds", "inf"],
        ["--max-case-seconds", "-1"],
        ["--sample-index", "123"],
        ["--sample-index", "7", "--sample-index", "123"],
        ["--sample-index", "7", "--corpus-row", "0"],
    ],
)
def test_validator_rejects_false_pass_arguments(
    validator, monkeypatch, tmp_path, options
):
    path = tmp_path / "pipeline.npz"
    _pipeline_fixture(path)
    monkeypatch.setattr(sys, "argv", ["validator", str(path), *options])
    with pytest.raises(SystemExit) as error:
        validator.main()
    assert error.value.code == 2


def test_validator_rejects_empty_corpus(validator, monkeypatch, tmp_path):
    path = tmp_path / "empty.npz"
    _pipeline_fixture(path, count=0)
    monkeypatch.setattr(sys, "argv", ["validator", str(path)])
    with pytest.raises(SystemExit) as error:
        validator.main()
    assert error.value.code == 2


def test_validator_reports_gate_error_and_latency_once(
    validator, monkeypatch, tmp_path
):
    path = tmp_path / "pipeline.npz"
    _pipeline_fixture(path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            str(path),
            "--max-case-seconds",
            "0.5",
            "--corpus-row",
            "0",
            "--corpus-row",
            "0",
        ],
    )

    def bad_gate(_point):
        raise ValueError("bad gate fixture")

    monkeypatch.setattr(validator, "_compile_target", lambda *args: bad_gate(None))
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(validator.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(AssertionError) as error:
        validator.main()
    message = str(error.value)
    assert "1 failed rows in 1 attempted rows" in message
    assert "bad gate fixture" in message
    assert "'ValueError': 1" in message
    assert "'latency': 1" in message


def test_locked_fixture_still_matches_published_hashes(generator):
    path = (
        Path(generator.__file__).resolve().parents[1] / "corpus/feasible_stratified.npy"
    )
    assert generator._existing_is_current(
        path,
        generator.DEFAULT_SAMPLES_PER_PAIR,
        generator.DEFAULT_TARGET_ROUNDS,
        0xC0FFEE,
        None,
    )
