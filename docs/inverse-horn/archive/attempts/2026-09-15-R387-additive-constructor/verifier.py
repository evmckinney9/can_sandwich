"""Independent R387 matrix checks; does not import constructor or its helpers."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
inputs = json.loads((ROOT / "inputs.json").read_text())
outputs = json.loads((ROOT / "results.json").read_text())
plants = json.loads((ROOT / "plants.json").read_text())
cases = {x["name"]: x for x in inputs}
assert len(cases) == len(inputs) == 12
expected = {(name, blocks) for name in cases for blocks in (False, True)}
assert {(x["name"], x["blocks"]) for x in outputs} == expected
assert len(outputs) == 24

def measurements(case, frame):
    a, b, c = [np.array(case[k], dtype=np.float64)
               for k in ("alpha", "beta", "gamma")]
    q = np.array(frame, dtype=np.float64)
    assert q.shape == (4, 4)
    assert all(np.isfinite(x).all() for x in (a, b, c, q))
    matrix_a = np.diag(a)
    matrix_b = q @ np.diag(b) @ q.T
    actual = np.linalg.eigvalsh(matrix_a + matrix_b)
    spectrum_error = float(np.max(np.abs(actual - np.sort(c))))
    second_error = float(np.max(np.abs(np.linalg.eigvalsh(matrix_b) - np.sort(b))))
    orth = float(np.linalg.norm(q.T @ q - np.eye(4), ord=np.inf))
    determinant = float(np.linalg.det(q))
    return dict(spectrum_error=spectrum_error, second_spectrum_error=second_error,
                orthogonality_inf=orth, determinant=determinant,
                determinant_error=abs(determinant - 1.0))

feasibility = []
for plant in plants:
    data = measurements(cases[plant["name"]], plant["Q"])
    assert data["spectrum_error"] <= 1e-12
    assert data["orthogonality_inf"] <= 1e-12
    assert abs(abs(data["determinant"]) - 1) <= 1e-12
    feasibility.append(dict(name=plant["name"], **data))
assert {x["name"] for x in plants} == set(cases)

checked = []
for row in outputs:
    record = dict(name=row["name"], blocks=row["blocks"], method=row["method"],
                  reported_steps=row.get("steps"), reported_seconds=row["seconds"])
    if row["Q"] is None:
        assert row.get("status") == "UNKNOWN"
        record.update(status="UNKNOWN", reported_unverified_last_error=row["error"])
    else:
        data = measurements(cases[row["name"]], row["Q"])
        assert data["spectrum_error"] <= 1e-8
        assert data["second_spectrum_error"] <= 1e-12
        assert data["orthogonality_inf"] <= 1e-12
        assert data["determinant_error"] <= 1e-12
        assert abs(data["spectrum_error"] - row["error"]) <= 1e-12
        record.update(status="ACCEPTED", **data)
    checked.append(record)

# One requested independent failure replay. No constructor code is called.
wall = cases["wall22"]
a, b, c = [np.array(wall[k]) for k in ("alpha", "beta", "gamma")]
denominator = 12 * max(1.0, max(abs(float(x)) for x in np.concatenate((a,b,c))))
spectra = (a / denominator + 1/3, b / denominator + 1/3,
           np.sort(-c)[::-1] / denominator + 1/3)
generator = np.random.default_rng(387)
bases = [np.linalg.qr(generator.standard_normal((4,4)))[0] for _ in range(3)]
minimum_seen = float("inf")
for sweep in range(6001):
    relative = bases[0].T @ bases[1]
    data = measurements(wall, relative)
    minimum_seen = min(minimum_seen, data["spectrum_error"])
    if sweep == 6000:
        break
    marginal = np.zeros((4,4))
    for basis, spectrum in zip(bases, spectra):
        marginal += basis @ np.diag(spectrum) @ basis.T
    lower = np.linalg.cholesky(marginal)
    bases = [np.linalg.qr(np.linalg.solve(lower, basis))[0] for basis in bases]
reported_wall = next(x for x in outputs if x["name"] == "wall22" and not x["blocks"])
assert minimum_seen > 5e-9
assert abs(data["spectrum_error"] - reported_wall["error"]) <= 1e-10
wall_replay = dict(sweeps=6000, seed=387, minimum_seen_error=minimum_seen,
                   final_error=data["spectrum_error"],
                   reported_error=reported_wall["error"],
                   comparison_tolerance=1e-10, outcome="UNKNOWN_REPRODUCED")

summary = {}
for blocks in (False, True):
    rows = [r for r in checked if r["blocks"] == blocks]
    accepted = [r for r in rows if r["status"] == "ACCEPTED"]
    summary["wrapped" if blocks else "plain"] = dict(
        accepted=len(accepted), attempted=len(rows),
        accepted_names=[r["name"] for r in accepted],
        unknown_names=[r["name"] for r in rows if r["status"] == "UNKNOWN"],
        max_accepted_spectrum_error=max(r["spectrum_error"] for r in accepted),
        max_orthogonality_inf=max(r["orthogonality_inf"] for r in accepted),
        max_determinant_error=max(r["determinant_error"] for r in accepted))
names = ["attempt.md", "gate-0.md", "constructor.py", "generate.py", "inputs.json",
         "plants.json", "run.py", "results.json", "verifier.py"]
hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
report = dict(status="FINITE_OUTPUT_CHECKS_PASS", source_helpers_imported=False,
              hashes=hashes, summary=summary, wall22_plain_replay=wall_replay,
              outputs=checked, input_feasibility=feasibility)
(ROOT/"independent.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
print(json.dumps(dict(summary=summary, wall22_plain_replay=wall_replay), indent=2))
