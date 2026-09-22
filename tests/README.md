# Solver tests

`cargo test --release` runs the Rust tests, including all 1,077,823 fixture cases.
The checker verifies SO(4) and the actual product eigenvalues at `1e-8`, using
faer independently of the solver's certificate. Declines and invalid frames fail.
Production still has known failures.

To compare a prototype, replace the local `candidate` function in `solver.rs`
(or call your own module from it), then run:

```sh
cargo test --release --test solver compare_candidate -- --ignored --nocapture
```

Both algorithms use the same cases and checker. The output shows passes,
solver time excluding verification, and fixed/regressed row indices.
The candidate initially calls production so the comparison can be checked.

`cases.bin` is the only data file: 1,077,823 records of nine little-endian IEEE-754
binary64 values, ordered `[C0,C1,C2,G0,G1,G2,T0,T1,T2]`. No header or padding;
72 bytes per record, 77,603,256 bytes total. Rust reads it with `f64::from_le_bytes`.
SHA-256: `7624f29bbe40cab22bef69ffc8c0e3c31b73b867b76a85ded94f8ca10c67eac1`.

Rows preserve the original bits, order, and duplicates. Half-open ranges:

| Source | Rows |
| --- | --- |
| Basic | 0–6 |
| Stratified | 6–13,187 |
| Linspace | 13,187–774,495 |
| Haar | 774,495–1,074,495 |
| Targeted | 1,074,495–1,075,950 |
| GULPS calls (`38014b5`, solver `af4e5ea`) | 1,075,950–1,077,823 |

`generate.py` contains the original grid/Haar, stratified, and targeted
construction recipes. Basic examples and captured GULPS calls are fixed
regression literals in that script. Generation uses NumPy, SciPy, and GULPS;
Rust tests need none of them. No old arrays or Git checkout are read as inputs.

```sh
python3 tests/generate.py /tmp/cases.bin
```

Omit the output path to replace `tests/cases.bin`. Seeds and sample counts are
fixed. Numerical results depend on NumPy/SciPy/GULPS versions: the current GULPS
coordinate routines change some stratified and Haar values compared with the
historical fixture. The checked-in fixture retains its original float bits.
