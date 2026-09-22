//! Compare an external candidate with production. Enable the `corpus` feature.
//!
//! Call [`compare`] from the candidate executable's `main`. See
//! `docs/researcher.md` for the mathematical contract and command-line options.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use nalgebra::{Complex, Matrix4};
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use crate::solve;
type Case = [[f64; 3]; 3];
type Frame = Matrix4<f64>;
type Solver = fn([f64; 3], [f64; 3], [f64; 3]) -> Option<Frame>;
const TOLERANCE: f64 = 1e-8;
const HELP: &str = "Compare a candidate with can_sandwich::solve.
Usage: candidate --corpus PATH [--case INDEX] [--report PATH]
  --corpus PATH  Nine little-endian f64 values per row (c, g, t).
  --case INDEX   Run one zero-based corpus row and print its input.
  --report PATH  Create a CSV with both results for every tested row.
Exit: 0 = candidate passed all tested rows; 1 = candidate failures; 2 = runner error.";

fn cases(path: &Path) -> io::Result<Vec<Case>> {
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(72) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "empty or truncated corpus",
        ));
    }
    let mut rows = Vec::with_capacity(bytes.len() / 72);
    for row in bytes.chunks_exact(72) {
        let case: Case = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let offset = (3 * i + j) * 8;
                f64::from_le_bytes(row[offset..offset + 8].try_into().unwrap())
            })
        });
        if case.iter().flatten().any(|v| !v.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("nonfinite input at row {}", rows.len()),
            ));
        }
        rows.push(case);
    }
    Ok(rows)
}

fn spectrum(m: [f64; 3]) -> [Complex<f64>; 4] {
    let [x, y, z] = [m[0] + m[1], m[0] + m[2], m[1] + m[2]];
    [x - y + z, x + y - z, -x - y - z, -x + y + z]
        .map(|v| Complex::from_polar(1.0, std::f64::consts::PI * v))
}

// Entries: spectral matching error, max-entry orthogonality defect, determinant error.
// Infinity denotes an unavailable or nonfinite measurement.
fn errors([c, g, t]: Case, o: &Frame) -> [f64; 3] {
    if o.iter().any(|v| !v.is_finite()) {
        return [f64::INFINITY; 3];
    }
    let finite = |x: f64| if x.is_finite() { x } else { f64::INFINITY };
    let mut result = [
        f64::INFINITY,
        finite((o.transpose() * o - Frame::identity()).amax()),
        finite((o.determinant() - 1.0).abs()),
    ];
    // Do not send malformed matrices to the eigensolver.
    if result[1] > TOLERANCE || result[2] > TOLERANCE {
        return result;
    }
    let [a, b, target] = [c, g, t].map(spectrum);
    if a.iter()
        .chain(&b)
        .chain(&target)
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return result;
    }
    let o = o.map(|v| Complex::new(v, 0.0));
    let a = Matrix4::from_diagonal(&nalgebra::Vector4::from(a));
    let b = Matrix4::from_diagonal(&nalgebra::Vector4::from(b));
    let Some(actual) = nalgebra::linalg::Schur::try_new(a * o * b * o.transpose(), 1e-14, 1000)
        .and_then(|schur| schur.eigenvalues())
    else {
        return result;
    };
    // All 24 bijections and both global signs preserve root multiplicities.
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                if i == j || i == k || j == k {
                    continue;
                }
                let order = [i, j, k, 6 - i - j - k];
                for sign in [-1.0, 1.0] {
                    let error = (0..4)
                        .map(|n| finite((actual[n] - target[order[n]] * sign).norm()))
                        .fold(0.0, f64::max);
                    result[0] = result[0].min(error);
                }
            }
        }
    }
    result
}

struct Outcome {
    elapsed: Duration,
    errors: Option<[f64; 3]>,
}
impl Outcome {
    fn passed(&self) -> bool {
        self.errors
            .is_some_and(|e| e.into_iter().all(|v| v <= TOLERANCE))
    }
    fn status(&self) -> &'static str {
        if self.errors.is_none() {
            "declined"
        } else if self.passed() {
            "passed"
        } else {
            "invalid"
        }
    }
}
fn evaluate(case: Case, solver: Solver) -> Outcome {
    let [c, g, t] = case;
    let start = Instant::now();
    let o = solver(c, g, t);
    let elapsed = start.elapsed();
    Outcome {
        elapsed,
        errors: o.map(|o| errors(case, &o)),
    }
}

#[derive(Default)]
struct Summary {
    passed: usize,
    declined: usize,
    elapsed: Duration,
    errors: Vec<[f64; 3]>,
    failed: Vec<usize>,
}
impl Summary {
    fn add(&mut self, index: usize, outcome: &Outcome) {
        self.elapsed += outcome.elapsed;
        self.passed += usize::from(outcome.passed());
        if !outcome.passed() {
            self.failed.push(index);
        }
        if let Some(errors) = outcome.errors {
            self.errors.push(errors);
        } else {
            self.declined += 1;
        }
    }
    fn print(&self, name: &str, count: usize) {
        println!(
            "{name}: {}/{} passed, {} declined, {} invalid; {:.6}s in solver ({:.3} us/case)",
            self.passed,
            count,
            self.declined,
            count - self.passed - self.declined,
            self.elapsed.as_secs_f64(),
            self.elapsed.as_secs_f64() * 1e6 / count as f64
        );
        if !self.failed.is_empty() {
            println!(
                "  failed rows (first 20): {:?}",
                &self.failed[..self.failed.len().min(20)]
            );
        }
        if self.errors.is_empty() {
            return;
        }
        for (i, label) in ["spectrum", "orthogonality", "determinant"]
            .iter()
            .enumerate()
        {
            let mut values: Vec<_> = self.errors.iter().map(|e| e[i]).collect();
            values.sort_unstable_by(f64::total_cmp);
            let last = values.len() - 1;
            println!(
                "  {label}: p50={:.3e} p99={:.3e} max={:.3e}",
                values[last / 2],
                values[last * 99 / 100],
                values[last]
            );
        }
    }
}

/// Run the corpus comparison using arguments from the candidate executable.
///
/// The candidate has the same signature as [`crate::solve`]. Return this exit
/// code from `main`. Both algorithms run in this process; panics and hangs are
/// not isolated. Use an external sandbox or timeout when needed.
///
/// ```no_run
/// fn main() -> std::process::ExitCode {
///     can_sandwich::corpus::compare(can_sandwich::solve)
/// }
/// ```
pub fn compare(candidate: Solver) -> ExitCode {
    match run(candidate) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

fn run(candidate: Solver) -> Result<bool, Box<dyn std::error::Error>> {
    let mut corpus = None;
    let mut selected = None;
    let mut report = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("{HELP}");
            return Ok(true);
        }
        match arg.to_str() {
            Some("--corpus") if corpus.is_none() => {
                corpus = Some(PathBuf::from(args.next().ok_or("--corpus needs a path")?))
            }
            Some("--case") if selected.is_none() => {
                selected = Some(
                    args.next()
                        .ok_or("--case needs an index")?
                        .to_str()
                        .ok_or("invalid case index")?
                        .parse::<usize>()?,
                )
            }
            Some("--report") if report.is_none() => {
                report = Some(PathBuf::from(args.next().ok_or("--report needs a path")?))
            }
            _ => {
                return Err(format!(
                    "unknown or repeated argument: {}\n{HELP}",
                    arg.to_string_lossy()
                )
                .into());
            }
        }
    }
    let path = corpus.ok_or(HELP)?;
    let rows = cases(&path)?;
    let range = match selected {
        Some(index) if index < rows.len() => {
            println!("case {index}: {:?}", rows[index]);
            index..index + 1
        }
        Some(_) => return Err(format!("case index must be below {}", rows.len()).into()),
        None => 0..rows.len(),
    };
    // Never overwrite an existing report, corpus, or source file.
    let mut report = report
        .map(|p| File::create_new(p).map(BufWriter::new))
        .transpose()?;
    if let Some(w) = &mut report {
        writeln!(
            w,
            "case,solver,status,nanoseconds,spectrum_error,orthogonality_error,determinant_error"
        )?;
    }
    println!(
        "corpus: {} ({} rows); testing {}; tolerance: {TOLERANCE:e}",
        path.display(),
        rows.len(),
        range.len()
    );
    // Warm both paths before timing. Alternate call order to reduce systematic bias.
    for index in range.clone().take(16) {
        let [c, g, t] = rows[index];
        std::hint::black_box(solve(c, g, t));
        std::hint::black_box(candidate(c, g, t));
    }
    let mut summaries = [Summary::default(), Summary::default()];
    let mut fixed = Vec::new();
    let mut regressed = Vec::new();
    let mut spectral_changes = [0usize; 3];
    for index in range.clone() {
        let case = rows[index];
        let results = if index % 2 == 0 {
            [evaluate(case, solve), evaluate(case, candidate)]
        } else {
            let proposed = evaluate(case, candidate);
            [evaluate(case, solve), proposed]
        };
        if !results[0].passed() && results[1].passed() {
            fixed.push(index);
        }
        if results[0].passed() && !results[1].passed() {
            regressed.push(index);
        }
        if let (Some(a), Some(b)) = (results[0].errors, results[1].errors)
            && a[0].is_finite()
            && b[0].is_finite()
        {
            spectral_changes[match b[0].total_cmp(&a[0]) {
                std::cmp::Ordering::Less => 0,
                std::cmp::Ordering::Equal => 1,
                std::cmp::Ordering::Greater => 2,
            }] += 1;
        }
        for (slot, name) in ["production", "candidate"].iter().enumerate() {
            let result = &results[slot];
            summaries[slot].add(index, result);
            if let Some(w) = &mut report {
                write!(
                    w,
                    "{index},{name},{},{}",
                    result.status(),
                    result.elapsed.as_nanos()
                )?;
                if let Some(e) = result.errors {
                    writeln!(w, ",{:.17e},{:.17e},{:.17e}", e[0], e[1], e[2])?;
                } else {
                    writeln!(w, ",,,")?;
                }
            }
        }
    }
    if let Some(w) = &mut report {
        w.flush()?;
    }
    for (summary, name) in summaries.iter().zip(["production", "candidate"]) {
        summary.print(name, range.len());
    }
    for (label, indices) in [("fixed", fixed), ("regressed", regressed)] {
        println!(
            "{label}: {} rows; first 20: {:?}",
            indices.len(),
            &indices[..indices.len().min(20)]
        );
    }
    println!(
        "spectral error smaller/equal/larger: {}/{}/{} (finite comparisons, no rounding threshold)",
        spectral_changes[0], spectral_changes[1], spectral_changes[2]
    );
    println!(
        "Replay a row with --corpus {} --case INDEX. Use --report PATH for all row results.",
        path.display()
    );
    Ok(summaries[1].passed == range.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn production(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Frame> {
        let (o, left, right, phase) = crate::solve_with_factors(c, g, t)?;
        let diagonal = |m: [f64; 3]| {
            let [x, y, z] = [m[0] + m[1], m[0] + m[2], m[1] + m[2]];
            let d = [x - y + z, x + y - z, -x - y - z, -x + y + z]
                .map(|v| Complex::from_polar(1.0, std::f64::consts::FRAC_PI_2 * v));
            Matrix4::from_diagonal(&nalgebra::Vector4::from(d))
        };
        for factor in [&left, &right] {
            assert!((factor.transpose() * factor - Frame::identity()).amax() < 1e-8);
            assert!((factor.determinant() - 1.0).abs() < 1e-8);
        }
        let complex = |v| Complex::new(v, 0.0);
        let actual = diagonal(c) * o.map(complex) * diagonal(g);
        let reconstructed =
            left.map(complex) * diagonal(t) * right.map(complex) * Complex::from_polar(1.0, phase);
        assert!((actual - reconstructed).iter().all(|v| v.norm() < 1e-8));
        Some(o)
    }

    #[test]
    fn rejects_invalid_inputs() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for index in 0..3 {
                let mut case = [[0.0; 3]; 3];
                case[index][0] = value;
                assert!(solve(case[0], case[1], case[2]).is_none());
            }
        }
        let c = [0.17, 0.04, -0.09];
        let wrong = [0.21, 0.03, -0.08];
        for (left, right) in [(c, [0.0; 3]), ([0.0; 3], c)] {
            assert!(solve(left, right, wrong).is_none());
        }
    }

    #[test]
    fn checker() {
        let check = |case, o: &Frame| errors(case, o).into_iter().all(|e| e <= TOLERANCE);
        let identity = Frame::identity();
        assert!(check([[0.0; 3]; 3], &identity));
        assert!(check([[0.0; 3], [0.0; 3], [0.5; 3]], &identity));
        // Repeated spectrum: the previous general eigensolver returned a spurious zero.
        let case = [
            [0.3125, 0.125, -0.125],
            [0.3125, 0.0625, -0.0625],
            [0.125; 3],
        ];
        let (a, b) = (0.8408964152537146, 0.5411961001461969);
        let frame = Frame::from_row_slice(&[
            -a, 0.0, b, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, b, 0.0, a, 0.0,
        ]);
        assert!(check(case, &frame));
        assert!(!check([[0.0; 3], [0.0; 3], [0.125, 0.0, 0.0]], &identity));
        assert!(!check([[0.0; 3]; 3], &(identity * 2.0)));
        let mut reflection = identity;
        reflection[(0, 0)] = -1.0;
        assert!(!check([[0.0; 3]; 3], &reflection));
        assert!(!check([[0.0; 3]; 3], &Frame::repeat(f64::NAN)));
    }

    #[test]
    fn production_cases() {
        let rows = cases(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/cases.bin"
        )))
        .unwrap();
        for (index, case) in rows.into_iter().enumerate() {
            let result = evaluate(case, production);
            assert!(
                result.passed(),
                "row {index}: {} {:?}",
                result.status(),
                result.errors
            );
        }
    }
}
