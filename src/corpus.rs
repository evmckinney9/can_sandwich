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
const TOLERANCE: f64 = 1e-12;
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
    // Shift and scale clustered spectra before the independent complex Schur
    // solve. Near a scalar matrix, unscaled QR can stall despite a valid frame.
    let matrix = a * o * b * o.transpose();
    let center = matrix.trace() / 4.0;
    let shifted = matrix - Matrix4::identity() * center;
    let scale = shifted.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let actual = if scale == 0.0 {
        nalgebra::Vector4::repeat(center)
    } else {
        let normalized = shifted / Complex::new(scale, 0.0);
        // An oblique rotation also changes the QR path when swapping real
        // and imaginary parts is insufficient near a repeated root.
        let Some(roots) = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.6, 0.8),
        ]
        .into_iter()
        .find_map(|phase| {
            nalgebra::linalg::Schur::try_new(normalized * phase, 1e-14, 1000)
                .and_then(|schur| schur.eigenvalues())
                .map(|roots| roots.map(|z| z / phase))
        }) else {
            return result;
        };
        roots.map(|z| z * scale + center)
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
    infeasible: bool,
}
impl Outcome {
    fn passed(&self) -> bool {
        if self.infeasible {
            self.errors.is_none()
        } else {
            self.errors
                .is_some_and(|e| e.into_iter().all(|v| v <= TOLERANCE))
        }
    }
    fn status(&self) -> &'static str {
        if self.infeasible && self.errors.is_none() {
            "rejected_infeasible"
        } else if self.errors.is_none() {
            "declined"
        } else if self.passed() {
            "passed"
        } else {
            "invalid"
        }
    }
}

/// Four rank-two Horn inequalities already used by tests/generate.py imply
/// |t1+t2| <= min(a, 1-a), a=c0+c2+g0+g2, for ordered alcove coordinates.
/// The other central lift negates t1+t2, so the same bound excludes both.
/// A 1e-12 margin in phase turns exceeds the checker’s root-error allowance.
/// This is a sufficient rejection certificate, not a feasibility oracle.
fn proven_infeasible([c, g, t]: Case) -> bool {
    let in_alcove = |m: [f64; 3]| {
        let fourth = -m.iter().sum::<f64>();
        m[0] >= m[1] && m[1] >= m[2] && m[2] >= fourth && m[0] - fourth <= 1.0
    };
    let a = c[0] + c[2] + g[0] + g[2];
    [c, g, t].into_iter().all(in_alcove) && (t[1] + t[2]).abs() > a.min(1.0 - a) + 1e-12
}

fn evaluate(case: Case, solver: Solver) -> Outcome {
    let [c, g, t] = case;
    let start = Instant::now();
    let o = solver(c, g, t);
    let elapsed = start.elapsed();
    Outcome {
        elapsed,
        errors: o.map(|o| errors(case, &o)),
        infeasible: proven_infeasible(case),
    }
}

#[derive(Default)]
struct Summary {
    passed: usize,
    declined: usize,
    rejected_infeasible: usize,
    elapsed: Duration,
    timings: Vec<Duration>,
    slowest: (usize, Duration),
    errors: Vec<[f64; 3]>,
    failed: Vec<usize>,
}
impl Summary {
    fn add(&mut self, index: usize, outcome: &Outcome) {
        self.elapsed += outcome.elapsed;
        self.timings.push(outcome.elapsed);
        if self.timings.len() == 1 || outcome.elapsed > self.slowest.1 {
            self.slowest = (index, outcome.elapsed);
        }
        self.passed += usize::from(outcome.passed());
        if !outcome.passed() {
            self.failed.push(index);
        }
        if let Some(errors) = outcome.errors {
            self.errors.push(errors);
        } else if outcome.infeasible {
            self.rejected_infeasible += 1;
        } else {
            self.declined += 1;
        }
    }
    fn print(&self, name: &str, count: usize) {
        println!(
            "{name}: {}/{} passed ({} rejected infeasible), {} declined, {} invalid; {:.6}s in solver ({:.3} us/case)",
            self.passed,
            count,
            self.rejected_infeasible,
            self.declined,
            count - self.passed - self.declined,
            self.elapsed.as_secs_f64(),
            self.elapsed.as_secs_f64() * 1e6 / count as f64
        );
        let mut timings = self.timings.clone();
        timings.sort_unstable();
        let n = timings.len();
        let median = (timings[(n - 1) / 2].as_secs_f64() + timings[n / 2].as_secs_f64()) * 0.5;
        println!(
            "  latency us: median={:.3} p95={:.3} p99={:.3} p99.9={:.3} max={:.3} (row {})",
            median * 1e6,
            timings[(n * 95).div_ceil(100) - 1].as_secs_f64() * 1e6,
            timings[(n * 99).div_ceil(100) - 1].as_secs_f64() * 1e6,
            timings[(n * 999).div_ceil(1000) - 1].as_secs_f64() * 1e6,
            self.slowest.1.as_secs_f64() * 1e6,
            self.slowest.0
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
                "  {label}: p50={:.3e} p99={:.3e} p99.9={:.3e} max={:.3e}",
                values[last / 2],
                values[last * 99 / 100],
                values[last * 999 / 1000],
                values[last]
            );
            if i == 0 {
                println!(
                    "  spectral errors above: 1e-13={} 1e-12={} 1e-10={}",
                    values.iter().filter(|&&v| v > 1e-13).count(),
                    values.iter().filter(|&&v| v > 1e-12).count(),
                    values.iter().filter(|&&v| v > 1e-10).count(),
                );
            }
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
        let (o, left, right, phase) = crate::solve_with_factors(c, g, t).ok()?;
        let diagonal = |m: [f64; 3]| {
            let [x, y, z] = [m[0] + m[1], m[0] + m[2], m[1] + m[2]];
            let d = [x - y + z, x + y - z, -x - y - z, -x + y + z]
                .map(|v| Complex::from_polar(1.0, std::f64::consts::FRAC_PI_2 * v));
            Matrix4::from_diagonal(&nalgebra::Vector4::from(d))
        };
        for factor in [&left, &right] {
            assert!((factor.transpose() * factor - Frame::identity()).amax() < TOLERANCE);
            assert!((factor.determinant() - 1.0).abs() < TOLERANCE);
        }
        let complex = |v| Complex::new(v, 0.0);
        let actual = diagonal(c) * o.map(complex) * diagonal(g);
        let reconstructed =
            left.map(complex) * diagonal(t) * right.map(complex) * Complex::from_polar(1.0, phase);
        assert!(
            (actual - reconstructed)
                .iter()
                .all(|v| v.norm() < TOLERANCE)
        );
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
        // Independent LAPACK verification gives a 6.7e-15 spectral error.
        // The first two Schur phases exhaust their iteration budgets here.
        assert!(check(
            [
                [
                    0.4999999999998125,
                    0.49999999999968747,
                    -0.49999999999956246
                ],
                [0.13619345023463234, 0.13619345023450732, 0.1361934502342573],
                [0.13619345023472765, 0.1361934502344448, 0.13619345023416196]
            ],
            &Frame::from_column_slice(&[
                -0.5440939537008301,
                0.40100367204652826,
                -0.33248713478895436,
                0.6577310466681887,
                0.6962930678850977,
                0.5431705545411705,
                0.27197803806683357,
                0.38232140811614473,
                0.2125518471778125,
                -0.7376324064276074,
                0.029021196984141157,
                0.6402170845695422,
                -0.41708445838062264,
                0.007761228797976285,
                0.9025730010003377,
                0.10650021488303599
            ]),
        ));
        assert!(!check([[0.0; 3], [0.0; 3], [0.125, 0.0, 0.0]], &identity));
        assert!(!check([[0.0; 3]; 3], &(identity * 2.0)));
        let mut reflection = identity;
        reflection[(0, 0)] = -1.0;
        assert!(!check([[0.0; 3]; 3], &reflection));
        assert!(!check([[0.0; 3]; 3], &Frame::repeat(f64::NAN)));

        // The corpus also contains near-feasible inputs. A certified rejection
        // must pass; an approximate witness for that input must not.
        let impossible = [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.2, -0.20000001]];
        assert!(proven_infeasible(impossible));
        assert!(evaluate(impossible, |_, _, _| None).passed());
        assert!(!evaluate(impossible, |_, _, _| Some(Frame::identity())).passed());
        assert!(!proven_infeasible([
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.3, 0.2, -0.2],
        ]));
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
