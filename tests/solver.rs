use can_sandwich::{Rung, solve};
use nalgebra::{Complex, Matrix4};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

type Case = [[f64; 3]; 3];
type Frame = Matrix4<f64>;
type Solver = fn([f64; 3], [f64; 3], [f64; 3]) -> Option<Frame>;

fn cases() -> Vec<Case> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cases.bin")).unwrap();
    assert!(
        !bytes.is_empty() && bytes.len().is_multiple_of(72),
        "empty or truncated fixture"
    );
    bytes
        .chunks_exact(72)
        .map(|row| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let offset = (3 * i + j) * 8;
                    let value = f64::from_le_bytes(row[offset..offset + 8].try_into().unwrap());
                    assert!(value.is_finite(), "nonfinite fixture coordinate");
                    value
                })
            })
        })
        .collect()
}

fn production(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Frame> {
    let solution = solve(c, g, t);
    if solution.rung == Rung::Unsolved || solution.o.iter().any(|z| z.im != 0.0) {
        return None;
    }
    Some(solution.o.map(|z| z.re))
}

fn spectrum(m: [f64; 3]) -> [Complex<f64>; 4] {
    let [x, y, z] = [m[0] + m[1], m[0] + m[2], m[1] + m[2]];
    [x - y + z, x + y - z, -x - y - z, -x + y + z]
        .map(|v| Complex::from_polar(1.0, std::f64::consts::PI * v))
}

// Check the actual matrix spectrum, independently of the solver's certificate.
fn check([c, g, t]: Case, o: &Frame) -> bool {
    if o.iter().any(|v| !v.is_finite())
        || !(o.transpose() * o - Frame::identity())
            .iter()
            .all(|v| v.abs() <= 1e-8)
        || !o.determinant().is_finite()
        || (o.determinant() - 1.0).abs() > 1e-8
    {
        return false;
    }
    let [a, b, target] = [c, g, t].map(spectrum);
    let o = o.map(|v| Complex::new(v, 0.0));
    let a = Matrix4::from_diagonal(&nalgebra::Vector4::from(a));
    let b = Matrix4::from_diagonal(&nalgebra::Vector4::from(b));
    let product = a * o * b * o.transpose();
    let Some(actual) = nalgebra::linalg::Schur::try_new(product, 1e-14, 1000)
        .and_then(|schur| schur.eigenvalues())
    else {
        return false;
    };
    // All 24 bijections; repeated eigenvalues must retain their multiplicities.
    for i in 0..4 {
        for j in 0..4 {
            if j == i {
                continue;
            }
            for k in 0..4 {
                if k == i || k == j {
                    continue;
                }
                let order = [i, j, k, 6 - i - j - k];
                for sign in [-1.0, 1.0] {
                    if (0..4).all(|n| (actual[n] - target[order[n]] * sign).norm() <= 1e-8) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn evaluate(rows: &[Case], solver: Solver) -> (BTreeSet<usize>, Duration) {
    let mut failures = BTreeSet::new();
    let mut time = Duration::ZERO;
    for (index, &[c, g, t]) in rows.iter().enumerate() {
        let start = Instant::now();
        let frame = solver(c, g, t);
        time += start.elapsed();
        if !frame.is_some_and(|o| check([c, g, t], &o)) {
            failures.insert(index);
        }
    }
    (failures, time)
}

#[test]
fn production_cases() {
    let rows = cases();
    let (failed, time) = evaluate(&rows, production);
    assert!(
        failed.is_empty(),
        "{}/{} failed, {time:?} in solver; rows: {:?}",
        failed.len(),
        rows.len(),
        failed
    );
}

#[test]
fn rejects_invalid_inputs() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in 0..3 {
            let mut case = [[0.0; 3]; 3];
            case[index][0] = value;
            assert_eq!(solve(case[0], case[1], case[2]).rung, Rung::Unsolved);
        }
    }
    let c = [0.17, 0.04, -0.09];
    let wrong = [0.21, 0.03, -0.08];
    for (left, right) in [(c, [0.0; 3]), ([0.0; 3], c)] {
        assert_eq!(solve(left, right, wrong).rung, Rung::Unsolved);
    }
}

#[test]
fn checker() {
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
#[ignore = "replace candidate with a prototype, then run explicitly"]
#[allow(clippy::print_stdout)]
fn compare_candidate() {
    // Replace this function with the algorithm under test, or call a local module.
    fn candidate(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Frame> {
        production(c, g, t)
    }
    let rows = cases();
    let (baseline, baseline_time) = evaluate(&rows, production);
    let (proposed, proposed_time) = evaluate(&rows, candidate);
    println!(
        "production: {}/{} passed, {baseline_time:?}",
        rows.len() - baseline.len(),
        rows.len()
    );
    println!(
        "candidate:  {}/{} passed, {proposed_time:?}",
        rows.len() - proposed.len(),
        rows.len()
    );
    println!(
        "fixed rows: {:?}",
        baseline.difference(&proposed).collect::<Vec<_>>()
    );
    println!(
        "regressed rows: {:?}",
        proposed.difference(&baseline).collect::<Vec<_>>()
    );
    assert!(
        proposed.is_subset(&baseline),
        "candidate regressed on previously passing cases"
    );
}
