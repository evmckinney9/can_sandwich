//! Corpus driver for the depth-two realization solver: coverage, residuals,
//! latency, and rung attribution over an `(N, 3, 3)` array of monodromy
//! triples `[C, G, T]`.
//!
//! - `bench-npy <path> [stride]`: every stride-th row, aggregate report.
//! - `bench-row <path> <row> [repetitions]`: one row repeated, latency quantiles.
//! - `bench-triple c0 c1 c2 g0 g1 g2 t0 t1 t2`: one literal triple.
//!
//! The exit status is nonzero when any sampled row is unsolved.
#![allow(clippy::print_stdout, clippy::print_stderr)]
use std::collections::BTreeMap;

use can_sandwich::{branch_signature, init_tables, prof, solve, Rung, Solution};

/// Parse a `.npy` header, returning `(shape, data_offset)`.
fn npy_header(bytes: &[u8]) -> (Vec<usize>, usize) {
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a .npy file");
    let (hlen, hstart) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => panic!("unsupported .npy version {v}"),
    };
    let head = std::str::from_utf8(&bytes[hstart..hstart + hlen]).unwrap();
    assert!(head.contains("'fortran_order': False"), "need C-order .npy");
    let s = head.find("'shape':").unwrap();
    let open = head[s..].find('(').unwrap() + s + 1;
    let close = head[open..].find(')').unwrap() + open;
    let shape = head[open..close]
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    (shape, hstart + hlen)
}

/// Read a little-endian C-order `(N, 3, 3)` f64 array as flat `[C, G, T]` rows.
fn read_triples(path: &str) -> Vec<[f64; 9]> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let (shape, off) = npy_header(&bytes);
    assert_eq!(&shape[1..], &[3, 3], "expected (N,3,3), got {shape:?}");
    (0..shape[0])
        .map(|i| {
            let mut row = [0.0f64; 9];
            for (k, slot) in row.iter_mut().enumerate() {
                let p = off + (i * 9 + k) * 8;
                *slot = f64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
            }
            row
        })
        .collect()
}

fn solve_row(r: &[f64; 9]) -> Solution {
    let (c, g, t) = ([r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]);
    if std::env::var_os("CHARTS_ONLY").is_some() {
        can_sandwich::solve_charts_only(c, g, t)
    } else {
        solve(c, g, t)
    }
}

fn usable(sol: &Solution) -> bool {
    sol.rung != Rung::Unsolved
}

fn micros(ns: u128) -> f64 {
    ns as f64 / 1e3
}

fn bench_npy(path: &str, stride: usize) -> bool {
    let triples = read_triples(path);
    init_tables();
    let mut by_rung: BTreeMap<Rung, usize> = BTreeMap::new();
    let mut timings: Vec<(u128, usize, Rung)> = Vec::new();
    let (mut solved, mut machine) = (0usize, 0usize);
    let (mut worst_residual, mut worst_det) = (0.0f64, 0.0f64);
    let t0 = std::time::Instant::now();
    for (k, r) in triples.iter().enumerate().step_by(stride) {
        let ts = std::time::Instant::now();
        let sol = solve_row(r);
        timings.push((ts.elapsed().as_nanos(), k, sol.rung));
        *by_rung.entry(sol.rung).or_default() += 1;
        if usable(&sol) {
            solved += 1;
            worst_residual = worst_residual.max(sol.residual);
            worst_det = worst_det.max((sol.o.map(|z| z.re).determinant() - 1.0).abs());
            machine += usize::from(sol.residual < 1e-12);
        } else {
            println!(
                "UNSOLVED {k} C {:?} G {:?} T {:?}",
                &r[0..3],
                &r[3..6],
                &r[6..9]
            );
        }
    }
    let n = timings.len();
    let us = t0.elapsed().as_secs_f64() * 1e6 / n.max(1) as f64;
    timings.sort_unstable_by_key(|x| x.0);
    let quantile = |q: f64| timings[(((n - 1) as f64) * q).round() as usize];
    println!(
        "dataset {path} rows {} stride {stride} sampled {n}",
        triples.len()
    );
    println!(
        "OVERALL: {solved}/{n} = {:.4}% solved; {us:.2} us/triple; worst residual {worst_residual:.3e}; \
         worst |det(O)-1| {worst_det:.3e}; machine-precise {machine}/{solved} = {:.4}%",
        100.0 * solved as f64 / n.max(1) as f64,
        100.0 * machine as f64 / solved.max(1) as f64,
    );
    println!("by rung: {by_rung:?}");
    let mut per_rung: BTreeMap<Rung, (u128, usize, u128, usize)> = BTreeMap::new();
    for &(ns, row, rung) in &timings {
        let e = per_rung.entry(rung).or_default();
        e.0 += ns;
        e.1 += 1;
        if ns > e.2 {
            (e.2, e.3) = (ns, row);
        }
    }
    for (rung, (ns, cnt, max_ns, max_row)) in &per_rung {
        println!(
            "  rung {rung:?}: n {cnt}, total {:.1} ms, mean {:.2} us, max {:.2} us at row {max_row}",
            *ns as f64 / 1e6,
            micros(*ns) / *cnt as f64,
            micros(*max_ns),
        );
    }
    let worst = timings[n - 1];
    println!(
        "latency us: p50 {:.2}; p99 {:.2}; p99.9 {:.2}; max {:.2} at row {} rung {:?}",
        micros(quantile(0.5).0),
        micros(quantile(0.99).0),
        micros(quantile(0.999).0),
        micros(worst.0),
        worst.1,
        worst.2,
    );
    prof::dump();
    solved == n
}

/// Full-corpus routing census. Each row contributes to a stable endpoint
/// signature and the first certified solver rung, making branch changes
/// auditable across frozen corpora.
fn census_npy(path: &str, stride: usize) -> bool {
    let triples = read_triples(path);
    init_tables();
    let mut bins: BTreeMap<(String, Rung), (usize, u128)> = BTreeMap::new();
    let mut signature_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut solved = 0usize;
    for (row_index, r) in triples.iter().enumerate().step_by(stride) {
        let c = [r[0], r[1], r[2]];
        let g = [r[3], r[4], r[5]];
        let t = [r[6], r[7], r[8]];
        let sig = branch_signature(c, g, t);
        let started = std::time::Instant::now();
        let sol = solve(c, g, t);
        let elapsed = started.elapsed().as_nanos();
        if sol.rung != Rung::Unsolved {
            solved += 1;
        }
        *signature_totals.entry(sig.clone()).or_default() += 1;
        let entry = bins.entry((sig.clone(), sol.rung)).or_default();
        entry.0 += 1;
        entry.1 += elapsed;
        if std::env::var_os("CENSUS_LATE_ROWS").is_some() && sol.rung == Rung::Chart {
            println!("LATE row={row_index} mean_us={:.2} {sig}", elapsed as f64 / 1e3);
        }
    }
    println!("dataset {path} rows {} stride {stride} sampled {} solved {solved}", triples.len(), triples.len().div_ceil(stride));
    println!("SIGNATURE_TOTALS");
    for (sig, n) in &signature_totals {
        println!("{n}\t{sig}");
    }
    println!("SIGNATURE_X_RUNG");
    println!("columns: count<TAB>mean_us<TAB>tier<TAB>rung<TAB>signature");
    for ((sig, rung), (n, ns)) in &bins {
        println!("{n}\t{:.2}\t{}\t{rung:?}\t{sig}", *ns as f64 / *n as f64 / 1e3, rung_tier(*rung));
    }
    solved == triples.len().div_ceil(stride)
}

fn rung_tier(rung: Rung) -> &'static str {
    match rung {
        Rung::Vertex | Rung::Edge | Rung::Face | Rung::OnePlusThree |
        Rung::RankOne31 | Rung::Pair22 | Rung::Chart | Rung::Radical => "exact-certified-finite",
        Rung::NearRankOne31 => "certified-near-candidate",
        // These are non-iterative and certificate-gated, but their dispatch
        // schedules are incomplete rather than universal closed-form leaves.
        Rung::Interior | Rung::Klein => "exact-certified-incomplete",
        Rung::Unsolved => "none",
    }
}

fn bench_row(path: &str, k: usize, reps: usize) {
    let triples = read_triples(path);
    init_tables();
    let r = triples[k];
    let mut nanos = Vec::with_capacity(reps);
    let mut answer = None;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let sol = solve_row(&r);
        nanos.push(t0.elapsed().as_nanos());
        answer = Some((sol.rung, sol.residual));
    }
    nanos.sort_unstable();
    prof::dump();
    let (rung, residual) = answer.unwrap();
    println!(
        "row {k} reps {reps} rung {rung:?} residual {residual:.3e} \
         latency us average {:.2} min {:.2} median {:.2} max {:.2}",
        micros(nanos.iter().sum::<u128>()) / reps as f64,
        micros(nanos[0]),
        micros(nanos[nanos.len() / 2]),
        micros(nanos[nanos.len() - 1]),
    );
}

fn bench_triple(values: &[f64]) {
    let r: [f64; 9] = values.try_into().expect("nine coordinates");
    let started = std::time::Instant::now();
    let sol = solve_row(&r);
    println!(
        "rung {:?} residual {:.6e} latency_us {:.3}",
        sol.rung,
        sol.residual,
        started.elapsed().as_secs_f64() * 1e6
    );
    prof::dump();
}

/// Separate from the production coefficient/certificate path: diagonalize the
/// actual sandwich and compare all root permutations and both target lifts.
fn paired_check(r: &[f64; 9], sol: &Solution) -> f64 {
    use nalgebra::{Complex, Matrix4};
    let roots = |m: &[f64]| {
        let w = [m[0] + m[1], m[0] + m[2], m[1] + m[2]];
        [
            w[0] - w[1] + w[2],
            w[0] + w[1] - w[2],
            -w[0] - w[1] - w[2],
            -w[0] + w[1] + w[2],
        ]
        .map(|x| Complex::from_polar(1.0, std::f64::consts::PI * x))
    };
    if sol.o.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return f64::INFINITY;
    }
    let real = sol.o.map(|z| z.re);
    let gram = (real.transpose() * real - Matrix4::identity()).amax();
    let frame = gram
        .max((real.determinant() - 1.0).abs())
        .max(sol.o.iter().map(|z| z.im.abs()).fold(0.0, f64::max));
    let a = Matrix4::from_diagonal(&nalgebra::Vector4::from(roots(&r[..3])));
    let b = Matrix4::from_diagonal(&nalgebra::Vector4::from(roots(&r[3..6])));
    let Some(ev) = (a * sol.o * b * sol.o.transpose()).eigenvalues() else {
        return f64::INFINITY;
    };
    let target = roots(&r[6..]);
    let mut best = f64::INFINITY;
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                for l in 0..4 {
                    if i == j || i == k || i == l || j == k || j == l || k == l {
                        continue;
                    }
                    for sign in [-1.0, 1.0] {
                        let e = [i, j, k, l]
                            .iter()
                            .enumerate()
                            .map(|(n, &p)| (ev[n] - target[p] * sign).norm())
                            .fold(0.0, f64::max);
                        best = best.min(e);
                    }
                }
            }
        }
    }
    frame.max(best)
}

fn bench_paired(path: &str, stride: usize) -> bool {
    use can_sandwich::{paired_edge_scope, solve_paired_edges, solve_paired_edges_forward};
    let triples = read_triples(path);
    init_tables();
    let mut times: [Vec<u128>; 3] = std::array::from_fn(|_| Vec::new());
    let mut counts = [0usize; 3];
    let mut worst = [0.0f64; 3];
    let (mut exact, mut near, mut invalid, mut added, mut lost) = (0, 0, 0, 0, 0);
    for (row, r) in triples.iter().enumerate().step_by(stride) {
        let (c, g, t) = ([r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]);
        let Some(gap) = paired_edge_scope(c, g, t) else {
            continue;
        };
        if gap <= 1e-12 {
            exact += 1;
        } else {
            near += 1;
        }
        let mut success = [false; 3];
        for offset in 0..3 {
            let method = (row + offset) % 3;
            let started = std::time::Instant::now();
            let sol = match method {
                0 => solve(c, g, t),
                1 => solve_paired_edges_forward(c, g, t),
                _ => solve_paired_edges(c, g, t),
            };
            times[method].push(started.elapsed().as_nanos());
            if usable(&sol) {
                let err = paired_check(r, &sol);
                worst[method] = worst[method].max(err);
                success[method] = err <= 1e-8;
                counts[method] += usize::from(success[method]);
                if !success[method] {
                    invalid += 1;
                    println!("INVALID row {row} method {method} gap {gap:.3e} error {err:.3e}");
                }
            }
        }
        added += usize::from(success[2] && !success[1]);
        lost += usize::from(success[0] && !success[2]);
        if !success[2] || (success[2] && !success[1]) {
            println!("CASE row {row} gap {gap:.3e} production {} forward {} bidirectional {} C {:?} G {:?} T {:?}", success[0],success[1],success[2],c,g,t);
        }
    }
    println!("PAIRED dataset {path} total {} stride {stride} applicable {} exact_gap_le_1e-12 {exact} near {near} added_by_backward {added} lost_vs_production {lost} invalid {invalid}", triples.len(),exact+near);
    for method in 0..3 {
        times[method].sort_unstable();
        let ns = &times[method];
        let q = |p: f64| {
            if ns.is_empty() {
                0.0
            } else {
                micros(ns[((ns.len() - 1) as f64 * p).round() as usize])
            }
        };
        println!("METHOD {} solved {}/{} mean_us {:.3} p50_us {:.3} p99_us {:.3} max_us {:.3} worst_check {:.3e}", ["production","forward","bidirectional"][method],counts[method],exact+near,micros(ns.iter().sum())/ns.len().max(1) as f64,q(0.5),q(0.99),q(1.0),worst[method]);
    }
    invalid == 0 && counts[2] == exact + near
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parse = |s: &String| s.parse().expect("integer argument");
    let ok = match args.first().map(String::as_str) {
        Some("bench-paired") => bench_paired(&args[1], args.get(2).map_or(1, parse)),
        Some("bench-npy") => bench_npy(&args[1], args.get(2).map_or(1, parse)),
        Some("census-npy") => census_npy(&args[1], args.get(2).map_or(1, parse)),
        Some("bench-row") => {
            bench_row(&args[1], parse(&args[2]), args.get(3).map_or(1, parse));
            true
        }
        Some("bench-triple") => {
            let values: Vec<f64> = args[1..]
                .iter()
                .map(|v| v.parse().expect("coordinate"))
                .collect();
            bench_triple(&values);
            true
        }
        _ => {
            eprintln!("usage: can_sandwich bench-npy <path> [stride] | census-npy <path> [stride] | bench-row <path> <row> [reps] | bench-triple <9 coordinates>");
            false
        }
    };
    if !ok {
        std::process::exit(1);
    }
}
