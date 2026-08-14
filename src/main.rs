//! Driver: run the whole `feasible_linspace` triple dataset through the depth-2
//! can-sandwich solver, end-to-end, and report per-stratum coverage/residual.
//! No select/witness/transport -- the problem is already a depth-2 subproblem.
mod data;

use can_sandwich::can_sandwich;

use std::collections::BTreeMap;

const SEG: &str = "/home/evm9/gulps/.claude/scripts/diagnostics/segments";

/// Number of distinct target eigenvalues exp(2i·eigphase): ndist=4 generic, <4 = degenerate spectrum.
/// The sin2 agent's law: ndist=2 -> Y=0 face (deg-2), ndist=3 -> interior fold (deg-8 tangent).
fn ndist(t: [f64; 3]) -> usize {
    let ph = can_sandwich::eigphases(can_sandwich::weyl_from_monodromy(t));
    let mut u: Vec<(f64, f64)> = vec![];
    for p in ph {
        let (c, s) = ((2.0 * p).cos(), (2.0 * p).sin());
        if !u.iter().any(|&(a, b)| (a - c).hypot(b - s) < 1e-6) {
            u.push((c, s));
        }
    }
    u.len()
}

fn main() {
    // `bench-npy <path> [stride]`: aggregate coverage, residual, determinant,
    // timing, and rung attribution for any unlabeled (N,3,3) triple corpus.
    let args: Vec<String> = std::env::args().collect();
    if let Some(p) = args.iter().position(|a| a == "bench-triple") {
        let values: Vec<f64> = args[p + 1..p + 10]
            .iter()
            .map(|value| value.parse().expect("triple coordinate"))
            .collect();
        let started = std::time::Instant::now();
        let solution = can_sandwich::solve(
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], values[8]],
        );
        println!(
            "rung {:?} residual {:.6e} latency_us {:.3}",
            solution.rung,
            solution.residual,
            started.elapsed().as_secs_f64() * 1e6
        );
        can_sandwich::prof::dump();
        return;
    }
    #[cfg(feature = "research-spin")]
    {
        if let Some(p) = args.iter().position(|a| a == "spin-replay-row") {
            let triples = data::read_triples(&args[p + 1]);
            let row: usize = args[p + 2].parse().expect("row index");
            let triple = triples[row];
            let production = can_sandwich::solve(
                [triple[0], triple[1], triple[2]],
                [triple[3], triple[4], triple[5]],
                [triple[6], triple[7], triple[8]],
            );
            let started = std::time::Instant::now();
            let replay = can_sandwich::replay_spin_action(
                [triple[0], triple[1], triple[2]],
                [triple[3], triple[4], triple[5]],
                [triple[6], triple[7], triple[8]],
                &production.o,
            );
            println!(
                "row {row} production {:?} residual {:.3e}; replay {:?} residual {:.3e}; replay_us {:.3}",
                production.rung,
                production.residual,
                replay.rung,
                replay.residual,
                started.elapsed().as_secs_f64() * 1e6,
            );
            return;
        }
    }
    // Compare the generic Spin scheduler against rows currently owned by the
    // expensive per-chart axis tail. This is a census only; it does not alter
    // production routing.
    #[cfg(feature = "research-spin")]
    if let Some((p, spread)) = args
        .iter()
        .position(|a| a == "spin-spread-tail-census")
        .map(|p| (p, true))
        .or_else(|| {
            args.iter()
                .position(|a| a == "spin-tail-census")
                .map(|p| (p, false))
        })
    {
        let triples = data::read_triples(&args[p + 1]);
        let bound: usize = args
            .get(p + 2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(if spread { 64 } else { 2 });
        can_sandwich::init_tables();
        let mut owned = 0usize;
        let mut solved = 0usize;
        let mut replayed = 0usize;
        let mut replay_elapsed_total = 0u128;
        let mut replay_elapsed_max = 0u128;
        let mut replay_elapsed_max_row = 0usize;
        let mut attempts_total = 0usize;
        let mut attempts_max = 0usize;
        let mut actions_total = 0usize;
        let mut actions_max = 0usize;
        let mut elapsed_total = 0u128;
        let mut elapsed_max = 0u128;
        let mut elapsed_max_row = 0usize;
        for (row, triple) in triples.iter().enumerate() {
            let production = can_sandwich::solve(
                [triple[0], triple[1], triple[2]],
                [triple[3], triple[4], triple[5]],
                [triple[6], triple[7], triple[8]],
            );
            if production.rung != can_sandwich::Rung::AxisQuartic {
                continue;
            }
            owned += 1;
            let replay_started = std::time::Instant::now();
            let replay = can_sandwich::replay_spin_action(
                [triple[0], triple[1], triple[2]],
                [triple[3], triple[4], triple[5]],
                [triple[6], triple[7], triple[8]],
                &production.o,
            );
            let replay_elapsed = replay_started.elapsed().as_nanos();
            replay_elapsed_total += replay_elapsed;
            if replay_elapsed > replay_elapsed_max {
                replay_elapsed_max = replay_elapsed;
                replay_elapsed_max_row = row;
            }
            replayed += usize::from(replay.rung == can_sandwich::Rung::Spin);
            let started = std::time::Instant::now();
            let (spin, attempts, actions) = if spread {
                can_sandwich::solve_spin_spread(
                    [triple[0], triple[1], triple[2]],
                    [triple[3], triple[4], triple[5]],
                    [triple[6], triple[7], triple[8]],
                    bound,
                )
            } else {
                let (spin, attempts) = can_sandwich::solve_spin_dyadic(
                    [triple[0], triple[1], triple[2]],
                    [triple[3], triple[4], triple[5]],
                    [triple[6], triple[7], triple[8]],
                    bound as i32,
                );
                (spin, attempts, 0)
            };
            let elapsed = started.elapsed().as_nanos();
            attempts_total += attempts;
            attempts_max = attempts_max.max(attempts);
            actions_total += actions;
            actions_max = actions_max.max(actions);
            elapsed_total += elapsed;
            if elapsed > elapsed_max {
                elapsed_max = elapsed;
                elapsed_max_row = row;
            }
            solved += usize::from(spin.rung == can_sandwich::Rung::Spin);
            if std::env::var_os("DUMP_SPIN_TAIL").is_some() {
                println!(
                    "SPIN_TAIL {row} replay {:?} spin {:?} actions {actions} attempts {attempts} latency_us {:.3} residual {:.3e}",
                    replay.rung,
                    spin.rung,
                    elapsed as f64 / 1e3,
                    spin.residual,
                );
            }
        }
        println!(
            "spin true-action replay: {replayed}/{owned} axis rows; average {:.3} us; slowest {:.3} us at row {replay_elapsed_max_row}",
            replay_elapsed_total as f64 / 1e3 / owned.max(1) as f64,
            replay_elapsed_max as f64 / 1e3,
        );
        if spread {
            println!(
                "spin spread prefix {bound}: {solved}/{owned} axis rows; average {:.3} us; slowest {:.3} us at row {elapsed_max_row}; average actions {:.2}; max actions {actions_max}; average kernel calls {:.2}; max kernel calls {attempts_max}",
                elapsed_total as f64 / 1e3 / owned.max(1) as f64,
                elapsed_max as f64 / 1e3,
                actions_total as f64 / owned.max(1) as f64,
                attempts_total as f64 / owned.max(1) as f64,
            );
        } else {
            println!(
                "spin tail denominator {bound}: {solved}/{owned} axis rows; average {:.3} us; slowest {:.3} us at row {elapsed_max_row}; average attempts {:.2}; max attempts {attempts_max}",
                elapsed_total as f64 / 1e3 / owned.max(1) as f64,
                elapsed_max as f64 / 1e3,
                attempts_total as f64 / owned.max(1) as f64,
            );
        }
        return;
    }
    if args.iter().any(|a| a == "hot-classes") {
        let hot = can_sandwich::interior_hot_classes();
        let s: Vec<String> = hot.iter().map(|x| x.to_string()).collect();
        println!("{}", s.join(","));
        return;
    }
    if args.iter().any(|a| a == "class-map") {
        // 16 lines (plane_idx), 24 classes each (perm_idx)
        let m = can_sandwich::interior_class_map();
        for plane_idx in 0..16 {
            let row: Vec<String> = (0..24).map(|p| m[plane_idx * 24 + p].to_string()).collect();
            println!("{}", row.join(","));
        }
        return;
    }
    // `bench-row <path> <index> [repetitions]`: isolate a corpus row and
    // report both latency and exact interior-chart work.
    if let Some(p) = args.iter().position(|a| a == "bench-row") {
        let triples = data::read_triples(&args[p + 1]);
        can_sandwich::init_tables();
        let k: usize = args[p + 2].parse().expect("row index");
        let reps: usize = args.get(p + 3).and_then(|s| s.parse().ok()).unwrap_or(1);
        let r = triples[k];
        let mut nanos = Vec::with_capacity(reps);
        let mut answer = None;
        for _ in 0..reps {
            can_sandwich::reset_interior_instr();
            let t0 = std::time::Instant::now();
            let sol = if std::env::var_os("AXIS_ONLY").is_some() {
                can_sandwich::solve_axis_only(
                    [r[0], r[1], r[2]],
                    [r[3], r[4], r[5]],
                    [r[6], r[7], r[8]],
                )
            } else if std::env::var_os("CANONICAL_THREE_GIVENS_ONLY").is_some() {
                can_sandwich::solve_canonical_three_givens_only(
                    [r[0], r[1], r[2]],
                    [r[3], r[4], r[5]],
                    [r[6], r[7], r[8]],
                )
            } else if std::env::var_os("THREE_GIVENS_ONLY").is_some() {
                can_sandwich::solve_three_givens_only(
                    [r[0], r[1], r[2]],
                    [r[3], r[4], r[5]],
                    [r[6], r[7], r[8]],
                )
            } else {
                can_sandwich::solve([r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]])
            };
            nanos.push(t0.elapsed().as_nanos());
            answer = Some((sol.rung, sol.residual, can_sandwich::interior_instr()));
        }
        nanos.sort_unstable();
        can_sandwich::prof::dump();
        let (rung, residual, instr) = answer.unwrap();
        if std::env::var_os("GULPS_FUNNEL").is_some() {
            let counts = can_sandwich::funnel::snapshot();
            for (name, count) in can_sandwich::funnel::NAMES.iter().zip(counts) {
                if count != 0 {
                    println!("FUNNEL {name} {count}");
                }
            }
        }
        println!(
            "row {k} reps {reps} rung {rung:?} residual {residual:.3e} interior {:?} latency us average {:.2} min {:.2} median {:.2} max {:.2}",
            instr,
            nanos.iter().sum::<u128>() as f64 / reps as f64 / 1e3,
            nanos[0] as f64 / 1e3,
            nanos[nanos.len() / 2] as f64 / 1e3,
            nanos[nanos.len() - 1] as f64 / 1e3,
        );
        return;
    }
    if let Some(p) = args.iter().position(|a| a == "bench-npy") {
        let triples = data::read_triples(&args[p + 1]);
        can_sandwich::init_tables();
        let stride = args
            .get(p + 2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(1usize);
        let mut by_rung: BTreeMap<can_sandwich::Rung, usize> = BTreeMap::new();
        let mut by_signature: BTreeMap<([usize; 3], can_sandwich::Rung), usize> = BTreeMap::new();
        let mut timings: Vec<(u128, usize, can_sandwich::Rung)> = Vec::new();
        let (mut n, mut solved, mut machine) = (0usize, 0usize, 0usize);
        let (mut worst_residual, mut worst_det) = (0.0f64, 0.0f64);
        let audit_gram = std::env::var_os("GULPS_AUDIT_GRAM").is_some();
        let (mut worst_gram, mut worst_gram_row) = (0.0f64, 0usize);
        let mut gram_failures: BTreeMap<can_sandwich::Rung, usize> = BTreeMap::new();
        let mut gram_failure_signatures: BTreeMap<[usize; 3], usize> = BTreeMap::new();
        let t0 = std::time::Instant::now();
        for k in (0..triples.len()).step_by(stride) {
            let r = triples[k];
            can_sandwich::reset_interior_instr();
            let ts = std::time::Instant::now();
            let sol = if std::env::var_os("AXIS_ONLY").is_some() {
                can_sandwich::solve_axis_only(
                    [r[0], r[1], r[2]],
                    [r[3], r[4], r[5]],
                    [r[6], r[7], r[8]],
                )
            } else if std::env::var_os("CANONICAL_THREE_GIVENS_ONLY").is_some() {
                can_sandwich::solve_canonical_three_givens_only(
                    [r[0], r[1], r[2]],
                    [r[3], r[4], r[5]],
                    [r[6], r[7], r[8]],
                )
            } else if std::env::var_os("THREE_GIVENS_ONLY").is_some() {
                can_sandwich::solve_three_givens_only(
                    [r[0], r[1], r[2]],
                    [r[3], r[4], r[5]],
                    [r[6], r[7], r[8]],
                )
            } else {
                can_sandwich::solve([r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]])
            };
            timings.push((ts.elapsed().as_nanos(), k, sol.rung));
            if std::env::var_os("DUMP_RUNGS").is_some() {
                let (tries, plane, perm) = can_sandwich::interior_instr();
                println!(
                    "TRIES {k} {:?} {tries} {plane} {perm} {:.6e}",
                    sol.rung, sol.residual
                );
            }
            if std::env::var_os("GULPS_DUMP_CHART").is_some()
                && sol.rung == can_sandwich::Rung::Interior
            {
                let (tries, _, _) = can_sandwich::interior_instr();
                let (class, root) = can_sandwich::interior_class_root();
                println!("CHART {k} {tries} {class} {root}");
            }
            if std::env::var_os("GULPS_CHARTALL").is_some() {
                let vc = can_sandwich::interior_valid_classes();
                let s: Vec<String> = vc.iter().map(|x| x.to_string()).collect();
                println!("VALIDSET {k} {}", s.join(","));
            }
            *by_rung.entry(sol.rung).or_default() += 1;
            if std::env::var_os("GULPS_SIGNATURES").is_some() {
                let sig = [
                    ndist([r[0], r[1], r[2]]),
                    ndist([r[3], r[4], r[5]]),
                    ndist([r[6], r[7], r[8]]),
                ];
                *by_signature.entry((sig, sol.rung)).or_default() += 1;
            }
            if sol.rung != can_sandwich::Rung::Unsolved {
                solved += 1;
                worst_residual = worst_residual.max(sol.residual);
                worst_det = worst_det.max((sol.o.map(|z| z.re).determinant() - 1.0).abs());
                machine += usize::from(sol.residual < 1e-12);
                if audit_gram {
                    let real = sol.o.map(|value| value.re);
                    let gram = real.transpose() * real - nalgebra::Matrix4::identity();
                    let residual = gram
                        .iter()
                        .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
                    if residual > worst_gram {
                        worst_gram = residual;
                        worst_gram_row = k;
                    }
                    if !residual.is_finite() || residual > 2e-10 {
                        *gram_failures.entry(sol.rung).or_default() += 1;
                        *gram_failure_signatures
                            .entry([
                                ndist([r[0], r[1], r[2]]),
                                ndist([r[3], r[4], r[5]]),
                                ndist([r[6], r[7], r[8]]),
                            ])
                            .or_default() += 1;
                    }
                }
            } else if std::env::var_os("DUMP_UNSOLVED").is_some() {
                println!(
                    "UNSOLVED {k} C {:?} G {:?} T {:?}",
                    &r[0..3],
                    &r[3..6],
                    &r[6..9]
                );
            }
            n += 1;
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / n.max(1) as f64;
        if std::env::var_os("GULPS_FUNNEL").is_some() {
            let f = can_sandwich::funnel::snapshot();
            let att = f[0].max(1);
            println!("FUNNEL interior/reanchor chart attempts: {}", f[0]);
            let solved = f[0] as i64 - (f[1] + f[2] + f[3] + f[4] + f[5] + f[6]) as i64;
            println!(
                "  {:<38} {:>10}  {:>7.3}%",
                "SOLVED",
                solved,
                100.0 * solved as f64 / att as f64
            );
            for i in 1..can_sandwich::funnel::N {
                println!(
                    "  {:<38} {:>10}  {:>7.3}%",
                    can_sandwich::funnel::NAMES[i],
                    f[i],
                    100.0 * f[i] as f64 / att as f64
                );
            }
        }
        if std::env::var_os("DUMP_RUNGS").is_some() {
            for &(ns, k, rung) in &timings {
                println!("RUNG {k} {rung:?} {ns}");
            }
        }
        timings.sort_unstable_by_key(|x| x.0);
        let quantile = |q: f64| {
            let i = ((timings.len().saturating_sub(1)) as f64 * q).round() as usize;
            timings[i.min(timings.len().saturating_sub(1))]
        };
        let p50 = quantile(0.5);
        let p99 = quantile(0.99);
        let p999 = quantile(0.999);
        let worst = *timings.last().unwrap();
        println!(
            "dataset {} rows {} stride {} sampled {}",
            args[p + 1],
            triples.len(),
            stride,
            n
        );
        println!(
            "OVERALL: {solved}/{n} = {:.4}% solved; {:.2} us/triple; worst residual {:.3e}; \
             worst |det(O)-1| {:.3e}; machine-precise {machine}/{solved} = {:.4}%",
            100.0 * solved as f64 / n.max(1) as f64,
            us,
            worst_residual,
            worst_det,
            100.0 * machine as f64 / solved.max(1) as f64,
        );
        println!("by rung: {by_rung:?}");
        if audit_gram {
            println!(
                "Gram audit: worst {:.3e} at row {}; failures above 2e-10 {:?}",
                worst_gram, worst_gram_row, gram_failures
            );
            println!("Gram failure signatures: {gram_failure_signatures:?}");
        }
        if std::env::var_os("GULPS_SIGNATURES").is_some() {
            println!("by spectral signature: {by_signature:?}");
        }
        let mut per_rung: BTreeMap<can_sandwich::Rung, (u128, usize, u128, usize)> =
            BTreeMap::new();
        for &(ns, row, rung) in &timings {
            let e = per_rung.entry(rung).or_default();
            e.0 += ns;
            e.1 += 1;
            if ns > e.2 {
                e.2 = ns;
                e.3 = row;
            }
        }
        for (rung, (ns, cnt, max_ns, max_row)) in &per_rung {
            println!(
                "  rung {rung:?}: n {cnt}, total {:.1} ms, mean {:.2} us, max {:.2} us at row {max_row}",
                *ns as f64 / 1e6,
                *ns as f64 / 1e3 / *cnt as f64,
                *max_ns as f64 / 1e3,
            );
        }
        println!(
            "latency us: p50 {:.2}; p99 {:.2}; p99.9 {:.2}; max {:.2} at row {} rung {:?}",
            p50.0 as f64 / 1e3,
            p99.0 as f64 / 1e3,
            p999.0 as f64 / 1e3,
            worst.0 as f64 / 1e3,
            worst.1,
            worst.2,
        );
        can_sandwich::prof::dump();
        return;
    }

    // `dump-o <path> <row>`: solve one row and print the frame O (row-major) +
    // the row's Weyl eigenphases, for offline fixture construction.
    if let Some(p) = args.iter().position(|a| a == "dump-o") {
        let triples = data::read_triples(&args[p + 1]);
        let k: usize = args[p + 2].parse().unwrap();
        let r = triples[k];
        let sol = can_sandwich::solve([r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]);
        println!("row {k} rung {:?} residual {:.3e}", sol.rung, sol.residual);
        for i in 0..4 {
            for j in 0..4 {
                print!("{:+.17e} ", sol.o[(i, j)].re);
            }
            println!();
        }
        for (nm, m) in [
            ("eb", [r[0], r[1], r[2]]),
            ("ep", [r[3], r[4], r[5]]),
            ("et", [r[6], r[7], r[8]]),
        ] {
            let e = can_sandwich::eigphases(can_sandwich::weyl_from_monodromy(m));
            println!(
                "{nm} {:+.17e} {:+.17e} {:+.17e} {:+.17e}",
                e[0], e[1], e[2], e[3]
            );
        }
        return;
    }
    // `solve-npy <path>` mode: run solve() on every triple of a (N,3,3) npy, print its rung per line.
    if let Some(p) = args.iter().position(|a| a == "solve-npy") {
        let triples = data::read_triples(&args[p + 1]);
        for r in &triples {
            let sol =
                can_sandwich::solve([r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]);
            println!("{:?}", sol.rung);
        }
        return;
    }
    let dir = std::env::args()
        .nth(1)
        .filter(|a| a != "chart-table")
        .unwrap_or_else(|| SEG.to_string());
    let triples = data::read_triples(&format!("{dir}/feasible_linspace.npy"));
    let labels = data::read_labels(&format!("{dir}/linspace_strata.npy"));
    assert_eq!(triples.len(), labels.len());
    println!("loaded {} triples (monodromy [C,G,T])", triples.len());

    // reader sanity: the label distribution must match numpy's.
    let mut dist: BTreeMap<i8, usize> = BTreeMap::new();
    for &l in &labels {
        *dist.entry(l).or_default() += 1;
    }
    println!("stratum-label distribution (-1 sliver .. 3 vertex): {dist:?}");

    // Run a stride sample through solve(), reporting coverage by (dataset label, rung).
    let stride = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40usize);
    let mut total: BTreeMap<i8, usize> = BTreeMap::new();
    let mut by_rung: BTreeMap<(i8, can_sandwich::Rung), usize> = BTreeMap::new();
    let mut nanos: BTreeMap<i8, u128> = BTreeMap::new();
    let mut worst: BTreeMap<i8, f64> = BTreeMap::new(); // worst residual among SOLVED, per label
    let (mut machine, mut tail) = (0usize, 0usize); // solves at <1e-12 vs the e-9 tail
    let mut worst_det = 0.0f64; // worst |det(O)−1| among accepted frames (SO(4) check)
    let mut sliver_nd: BTreeMap<usize, (usize, usize)> = BTreeMap::new(); // ndist -> (solved, total) for label -1
    let t0 = std::time::Instant::now();
    let mut n = 0;
    for k in (0..triples.len()).step_by(stride) {
        let r = &triples[k];
        let c = [r[0], r[1], r[2]];
        let g = [r[3], r[4], r[5]];
        let t = [r[6], r[7], r[8]];
        let ts = std::time::Instant::now();
        let sol = can_sandwich::solve(c, g, t);
        *nanos.entry(labels[k]).or_default() += ts.elapsed().as_nanos();
        *total.entry(labels[k]).or_default() += 1;
        *by_rung.entry((labels[k], sol.rung)).or_default() += 1;
        if sol.rung == can_sandwich::Rung::Unsolved && std::env::var_os("DUMP_UNSOLVED").is_some() {
            println!(
                "UNSOLVED {k} label {} C {:?} G {:?} T {:?}",
                labels[k], c, g, t
            );
        }
        if labels[k] == -1 {
            let e = sliver_nd.entry(ndist(t)).or_default();
            e.1 += 1;
            if sol.rung != can_sandwich::Rung::Unsolved {
                e.0 += 1;
            }
        }
        if sol.rung != can_sandwich::Rung::Unsolved {
            let w = worst.entry(labels[k]).or_insert(0.0);
            *w = w.max(sol.residual);
            // Every accepted frame must be genuine SO(4): det(O)=+1 (the construction is real).
            let det = sol.o.map(|z| z.re).determinant();
            worst_det = worst_det.max((det - 1.0).abs());
            if sol.residual < 1e-12 {
                machine += 1;
            } else {
                tail += 1;
            }
        }
        n += 1;
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    let solved_all: usize = by_rung
        .iter()
        .filter(|((_, rr), _)| *rr != can_sandwich::Rung::Unsolved)
        .map(|(_, &v)| v)
        .sum();
    println!(
        "\nOVERALL: {solved_all}/{n} = {:.2}% solved; worst residual = {:.1e}; worst |det(O)−1| = {:.1e}; \
         machine-precise (<1e-12): {}/{} = {:.2}%; e-9 tail: {}",
        100.0 * solved_all as f64 / n as f64,
        worst.values().cloned().fold(0.0, f64::max),
        worst_det,
        machine,
        machine + tail,
        100.0 * machine as f64 / (machine + tail).max(1) as f64,
        tail,
    );
    println!("\nsliver (label -1) coverage by target ndist (degeneracy):");
    for (nd, (s, t)) in &sliver_nd {
        println!(
            "  ndist={nd}: {s:>4}/{t:<4} = {:>3.0}%  ({})",
            100.0 * *s as f64 / *t as f64,
            match nd {
                2 => "Y=0 face / deg-2",
                3 => "interior fold / deg-8 tangent",
                4 => "generic spectrum / crossing?",
                _ => "",
            }
        );
    }
    println!("\nsolve() over {n} triples (stride {stride}, {us:.1} us/triple):");
    println!("  label    n   solved  us/triple  by rung");
    for (&l, &tot) in &total {
        let solved: usize = by_rung
            .iter()
            .filter(|((ll, rr), _)| *ll == l && *rr != can_sandwich::Rung::Unsolved)
            .map(|(_, &v)| v)
            .sum();
        let rungs: Vec<String> = by_rung
            .iter()
            .filter(|((ll, rr), _)| *ll == l && *rr != can_sandwich::Rung::Unsolved)
            .map(|((_, rr), &v)| format!("{rr:?}:{v}"))
            .collect();
        let us_l = nanos[&l] as f64 / 1e3 / tot as f64;
        println!(
            "  {l:>4} {tot:>6} {:>6} ({:>3.0}%) {us_l:>9.2}  {}",
            solved,
            100.0 * solved as f64 / tot as f64,
            rungs.join(" ")
        );
    }
}
