//! Dump production solutions for given feasible_linspace rows: the solved
//! O matrix (row-major), rung, and residual. Consumed by the Python
//! boundary-construction pipeline as certified R-member entry data.
mod data;

use can_sandwich::can_sandwich;

const SEG: &str = "/home/evm9/gulps/.claude/scripts/diagnostics/segments";

fn main() {
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "feasible_linspace".into());
    let triples = data::read_triples(&format!("{SEG}/{dataset}.npy"));
    for arg in std::env::args().skip(1) {
        let idx: usize = arg.parse().expect("row index");
        let r = triples[idx];
        let c = [r[0], r[1], r[2]];
        let g = [r[3], r[4], r[5]];
        let t = [r[6], r[7], r[8]];
        let t0 = std::time::Instant::now();
        let sol = if std::env::var_os("THREE_GIVENS_ONLY").is_some() {
            can_sandwich::solve_three_givens_only(c, g, t)
        } else if std::env::var_os("AXIS_ONLY").is_some() {
            can_sandwich::solve_axis_only(c, g, t)
        } else {
            can_sandwich::solve(c, g, t)
        };
        let us = t0.elapsed().as_micros();
        println!(
            "row {idx} rung {:?} residual {:.3e} time_us {us}",
            sol.rung, sol.residual
        );
        let o = sol.o;
        let mut flat = String::new();
        for i in 0..4 {
            for j in 0..4 {
                flat.push_str(&format!("{:.17e} ", o[(i, j)]));
            }
        }
        println!("O {flat}");
    }
}
