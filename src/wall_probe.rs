//! Parity probe for the wall constructor.
//! Prototype parity run, not a benchmark claim.

mod data;
mod wall_rung;

use std::time::Instant;

/// Hard rows: all must hit at residual < 1e-8. Row 36581 (fold class) is
/// handled separately below as an EXPECTED MISS: its selector zero has even
/// multiplicity in every f64-sampled resultant (thrice-confirmed); it needs
/// double-double or exact isolation (future increment).
const PARITY_ROWS: &[usize] = &[
    83073, 88000, 236178, 257245, 268000, 274931, 289362, 336133, 455816, 635232,
];
const FOLD_ROW: usize = 36581;

const CORPUS: &str = "/home/evm9/gulps/.claude/scripts/diagnostics/segments/feasible_linspace.npy";

fn main() {
    println!("wall_probe: prototype parity run, not a benchmark claim");
    println!();

    let triples = data::read_triples(CORPUS);
    let n = triples.len();
    println!("corpus: {} rows", n);
    println!();

    let mut n_pass = 0usize;
    let mut n_fail = 0usize;

    let all_rows: Vec<(usize, bool)> = PARITY_ROWS
        .iter()
        .map(|&r| (r, true))
        .chain(std::iter::once((FOLD_ROW, false)))
        .collect();

    for (row, must_hit) in &all_rows {
        let row = *row;
        if row >= n {
            println!("row {:>7}: OUT OF RANGE (corpus has {} rows)", row, n);
            if *must_hit {
                n_fail += 1;
            }
            continue;
        }
        let triple = &triples[row];
        let t0 = Instant::now();

        // Try interior, then boundary, then DD fold path
        let result = wall_rung::run_row_interior(triple)
            .or_else(|| wall_rung::run_row_boundary(triple))
            .or_else(|| wall_rung::run_row_fold_dd(triple));

        let elapsed = t0.elapsed();

        match result {
            Some(hit) => {
                let verdict = if *must_hit { "PASS" } else { "unexpected HIT" };
                println!(
                    "row {:>7}: HIT  resid={:.3e}  wb={}  z={:.4}  {}  {:?}  {}",
                    row,
                    hit.residual,
                    hit.w_branch,
                    hit.z0,
                    if hit.is_boundary {
                        format!("boundary bentry={}", hit.bentry.unwrap_or(99))
                    } else {
                        "interior".to_string()
                    },
                    elapsed,
                    verdict,
                );
                if *must_hit {
                    n_pass += 1;
                }
            }
            None => {
                if *must_hit {
                    println!("row {:>7}: MISS (FAIL -- must hit)  {:?}", row, elapsed);
                    n_fail += 1;
                } else {
                    println!("row {:>7}: MISS (expected: fold class)  {:?}", row, elapsed);
                }
            }
        }
    }

    println!();
    println!(
        "parity: {}/{} required rows passed",
        n_pass,
        PARITY_ROWS.len()
    );
    if n_fail == 0 {
        println!("PARITY OK");
    } else {
        println!("PARITY FAIL ({} misses)", n_fail);
        std::process::exit(1);
    }
}
