//! Depth-two two-qubit realization in binary64 arithmetic.
//!
//! Both public functions prepare the same problem, search for a frame, and
//! verify it against the input spectrum. See README.md for coordinate conventions.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]
use nalgebra::{Complex, Matrix4};
type C = Complex<f64>;
type ComplexMatrix = Matrix4<C>;
mod algebraic;
mod diagnostics;
mod numerical;
mod problem;
mod spectral;
use algebraic::compiler_solution;
#[cfg(not(feature = "diagnostics"))]
use algebraic::{Rung, Solution};
#[cfg(feature = "diagnostics")]
pub use algebraic::{Rung, Solution};
#[cfg(not(feature = "diagnostics"))]
use diagnostics::prof;
#[cfg(feature = "diagnostics")]
pub use diagnostics::prof;
use problem::Problem;
#[cfg(feature = "diagnostics")]
pub type Mat4 = ComplexMatrix;
#[cfg(feature = "diagnostics")]
pub use diagnostics::{
    branch_signature, certify_frame, endpoint_gauge_residual, endpoint_right_gauge,
    factor_through_berkeley, factorized_gate_collapse_residual, factorized_waypoint_mass_residual,
    init_tables, ordered_chart_solutions, paired_edge_scope, solve_charts_only,
    solve_factorized_waypoint, solve_factorized_waypoint_direct, solve_paired_edges,
    solve_paired_edges_forward, solve_via_fixed_berkeley, solve_via_fixed_factor,
};

/// Return a verified real SO(4) frame, or decline.
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Matrix4<f64>> {
    solve_using(c, g, t, |_, solution, _| Some(solution.o.map(|z| z.re)))
}

/// Return `(O, L, R, phase)` with all three matrices real SO(4), where
/// `D(c) O D(g) = exp(i phase) L D(t) R` in the magic basis.
/// Reuses the eigenbasis from verification; no second diagonalization.
#[allow(clippy::type_complexity)]
pub fn solve_with_factors(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
) -> Option<(Matrix4<f64>, Matrix4<f64>, Matrix4<f64>, f64)> {
    solve_using(c, g, t, |problem, solution, state| {
        state.factors(problem, t, solution.o.map(|z| z.re))
    })
}

fn solve_using<T>(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    finish: impl FnOnce(&Problem, Solution, spectral::State) -> Option<T>,
) -> Option<T> {
    if c.iter().chain(&g).chain(&t).any(|x| !x.is_finite()) {
        return None;
    }
    let tp = prof::start();
    let problem = Problem::new(c, g, t);
    prof::rec(prof::SEG_PREPARE, tp);
    let result = algebraic::solve(&problem);
    if let Some(solution) = result
        && let Some(state) = solution.verified
    {
        return finish(&problem, solution, state);
    }
    let certify = |o: Matrix4<f64>| {
        let solution =
            compiler_solution(&problem, o.map(|v| C::new(v, 0.0)), Rung::Numerical, 0.0)?;
        let state = solution.verified?;
        Some((solution, state))
    };
    let (solution, state) = numerical::solve(&problem, c, g, t).and_then(certify)?;
    finish(&problem, solution, state)
}
