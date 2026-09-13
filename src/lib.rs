//! Exact, non-iterative depth-two two-qubit realization.
//!
//! The public boundary is intentionally small: [`solve`] accepts
//! one depth-two triple in monodromy coordinates and returns a
//! forward-certified real frame.
//! Diagnostic binaries live beside this library, but the realization engine is
//! compiled and tested only once.

// Index loops mirror the matrix formulas they implement, and negated float
// comparisons are deliberate NaN guards.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

mod cpoly;

mod cascade;

#[cfg(feature = "diagnostics")]
pub use cascade::solve_charts_only;
pub use cascade::{
    branch_signature, certify_frame, endpoint_gauge_residual, endpoint_right_gauge,
    factor_through_berkeley, factorized_gate_collapse_residual, factorized_waypoint_mass_residual,
    ordered_chart_solutions, solve, solve_factorized_waypoint, solve_factorized_waypoint_direct,
    solve_via_fixed_berkeley, solve_via_fixed_factor, Mat4, Rung, Solution,
};
#[cfg(feature = "diagnostics")]
pub use cascade::{init_tables, prof};
#[cfg(feature = "diagnostics")]
pub use cascade::{paired_edge_scope, solve_paired_edges, solve_paired_edges_forward};

mod radical;
