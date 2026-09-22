//! Non-iterative depth-two two-qubit realization in binary64 arithmetic.
//!
//! The public boundary is [`solve`]: one depth-two triple in monodromy
//! coordinates in, a forward-certified real frame out. Everything else is
//! research tooling behind the `diagnostics` feature.

// Index loops mirror the matrix formulas they implement, and negated float
// comparisons are deliberate NaN guards.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

mod cascade;
mod cpoly;
mod radical;

pub use cascade::{Mat4, Rung, Solution, solve};

#[cfg(feature = "diagnostics")]
pub use cascade::{
    branch_signature, certify_frame, endpoint_gauge_residual, endpoint_right_gauge,
    factor_through_berkeley, factorized_gate_collapse_residual, factorized_waypoint_mass_residual,
    init_tables, ordered_chart_solutions, paired_edge_scope, prof, solve_charts_only,
    solve_factorized_waypoint, solve_factorized_waypoint_direct, solve_paired_edges,
    solve_paired_edges_forward, solve_via_fixed_berkeley, solve_via_fixed_factor,
};
