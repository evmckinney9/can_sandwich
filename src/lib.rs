//! Depth-two two-qubit realization in binary64 arithmetic.
//!
//! Given left, right, and target canonical gates, construct the local gate
//! between the first two that produces the target nonlocal action.
//!
//! Inputs are dimensionless monodromy triples. For `m = [m0, m1, m2]`, the
//! diagonal gate in the magic basis is
//! `D(m) = diag(exp(iπm1), exp(iπm0), exp(-iπ(m0+m1+m2)), exp(iπm2))`.
//! See the [researcher guide](https://github.com/evmckinney9/can_sandwich/blob/main/docs/researcher.md)
//! for the mathematical contract and how to compare a candidate algorithm.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]
use nalgebra::{Complex, Matrix4};
type C = Complex<f64>;
type ComplexMatrix = Matrix4<C>;
mod algebraic;
#[cfg(any(test, feature = "corpus"))]
pub mod corpus;
mod diagnostics;
mod numerical;
mod problem;
mod spectral;
#[cfg(not(feature = "diagnostics"))]
use algebraic::{Rung, Solution};
#[cfg(feature = "diagnostics")]
pub use algebraic::{Rung, Solution};
#[cfg(not(feature = "diagnostics"))]
use diagnostics::prof;
#[cfg(feature = "diagnostics")]
pub use diagnostics::prof;
use problem::Problem;
/// Largest root error, including the eigenbasis residual, of an accepted witness.
pub use spectral::SPECTRAL_TOLERANCE;
#[cfg(feature = "diagnostics")]
pub type Mat4 = ComplexMatrix;
#[cfg(feature = "diagnostics")]
pub use diagnostics::{branch_signature, certify_frame, solve_report};

/// The crate version, for bug reports.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Construct a real SO(4) matrix `O` for left gate `c`, right gate `g`, and target `t`.
///
/// The spectrum of `D(c)² O D(g)² Oᵀ` must match that of `s D(t)²` for one
/// global sign `s = ±1`, within numerical tolerance. The crate documentation
/// defines `D` and the input coordinates.
///
/// Returns `None` for nonfinite inputs or when the bounded search finds no
/// accepted witness. A decline does not establish that the input is infeasible.
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Matrix4<f64>> {
    witness(c, g, t).map(|(_, solution)| solution.o)
}

/// Why [`solve_with_factors`] returned no result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclineKind {
    /// The input is nonfinite, or the bounded search found no frame that
    /// passes spectral verification.
    NoWitness,
    /// A verified frame was found, but `L` and `R` failed their checks.
    Reconstruction,
}

/// A [`solve_with_factors`] decline with the inputs that produced it.
///
/// `Display` renders a bug report: the reason, the inputs, and the crate
/// version, followed by the issue URL. Debug formatting preserves the finite
/// input values so they can be copied into a reproducer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decline {
    /// Why the solver declined.
    pub kind: DeclineKind,
    /// Left gate monodromy coordinates.
    pub c: [f64; 3],
    /// Right gate monodromy coordinates.
    pub g: [f64; 3],
    /// Target monodromy coordinates.
    pub t: [f64; 3],
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self.kind {
            DeclineKind::NoWitness => "no witness",
            DeclineKind::Reconstruction => "endpoint reconstruction failed",
        };
        write!(
            f,
            "can_sandwich {VERSION} declined ({reason}) for c={:?}, g={:?}, t={:?}. \
             Please report this message at https://github.com/evmckinney9/can_sandwich/issues/1",
            self.c, self.g, self.t
        )
    }
}

impl std::error::Error for Decline {}

/// Return `(O, L, R, phase)` with all three matrices real SO(4), where
/// `D(c) O D(g) = exp(i phase) L D(t) R` in the magic basis.
///
/// Inputs follow the same convention as [`solve`], and `phase` is in radians.
/// The result is verified numerically, rather than certified in exact
/// arithmetic. A [`Decline`] does not establish that the input is infeasible.
#[allow(clippy::type_complexity)]
pub fn solve_with_factors(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
) -> Result<(Matrix4<f64>, Matrix4<f64>, Matrix4<f64>, f64), Decline> {
    let decline = |kind| Decline { kind, c, g, t };
    let (problem, solution) = witness(c, g, t).ok_or_else(|| decline(DeclineKind::NoWitness))?;
    solution
        .state
        .factors(&problem, t, solution.o)
        .ok_or_else(|| decline(DeclineKind::Reconstruction))
}

fn witness(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<(Problem, Solution)> {
    if c.iter().chain(&g).chain(&t).any(|x| !x.is_finite()) {
        return None;
    }
    let tp = prof::start();
    let problem = Problem::new(c, g, t);
    prof::rec(prof::SEG_PREPARE, tp);
    let mut solution = algebraic::solve(&problem).or_else(|| {
        let o = numerical::solve(&problem, c, g, t)?;
        let state = spectral::verify(&problem, &o)?;
        Some(Solution {
            o,
            rung: Rung::Numerical,
            residual: state.error,
            state,
        })
    })?;
    // Polish when the verified bound exceeds the refinement target. Measure
    // both frames with the joint eigenbasis and require improvement beyond
    // both of its residuals. A large residual can reflect orthogonality
    // drift, which refinement removes; then the error bound must halve.
    if solution.state.error > numerical::ROOT_TOLERANCE {
        let current = problem.joint(&solution.o, solution.state, 0.0);
        let lower_error = current.root_error - 4.0 * current.off_diagonal_error;
        if current.root_error > numerical::ROOT_TOLERANCE {
            let o = problem.refine(solution.o);
            if let Some(state) = spectral::verify(&problem, &o)
                && problem.joint(&o, state, 0.0).error < lower_error.max(0.5 * current.error)
            {
                solution.o = o;
                solution.residual = state.error;
                solution.state = state;
            }
        }
    }
    // Targets on a Horn wall, where a routed root equals a target root, can
    // stall above the refinement target in every free frame. Retaining that
    // root exactly while refining the complementary SO(3) block reaches it.
    // Routed roots within the multiplicity threshold (`1e-11`) qualify.
    if solution.state.error > numerical::ROOT_TOLERANCE {
        let current = problem.joint(&solution.o, solution.state, 0.0);
        if current.root_error > numerical::ROOT_TOLERANCE
            && let Some(o) = problem.split(
                numerical::seed(c, g, t),
                1e-11,
                numerical::ROOT_TOLERANCE,
                Some(2),
            )
            && let Some(state) = spectral::verify(&problem, &o)
            && problem.joint(&o, state, 0.0).error
                < (current.root_error - 4.0 * current.off_diagonal_error).max(0.5 * current.error)
        {
            solution.o = o;
            solution.residual = state.error;
            solution.state = state;
        }
    }
    Some((problem, solution))
}
