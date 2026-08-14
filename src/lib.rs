//! Exact, non-iterative depth-two two-qubit realization.
//!
//! The public boundary is intentionally small: [`solve`] accepts
//! one depth-two triple in monodromy coordinates and returns a
//! forward-certified real frame.
//! Diagnostic binaries live beside this library, but the realization engine is
//! compiled and tested only once.

#[cfg(feature = "diagnostics")]
pub mod can_sandwich;
#[cfg(not(feature = "diagnostics"))]
mod can_sandwich;

#[cfg(feature = "research-spin")]
pub use can_sandwich::{replay_spin_action, solve_spin_dyadic, solve_spin_spread};
pub use can_sandwich::{solve, Mat4, Rung, Solution};

// The confluent Schubert/radical constructor is an implementation detail used
// by `can_sandwich::secular`; it is not part of the black-box API.
mod sandwich;
