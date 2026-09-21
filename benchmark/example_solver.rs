//! Minimal submission: solve identity inputs and decline everything else.
//!
//! Replace this function with your algorithm. Available dependencies are
//! std, nalgebra 0.35, and serde_json 1. Keep stdout free for the benchmark.

/// Return a real SO(4) frame, or None when no frame was found.
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<[[f64; 4]; 4]> {
    if [c, g, t].iter().flatten().all(|value| *value == 0.0) {
        Some(std::array::from_fn(|i| {
            std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })
        }))
    } else {
        None
    }
}
