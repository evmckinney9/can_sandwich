//! Bounded Levenberg–Marquardt refinement and deterministic restarts.
use crate::{
    C as Z,
    problem::{PERMS24, PLANES, Problem},
};
use nalgebra::{Matrix4, SMatrix, SVector};
use std::f64::consts::PI;
type R4 = Matrix4<f64>;
const ACCEPT: f64 = 4e-9;
fn rotate(o: &mut R4, p: usize, q: usize, angle: f64) {
    let (s, c) = angle.sin_cos();
    for j in 0..4 {
        let x = o[(p, j)];
        let y = o[(q, j)];
        o[(p, j)] = c * x + s * y;
        o[(q, j)] = -s * x + c * y;
    }
}
impl Problem {
    fn early(&self) -> Option<R4> {
        let trace: Z = self.target_roots[0].iter().sum();
        for perm in *PERMS24 {
            let roots = std::array::from_fn(|i| self.left[i] * self.right[perm[i]]);
            let sum: Z = roots.iter().sum();
            let mut o = R4::zeros();
            for i in 0..4 {
                o[(i, perm[i])] = 1.0;
            }
            if o.determinant() < 0.0 {
                for i in 0..4 {
                    o[(i, 0)] *= -1.0;
                }
            }
            if ((sum - trace).norm() < 1e-7 || (sum + trace).norm() < 1e-7)
                && self.match_roots(roots) < ACCEPT
            {
                return Some(o);
            }
            for (p, q) in PLANES {
                let slope =
                    (self.left[p] - self.left[q]) * (self.right[perm[q]] - self.right[perm[p]]);
                if slope.norm_sqr() < 1e-28 {
                    continue;
                }
                for sign in [-1.0, 1.0] {
                    let delta = trace * sign - sum;
                    let u = (delta * slope.conj()).re / slope.norm_sqr();
                    if !(-1e-12..=1.0 + 1e-12).contains(&u) || (delta - slope * u).norm() > 1e-9 {
                        continue;
                    }
                    let mut candidate = o;
                    rotate(&mut candidate, p, q, u.clamp(0.0, 1.0).sqrt().asin());
                    let state = self.state(&candidate, 0.0);
                    if state.cost.is_finite() && state.error < ACCEPT {
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }
    fn iterate(&self, mut o: R4, branch: f64) -> Option<R4> {
        let ratios: [Z; 6] = PLANES.map(|(p, q)| self.dc[(p, p)] * self.dc[(q, q)].conj());
        let mut state = self.state(&o, branch);
        let mut damping = 1e-3;
        let mut stalled = 0;
        for _ in 0..40 {
            if state.cost.is_finite() && state.error < ACCEPT {
                return Some(o);
            }
            if !state.cost.is_finite() || state.off_diagonal_error > 1e-10 {
                break;
            }
            // Variable-projection GN: eliminate target eigenbasis rotations
            // pairwise while retaining splitting curvature at repeated targets.
            let mut h = SMatrix::<f64, 6, 6>::zeros();
            let mut gradient = SVector::<f64, 6>::zeros();
            for i in 0..4 {
                for l in i..4 {
                    let derivative: [Z; 6] = std::array::from_fn(|k| {
                        let (p, q) = PLANES[k];
                        let ratio = ratios[k];
                        (state.roots[l] * ratio - state.roots[i] * ratio.conj())
                            * state.eigenvectors[(p, i)]
                            * state.eigenvectors[(q, l)]
                            + (state.roots[i] * ratio - state.roots[l] * ratio.conj())
                                * state.eigenvectors[(q, i)]
                                * state.eigenvectors[(p, l)]
                    });
                    let weight = if i == l { 1.0 } else { 2.0 };
                    let direction = state.target[l] - state.target[i];
                    let regularization = state.cost.max(1e-28);
                    let denominator = direction.norm_sqr() + regularization;
                    let alpha: [f64; 6] = std::array::from_fn(|k| {
                        if i == l {
                            0.0
                        } else {
                            (derivative[k].conj() * direction).re / denominator
                        }
                    });
                    let projected: [Z; 6] =
                        std::array::from_fn(|k| derivative[k] - alpha[k] * direction);
                    for k in 0..6 {
                        if i == l {
                            gradient[k] +=
                                (derivative[k].conj() * (state.roots[i] - state.target[i])).re;
                        }
                        for n in 0..=k {
                            let value = (projected[k].conj() * projected[n]).re
                                + regularization * alpha[k] * alpha[n];
                            h[(k, n)] += weight * value;
                            if k != n {
                                h[(n, k)] += weight * value;
                            }
                        }
                    }
                }
            }
            let scale = h.diagonal().amax().max(1e-24);
            let mut accepted = false;
            for _ in 0..12 {
                let mut regularized = h;
                for k in 0..6 {
                    regularized[(k, k)] += damping * h[(k, k)].max(scale * 1e-14).max(1e-30);
                }
                let Some(chol) = regularized.cholesky() else {
                    damping *= 10.0;
                    continue;
                };
                let mut step = -chol.solve(&gradient);
                let norm = step.norm();
                if norm > 0.8 {
                    step *= 0.8 / norm;
                }
                let mut candidate = o;
                for k in 0..6 {
                    let (p, q) = PLANES[k];
                    rotate(&mut candidate, p, q, step[k]);
                }
                let next = self.state(&candidate, branch);
                if next.cost < state.cost {
                    let gain = state.cost - next.cost;
                    if gain < state.cost * 1e-8 {
                        stalled += 1;
                    } else {
                        stalled = 0;
                    }
                    o = candidate;
                    state = next;
                    damping = (damping * 0.25).max(1e-14);
                    accepted = true;
                    break;
                } else {
                    damping = (damping * 8.0).min(1e14);
                }
            }
            if !accepted || stalled > 8 {
                break;
            }
        }
        if state.cost.is_finite() && state.error < ACCEPT {
            Some(o)
        } else {
            None
        }
    }
}
fn random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}
fn random_starts(problem: &Problem, seed: &mut u64, branches: &[f64]) -> Option<R4> {
    // Try another orientation after four starts instead of exhausting one
    // parameterization. Each start has its own bounded refinement budget.
    for attempt in 0..4 {
        let mut o = R4::identity();
        for (p, q) in PLANES {
            rotate(&mut o, p, q, (2.0 * random(seed) - 1.0) * PI);
        }
        if let Some(solution) = problem.iterate(o, branches[attempt % branches.len()]) {
            return Some(solution);
        }
    }
    None
}

fn solve_primary(problem: &Problem, c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<R4> {
    if let Some(o) = problem.early() {
        return Some(o);
    }
    let mut seed = 0x123456789abcdefu64;
    for value in c.into_iter().chain(g).chain(t) {
        seed = seed.rotate_left(7) ^ value.to_bits();
    }
    if let Some(o) = random_starts(problem, &mut seed, &[0.0]) {
        return Some(o);
    }
    let swapped = Problem::new(g, c, t);
    swapped
        .early()
        .or_else(|| random_starts(&swapped, &mut seed, &[0.0]))
        .map(|o| o.transpose())
        .or_else(|| random_starts(problem, &mut seed, &[1.0, -1.0]))
}

/// Reorder the real eigenbasis of A^-1/2 W C W^T A^-1/2 to B's spectrum.
/// This maps an inverse-factor problem back to the original problem.
fn inverse_factor(original: &Problem, c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<R4> {
    let inverse_c = c.map(|x| -x);
    let dual = Problem::new(inverse_c, t, g);
    let frame = solve_primary(&dual, inverse_c, t, g)?;
    let state = dual.state(&frame, 0.0);
    let o = state.ordered_basis();
    let check = original.state(&o, 0.0);
    if check.cost.is_finite() && check.error < ACCEPT {
        Some(o)
    } else {
        original.iterate(o, 0.0)
    }
}

fn near_commuting(problem: &Problem) -> Option<R4> {
    let mut starts = PERMS24.map(|perm| {
        let roots = std::array::from_fn(|i| problem.left[i] * problem.right[perm[i]]);
        (problem.match_roots(roots), perm)
    });
    starts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut seed = 0x987654321abcdefu64;
    for (error, perm) in starts.into_iter().take(4) {
        // At a commuting frame the first spectral derivative vanishes;
        // a spectral displacement of size e can require rotations of sqrt(e).
        let scale = error.sqrt().clamp(1e-5, 0.3);
        for multiplier in [0.1, 1.0, 10.0] {
            let mut o = R4::zeros();
            for i in 0..4 {
                o[(i, perm[i])] = 1.0;
            }
            if o.determinant() < 0.0 {
                o.column_mut(0).neg_mut();
            }
            for (p, q) in PLANES {
                rotate(
                    &mut o,
                    p,
                    q,
                    (2.0 * random(&mut seed) - 1.0) * scale * multiplier,
                );
            }
            if let Some(o) = problem.iterate(o, 0.0) {
                return Some(o);
            }
        }
    }
    None
}

pub(crate) fn solve(problem: &Problem, c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<R4> {
    solve_primary(problem, c, g, t)
        .or_else(|| near_commuting(problem))
        .or_else(|| inverse_factor(problem, c, g, t))
        .or_else(|| inverse_factor(&Problem::new(g, c, t), g, c, t).map(|o| o.transpose()))
}
