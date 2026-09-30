//! Bounded Levenberg–Marquardt refinement and deterministic restarts.
use crate::{
    C as Z,
    problem::{PERMS24, PLANES, Problem},
    spectral::{SPECTRAL_TOLERANCE, State},
};
use nalgebra::{Matrix4, SMatrix, SVector, SymmetricEigen};
use std::f64::consts::PI;
type R4 = Matrix4<f64>;
/// Refinement target for root distances; the spectral verifier separately
/// bounds error from the computed eigenbasis.
pub(crate) const ROOT_TOLERANCE: f64 = 1e-14;

fn converged(state: &State, root_tolerance: f64) -> bool {
    state.cost.is_finite() && state.root_error < root_tolerance && state.error < SPECTRAL_TOLERANCE
}
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
                && self.match_roots(roots) < SPECTRAL_TOLERANCE
            {
                return Some(o);
            }
            for (p, q) in PLANES {
                for sign in [-1.0, 1.0] {
                    let targets = self.target_roots[0].map(|z| z * sign);
                    if (0..4)
                        .filter(|&i| i != p && i != q)
                        .any(|i| targets.iter().all(|z| (roots[i] - z).norm() > 1e-13))
                    {
                        continue;
                    }
                    for root in targets {
                        let Some(angle) = self.block_angle((p, q), (perm[p], perm[q]), root) else {
                            continue;
                        };
                        let mut candidate = o;
                        rotate(&mut candidate, p, q, angle);
                        let state = self.state(&candidate, 0.0);
                        if converged(&state, SPECTRAL_TOLERANCE) {
                            return Some(candidate);
                        }
                    }
                }
            }
        }
        None
    }
    pub(crate) fn iterate(&self, o: R4, branch: f64) -> Option<R4> {
        self.iterate_fixed(o, branch, None)
    }

    /// Keep useful progress even when refinement cannot reach its target.
    pub(crate) fn refine(&self, o: R4) -> R4 {
        let (first, plain) = self.polish(o, false);
        let state = self.joint(&first, plain, 0.0);
        if converged(&state, ROOT_TOLERANCE) {
            return first;
        }
        // Continue with the joint eigenbasis, whose smaller residual lets the
        // iteration resolve nearly repeated roots.
        let (second, next) = self.polish(first, true);
        if next.error < state.error {
            second
        } else {
            first
        }
    }

    fn polish(&self, o: R4, joint: bool) -> (R4, State) {
        let (o, state) = self.refine_fixed(o, 0.0, None, ROOT_TOLERANCE, joint);
        if converged(&state, ROOT_TOLERANCE) {
            return (o, state);
        }
        self.escape(o, state, joint)
    }

    /// Restart a stalled refinement along directions that leave every root
    /// fixed to first order. Vertex, edge, and face frames are critical points
    /// of the spectral map, where a root error `e` needs rotations of order
    /// `sqrt(e / sigma_max)`.
    fn escape(&self, o: R4, state: State, joint: bool) -> (R4, State) {
        let jacobian = SMatrix::<f64, 8, 6>::from_fn(|row, k| {
            let (p, q) = PLANES[k];
            let (i, v) = (row / 2, &state.eigenvectors);
            let ratio = self.dc[(p, p)] * self.dc[(q, q)].conj();
            let change = state.roots[i] * ratio - state.roots[i] * ratio.conj();
            let derivative = change * v[(p, i)] * v[(q, i)] + change * v[(q, i)] * v[(p, i)];
            if row % 2 == 0 {
                derivative.re
            } else {
                derivative.im
            }
        });
        let eigen = SymmetricEigen::new(jacobian.transpose() * jacobian);
        let largest = eigen.eigenvalues.max();
        let mut best = (o, state);
        if !state.cost.is_finite() {
            return best;
        }
        let step = (state.root_error / largest.sqrt()).sqrt().min(0.5);
        for k in 0..6 {
            if largest > 0.0 && eigen.eigenvalues[k] > 1e-4 * largest {
                continue;
            }
            for sign in [-1.0, 1.0] {
                let mut start = o;
                for (m, (p, q)) in PLANES.into_iter().enumerate() {
                    rotate(&mut start, p, q, sign * step * eigen.eigenvectors[(m, k)]);
                }
                let (candidate, next) = self.refine_fixed(start, 0.0, None, ROOT_TOLERANCE, joint);
                if next.error < best.1.error && next.root_error < best.1.root_error {
                    best = (candidate, next);
                }
                if converged(&best.1, ROOT_TOLERANCE) {
                    return best;
                }
            }
        }
        best
    }

    /// A routed root within `gate` of a target root can be retained exactly
    /// while solving the complementary SO(3) block. Free rotations would only
    /// preserve that boundary root to second order, which stalls refinement
    /// near repeated spectra. `closest` limits the search to the routes whose
    /// roots are nearest a target root.
    pub(crate) fn split(
        &self,
        mut seed: u64,
        gate: f64,
        tolerance: f64,
        closest: Option<usize>,
    ) -> Option<R4> {
        let mut routes = Vec::new();
        for fixed in 0..4 {
            for column in 0..4 {
                for sign in [-1.0, 1.0] {
                    let root = self.left[fixed] * self.right[column];
                    let distance = self.target_roots[0]
                        .iter()
                        .map(|z| (root - z * sign).norm())
                        .fold(f64::INFINITY, f64::min);
                    if distance <= gate {
                        routes.push((distance, fixed, column, sign));
                    }
                }
            }
        }
        if let Some(count) = closest {
            routes.sort_by(|a, b| a.0.total_cmp(&b.0));
            routes.truncate(count);
        }
        for &(distance, fixed, column, sign) in &routes {
            // The fixed root is exactly the routed root, so a route cannot
            // converge unless it already meets the root tolerance.
            if distance >= tolerance {
                continue;
            }
            for start in self.wall_frames(fixed, column, sign) {
                let (o, state) = self.refine_fixed(start, sign, Some(fixed), tolerance, false);
                if converged(&state, tolerance) {
                    return Some(o);
                }
            }
            for _ in 0..(if closest.is_some() { 2 } else { 8 }) {
                let mut start = R4::identity();
                start.swap_columns(fixed, column);
                if start.determinant() < 0.0 {
                    start.column_mut(0).neg_mut();
                }
                for (p, q) in PLANES {
                    if p != fixed && q != fixed {
                        rotate(&mut start, p, q, (2.0 * random(&mut seed) - 1.0) * PI);
                    }
                }
                let (o, state) = self.refine_fixed(start, sign, Some(fixed), tolerance, false);
                if converged(&state, tolerance) {
                    return Some(o);
                }
            }
        }
        None
    }

    /// Exact frames through a Horn wall, where the routed root
    /// `left[fixed] * right[column]` equals a target root. With row `fixed`
    /// of O equal to ±e_column the sandwich splits into that root and a 3×3
    /// sandwich with O₃ ∈ O(3). Every 2×2 minor of O₃ is ± its complementary
    /// entry, so with B = O₃∘O₃ the reduced coefficients are e₁ = aᵀBl,
    /// e₂ = e₃·conj(e₁) and e₃ = det: on the unit circle the single complex
    /// equation aᵀBl = w₁ + w₂ + w₃ fixes the spectrum. On the chart
    /// O₃ = G₁₂(θ)G₂₃(φ), which has one zero entry, aᵀBl is bilinear in
    /// (x, y) = (cos²θ, cos²φ), and y is real exactly at the roots of a
    /// quadratic in x. All nine placements of the zero are tried.
    fn wall_frames(&self, fixed: usize, column: usize, sign: f64) -> Vec<R4> {
        let rows: Vec<usize> = (0..4).filter(|&i| i != fixed).collect();
        let cols: Vec<usize> = (0..4).filter(|&j| j != column).collect();
        let routed = self.left[fixed] * self.right[column];
        let targets = self.target_roots[0].map(|z| z * sign);
        let distance = |k: usize| (targets[k] - routed).norm();
        let matched = (0..4)
            .min_by(|&i, &j| distance(i).total_cmp(&distance(j)))
            .unwrap_or(0);
        let e1: Z = (0..4).filter(|&k| k != matched).map(|k| targets[k]).sum();
        let mut frames = Vec::new();
        for zero_row in 0..3 {
            for zero_column in 0..3 {
                let mut r = [0usize; 3];
                let mut c = [0usize; 3];
                let (mut ri, mut ci) = (0, 1);
                for k in 0..3 {
                    if k != zero_row {
                        r[ri] = rows[k];
                        ri += 1;
                    }
                    if k != zero_column {
                        c[ci] = cols[k];
                        ci += 1;
                    }
                }
                r[2] = rows[zero_row];
                c[0] = cols[zero_column];
                let chart = |x: f64, y: f64| -> Z {
                    let b = [
                        [x, (1.0 - x) * y, (1.0 - x) * (1.0 - y)],
                        [1.0 - x, x * y, x * (1.0 - y)],
                        [0.0, 1.0 - y, y],
                    ];
                    let mut sum = Z::new(0.0, 0.0);
                    for i in 0..3 {
                        for j in 0..3 {
                            sum += self.left[r[i]] * self.right[c[j]] * b[i][j];
                        }
                    }
                    sum
                };
                let (f00, f10, f01, f11) = (
                    chart(0.0, 0.0),
                    chart(1.0, 0.0),
                    chart(0.0, 1.0),
                    chart(1.0, 1.0),
                );
                // e1 - chart(x, y) = P(x) - Q(x) y with P, Q affine in x.
                let (p0, p1) = (e1 - f00, f00 - f10);
                let (q0, q1) = (f01 - f00, f11 - f10 - f01 + f00);
                let cross = |u: Z, v: Z| (u * v.conj()).im;
                let quadratic = [cross(p1, q1), cross(p0, q1) + cross(p1, q0), cross(p0, q0)];
                for x in real_roots(quadratic) {
                    let q = q0 + q1 * x;
                    if !(-1e-12..=1.0 + 1e-12).contains(&x) || q.norm_sqr() == 0.0 {
                        continue;
                    }
                    let y = ((p0 + p1 * x) * q.conj()).re / q.norm_sqr();
                    if !(-1e-12..=1.0 + 1e-12).contains(&y) {
                        continue;
                    }
                    let (x, y) = (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
                    let (ct, st, cp, sp) = (x.sqrt(), (1.0 - x).sqrt(), y.sqrt(), (1.0 - y).sqrt());
                    let o3 = [
                        [ct, -st * cp, st * sp],
                        [st, ct * cp, -ct * sp],
                        [0.0, sp, cp],
                    ];
                    let mut o = R4::zeros();
                    o[(fixed, column)] = 1.0;
                    for i in 0..3 {
                        for j in 0..3 {
                            o[(r[i], c[j])] = o3[i][j];
                        }
                    }
                    if o.determinant() < 0.0 {
                        o[(fixed, column)] = -1.0;
                    }
                    frames.push(o);
                }
            }
        }
        frames
    }

    /// Starts next to a Horn wall. A routed root within `gate` of a target
    /// root puts the solution close to an exact wall frame of the nearby wall
    /// problem, in which that target root is replaced by the routed one; free
    /// refinement from those frames then leaves the wall.
    fn near_wall(&self, gate: f64) -> Option<R4> {
        let mut routes = Vec::new();
        for fixed in 0..4 {
            for column in 0..4 {
                for sign in [-1.0, 1.0] {
                    let root = self.left[fixed] * self.right[column];
                    let distance = self.target_roots[0]
                        .iter()
                        .map(|z| (root - z * sign).norm())
                        .fold(f64::INFINITY, f64::min);
                    if distance <= gate {
                        routes.push((distance, fixed, column, sign));
                    }
                }
            }
        }
        routes.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, fixed, column, sign) in &routes {
            for start in self.wall_frames(fixed, column, sign).into_iter().take(4) {
                if let Some(o) = self.iterate(start, sign) {
                    return Some(o);
                }
            }
        }
        None
    }

    /// Levenberg–Marquardt on the matched roots themselves. The refinement
    /// above fits a damped variable-projection model that stalls next to
    /// critical frames, where the solution lies a square-root distance away
    /// along a nearly flat direction. Here the residual is the vector of
    /// root errors indexed by target, the Jacobian is a forward difference
    /// at the square root of machine epsilon, and the damping is Marquardt's
    /// diagonal scaling with Nielsen's update. If that stalls, one step of
    /// 1e-6 along the flattest direction, on either side, leaves the
    /// critical point before the iteration restarts.
    pub(crate) fn newton_polish(&self, o: R4) -> R4 {
        let (sign, _) = self.state(&o, 0.0).residual_by_target();
        let Some((mut best, mut r, j)) = self.root_lm(o, sign) else {
            return o;
        };
        if r.amax() > 1e-14 {
            let svd = j.svd(false, true);
            let (Some(v_t), flattest) = (svd.v_t, svd.singular_values.imin()) else {
                return best;
            };
            let start = best;
            for side in [1.0, -1.0] {
                let mut shifted = start;
                for (k, (p, q)) in PLANES.into_iter().enumerate() {
                    rotate(&mut shifted, p, q, side * 1e-6 * v_t[(flattest, k)]);
                }
                if let Some((o, next, _)) = self.root_lm(orthogonalize(shifted), sign)
                    && next.amax() < r.amax()
                {
                    (best, r) = (o, next);
                    if r.amax() < 1e-14 {
                        break;
                    }
                }
            }
        }
        best
    }

    fn root_lm(&self, mut o: R4, sign: f64) -> Option<(R4, SVector<f64, 8>, SMatrix<f64, 8, 6>)> {
        let residual = |o: &R4| -> Option<SVector<f64, 8>> {
            let s = self.state(o, sign);
            let (branch, r) = s.residual_by_target();
            (branch == sign && s.cost.is_finite()).then(|| {
                SVector::<f64, 8>::from_fn(
                    |k, _| if k % 2 == 0 { r[k / 2].re } else { r[k / 2].im },
                )
            })
        };
        let h = f64::EPSILON.sqrt();
        let jacobian = |o: &R4, r: &SVector<f64, 8>| -> Option<SMatrix<f64, 8, 6>> {
            let mut j = SMatrix::<f64, 8, 6>::zeros();
            for (k, (p, q)) in PLANES.into_iter().enumerate() {
                let mut shifted = *o;
                rotate(&mut shifted, p, q, h);
                j.set_column(k, &((residual(&shifted)? - r) / h));
            }
            Some(j)
        };
        let mut r = residual(&o)?;
        let mut j = jacobian(&o, &r)?;
        let mut normal = j.transpose() * j;
        let mut damping = 1e-3 * normal.diagonal().max();
        let mut growth = 2.0;
        for _ in 0..40 {
            if r.amax() < 1e-15 {
                break;
            }
            let gradient = j.transpose() * r;
            let mut system = normal;
            for k in 0..6 {
                system[(k, k)] += damping * normal[(k, k)].max(1e-30);
            }
            let Some(step) = system.lu().solve(&(-gradient)) else {
                break;
            };
            let mut candidate = o;
            for (k, (p, q)) in PLANES.into_iter().enumerate() {
                rotate(&mut candidate, p, q, step[k]);
            }
            let candidate = orthogonalize(candidate);
            match residual(&candidate) {
                Some(next) if next.norm_squared() < r.norm_squared() => {
                    let Some(next_j) = jacobian(&candidate, &next) else {
                        break;
                    };
                    (o, r, j) = (candidate, next, next_j);
                    normal = j.transpose() * j;
                    damping /= 3.0;
                    growth = 2.0;
                }
                _ => {
                    damping *= growth;
                    growth *= 2.0;
                }
            }
        }
        Some((o, r, j))
    }

    fn measure(&self, o: &R4, branch: f64, joint: bool) -> State {
        let state = self.state(o, branch);
        if joint {
            self.joint(o, state, branch)
        } else {
            state
        }
    }

    fn iterate_fixed(&self, o: R4, branch: f64, fixed: Option<usize>) -> Option<R4> {
        let (o, state) = self.refine_fixed(o, branch, fixed, SPECTRAL_TOLERANCE, false);
        converged(&state, SPECTRAL_TOLERANCE).then_some(o)
    }

    fn refine_fixed(
        &self,
        mut o: R4,
        branch: f64,
        fixed: Option<usize>,
        root_tolerance: f64,
        joint: bool,
    ) -> (R4, State) {
        let ratios: [Z; 6] = PLANES.map(|(p, q)| self.dc[(p, p)] * self.dc[(q, q)].conj());
        o = orthogonalize(o);
        let mut state = self.measure(&o, branch, joint);
        let mut best = (o, state);
        let mut damping = 1e-3;
        let mut stalled = 0;
        for _ in 0..120 {
            if converged(&state, root_tolerance) {
                return (o, state);
            }
            if !state.cost.is_finite() || state.off_diagonal_error > 1e-10 {
                break;
            }
            // Variable-projection GN: eliminate target eigenbasis rotations
            // pairwise while retaining splitting curvature at repeated targets.
            let mut jacobian = SMatrix::<f64, 30, 6>::zeros();
            let mut residual = SVector::<f64, 30>::zeros();
            let mut row = 0;
            for i in 0..4 {
                for l in i..4 {
                    let derivative: [Z; 6] = std::array::from_fn(|k| {
                        let (p, q) = PLANES[k];
                        if fixed.is_some_and(|fixed| p == fixed || q == fixed) {
                            return Z::new(0.0, 0.0);
                        }
                        let ratio = ratios[k];
                        (state.roots[l] * ratio - state.roots[i] * ratio.conj())
                            * state.eigenvectors[(p, i)]
                            * state.eigenvectors[(q, l)]
                            + (state.roots[i] * ratio - state.roots[l] * ratio.conj())
                                * state.eigenvectors[(q, i)]
                                * state.eigenvectors[(p, l)]
                    });
                    let weight = if i == l {
                        1.0
                    } else {
                        std::f64::consts::SQRT_2
                    };
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
                        jacobian[(row, k)] = weight * projected[k].re;
                        jacobian[(row + 1, k)] = weight * projected[k].im;
                        jacobian[(row + 2, k)] = weight * regularization.sqrt() * alpha[k];
                    }
                    if i == l {
                        let error = state.roots[i] - state.target[i];
                        residual[row] = error.re;
                        residual[row + 1] = error.im;
                    }
                    row += 3;
                }
            }
            let norms: [f64; 6] = std::array::from_fn(|k| jacobian.column(k).norm_squared());
            let scale = norms.into_iter().fold(1e-24, f64::max);
            let mut accepted = false;
            for _ in 0..12 {
                // Solve the damped least-squares system directly. Forming JᵀJ
                // squares the condition number and loses directions that
                // distinguish nearly repeated roots.
                let mut augmented = SMatrix::<f64, 36, 6>::zeros();
                augmented.fixed_rows_mut::<30>(0).copy_from(&jacobian);
                for k in 0..6 {
                    augmented[(30 + k, k)] =
                        (damping * norms[k].max(scale * 1e-28).max(1e-30)).sqrt();
                }
                let qr = augmented.qr();
                let mut rhs = SVector::<f64, 36>::zeros();
                rhs.fixed_rows_mut::<30>(0).copy_from(&(-residual));
                qr.q_tr_mul(&mut rhs);
                let Some(mut step) = qr.r().solve_upper_triangular(&rhs.fixed_rows::<6>(0)) else {
                    damping *= 10.0;
                    continue;
                };
                // A frozen coordinate stays exactly zero, including roundoff
                // from the least-squares solve. Otherwise its boundary root drifts.
                for (k, (p, q)) in PLANES.into_iter().enumerate() {
                    if fixed.is_some_and(|fixed| p == fixed || q == fixed) {
                        step[k] = 0.0;
                    }
                }
                let norm = step.norm();
                if norm > 0.8 {
                    step *= 0.8 / norm;
                }
                let mut candidate = o;
                for k in 0..6 {
                    let (p, q) = PLANES[k];
                    rotate(&mut candidate, p, q, step[k]);
                }
                candidate = orthogonalize(candidate);
                let next = self.measure(&candidate, branch, joint);
                if next.cost < state.cost {
                    let gain = state.cost - next.cost;
                    if gain < state.cost * 1e-4 {
                        stalled += 1;
                    } else {
                        stalled = 0;
                    }
                    o = candidate;
                    state = next;
                    if state.error < best.1.error && state.root_error < best.1.root_error {
                        best = (o, state);
                    }
                    damping = (damping * 0.25).max(1e-30);
                    accepted = true;
                    break;
                } else {
                    damping = (damping * 8.0).min(1e14);
                }
            }
            if !accepted || stalled > 5 {
                break;
            }
        }
        if converged(&state, root_tolerance) {
            (o, state)
        } else {
            best
        }
    }
}

/// Real roots of `a x² + b x + c`, by the cancellation-free quadratic formula.
fn real_roots([a, b, c]: [f64; 3]) -> Vec<f64> {
    let scale = a.abs().max(b.abs()).max(c.abs());
    if scale == 0.0 {
        return vec![];
    }
    if a.abs() <= 1e-14 * scale {
        return if b == 0.0 { vec![] } else { vec![-c / b] };
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return vec![];
    }
    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
    if q == 0.0 {
        vec![0.0]
    } else {
        vec![q / a, c / q]
    }
}

/// One polar Newton step removes the rounding drift of successive rotations.
/// The input is already orthogonal to working precision.
pub(crate) fn orthogonalize(o: R4) -> R4 {
    o * ((R4::identity() * 3.0 - o.transpose() * o) * 0.5)
}

pub(crate) fn seed(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> u64 {
    let mut seed = 0x123456789abcdefu64;
    for value in c.into_iter().chain(g).chain(t) {
        seed = seed.rotate_left(7) ^ value.to_bits();
    }
    seed
}

fn random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}
fn random_starts(problem: &Problem, seed: &mut u64, branches: &[f64]) -> Option<R4> {
    // Each orientation and each refinement has a bounded search budget.
    for attempt in 0..24 {
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
    let mut seed = seed(c, g, t);
    if let Some(o) = problem.split(seed, 1e-14, SPECTRAL_TOLERANCE, None) {
        return Some(o);
    }
    if let Some(o) = problem.near_wall(1e-6) {
        return Some(o);
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
    if converged(&check, SPECTRAL_TOLERANCE) {
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
    // Rank-two Horn bounds for ordered alcove coordinates, on both central
    // target lifts. Do not spend restart budgets on a certified violation.
    let ordered = |m: [f64; 3]| {
        let last = -m[0] - m[1] - m[2];
        m[0] >= m[1] && m[1] >= m[2] && m[2] >= last && m[0] - last <= 1.0
    };
    let paired_sum = (c[0] + c[2]) + (g[0] + g[2]);
    if [c, g, t].into_iter().all(ordered)
        && (t[1] + t[2]).abs() > paired_sum.min(1.0 - paired_sum) + 1e-12
    {
        return None;
    }
    solve_primary(problem, c, g, t)
        .or_else(|| near_commuting(problem))
        .or_else(|| inverse_factor(problem, c, g, t))
        .or_else(|| inverse_factor(&Problem::new(g, c, t), g, c, t).map(|o| o.transpose()))
}

/// Nonnegative least squares for a real 4×6 system by support enumeration
/// (supports of at most three columns; the phase shifts sum to zero, so
/// the system has rank three). Returns the solution with least residual.
fn nonnegative_squares(w: &[[f64; 6]; 4], phi: &[f64; 4]) -> Option<[f64; 6]> {
    let mut best: Option<(f64, [f64; 6])> = None;
    for mask in 1u32..64 {
        let support: Vec<usize> = (0..6).filter(|&q| mask & (1 << q) != 0).collect();
        if support.len() > 3 {
            continue;
        }
        let n = support.len();
        let mut a = nalgebra::DMatrix::<f64>::zeros(4, n);
        for k in 0..4 {
            for (col, &q) in support.iter().enumerate() {
                a[(k, col)] = w[k][q];
            }
        }
        let b = nalgebra::DVector::<f64>::from_row_slice(phi);
        let Some(inverse) = (a.transpose() * &a).try_inverse() else {
            continue;
        };
        let y = inverse * a.transpose() * &b;
        if y.iter().any(|&v| !(v >= 0.0)) {
            continue;
        }
        let residual = (&a * &y - &b).norm();
        let mut x = [0.0; 6];
        for (col, &q) in support.iter().enumerate() {
            x[q] = y[col];
        }
        if best.as_ref().is_none_or(|(r, _)| residual < *r) {
            best = Some((residual, x));
        }
    }
    best.map(|(_, x)| x)
}

/// exp(B) for a small real skew matrix by its Taylor series.
fn exp_skew(b: &R4) -> R4 {
    let mut out = R4::identity();
    let mut term = R4::identity();
    for n in 1..30 {
        term = term * b / n as f64;
        out += term;
        if term.amax() < 1e-18 {
            break;
        }
    }
    out
}

impl Problem {
    /// Second-order vertex frames. At a permutation frame P the spectral
    /// map is critical: under O = exp(B) P root k of D_c^2 O D_g^2 O^T
    /// moves only at second order, by sum_j B_kj^2 w_kj with
    /// w_kj = d_k (g_j - g_k) + d_k d_j (g_j - g_k)^2 / (λ_k - λ_j),
    /// d the left roots, g the right roots routed through P and
    /// λ_k = d_k g_k. A target near a vertex spectrum (Newton from P stalls
    /// on the vanishing Jacobian) is met to second order by the squared
    /// angles x_kj = B_kj^2 >= 0 of the real phase-shift system.
    pub(crate) fn vertex_quadratic(&self) -> Vec<R4> {
        // Vertices whose spectrum lies within 1e-7 of a target lift, nearest
        // first. The window keeps the higher-order remainder small; a frame
        // that still misses the refinement target falls through.
        let mut near: Vec<(f64, [usize; 4], f64, [usize; 4])> = Vec::new();
        // Every root of a vertex spectrum is a routed product left_i right_j, so a
        // lift whose first target root has no routed product within the window
        // has no vertex in it.
        let reachable = |sign: f64| {
            let z = self.target_roots[0][0] * sign;
            self.routed
                .iter()
                .flatten()
                .any(|r| (r - z).norm_sqr() <= 1e-14)
        };
        let lifts: Vec<f64> = [1.0, -1.0].into_iter().filter(|&s| reachable(s)).collect();
        if lifts.is_empty() {
            return Vec::new();
        }
        let trace: Z = self.target_roots[0].iter().sum();
        for perm in *PERMS24 {
            let lam: [Z; 4] = std::array::from_fn(|k| self.left[k] * self.right[perm[k]]);
            // Nondegenerate second-order theory needs a simple vertex spectrum,
            // with gaps far (1e3x) above the window's root shifts.
            if PLANES.iter().any(|&(a, b)| (lam[a] - lam[b]).norm() < 1e-4) {
                continue;
            }
            let sum: Z = lam.iter().sum();
            for &sign in &lifts {
                // A root match within the window moves the trace by at most 4x.
                if (sum - trace * sign).norm() > 4e-7 {
                    continue;
                }
                let mut matched: Option<(f64, [usize; 4])> = None;
                for s in *PERMS24 {
                    let dev = (0..4)
                        .map(|k| (lam[k] - self.target_roots[0][s[k]] * sign).norm())
                        .fold(0.0, f64::max);
                    if matched.is_none_or(|(d, _)| dev < d) {
                        matched = Some((dev, s));
                    }
                }
                if let Some((dev, s)) = matched
                    && dev <= 1e-7
                    && dev > 0.0
                {
                    near.push((dev, perm, sign, s));
                }
            }
        }
        // An isolated vertex only: several permutations or lifts inside the
        // window mean degenerate factor spectra, where the vertex is not a
        // locally unique critical frame.
        if near.len() != 1 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &(_, perm, sign, s) in near.iter().take(4) {
            let g: [Z; 4] = std::array::from_fn(|k| self.right[perm[k]]);
            let lam: [Z; 4] = std::array::from_fn(|k| self.left[k] * g[k]);
            {
                let mut w = [[0.0f64; 6]; 4];
                for (q, &(a, b)) in PLANES.iter().enumerate() {
                    for (k, j) in [(a, b), (b, a)] {
                        let denominator = lam[k] - lam[j];
                        if denominator.norm() < 1e-8 {
                            continue;
                        }
                        let spread = g[j] - g[k];
                        let shift = self.left[k] * spread
                            + self.left[k] * self.left[j] * spread * spread / denominator;
                        w[k][q] = (shift / lam[k]).im;
                    }
                }
                let phi: [f64; 4] =
                    std::array::from_fn(|k| (self.target_roots[0][s[k]] * sign / lam[k]).arg());
                let Some(x) = nonnegative_squares(&w, &phi) else {
                    continue;
                };
                let mut b = R4::zeros();
                for (q, &(i, j)) in PLANES.iter().enumerate() {
                    b[(i, j)] = x[q].sqrt();
                    b[(j, i)] = -b[(i, j)];
                }
                let mut p = R4::zeros();
                for i in 0..4 {
                    p[(i, perm[i])] = 1.0;
                }
                let mut o = orthogonalize(exp_skew(&b) * p);
                if o.determinant() < 0.0 {
                    for i in 0..4 {
                        o[(i, 0)] *= -1.0;
                    }
                }
                out.push(o);
            }
        }
        out
    }
}
