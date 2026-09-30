//! Spectral verification and endpoint factors from the same real eigenbasis.
use crate::{
    C as Z,
    problem::{PERMS24, PLANES, Problem, phases},
};
use nalgebra::{Matrix4, SymmetricEigen};
use std::f64::consts::PI;
type R4 = Matrix4<f64>;
/// Maximum accepted root error, including the eigenbasis residual. This is
/// an acceptance ceiling; numerical refinement aims for substantially less.
pub const SPECTRAL_TOLERANCE: f64 = 1e-13;
#[derive(Clone, Copy)]
pub(crate) struct State {
    pub(crate) cost: f64,
    pub(crate) error: f64,
    pub(crate) root_error: f64,
    pub(crate) roots: [Z; 4],
    pub(crate) target: [Z; 4],
    pub(crate) eigenvectors: R4,
    pub(crate) off_diagonal_error: f64,
    order: [usize; 4],
    sign: f64,
}
impl Problem {
    /// Diagonal of the sandwich projected onto a given real basis: the roots
    /// to second order in the basis's off-diagonal coupling.
    pub(crate) fn roots_in_basis(&self, o: &R4, v: &R4) -> [Z; 4] {
        let d = project(&self.sandwich(o), v);
        std::array::from_fn(|i| d[(i, i)])
    }

    pub(crate) fn match_roots(&self, roots: [Z; 4]) -> f64 {
        let mut best = f64::INFINITY;
        for sign in [-1.0, 1.0] {
            let distances: [[f64; 4]; 4] = std::array::from_fn(|i| {
                std::array::from_fn(|j| (roots[i] - self.target_roots[0][j] * sign).norm_sqr())
            });
            for perm in *PERMS24 {
                let worst = (0..4).map(|i| distances[i][perm[i]]).fold(0.0, f64::max);
                best = best.min(worst);
            }
        }
        best.sqrt()
    }

    fn sandwich(&self, o: &R4) -> Matrix4<Z> {
        let mut s = Matrix4::<Z>::zeros();
        for i in 0..4 {
            for j in i..4 {
                let mut z = Z::new(0.0, 0.0);
                for k in 0..4 {
                    z += self.right[k] * (o[(i, k)] * o[(j, k)]);
                }
                z *= self.dc[(i, i)] * self.dc[(j, j)];
                s[(i, j)] = z;
                s[(j, i)] = z;
            }
        }
        s
    }

    pub(crate) fn state(&self, o: &R4, branch: f64) -> State {
        let s = self.sandwich(o);
        let mut eigenvectors = R4::identity();
        let mut roots = [Z::new(0.0, 0.0); 4];
        let mut off = f64::INFINITY;
        for weight in [
            0.6180339887498949,
            -std::f64::consts::SQRT_2,
            0.0,
            std::f64::consts::E,
        ] {
            let mut h = R4::from_fn(|i, j| s[(i, j)].re + weight * s[(i, j)].im);
            let center = h.trace() / 4.0;
            for i in 0..4 {
                h[(i, i)] -= center;
            }
            let scale = h.amax();
            if scale > 0.0 {
                h /= scale;
            }
            let Some(eigen) = SymmetricEigen::try_new(h, f64::EPSILON, 100) else {
                continue;
            };
            let v = eigen.eigenvectors;
            let diag = project(&s, &v);
            let err = coupling(&diag);
            if err < off {
                off = err;
                eigenvectors = v;
                roots = std::array::from_fn(|i| diag[(i, i)]);
            }
            if off < 1e-14 {
                break;
            }
        }
        self.assign(roots, eigenvectors, off, branch)
    }

    /// Closed-form state of a vertex or one-Givens edge frame. An exactly
    /// diagonal sandwich is its own eigenbasis. With one coupled plane the
    /// real and imaginary parts of the 2x2 block commute, so one Jacobi
    /// rotation is its eigenbasis; that state is kept only when its residual
    /// is far below every downstream refinement trigger.
    pub(crate) fn sparse_state(&self, o: &R4, branch: f64) -> Option<State> {
        let s = self.sandwich(o);
        let mut coupled = PLANES
            .iter()
            .filter(|&&(i, j)| s[(i, j)].re != 0.0 || s[(i, j)].im != 0.0);
        match (coupled.next(), coupled.next(), coupled.next()) {
            (None, _, _) => Some(self.assign(
                std::array::from_fn(|i| s[(i, i)]),
                R4::identity(),
                0.0,
                branch,
            )),
            // One coupled plane, or two disjoint ones (a two-block face
            // frame): each block is diagonalized by its own rotation.
            (Some(_), None, _) | (Some(_), Some(_), None) if coupled_planes_disjoint(&s) => {
                let (v, diag) = jacobi(R4::identity(), s);
                let off = coupling(&diag);
                let state = self.assign(std::array::from_fn(|i| diag[(i, i)]), v, off, branch);
                (state.error <= 2e-15).then_some(state)
            }
            _ => None,
        }
    }

    /// Closed-form state of a constructed frame whose spectrum is known: the
    /// real projection `h` has eigenvalues at the projected target roots, and
    /// each eigenvector spans the adjugate of `h - mu I`. The basis is scored
    /// by the same coupling bound as `state`; it is kept only when that bound
    /// is far below every downstream refinement trigger.
    pub(crate) fn targeted_state(&self, o: &R4) -> Option<State> {
        let s = self.sandwich(o);
        let trace = (0..4).map(|i| s[(i, i)]).sum::<Z>();
        let sum = self.target_roots[0].iter().sum::<Z>();
        let (plus, minus) = ((trace - sum).norm_sqr(), (trace + sum).norm_sqr());
        if (plus - minus).abs() <= 1e-6 * (plus + minus) {
            // e1 cannot name the sign branch; try both.
            return self
                .targeted_state_signed(&s, 1.0)
                .or_else(|| self.targeted_state_signed(&s, -1.0));
        }
        self.targeted_state_signed(&s, if plus <= minus { 1.0 } else { -1.0 })
    }

    fn targeted_state_signed(&self, s: &Matrix4<Z>, sign: f64) -> Option<State> {
        let roots = self.target_roots[0].map(|root| root * sign);
        // Real projection with the widest relative root separation.
        let mut weight = 0.0;
        let mut best = -1.0;
        for w in [
            0.6180339887498949,
            -std::f64::consts::SQRT_2,
            0.0,
            std::f64::consts::E,
            1.0,
            -0.5,
            0.3,
            -3.0,
        ] {
            let p = roots.map(|r| r.re + w * r.im);
            let spread = p.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - p.iter().copied().fold(f64::INFINITY, f64::min);
            let mut gap = f64::INFINITY;
            for (i, j) in PLANES {
                gap = gap.min((p[i] - p[j]).abs());
            }
            let separation = gap / spread;
            if separation > best {
                best = separation;
                weight = w;
            }
        }
        if !(best > 1e-3) {
            return None;
        }
        let mut h = R4::from_fn(|i, j| s[(i, j)].re + weight * s[(i, j)].im);
        let center = h.trace() / 4.0;
        for i in 0..4 {
            h[(i, i)] -= center;
        }
        let scale = h.amax();
        if !(scale > 0.0) || !scale.is_finite() {
            return None;
        }
        h /= scale;
        let mut v = R4::zeros();
        for (k, root) in roots.iter().enumerate() {
            let mu = (root.re + weight * root.im - center) / scale;
            let mut m = h;
            for i in 0..4 {
                m[(i, i)] -= mu;
            }
            v.set_column(k, &null_vector(&m)?);
        }
        // One Newton-Schulz step restores orthogonality to second order.
        let gram = v.transpose() * v;
        v = v * (R4::identity() * 3.0 - gram) * 0.5;
        let mut diag = project(s, &v);
        let mut off = coupling(&diag);
        if off > 1e-15 {
            // First-order eigenvector correction: rotate each plane by the
            // real ratio D_il / (D_ll - D_ii) (the parts commute), then
            // re-orthogonalize. No trigonometry is needed.
            let mut e = R4::identity();
            for (i, l) in PLANES {
                let spread = diag[(l, l)] - diag[(i, i)];
                let denominator = spread.norm_sqr();
                if !(denominator > 0.0) {
                    return None;
                }
                let angle = (diag[(i, l)] * spread.conj()).re / denominator;
                e[(i, l)] = angle;
                e[(l, i)] = -angle;
            }
            v *= e;
            let gram = v.transpose() * v;
            v = v * (R4::identity() * 3.0 - gram) * 0.5;
            diag = project(s, &v);
            off = coupling(&diag);
        }
        let state = self.assign(std::array::from_fn(|i| diag[(i, i)]), v, off, 0.0);
        (state.cost.is_finite() && state.error <= 5e-15).then_some(state)
    }

    /// Finish a basis with a large residual by Jacobi rotations of the complex
    /// matrix. One real projection cannot separate roots whose projected
    /// values nearly coincide.
    pub(crate) fn joint(&self, o: &R4, state: State, branch: f64) -> State {
        if state.off_diagonal_error <= 1e-15 {
            return state;
        }
        let s = self.sandwich(o);
        let (v, diag) = jacobi(state.eigenvectors, project(&s, &state.eigenvectors));
        let off = coupling(&diag);
        if off >= state.off_diagonal_error {
            return state;
        }
        self.assign(std::array::from_fn(|i| diag[(i, i)]), v, off, branch)
    }

    fn assign(&self, roots: [Z; 4], eigenvectors: R4, off: f64, branch: f64) -> State {
        let mut best = f64::INFINITY;
        let mut order = [0, 1, 2, 3];
        let mut chosen_sign = 1.0;
        for sign in [-1.0, 1.0] {
            if branch != 0.0 && sign != branch {
                continue;
            }
            let distances: [[f64; 4]; 4] = std::array::from_fn(|i| {
                std::array::from_fn(|j| (roots[i] - self.target_roots[0][j] * sign).norm_sqr())
            });
            for perm in *PERMS24 {
                let cost: f64 = (0..4).map(|i| distances[i][perm[i]]).sum();
                if cost < best {
                    best = cost;
                    order = perm;
                    chosen_sign = sign;
                }
            }
        }
        let chosen: [Z; 4] = std::array::from_fn(|i| self.target_roots[0][order[i]] * chosen_sign);
        let root_error = (0..4)
            .map(|i| (roots[i] - chosen[i]).norm_sqr())
            .fold(0.0, f64::max)
            .sqrt();
        let error = root_error + 4.0 * off;
        State {
            cost: best,
            error,
            root_error,
            roots,
            target: chosen,
            eigenvectors,
            off_diagonal_error: off,
            order,
            sign: chosen_sign,
        }
    }
}

/// Whether the coupled off-diagonal entries of `s` form at most two planes
/// with no shared index.
fn coupled_planes_disjoint(s: &Matrix4<Z>) -> bool {
    let mut used = [false; 4];
    for (i, j) in PLANES {
        if s[(i, j)].re != 0.0 || s[(i, j)].im != 0.0 {
            if used[i] || used[j] {
                return false;
            }
            used[i] = true;
            used[j] = true;
        }
    }
    true
}

/// Unit null vector of a symmetric matrix of corank one: the adjugate is
/// `c v v^T`, so its largest diagonal entry names the best column. The
/// adjugate is formed from the twelve 2x2 minors of the row pairs.
fn null_vector(m: &R4) -> Option<nalgebra::Vector4<f64>> {
    let a = |i: usize, j: usize| m[(i, j)];
    let s0 = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    let s1 = a(0, 0) * a(1, 2) - a(1, 0) * a(0, 2);
    let s2 = a(0, 0) * a(1, 3) - a(1, 0) * a(0, 3);
    let s3 = a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2);
    let s4 = a(0, 1) * a(1, 3) - a(1, 1) * a(0, 3);
    let s5 = a(0, 2) * a(1, 3) - a(1, 2) * a(0, 3);
    let c5 = a(2, 2) * a(3, 3) - a(3, 2) * a(2, 3);
    let c4 = a(2, 1) * a(3, 3) - a(3, 1) * a(2, 3);
    let c3 = a(2, 1) * a(3, 2) - a(3, 1) * a(2, 2);
    let c2 = a(2, 0) * a(3, 3) - a(3, 0) * a(2, 3);
    let c1 = a(2, 0) * a(3, 2) - a(3, 0) * a(2, 2);
    let c0 = a(2, 0) * a(3, 1) - a(3, 0) * a(2, 1);
    let adj = [
        [
            a(1, 1) * c5 - a(1, 2) * c4 + a(1, 3) * c3,
            -a(0, 1) * c5 + a(0, 2) * c4 - a(0, 3) * c3,
            a(3, 1) * s5 - a(3, 2) * s4 + a(3, 3) * s3,
            -a(2, 1) * s5 + a(2, 2) * s4 - a(2, 3) * s3,
        ],
        [
            -a(1, 0) * c5 + a(1, 2) * c2 - a(1, 3) * c1,
            a(0, 0) * c5 - a(0, 2) * c2 + a(0, 3) * c1,
            -a(3, 0) * s5 + a(3, 2) * s2 - a(3, 3) * s1,
            a(2, 0) * s5 - a(2, 2) * s2 + a(2, 3) * s1,
        ],
        [
            a(1, 0) * c4 - a(1, 1) * c2 + a(1, 3) * c0,
            -a(0, 0) * c4 + a(0, 1) * c2 - a(0, 3) * c0,
            a(3, 0) * s4 - a(3, 1) * s2 + a(3, 3) * s0,
            -a(2, 0) * s4 + a(2, 1) * s2 - a(2, 3) * s0,
        ],
        [
            -a(1, 0) * c3 + a(1, 1) * c1 - a(1, 2) * c0,
            a(0, 0) * c3 - a(0, 1) * c1 + a(0, 2) * c0,
            -a(3, 0) * s3 + a(3, 1) * s1 - a(3, 2) * s0,
            a(2, 0) * s3 - a(2, 1) * s1 + a(2, 2) * s0,
        ],
    ];
    let mut j = 0;
    for k in 1..4 {
        if adj[k][k].abs() > adj[j][j].abs() {
            j = k;
        }
    }
    let column = nalgebra::Vector4::new(adj[0][j], adj[1][j], adj[2][j], adj[3][j]);
    let norm = column.norm();
    (norm > 0.0 && norm.is_finite()).then(|| column / norm)
}

fn project(s: &Matrix4<Z>, v: &R4) -> Matrix4<Z> {
    let re = v.transpose() * s.map(|z| z.re) * v;
    let im = v.transpose() * s.map(|z| z.im) * v;
    Matrix4::from_fn(|i, j| Z::new(re[(i, j)], im[(i, j)]))
}

/// Largest off-diagonal entry, the residual of an approximate eigenbasis.
fn coupling(d: &Matrix4<Z>) -> f64 {
    let mut err: f64 = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                err = err.max(d[(i, j)].norm_sqr());
            }
        }
    }
    err.sqrt()
}

/// One sweep of real Jacobi rotations on the complex matrix `VᵀSV`. Real and
/// imaginary parts commute, so tan 2φ = 2 D_il / (D_ll - D_ii) is real.
fn jacobi(mut v: R4, mut diag: Matrix4<Z>) -> (R4, Matrix4<Z>) {
    for (i, l) in PLANES {
        let entry = diag[(i, l)];
        let spread = diag[(l, l)] - diag[(i, i)];
        if entry.norm_sqr() <= 1e-36 * spread.norm_sqr() {
            continue;
        }
        let angle = 0.5 * (2.0 * entry * spread.conj()).re.atan2(spread.norm_sqr());
        let (sine, cosine) = angle.sin_cos();
        for k in 0..4 {
            let (x, y) = (diag[(k, i)], diag[(k, l)]);
            diag[(k, i)] = x * cosine - y * sine;
            diag[(k, l)] = x * sine + y * cosine;
        }
        for k in 0..4 {
            let (x, y) = (diag[(i, k)], diag[(l, k)]);
            diag[(i, k)] = x * cosine - y * sine;
            diag[(l, k)] = x * sine + y * cosine;
        }
        for k in 0..4 {
            let (x, y) = (v[(k, i)], v[(k, l)]);
            v[(k, i)] = x * cosine - y * sine;
            v[(k, l)] = x * sine + y * cosine;
        }
    }
    (v, diag)
}

/// Spectral screen for algebraic and numerical candidates.
pub(crate) fn verify(problem: &Problem, o: &R4) -> Option<State> {
    if !o.iter().all(|v| v.is_finite()) {
        return None;
    }
    if (o.transpose() * o - R4::identity()).amax() > 1e-12 || (o.determinant() - 1.0).abs() > 1e-12
    {
        return None;
    }
    verify_frame(problem, o, false)
}

/// Spectral screen for a frame whose finiteness, orthogonality, and
/// determinant were already checked at the same `1e-12` bound.
pub(crate) fn verify_frame(problem: &Problem, o: &R4, sparse: bool) -> Option<State> {
    let fast = if sparse {
        problem.sparse_state(o, 0.0)
    } else {
        problem.targeted_state(o)
    };
    let state = match fast {
        Some(state) => state,
        None => problem.state(o, 0.0),
    };
    (state.cost.is_finite() && state.error < SPECTRAL_TOLERANCE).then_some(state)
}

impl State {
    /// Root residuals indexed by target position, with the sign branch.
    pub(crate) fn residual_by_target(&self) -> (f64, [Z; 4]) {
        let mut residual = [Z::new(0.0, 0.0); 4];
        for i in 0..4 {
            residual[self.order[i]] = self.roots[i] - self.target[i];
        }
        (self.sign, residual)
    }

    pub(crate) fn ordered_basis(&self) -> R4 {
        let mut left = R4::zeros();
        for i in 0..4 {
            left.set_column(self.order[i], &self.eigenvectors.column(i));
        }
        if left.determinant() < 0.0 {
            left.column_mut(0).neg_mut();
        }
        left
    }

    /// U = D_c O D_g = exp(i phase) L D_t R. The real and imaginary
    /// parts of U U^T commute, so its real eigenbasis supplies L.
    pub(crate) fn factors(&self, p: &Problem, t: [f64; 3], o: R4) -> Option<(R4, R4, R4, f64)> {
        let left = self.ordered_basis();
        let dt = phases(t).map(|v| Z::from_polar(1.0, v));
        let dg = p.right_phases.map(|v| Z::from_polar(1.0, v));
        let u = Matrix4::from_fn(|i, j| p.dc[(i, i)] * o[(i, j)] * dg[j]);
        let phase = if self.sign < 0.0 { PI / 2.0 } else { 0.0 };
        let scalar = Z::from_polar(1.0, phase);
        let complex_left = left.map(|v| Z::new(v, 0.0));
        let mut right = complex_left.transpose() * u;
        for i in 0..4 {
            for j in 0..4 {
                right[(i, j)] *= (scalar * dt[i]).conj();
            }
        }
        let right = right.map(|v| v.re);
        let rebuilt = complex_left * Matrix4::from_fn(|i, j| scalar * dt[i] * right[(i, j)]);
        if right.iter().all(|v| v.is_finite())
            && (right.transpose() * right - R4::identity()).amax() < 1e-12
            && (right.determinant() - 1.0).abs() < 1e-12
            && (rebuilt - u).iter().all(|v| v.norm() < 1e-12)
        {
            Some((o, left, right, phase))
        } else {
            None
        }
    }
}
