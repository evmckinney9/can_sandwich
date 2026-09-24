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
    let state = problem.state(o, 0.0);
    (state.cost.is_finite() && state.error < SPECTRAL_TOLERANCE).then_some(state)
}

impl State {
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
