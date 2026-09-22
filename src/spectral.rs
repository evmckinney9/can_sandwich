//! Spectral verification and endpoint factors from the same real eigenbasis.
use crate::{
    C as Z,
    problem::{PERMS24, Problem, eigphases, weyl_from_monodromy},
};
use nalgebra::{Matrix4, SymmetricEigen};
use std::f64::consts::PI;
type R4 = Matrix4<f64>;
#[derive(Clone, Copy)]
pub(crate) struct State {
    pub(crate) cost: f64,
    pub(crate) error: f64,
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

    pub(crate) fn state(&self, o: &R4, branch: f64) -> State {
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
            let cv = v.map(|x| Z::new(x, 0.0));
            let diag = cv.transpose() * s * cv;
            let mut err: f64 = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    if i != j {
                        err = err.max(diag[(i, j)].norm());
                    }
                }
            }
            if err < off {
                off = err;
                eigenvectors = v;
                roots = std::array::from_fn(|i| diag[(i, i)]);
            }
            if off < 2e-13 {
                break;
            }
        }
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
        let error = (0..4)
            .map(|i| (roots[i] - chosen[i]).norm())
            .fold(0.0, f64::max)
            + 4.0 * off;
        State {
            cost: best,
            error,
            roots,
            target: chosen,
            eigenvectors,
            off_diagonal_error: off,
            order,
            sign: chosen_sign,
        }
    }
}
/// Spectral screen for algebraic and numerical candidates.
pub(crate) fn verify(problem: &Problem, o: &R4) -> Option<State> {
    if !o.iter().all(|v| v.is_finite()) {
        return None;
    }
    if (o.transpose() * o - R4::identity()).amax() > 1e-11 || (o.determinant() - 1.0).abs() > 1e-11
    {
        return None;
    }
    let state = problem.state(o, 0.0);
    (state.cost.is_finite() && state.error < 8e-9).then_some(state)
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
        let dt = eigphases(weyl_from_monodromy(t)).map(|v| Z::from_polar(1.0, v));
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
            && (right.transpose() * right - R4::identity()).amax() < 1e-11
            && (right.determinant() - 1.0).abs() < 1e-11
            && (rebuilt - u).iter().all(|v| v.norm() < 8e-9)
        {
            Some((o, left, right, phase))
        } else {
            None
        }
    }
}
