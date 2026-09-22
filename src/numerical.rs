//! Bounded LM fallback, adapted from the 2026-09-21 spectral prototype.
use nalgebra::{Complex, Matrix4, SMatrix, SVector, SymmetricEigen};
use std::f64::consts::PI;
type Z = Complex<f64>;
type R4 = Matrix4<f64>;
const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const ACCEPT: f64 = 4e-9;
fn phases(m: [f64; 3]) -> [f64; 4] {
    let w = [m[0] + m[1], m[0] + m[2], m[1] + m[2]];
    [
        w[0] - w[1] + w[2],
        w[0] + w[1] - w[2],
        -w[0] - w[1] - w[2],
        -w[0] + w[1] + w[2],
    ]
    .map(|v| PI * v)
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
struct Problem {
    a: [Z; 4],
    b: [Z; 4],
    d: [Z; 4],
    target: [Z; 4],
    ratios: [Z; 6],
    perms: [[usize; 4]; 24],
}
struct State {
    cost: f64,
    error: f64,
    roots: [Z; 4],
    target: [Z; 4],
    v: R4,
    off: f64,
}
impl Problem {
    fn new(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Self {
        let ap = phases(c);
        let bp = phases(g);
        let tp = phases(t);
        let d = ap.map(|p| Z::from_polar(1.0, p / 2.0));
        Self {
            a: ap.map(|p| Z::from_polar(1.0, p)),
            b: bp.map(|p| Z::from_polar(1.0, p)),
            d,
            target: tp.map(|p| Z::from_polar(1.0, p)),
            ratios: std::array::from_fn(|k| {
                let (p, q) = PLANES[k];
                d[p] * d[q].conj()
            }),
            perms: *super::PERMS24,
        }
    }
    fn match_roots(&self, roots: [Z; 4]) -> f64 {
        let mut best = f64::INFINITY;
        for sign in [-1.0, 1.0] {
            for perm in self.perms {
                let mut worst: f64 = 0.0;
                for i in 0..4 {
                    worst = worst.max((roots[i] - self.target[perm[i]] * sign).norm());
                }
                best = best.min(worst);
            }
        }
        best
    }
    fn state(&self, o: &R4, branch: f64) -> State {
        let mut s = Matrix4::<Z>::zeros();
        for i in 0..4 {
            for j in i..4 {
                let mut z = Z::new(0.0, 0.0);
                for k in 0..4 {
                    z += self.b[k] * (o[(i, k)] * o[(j, k)]);
                }
                z *= self.d[i] * self.d[j];
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
        let mut chosen = self.target;
        for sign in [-1.0, 1.0] {
            if branch != 0.0 && sign != branch {
                continue;
            }
            for perm in self.perms {
                let target = std::array::from_fn(|i| self.target[perm[i]] * sign);
                let cost: f64 = (0..4).map(|i| (roots[i] - target[i]).norm_sqr()).sum();
                if cost < best {
                    best = cost;
                    chosen = target;
                }
            }
        }
        let error = (0..4)
            .map(|i| (roots[i] - chosen[i]).norm())
            .fold(0.0, f64::max)
            + 4.0 * off;
        State {
            cost: best,
            error,
            roots,
            target: chosen,
            v: eigenvectors,
            off,
        }
    }

    fn early(&self) -> Option<R4> {
        let trace: Z = self.target.iter().sum();
        for perm in self.perms {
            let roots = std::array::from_fn(|i| self.a[i] * self.b[perm[i]]);
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
                let slope = (self.a[p] - self.a[q]) * (self.b[perm[q]] - self.b[perm[p]]);
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
        let mut state = self.state(&o, branch);
        let mut damping = 1e-3;
        let mut stalled = 0;
        for _ in 0..120 {
            if state.cost.is_finite() && state.error < ACCEPT {
                return Some(o);
            }
            if !state.cost.is_finite() || state.off > 1e-10 {
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
                        let ratio = self.ratios[k];
                        (state.roots[l] * ratio - state.roots[i] * ratio.conj())
                            * state.v[(p, i)]
                            * state.v[(q, l)]
                            + (state.roots[i] * ratio - state.roots[l] * ratio.conj())
                                * state.v[(q, i)]
                                * state.v[(p, l)]
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
fn solve_primary(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<[[f64; 4]; 4]> {
    let problem = Problem::new(c, g, t);
    let pack = |o: R4| std::array::from_fn(|i| std::array::from_fn(|j| o[(i, j)]));
    if let Some(o) = problem.early() {
        return Some(pack(o));
    }
    let mut seed = 0x123456789abcdefu64;
    for value in c.into_iter().chain(g).chain(t) {
        seed = seed.rotate_left(7) ^ value.to_bits();
    }
    for _ in 0..24 {
        let mut o = R4::identity();
        for (p, q) in PLANES {
            rotate(&mut o, p, q, (2.0 * random(&mut seed) - 1.0) * PI);
        }
        if let Some(solution) = problem.iterate(o, 0.0) {
            return Some(pack(solution));
        }
    }
    let swapped = Problem::new(g, c, t);
    if let Some(o) = swapped.early() {
        return Some(pack(o.transpose()));
    }
    for _ in 0..24 {
        let mut o = R4::identity();
        for (p, q) in PLANES {
            rotate(&mut o, p, q, (2.0 * random(&mut seed) - 1.0) * PI);
        }
        if let Some(solution) = swapped.iterate(o, 0.0) {
            return Some(pack(solution.transpose()));
        }
    }
    for attempt in 0..24 {
        let mut o = R4::identity();
        for (p, q) in PLANES {
            rotate(&mut o, p, q, (2.0 * random(&mut seed) - 1.0) * PI);
        }
        let branch = if attempt % 2 == 0 { 1.0 } else { -1.0 };
        if let Some(solution) = problem.iterate(o, branch) {
            return Some(pack(solution));
        }
    }
    None
}

/// Reorder the real eigenbasis of A^-1/2 W C W^T A^-1/2 to B's spectrum.
/// This maps an inverse-factor problem back to the original problem.
fn inverse_factor(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<[[f64; 4]; 4]> {
    let inverse_c = c.map(|x| -x);
    let frame = solve_primary(inverse_c, t, g)?;
    let dual = Problem::new(inverse_c, t, g);
    let state = dual.state(&R4::from_fn(|i, j| frame[i][j]), 0.0);
    let mut best = f64::INFINITY;
    let mut ordering = [0, 1, 2, 3];
    for sign in [-1.0, 1.0] {
        for perm in dual.perms {
            let cost: f64 = (0..4)
                .map(|i| (state.roots[i] - dual.target[perm[i]] * sign).norm_sqr())
                .sum();
            if cost < best {
                best = cost;
                ordering = perm;
            }
        }
    }
    let mut o = R4::zeros();
    for i in 0..4 {
        o.set_column(ordering[i], &state.v.column(i));
    }
    if o.determinant() < 0.0 {
        o.column_mut(0).neg_mut();
    }
    let pack = |o: R4| std::array::from_fn(|i| std::array::from_fn(|j| o[(i, j)]));
    let original = Problem::new(c, g, t);
    let check = original.state(&o, 0.0);
    if check.cost.is_finite() && check.error < ACCEPT {
        Some(pack(o))
    } else {
        original.iterate(o, 0.0).map(pack)
    }
}

fn near_commuting(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<[[f64; 4]; 4]> {
    let problem = Problem::new(c, g, t);
    let mut starts: Vec<_> = problem
        .perms
        .into_iter()
        .map(|perm| {
            let roots = std::array::from_fn(|i| problem.a[i] * problem.b[perm[i]]);
            (problem.match_roots(roots), perm)
        })
        .collect();
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
                return Some(std::array::from_fn(|i| std::array::from_fn(|j| o[(i, j)])));
            }
        }
    }
    None
}

pub(super) fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<[[f64; 4]; 4]> {
    solve_primary(c, g, t)
        .or_else(|| near_commuting(c, g, t))
        .or_else(|| inverse_factor(c, g, t))
        .or_else(|| {
            inverse_factor(g, c, t)
                .map(|o| std::array::from_fn(|i| std::array::from_fn(|j| o[j][i])))
        })
}

/// Spectral screen for algebraic and numerical candidates.
pub(super) fn verify(c: [f64; 3], g: [f64; 3], t: [f64; 3], frame: [[f64; 4]; 4]) -> bool {
    let o = R4::from_fn(|i, j| frame[i][j]);
    if !o.iter().all(|v| v.is_finite()) {
        return false;
    }
    if (o.transpose() * o - R4::identity()).amax() > 1e-11 || (o.determinant() - 1.0).abs() > 1e-11
    {
        return false;
    }
    let state = Problem::new(c, g, t).state(&o, 0.0);
    state.cost.is_finite() && state.error < 8e-9
}

/// Polish an existing frame before paying for fresh deterministic starts.
pub(super) fn refine(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    frame: [[f64; 4]; 4],
) -> Option<[[f64; 4]; 4]> {
    let mut o = R4::from_fn(|i, j| frame[i][j]);
    if !o.iter().all(|v| v.is_finite()) {
        return None;
    }
    // Reorthogonalize the supplied frame; spectra and target are unchanged.
    for j in 0..4 {
        for k in 0..j {
            let dot = (0..4).map(|i| o[(i, j)] * o[(i, k)]).sum::<f64>();
            for i in 0..4 {
                o[(i, j)] -= dot * o[(i, k)];
            }
        }
        let norm = (0..4).map(|i| o[(i, j)] * o[(i, j)]).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return None;
        }
        for i in 0..4 {
            o[(i, j)] /= norm;
        }
    }
    if o.determinant() < 0.0 {
        for i in 0..4 {
            o[(i, 0)] *= -1.0;
        }
    }
    let result = Problem::new(c, g, t).iterate(o, 0.0).or_else(|| {
        Problem::new(g, c, t)
            .iterate(o.transpose(), 0.0)
            .map(|v| v.transpose())
    })?;
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| result[(i, j)])
    }))
}
