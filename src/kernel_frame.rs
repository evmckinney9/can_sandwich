//! Kernel-frame realization solver (research port of the s336 prototype).
//!
//! Representation: find `Q` in `SO(4)` with `det(Q ∘ C_j) = 0` for the sine
//! masks `C_j[i,k] = sin(pt_k − pc_i − pg_j)`; then
//! `S = (Sin∘Q)(Cos∘Q)^{-1}` is real symmetric with spectrum `tan(pg)` and its
//! eigenframe realizes the sandwich. `Q = L(l) R(r̄)` is linear in each spin
//! factor, so for a fixed factor the three independent conditions form a
//! 3-parameter multiparameter eigenvalue problem solved by one
//! staircase-compressed generalized eigensolve (Atkinson operator
//! determinants). Support strata are sparse-`Q` closed forms. Every accept is
//! gated by the forward spectral certificate.
//!
//! Research-gated: additive, no production dispatch changes.

use nalgebra::{Complex, DMatrix, Matrix4, SMatrix};

use super::{eigphases, poly_roots, rho_weyl, weyl_from_monodromy, C};

type M4 = SMatrix<f64, 4, 4>;
type M3 = SMatrix<f64, 3, 3>;
type V4 = SMatrix<f64, 4, 1>;
pub type Mat4c = Matrix4<C>;

const ACCEPT_SPEC: f64 = 1e-9;

// ---------- deterministic rng (xorshift* + polar normals) ----------
pub struct Rng {
    s: u64,
    spare: Option<f64>,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            s: seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
            spare: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * f);
                return u * f;
            }
        }
    }
}

// ---------- quaternions ----------
pub fn quat_l(q: [f64; 4]) -> M4 {
    let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
    M4::from_row_slice(&[a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a])
}
pub fn quat_r(q: [f64; 4]) -> M4 {
    let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
    M4::from_row_slice(&[a, -b, -c, -d, b, a, d, -c, c, -d, a, b, d, c, -b, a])
}
fn qconj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
fn qmul(p: [f64; 4], q: [f64; 4]) -> [f64; 4] {
    let (a1, b1, c1, d1) = (p[0], p[1], p[2], p[3]);
    let (a2, b2, c2, d2) = (q[0], q[1], q[2], q[3]);
    [
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    ]
}
fn qnormalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Split `Q ∈ SO(4)` into its spin pair `(l, r)` with `Q = L(l) R(r̄)`.
fn split_quat(q: &M4) -> Option<([f64; 4], [f64; 4])> {
    let e0 = [1.0, 0.0, 0.0, 0.0];
    let p = {
        let col = q * V4::from_column_slice(&e0);
        [col[0], col[1], col[2], col[3]]
    };
    let rot = quat_r(qconj(p)) * q;
    let r3: M3 = rot.fixed_view::<3, 3>(1, 1).into_owned();
    let tr = r3.trace();
    let l = if 1.0 + tr > 1e-10 {
        let l0 = (1.0 + tr).sqrt() / 2.0;
        [
            l0,
            (r3[(2, 1)] - r3[(1, 2)]) / (4.0 * l0),
            (r3[(0, 2)] - r3[(2, 0)]) / (4.0 * l0),
            (r3[(1, 0)] - r3[(0, 1)]) / (4.0 * l0),
        ]
    } else {
        let d = [
            ((r3[(0, 0)] + 1.0) / 2.0).max(0.0).sqrt(),
            ((r3[(1, 1)] + 1.0) / 2.0).max(0.0).sqrt(),
            ((r3[(2, 2)] + 1.0) / 2.0).max(0.0).sqrt(),
        ];
        let k = (0..3).max_by(|&a, &b| d[a].total_cmp(&d[b]))?;
        let mut v = [0.0; 3];
        v[k] = d[k];
        for i in 0..3 {
            if i != k && d[k] > 1e-9 {
                v[i] = (r3[(i, k)] + r3[(k, i)]) / (4.0 * d[k]);
            }
        }
        [0.0, v[0], v[1], v[2]]
    };
    let l = qnormalize(l);
    let r = qconj(qmul(qconj(l), p));
    Some((l, qnormalize(r)))
}

// ---------- spectral utilities ----------
fn eig4c(m: &Mat4c) -> Option<[C; 4]> {
    let fm = faer::Mat::<C>::from_fn(4, 4, |i, j| m[(i, j)]);
    let ev = fm.eigenvalues().ok()?;
    Some(std::array::from_fn(|i| ev[i]))
}

const PERMS4: [[usize; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

fn spec_dist(a: &[C; 4], b: &[C; 4]) -> f64 {
    let mut best = f64::INFINITY;
    for p in PERMS4 {
        let mut worst = 0.0f64;
        for i in 0..4 {
            worst = worst.max((a[i] - b[p[i]]).norm());
        }
        best = best.min(worst);
    }
    best
}

pub fn pole_gamma(pc: &[f64; 4], pg: &[f64; 4], pt: &[f64; 4]) -> f64 {
    let mut best = -1.0;
    let mut bg = 0.0;
    for n in 0..32 {
        let gam = std::f64::consts::PI * n as f64 / 32.0;
        let mut m = f64::INFINITY;
        for j in 0..4 {
            m = m.min((pg[j] + gam / 2.0).cos().abs());
        }
        for i in 0..4 {
            for k in 0..4 {
                m = m.min((pt[k] - pc[i] + gam / 2.0).cos().abs());
            }
        }
        if m > best {
            best = m;
            bg = gam;
        }
    }
    bg
}

/// The forward certificate: recover `S`, its eigenframe `O`, and check the
/// sandwich spectrum against the target roots (exact matching metric).
pub struct Verifier {
    pc: [f64; 4],
    pg: [f64; 4],
    sinx: M4,
    cosx: M4,
    tan_g: [f64; 4],
    troots: [C; 4],
}
impl Verifier {
    pub fn new(pc: [f64; 4], pg: [f64; 4], pt: [f64; 4]) -> Self {
        let gam = pole_gamma(&pc, &pg, &pt);
        let mut sinx = M4::zeros();
        let mut cosx = M4::zeros();
        for i in 0..4 {
            for k in 0..4 {
                let x = pt[k] - pc[i] + gam / 2.0;
                sinx[(i, k)] = x.sin();
                cosx[(i, k)] = x.cos();
            }
        }
        Self {
            pc,
            pg,
            sinx,
            cosx,
            tan_g: std::array::from_fn(|j| (pg[j] + gam / 2.0).tan()),
            troots: std::array::from_fn(|k| C::from_polar(1.0, 2.0 * pt[k])),
        }
    }

    pub fn verify(&self, q: &M4) -> Option<(M4, f64)> {
        let mq = self.cosx.component_mul(q);
        let inv = mq.try_inverse()?;
        let s = self.sinx.component_mul(q) * inv;
        let mut asym = 0.0f64;
        for i in 0..4 {
            for j in 0..4 {
                asym = asym.max((s[(i, j)] - s[(j, i)]).abs());
            }
        }
        if !asym.is_finite() || asym > 1e-5 {
            return None;
        }
        let s = (s + s.transpose()) * 0.5;
        let se = nalgebra::SymmetricEigen::new(s);
        // greedy eigenvalue match to tan(pg + gam/2)
        let mut used = [false; 4];
        let mut order = [0usize; 4];
        for j in 0..4 {
            let mut bi = usize::MAX;
            let mut bd = f64::INFINITY;
            for k in 0..4 {
                if !used[k] {
                    let d = (se.eigenvalues[k] - self.tan_g[j]).abs();
                    if d < bd {
                        bd = d;
                        bi = k;
                    }
                }
            }
            if bd > 1e-5 {
                return None;
            }
            used[bi] = true;
            order[j] = bi;
        }
        let mut o = M4::zeros();
        for j in 0..4 {
            o.set_column(j, &se.eigenvectors.column(order[j]));
        }
        if o.determinant() < 0.0 {
            for i in 0..4 {
                o[(i, 0)] = -o[(i, 0)];
            }
        }
        let dc = Mat4c::from_fn(|i, j| {
            if i == j {
                C::from_polar(1.0, self.pc[i])
            } else {
                C::new(0.0, 0.0)
            }
        });
        let lam = Mat4c::from_fn(|i, j| {
            if i == j {
                C::from_polar(1.0, 2.0 * self.pg[i])
            } else {
                C::new(0.0, 0.0)
            }
        });
        let oc = Mat4c::from_fn(|i, j| C::new(o[(i, j)], 0.0));
        let m = dc * oc * lam * oc.transpose() * dc;
        let got = eig4c(&m)?;
        let err = spec_dist(&got, &self.troots);
        (err < ACCEPT_SPEC).then_some((o, err))
    }
}

// ---------- staircase-compressed generalized eigenvalues ----------
fn regular_eigs(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Vec<Complex<f64>> {
    // The pencil is structurally singular; a fixed tiny regularization makes
    // the standard-eigen reduction well posed, and every candidate is re-gated
    // by correction + the forward certificate downstream.
    let n = a.nrows();
    let scale_a = a.amax();
    let scale_b = b.amax();
    for (round, tau) in [1e-10f64, 1e-7].into_iter().enumerate() {
        let mut ac = a.clone();
        let mut bc = b.clone();
        let mut prng = Rng::new(7 + round as u64);
        for v in ac.iter_mut() {
            *v += tau * scale_a * prng.normal();
        }
        for v in bc.iter_mut() {
            *v += tau * scale_b * prng.normal();
        }
        let out = gevd_pair(&ac, &bc);
        if out.len() > n / 3 || round == 1 {
            return out;
        }
    }
    vec![]
}

fn gevd_pair(ac: &DMatrix<f64>, bc: &DMatrix<f64>) -> Vec<Complex<f64>> {
    // Standard-eigen reduction of the (regularized) pencil: eig(B^-1 A).
    // Uses the same faer real-eigenvalue path as `poly_roots`.
    let n = ac.nrows();
    let lu = bc.clone().lu();
    let Some(m) = lu.solve(ac) else {
        return vec![];
    };
    let fm = faer::Mat::<f64>::from_fn(n, n, |i, j| m[(i, j)]);
    let Ok(ev) = fm.eigenvalues() else {
        return vec![];
    };
    let mut out = Vec::with_capacity(n);
    for z in ev {
        let z: C = z;
        if z.re.is_finite() && z.im.is_finite() {
            out.push(Complex::new(z.re, z.im));
        }
    }
    out
}

// ---------- small polynomial helpers ----------
/// Bivariate quartic coefficients from a 5x5 tensor grid at nodes -2..2.
fn biquartic<F: Fn(f64, f64) -> f64>(f: F) -> [[f64; 5]; 5] {
    let nodes = [-2.0, -1.0, 0.0, 1.0, 2.0];
    // inverse Vandermonde for those nodes (row i = coefficients of node^i fit)
    let mut vm = SMatrix::<f64, 5, 5>::zeros();
    for (r, &x) in nodes.iter().enumerate() {
        let mut p = 1.0;
        for c in 0..5 {
            vm[(r, c)] = p;
            p *= x;
        }
    }
    let vi = vm.try_inverse().expect("fixed vandermonde");
    let mut fv = SMatrix::<f64, 5, 5>::zeros();
    for (r, &x) in nodes.iter().enumerate() {
        for (c, &y) in nodes.iter().enumerate() {
            fv[(r, c)] = f(x, y);
        }
    }
    let co = vi * fv * vi.transpose();
    let mut out = [[0.0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            out[i][j] = co[(i, j)];
        }
    }
    out
}

fn poly_eval(c: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &ci in c.iter().rev() {
        acc = acc * x + ci;
    }
    acc
}

/// Real candidate y-values of `Res_x(P1, P2)(y)` by evaluation + fit + roots.
fn sylvester_roots(p1: &[[f64; 5]; 5], p2: &[[f64; 5]; 5]) -> Vec<f64> {
    let ny = 33usize;
    let mut ys = [0.0f64; 33];
    let mut vals = [0.0f64; 33];
    for t in 0..ny {
        let y = (std::f64::consts::PI * (t as f64 + 0.5) / ny as f64).cos() * 4.0;
        ys[t] = y;
        let a: [f64; 5] = std::array::from_fn(|i| poly_eval(&p1[i], y));
        let b: [f64; 5] = std::array::from_fn(|i| poly_eval(&p2[i], y));
        let scale_a = a.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        let scale_b = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        let na = (0..5)
            .rev()
            .find(|&i| a[i].abs() > 1e-13 * scale_a)
            .unwrap_or(0);
        let nb = (0..5)
            .rev()
            .find(|&i| b[i].abs() > 1e-13 * scale_b)
            .unwrap_or(0);
        if na == 0 || nb == 0 {
            vals[t] = if na == 0 && a[0].abs() < 1e-13 * scale_a {
                0.0
            } else {
                1.0
            };
            continue;
        }
        let m = na + nb;
        let mut sy = DMatrix::<f64>::zeros(m, m);
        for r in 0..nb {
            for c in 0..=na {
                sy[(r, r + c)] = a[na - c];
            }
        }
        for r in 0..na {
            for c in 0..=nb {
                sy[(nb + r, r + c)] = b[nb - c];
            }
        }
        vals[t] = sy.determinant();
    }
    let scale = vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return vec![];
    }
    let deg = 16usize;
    let mut vm = DMatrix::<f64>::zeros(ny, deg + 1);
    for t in 0..ny {
        let mut p = 1.0;
        for c in 0..=deg {
            vm[(t, c)] = p;
            p *= ys[t];
        }
    }
    let rhs = DMatrix::<f64>::from_fn(ny, 1, |t, _| vals[t] / scale);
    let svd = vm.svd(true, true);
    let cf = match svd.solve(&rhs, 1e-12) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    let coeffs: Vec<f64> = (0..=deg).map(|i| cf[(i, 0)]).collect();
    poly_roots(&coeffs)
        .into_iter()
        .filter(|z| z.im.abs() < 1e-3)
        .map(|z| z.re)
        .collect()
}

// ---------- the MEP core ----------
const PERMS3: [([usize; 3], f64); 6] = [
    ([0, 1, 2], 1.0),
    ([0, 2, 1], -1.0),
    ([1, 0, 2], -1.0),
    ([1, 2, 0], 1.0),
    ([2, 0, 1], 1.0),
    ([2, 1, 0], -1.0),
];

fn kron3(a: &M4, b: &M4, c: &M4, out: &mut DMatrix<f64>, sign: f64) {
    for i0 in 0..4 {
        for j0 in 0..4 {
            let ab = a[(i0, j0)];
            if ab == 0.0 {
                continue;
            }
            for i1 in 0..4 {
                for j1 in 0..4 {
                    let bb = ab * b[(i1, j1)];
                    if bb == 0.0 {
                        continue;
                    }
                    for i2 in 0..4 {
                        for j2 in 0..4 {
                            out[(16 * i0 + 4 * i1 + i2, 16 * j0 + 4 * j1 + j2)] +=
                                sign * bb * c[(i2, j2)];
                        }
                    }
                }
            }
        }
    }
}

struct FixedSlice {
    /// A[j][m]: condition j, coefficient of the m-th quaternion component.
    a: [[M4; 4]; 4],
}

impl FixedSlice {
    fn new(pc: &[f64; 4], pg: &[f64; 4], pt: &[f64; 4], q: [f64; 4], fix_right: bool) -> Self {
        Self::with_mode(pc, pg, pt, q, fix_right, 0)
    }

    /// mode 0: three distinct-gate masks. mode 1/2: replace the last one/two
    /// conditions with fixed generic masks -- a deterministic gauge cut for
    /// confluent rows whose solution sets are positive-dimensional (the
    /// certificate still gates every accept).
    fn with_mode(
        pc: &[f64; 4],
        pg: &[f64; 4],
        pt: &[f64; 4],
        q: [f64; 4],
        fix_right: bool,
        mode: usize,
    ) -> Self {
        // pick three masks with pairwise-distinct gate phases (mod pi); with a
        // gate collision the duplicate mask is a repeated condition that only
        // degrades the operator pencil.
        let mut picks: Vec<usize> = Vec::new();
        for j in 0..4 {
            if picks.iter().all(|&jj| {
                let d = (pg[j] - pg[jj]) % std::f64::consts::PI;
                let d = d.min(std::f64::consts::PI - d.abs()).abs();
                d > 1e-9
            }) {
                picks.push(j);
            }
            if picks.len() == 3 {
                break;
            }
        }
        while picks.len() < 3 {
            for j in 0..4 {
                if !picks.contains(&j) {
                    picks.push(j);
                    break;
                }
            }
        }
        let jsel = [
            picks[0],
            picks[1],
            picks[2],
            (0..4).find(|j| !picks.contains(j)).unwrap_or(3),
        ];
        let mut cs = [[0.0f64; 16]; 4];
        for (slot, &j) in jsel.iter().enumerate() {
            for i in 0..4 {
                for k in 0..4 {
                    cs[slot][4 * i + k] = (pt[k] - pc[i] - pg[j]).sin();
                }
            }
        }
        if mode >= 1 {
            let mut grng = Rng::new(1234);
            for e in 0..16 {
                cs[2][e] = grng.normal();
            }
            if mode >= 2 {
                for e in 0..16 {
                    cs[1][e] = grng.normal();
                }
            }
            // conditions 1..2 replaced by gauge cuts; the true masks stay in
            // slots 0 (and 3 for the correction rows)
        }
        let conj = M4::from_diagonal(&V4::from_column_slice(&[1.0, -1.0, -1.0, -1.0]).into());
        let mut a = [[M4::zeros(); 4]; 4];
        for m in 0..4 {
            let mut e = [0.0; 4];
            e[m] = 1.0;
            let base = if fix_right {
                quat_l(e)
                    * quat_r({
                        let cq = conj * V4::from_column_slice(&q);
                        [cq[0], cq[1], cq[2], cq[3]]
                    })
            } else {
                quat_l(q)
                    * quat_r({
                        let ce = conj * V4::from_column_slice(&e);
                        [ce[0], ce[1], ce[2], ce[3]]
                    })
            };
            for j in 0..4 {
                let mut mj = M4::zeros();
                for i in 0..4 {
                    for k in 0..4 {
                        mj[(i, k)] = base[(i, k)] * cs[j][4 * i + k];
                    }
                }
                a[j][m] = mj;
            }
        }
        Self { a }
    }

    /// One first-order singular-vector correction pass (fixed count).
    fn sv_correct(&self, r: [f64; 4]) -> [f64; 4] {
        let mut r = qnormalize(r);
        for _ in 0..2 {
            let mut rows = M4::zeros();
            for j in 0..4 {
                let mut w = M4::zeros();
                for m in 0..4 {
                    w += self.a[j][m] * r[m];
                }
                let svd = w.svd(true, true);
                let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
                let mut mi = 0usize;
                for e in 1..4 {
                    if svd.singular_values[e] < svd.singular_values[mi] {
                        mi = e;
                    }
                }
                let uv = u.column(mi);
                let xv = vt.row(mi);
                for m in 0..4 {
                    let mut acc = 0.0;
                    for i in 0..4 {
                        for k in 0..4 {
                            acc += uv[i] * self.a[j][m][(i, k)] * xv[k];
                        }
                    }
                    rows[(j, m)] = acc;
                }
            }
            let svd = rows.svd(true, true);
            let vt = svd.v_t.unwrap();
            let mut mi = 0usize;
            for e in 1..4 {
                if svd.singular_values[e] < svd.singular_values[mi] {
                    mi = e;
                }
            }
            let vlast = vt.row(mi);
            let cand = [vlast[0], vlast[1], vlast[2], vlast[3]];
            let n = (cand.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if n < 1e-12 {
                return r;
            }
            r = [cand[0] / n, cand[1] / n, cand[2] / n, cand[3] / n];
        }
        r
    }

    fn solve(
        &self,
        ad: &M4,
        q_fixed: [f64; 4],
        fix_right: bool,
        ver: &Verifier,
    ) -> Option<(M4, f64)> {
        // B[j][i] = sum_m Ad[m][i] A[j][m], j = 0..2 only for the MEP
        let mut b = [[M4::zeros(); 4]; 3];
        for j in 0..3 {
            for i in 0..4 {
                let mut acc = M4::zeros();
                for m in 0..4 {
                    acc += self.a[j][m] * ad[(m, i)];
                }
                b[j][i] = acc;
            }
        }
        let mut d0 = DMatrix::<f64>::zeros(64, 64);
        let mut d1 = DMatrix::<f64>::zeros(64, 64);
        for (p, sg) in PERMS3 {
            kron3(
                &b[0][1 + p[0]],
                &b[1][1 + p[1]],
                &b[2][1 + p[2]],
                &mut d0,
                sg,
            );
            let mut pick = [1 + p[0], 1 + p[1], 1 + p[2]];
            let slot = p.iter().position(|&x| x == 0).unwrap();
            pick[slot] = 0;
            kron3(&b[0][pick[0]], &b[1][pick[1]], &b[2][pick[2]], &mut d1, -sg);
        }
        let lam1s: Vec<f64> = regular_eigs(&d1, &d0)
            .into_iter()
            .filter(|z| z.im.abs() < 1e-3 && z.re.abs() < 1e6)
            .map(|z| z.re)
            .collect();
        let mut best: Option<(M4, f64)> = None;
        let mut seen: Vec<f64> = Vec::new();
        for l1 in lam1s {
            if seen.iter().any(|&s| (s - l1).abs() < 1e-9) {
                continue;
            }
            seen.push(l1);
            let w0 = |j: usize| b[j][0] + b[j][1] * l1;
            let f1 = {
                let w = w0(0);
                let (b2, b3) = (b[0][2], b[0][3]);
                move |x: f64, y: f64| (w + b2 * x + b3 * y).determinant()
            };
            let f2 = {
                let w = w0(1);
                let (b2, b3) = (b[1][2], b[1][3]);
                move |x: f64, y: f64| (w + b2 * x + b3 * y).determinant()
            };
            let p1 = biquartic(&f1);
            let p2 = biquartic(&f2);
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for l3 in sylvester_roots(&p1, &p2) {
                let a: Vec<f64> = (0..5).map(|i| poly_eval(&p1[i], l3)).collect();
                for z in poly_roots(&a) {
                    if z.im.abs() < 1e-3 {
                        pairs.push((z.re, l3));
                    }
                }
            }
            // transposed elimination order
            let p1t = transpose5(&p1);
            let p2t = transpose5(&p2);
            for l2 in sylvester_roots(&p1t, &p2t) {
                let a: Vec<f64> = (0..5).map(|i| poly_eval(&p1t[i], l2)).collect();
                for z in poly_roots(&a) {
                    if z.im.abs() < 1e-3 {
                        pairs.push((l2, z.re));
                    }
                }
            }
            for (l2, l3) in pairs {
                let lam = V4::from_column_slice(&[1.0, l1, l2, l3]);
                let rv = ad * lam;
                let n = rv.norm();
                if n < 1e-12 {
                    continue;
                }
                let r = self.sv_correct([rv[0] / n, rv[1] / n, rv[2] / n, rv[3] / n]);
                let conj = [r[0], -r[1], -r[2], -r[3]];
                let q = if fix_right {
                    quat_l(r) * quat_r(qconj(q_fixed))
                } else {
                    quat_l(q_fixed) * quat_r(conj)
                };
                if let Some((o, err)) = ver.verify(&q) {
                    if best.as_ref().is_none_or(|(_, e)| err < *e) {
                        best = Some((o, err));
                        if err < 1e-11 {
                            return best;
                        }
                    }
                }
            }
        }
        best
    }
}

fn transpose5(p: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let mut out = [[0.0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            out[i][j] = p[j][i];
        }
    }
    out
}

// ---------- the forward dictionary ----------
struct Seed {
    dist: f64,
    l: [f64; 4],
    r: [f64; 4],
}

fn dictionary(pc: &[f64; 4], pg: &[f64; 4], pt: &[f64; 4], n: usize, seed: u64) -> Vec<Seed> {
    let gam = pole_gamma(pc, pg, pt);
    let tan_g: [f64; 4] = std::array::from_fn(|j| (pg[j] + gam / 2.0).tan());
    let troots: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * pt[k]));
    let dc = Mat4c::from_fn(|i, j| {
        if i == j {
            C::from_polar(1.0, pc[i])
        } else {
            C::new(0.0, 0.0)
        }
    });
    let lam = Mat4c::from_fn(|i, j| {
        if i == j {
            C::from_polar(1.0, 2.0 * pg[i])
        } else {
            C::new(0.0, 0.0)
        }
    });
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let g = M4::from_fn(|_, _| rng.normal());
        let qr = g.qr();
        let mut a = qr.q();
        if a.determinant() < 0.0 {
            for i in 0..4 {
                a[(i, 0)] = -a[(i, 0)];
            }
        }
        let ac = Mat4c::from_fn(|i, j| C::new(a[(i, j)], 0.0));
        let m = dc * ac * lam * ac.transpose() * dc;
        let Some(w) = eig4c(&m) else { continue };
        let dist = spec_dist(&w, &troots);
        let pts: [f64; 4] = std::array::from_fn(|k| w[k].arg() / 2.0);
        let s = a * M4::from_diagonal(&V4::from_column_slice(&tan_g).into()) * a.transpose();
        let mut qs = M4::zeros();
        let mut ok = true;
        for k in 0..4 {
            let mut hk = M4::zeros();
            for i in 0..4 {
                hk[(i, i)] = (pts[k] - pc[i] + gam / 2.0).tan();
            }
            let se = nalgebra::SymmetricEigen::new(s - hk);
            // repeated target roots give multi-dimensional kernels shared by
            // duplicate columns; pick the kernel direction orthogonal to the
            // columns already assigned from the same kernel.
            let mut cand: Vec<usize> = (0..4).filter(|&e| se.eigenvalues[e].abs() < 1e-6).collect();
            if cand.is_empty() {
                ok = false;
                break;
            }
            cand.sort_by(|&a, &b| se.eigenvalues[a].abs().total_cmp(&se.eigenvalues[b].abs()));
            let mut chosen: Option<nalgebra::SVector<f64, 4>> = None;
            for &e in &cand {
                let mut v: nalgebra::SVector<f64, 4> = se.eigenvectors.column(e).into_owned();
                for kp in 0..k {
                    if (pts[k] - pts[kp]).abs() < 1e-6 {
                        let mut prev = nalgebra::SVector::<f64, 4>::zeros();
                        for i in 0..4 {
                            let cx = (pts[kp] - pc[i] + gam / 2.0).cos();
                            prev[i] = qs[(i, kp)] * cx;
                        }
                        let nn = prev.norm();
                        if nn > 1e-12 {
                            let prev = prev / nn;
                            v -= prev * prev.dot(&v);
                        }
                    }
                }
                if v.norm() > 1e-6 {
                    chosen = Some(v.normalize());
                    break;
                }
            }
            let Some(v) = chosen else {
                ok = false;
                break;
            };
            for i in 0..4 {
                let cx = (pts[k] - pc[i] + gam / 2.0).cos();
                if cx.abs() < 1e-8 {
                    ok = false;
                    break;
                }
                qs[(i, k)] = v[i] / cx;
            }
            if !ok {
                break;
            }
            let nrm = qs.column(k).norm();
            if nrm < 1e-12 {
                ok = false;
                break;
            }
            for i in 0..4 {
                qs[(i, k)] /= nrm;
            }
        }
        if !ok {
            continue;
        }
        if (qs.determinant().abs() - 1.0).abs() > 1e-6 {
            continue;
        }
        if qs.determinant() < 0.0 {
            for i in 0..4 {
                qs[(i, 0)] = -qs[(i, 0)];
            }
        }
        let Some((l, r)) = split_quat(&qs) else {
            continue;
        };
        out.push(Seed { dist, l, r });
    }
    out
}

fn fps_order(seeds: &mut Vec<Seed>, m: usize) -> Vec<usize> {
    if seeds.is_empty() {
        return vec![];
    }
    seeds.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    let dl = |a: &[f64; 4], b: &[f64; 4]| -> f64 {
        let dp: f64 = (0..4).map(|i| (a[i] - b[i]) * (a[i] - b[i])).sum();
        let dm: f64 = (0..4).map(|i| (a[i] + b[i]) * (a[i] + b[i])).sum();
        dp.min(dm).sqrt()
    };
    let mut chosen = vec![0usize];
    let mut dmin: Vec<f64> = seeds.iter().map(|s| dl(&s.l, &seeds[0].l)).collect();
    while chosen.len() < m.min(seeds.len()) {
        let (i, _) = dmin
            .iter()
            .enumerate()
            .filter(|(i, _)| !chosen.contains(i))
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        chosen.push(i);
        for (j, s) in seeds.iter().enumerate() {
            let d = dl(&s.l, &seeds[i].l);
            if d < dmin[j] {
                dmin[j] = d;
            }
        }
    }
    chosen
}

// ---------- sparse support pass ----------
fn block2(theta: f64, s: f64) -> [[f64; 2]; 2] {
    let (c, sn) = (theta.cos(), theta.sin());
    [[c, -s * sn], [sn, s * c]]
}

fn block_thetas(c2: &[[f64; 2]; 2]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for s in [1.0, -1.0] {
        let num = -c2[0][0] * c2[1][1];
        let den = c2[0][1] * c2[1][0];
        if den.abs() < 1e-15 {
            if num.abs() < 1e-12 {
                for th in [
                    0.0,
                    std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_2,
                ] {
                    out.push((th, s));
                }
            }
            continue;
        }
        let t2 = num / den;
        if t2 < -1e-9 {
            continue;
        }
        let th = t2.max(0.0).sqrt().atan();
        out.push((th, s));
        out.push((-th, s));
    }
    out
}

fn solve_sparse(pc: &[f64; 4], pg: &[f64; 4], pt: &[f64; 4], ver: &Verifier) -> Option<(M4, f64)> {
    const TOL: f64 = 1e-9;
    let mut cs = [[[0.0f64; 4]; 4]; 4];
    for j in 0..4 {
        for i in 0..4 {
            for k in 0..4 {
                cs[j][i][k] = (pt[k] - pc[i] - pg[j]).sin();
            }
        }
    }
    let mut best: Option<(M4, f64)> = None;
    let mut try_q = |q: &M4, best: &mut Option<(M4, f64)>| {
        if let Some((o, err)) = ver.verify(q) {
            if best.as_ref().is_none_or(|(_, e)| err < *e) {
                *best = Some((o, err));
            }
        }
    };
    // vertex
    for p in PERMS4 {
        let all = (0..3).all(|j| {
            (0..4)
                .map(|k| cs[j][p[k]][k].abs())
                .fold(f64::INFINITY, f64::min)
                < TOL
        });
        if all {
            let mut q = M4::zeros();
            for k in 0..4 {
                q[(p[k], k)] = 1.0;
            }
            try_q(&q, &mut best);
            if best.as_ref().is_some_and(|(_, e)| *e < 1e-12) {
                return best;
            }
        }
    }
    // edge: one 2x2 block
    for i2a in 0..4 {
        for i2b in i2a + 1..4 {
            let rest_i: Vec<usize> = (0..4).filter(|&i| i != i2a && i != i2b).collect();
            for k2a in 0..4 {
                for k2b in k2a + 1..4 {
                    let rest_k: Vec<usize> = (0..4).filter(|&k| k != k2a && k != k2b).collect();
                    for swap in [false, true] {
                        let pins = if swap {
                            [(rest_i[0], rest_k[1]), (rest_i[1], rest_k[0])]
                        } else {
                            [(rest_i[0], rest_k[0]), (rest_i[1], rest_k[1])]
                        };
                        let mut cands: Option<Vec<(f64, f64)>> = None;
                        let mut dead = false;
                        for j in 0..3 {
                            let pref: f64 = pins.iter().map(|&(i, k)| cs[j][i][k].abs()).product();
                            if pref < TOL {
                                continue;
                            }
                            let c2 = [
                                [cs[j][i2a][k2a], cs[j][i2a][k2b]],
                                [cs[j][i2b][k2a], cs[j][i2b][k2b]],
                            ];
                            let cur = block_thetas(&c2);
                            cands = Some(match cands {
                                None => cur,
                                Some(prev) => prev
                                    .into_iter()
                                    .filter(|(t, s)| {
                                        cur.iter().any(|(t2, s2)| (t - t2).abs() < 1e-7 && s == s2)
                                    })
                                    .collect(),
                            });
                            if cands.as_ref().unwrap().is_empty() {
                                dead = true;
                                break;
                            }
                        }
                        if dead {
                            continue;
                        }
                        let cands = cands.unwrap_or_else(|| {
                            vec![(0.0, 1.0), (std::f64::consts::FRAC_PI_4, 1.0)]
                        });
                        for (th, s) in cands {
                            let b = block2(th, s);
                            let mut q = M4::zeros();
                            for &(i, k) in &pins {
                                q[(i, k)] = 1.0;
                            }
                            q[(i2a, k2a)] = b[0][0];
                            q[(i2a, k2b)] = b[0][1];
                            q[(i2b, k2a)] = b[1][0];
                            q[(i2b, k2b)] = b[1][1];
                            try_q(&q, &mut best);
                        }
                    }
                }
            }
        }
    }
    if best.as_ref().is_some_and(|(_, e)| *e < 1e-12) {
        return best;
    }
    // face: two disjoint blocks
    let rowsplits = [
        [(0usize, 1usize), (2, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
    ];
    for ia in rowsplits {
        for ka in rowsplits {
            for flip in [false, true] {
                let (ra, rb) = (ia[0], ia[1]);
                let (ca, cb) = if flip { (ka[1], ka[0]) } else { (ka[0], ka[1]) };
                for mask in 0..8u32 {
                    let mut tha: Option<Vec<(f64, f64)>> = None;
                    let mut thb: Option<Vec<(f64, f64)>> = None;
                    let mut dead = false;
                    for j in 0..3 {
                        let (rows, colsp, store) = if mask >> j & 1 == 1 {
                            (ra, ca, &mut tha)
                        } else {
                            (rb, cb, &mut thb)
                        };
                        let c2 = [
                            [cs[j][rows.0][colsp.0], cs[j][rows.0][colsp.1]],
                            [cs[j][rows.1][colsp.0], cs[j][rows.1][colsp.1]],
                        ];
                        let cur = block_thetas(&c2);
                        if cur.is_empty() {
                            dead = true;
                            break;
                        }
                        *store = Some(match store.take() {
                            None => cur,
                            Some(prev) => prev
                                .into_iter()
                                .filter(|(t, s)| {
                                    cur.iter().any(|(t2, s2)| (t - t2).abs() < 1e-7 && s == s2)
                                })
                                .collect(),
                        });
                        if store.as_ref().unwrap().is_empty() {
                            dead = true;
                            break;
                        }
                    }
                    if dead {
                        continue;
                    }
                    let defaults = || {
                        vec![
                            (0.0, 1.0),
                            (std::f64::consts::FRAC_PI_4, 1.0),
                            (std::f64::consts::FRAC_PI_2, 1.0),
                        ]
                    };
                    let la = tha.unwrap_or_else(defaults);
                    let lb = thb.unwrap_or_else(defaults);
                    for &(t1, s1) in la.iter().take(4) {
                        for &(t2, s2) in lb.iter().take(4) {
                            let b1 = block2(t1, s1);
                            let b2m = block2(t2, s2);
                            let mut q = M4::zeros();
                            q[(ra.0, ca.0)] = b1[0][0];
                            q[(ra.0, ca.1)] = b1[0][1];
                            q[(ra.1, ca.0)] = b1[1][0];
                            q[(ra.1, ca.1)] = b1[1][1];
                            q[(rb.0, cb.0)] = b2m[0][0];
                            q[(rb.0, cb.1)] = b2m[0][1];
                            q[(rb.1, cb.0)] = b2m[1][0];
                            q[(rb.1, cb.1)] = b2m[1][1];
                            try_q(&q, &mut best);
                        }
                    }
                }
            }
        }
    }
    best
}

// ---------- deterministic measure-step path (the fast path) ----------
fn skew3(k: [f64; 3]) -> M3 {
    M3::from_row_slice(&[0.0, -k[2], k[1], k[2], 0.0, -k[0], -k[1], k[0], 0.0])
}
fn cayley3(k: [f64; 3]) -> M3 {
    let kk = skew3(k);
    (M3::identity() - kk)
        * (M3::identity() + kk)
            .try_inverse()
            .unwrap_or_else(M3::identity)
}

struct TerminalData {
    r0: V4,
    u0: SMatrix<f64, 4, 3>,
    bs: [M3; 4],
    gs: [M3; 4],
    perm: [usize; 4],
}

fn terminal_data(
    pc: &[f64; 4],
    pg: &[f64; 4],
    pt: &[f64; 4],
    sh: &[f64; 3],
) -> Option<TerminalData> {
    let mut cs: [f64; 4] = std::array::from_fn(|i| pc[i].tan());
    let mut perm = [0usize, 1, 2, 3];
    perm.sort_by(|&a, &b| cs[a].total_cmp(&cs[b]));
    cs.sort_by(|a, b| a.total_cmp(b));
    let mut shs = *sh;
    shs.sort_by(|a, b| a.total_cmp(b));
    let mut rho = [0.0f64; 4];
    for i in 0..4 {
        let mut num = 1.0;
        for m in 0..3 {
            num *= cs[i] - shs[m];
        }
        let mut den = 1.0;
        for ii in 0..4 {
            if ii != i {
                den *= cs[i] - cs[ii];
            }
        }
        rho[i] = num / den;
        if rho[i] <= 1e-14 {
            return None;
        }
    }
    let r0 = V4::from_column_slice(&[rho[0].sqrt(), rho[1].sqrt(), rho[2].sqrt(), rho[3].sqrt()]);
    let proj = M4::identity() - r0 * r0.transpose();
    let svd = proj.svd(true, false);
    let uu = svd.u.unwrap();
    let mut idx: Vec<usize> = (0..4).collect();
    idx.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
    let mut u0 = SMatrix::<f64, 4, 3>::zeros();
    for c in 0..3 {
        u0.set_column(c, &uu.column(idx[c]));
    }
    let c4 = M4::from_fn(|i, j| if i == j { cs[i] } else { 0.0 });
    let mut bs = [M3::zeros(); 4];
    let mut gs = [M3::zeros(); 4];
    for j in 0..4 {
        let d: [f64; 4] = std::array::from_fn(|k| (pt[k] - pg[j]).tan());
        let a_j = r0 * r0.transpose() * d[0] - c4;
        let inv = a_j.try_inverse()?;
        bs[j] = u0.transpose() * inv * u0;
        gs[j] = M3::from_fn(|i, jj| if i == jj { 1.0 / d[i + 1] } else { 0.0 });
    }
    Some(TerminalData {
        r0,
        u0,
        bs,
        gs,
        perm,
    })
}

impl TerminalData {
    fn frame(&self, r: &M3) -> M4 {
        let lower = r.transpose() * self.u0.transpose(); // 3x4
        let mut es = M4::zeros();
        for c in 0..4 {
            es[(0, c)] = self.r0[c];
            for rr in 0..3 {
                es[(rr + 1, c)] = lower[(rr, c)];
            }
        }
        // columns of the sorted eigenframe are the kernel-frame rows (pc order)
        let mut qt = M4::zeros();
        for s in 0..4 {
            for c in 0..4 {
                qt[(self.perm[s], c)] = es[(c, s)];
            }
        }
        qt
    }

    fn residual(&self, k: [f64; 3], j: usize) -> f64 {
        let r = cayley3(k);
        let m = self.bs[j] + r * self.gs[j] * r.transpose();
        m.determinant() / m.norm().max(1e-30).powi(3)
    }

    fn solve(&self, ver: &Verifier) -> Option<(M4, f64)> {
        let n_grid = 8usize;
        let kmax = 6.0f64;
        let node =
            |q: usize| (std::f64::consts::PI * (q as f64 + 0.5) / n_grid as f64).cos() * kmax;
        let mut cells: Vec<(f64, [f64; 3])> = Vec::with_capacity(n_grid.pow(3));
        for gi in 0..n_grid * n_grid * n_grid {
            let k = [
                node(gi / (n_grid * n_grid)),
                node(gi / n_grid % n_grid),
                node(gi % n_grid),
            ];
            let sc = self
                .residual(k, 0)
                .abs()
                .max(self.residual(k, 1).abs())
                .max(self.residual(k, 2).abs());
            if sc.is_finite() {
                cells.push((sc, k));
            }
        }
        cells.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, k0) in cells.iter().take(12) {
            let mut k = k0;
            let mut ok = false;
            for _ in 0..25 {
                let f0 = [
                    self.residual(k, 0),
                    self.residual(k, 1),
                    self.residual(k, 2),
                ];
                let fmax = f0.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                if !fmax.is_finite() {
                    break;
                }
                if fmax < 1e-13 {
                    ok = true;
                    break;
                }
                let h = 1e-7;
                let mut jac = M3::zeros();
                for d in 0..3 {
                    let mut kp = k;
                    kp[d] += h;
                    for j in 0..3 {
                        jac[(j, d)] = (self.residual(kp, j) - f0[j]) / h;
                    }
                }
                let Some(inv) = jac.try_inverse() else { break };
                let f0v = SMatrix::<f64, 3, 1>::from_column_slice(&f0);
                let dk = -(inv * f0v);
                let mut lam = 1.0;
                let mut moved = false;
                while lam > 1e-6 {
                    let kn = [k[0] + lam * dk[0], k[1] + lam * dk[1], k[2] + lam * dk[2]];
                    let nn = (kn[0] * kn[0] + kn[1] * kn[1] + kn[2] * kn[2]).sqrt();
                    if nn < 40.0 {
                        let fn_ = [
                            self.residual(kn, 0),
                            self.residual(kn, 1),
                            self.residual(kn, 2),
                        ];
                        let fnmax = fn_.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                        if fnmax.is_finite() && fnmax < fmax {
                            k = kn;
                            moved = true;
                            break;
                        }
                    }
                    lam *= 0.5;
                }
                if !moved {
                    break;
                }
            }
            if ok && self.residual(k, 3).abs() < 1e-8 {
                let r = cayley3(k);
                let qt = self.frame(&r);
                if let Some((o, err)) = ver.verify(&qt) {
                    return Some((o, err));
                }
                if let Some((o, err)) = ver.verify(&qt.transpose()) {
                    return Some((o, err));
                }
            }
        }
        None
    }
}

fn solve_deterministic(
    pc: &[f64; 4],
    pg: &[f64; 4],
    pt: &[f64; 4],
    ver: &Verifier,
) -> Option<(M4, f64)> {
    let mut cs: [f64; 4] = std::array::from_fn(|i| pc[i].tan());
    cs.sort_by(|a, b| a.total_cmp(b));
    let frac = |a: f64| -> [f64; 3] { std::array::from_fn(|i| cs[i] + a * (cs[i + 1] - cs[i])) };
    let schedule = [
        frac(0.5),
        frac(0.3),
        frac(0.7),
        [frac(0.3)[0], frac(0.5)[1], frac(0.7)[2]],
    ];
    for sh in schedule {
        if let Some(td) = terminal_data(pc, pg, pt, &sh) {
            if let Some(hit) = td.solve(ver) {
                return Some(hit);
            }
        }
    }
    None
}

// ---------- the composite entry ----------
pub struct KfSolution {
    pub o: M4,
    pub spec_err: f64,
}

/// Stage diagnostics for one row (research CLI).
pub fn debug_row(c: [f64; 3], g: [f64; 3], t: [f64; 3]) {
    let pc = eigphases(weyl_from_monodromy(c));
    let pg = eigphases(weyl_from_monodromy(g));
    let tw = weyl_from_monodromy(t);
    let pt = eigphases(tw);
    let ver = Verifier::new(pc, pg, pt);
    let sp = solve_sparse(&pc, &pg, &pt, &ver);
    eprintln!("[kf] sparse: {:?}", sp.map(|(_, e)| e));
    let mut seeds = dictionary(&pc, &pg, &pt, 40, 17);
    eprintln!("[kf] dictionary: {} seeds", seeds.len());
    seeds.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    if let Some(s0) = seeds.first() {
        eprintln!("[kf] best seed dist {:.3} l {:?}", s0.dist, s0.l);
        let fs = FixedSlice::new(&pc, &pg, &pt, s0.l, false);
        let mut rng = Rng::new(11);
        let gmat = M4::from_fn(|_, _| rng.normal());
        let ad = gmat.qr().q();
        // instrument the solve internals
        let mut b = [[M4::zeros(); 4]; 3];
        for j in 0..3 {
            for i in 0..4 {
                let mut acc = M4::zeros();
                for m in 0..4 {
                    acc += fs.a[j][m] * ad[(m, i)];
                }
                b[j][i] = acc;
            }
        }
        let mut d0 = DMatrix::<f64>::zeros(64, 64);
        let mut d1 = DMatrix::<f64>::zeros(64, 64);
        for (p, sg) in PERMS3 {
            kron3(
                &b[0][1 + p[0]],
                &b[1][1 + p[1]],
                &b[2][1 + p[2]],
                &mut d0,
                sg,
            );
            let mut pick = [1 + p[0], 1 + p[1], 1 + p[2]];
            let slot = p.iter().position(|&x| x == 0).unwrap();
            pick[slot] = 0;
            kron3(&b[0][pick[0]], &b[1][pick[1]], &b[2][pick[2]], &mut d1, -sg);
        }
        let evs = regular_eigs(&d1, &d0);
        let nreal = evs.iter().filter(|z| z.im.abs() < 1e-3).count();
        eprintln!(
            "[kf] regular_eigs: {} total, {} near-real",
            evs.len(),
            nreal
        );
        let got = fs.solve(&ad, s0.l, false, &ver);
        eprintln!("[kf] first-seed solve: {:?}", got.map(|(_, e)| e));
    }
}

pub fn solve_row(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<KfSolution> {
    let pc = eigphases(weyl_from_monodromy(c));
    let pg = eigphases(weyl_from_monodromy(g));
    let tw_direct = weyl_from_monodromy(t);
    let lifts = [tw_direct, rho_weyl(tw_direct)];
    // fixed dehomogenizations
    let mut rng = Rng::new(11);
    let gmat = M4::from_fn(|_, _| rng.normal());
    let ad = gmat.qr().q();
    let mut rng2 = Rng::new(23);
    let gmat2 = M4::from_fn(|_, _| rng2.normal());
    let ad2 = gmat2.qr().q();

    for tw in lifts {
        let pt = eigphases(tw);
        let ver = Verifier::new(pc, pg, pt);
        if let Some((o, err)) = solve_sparse(&pc, &pg, &pt, &ver) {
            return Some(KfSolution { o, spec_err: err });
        }
        if let Some((o, err)) = solve_deterministic(&pc, &pg, &pt, &ver) {
            return Some(KfSolution { o, spec_err: err });
        }
        // confluence: does any spectrum carry a collision?
        let collide = |ph: &[f64; 4]| {
            (0..4).any(|a| {
                (a + 1..4).any(|b| {
                    let d: C = C::from_polar(1.0, 2.0 * ph[a]) - C::from_polar(1.0, 2.0 * ph[b]);
                    d.norm() < 1e-6
                })
            })
        };
        let confluent = collide(&pc) || collide(&pg) || collide(&pt);
        let modes: &[usize] = if confluent { &[0, 1, 2] } else { &[0] };
        // stage 1: nearest-target dictionary seeds
        let mut seeds = dictionary(&pc, &pg, &pt, 40, 17);
        seeds.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        for s in seeds.iter().take(6) {
            for (q, fix_right) in [(s.l, false), (s.r, true)] {
                for &mode in modes {
                    let fs = FixedSlice::with_mode(&pc, &pg, &pt, q, fix_right, mode);
                    if let Some((o, err)) = fs.solve(&ad, q, fix_right, &ver) {
                        return Some(KfSolution { o, spec_err: err });
                    }
                }
            }
        }
        // stage 1b: medium dictionary
        let mut seeds2 = dictionary(&pc, &pg, &pt, 160, 19);
        seeds2.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        for s in seeds2.iter().take(10) {
            for (q, fix_right) in [(s.l, false), (s.r, true)] {
                let fs = FixedSlice::new(&pc, &pg, &pt, q, fix_right);
                if let Some((o, err)) = fs.solve(&ad, q, fix_right, &ver) {
                    return Some(KfSolution { o, spec_err: err });
                }
            }
        }
        // stage 2: farthest-point coverage escalation
        let mut pool = dictionary(&pc, &pg, &pt, 900, 71);
        let order = fps_order(&mut pool, 96);
        for &i in &order {
            let s = &pool[i];
            for (q, fix_right) in [(s.l, false), (s.r, true)] {
                let fs = FixedSlice::new(&pc, &pg, &pt, q, fix_right);
                if let Some((o, err)) = fs.solve(&ad2, q, fix_right, &ver) {
                    return Some(KfSolution { o, spec_err: err });
                }
            }
        }
    }
    None
}
