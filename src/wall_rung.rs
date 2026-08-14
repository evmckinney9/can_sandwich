//! Wall constructor -- PROTOTYPE. Not a benchmark claim.
//!
//! Ports exact_wall_row83073.py (chart algebra),
//! exact_wall_v2.py (interior Z-elimination via Chebyshev/Sylvester), and
//! exact_wall_v4_boundary.py (boundary chart for Birkhoff-boundary rows).
//!
//! Acceptance criterion: char-poly coefficients of
//!   M = diag(alpha) O diag(gamma) O^T
//! vs target polynomial -- NEVER sorted eigenvalues (sorted-phase comparison
//! inflates to the degeneracy gap on fold rows).

#![allow(non_snake_case)]

use std::f64::consts::PI;

type C = nalgebra::Complex<f64>;

// ---- polynomial helpers (descending power order = numpy convention) ------

fn polymul(p: &[f64], q: &[f64]) -> Vec<f64> {
    if p.is_empty() || q.is_empty() {
        return vec![];
    }
    let mut r = vec![0.0f64; p.len() + q.len() - 1];
    for (i, &pi) in p.iter().enumerate() {
        for (j, &qj) in q.iter().enumerate() {
            r[i + j] += pi * qj;
        }
    }
    r
}

fn polyval(p: &[f64], x: f64) -> f64 {
    p.iter().fold(0.0f64, |acc, &c| acc * x + c)
}

/// Trim near-zero leading coefficients (relative threshold).
fn poly_trim(p: &[f64]) -> &[f64] {
    let scale = p.iter().fold(0.0f64, |m, &v| m.max(v.abs())) + 1e-300;
    let i = p
        .iter()
        .position(|&v| v.abs() >= 1e-12 * scale)
        .unwrap_or(p.len().saturating_sub(1));
    &p[i..]
}

// ---- Gaussian-elimination LU determinant (in-place, row-major) -----------

fn lu_det(mat: &mut Vec<f64>, n: usize) -> f64 {
    let mut sign = 1.0f64;
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return 0.0;
        }
        if max_row != col {
            for k in 0..n {
                mat.swap(col * n + k, max_row * n + k);
            }
            sign = -sign;
        }
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for k in col..n {
                let v = factor * mat[col * n + k];
                mat[row * n + k] -= v;
            }
        }
    }
    sign * (0..n).map(|i| mat[i * n + i]).product::<f64>()
}

/// Resultant of p and q (descending order) via Sylvester matrix determinant.
pub fn sylvester_det(p_raw: &[f64], q_raw: &[f64]) -> f64 {
    let p = poly_trim(p_raw);
    let q = poly_trim(q_raw);
    let n = p.len().saturating_sub(1);
    let m = q.len().saturating_sub(1);
    if n < 1 || m < 1 {
        return 0.0;
    }
    let sz = n + m;
    let mut s = vec![0.0f64; sz * sz];
    for i in 0..m {
        for (j, &v) in p.iter().enumerate() {
            s[i * sz + i + j] = v;
        }
    }
    for i in 0..n {
        for (j, &v) in q.iter().enumerate() {
            s[(m + i) * sz + i + j] = v;
        }
    }
    lu_det(&mut s, sz)
}

// ---- Chebyshev roots via colleague matrix --------------------------------

/// Chebyshev-1 nodes on [-1,1] in descending order (matching numpy chebpts1).
/// x_j = cos(π(2j+1)/(2N)) for j=0..N-1.
fn chebpts1(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (PI * (2.0 * j as f64 + 1.0) / (2.0 * n as f64)).cos())
        .collect()
}

/// DCT-II to obtain ascending-order Chebyshev coefficients.
/// nodes = chebpts1(N), y = function values at those nodes.
fn cheb_fit_asc(nodes: &[f64], y: &[f64]) -> Vec<f64> {
    let n = nodes.len();
    assert_eq!(n, y.len());
    let n_f = n as f64;
    // For each node x_j = cos(θ_j) with θ_j = π(2j+1)/(2N),
    // cc[k] = (2/N) Σ_j y[j] cos(k θ_j), then cc[0] /= 2.
    let mut cc = vec![0.0f64; n];
    for k in 0..n {
        let kf = k as f64;
        let s: f64 = (0..n)
            .map(|j| {
                let theta = PI * (2.0 * j as f64 + 1.0) / (2.0 * n_f);
                y[j] * (kf * theta).cos()
            })
            .sum();
        cc[k] = 2.0 / n_f * s;
    }
    cc[0] *= 0.5;
    cc
}

/// Roots of a Chebyshev series (ascending order, cc[k] = coeff of T_k)
/// via the colleague matrix. Matches axis_quartic.rs colleague_roots.
fn colleague_roots(cc: &[f64]) -> Vec<C> {
    let scale = cc.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1e-300);
    let mut n = cc.len();
    while n > 1 && cc[n - 1].abs() < 1e-12 * scale {
        n -= 1;
    }
    if n < 2 {
        return vec![];
    }
    let deg = n - 1;
    let lead = cc[deg];
    let a = faer::Mat::<C>::from_fn(deg, deg, |i, j| {
        let mut v = 0.0f64;
        if i == 0 && j == 1 {
            v = 1.0;
        } else if i > 0 && (j + 1 == i || j == i + 1) {
            v = 0.5;
        }
        if i == deg - 1 {
            v += -cc[j] / (2.0 * lead);
        }
        C::new(v, 0.0)
    });
    match a.eigenvalues() {
        Ok(ev) => (0..deg).map(|i| ev[i]).collect(),
        Err(_) => vec![],
    }
}

/// Real roots of a polynomial (descending order) via companion matrix.
fn poly_roots_real(p_raw: &[f64], im_tol: f64) -> Vec<f64> {
    let p = poly_trim(p_raw);
    let deg = p.len().saturating_sub(1);
    if deg < 1 {
        return vec![];
    }
    let lead = p[0];
    if lead.abs() < 1e-300 {
        return vec![];
    }
    // Frobenius companion: subdiagonal = 1, last col = -p[deg-j]/lead
    let mat = faer::Mat::<C>::from_fn(deg, deg, |row, col| {
        let mut v = 0.0f64;
        if col == deg - 1 {
            v = -p[deg - row] / lead;
        } else if row == col + 1 {
            v = 1.0;
        }
        C::new(v, 0.0)
    });
    match mat.eigenvalues() {
        Ok(ev) => (0..deg)
            .filter_map(|k| {
                if ev[k].im.abs() <= im_tol {
                    Some(ev[k].re)
                } else {
                    None
                }
            })
            .collect(),
        Err(_) => vec![],
    }
}

// ---- Weyl / spectrum -----------------------------------------------------

fn weyl(m: [f64; 3]) -> [f64; 3] {
    [m[0] + m[1], m[0] + m[2], m[1] + m[2]]
}

fn eigph(v: [f64; 3]) -> [f64; 4] {
    let h = PI / 2.0;
    [
        h * (v[0] - v[1] + v[2]),
        h * (v[0] + v[1] - v[2]),
        h * (-v[0] - v[1] - v[2]),
        h * (-v[0] + v[1] + v[2]),
    ]
}

/// Returns (c, g, wp, wm) as [C; 4] each.
/// triple = [mC[0..3], mG[0..3], mT[0..3]].
pub fn row_spectra(triple: &[f64; 9]) -> ([C; 4], [C; 4], [C; 4], [C; 4]) {
    let mc = [triple[0], triple[1], triple[2]];
    let mg = [triple[3], triple[4], triple[5]];
    let mt = [triple[6], triple[7], triple[8]];

    let ec = eigph(weyl(mc));
    let eg = eigph(weyl(mg));
    let et = eigph(weyl(mt));

    let c: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, -2.0 * ec[k]));
    let g: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * eg[k]));
    let wp: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * et[k]));
    let wm: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, -2.0 * et[k]));

    (c, g, wp, wm)
}

/// Elementary symmetric polynomials e1, e2 of w (for the linear chart).
fn char_coeffs(w: &[C; 4]) -> (C, C) {
    let e1: C = w.iter().copied().sum();
    let e2: C = (0..4)
        .flat_map(|i| ((i + 1)..4).map(move |j| w[i] * w[j]))
        .sum();
    (e1, e2)
}

// ---- Branch linear coefficients ------------------------------------------

/// Compute the 3-equation linear system coefficients for fixed (alpha, gamma, z0).
/// Returns (c1_0, c1_c[6], c2_0, c2_c[6], rho) in complex form.
fn branch_linear_coeffs(alpha: &[C; 4], gamma: &[C; 4], z0: f64) -> ([C; 6], C, [C; 6], C, C) {
    // z0 plays the role of the chart parameter (different from capital Z)
    let a = (alpha[0] + alpha[1] + (alpha[0] - alpha[1]) * z0) / 2.0;
    let b = (alpha[0] + alpha[1] - (alpha[0] - alpha[1]) * z0) / 2.0;
    let e = (gamma[0] + gamma[1] + (gamma[0] - gamma[1]) * z0) / 2.0;
    let f = (gamma[0] + gamma[1] - (gamma[0] - gamma[1]) * z0) / 2.0;
    let uv = (alpha[0] - alpha[1]) * (gamma[0] - gamma[1]) * C::new(1.0 - z0 * z0, 0.0) / 4.0;

    let avec = [b, alpha[2], alpha[3]];
    let gvec = [f, gamma[2], gamma[3]];
    let ap = [alpha[0] * alpha[1], a * alpha[2], a * alpha[3]];
    let am = [alpha[2] * alpha[3], alpha[3] * b, b * alpha[2]];
    let gp = [gamma[0] * gamma[1], e * gamma[2], e * gamma[3]];
    let gm = [gamma[2] * gamma[3], gamma[3] * f, f * gamma[2]];

    // B0, dBdX, dBdY, dBdZ, dBdW (real 3x3 matrices, flattened row-major)
    #[rustfmt::skip]
    let b0:   [f64; 9] = [0., 0., 1.,  0., 0., 1.,  1., 1.,-1.];
    #[rustfmt::skip]
    let dbdx: [f64; 9] = [1., 0.,-1.,  0., 0., 0., -1., 0., 1.];
    #[rustfmt::skip]
    let dbdy: [f64; 9] = [0., 1.,-1.,  0., 0., 0.,  0.,-1., 1.];
    #[rustfmt::skip]
    let dbdz: [f64; 9] = [0., 0., 0.,  1., 0.,-1., -1., 0., 1.];
    #[rustfmt::skip]
    let dbdw: [f64; 9] = [0., 0., 0.,  0., 1.,-1.,  0.,-1., 1.];

    // outer product contraction: sum_ij (avec_i * gvec_j) * mat_ij
    let dot2d_agg = |mat: &[f64; 9]| -> C {
        let mut s = C::new(0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                s += avec[i] * gvec[j] * mat[i * 3 + j];
            }
        }
        s
    };
    let dot2d_apgp = |mat: &[f64; 9]| -> C {
        let mut s = C::new(0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                s += (ap[i] * gp[j] + am[i] * gm[j]) * mat[i * 3 + j];
            }
        }
        s
    };

    // C1 at X=Y=Z=W=r=s=0
    let c1_0: C = a * e + (0..9).map(|k| avec[k / 3] * gvec[k % 3] * b0[k]).sum::<C>();
    // C1 coefficients [X, Y, Z, W, r, s]
    let c1_c: [C; 6] = [
        dot2d_agg(&dbdx),
        dot2d_agg(&dbdy),
        dot2d_agg(&dbdz),
        dot2d_agg(&dbdw),
        uv * 2.0,
        C::new(0.0, 0.0),
    ];

    // C2 at X=Y=Z=W=r=s=0
    let c2_0: C = (0..9)
        .map(|k| (ap[k / 3] * gp[k % 3] + am[k / 3] * gm[k % 3]) * b0[k])
        .sum::<C>();
    // C2 coefficients [X, Y, Z, W, r, s]
    let c2_c: [C; 6] = [
        dot2d_apgp(&dbdx),
        dot2d_apgp(&dbdy),
        dot2d_apgp(&dbdz),
        dot2d_apgp(&dbdw),
        uv * 2.0 * (alpha[2] * gamma[2] + alpha[3] * gamma[3]),
        uv * 2.0 * (alpha[2] - alpha[3]) * (gamma[2] - gamma[3]),
    ];

    let rho: C = (alpha.iter().product::<C>() * gamma.iter().product::<C>()).sqrt();

    (c1_c, c1_0, c2_c, c2_0, rho)
}

/// Returns (A_WRS, b_WRS) where [W, r, s] = A_WRS @ [X, Y, Z] + b_WRS.
/// Returns None if pivot matrix is singular.
fn linear_chart(
    alpha: &[C; 4],
    gamma: &[C; 4],
    tc: (C, C),
    z0: f64,
) -> Option<([[f64; 3]; 3], [f64; 3])> {
    let (c1_c, c1_0, c2_c, c2_0, rho) = branch_linear_coeffs(alpha, gamma, z0);
    let (t1, t2) = tc;

    let c1_adj = c1_0 - t1;
    let c2_adj = c2_0 - t2;

    // Build real 3x6 matrix M and constant vector v
    // Row 0: Re(C1 - t1) = 0
    // Row 1: Im(C1 - t1) = 0
    // Row 2: Re((C2 - t2)/rho) = 0
    let row0: [f64; 6] = std::array::from_fn(|j| c1_c[j].re);
    let row1: [f64; 6] = std::array::from_fn(|j| c1_c[j].im);
    let row2: [f64; 6] = std::array::from_fn(|j| (c2_c[j] / rho).re);

    let v = [c1_adj.re, c1_adj.im, (c2_adj / rho).re];

    // A_pivot = M[:, 3:6] (columns W, r, s)
    let ap = nalgebra::Matrix3::from_fn(|i, j| [row0, row1, row2][i][j + 3]);
    let det = ap.determinant();
    if det.abs() < 1e-12 {
        return None;
    }
    let ainv = ap.try_inverse()?;

    // A_free = M[:, 0:3] (columns X, Y, Z)
    let af = nalgebra::Matrix3::from_fn(|i, j| [row0, row1, row2][i][j]);
    let vv = nalgebra::Vector3::from(v);

    let a_wrs_mat = -ainv * af;
    let b_wrs_vec = -ainv * vv;

    let a_wrs: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| a_wrs_mat[(i, j)]));
    let b_wrs: [f64; 3] = std::array::from_fn(|i| b_wrs_vec[i]);

    Some((a_wrs, b_wrs))
}

// ---- Quadric coefficient extraction at fixed (z0_chart, Z0_capital) ------

/// Returns polynomial-in-Y coefficients for Q1, Q2, Q3 at fixed Z0.
/// Each Qi = (a_i: f64, b_i: [f64;2], c_i: [f64;3]) where
///   b_i(Y) = b_i[0]*Y + b_i[1]  (high-degree first)
///   c_i(Y) = c_i[0]*Y^2 + c_i[1]*Y + c_i[2]
type Quadric = (f64, [f64; 2], [f64; 3]);

fn poly_coeffs(a_wrs: &[[f64; 3]; 3], b_wrs: &[f64; 3], z0: f64) -> (Quadric, Quadric, Quadric) {
    // r = p_r*X + q_r*Y + c0_r
    let p_r = a_wrs[1][0];
    let q_r = a_wrs[1][1];
    let c0_r = a_wrs[1][2] * z0 + b_wrs[1];
    // s = p_s*X + q_s*Y + c0_s
    let p_s = a_wrs[2][0];
    let q_s = a_wrs[2][1];
    let c0_s = a_wrs[2][2] * z0 + b_wrs[2];
    // W = p_w*X + q_w*Y + c0_w
    let p_w = a_wrs[0][0];
    let q_w = a_wrs[0][1];
    let c0_w = a_wrs[0][2] * z0 + b_wrs[0];

    // Q1 = r^2 - X
    let a1 = p_r * p_r;
    let b1 = [2.0 * p_r * q_r, 2.0 * p_r * c0_r - 1.0];
    let c1 = [q_r * q_r, 2.0 * q_r * c0_r, c0_r * c0_r];

    // Q2 = s^2 - (1-Z0-W)*(1-Y-W)
    let a_s = 1.0 - z0 - c0_w;
    let b_s = 1.0 - c0_w;
    let a2 = p_s * p_s - p_w * p_w;
    let b2 = [
        2.0 * p_s * q_s - p_w * (1.0 + 2.0 * q_w),
        2.0 * p_s * c0_s + p_w * (a_s + b_s),
    ];
    let c2 = [
        q_s * q_s - q_w * (1.0 + q_w),
        2.0 * q_s * c0_s + a_s * (1.0 + q_w) + b_s * q_w,
        c0_s * c0_s - a_s * b_s,
    ];

    // Q3 = (r+s)^2 - W*(X+Y+Z0+W-1)
    let prs = p_r + p_s;
    let qrs = q_r + q_s;
    let c0rs = c0_r + c0_s;
    let w_c = z0 + c0_w - 1.0;
    let a3 = prs * prs - p_w * (1.0 + p_w);
    let b3 = [
        2.0 * prs * qrs - (p_w + q_w + 2.0 * p_w * q_w),
        2.0 * prs * c0rs - p_w * w_c - c0_w * (1.0 + p_w),
    ];
    let c3 = [
        qrs * qrs - q_w * (1.0 + q_w),
        2.0 * qrs * c0rs - q_w * w_c - c0_w * (1.0 + q_w),
        c0rs * c0rs - c0_w * w_c,
    ];

    ((a1, b1, c1), (a2, b2, c2), (a3, b3, c3))
}

fn eval_quadric(q: &Quadric, x: f64, y: f64) -> f64 {
    let (a, b, c) = q;
    a * x * x + polyval(b, y) * x + polyval(c, y)
}

// ---- X-elimination from Q1=0, Q2=0 -> degree-4 polynomial in Y ----------

/// Returns R(Y) (degree ≤ 4) and auxiliary (num, den) polys.
fn elim_x_poly(q1: &Quadric, q2: &Quadric) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (a1, b1, c1) = q1;
    let (a2, b2, c2) = q2;

    // num = a2*c1 - a1*c2  (degree 2 in Y)
    let num: Vec<f64> = c1
        .iter()
        .zip(c2.iter())
        .map(|(&x, &y)| a2 * x - a1 * y)
        .collect();
    // den = a1*b2 - a2*b1  (degree 1 in Y)
    let den: Vec<f64> = b1
        .iter()
        .zip(b2.iter())
        .map(|(&x, &y)| a1 * y - a2 * x)
        .collect();

    // R = a1*num^2 + b1*num*den + c1*den^2  (degree 4)
    let num_sq = polymul(&num, &num);
    let b1_num = polymul(b1, &num);
    let b1_nd = polymul(&b1_num, &den);
    let den_sq = polymul(&den, &den);
    let c1_dsq = polymul(c1, &den_sq);

    let n = num_sq.len().max(b1_nd.len()).max(c1_dsq.len());
    let mut r = vec![0.0f64; n];
    for (i, &v) in num_sq.iter().enumerate() {
        r[n - num_sq.len() + i] += a1 * v;
    }
    for (i, &v) in b1_nd.iter().enumerate() {
        r[n - b1_nd.len() + i] += v;
    }
    for (i, &v) in c1_dsq.iter().enumerate() {
        r[n - c1_dsq.len() + i] += v;
    }

    (r, num, den)
}

// ---- SO(3) sign lift ------------------------------------------------------

/// Enumerate 64 sign combos for first two columns of R ∈ SO(3).
/// Filter: col0·col1=0; col2=col0×col1; det=+1; R[0,0]≈r_target; R[1,2]*R[2,1]≈s_target.
fn so3_sign_lift(
    b_mat: &[[f64; 3]; 3],
    r_target: f64,
    s_target: f64,
    tol: f64,
) -> Vec<[[f64; 3]; 3]> {
    let sqb: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| b_mat[i][j].max(0.0).sqrt()));
    let mut results = Vec::new();

    for bits in 0u8..64 {
        let signs: [f64; 6] =
            std::array::from_fn(|k| if (bits >> k) & 1 == 1 { 1.0 } else { -1.0 });
        let col0 = [
            signs[0] * sqb[0][0],
            signs[1] * sqb[1][0],
            signs[2] * sqb[2][0],
        ];
        let col1 = [
            signs[3] * sqb[0][1],
            signs[4] * sqb[1][1],
            signs[5] * sqb[2][1],
        ];
        // orthogonality check
        let dot01 = col0[0] * col1[0] + col0[1] * col1[1] + col0[2] * col1[2];
        if dot01.abs() > tol {
            continue;
        }
        // col2 = col0 × col1
        let col2 = [
            col0[1] * col1[2] - col0[2] * col1[1],
            col0[2] * col1[0] - col0[0] * col1[2],
            col0[0] * col1[1] - col0[1] * col1[0],
        ];
        // Check col2^2 ≈ B[:,2]
        let ok = (0..3).all(|i| (col2[i] * col2[i] - b_mat[i][2]).abs() <= tol);
        if !ok {
            continue;
        }
        // Assemble R (columns 0,1,2)
        let r = [
            [col0[0], col1[0], col2[0]],
            [col0[1], col1[1], col2[1]],
            [col0[2], col1[2], col2[2]],
        ];
        // det = +1
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        if (det - 1.0).abs() > tol {
            continue;
        }
        // r_target = R[0,0] = col0[0]
        if (r[0][0] - r_target).abs() > tol {
            continue;
        }
        // s_target = R[1,2]*R[2,1] = col2[1]*col1[2]
        if (r[1][2] * r[2][1] - s_target).abs() > tol {
            continue;
        }
        results.push(r);
    }
    results
}

// ---- SO(4) reconstruction ------------------------------------------------

/// Build O for both signs of d.
/// conj = [[c,-d,0,0],[d,c,0,0],[0,0,1,0],[0,0,0,1]], block = [[1,0;0,R]].
/// O = (conj^T @ block) @ conj; flip col 0 if det < 0.
fn reconstruct_so4(z0: f64, r: &[[f64; 3]; 3]) -> Vec<[[f64; 4]; 4]> {
    let c_val = ((1.0 + z0) / 2.0).max(0.0).sqrt();
    let mut cands = Vec::new();

    for d_sign in [1.0f64, -1.0] {
        let d_val = d_sign * ((1.0 - z0) / 2.0).max(0.0).sqrt();

        // CB = conj^T @ block (4x4)
        // Row 0: [c, d*R[0,0], d*R[0,1], d*R[0,2]]
        // Row 1: [-d, c*R[0,0], c*R[0,1], c*R[0,2]]
        // Row 2: [0, R[1,0], R[1,1], R[1,2]]
        // Row 3: [0, R[2,0], R[2,1], R[2,2]]
        let cb: [[f64; 4]; 4] = [
            [c_val, d_val * r[0][0], d_val * r[0][1], d_val * r[0][2]],
            [-d_val, c_val * r[0][0], c_val * r[0][1], c_val * r[0][2]],
            [0.0, r[1][0], r[1][1], r[1][2]],
            [0.0, r[2][0], r[2][1], r[2][2]],
        ];

        // conj = [[c,-d,0,0],[d,c,0,0],[0,0,1,0],[0,0,0,1]]
        // O = CB @ conj
        let mut o = [[0.0f64; 4]; 4];
        for i in 0..4 {
            o[i][0] = cb[i][0] * c_val + cb[i][1] * d_val;
            o[i][1] = cb[i][0] * (-d_val) + cb[i][1] * c_val;
            o[i][2] = cb[i][2];
            o[i][3] = cb[i][3];
        }

        // Fix det if negative: flip column 0
        let det = mat4_det(&o);
        if det < 0.0 {
            for i in 0..4 {
                o[i][0] = -o[i][0];
            }
        }
        cands.push(o);
    }
    cands
}

fn mat4_det(o: &[[f64; 4]; 4]) -> f64 {
    // Laplace expansion along row 0
    let m = |r: usize, c: usize| o[r][c];
    let minor3 = |r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize| -> f64 {
        m(r0, c0) * (m(r1, c1) * m(r2, c2) - m(r1, c2) * m(r2, c1))
            - m(r0, c1) * (m(r1, c0) * m(r2, c2) - m(r1, c2) * m(r2, c0))
            + m(r0, c2) * (m(r1, c0) * m(r2, c1) - m(r1, c1) * m(r2, c0))
    };
    m(0, 0) * minor3(1, 2, 3, 1, 2, 3) - m(0, 1) * minor3(1, 2, 3, 0, 2, 3)
        + m(0, 2) * minor3(1, 2, 3, 0, 1, 3)
        - m(0, 3) * minor3(1, 2, 3, 0, 1, 2)
}

// ---- Smooth invariant check (char-poly coefficients, never sorted eigvals) -

/// Char-poly residual = max|e_k(M) - e_k(w)| for k=1,2,3
/// where e_k are elementary symmetric polynomials of the eigenvalues.
/// M = diag(alpha) O diag(gamma) O^T (complex 4×4).
pub fn char_poly_residual(alpha: &[C; 4], gamma: &[C; 4], o: &[[f64; 4]; 4], w: &[C; 4]) -> f64 {
    // Build M as 4x4 complex matrix
    // M[i,j] = alpha[i] * sum_k (O[i,k] * gamma[k] * O[j,k])
    let m = faer::Mat::<C>::from_fn(4, 4, |i, j| {
        let s: C = (0..4)
            .map(|k| C::new(o[i][k], 0.0) * gamma[k] * C::new(o[j][k], 0.0))
            .sum();
        alpha[i] * s
    });

    // Collect eigenvalues into a fixed array to avoid closure-move issues.
    let em: [C; 4] = match m.eigenvalues() {
        Ok(ev) => std::array::from_fn(|i| ev[i]),
        Err(_) => return f64::INFINITY,
    };

    // Elementary symmetric polys via explicit loops (no nested closures).
    fn esym(e: &[C; 4]) -> (C, C, C) {
        let mut e1 = C::new(0.0, 0.0);
        let mut e2 = C::new(0.0, 0.0);
        let mut e3 = C::new(0.0, 0.0);
        for i in 0..4 {
            e1 += e[i];
            for j in (i + 1)..4 {
                e2 += e[i] * e[j];
                for k in (j + 1)..4 {
                    e3 += e[i] * e[j] * e[k];
                }
            }
        }
        (e1, e2, e3)
    }

    let (e1_m, e2_m, e3_m) = esym(&em);
    let (e1_w, e2_w, e3_w) = esym(w);

    // max|(e_k_M - e_k_w)| for k=1,2,3
    (e1_m - e1_w)
        .norm()
        .max((e2_m - e2_w).norm())
        .max((e3_m - e3_w).norm())
}

// ---- Alpha orderings (12) and gamma permutations (24) --------------------

pub const ALPHA_ORDERINGS: [[usize; 4]; 12] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
];

fn all_gamma_perms() -> Vec<[usize; 4]> {
    let mut perms = Vec::new();
    let mut a = [0, 1, 2, 3usize];
    // Heap's algorithm
    let mut c = [0usize; 4];
    perms.push(a);
    let mut i = 0;
    while i < 4 {
        if c[i] < i {
            if i & 1 == 0 {
                a.swap(0, i);
            } else {
                a.swap(c[i], i);
            }
            perms.push(a);
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    perms
}

// ---- Interior solver constants -------------------------------------------

const NS: usize = 64;
const Z_LO: f64 = 1e-6;
const Z_HI: f64 = 1.0 - 1e-6;

pub const Z_VALS: [f64; 9] = [0.0, 0.35, -0.35, 0.65, -0.65, 0.15, -0.15, 0.85, -0.85];

/// Try to construct O for fixed (alpha, gamma, w, z0) via interior Z-elimination.
/// Returns char-poly residual on success, None on miss.
fn solve_interior_branch(
    alpha: &[C; 4],
    gamma: &[C; 4],
    w: &[C; 4],
    z0: f64,
    tol: f64,
) -> Option<(f64, [[f64; 4]; 4])> {
    let tc = char_coeffs(w);
    let (a_wrs, b_wrs) = linear_chart(alpha, gamma, tc, z0)?;

    // Sample E(Z) at 64 Chebyshev-1 nodes on [Z_LO, Z_HI]
    let nodes = chebpts1(NS);
    let mut es = vec![0.0f64; NS];
    for (k, &t) in nodes.iter().enumerate() {
        let zk = 0.5 * (Z_HI + Z_LO) + 0.5 * (Z_HI - Z_LO) * t;
        let (q1, q2, q3) = poly_coeffs(&a_wrs, &b_wrs, zk);
        let (r12, _, _) = elim_x_poly(&q1, &q2);
        let (r13, _, _) = elim_x_poly(&q1, &q3);
        es[k] = sylvester_det(&r12, &r13);
    }

    let scale = es.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale < 1e-300 {
        return None;
    }
    let es_scaled: Vec<f64> = es.iter().map(|&e| e / scale).collect();

    // Chebyshev fit (ascending-order coefficients)
    let mut cc = cheb_fit_asc(&nodes, &es_scaled);
    // Trim trailing (high-degree) near-zero coefficients
    while cc.len() > 1 && cc.last().copied().unwrap_or(0.0).abs() < 1e-12 {
        cc.pop();
    }
    if cc.len() < 2 {
        return None;
    }

    let roots = colleague_roots(&cc);

    for rt in roots {
        if rt.im.abs() > 1e-7 || rt.re < -1.001 || rt.re > 1.001 {
            continue;
        }
        let z_cap = 0.5 * (Z_HI + Z_LO) + 0.5 * (Z_HI - Z_LO) * rt.re;
        if !(0.0..=1.0).contains(&z_cap) {
            continue;
        }

        let (q1, q2, q3) = poly_coeffs(&a_wrs, &b_wrs, z_cap);
        let (r12, _, _) = elim_x_poly(&q1, &q2);
        let r12t = poly_trim(&r12);
        if r12t.len() < 2 {
            continue;
        }

        let y_roots = poly_roots_real(r12t, 1e-7);
        for y0 in y_roots {
            if y0 < -1e-9 || y0 > 1.0 + 1e-9 {
                continue;
            }

            // X via Q1 discriminant
            let (a1, b1, c1) = q1;
            let b1v = polyval(&b1, y0);
            let c1v = polyval(&c1, y0);
            let disc = b1v * b1v - 4.0 * a1 * c1v;
            if disc < -1e-8 {
                continue;
            }
            let sq = disc.max(0.0).sqrt();
            for &x0 in &[(-b1v + sq) / (2.0 * a1), (-b1v - sq) / (2.0 * a1)] {
                if x0 < -1e-9 || x0 > 1.0 + 1e-9 {
                    continue;
                }
                // Check Q2 and Q3
                let q2v = eval_quadric(&q2, x0, y0).abs();
                let q3v = eval_quadric(&q3, x0, y0).abs();
                if q2v.max(q3v) > 1e-7 {
                    continue;
                }

                let w0 = a_wrs[0][0] * x0 + a_wrs[0][1] * y0 + a_wrs[0][2] * z_cap + b_wrs[0];
                let r0 = a_wrs[1][0] * x0 + a_wrs[1][1] * y0 + a_wrs[1][2] * z_cap + b_wrs[1];
                let s0 = a_wrs[2][0] * x0 + a_wrs[2][1] * y0 + a_wrs[2][2] * z_cap + b_wrs[2];

                let b_birk = [
                    [x0, y0, 1.0 - x0 - y0],
                    [z_cap, w0, 1.0 - z_cap - w0],
                    [1.0 - x0 - z_cap, 1.0 - y0 - w0, x0 + y0 + z_cap + w0 - 1.0],
                ];
                if b_birk.iter().flatten().any(|&v| v < -1e-8) {
                    continue;
                }

                for r_mat in so3_sign_lift(&b_birk, r0, s0, 1e-5) {
                    for o in reconstruct_so4(z0, &r_mat) {
                        let resid = char_poly_residual(alpha, gamma, &o, w);
                        if resid < tol {
                            return Some((resid, o));
                        }
                    }
                }
            }
        }
    }
    None
}

// ---- Boundary bivariate quadric algebra ----------------------------------

/// Affine function of two free variables (u, v): val = u_coef*u + v_coef*v + c
#[derive(Clone, Copy)]
struct AffFn {
    u: f64,
    v: f64,
    c: f64,
}

/// Bivariate quadratic: sum_{keys} coef * monomial
#[derive(Clone, Default)]
struct BivQuad {
    uu: f64,
    vv: f64,
    uv: f64,
    u: f64,
    v: f64,
    c: f64,
}

fn biv_mul(a: AffFn, b: AffFn) -> BivQuad {
    BivQuad {
        uu: a.u * b.u,
        vv: a.v * b.v,
        uv: a.u * b.v + a.v * b.u,
        u: a.u * b.c + a.c * b.u,
        v: a.v * b.c + a.c * b.v,
        c: a.c * b.c,
    }
}

fn aff_add(a: AffFn, b: AffFn, c0: f64) -> AffFn {
    AffFn {
        u: a.u + b.u,
        v: a.v + b.v,
        c: a.c + b.c + c0,
    }
}

fn aff_scale(a: AffFn, k: f64) -> AffFn {
    AffFn {
        u: k * a.u,
        v: k * a.v,
        c: k * a.c,
    }
}

fn biv_sub(a: &BivQuad, b: &BivQuad) -> BivQuad {
    BivQuad {
        uu: a.uu - b.uu,
        vv: a.vv - b.vv,
        uv: a.uv - b.uv,
        u: a.u - b.u,
        v: a.v - b.v,
        c: a.c - b.c,
    }
}

/// Convert BivQuad to Quadric treating u as X:
/// Q = uu*u^2 + (uv*v + u_coef)*u + (vv*v^2 + v_coef*v + c)
fn biv_as_quadric(q: &BivQuad) -> Quadric {
    (q.uu, [q.uv, q.u], [q.vv, q.v, q.c])
}

fn biv_eval(q: &BivQuad, u: f64, v: f64) -> f64 {
    q.uu * u * u + q.vv * v * v + q.uv * u * v + q.u * u + q.v * v + q.c
}

// Birkhoff boundary entries: 9 (i,j) pairs with ([coef; 6], const)
// coef dot (X,Y,Z,W,r,s) + const = 0 means B[i][j] = 0
const BIRKHOFF: [([f64; 6], f64); 9] = [
    ([1., 0., 0., 0., 0., 0.], 0.),   // B[0,0] = X = 0
    ([0., 1., 0., 0., 0., 0.], 0.),   // B[0,1] = Y = 0
    ([-1., -1., 0., 0., 0., 0.], 1.), // B[0,2] = 1-X-Y = 0
    ([0., 0., 1., 0., 0., 0.], 0.),   // B[1,0] = Z = 0
    ([0., 0., 0., 1., 0., 0.], 0.),   // B[1,1] = W = 0
    ([0., 0., -1., -1., 0., 0.], 1.), // B[1,2] = 1-Z-W = 0
    ([-1., 0., -1., 0., 0., 0.], 1.), // B[2,0] = 1-X-Z = 0
    ([0., -1., 0., -1., 0., 0.], 1.), // B[2,1] = 1-Y-W = 0
    ([1., 1., 1., 1., 0., 0.], -1.),  // B[2,2] = X+Y+Z+W-1 = 0
];

// ---- Boundary branch construction ----------------------------------------

const Z_A: f64 = -0.999;
const Z_B: f64 = 0.999;

/// Build (R12, R13, back_fn_data) for fixed z0 and Birkhoff entry.
/// Returns None if the 4x4 pivot is singular.
/// back_fn_data = (P6: 6x2, h6: [f64;6], Q1, Q2, Q3 as BivQuad) for back-sub.
fn boundary_branch_polys(
    alpha: &[C; 4],
    gamma: &[C; 4],
    tc: (C, C),
    z0: f64,
    bentry: usize,
) -> Option<(
    Vec<f64>,
    Vec<f64>,
    [[f64; 2]; 6],
    [f64; 6],
    BivQuad,
    BivQuad,
    BivQuad,
)> {
    let (c1_c, c1_0, c2_c, c2_0, rho) = branch_linear_coeffs(alpha, gamma, z0);
    let (t1, t2) = tc;
    let c1_adj = c1_0 - t1;
    let c2_adj = c2_0 - t2;

    let row0: [f64; 6] = std::array::from_fn(|j| c1_c[j].re);
    let row1: [f64; 6] = std::array::from_fn(|j| c1_c[j].im);
    let row2: [f64; 6] = std::array::from_fn(|j| (c2_c[j] / rho).re);
    let (brow, bconst) = BIRKHOFF[bentry];
    let row3: [f64; 6] = brow;

    let m4 = [row0, row1, row2, row3];
    let v4 = [c1_adj.re, c1_adj.im, (c2_adj / rho).re, bconst];

    // A_piv = M[:, 2:6] (4x4), free vars: u=col0 (X), v=col1 (Y)
    let ap = nalgebra::Matrix4::from_fn(|i, j| m4[i][j + 2]);
    if ap.determinant().abs() < 1e-14 {
        return None;
    }
    let ainv = ap.try_inverse()?;
    let vv = nalgebra::Vector4::from(v4);
    let af = nalgebra::Matrix4x2::from_fn(|i, j| m4[i][j]);

    // (Z,W,r,s) = -ainv @ af @ (u,v) - ainv @ v4
    let p_free = -ainv * af; // 4x2
    let h_free = -ainv * vv; // 4

    // Full 6x2 map: y6 = P6 @ (u,v) + h6
    // y6 = [X, Y, Z, W, r, s], X=u (row 0 = [1,0]), Y=v (row 1 = [0,1])
    let mut p6 = [[0.0f64; 2]; 6];
    let mut h6 = [0.0f64; 6];
    p6[0] = [1.0, 0.0];
    p6[1] = [0.0, 1.0];
    h6[0] = 0.0;
    h6[1] = 0.0;
    for i in 0..4 {
        p6[i + 2] = [p_free[(i, 0)], p_free[(i, 1)]];
        h6[i + 2] = h_free[i];
    }

    // Affine functions of (u, v) for each y6 slot
    let aff = |slot: usize| AffFn {
        u: p6[slot][0],
        v: p6[slot][1],
        c: h6[slot],
    };
    let x_ = AffFn {
        u: 1.0,
        v: 0.0,
        c: 0.0,
    };
    let y_ = AffFn {
        u: 0.0,
        v: 1.0,
        c: 0.0,
    };
    let z_ = aff(2);
    let w_ = aff(3);
    let r_ = aff(4);
    let s_ = aff(5);
    let one = AffFn {
        u: 0.0,
        v: 0.0,
        c: 1.0,
    };

    // Q1 = r^2 - X*1
    let q1 = biv_sub(&biv_mul(r_, r_), &biv_mul(x_, one));
    // Q2 = s^2 - (1-Z-W)*(1-Y-W)
    let ta = aff_add(aff_scale(z_, -1.0), aff_scale(w_, -1.0), 1.0);
    let tb = aff_add(aff_scale(y_, -1.0), aff_scale(w_, -1.0), 1.0);
    let q2 = biv_sub(&biv_mul(s_, s_), &biv_mul(ta, tb));
    // Q3 = (r+s)^2 - W*(X+Y+Z+W-1)
    let rs = aff_add(r_, s_, 0.0);
    let tc_sum = aff_add(aff_add(x_, y_, 0.0), aff_add(z_, w_, -1.0), 0.0);
    let q3 = biv_sub(&biv_mul(rs, rs), &biv_mul(w_, tc_sum));

    // Convert to Quadric (u as X), eliminate u from Q1,Q2 and Q1,Q3
    let uq1 = biv_as_quadric(&q1);
    let uq2 = biv_as_quadric(&q2);
    let uq3 = biv_as_quadric(&q3);

    let (r12, _, _) = elim_x_poly(&uq1, &uq2);
    let (r13, _, _) = elim_x_poly(&uq1, &uq3);

    Some((r12, r13, p6, h6, q1, q2, q3))
}

/// Interior boundary solver for one (alpha, gamma, w, Birkhoff entry).
fn solve_boundary_branch(
    alpha: &[C; 4],
    gamma: &[C; 4],
    w: &[C; 4],
    bentry: usize,
    tol: f64,
) -> Option<f64> {
    let tc = char_coeffs(w);

    // Sample E(z) at 64 Chebyshev-1 nodes on [Z_A, Z_B]
    let nodes = chebpts1(NS);
    let mut es = vec![0.0f64; NS];
    let mut any_valid = false;
    for (k, &t) in nodes.iter().enumerate() {
        let zk = 0.5 * (Z_B + Z_A) + 0.5 * (Z_B - Z_A) * t;
        if let Some((r12, r13, _, _, _, _, _)) = boundary_branch_polys(alpha, gamma, tc, zk, bentry)
        {
            if r12.iter().all(|v| v.is_finite()) && r13.iter().all(|v| v.is_finite()) {
                es[k] = sylvester_det(&r12, &r13);
                if es[k].is_finite() {
                    any_valid = true;
                } else {
                    es[k] = 0.0;
                }
            }
        }
    }
    if !any_valid {
        return None;
    }

    let scale = es.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale < 1e-300 {
        return None;
    }
    let es_scaled: Vec<f64> = es.iter().map(|&e| e / scale).collect();
    let mut cc = cheb_fit_asc(&nodes, &es_scaled);
    while cc.len() > 1 && cc.last().copied().unwrap_or(0.0).abs() < 1e-12 {
        cc.pop();
    }
    if cc.len() < 2 {
        return None;
    }

    let roots = colleague_roots(&cc);
    for rt in roots {
        if rt.im.abs() > 1e-7 || rt.re < -1.001 || rt.re > 1.001 {
            continue;
        }
        let z0 = 0.5 * (Z_B + Z_A) + 0.5 * (Z_B - Z_A) * rt.re;
        if !(-1.0..=1.0).contains(&z0) {
            continue;
        }

        let (r12, r13, p6, h6, q1, q2, q3) = boundary_branch_polys(alpha, gamma, tc, z0, bentry)?;
        if !r12.iter().all(|v| v.is_finite()) || !r13.iter().all(|v| v.is_finite()) {
            continue;
        }

        let r12t = poly_trim(&r12);
        if r12t.len() < 2 {
            continue;
        }
        let s13 = r13.iter().fold(0.0f64, |m, &v| m.max(v.abs())) + 1e-300;
        let v_roots = poly_roots_real(r12t, 1e-6);

        for v0 in v_roots {
            // Check R13(v0)
            if polyval(&r13, v0).abs() / s13 > 1e-5 {
                continue;
            }
            // Back-substitute u from Q1 (as a quadratic in u at fixed v)
            let uq1 = biv_as_quadric(&q1);
            let (au, bu_arr, cu_arr) = uq1;
            let buv = polyval(&bu_arr, v0);
            let cuv = polyval(&cu_arr, v0);
            let us = if au.abs() < 1e-14 {
                if buv.abs() < 1e-14 {
                    vec![]
                } else {
                    vec![-cuv / buv]
                }
            } else {
                let disc = buv * buv - 4.0 * au * cuv;
                if disc < -1e-8 {
                    vec![]
                } else {
                    let sq = disc.max(0.0).sqrt();
                    vec![(-buv + sq) / (2.0 * au), (-buv - sq) / (2.0 * au)]
                }
            };

            for u0 in us {
                // Reconstruct y6 = P6 @ (u0, v0) + h6
                let y6: [f64; 6] = std::array::from_fn(|i| p6[i][0] * u0 + p6[i][1] * v0 + h6[i]);
                let (x0, y0, z0_cap, w0, r0, s0) = (y6[0], y6[1], y6[2], y6[3], y6[4], y6[5]);

                // Check quadric residuals
                let qres = biv_eval(&q1, u0, v0)
                    .abs()
                    .max(biv_eval(&q2, u0, v0).abs())
                    .max(biv_eval(&q3, u0, v0).abs());
                if qres > 1e-6 {
                    continue;
                }

                let b_birk = [
                    [x0, y0, 1.0 - x0 - y0],
                    [z0_cap, w0, 1.0 - z0_cap - w0],
                    [
                        1.0 - x0 - z0_cap,
                        1.0 - y0 - w0,
                        x0 + y0 + z0_cap + w0 - 1.0,
                    ],
                ];
                if b_birk.iter().flatten().any(|&v| v < -1e-6) {
                    continue;
                }
                let b_pos = [
                    [
                        b_birk[0][0].max(0.0),
                        b_birk[0][1].max(0.0),
                        b_birk[0][2].max(0.0),
                    ],
                    [
                        b_birk[1][0].max(0.0),
                        b_birk[1][1].max(0.0),
                        b_birk[1][2].max(0.0),
                    ],
                    [
                        b_birk[2][0].max(0.0),
                        b_birk[2][1].max(0.0),
                        b_birk[2][2].max(0.0),
                    ],
                ];

                for r_mat in so3_sign_lift(&b_pos, r0, s0, 1e-4) {
                    for o in reconstruct_so4(z0, &r_mat) {
                        let resid = char_poly_residual(alpha, gamma, &o, w);
                        if resid < tol {
                            return Some(resid);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Debug: stage values on row 36581's known branch (wb1, identity a_ord,
/// g_perm (1,0,3,2)); compare with validate_dd_dual_locator.py.
pub fn debug_fold_known_branch(triple: &[f64; 9]) {
    let (c, g, _wp, wm) = row_spectra(triple);
    let alpha: [C; 4] = c;
    let g_perm = [1usize, 0, 3, 2];
    let gamma: [C; 4] = std::array::from_fn(|k| g[g_perm[k]]);
    let tc = char_coeffs(&wm);

    for &zcap in &[0.515059f64, 0.515, 0.52, 0.5, 0.55] {
        if let Some((dip_z, dip_v)) = fold_dip(&alpha, &gamma, tc, zcap) {
            println!("zcap {zcap:.6}: dip at z={dip_z:.4} val={dip_v:.3e}");
        } else {
            println!("zcap {zcap:.6}: no dip data");
        }
        let cands = fold_z_candidates(&alpha, &gamma, tc, zcap, 0.75, 0.90, 5e-2);
        println!(
            "  windowed candidates: {:?}",
            cands
                .iter()
                .map(|(re, im)| (format!("{re:.6}"), format!("{im:.1e}")))
                .collect::<Vec<_>>()
        );
        for &(z_re, _) in &cands {
            match solve_interior_branch(&alpha, &gamma, &wm, z_re, 1e-8) {
                Some((r, _)) => println!("  construct at {z_re:.6}: HIT {r:.2e}"),
                None => println!("  construct at {z_re:.6}: miss"),
            }
        }
    }
    println!("staged: {:?}", fold_branch_staged(&alpha, &gamma, &wm));
}

// ---- Public row runner ---------------------------------------------------

/// Result of a row run.
pub struct WallHit {
    pub residual: f64,
    pub w_branch: u8,
    pub a_ord: [usize; 4],
    pub g_perm: [usize; 4],
    pub z0: f64,
    pub is_boundary: bool,
    pub bentry: Option<usize>,
}

const TOL: f64 = 1e-8;

/// Interior solver: 2 w-branches × 12 alpha orderings × 24 gamma perms × 9 z-values.
pub fn run_row_interior(triple: &[f64; 9]) -> Option<WallHit> {
    let (c, g, wp, wm) = row_spectra(triple);
    let gamma_perms = all_gamma_perms();

    for (wb, w) in [(0u8, wp), (1u8, wm)] {
        for a_ord in ALPHA_ORDERINGS {
            let alpha: [C; 4] = std::array::from_fn(|k| c[a_ord[k]]);
            for g_perm in &gamma_perms {
                let gamma: [C; 4] = std::array::from_fn(|k| g[g_perm[k]]);
                for &z0 in &Z_VALS {
                    if let Some((resid, _o)) = solve_interior_branch(&alpha, &gamma, &w, z0, TOL) {
                        return Some(WallHit {
                            residual: resid,
                            w_branch: wb,
                            a_ord,
                            g_perm: *g_perm,
                            z0,
                            is_boundary: false,
                            bentry: None,
                        });
                    }
                }
            }
        }
    }
    None
}

/// Production-additive capped interior attempt for the cascade.
///
/// Bounded enumeration: a fixed combo cap (an enumeration bound in the
/// CAND_K/HEAD_K family, not a tuning knob) so a decline is cheap; tight
/// acceptance so near-misses fall through to the incumbent rungs. The hit O
/// is un-permuted back to the ORIGINAL (c, g) frame and re-verified there
/// before being returned, so a frame-convention error cannot ship a wrong
/// matrix.
pub fn try_wall_capped(
    triple: &[f64; 9],
    max_combos: usize,
    tol: f64,
) -> Option<([[f64; 4]; 4], f64)> {
    let (c, g, wp, wm) = row_spectra(triple);
    let gamma_perms = all_gamma_perms();
    let mut combos = 0usize;
    for w in [wp, wm] {
        for a_ord in ALPHA_ORDERINGS {
            let alpha: [C; 4] = std::array::from_fn(|k| c[a_ord[k]]);
            for g_perm in &gamma_perms {
                let gamma: [C; 4] = std::array::from_fn(|k| g[g_perm[k]]);
                for &z0 in &Z_VALS {
                    combos += 1;
                    if combos > max_combos {
                        return None;
                    }
                    if let Some((_, o)) = solve_interior_branch(&alpha, &gamma, &w, z0, tol) {
                        // un-permute: O_orig[a_ord[r]][g_perm[s]] = O[r][s]
                        let mut o_orig = [[0.0f64; 4]; 4];
                        for r in 0..4 {
                            for s in 0..4 {
                                o_orig[a_ord[r]][g_perm[s]] = o[r][s];
                            }
                        }
                        let resid = char_poly_residual(&c, &g, &o_orig, &w);
                        if resid < tol {
                            return Some((o_orig, resid));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Boundary solver: 2 w-branches × 12 alpha orderings × 24 gamma perms × 9 Birkhoff entries.
pub fn run_row_boundary(triple: &[f64; 9]) -> Option<WallHit> {
    let (c, g, wp, wm) = row_spectra(triple);
    let gamma_perms = all_gamma_perms();

    for (wb, w) in [(0u8, wp), (1u8, wm)] {
        for a_ord in ALPHA_ORDERINGS {
            let alpha: [C; 4] = std::array::from_fn(|k| c[a_ord[k]]);
            for g_perm in &gamma_perms {
                let gamma: [C; 4] = std::array::from_fn(|k| g[g_perm[k]]);
                for bentry in 0..9 {
                    if let Some(resid) = solve_boundary_branch(&alpha, &gamma, &w, bentry, TOL) {
                        return Some(WallHit {
                            residual: resid,
                            w_branch: wb,
                            a_ord,
                            g_perm: *g_perm,
                            z0: 0.0,
                            is_boundary: true,
                            bentry: Some(bentry),
                        });
                    }
                }
            }
        }
    }
    None
}

// ---- Double-double (Dekker) arithmetic for the fold-class eliminant ------
//
// The fold row has a DOUBLE ROOT of E(Z): at float distance ~4e-4 the signal
// is ~1e-8, below f64 noise.  Computing the Sylvester samples in DD lifts the
// noise floor to ~1e-10 (106-bit mantissa) -- well inside the capture window.
// Only the sample-computation stages run in DD; fit + root-finding run in f64.

#[derive(Clone, Copy, Debug)]
struct DD {
    hi: f64,
    lo: f64,
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl DD {
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        DD { hi: x, lo: 0.0 }
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
    #[inline(always)]
    fn neg(self) -> Self {
        DD {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
    #[inline(always)]
    fn quicksum(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum(hi, lo);
        DD { hi: s, lo: e }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        let (s, e) = two_sum(self.hi, b.hi);
        DD::quicksum(s, e + self.lo + b.lo)
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        let (p, e) = two_prod(self.hi, b.hi);
        DD::quicksum(p, e + self.hi * b.lo + self.lo * b.hi)
    }
    #[inline(always)]
    fn div(self, b: Self) -> Self {
        let q1 = self.hi / b.hi;
        let r = self.sub(DD::from_f64(q1).mul(b));
        DD::quicksum(q1, r.hi / b.hi)
    }
}

fn dd_polymul(p: &[DD], q: &[DD]) -> Vec<DD> {
    if p.is_empty() || q.is_empty() {
        return vec![];
    }
    let mut r = vec![DD::from_f64(0.0); p.len() + q.len() - 1];
    for (i, &pi) in p.iter().enumerate() {
        for (j, &qj) in q.iter().enumerate() {
            r[i + j] = r[i + j].add(pi.mul(qj));
        }
    }
    r
}

/// DDQ = (a, [b_Y, b_0], [c_Y2, c_Y, c_0]) -- same shape as Quadric but in DD.
type DDQ = (DD, [DD; 2], [DD; 3]);

fn dd_poly_coeffs(a_wrs: &[[f64; 3]; 3], b_wrs: &[f64; 3], z_dd: DD) -> (DDQ, DDQ, DDQ) {
    let f = DD::from_f64;
    let one = f(1.0);
    let two = f(2.0);

    let c0_w = f(a_wrs[0][2]).mul(z_dd).add(f(b_wrs[0]));
    let c0_r = f(a_wrs[1][2]).mul(z_dd).add(f(b_wrs[1]));
    let c0_s = f(a_wrs[2][2]).mul(z_dd).add(f(b_wrs[2]));

    let p_r = f(a_wrs[1][0]);
    let q_r = f(a_wrs[1][1]);
    let p_s = f(a_wrs[2][0]);
    let q_s = f(a_wrs[2][1]);
    let p_w = f(a_wrs[0][0]);
    let q_w = f(a_wrs[0][1]);

    // Q1 = r^2 - X
    let a1 = p_r.mul(p_r);
    let b1 = [p_r.mul(q_r).mul(two), p_r.mul(c0_r).mul(two).sub(one)];
    let c1 = [q_r.mul(q_r), q_r.mul(c0_r).mul(two), c0_r.mul(c0_r)];

    // Q2 = s^2 - (1-Z-W)(1-Y-W)
    let a_s = one.sub(z_dd).sub(c0_w);
    let b_s = one.sub(c0_w);
    let a2 = p_s.mul(p_s).sub(p_w.mul(p_w));
    let b2 = [
        p_s.mul(q_s).mul(two).sub(p_w.mul(one.add(q_w.mul(two)))),
        p_s.mul(c0_s).mul(two).add(p_w.mul(a_s.add(b_s))),
    ];
    let c2 = [
        q_s.mul(q_s).sub(q_w.mul(one.add(q_w))),
        q_s.mul(c0_s)
            .mul(two)
            .add(a_s.mul(one.add(q_w)))
            .add(b_s.mul(q_w)),
        c0_s.mul(c0_s).sub(a_s.mul(b_s)),
    ];

    // Q3 = (r+s)^2 - W*(X+Y+Z+W-1)
    let prs = p_r.add(p_s);
    let qrs = q_r.add(q_s);
    let c0rs = c0_r.add(c0_s);
    let w_c = z_dd.add(c0_w).sub(one);
    let a3 = prs.mul(prs).sub(p_w.mul(one.add(p_w)));
    let b3 = [
        prs.mul(qrs)
            .mul(two)
            .sub(p_w.add(q_w).add(p_w.mul(q_w).mul(two))),
        prs.mul(c0rs)
            .mul(two)
            .sub(p_w.mul(w_c))
            .sub(c0_w.mul(one.add(p_w))),
    ];
    let c3 = [
        qrs.mul(qrs).sub(q_w.mul(one.add(q_w))),
        qrs.mul(c0rs)
            .mul(two)
            .sub(q_w.mul(w_c))
            .sub(c0_w.mul(one.add(q_w))),
        c0rs.mul(c0rs).sub(c0_w.mul(w_c)),
    ];

    ((a1, b1, c1), (a2, b2, c2), (a3, b3, c3))
}

fn dd_elim_x_poly(q1: &DDQ, q2: &DDQ) -> Vec<DD> {
    let (a1, b1, c1) = q1;
    let (a2, b2, c2) = q2;

    let num: Vec<DD> = c1
        .iter()
        .zip(c2.iter())
        .map(|(&x, &y)| a2.mul(x).sub(a1.mul(y)))
        .collect();
    let den: Vec<DD> = b1
        .iter()
        .zip(b2.iter())
        .map(|(&x, &y)| a1.mul(y).sub(a2.mul(x)))
        .collect();

    let num_sq = dd_polymul(&num, &num);
    let b1_num = dd_polymul(b1, &num);
    let b1_nd = dd_polymul(&b1_num, &den);
    let den_sq = dd_polymul(&den, &den);
    let c1_dsq = dd_polymul(c1, &den_sq);

    let n = num_sq.len().max(b1_nd.len()).max(c1_dsq.len());
    let mut r = vec![DD::from_f64(0.0); n];
    let place = |r: &mut Vec<DD>, src: &[DD], scale: Option<DD>| {
        for (i, &v) in src.iter().enumerate() {
            let idx = n - src.len() + i;
            let sv = if let Some(s) = scale { s.mul(v) } else { v };
            r[idx] = r[idx].add(sv);
        }
    };
    place(&mut r, &num_sq, Some(*a1));
    place(&mut r, &b1_nd, None);
    place(&mut r, &c1_dsq, None);
    r
}

fn dd_trim(v: &[DD]) -> &[DD] {
    let sc = v.iter().fold(0.0f64, |m, d| m.max(d.hi.abs())).max(1e-300);
    let i = v
        .iter()
        .position(|d| d.hi.abs() >= 1e-10 * sc)
        .unwrap_or(v.len().saturating_sub(1));
    &v[i..]
}

fn dd_sylvester_det(p_raw: &[DD], q_raw: &[DD]) -> f64 {
    let p = dd_trim(p_raw);
    let q = dd_trim(q_raw);
    let n = p.len().saturating_sub(1);
    let m = q.len().saturating_sub(1);
    if n < 1 || m < 1 {
        return 0.0;
    }
    let sz = n + m;
    let mut s = vec![DD::from_f64(0.0); sz * sz];

    for i in 0..m {
        for (j, &v) in p.iter().enumerate() {
            s[i * sz + i + j] = v;
        }
    }
    for i in 0..n {
        for (j, &v) in q.iter().enumerate() {
            s[(m + i) * sz + i + j] = v;
        }
    }

    let mut sign = DD::from_f64(1.0);
    for col in 0..sz {
        let mut max_row = col;
        let mut max_val = s[col * sz + col].hi.abs();
        for row in (col + 1)..sz {
            let v = s[row * sz + col].hi.abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-280 {
            return 0.0;
        }
        if max_row != col {
            for k in 0..sz {
                s.swap(col * sz + k, max_row * sz + k);
            }
            sign = sign.neg();
        }
        let pivot = s[col * sz + col];
        for row in (col + 1)..sz {
            let factor = s[row * sz + col].div(pivot);
            for k in col..sz {
                let v = factor.mul(s[col * sz + k]);
                s[row * sz + k] = s[row * sz + k].sub(v);
            }
        }
    }
    (0..sz).fold(sign, |acc, i| acc.mul(s[i * sz + i])).to_f64()
}

fn solve_fold_dd_branch(
    alpha: &[C; 4],
    gamma: &[C; 4],
    w: &[C; 4],
    z0: f64,
    tol: f64,
) -> Option<f64> {
    let tc = char_coeffs(w);
    let (a_wrs, b_wrs) = linear_chart(alpha, gamma, tc, z0)?;

    let nodes = chebpts1(NS);
    let mut es = vec![0.0f64; NS];
    for (k, &t) in nodes.iter().enumerate() {
        let zk = 0.5 * (Z_HI + Z_LO) + 0.5 * (Z_HI - Z_LO) * t;
        let (q1, q2, q3) = dd_poly_coeffs(&a_wrs, &b_wrs, DD::from_f64(zk));
        let r12 = dd_elim_x_poly(&q1, &q2);
        let r13 = dd_elim_x_poly(&q1, &q3);
        let e = dd_sylvester_det(&r12, &r13);
        es[k] = if e.is_finite() { e } else { 0.0 };
    }

    let scale = es.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale < 1e-300 {
        return None;
    }
    let es_scaled: Vec<f64> = es.iter().map(|&e| e / scale).collect();
    let mut cc = cheb_fit_asc(&nodes, &es_scaled);
    while cc.len() > 1 && cc.last().copied().unwrap_or(0.0).abs() < 1e-12 {
        cc.pop();
    }
    if cc.len() < 2 {
        return None;
    }

    let roots = colleague_roots(&cc);
    for rt in roots {
        if rt.im.abs() > 1e-6 || rt.re < -1.001 || rt.re > 1.001 {
            continue;
        }
        let z_cap = 0.5 * (Z_HI + Z_LO) + 0.5 * (Z_HI - Z_LO) * rt.re;
        if !(0.0..=1.0).contains(&z_cap) {
            continue;
        }
        let (q1, q2, q3) = poly_coeffs(&a_wrs, &b_wrs, z_cap);
        let (r12, _, _) = elim_x_poly(&q1, &q2);
        let r12t = poly_trim(&r12);
        if r12t.len() < 2 {
            continue;
        }
        let y_roots = poly_roots_real(r12t, 1e-6);
        for y0 in y_roots {
            if y0 < -1e-9 || y0 > 1.0 + 1e-9 {
                continue;
            }
            let (a1, b1, c1) = q1;
            let b1v = polyval(&b1, y0);
            let c1v = polyval(&c1, y0);
            let disc = b1v * b1v - 4.0 * a1 * c1v;
            if disc < -1e-8 {
                continue;
            }
            let sq = disc.max(0.0).sqrt();
            for &x0 in &[(-b1v + sq) / (2.0 * a1), (-b1v - sq) / (2.0 * a1)] {
                if x0 < -1e-9 || x0 > 1.0 + 1e-9 {
                    continue;
                }
                let q2v = eval_quadric(&q2, x0, y0).abs();
                let q3v = eval_quadric(&q3, x0, y0).abs();
                if q2v.max(q3v) > 1e-6 {
                    continue;
                }
                let w0 = a_wrs[0][0] * x0 + a_wrs[0][1] * y0 + a_wrs[0][2] * z_cap + b_wrs[0];
                let r0 = a_wrs[1][0] * x0 + a_wrs[1][1] * y0 + a_wrs[1][2] * z_cap + b_wrs[1];
                let s0 = a_wrs[2][0] * x0 + a_wrs[2][1] * y0 + a_wrs[2][2] * z_cap + b_wrs[2];
                let b_birk = [
                    [x0, y0, 1.0 - x0 - y0],
                    [z_cap, w0, 1.0 - z_cap - w0],
                    [1.0 - x0 - z_cap, 1.0 - y0 - w0, x0 + y0 + z_cap + w0 - 1.0],
                ];
                if b_birk.iter().flatten().any(|&v| v < -1e-8) {
                    continue;
                }
                for r_mat in so3_sign_lift(&b_birk, r0, s0, 1e-5) {
                    for o in reconstruct_so4(z0, &r_mat) {
                        let resid = char_poly_residual(alpha, gamma, &o, w);
                        if resid < tol {
                            return Some(resid);
                        }
                    }
                }
            }
        }
    }
    None
}

/// DD dual z-locator. The fold's zero has even multiplicity in every f64
/// resultant, so grid-z enumeration can never land in its ~4e-4 window and
/// DD precision at grid z is useless (no real root exists there at any
/// precision). Instead: fix the Birkhoff coordinate Z0, sample
/// Etilde(z) = Res_Y(R12, R13)(z; Z0) with the DD elimination chain (the
/// cancellation lives in elim/Sylvester, not the chart), fit, and take real
/// parts of near-real colleague roots -- an even-order zero splits into a
/// conjugate pair ~1e-7 off-axis whose real part lands deep inside the
/// window. Construction then reuses the VERIFIED f64 chain at that z.
/// Returns (z_real_part, |imag|) of near-real root pairs of the smoothly
/// normalized Etilde(z; zcap). |imag| MEASURES the distance |zcap - Z*|
/// (validated: linear, validate_dd_dual_locator.py), which drives the staged
/// refinement in the runner.
/// Smoothly normalized Etilde samples over an arbitrary z window.
fn fold_samples(
    alpha: &[C; 4],
    gamma: &[C; 4],
    tc: (C, C),
    zcap: f64,
    lo: f64,
    hi: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nodes = chebpts1(NS);
    let mut es = vec![0.0f64; NS];
    for (k, &t) in nodes.iter().enumerate() {
        let z = 0.5 * (hi + lo) + 0.5 * (hi - lo) * t;
        if let Some((a_wrs, b_wrs)) = linear_chart(alpha, gamma, tc, z) {
            let (q1, q2, q3) = dd_poly_coeffs(&a_wrs, &b_wrs, DD::from_f64(zcap));
            let r12 = dd_elim_x_poly(&q1, &q2);
            let r13 = dd_elim_x_poly(&q1, &q3);
            // smooth normalization: resultant homogeneity in 2-norms
            // (max-abs has kinks that make Chebyshev fits ring)
            let n1 = r12.iter().map(|d| d.hi * d.hi).sum::<f64>().sqrt();
            let n2 = r13.iter().map(|d| d.hi * d.hi).sum::<f64>().sqrt();
            let (dm, dk) = (r12.len() - 1, r13.len() - 1);
            let e = dd_sylvester_det(&r12, &r13);
            let denom = n1.powi(dk as i32) * n2.powi(dm as i32);
            es[k] = if e.is_finite() && denom > 0.0 && denom.is_finite() {
                e / denom
            } else {
                0.0
            };
        }
    }
    (nodes, es)
}

/// Windowed pair extraction: fit the samples, return (z_re, |im|) of
/// near-real root pairs, mapped back to the window.
fn fold_z_candidates(
    alpha: &[C; 4],
    gamma: &[C; 4],
    tc: (C, C),
    zcap: f64,
    lo: f64,
    hi: f64,
    im_gate: f64,
) -> Vec<(f64, f64)> {
    let (nodes, es) = fold_samples(alpha, gamma, tc, zcap, lo, hi);
    let scale = es.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale < 1e-300 {
        return vec![];
    }
    let es_scaled: Vec<f64> = es.iter().map(|&e| e / scale).collect();
    let mut cc = cheb_fit_asc(&nodes, &es_scaled);
    while cc.len() > 1 && cc.last().copied().unwrap_or(0.0).abs() < 1e-12 {
        cc.pop();
    }
    if cc.len() < 2 {
        return vec![];
    }
    colleague_roots(&cc)
        .into_iter()
        .filter(|rt| rt.im.abs() < im_gate && rt.re.abs() <= 1.0)
        .map(|rt| (0.5 * (hi + lo) + 0.5 * (hi - lo) * rt.re, rt.im.abs()))
        .collect()
}

/// Global dip detector: (z at min |normalized sample|, dip value).
fn fold_dip(alpha: &[C; 4], gamma: &[C; 4], tc: (C, C), zcap: f64) -> Option<(f64, f64)> {
    const ZW: f64 = 0.985;
    let (nodes, es) = fold_samples(alpha, gamma, tc, zcap, -ZW, ZW);
    let scale = es.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale < 1e-300 {
        return None;
    }
    let mut best = (f64::INFINITY, 0.0f64);
    for (k, &e) in es.iter().enumerate() {
        let v = (e / scale).abs();
        if v < best.0 {
            best = (v, ZW * nodes[k]);
        }
    }
    Some((best.1, best.0))
}

/// Staged Z refinement per branch (fixed depth 3, five-point stages: a
/// finite bounded enumeration, no convergence test). Stage-1 coarse grid
/// detects the fold pair (|im| gate 0.05, detection radius ~2e-2 measured);
/// each later stage re-centers on the best zcap with step = |im| (the
/// measured linear error law), shrinking |im| ~10x per stage; the final
/// pair's real part carries z* to ~1e-6, inside the construction window.
fn fold_branch_staged(alpha: &[C; 4], gamma: &[C; 4], w: &[C; 4]) -> Option<(f64, f64)> {
    let tc = char_coeffs(w);
    // stage 1: dip detection over a coarse zcap grid (samples only, no fit)
    let mut det: Option<(f64, f64, f64)> = None; // (dip_val, zcap, dip_z)
    let mut zc = 0.04f64;
    while zc < 0.97 {
        if let Some((dip_z, dip_v)) = fold_dip(alpha, gamma, tc, zc) {
            if det.map_or(true, |(bv, _, _)| dip_v < bv) {
                det = Some((dip_v, zc, dip_z));
            }
        }
        zc += 0.02;
    }
    let (dip_v, mut bzc, dip_z) = det?;
    if dip_v > 0.2 {
        return None; // no fold signal on this branch
    }
    // stages 2..3: windowed pair extraction around the dip; |im| = |zcap-Z*|
    // (measured linear law) drives the zcap re-centering; fixed depth.
    let (mut lo, mut hi) = ((dip_z - 0.08).max(-0.985), (dip_z + 0.08).min(0.985));
    let mut best: Option<(f64, f64)> = None; // (z_re, im)
    for _ in 0..3 {
        let gate = best.map_or(5e-2, |(_, im)| (im * 1.5).max(1e-6));
        let step = best.map_or(1e-2, |(_, im)| im.max(1e-9));
        let mut improved = false;
        for &dz in &[0.0, -step, step, -0.5 * step, 0.5 * step] {
            let zc2 = bzc + dz;
            if !(0.0..=1.0).contains(&zc2) {
                continue;
            }
            for (z_re, im) in fold_z_candidates(alpha, gamma, tc, zc2, lo, hi, gate) {
                if best.map_or(true, |(_, bim)| im < bim) {
                    best = Some((z_re, im));
                    bzc = zc2;
                    improved = true;
                }
            }
        }
        if let Some((z_re, _)) = best {
            lo = (z_re - 0.02).max(-0.985);
            hi = (z_re + 0.02).min(0.985);
        }
        if !improved {
            break;
        }
    }
    best
}

pub fn run_row_fold_dd(triple: &[f64; 9]) -> Option<WallHit> {
    let (c, g, wp, wm) = row_spectra(triple);
    let gamma_perms = all_gamma_perms();

    for (wb, w) in [(0u8, wp), (1u8, wm)] {
        for a_ord in ALPHA_ORDERINGS {
            let alpha: [C; 4] = std::array::from_fn(|k| c[a_ord[k]]);
            for g_perm in &gamma_perms {
                let gamma: [C; 4] = std::array::from_fn(|k| g[g_perm[k]]);
                if let Some((z_star, _im)) = fold_branch_staged(&alpha, &gamma, &w) {
                    if let Some((resid, _o)) =
                        solve_interior_branch(&alpha, &gamma, &w, z_star, TOL)
                    {
                        return Some(WallHit {
                            residual: resid,
                            w_branch: wb,
                            a_ord,
                            g_perm: *g_perm,
                            z0: z_star,
                            is_boundary: false,
                            bentry: None,
                        });
                    }
                }
            }
        }
    }
    None
}
