//! The genuine sextic of one three-Givens chart.
//!
//! For fixed `x`, three trilinear residuals are three planes in the Segre
//! coordinates `[1,y,z,yz]`.  Their projective kernel is the vector of signed
//! 3-by-3 cofactors.  It represents an actual `(y,z)` exactly when it lies on
//! the Segre quadric `n0*n3-n1*n2=0`.  Since every cofactor is cubic in `x`,
//! this constructs the genuine sextic directly from the original equations:
//! no degree-eight resultant and no numerical quadratic deflation.

pub(crate) struct ChartPolynomial {
    pub coefficients: [f64; 7],
    pub(crate) kernel: [[f64; 4]; 4],
}

/// Recover all `(y,z)` candidates at a computed `x` root.
///
/// Each trilinear becomes `(a_i+b_i*y)+(c_i+d_i*y)z=0`. Eliminate `z`
/// with the best-conditioned of the three row pairs, solve one quadratic,
/// then use the strongest row for the linear `z` solve.
pub(crate) fn recover(residuals: &[[f64; 8]; 3], x: f64) -> Vec<(f64, f64)> {
    let rows: [[f64; 4]; 3] = std::array::from_fn(|i| {
        let r = residuals[i];
        [
            r[0] + r[1] * x,
            r[2] + r[4] * x,
            r[3] + r[5] * x,
            r[6] + r[7] * x,
        ]
    });
    let mut quadratic = [0.0; 3];
    let mut best = -1.0;
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let ([ai, bi, ci, di], [aj, bj, cj, dj]) = (rows[i], rows[j]);
        let q = [
            ai * cj - aj * ci,
            ai * dj + bi * cj - aj * di - bj * ci,
            bi * dj - bj * di,
        ];
        let scale = q.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        if scale > best {
            best = scale;
            quadratic = q;
        }
    }

    let [c, b, a] = quadratic;
    let scale = a.abs().max(b.abs()).max(c.abs());
    let ys = if scale == 0.0 {
        Vec::new()
    } else if a.abs() <= 64.0 * f64::EPSILON * scale {
        if b == 0.0 { Vec::new() } else { vec![-c / b] }
    } else {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            Vec::new()
        } else {
            let root = discriminant.sqrt();
            let q = -0.5 * (b + b.signum() * root);
            if q == 0.0 {
                vec![-b / (2.0 * a)]
            } else {
                vec![q / a, c / q]
            }
        }
    };

    ys.into_iter()
        .filter_map(|y| {
            let row = rows
                .iter()
                .max_by(|u, v| (u[2] + u[3] * y).abs().total_cmp(&(v[2] + v[3] * y).abs()))?;
            let denominator = row[2] + row[3] * y;
            (denominator != 0.0).then_some((y, -(row[0] + row[1] * y) / denominator))
        })
        .collect()
}

/// Genuine chart eliminant and its projective recovery map.
pub(crate) fn eliminate(residuals: &[[f64; 8]; 3]) -> ChartPolynomial {
    // A(x) [1,y,z,yz]^T = 0. Each entry of A is affine in x.
    let a: [[[f64; 2]; 4]; 3] = std::array::from_fn(|row| {
        let r = residuals[row];
        [[r[0], r[1]], [r[2], r[4]], [r[3], r[5]], [r[6], r[7]]]
    });

    let mut n = [[0.0; 4]; 4];
    for omitted in 0..4 {
        let cols = std::array::from_fn::<_, 3, _>(|k| if k < omitted { k } else { k + 1 });
        n[omitted] = determinant(&a, cols);
        if omitted % 2 == 1 {
            for coefficient in &mut n[omitted] {
                *coefficient = -*coefficient;
            }
        }
    }

    let n03 = cubic_product(&n[0], &n[3]);
    let n12 = cubic_product(&n[1], &n[2]);
    ChartPolynomial {
        coefficients: std::array::from_fn(|k| n03[k] - n12[k]),
        kernel: n,
    }
}

// ---------------------------------------------------------------------
// Fixed-size 6x6 eigenvalues-only real QZ (Moler-Stewart), specialized to
// the block companion pencil of the 2x2 cubic matrix polynomial K(x):
// no allocation, no vectors, Givens Hessenberg-triangular reduction and an
// implicit double-shift chase. A negligible B diagonal in the active window
// (the exact degree-drop stratum, which the caller's scale check already
// routes to the dynamic scalar rooter) or non-convergence returns None and
// the caller keeps its current path, so completeness never depends on this
// kernel.

mod qz6 {
    const N: usize = 6;
    pub type M6 = [[f64; N]; N];

    #[inline]
    fn rot(a: f64, b: f64) -> (f64, f64) {
        // (c, s) with -s*a + c*b = 0: left rotation zeroing the second entry.
        let r = a.hypot(b);
        if r == 0.0 { (1.0, 0.0) } else { (a / r, b / r) }
    }

    #[inline]
    fn lrot(m: &mut M6, c: f64, s: f64, i: usize, j: usize, c0: usize) {
        for k in c0..N {
            let (x, y) = (m[i][k], m[j][k]);
            m[i][k] = c * x + s * y;
            m[j][k] = -s * x + c * y;
        }
    }

    #[inline]
    fn rrot(m: &mut M6, c: f64, s: f64, i: usize, j: usize, r1: usize) {
        // columns i, j: new_i = c*col_i + s*col_j, new_j = -s*col_i + c*col_j
        for k in 0..=r1 {
            let (x, y) = (m[k][i], m[k][j]);
            m[k][i] = c * x + s * y;
            m[k][j] = -s * x + c * y;
        }
    }

    /// Unit Householder vector sending v to a multiple of e1, or None if v=0.
    #[inline]
    fn house3(v: [f64; 3]) -> Option<[f64; 3]> {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n == 0.0 {
            return None;
        }
        let alpha = if v[0] >= 0.0 { -n } else { n };
        let mut u = [v[0] - alpha, v[1], v[2]];
        let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        if un == 0.0 {
            return None;
        }
        for x in &mut u {
            *x /= un;
        }
        Some(u)
    }

    #[inline]
    fn lhouse3(m: &mut M6, u: [f64; 3], r: usize, c0: usize) {
        for c in c0..N {
            let d = 2.0 * (u[0] * m[r][c] + u[1] * m[r + 1][c] + u[2] * m[r + 2][c]);
            m[r][c] -= d * u[0];
            m[r + 1][c] -= d * u[1];
            m[r + 2][c] -= d * u[2];
        }
    }

    #[inline]
    fn rhouse3(m: &mut M6, u: [f64; 3], c: usize, r1: usize) {
        for r in 0..=r1 {
            let d = 2.0 * (u[0] * m[r][c] + u[1] * m[r][c + 1] + u[2] * m[r][c + 2]);
            m[r][c] -= d * u[0];
            m[r][c + 1] -= d * u[1];
            m[r][c + 2] -= d * u[2];
        }
    }

    /// Reduce (A, B) to (upper Hessenberg, upper triangular) by left Givens
    /// sweeps and B-restoring column rotations.
    fn hess_tri(a: &mut M6, b: &mut M6) {
        for c in 0..N - 1 {
            for r in (c + 1..N).rev() {
                if b[r][c] != 0.0 {
                    let (gc, gs) = rot(b[r - 1][c], b[r][c]);
                    lrot(b, gc, gs, r - 1, r, c);
                    b[r][c] = 0.0;
                    lrot(a, gc, gs, r - 1, r, 0);
                }
            }
        }
        for c in 0..N - 2 {
            for r in (c + 2..N).rev() {
                if a[r][c] != 0.0 {
                    let (gc, gs) = rot(a[r - 1][c], a[r][c]);
                    lrot(a, gc, gs, r - 1, r, c);
                    a[r][c] = 0.0;
                    lrot(b, gc, gs, r - 1, r, r - 1);
                    if b[r][r - 1] != 0.0 {
                        // zero B[r][r-1] mixing columns (r-1, r); rows to N-1:
                        // transient below-Hessenberg fill can sit at row r+2
                        let (hc, hs) = rot(b[r][r], b[r][r - 1]);
                        rrot(b, hc, -hs, r - 1, r, N - 1);
                        b[r][r - 1] = 0.0;
                        rrot(a, hc, -hs, r - 1, r, N - 1);
                    }
                }
            }
        }
    }

    /// Eigenvalues (alpha_re, alpha_im, beta) of the regular pencil (A, B).
    /// None on a negligible B diagonal in an active window or non-convergence.
    pub fn eigenvalues(a: &mut M6, b: &mut M6) -> Option<[(f64, f64, f64); N]> {
        hess_tri(a, b);
        let anorm: f64 = a
            .iter()
            .flat_map(|r| r.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));
        let bnorm: f64 = b
            .iter()
            .flat_map(|r| r.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));
        if anorm == 0.0 || bnorm == 0.0 {
            return None;
        }
        let epsa = f64::EPSILON * anorm * 16.0;
        let epsb = f64::EPSILON * bnorm * 16.0;

        let mut out = [(0.0, 0.0, 0.0); N];
        let mut hi = N - 1;
        let mut its = 0usize;
        let mut total = 0usize;
        loop {
            total += 1;
            if total > 40 * N {
                return None;
            }
            // deflate negligible subdiagonals
            let mut lo = hi;
            while lo > 0 {
                if a[lo][lo - 1].abs()
                    <= epsa.max(f64::EPSILON * (a[lo - 1][lo - 1].abs() + a[lo][lo].abs()))
                {
                    a[lo][lo - 1] = 0.0;
                    break;
                }
                lo -= 1;
            }
            if lo == hi {
                // 1x1 block
                if b[hi][hi].abs() <= epsb {
                    return None; // infinite eigenvalue: decline to the caller
                }
                out[hi] = (a[hi][hi], 0.0, b[hi][hi]);
                if hi == 0 {
                    return Some(out);
                }
                hi -= 1;
                its = 0;
                continue;
            }
            if lo == hi - 1 {
                // 2x2 block: det(A - x B) quadratic
                let (a11, a12, a21, a22) = (a[lo][lo], a[lo][hi], a[hi][lo], a[hi][hi]);
                let (b11, b12, b22) = (b[lo][lo], b[lo][hi], b[hi][hi]);
                if b11.abs() <= epsb || b22.abs() <= epsb {
                    return None;
                }
                let p2 = b11 * b22;
                let p1 = -(a11 * b22 + a22 * b11 - a21 * b12);
                let p0 = a11 * a22 - a12 * a21;
                let disc = p1 * p1 - 4.0 * p2 * p0;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    let q = -0.5 * (p1 + p1.signum() * sq);
                    let (r1, r2) = if q == 0.0 {
                        (0.0, 0.0)
                    } else {
                        (q / p2, p0 / q)
                    };
                    out[lo] = (r1, 0.0, 1.0);
                    out[hi] = (r2, 0.0, 1.0);
                } else {
                    let re = -p1 / (2.0 * p2);
                    let im = (-disc).sqrt() / (2.0 * p2);
                    out[lo] = (re, im, 1.0);
                    out[hi] = (re, -im, 1.0);
                }
                if lo == 0 {
                    return Some(out);
                }
                hi = lo - 1;
                its = 0;
                continue;
            }
            // active window [lo..=hi], size >= 3
            for k in lo..=hi {
                if b[k][k].abs() <= epsb {
                    return None; // degree-drop stratum: caller keeps its path
                }
            }
            its += 1;
            if its > 30 {
                return None;
            }
            // implicit double shift from the trailing 2x2 of M = A B^-1
            let (x, y, z) = {
                let (h, t) = (&*a, &*b);
                let m11 = h[lo][lo] / t[lo][lo];
                let m21 = h[lo + 1][lo] / t[lo][lo];
                let m12 = (h[lo][lo + 1] - m11 * t[lo][lo + 1]) / t[lo + 1][lo + 1];
                let m22 = (h[lo + 1][lo + 1] - m21 * t[lo][lo + 1]) / t[lo + 1][lo + 1];
                let m32 = h[lo + 2][lo + 1] / t[lo + 1][lo + 1];
                // trailing 2x2 of M
                let e = hi - 1;
                let me11 =
                    h[e][e] / t[e][e] - h[e][e - 1] * t[e - 1][e] / (t[e - 1][e - 1] * t[e][e]);
                let me21 = h[hi][e] / t[e][e];
                let me12 = (h[e][hi] - me11 * t[e][hi]) / t[hi][hi]
                    - h[e][e - 1] * t[e - 1][hi] / (t[e - 1][e - 1] * t[hi][hi]);
                let me22 = (h[hi][hi] - me21 * t[e][hi]) / t[hi][hi];
                let (mut s, mut p) = (me11 + me22, me11 * me22 - me12 * me21);
                if its.is_multiple_of(10) {
                    // exceptional shift: perturb to break rare cycles
                    s = 1.5 * (h[hi][e].abs() + h[e][e - 1].abs()) / t[hi][hi].abs();
                    p = s * s * 0.25;
                }
                (
                    m11 * m11 + m12 * m21 - s * m11 + p,
                    m21 * (m11 + m22 - s),
                    m21 * m32,
                )
            };
            let (mut x, mut y, mut z) = (x, y, z);
            for k in lo..=hi - 2 {
                if let Some(u) = house3([x, y, z]) {
                    let c0 = if k > lo { k - 1 } else { lo };
                    lhouse3(a, u, k, c0);
                    lhouse3(b, u, k, c0);
                }
                // restore B: right Householder zeroing row k+2 of B left of diag
                let rowv = [b[k + 2][k + 2], b[k + 2][k + 1], b[k + 2][k]];
                if let Some(u3) = house3(rowv) {
                    // reflects (col k+2, col k+1, col k) order; remap to columns
                    let u = [u3[2], u3[1], u3[0]];
                    let r1 = (k + 3).min(hi);
                    rhouse3(a, u, k, r1);
                    rhouse3(b, u, k, r1);
                    // exact zeros
                    b[k + 2][k] = 0.0;
                    b[k + 2][k + 1] = 0.0;
                }
                if b[k + 1][k] != 0.0 {
                    let (hc, hs) = rot(b[k + 1][k + 1], b[k + 1][k]);
                    let r1 = (k + 3).min(hi);
                    rrot(b, hc, -hs, k, k + 1, r1);
                    b[k + 1][k] = 0.0;
                    rrot(a, hc, -hs, k, k + 1, r1);
                }
                if k > lo {
                    a[k + 1][k - 1] = 0.0;
                    if k + 2 <= hi {
                        a[k + 2][k - 1] = 0.0;
                    }
                }
                x = a[k + 1][k];
                y = a[k + 2][k];
                z = if k + 3 <= hi { a[k + 3][k] } else { 0.0 };
            }
            // final 2-row step: zero A[hi][hi-2], then B[hi][hi-1]
            let k = hi - 1;
            if a[hi][k - 1] != 0.0 {
                let (gc, gs) = rot(a[k][k - 1], a[hi][k - 1]);
                lrot(a, gc, gs, k, hi, k - 1);
                a[hi][k - 1] = 0.0;
                lrot(b, gc, gs, k, hi, k);
            }
            if b[hi][k] != 0.0 {
                let (hc, hs) = rot(b[hi][hi], b[hi][k]);
                rrot(b, hc, -hs, k, hi, hi);
                b[hi][k] = 0.0;
                rrot(a, hc, -hs, k, hi, hi);
            }
        }
    }
}

impl ChartPolynomial {
    /// All finite unit-interval roots via the fixed 6x6 eigenvalues-only QZ.
    /// None = kernel declined (degree drop / non-convergence): keep the
    /// caller's existing path.
    pub fn qz6_roots(&self) -> Option<Vec<f64>> {
        let mut k = [[[0.0f64; 2]; 2]; 4];
        for entry in 0..4 {
            for degree in 0..=3 {
                k[degree][entry / 2][entry % 2] = self.kernel[entry][degree];
            }
        }
        let mut a = [[0.0f64; 6]; 6];
        let mut b = [[0.0f64; 6]; 6];
        for i in 0..4 {
            a[i][i + 2] = 1.0;
            b[i][i] = 1.0;
        }
        for i in 0..2 {
            for j in 0..6 {
                a[i + 4][j] = -k[j / 2][i][j % 2];
            }
            for j in 0..2 {
                b[i + 4][j + 4] = k[3][i][j];
            }
        }
        let eigs = qz6::eigenvalues(&mut a, &mut b)?;
        Some(
            eigs.iter()
                .filter_map(|&(re, im, beta)| {
                    if beta == 0.0 {
                        return None;
                    }
                    let (re, im) = (re / beta, im / beta);
                    (im.abs() < 1e-6 && (-1e-12..=1.0 + 1e-12).contains(&re))
                        .then_some(re.clamp(0.0, 1.0))
                })
                .collect(),
        )
    }
}

fn determinant(a: &[[[f64; 2]; 4]; 3], c: [usize; 3]) -> [f64; 4] {
    // Leibniz formula. The six products are only cubics, so the direct form is
    // both smaller and better conditioned than building a generic polynomial
    // determinant.
    let mut out = [0.0; 4];
    add_triple(&mut out, &a[0][c[0]], &a[1][c[1]], &a[2][c[2]], 1.0);
    add_triple(&mut out, &a[0][c[1]], &a[1][c[2]], &a[2][c[0]], 1.0);
    add_triple(&mut out, &a[0][c[2]], &a[1][c[0]], &a[2][c[1]], 1.0);
    add_triple(&mut out, &a[0][c[2]], &a[1][c[1]], &a[2][c[0]], -1.0);
    add_triple(&mut out, &a[0][c[1]], &a[1][c[0]], &a[2][c[2]], -1.0);
    add_triple(&mut out, &a[0][c[0]], &a[1][c[2]], &a[2][c[1]], -1.0);
    out
}

fn add_triple(out: &mut [f64; 4], a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], sign: f64) {
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                out[i + j + k] += sign * a[i] * b[j] * c[k];
            }
        }
    }
}

fn cubic_product(a: &[f64; 4], b: &[f64; 4]) -> [f64; 7] {
    let mut out = [0.0; 7];
    for i in 0..4 {
        for j in 0..4 {
            out[i + j] += a[i] * b[j];
        }
    }
    out
}
