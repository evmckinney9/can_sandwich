//! The chart tier: closed-form frames from the row-in-plane atlas.
//!
//! A chart fixes one row of `O` to a coordinate 2-plane, `O = embed_a(R) G_(a,l)(w)` with
//! `t' = sin^2 w`. For fixed `t'` the two independent characteristic coefficients
//! `(Re e1, Im e1, e2)` are affine in the 3x3 orthostochastic block `A = R o R`, so the inner
//! solve is a line in the doubly stochastic space cut by the Heron quartic. The atlas is
//! 48 charts (rows and columns) x 2 target lifts x 3 roles of the spectral triple; roles
//! reconstruct `O` through spectral projectors.
//!
//! Three passes share the atlas: the fast pass probes each chart at fixed fractions of its
//! numerical-range interval; the wall pass probes the charts through the block frames nearest
//! a target root at the second-order `t'`; the exact pass selects `t'` from the discriminant
//! of the chart quartic (denominator-cleared coefficients are degree-12 polynomials in `t'`
//! for a fixed pivot pattern, interpolated from 13 Chebyshev nodes). Every candidate is a
//! closed-form frame accepted only by the production certificate. Block-stratum leaves
//! (`3+1`, `2+2`, near-scalar, degenerate pairs) handle clustered spectra.
use super::{Mat4, C};
use nalgebra::{Matrix3, Matrix4, RowVector4, Vector3};
type M3 = Matrix3<f64>;
type V3 = Vector3<f64>;
/// Real frame in `SO(4)`; the tier works with real matrices and lifts to `Mat4` only for the certificate.
type Real4 = Matrix4<f64>;
/// One chart line: a base point and a direction in the affine `A` coordinates.
type Line = ([f64; 4], [f64; 4]);
/// One endpoint's chart forms in double-double precision: coefficient rows and right-hand side.
type PreciseRows = (
    [[super::chart_precision::D; 4]; 3],
    [super::chart_precision::D; 3],
);
/// Halving steps of the discriminant sign isolation in the exact pass.
const BIS: usize = 25;
/// Probe points of the fast pass: the chord point first (`CHORD`), then two fractions of the
/// numerical-range interval.
const FRACTIONS: [f64; 3] = [CHORD, 0.25, 0.75];
/// Sentinel fraction: probe at the chord point nearest a target root (see [`chord_point`]).
const CHORD: f64 = -1.0;

/// The `t'` at which the chart's `(a, a)` sheet value `l_a (m_j + t' (m_k - m_j))` passes nearest a
/// target root, when that point lies inside the record's interval; otherwise the midpoint. Measured
/// on Haar rows this single probe lands in a window about twice as often as the midpoint.
fn chord_point(
    l: &[C; 4],
    m: &[C; 4],
    lt: &[C; 4],
    a: usize,
    j: usize,
    k: usize,
    (t0, t1): (f64, f64),
) -> f64 {
    let dm = m[k] - m[j];
    let mut best = (f64::INFINITY, 0.5 * (t0 + t1));
    if dm.norm() > 1e-12 {
        for &root in lt {
            let z = (root / l[a] - m[j]) / dm;
            let d = (z.im * (l[a] * dm).norm()).abs();
            if z.re >= t0 && z.re <= t1 && d < best.0 {
                best = (d, z.re);
            }
        }
    }
    best.1
}
fn complex(o: &Real4) -> Mat4 {
    o.map(|x| C::new(x, 0.0))
}
/// Nearest orthogonal frame to `o`: two terms of the polar series `O (I - E/2 + 3E^2/8)` with
/// `E = O^T O - I`, a fixed polynomial exact to `O(|E|^3)`.
fn polar(o: &Real4) -> Real4 {
    let e = o.transpose() * o - Real4::identity();
    o * (Real4::identity() - 0.5 * e + 0.375 * (e * e))
}
fn finite(o: &Real4) -> bool {
    o.iter().all(|x| x.is_finite())
}
fn e1e2_of_spectrum(l: &[C; 4]) -> [f64; 3] {
    let e1 = l[0] + l[1] + l[2] + l[3];
    let mut e2 = C::default();
    for a in 0..4 {
        for b in (a + 1)..4 {
            e2 += l[a] * l[b];
        }
    }
    [e1.re, e1.im, e2.re]
}
fn char_e1e2(d1: &[C; 4], d2: &[C; 4], o: &Real4) -> (C, C) {
    // W = D1 O D2 O^T ; e1 = tr W, e2 = (tr^2 - tr W^2)/2
    let mut w = Mat4::zeros();
    for i in 0..4 {
        for j in 0..4 {
            let mut s = C::default();
            for k in 0..4 {
                s += C::new(o[(i, k)] * o[(j, k)], 0.0) * d2[k];
            }
            w[(i, j)] = d1[i] * s;
        }
    }
    let mut tr = C::default();
    let mut tr2 = C::default();
    for i in 0..4 {
        tr += w[(i, i)];
        for k in 0..4 {
            tr2 += w[(i, k)] * w[(k, i)];
        }
    }
    (tr, (tr * tr - tr2) * 0.5)
}
/// Largest real root of the monic cubic `x^3 + a x^2 + b x + c`.
fn cubic_largest_root(a: f64, b: f64, c: f64) -> f64 {
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let z = if disc > 0.0 {
        let sd = disc.sqrt();
        (-q / 2.0 + sd).cbrt() + (-q / 2.0 - sd).cbrt()
    } else if p >= 0.0 {
        0.0
    } else {
        let r = 2.0 * (-p / 3.0).sqrt();
        let arg = (3.0 * q / (p * r)).clamp(-1.0, 1.0);
        r * (arg.acos() / 3.0).cos()
    };
    z - a / 3.0
}
/// Real roots of the quadratic `x^2 + b x + c`; a near-double root (imaginary part below the tolerance
/// used everywhere for near-real roots) is returned once at its real part.
fn quad_roots(b: f64, c: f64, out: &mut [f64; 6], n: &mut usize) {
    let d = b * b / 4.0 - c;
    let re = -b / 2.0;
    if d >= 0.0 {
        let sd = d.sqrt();
        out[*n] = re + sd;
        out[*n + 1] = re - sd;
        *n += 2;
    } else if (-d).sqrt() <= 1e-5 * (1.0 + re.abs()) {
        out[*n] = re;
        *n += 1;
    }
}
/// Real roots (near-double roots included at their real part) of a quartic given low to high.
fn real_roots(h: &[f64; 5]) -> ([f64; 6], usize) {
    let scale = h.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let mut out = [0.0; 6];
    let mut n = 0;
    if !(scale > 0.0) || !scale.is_finite() {
        return (out, 0);
    }
    let q: [f64; 5] = std::array::from_fn(|i| h[i] / scale);
    let mut deg = 4usize;
    while deg > 0 && q[deg].abs() <= 1e-6 {
        deg -= 1;
    }
    match deg {
        4 => {
            let (b, c, d, e) = (q[3] / q[4], q[2] / q[4], q[1] / q[4], q[0] / q[4]);
            let p = c - 3.0 * b * b / 8.0;
            let qd = d - b * c / 2.0 + b * b * b / 8.0;
            let r = e - b * d / 4.0 + b * b * c / 16.0 - 3.0 * b * b * b * b / 256.0;
            if qd.abs() < 1e-14 * (1.0 + p.abs() + r.abs()) {
                // biquadratic: y^2 = w
                let mut ws = [0.0; 6];
                let mut nw = 0;
                quad_roots(p, r, &mut ws, &mut nw);
                for &w in &ws[..nw] {
                    if w >= 0.0 {
                        let y = w.sqrt();
                        out[n] = y;
                        out[n + 1] = -y;
                        n += 2;
                    } else if (-w).sqrt() <= 1e-5 {
                        out[n] = 0.0;
                        n += 1;
                    }
                }
            } else {
                let m = cubic_largest_root(p, (p * p - 4.0 * r) / 4.0, -qd * qd / 8.0).max(0.0);
                let s = (2.0 * m).sqrt();
                for sg in [1.0, -1.0] {
                    quad_roots(-s * sg, p / 2.0 + m + sg * qd / (2.0 * s), &mut out, &mut n);
                }
            }
            for y in out[..n].iter_mut() {
                *y -= b / 4.0;
            }
        }
        3 => {
            let (a, b, c) = (q[2] / q[3], q[1] / q[3], q[0] / q[3]);
            // Factor an exact zero constant before approximating a cubic root:
            // dividing by a tiny approximation to zero corrupts Vieta deflation.
            if h[0] == 0.0 {
                out[n] = 0.0;
                n += 1;
                quad_roots(a, b, &mut out, &mut n);
                return (out, n);
            }
            // A small leading coefficient can put one cubic root far away.
            // Recover that root without cancellation. Deflating by the
            // largest (rather than largest-magnitude) root can lose the two
            // physical O(1) roots in a subtraction of O(|a|) terms.
            let positive = cubic_largest_root(a, b, c);
            let negative = -cubic_largest_root(-a, b, -c);
            let r1 = if positive.abs() >= negative.abs() {
                positive
            } else {
                negative
            };
            out[n] = r1;
            n += 1;
            if r1 != 0.0 {
                // Both deflations are algebraically equivalent for an exact
                // root. Compare their reconstructed cubic coefficients: Vieta
                // avoids cancellation for a distant root, while ordinary
                // deflation avoids dividing by an approximate zero root.
                let ordinary_sum = a + r1;
                let ordinary_product = b + r1 * ordinary_sum;
                let product = -c / r1;
                let sum = (product - b) / r1;
                let error = |sum: f64, product: f64| {
                    if !sum.is_finite() || !product.is_finite() {
                        return f64::INFINITY;
                    }
                    (r1 + a - sum)
                        .abs()
                        .max(r1.mul_add(sum, b - product).abs())
                        .max(r1.mul_add(product, c).abs())
                };
                let (sum, product) = if error(sum, product) < error(ordinary_sum, ordinary_product)
                {
                    (sum, product)
                } else {
                    (ordinary_sum, ordinary_product)
                };
                quad_roots(sum, product, &mut out, &mut n);
            } else {
                quad_roots(a, b, &mut out, &mut n);
            }
        }
        2 => quad_roots(q[1] / q[2], q[0] / q[2], &mut out, &mut n),
        1 => {
            out[0] = -q[0] / q[1];
            n = 1;
        }
        _ => {}
    }
    (out, n)
}
/// Discriminant of a quartic (coefficients low to high), normalized by the leading coefficient scale.
fn quartic_disc(h: &[f64; 5]) -> f64 {
    let (e, d, c, b, a) = (h[0], h[1], h[2], h[3], h[4]);
    256.0 * a * a * a * e * e * e - 192.0 * a * a * b * d * e * e - 128.0 * a * a * c * c * e * e
        + 144.0 * a * a * c * d * d * e
        - 27.0 * a * a * d * d * d * d
        + 144.0 * a * b * b * c * e * e
        - 6.0 * a * b * b * d * d * e
        - 80.0 * a * b * c * c * d * e
        + 18.0 * a * b * c * d * d * d
        + 16.0 * a * c * c * c * c * e
        - 4.0 * a * c * c * c * d * d
        - 27.0 * b * b * b * b * e * e
        + 18.0 * b * b * b * c * d * e
        - 4.0 * b * b * b * d * d * d
        - 4.0 * b * b * c * c * c * e
        + b * b * c * c * d * d
}
fn perm_swap(i: usize, j: usize) -> Real4 {
    let mut p = Real4::identity();
    if i != j {
        p[(i, i)] = 0.0;
        p[(j, j)] = 0.0;
        p[(i, j)] = 1.0;
        p[(j, i)] = -1.0;
    }
    p
}
/// Real orthonormal eigenvectors of the symmetric unitary `s = U diag(d) U^T`, from its spectral
/// projectors (polynomials in `s`), so repeated eigenvalues are handled exactly.
fn real_eigvecs(s: &Mat4, d: &[C; 4]) -> Option<Real4> {
    let mut rep = [0usize; 4];
    let mut ng = 0;
    let mut grp = [0usize; 4];
    for k in 0..4 {
        match (0..ng).find(|&g| (d[rep[g]] - d[k]).norm() < 1e-8) {
            Some(g) => grp[k] = g,
            None => {
                rep[ng] = k;
                grp[k] = ng;
                ng += 1;
            }
        }
    }
    let mut u = Real4::zeros();
    let mut taken = [[0.0f64; 4]; 4];
    let mut nt = 0;
    for g in 0..ng {
        let mut p = Mat4::identity();
        for h in 0..ng {
            if h != g {
                let mut sh = *s;
                for i in 0..4 {
                    sh[(i, i)] -= d[rep[h]];
                }
                p = sh * p;
                let inv = 1.0 / (d[rep[g]] - d[rep[h]]);
                p *= inv;
            }
        }
        let mut cols = [[0.0f64; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                if p[(r, c)].im.abs() > 1e-6 {
                    return None;
                }
                cols[c][r] = p[(r, c)].re;
            }
        }
        for k in 0..4 {
            if grp[k] != g {
                continue;
            }
            for v in &taken[..nt] {
                for c in cols.iter_mut() {
                    let dd: f64 = (0..4).map(|r| v[r] * c[r]).sum();
                    for r in 0..4 {
                        c[r] -= dd * v[r];
                    }
                }
            }
            let (bi, bn) = cols
                .iter()
                .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
                .enumerate()
                .fold((0, 0.0), |a, (i, n)| if n > a.1 { (i, n) } else { a });
            if bn < 1e-7 {
                return None;
            }
            let v: [f64; 4] = std::array::from_fn(|r| cols[bi][r] / bn);
            for r in 0..4 {
                u[(r, k)] = v[r];
            }
            taken[nt] = v;
            nt += 1;
        }
    }
    Some(u)
}
/// `O` for `(d1, d2; lam_t)` from a solution `V` of `spec(lam_t V d2^{-1} V^T) = d1`.
fn reconstruct_from_role1(d1: &[C; 4], d2: &[C; 4], lam_t: &[C; 4], v: &Real4) -> Option<Real4> {
    let lt: [C; 4] = std::array::from_fn(|k| lam_t[k].sqrt());
    let d2h: [C; 4] = std::array::from_fn(|k| d2[k].sqrt());
    let d1h: [C; 4] = std::array::from_fn(|k| d1[k].sqrt());
    let vm = complex(v);
    let z = Mat4::from_diagonal(&nalgebra::Vector4::new(lt[0], lt[1], lt[2], lt[3]))
        * vm
        * Mat4::from_diagonal(&nalgebra::Vector4::new(
            1.0 / d2h[0],
            1.0 / d2h[1],
            1.0 / d2h[2],
            1.0 / d2h[3],
        ));
    let s = z * z.transpose();
    let u = real_eigvecs(&s, d1)?;
    let um = complex(&u);
    let o = Mat4::from_diagonal(&nalgebra::Vector4::new(
        1.0 / d1h[0],
        1.0 / d1h[1],
        1.0 / d1h[2],
        1.0 / d1h[3],
    )) * um.transpose()
        * z;
    if o.iter().any(|z| z.im.abs() > 1e-6) {
        return None;
    }
    let mut or = o.map(|z| z.re);
    if or.determinant() < 0.0 {
        or.row_mut(0).neg_mut();
    }
    Some(or)
}

/// Map a frame reconstructed in one of the three spectral roles back to the
/// production role.  Both the chart fast path and the exact leaf path use the
/// same role transport; keeping it here prevents their transpose conventions
/// from drifting apart.
fn reconstruct_role(
    role: usize,
    d1: &[C; 4],
    d2: &[C; 4],
    target: &[C; 4],
    frame: &Real4,
) -> Option<Real4> {
    match role {
        0 => Some(*frame),
        1 => reconstruct_from_role1(d1, d2, target, frame),
        _ => reconstruct_from_role1(d2, d1, target, frame).map(|x| x.transpose()),
    }
}

/// Shared chart candidate gate.  All chart/leaf constructors produce a real
/// matrix that is only approximately orthogonal; normalize it once, apply the
/// same coefficient pre-gate, and then use the production rootwise certificate.
fn check_chart_candidate(
    problem: &super::PreparedSandwich,
    frame: Real4,
) -> Option<super::Solution> {
    if !finite(&frame) {
        return None;
    }
    let dev = ((0..4).map(|k| frame[(0, k)] * frame[(1, k)]).sum::<f64>())
        .abs()
        .max(((0..4).map(|k| frame[(2, k)] * frame[(3, k)]).sum::<f64>()).abs())
        .max(((0..4).map(|k| frame[(0, k)] * frame[(0, k)]).sum::<f64>() - 1.0).abs());
    let mut frame = if dev > 1e-13 { polar(&frame) } else { frame };
    if frame.determinant() < 0.0 {
        frame.row_mut(0).neg_mut();
    }
    let (f1, f2) = char_e1e2(&problem.left, &problem.right, &frame);
    let taus = [
        e1e2_of_spectrum(&problem.target_roots[0]),
        e1e2_of_spectrum(&problem.target_roots[1]),
    ];
    let residual = taus
        .iter()
        .map(|t| (f1 - C::new(t[0], t[1])).norm() + (f2 - C::new(t[2], 0.0)).norm())
        .fold(f64::INFINITY, f64::min);
    (residual < 1e-5)
        .then(|| {
            certified(
                problem,
                &problem.left,
                &problem.right,
                &problem.target_roots,
                &frame,
                residual,
            )
        })
        .flatten()
}

/// Vertices of the convex hull of four unit-circle points, in angular order.
fn hull_of(l: &[C; 4]) -> Vec<(f64, f64)> {
    let ang: [f64; 4] = std::array::from_fn(|k| l[k].im.atan2(l[k].re));
    let mut idx = [0usize, 1, 2, 3];
    idx.sort_unstable_by(|&a, &b| ang[a].partial_cmp(&ang[b]).unwrap());
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(4);
    for &k in &idx {
        let p = (l[k].re, l[k].im);
        if pts
            .last()
            .is_none_or(|q: &(f64, f64)| (q.0 - p.0).abs() + (q.1 - p.1).abs() >= 1e-12)
        {
            pts.push(p);
        }
    }
    pts
}
fn nr_interval(lam_a: C, mu_j: C, mu_l: C, hull: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = hull.len();
    if n < 3 {
        return Some((0.0, 1.0));
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let z0 = lam_a * mu_j;
    let dz = lam_a * (mu_l - mu_j);
    for t in 0..n {
        let (p, q) = (hull[t], hull[if t + 1 == n { 0 } else { t + 1 }]);
        let (ex, ey) = (q.0 - p.0, q.1 - p.1);
        let c0 = ex * (z0.im - p.1) - ey * (z0.re - p.0);
        let c1 = ex * dz.im - ey * dz.re;
        if c1.abs() < 1e-15 {
            if c0 < -1e-12 {
                return None;
            }
            continue;
        }
        let r = -c0 / c1;
        if c1 > 0.0 {
            lo = lo.max(r);
        } else {
            hi = hi.min(r);
        }
    }
    (lo < hi).then_some((lo, hi))
}

// -------------------------------------------------------------- the chart
// Chart (i; i,k): row i of O in the coordinate plane (i,k), t' = sin^2 of the angle.  Both
// characteristic coefficients are linear in the 3x3 orthostochastic A = R o R, with coefficient
// rows affine in t': rows(t') = R0 + t' R1, rhs(t') = q0 + t' q1 (four free coordinates of A).
struct Chart {
    i: usize,
    k: usize,
    r0: [[f64; 4]; 3],
    r1: [[f64; 4]; 3],
    q0: [f64; 3],
    q1: [f64; 3],
    precise: Option<[PreciseRows; 2]>,
}
fn embed(i: usize, r: &M3) -> Real4 {
    let o: Vec<usize> = (0..4).filter(|&j| j != i).collect();
    let mut v = Real4::zeros();
    v[(i, i)] = 1.0;
    for a in 0..3 {
        for b in 0..3 {
            v[(o[a], o[b])] = r[(a, b)];
        }
    }
    v
}
fn a_from_x(x: &[f64; 4]) -> M3 {
    let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
    M3::new(
        x1,
        x2,
        1.0 - x1 - x2,
        x3,
        x4,
        1.0 - x3 - x4,
        1.0 - x1 - x3,
        1.0 - x2 - x4,
        x1 + x2 + x3 + x4 - 1.0,
    )
}
fn restrict(cm: &[[f64; 3]; 3]) -> ([f64; 4], f64) {
    const ENT: [([f64; 4], f64); 9] = [
        ([1., 0., 0., 0.], 0.),
        ([0., 1., 0., 0.], 0.),
        ([-1., -1., 0., 0.], 1.),
        ([0., 0., 1., 0.], 0.),
        ([0., 0., 0., 1.], 0.),
        ([0., 0., -1., -1.], 1.),
        ([-1., 0., -1., 0.], 1.),
        ([0., -1., 0., -1.], 1.),
        ([1., 1., 1., 1.], -1.),
    ];
    let mut row = [0.0; 4];
    let mut k0 = 0.0;
    for j in 0..3 {
        for l in 0..3 {
            let (r, c0) = ENT[j * 3 + l];
            for m in 0..4 {
                row[m] += cm[j][l] * r[m];
            }
            k0 += cm[j][l] * c0;
        }
    }
    (row, k0)
}
fn forms_at(
    lam: &[C; 4],
    mu: &[C; 4],
    i: usize,
    k: usize,
    tp: f64,
) -> (C, [[C; 3]; 3], [[C; 3]; 3]) {
    let mut others = [0usize; 3];
    let mut n = 0;
    for j in 0..4 {
        if j != i {
            others[n] = j;
            n += 1;
        }
    }
    let kidx = (0..3).find(|&q| others[q] == k).unwrap();
    let a = mu[i] * (1.0 - tp) + mu[k] * tp;
    let cc = mu[i] * tp + mu[k] * (1.0 - tp);
    let b2 = (mu[i] - mu[k]) * (mu[i] - mu[k]) * (tp * (1.0 - tp));
    let mut mup: [C; 3] = std::array::from_fn(|q| mu[others[q]]);
    mup[kidx] = cc;
    let lo: [C; 3] = std::array::from_fn(|q| lam[others[q]]);
    let mut c1 = [[C::default(); 3]; 3];
    let mut c2 = [[C::default(); 3]; 3];
    for j in 0..3 {
        for l in 0..3 {
            c1[j][l] = lo[j] * mup[l];
            c2[j][l] += lam[i] * lo[j] * a * mup[l];
            if l == kidx {
                c2[j][l] -= lam[i] * lo[j] * b2;
            }
        }
    }
    let comp = |x: usize, y: usize| 3 - x - y;
    for j in 0..3 {
        for kk in (j + 1)..3 {
            for l1 in 0..3 {
                for l2 in (l1 + 1)..3 {
                    c2[comp(j, kk)][comp(l1, l2)] += lo[j] * lo[kk] * mup[l1] * mup[l2];
                }
            }
        }
    }
    (lam[i] * a, c1, c2)
}
impl Chart {
    fn new(lam: &[C; 4], mu: &[C; 4], i: usize, k: usize, tau: [f64; 3], target: &[C; 4]) -> Chart {
        let clustered = [lam, mu, target]
            .iter()
            .any(|s| (0..4).any(|a| ((a + 1)..4).any(|b| (s[a] - s[b]).norm() < 1e-4)));
        if clustered {
            let (ra, qa) = super::chart_precision::rows_at(lam, mu, i, k, 0.0, target);
            let (rb, qb) = super::chart_precision::rows_at(lam, mu, i, k, 1.0, target);
            return Chart {
                i,
                k,
                r0: ra.map(|row| row.map(|x| x.value())),
                r1: std::array::from_fn(|r| std::array::from_fn(|c| (rb[r][c] - ra[r][c]).value())),
                q0: qa.map(|x| x.value()),
                q1: std::array::from_fn(|r| (qb[r] - qa[r]).value()),
                precise: Some([(ra, qa), (rb, qb)]),
            };
        }
        let rows_at = |tp: f64| -> ([[f64; 4]; 3], [f64; 3]) {
            let (k1, c1, c2) = forms_at(lam, mu, i, k, tp);
            let re = |m: &[[C; 3]; 3]| -> [[f64; 3]; 3] {
                std::array::from_fn(|j| std::array::from_fn(|l| m[j][l].re))
            };
            let im = |m: &[[C; 3]; 3]| -> [[f64; 3]; 3] {
                std::array::from_fn(|j| std::array::from_fn(|l| m[j][l].im))
            };
            let (r1, b1) = restrict(&re(&c1));
            let (r2, b2) = restrict(&im(&c1));
            let (r3, b3) = restrict(&re(&c2));
            (
                [r1, r2, r3],
                [tau[0] - k1.re - b1, tau[1] - k1.im - b2, tau[2] - b3],
            )
        };
        let (ra, qa) = rows_at(0.0);
        let (rb, qb) = rows_at(1.0);
        let mut r1 = [[0.0; 4]; 3];
        let mut q1 = [0.0; 3];
        for r in 0..3 {
            for c in 0..4 {
                r1[r][c] = rb[r][c] - ra[r][c];
            }
            q1[r] = qb[r] - qa[r];
        }
        Chart {
            i,
            k,
            r0: ra,
            r1,
            q0: qa,
            q1,
            precise: None,
        }
    }
    /// Solution lines of the affine system at `tp`: up to 4 (particular, direction) pairs.
    fn lines(&self, tp: f64) -> ([Line; 4], usize) {
        use super::chart_precision::D;
        let mut m = [[D::from(0.0); 5]; 3];
        let mut scale = 0.0f64;
        for r in 0..3 {
            let mut row_scale = 0.0f64;
            for c in 0..4 {
                m[r][c] = self.precise.as_ref().map_or(
                    D::from(self.r0[r][c] + tp * self.r1[r][c]),
                    |[(ra, _), (rb, _)]| ra[r][c] + D::from(tp) * (rb[r][c] - ra[r][c]),
                );
                row_scale = row_scale.max(m[r][c].value().abs());
            }
            m[r][4] = self.precise.as_ref().map_or(
                D::from(self.q0[r] + tp * self.q1[r]),
                |[(_, qa), (_, qb)]| qa[r] + D::from(tp) * (qb[r] - qa[r]),
            );
            if row_scale > 0.0 {
                for c in 0..5 {
                    m[r][c] = m[r][c] / D::from(row_scale);
                }
            }
            scale = scale.max(
                m[r][..4]
                    .iter()
                    .map(|x| x.value().abs())
                    .fold(0.0, f64::max),
            );
        }
        let tol = 1e-13 * (1.0 + scale);
        let mut pivots = [(0usize, 0usize); 3];
        let mut np = 0;
        let mut used_r = [false; 3];
        let mut used_c = [false; 4];
        for _ in 0..3 {
            let mut best = (0.0f64, 0, 0);
            for r in 0..3 {
                if used_r[r] {
                    continue;
                }
                for c in 0..4 {
                    if !used_c[c] && m[r][c].value().abs() > best.0 {
                        best = (m[r][c].value().abs(), r, c);
                    }
                }
            }
            if best.0 <= tol {
                break;
            }
            let (_, pr, pc) = best;
            used_r[pr] = true;
            used_c[pc] = true;
            pivots[np] = (pr, pc);
            np += 1;
            for r in 0..3 {
                if r == pr {
                    continue;
                }
                let f = m[r][pc] / m[pr][pc];
                for c in 0..5 {
                    m[r][c] = m[r][c] - f * m[pr][c];
                }
            }
        }
        let out = [([0.0; 4], [0.0; 4]); 4];
        for r in 0..3 {
            if !used_r[r] && m[r][4].value().abs() > 1e-7 * (1.0 + scale) {
                return (out, 0);
            }
        }
        let mut x0 = [0.0; 4];
        for &(pr, pc) in &pivots[..np] {
            x0[pc] = (m[pr][4] / m[pr][pc]).value();
        }
        let mut basis = [[0.0; 4]; 4];
        let mut nb = 0;
        for f in 0..4 {
            if used_c[f] {
                continue;
            }
            let mut n = [0.0; 4];
            n[f] = 1.0;
            for &(pr, pc) in &pivots[..np] {
                n[pc] = -(m[pr][f] / m[pr][pc]).value();
            }
            basis[nb] = n;
            nb += 1;
        }
        let mut out = out;
        let mut no = 0;
        for a in 0..nb {
            out[no] = (x0, basis[a]);
            no += 1;
        }
        if nb == 2 {
            out[2] = (x0, std::array::from_fn(|i| basis[0][i] + basis[1][i]));
            out[3] = (x0, std::array::from_fn(|i| basis[0][i] - basis[1][i]));
            no = 4;
        }
        (out, no.min(4))
    }
    fn heron(x0: &[f64; 4], d: &[f64; 4]) -> (M3, M3, [f64; 5]) {
        let a0 = a_from_x(x0);
        let ad = a_from_x(&[x0[0] + d[0], x0[1] + d[1], x0[2] + d[2], x0[3] + d[3]]) - a0;
        (a0, ad, Self::heron_raw(&a0, &ad))
    }
    /// Heron quartic of the line `a0 + s ad` in the parameter `s` (coefficients low to high).
    fn heron_raw(a0: &M3, ad: &M3) -> [f64; 5] {
        let quad = |l: usize| -> [f64; 3] {
            let (p0, p1) = (a0[(0, l)], ad[(0, l)]);
            let (q0, q1) = (a0[(1, l)], ad[(1, l)]);
            [p0 * q0, p0 * q1 + p1 * q0, p1 * q1]
        };
        let u = [quad(0), quad(1), quad(2)];
        let mul = |a: [f64; 3], b: [f64; 3]| {
            let mut c = [0.0; 5];
            for i in 0..3 {
                for j in 0..3 {
                    c[i + j] += a[i] * b[j];
                }
            }
            c
        };
        let mut h = [0.0; 5];
        for l in 0..3 {
            for m in (l + 1)..3 {
                let t = mul(u[l], u[m]);
                for i in 0..5 {
                    h[i] += 2.0 * t[i];
                }
            }
        }
        for l in 0..3 {
            let t = mul(u[l], u[l]);
            for i in 0..5 {
                h[i] -= t[i];
            }
        }
        h
    }
    /// Denominator-cleared Heron quartic at `tp` for a fixed pivot pattern: Cramer numerators `N`, the
    /// scaled null direction and `det` are cubic in `tp`, so these five coefficients are polynomials of
    /// degree <= 12 in `tp` and can be interpolated.
    fn cleared(&self, tp: f64, cols: [usize; 3], free: usize) -> [f64; 5] {
        let mut m = [[0.0f64; 4]; 3];
        let mut q = [0.0f64; 3];
        for r in 0..3 {
            for c in 0..4 {
                m[r][c] = self.r0[r][c] + tp * self.r1[r][c];
            }
            q[r] = self.q0[r] + tp * self.q1[r];
        }
        let det3 = |a: [[f64; 3]; 3]| {
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        };
        let base: [[f64; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| m[r][cols[c]]));
        let det = det3(base);
        let mut nn = [0.0; 4];
        let mut nh = [0.0; 4];
        nh[free] = det;
        for i in 0..3 {
            let mut mq = base;
            let mut mf = base;
            for r in 0..3 {
                mq[r][i] = q[r];
                mf[r][i] = m[r][free];
            }
            nn[cols[i]] = det3(mq);
            nh[cols[i]] = -det3(mf);
        }
        let ca = a_from_x(&[0.0; 4]);
        let a0 = a_from_x(&nn) + ca * (det - 1.0);
        let ad = a_from_x(&nh) - ca;
        Self::heron_raw(&a0, &ad)
    }
    fn disc(&self, tp: f64) -> f64 {
        let (l, n) = self.lines(tp);
        if n != 1 {
            return 0.0;
        }
        let (_, _, h) = Self::heron(&l[0].0, &l[0].1);
        let sc = h.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if !(sc > 0.0) {
            return 0.0;
        }
        quartic_disc(&std::array::from_fn(|q| h[q] / sc))
    }
    /// Frames at `tp`, handed to `emit` as they are built; stops (true) as soon as `emit` accepts one.
    fn frames(&self, tp: f64, emit: &mut dyn FnMut(Real4) -> bool) -> bool {
        let w = tp.clamp(0.0, 1.0).sqrt().asin();
        let (cw, sw) = (w.cos(), w.sin());
        let (ls, nl) = self.lines(tp);
        for (x0, d) in &ls[..nl] {
            let (mut a0, mut ad, _) = Self::heron(x0, d);
            // The free coordinate can have a very small feasible interval.
            // Evaluate the same Heron polynomial in a coordinate centered on
            // that interval, so Ferrari does not subtract large, nearly equal
            // roots to recover a tiny physical displacement.
            let mut lo = f64::NEG_INFINITY;
            let mut hi = f64::INFINITY;
            for row in 0..3 {
                for column in 0..3 {
                    let (base, direction) = (a0[(row, column)], ad[(row, column)]);
                    if direction > 0.0 {
                        lo = lo.max((-1e-6 - base) / direction);
                    } else if direction < 0.0 {
                        hi = hi.min((-1e-6 - base) / direction);
                    }
                }
            }
            if lo.is_finite() && hi.is_finite() && lo < hi {
                let center = 0.5 * (lo + hi);
                let radius = 0.5 * (hi - lo);
                a0 += center * ad;
                ad *= radius;
            }
            let h = Self::heron_raw(&a0, &ad);
            let sc = h.iter().map(|x| x.abs()).fold(0.0, f64::max);
            if !(sc > 0.0) || !sc.is_finite() {
                continue;
            }
            let hn: [f64; 5] = std::array::from_fn(|q| h[q] / sc);
            let (roots, nr) = if hn.iter().skip(1).all(|x| x.abs() <= 1e-10) {
                let mut lo = f64::NEG_INFINITY;
                let mut hi = f64::INFINITY;
                let mut ok = true;
                for r in 0..3 {
                    for c in 0..3 {
                        let (p0, p1) = (a0[(r, c)], ad[(r, c)]);
                        if p1.abs() < 1e-14 {
                            if p0 < -1e-9 {
                                ok = false;
                            }
                            continue;
                        }
                        let s0 = -p0 / p1;
                        if p1 > 0.0 {
                            lo = lo.max(s0);
                        } else {
                            hi = hi.min(s0);
                        }
                    }
                }
                if ok && lo <= hi {
                    ([0.5 * (lo + hi), 0.0, 0.0, 0.0, 0.0, 0.0], 1)
                } else {
                    ([0.0; 6], 0)
                }
            } else {
                real_roots(&hn)
            };
            for &s in &roots[..nr] {
                let a = a0 + s * ad;
                if a.iter().any(|&v| v < -1e-6) {
                    continue;
                }
                let r1 = V3::new(
                    a[(0, 0)].max(0.0).sqrt(),
                    a[(0, 1)].max(0.0).sqrt(),
                    a[(0, 2)].max(0.0).sqrt(),
                );
                let mut best: Option<(f64, V3)> = None;
                for sg in 0..4u32 {
                    let e2 = if sg & 1 == 1 { -1.0 } else { 1.0 };
                    let e3 = if sg & 2 == 2 { -1.0 } else { 1.0 };
                    let r2 = V3::new(
                        a[(1, 0)].max(0.0).sqrt(),
                        e2 * a[(1, 1)].max(0.0).sqrt(),
                        e3 * a[(1, 2)].max(0.0).sqrt(),
                    );
                    let dot = r1.dot(&r2).abs();
                    if best.is_none_or(|(b, _)| dot < b) {
                        best = Some((dot, r2));
                    }
                }
                let (dot, r2) = best.unwrap();
                if dot > 1e-4 {
                    continue;
                }
                // a vanishing entry of A gives sqrt(rounding) = 1e-8 in the frame: recompute small entries of
                // the two rows from orthogonality with the other row where that column is large (linear, exact)
                let (mut r1, mut r2) = (r1, r2);
                for n in 0..3 {
                    let (o1, o2) = ((n + 1) % 3, (n + 2) % 3);
                    if r1[n].abs() < 1e-3 && r2[n].abs() > 0.1 {
                        r1[n] = -(r1[o1] * r2[o1] + r1[o2] * r2[o2]) / r2[n];
                    } else if r2[n].abs() < 1e-3 && r1[n].abs() > 0.1 {
                        r2[n] = -(r1[o1] * r2[o1] + r1[o2] * r2[o2]) / r1[n];
                    }
                }
                let r1 = r1.normalize();
                let r2 = (r2 - r1 * r1.dot(&r2)).normalize();
                let r3 = r1.cross(&r2);
                let r = M3::from_rows(&[r1.transpose(), r2.transpose(), r3.transpose()]);
                // embed(i, R) times the Givens rotation in the (i, k) plane
                let mut o = embed(self.i, &r);
                let (i, k) = (self.i, self.k);
                for row in 0..4 {
                    let (ei, ek) = (o[(row, i)], o[(row, k)]);
                    o[(row, i)] = ei * cw + ek * sw;
                    o[(row, k)] = ek * cw - ei * sw;
                }
                if emit(o) {
                    return true;
                }
            }
        }
        false
    }
}
/// Degree-12 interpolant of the five cleared quartic coefficients on 13 Chebyshev-Lobatto nodes of [t0, t1].
struct Interp {
    t0: f64,
    t1: f64,
    nodes: [f64; 13],
    vals: [[f64; 5]; 13],
}
impl Interp {
    fn new(ch: &Chart, t0: f64, t1: f64) -> Option<Interp> {
        let nodes: [f64; 13] =
            std::array::from_fn(|j| 0.5 * (1.0 - (std::f64::consts::PI * j as f64 / 12.0).cos()));
        // the pivot pattern whose determinant stays farthest from zero across the nodes
        let det3 = |a: [[f64; 3]; 3]| {
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        };
        let mut best = (0.0f64, 0usize);
        for free in 0..4 {
            let cols: Vec<usize> = (0..4).filter(|&c| c != free).collect();
            let mut mn = f64::INFINITY;
            for j in 0..13 {
                let tp = t0 + (t1 - t0) * nodes[j];
                let mut scale = 0.0f64;
                let a: [[f64; 3]; 3] = std::array::from_fn(|r| {
                    std::array::from_fn(|c| {
                        let v = ch.r0[r][cols[c]] + tp * ch.r1[r][cols[c]];
                        scale = scale.max(v.abs());
                        v
                    })
                });
                mn = mn.min(det3(a).abs() / (scale * scale * scale).max(1e-300));
            }
            if mn > best.0 {
                best = (mn, free);
            }
        }
        if best.0 < 1e-6 {
            return None;
        }
        let free = best.1;
        let mut cols = [0usize; 3];
        let mut n = 0;
        for c in 0..4 {
            if c != free {
                cols[n] = c;
                n += 1;
            }
        }
        let vals: [[f64; 5]; 13] =
            std::array::from_fn(|j| ch.cleared(t0 + (t1 - t0) * nodes[j], cols, free));
        Some(Interp {
            t0,
            t1,
            nodes,
            vals,
        })
    }
    /// Normalized discriminant at `tp` (equal to `Chart::disc` up to rounding).
    fn disc(&self, tp: f64) -> f64 {
        let x = ((tp - self.t0) / (self.t1 - self.t0)).clamp(0.0, 1.0);
        let mut num = [0.0f64; 5];
        let mut den = 0.0f64;
        for j in 0..13 {
            let dx = x - self.nodes[j];
            if dx.abs() < 1e-15 {
                num = self.vals[j];
                den = 1.0;
                break;
            }
            let w = (if j % 2 == 0 { 1.0 } else { -1.0 })
                * (if j == 0 || j == 12 { 0.5 } else { 1.0 })
                / dx;
            den += w;
            for i in 0..5 {
                num[i] += w * self.vals[j][i];
            }
        }
        let h: [f64; 5] = std::array::from_fn(|i| num[i] / den);
        let sc = h.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if !(sc > 0.0) || !sc.is_finite() {
            return 0.0;
        }
        quartic_disc(&std::array::from_fn(|q| h[q] / sc))
    }
}
/// Chart (a; j,l) for a general lambda slot: slot swap plus the inner chart.
fn make_chart(
    lam: &[C; 4],
    mu: &[C; 4],
    a: usize,
    j: usize,
    l: usize,
    tau: [f64; 3],
    target: &[C; 4],
) -> (Chart, Option<Real4>) {
    if a == j {
        return (Chart::new(lam, mu, j, l, tau, target), None);
    }
    if a == l {
        return (Chart::new(lam, mu, l, j, tau, target), None);
    }
    let mut lp = *lam;
    lp.swap(a, j);
    (
        Chart::new(&lp, mu, j, l, tau, target),
        Some(perm_swap(a, j)),
    )
}
/// Exact selection on [t0,t1]: 33 samples of the discriminant; the centres of the positive runs are
/// tested first, then the zeros (sign changes, bisected) with the midpoints between consecutive events,
/// then the extrema next to near-zero samples (a fold that touches, or hugs an interval end).
fn exact_select(
    disc: &dyn Fn(f64) -> f64,
    t0: f64,
    t1: f64,
    cells: usize,
    mut test: impl FnMut(f64) -> bool,
) -> bool {
    let tp = |t: f64| (t0 + (t1 - t0) * t).clamp(0.0, 1.0);
    let d = |t: f64| disc(tp(t));
    let mut v = [0.0f64; 33];
    for c in 0..=cells {
        v[c] = d(c as f64 / cells as f64);
    }
    let scale = v.iter().map(|x| x.abs()).fold(0.0, f64::max);
    // stage 1: centres of the maximal runs of non-negative samples
    let mut c = 0;
    while c <= cells {
        if v[c] >= 0.0 {
            let s0 = c;
            while c < cells && v[c + 1] >= 0.0 {
                c += 1;
            }
            if test(tp(0.5 * (s0 + c) as f64 / cells as f64)) {
                return true;
            }
        }
        c += 1;
    }
    let bisect =
        |f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64, mut flo: f64, steps: usize| -> f64 {
            for _ in 0..steps {
                let mid = 0.5 * (lo + hi);
                let fm = f(mid);
                if fm == 0.0 {
                    return mid;
                }
                if fm.signum() == flo.signum() {
                    lo = mid;
                    flo = fm;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
    // stage 2: the ends, exact zeros of the samples, the full-precision sign changes (a window narrower than a
    // cell, or a touching fold) and the extrema next to near-zero samples; each tested once, no midpoints of
    // negative regions (no real roots there)
    let mut fine: Vec<f64> = vec![0.0, 1.0];
    for c in 0..cells {
        let (x0, x1) = (c as f64 / cells as f64, (c + 1) as f64 / cells as f64);
        if v[c] == 0.0 {
            fine.push(x0);
        } else if v[c].signum() != v[c + 1].signum() && v[c + 1] != 0.0 {
            fine.push(bisect(&d, x0, x1, v[c], BIS));
        }
    }
    {
        let dd = |t: f64| {
            let e = 1e-6;
            d((t + e).min(1.0)) - d((t - e).max(0.0))
        };
        let mut ddv = [f64::NAN; 33];
        for c in 0..cells {
            if v[c].abs().min(v[c + 1].abs()) > 1e-2 * scale {
                continue;
            }
            let (x0, x1) = (c as f64 / cells as f64, (c + 1) as f64 / cells as f64);
            if ddv[c].is_nan() {
                ddv[c] = dd(x0);
            }
            if ddv[c + 1].is_nan() {
                ddv[c + 1] = dd(x1);
            }
            let (f0, f1) = (ddv[c], ddv[c + 1]);
            if f0.signum() != f1.signum() && f0 != 0.0 && f1 != 0.0 {
                fine.push(bisect(&dd, x0, x1, f0, BIS));
            }
        }
    }
    fine.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fine.dedup_by(|a, b| (*a - *b).abs() < 1e-13);
    for q in 0..fine.len() {
        if test(tp(fine[q])) {
            return true;
        }
        // the midpoint between two consecutive events only where a window narrower than a cell can hide
        if q + 1 < fine.len()
            && fine[q + 1] - fine[q] < 2.0 / cells as f64
            && test(tp(0.5 * (fine[q] + fine[q + 1])))
        {
            return true;
        }
    }
    false
}

// ----------------------------------------------------------------- driver
/// Block frames: block traces are the roots of `z^2 - e1 z + (e2 - det1 - det2)`.
fn block_leaves(lam: &[C; 4], mu: &[C; 4], e1t: C, e2t: C) -> Vec<Real4> {
    let mut out = vec![];
    let parts: [[usize; 4]; 3] = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]];
    for pl in &parts {
        for pm in &parts {
            for matching in 0..2 {
                let lb = [[pl[0], pl[1]], [pl[2], pl[3]]];
                let mb = if matching == 0 {
                    [[pm[0], pm[1]], [pm[2], pm[3]]]
                } else {
                    [[pm[2], pm[3]], [pm[0], pm[1]]]
                };
                let alpha = |b: usize| lam[lb[b][0]] * mu[mb[b][0]] + lam[lb[b][1]] * mu[mb[b][1]];
                let beta = |b: usize| lam[lb[b][0]] * mu[mb[b][1]] + lam[lb[b][1]] * mu[mb[b][0]];
                let detb = |b: usize| lam[lb[b][0]] * lam[lb[b][1]] * mu[mb[b][0]] * mu[mb[b][1]];
                let cprod = e2t - detb(0) - detb(1);
                let disc0 = (e1t * e1t - cprod * 4.0).sqrt();
                // a boundary target has a double block trace: the exact root e1/2 is tried as well
                for order in 0..3 {
                    let disc = if order == 2 { C::default() } else { disc0 };
                    let t = if order == 0 {
                        [(e1t + disc) * 0.5, (e1t - disc) * 0.5]
                    } else {
                        [(e1t - disc) * 0.5, (e1t + disc) * 0.5]
                    };
                    let mut xs = [0.5; 2];
                    let mut ok = true;
                    for b in 0..2 {
                        let d = alpha(b) - beta(b);
                        if d.norm() < 1e-12 {
                            if (t[b] - beta(b)).norm() > 1e-6 {
                                ok = false;
                            }
                            continue;
                        }
                        let x = (t[b] - beta(b)) / d;
                        if x.im.abs() > 1e-6 || x.re < -1e-8 || x.re > 1.0 + 1e-8 {
                            ok = false;
                            break;
                        }
                        xs[b] = x.re.clamp(0.0, 1.0);
                    }
                    if !ok {
                        continue;
                    }
                    let mut o = Real4::zeros();
                    for b in 0..2 {
                        let (c, sn) = (xs[b].sqrt(), (1.0 - xs[b]).sqrt());
                        o[(lb[b][0], mb[b][0])] = c;
                        o[(lb[b][0], mb[b][1])] = -sn;
                        o[(lb[b][1], mb[b][0])] = sn;
                        o[(lb[b][1], mb[b][1])] = c;
                    }
                    out.push(o);
                }
            }
        }
    }
    out
}
/// One accepted chart record: residual, chart indices, lift, role, and the `t'` window.
type Rec = (f64, u8, u8, u8, u8, f64, f64);
/// Wall-rule candidate: (|sin m|, lift, swap, a, j, k, end, root).
type WallCand = (f64, usize, u8, u8, u8, u8, u8, u8);
/// One (role, lift) working set: spectra, target coefficients, and its accepted records.
type RoleSet = (usize, usize, [C; 4], [C; 4], [f64; 3], Vec<Rec>);
fn solve_charts_snapped(
    problem: &super::PreparedSandwich,
    lamc: [C; 4],
    muc: [C; 4],
    liftsc: [[C; 4]; 2],
) -> Option<super::Solution> {
    solve_charts_gated(problem, 2, lamc, muc, liftsc, true)
}

/// Charts built from `lam, mu, lifts` (snapped to the nearest stratum on the second pass); every
/// candidate is checked against the problem's true spectra. `mode` 1 stops after the fast and wall
/// passes, 2 continues with exact selection over every chart. `allow_clustered` admits spectra with
/// two roots within 1e-4.
fn solve_charts_gated(
    problem: &super::PreparedSandwich,
    mode: u8,
    lam: [C; 4],
    mu: [C; 4],
    lifts: [[C; 4]; 2],
    allow_clustered: bool,
) -> Option<super::Solution> {
    let gap = |l: &[C; 4]| {
        (0..4)
            .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
            .map(|(i, j)| (l[i] - l[j]).norm())
            .fold(f64::INFINITY, f64::min)
    };
    if !allow_clustered && gap(&lam).min(gap(&mu)).min(gap(&lifts[0])) < 1e-4 {
        return None;
    }
    let conj4 = |d: &[C; 4]| -> [C; 4] { std::array::from_fn(|q| d[q].conj()) };
    let role_data = |role: usize, lift: usize| -> ([C; 4], [C; 4], [f64; 3], [C; 4]) {
        let lam_t = lifts[lift];
        match role {
            0 => (lam, mu, e1e2_of_spectrum(&lam_t), lam_t),
            1 => (lam_t, conj4(&mu), e1e2_of_spectrum(&lam), lam),
            _ => (lam_t, conj4(&lam), e1e2_of_spectrum(&mu), mu),
        }
    };
    let records = |l0: &[C; 4], m0: &[C; 4], lt_r: &[C; 4]| -> Vec<Rec> {
        let hull = hull_of(lt_r);
        let mut raw = [(0.0f64, 0u8, 0u8, 0u8, 0u8, 0.0f64, 0.0f64); 48];
        let mut nr = 0;
        let mut wmax = 0.0f64;
        for swap in 0..2u8 {
            let (l, m) = if swap == 0 { (l0, m0) } else { (m0, l0) };
            for a in 0..4u8 {
                for j in 0..4u8 {
                    for k in (j + 1)..4u8 {
                        if let Some((t0, t1)) =
                            nr_interval(l[a as usize], m[j as usize], m[k as usize], &hull)
                        {
                            raw[nr] = (t1 - t0, swap, a, j, k, t0, t1);
                            nr += 1;
                            wmax = wmax.max(t1 - t0);
                        }
                    }
                }
            }
        }
        if nr == 0 {
            for swap in 0..2u8 {
                for a in 0..4u8 {
                    for j in 0..4u8 {
                        for k in (j + 1)..4u8 {
                            raw[nr] = (1.0, swap, a, j, k, 0.0, 1.0);
                            nr += 1;
                        }
                    }
                }
            }
            wmax = 1.0;
        }
        let mut recs: Vec<Rec> = vec![raw[0]; nr];
        let inv = 7.999 / wmax.max(1e-300);
        let mut bkt = [0u8; 48];
        let mut cnt = [0usize; 9];
        for r in 0..nr {
            let b = (raw[r].0 * inv) as usize;
            bkt[r] = b as u8;
            cnt[b + 1] += 1;
        }
        for b in 0..8 {
            cnt[b + 1] += cnt[b];
        }
        for r in 0..nr {
            let b = bkt[r] as usize;
            recs[cnt[b]] = raw[r];
            cnt[b] += 1;
        }
        recs
    };
    let map_back = |role: usize, lift: usize, swap: u8, oc: Real4| -> Option<Real4> {
        let lam_t = lifts[lift];
        let oc = if swap == 0 { oc } else { oc.transpose() };
        reconstruct_role(role, &lam, &mu, &lam_t, &oc)
    };
    let try_chart = |role: usize,
                     lift: usize,
                     l: &[C; 4],
                     m: &[C; 4],
                     tau: [f64; 3],
                     rec: &Rec,
                     f: f64|
     -> Option<super::Solution> {
        let &(_, swap, a, j, k, t0, t1) = rec;
        let (l, m) = if swap == 0 { (l, m) } else { (m, l) };
        let lt_r = role_data(role, lift).3;
        let tp = if f == CHORD {
            chord_point(l, m, &lt_r, a as usize, j as usize, k as usize, (t0, t1))
        } else {
            (t0 + (t1 - t0) * f).clamp(0.0, 1.0)
        };
        let (ch, pm) = make_chart(l, m, a as usize, j as usize, k as usize, tau, &lt_r);
        let mut hit = None;
        ch.frames(tp, &mut |oc| {
            let oc = pm.map_or(oc, |p| p * oc);
            match map_back(role, lift, swap, oc)
                .and_then(|frame| check_chart_candidate(problem, frame))
            {
                Some(h) => {
                    hit = Some(h);
                    true
                }
                None => false,
            }
        });
        hit
    };
    let exact_chart = |role: usize,
                       lift: usize,
                       l: &[C; 4],
                       m: &[C; 4],
                       tau: [f64; 3],
                       rec: &Rec|
     -> Option<super::Solution> {
        let &(_, swap, a, j, k, t0, t1) = rec;
        let (l, m) = if swap == 0 { (l, m) } else { (m, l) };
        let (ch, pm) = make_chart(
            l,
            m,
            a as usize,
            j as usize,
            k as usize,
            tau,
            &role_data(role, lift).3,
        );
        let mut found: Option<super::Solution> = None;
        let interp = Interp::new(&ch, t0, t1);
        let disc = |tp: f64| match &interp {
            Some(ip) => ip.disc(tp),
            None => ch.disc(tp),
        };
        exact_select(&disc, t0, t1, 32, |tp| {
            ch.frames(tp, &mut |oc| {
                let oc = pm.map_or(oc, |p| p * oc);
                match map_back(role, lift, swap, oc)
                    .and_then(|frame| check_chart_candidate(problem, frame))
                {
                    Some(h) => {
                        found = Some(h);
                        true
                    }
                    None => false,
                }
            })
        });
        found
    };
    // Phase W: the second-order wall rule (R0224, R0227) as the first probe.  For role 0, both lifts,
    // both swaps, every record (a; j, k) and both block ends, the target root nearest the split-off
    // product gives the phase margin m.  The block frames of the chart at that end on the target
    // projected onto the wall are points of the 3x3 core curve; the fibre leaves them at
    // t' = m / kappa(R_1) - m^2 kappa_2 / kappa^3, with kappa the exact first-order phase rate of the
    // split-off root, kappa_2 its second-order term and R_1 the first-order transverse shift of the
    // block frame.  Candidates are ranked by |sin m| and the first WALL_PROBES are probed.
    {
        const WALL_PROBES: usize = 4;
        let mut pre: Vec<WallCand> = Vec::with_capacity(192);
        for lift in 0..2 {
            let (l0, m0, _tau, lt_r) = role_data(0, lift);
            for swap in 0..2u8 {
                let (l, m) = if swap == 0 { (l0, m0) } else { (m0, l0) };
                for a in 0..4 {
                    for j in 0..4 {
                        for k in (j + 1)..4 {
                            if (m[k] - m[j]).norm() < 1e-12 {
                                continue;
                            }
                            // chart coordinates: row index i carries lam[a] (make_chart swaps the slots)
                            let (li, i, pp) = if a == k { (l[a], k, j) } else { (l[a], j, k) };
                            for end in 0..2u8 {
                                let rho0 = li * if end == 0 { m[i] } else { m[pp] };
                                let (n, key) = (0..4)
                                    .map(|q| {
                                        let z = lt_r[q] * rho0.conj();
                                        (
                                            q,
                                            if z.re > 0.0 {
                                                z.im.abs()
                                            } else {
                                                f64::INFINITY
                                            },
                                        )
                                    })
                                    .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                                    .unwrap();
                                if !(key > 1e-12) || !key.is_finite() {
                                    continue;
                                }
                                pre.push((
                                    key, lift, swap, a as u8, j as u8, k as u8, end, n as u8,
                                ));
                            }
                        }
                    }
                }
            }
        }
        pre.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let mut probed = 0usize;
        'cand: for &(_, lift, swap, a, j, k, end, n) in &pre {
            if probed >= WALL_PROBES {
                break;
            }
            let (l0, m0, tau, lt_r) = role_data(0, lift);
            let (l, m) = if swap == 0 { (l0, m0) } else { (m0, l0) };
            let (a, j, k, n) = (a as usize, j as usize, k as usize, n as usize);
            let (lp, i, pp) = if a == j {
                (l, j, k)
            } else if a == k {
                (l, k, j)
            } else {
                let mut lp = l;
                lp.swap(a, j);
                (lp, j, k)
            };
            let rho0 = lp[i] * if end == 0 { m[i] } else { m[pp] };
            let mrg = (lt_r[n] / rho0).arg();
            let ph = C::new(0.0, mrg / 3.0).exp();
            let proj: [C; 4] = std::array::from_fn(|q| if q == n { rho0 } else { lt_r[q] * ph });
            let tau_p = e1e2_of_spectrum(&proj);
            let (chp, _) = make_chart(&l, &m, a, j, k, tau_p, &proj);
            let mut blocks: Vec<Real4> = vec![];
            chp.frames(if end == 0 { 0.0 } else { 1.0 }, &mut |oc| {
                blocks.push(oc);
                false
            });
            if blocks.is_empty() {
                continue;
            }
            let (ch, pm) = make_chart(&l, &m, a, j, k, tau, &lt_r);
            let comp: Vec<usize> = (0..4).filter(|&q| q != i).collect();
            for oc in blocks.iter().take(2) {
                // the block frame in end-0 form (row i = +-e_i), with the partner spectrum swapped at end 1
                let (o, mu2) = if end == 0 {
                    (*oc, m)
                } else {
                    let mut o = *oc;
                    for row in 0..4 {
                        let (ei, ep) = (oc[(row, i)], oc[(row, pp)]);
                        o[(row, i)] = -ep;
                        o[(row, pp)] = ei;
                    }
                    let mut mu2 = m;
                    mu2.swap(i, pp);
                    (o, mu2)
                };
                if (o[(i, i)].abs() - 1.0).abs() > 1e-6 {
                    continue;
                }
                let (f1, f2) = char_e1e2(&lp, &mu2, &o);
                if (f1 - C::new(tau_p[0], tau_p[1])).norm() + (f2 - C::new(tau_p[2], 0.0)).norm()
                    > 1e-6
                {
                    continue;
                }
                // exact first- and second-order rates of the split-off root at this block frame
                let d = mu2[pp] - mu2[i];
                let mut w3 = nalgebra::Matrix3::<C>::zeros();
                for (x, &qx) in comp.iter().enumerate() {
                    for (y, &qy) in comp.iter().enumerate() {
                        let mut sum = C::default();
                        for q in 0..4 {
                            sum += mu2[q] * (o[(qx, q)] * o[(qy, q)]);
                        }
                        w3[(x, y)] = lp[qx] * sum;
                    }
                }
                let r = nalgebra::Vector3::<C>::from_iterator(
                    comp.iter().map(|&qx| C::new(o[(qx, pp)], 0.0)),
                );
                let d1r = nalgebra::Vector3::<C>::from_iterator(
                    comp.iter().map(|&qx| lp[qx] * o[(qx, pp)]),
                );
                let Some(q) = (nalgebra::Matrix3::<C>::identity() * rho0 - w3).try_inverse() else {
                    continue;
                };
                let g0 = (r.transpose() * q * d1r)[(0, 0)];
                let h = (r.transpose() * q * q * d1r)[(0, 0)];
                let kk = lp[i] * d * (C::new(1.0, 0.0) + d * g0);
                let k2 = lp[i] * d * d * (-d * g0 * g0 - g0 - h * kk);
                let psi1 = -d * (r.transpose() * d1r)[(0, 0)] - lp[i] * d * d * g0;
                let kap = (kk / rho0).im;
                let kap2 = (k2 / rho0).im;
                if !kap.is_finite() || kap == 0.0 {
                    continue;
                }
                // first-order transverse shift of the block frame: dPhi[X_1] = phi_1 - Psi_1 / kappa
                let mut rr = M3::zeros();
                for (x, &qx) in comp.iter().enumerate() {
                    for (y, &qy) in comp.iter().enumerate() {
                        rr[(x, y)] = o[(qx, qy)];
                    }
                }
                let rc = rr.map(|x| C::new(x, 0.0));
                let d1 = nalgebra::Matrix3::<C>::from_diagonal(&nalgebra::Vector3::from_iterator(
                    comp.iter().map(|&q| lp[q]),
                ));
                let m0d = nalgebra::Matrix3::<C>::from_diagonal(&nalgebra::Vector3::from_iterator(
                    comp.iter().map(|&q| mu2[q]),
                ));
                let core = rc.transpose() * d1 * rc;
                let gens: [M3; 3] = std::array::from_fn(|g| {
                    let (u, w) = ((g + 1) % 3, (g + 2) % 3);
                    let mut x = M3::zeros();
                    x[(u, w)] = 1.0;
                    x[(w, u)] = -1.0;
                    x
                });
                let dphi: [C; 3] = std::array::from_fn(|g| {
                    let x = gens[g].map(|v| C::new(v, 0.0));
                    (core * (x * m0d - m0d * x)).trace()
                });
                let phi1 = C::new(0.0, -1.0 / 3.0)
                    * (0..4)
                        .filter(|&q| q != n)
                        .map(|q| lt_r[q])
                        .fold(C::default(), |acc, z| acc + z);
                let b = phi1 - psi1 / kap;
                let aa = nalgebra::Matrix2x3::<f64>::from_rows(&[
                    nalgebra::RowVector3::new(dphi[0].re, dphi[1].re, dphi[2].re),
                    nalgebra::RowVector3::new(dphi[0].im, dphi[1].im, dphi[2].im),
                ]);
                let x1 = match (aa * aa.transpose()).try_inverse() {
                    Some(inv) => aa.transpose() * inv * nalgebra::Vector2::new(b.re, b.im),
                    None => nalgebra::Vector3::zeros(),
                };
                let xm = (gens[0] * x1[0] + gens[1] * x1[1] + gens[2] * x1[2]) * mrg;
                let th = (x1.norm() * mrg.abs()).max(1e-300);
                let ex = M3::identity()
                    + xm * (th.sin() / th)
                    + xm * xm * ((1.0 - th.cos()) / (th * th));
                let r1 = rr * ex;
                let mut o1 = o;
                for (x, &qx) in comp.iter().enumerate() {
                    for (y, &qy) in comp.iter().enumerate() {
                        o1[(qx, qy)] = r1[(x, y)];
                    }
                }
                let mut w31 = nalgebra::Matrix3::<C>::zeros();
                for (x, &qx) in comp.iter().enumerate() {
                    for (y, &qy) in comp.iter().enumerate() {
                        let mut sum = C::default();
                        for q in 0..4 {
                            sum += mu2[q] * (o1[(qx, q)] * o1[(qy, q)]);
                        }
                        w31[(x, y)] = lp[qx] * sum;
                    }
                }
                let r1v = nalgebra::Vector3::<C>::from_iterator(
                    comp.iter().map(|&qx| C::new(o1[(qx, pp)], 0.0)),
                );
                let d1r1 = nalgebra::Vector3::<C>::from_iterator(
                    comp.iter().map(|&qx| lp[qx] * o1[(qx, pp)]),
                );
                let Some(q1) = (nalgebra::Matrix3::<C>::identity() * rho0 - w31).try_inverse()
                else {
                    continue;
                };
                let g1 = (r1v.transpose() * q1 * d1r1)[(0, 0)];
                let kap1 = (lp[i] * d * (C::new(1.0, 0.0) + d * g1) / rho0).im;
                if !kap1.is_finite() || kap1 == 0.0 {
                    continue;
                }
                let t = mrg / kap1 - mrg * mrg * kap2 / (kap * kap * kap);
                if !(t > 0.0 && t < 1.0) {
                    continue;
                }
                let tp = if end == 0 { t } else { 1.0 - t };
                probed += 1;
                let mut hit = None;
                ch.frames(tp, &mut |oc| {
                    let oc = pm.map_or(oc, |p| p * oc);
                    match map_back(0, lift, swap, oc)
                        .and_then(|frame| check_chart_candidate(problem, frame))
                    {
                        Some(h) => {
                            hit = Some(h);
                            true
                        }
                        None => false,
                    }
                });
                if hit.is_some() {
                    return hit;
                }
                if probed >= WALL_PROBES {
                    break 'cand;
                }
            }
        }
    }
    // phase A: the three slot-native charts (0; 0,k) at their chord points
    let tpa = super::prof::start();
    {
        let (l0, m0, tau, lt_r) = role_data(0, 0);
        let hull = hull_of(&lt_r);
        let ivs: [Option<(f64, f64)>; 3] =
            std::array::from_fn(|k| nr_interval(l0[0], m0[0], m0[k + 1], &hull));
        let mut ord = [1u8, 2, 3];
        ord.sort_unstable_by(|&x, &y| {
            let w = |k: u8| ivs[k as usize - 1].map_or(-1.0, |(a, b)| b - a);
            w(y).partial_cmp(&w(x)).unwrap()
        });
        for k in ord {
            if let Some((t0, t1)) = ivs[k as usize - 1] {
                let rec: Rec = (t1 - t0, 0, 0, 0, k, t0, t1);
                if let Some(h) = try_chart(0, 0, &l0, &m0, tau, &rec, CHORD) {
                    super::prof::rec(super::prof::CHART_PHASEA, tpa);
                    return Some(h);
                }
            }
        }
    }
    super::prof::rec(super::prof::CHART_PHASEA, tpa);
    // the six (role, lift) sets, most productive first
    let schedule: [(usize, usize); 6] = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (2, 1)];
    let mut sets: Vec<RoleSet> = Vec::with_capacity(6);
    for (si, &(role, lift)) in schedule.iter().enumerate() {
        let (l0, m0, tau, lt_r) = role_data(role, lift);
        let recs = records(&l0, &m0, &lt_r);
        let fr: &[f64] = if si == 0 { &FRACTIONS } else { &FRACTIONS[..1] };
        for (fi, &f) in fr.iter().enumerate() {
            for rec in &recs {
                if fi == 0 && si == 0 && rec.1 == 0 && rec.2 == 0 && rec.3 == 0 {
                    continue;
                }
                if let Some(h) = try_chart(role, lift, &l0, &m0, tau, rec, f) {
                    return Some(h);
                }
            }
        }
        sets.push((role, lift, l0, m0, tau, recs));
    }
    // Wall rung: near a reach wall the fibre is a tube around the block stratum that attains
    // it.  1+3: a target root within O(margin) of lam_a mu_b; block frames over the projected target are the
    // chart (a; b, l) at t' = 0 and the second-order eigenvalue shift puts the solution at
    // t' = (lt_k - lam_a mu_b) / [lam_a (mu_l - mu_b) (1 + (mu_l - mu_b) r^T (lam_a mu_b - M')^{-1} Lambda' r)];
    // 2+2: a root-pair product within O(margin) of a block determinant; block frames from the trace quadratic
    // lie on the charts through their rows and columns at t' = O_al^2.  Without the reach DP's facet (it lives
    // in gulps-core, which depends on this crate) the candidates are the nearest products of every root.
    for lift in 0..2 {
        let lt = lifts[lift];
        let (l0, m0, tau, lt_r) = role_data(0, lift);
        let hull = hull_of(&lt_r);
        let mut c13: Vec<(f64, usize, usize, usize)> = vec![];
        for a in 0..4 {
            for b in 0..4 {
                for k in 0..4 {
                    let d = (lt[k] - l0[a] * m0[b]).norm();
                    if d < 0.15 {
                        c13.push((d, a, b, k));
                    }
                }
            }
        }
        c13.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        for &(_, a, b, k) in c13.iter().take(4) {
            let r3 = (l0[a] * m0[b] / lt[k]).powf(1.0 / 3.0);
            let mut lts = lt;
            lts[k] = l0[a] * m0[b];
            for q in 0..4 {
                if q != k {
                    lts[q] /= r3;
                }
            }
            let taus_p = e1e2_of_spectrum(&lts);
            for l in 0..4 {
                if l == b {
                    continue;
                }
                let Some((t0, t1)) = nr_interval(l0[a], m0[b], m0[l], &hull) else {
                    continue;
                };
                let (chs, pms) = make_chart(&l0, &m0, a, b, l, taus_p, &lts);
                let mut blocks: Vec<Real4> = vec![];
                chs.frames(0.0, &mut |oc| {
                    blocks.push(pms.map_or(oc, |p| p * oc));
                    false
                });
                if blocks.is_empty() {
                    continue;
                }
                let (ch, pm) = make_chart(&l0, &m0, a, b, l, tau, &lt_r);
                let rows: Vec<usize> = (0..4).filter(|&c| c != a).collect();
                let cols: Vec<usize> = (0..4).filter(|&c| c != b).collect();
                for o in blocks.iter().take(4) {
                    let mut mpc = [[C::default(); 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            let mut sum = C::default();
                            for d in 0..3 {
                                sum += m0[cols[d]] * o[(rows[i], cols[d])] * o[(rows[j], cols[d])];
                            }
                            mpc[i][j] = l0[rows[i]] * sum;
                        }
                    }
                    let z = l0[a] * m0[b];
                    let a3 = nalgebra::Matrix3::<C>::from_fn(|i, j| {
                        if i == j {
                            z - mpc[i][j]
                        } else {
                            -mpc[i][j]
                        }
                    });
                    let Some(inv) = a3.try_inverse() else {
                        continue;
                    };
                    let rv = nalgebra::Vector3::<C>::from_fn(|i, _| C::new(o[(rows[i], l)], 0.0));
                    let lr = nalgebra::Vector3::<C>::from_fn(|i, _| l0[rows[i]] * o[(rows[i], l)]);
                    let quad = (rv.transpose() * inv * lr)[(0, 0)];
                    let dm = m0[l] - m0[b];
                    let tp = ((lt[k] - z) / (l0[a] * dm * (C::new(1.0, 0.0) + dm * quad))).re;
                    if !(tp > 0.0) {
                        continue;
                    }
                    let mut hit = None;
                    ch.frames(tp, &mut |oc| {
                        let oc = pm.map_or(oc, |p| p * oc);
                        match map_back(0, lift, 0, oc)
                            .and_then(|frame| check_chart_candidate(problem, frame))
                        {
                            Some(h) => {
                                hit = Some(h);
                                true
                            }
                            None => false,
                        }
                    });
                    if let Some(h) = hit {
                        return Some(h);
                    }
                    let rec: Rec = (
                        t1 - t0,
                        0,
                        a as u8,
                        b as u8,
                        l as u8,
                        (0.5 * tp).max(t0),
                        (1.5 * tp).min(t1),
                    );
                    if rec.5 < rec.6 {
                        if let Some(h) = exact_chart(0, lift, &l0, &m0, tau, &rec) {
                            return Some(h);
                        }
                    }
                }
            }
        }
        let pairs = [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let mut c22: Vec<(f64, usize, usize, usize)> = vec![];
        for (pa, &(a, a2)) in pairs.iter().enumerate() {
            for (pb_, &(b, b2)) in pairs.iter().enumerate() {
                for (pk, &(k, k2)) in pairs.iter().enumerate() {
                    let d = (lt[k] * lt[k2] - l0[a] * l0[a2] * m0[b] * m0[b2]).norm();
                    if d < 0.15 {
                        c22.push((d, pa, pb_, pk));
                    }
                }
            }
        }
        c22.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        for &(d, pa, pb_, pk) in c22.iter().take(3) {
            let (k, k2) = pairs[pk];
            let (a, a2) = pairs[pa];
            let (b, b2) = pairs[pb_];
            let r = (l0[a] * l0[a2] * m0[b] * m0[b2] / (lt[k] * lt[k2])).sqrt();
            let mut lts = lt;
            lts[k] *= r;
            lts[k2] *= r;
            for q in 0..4 {
                if q != k && q != k2 {
                    lts[q] /= r;
                }
            }
            let taus_p = e1e2_of_spectrum(&lts);
            let (e1t, e2t) = (C::new(taus_p[0], taus_p[1]), C::new(taus_p[2], 0.0));
            let mut budget = 8usize;
            for o in block_leaves(&l0, &m0, e1t, e2t).into_iter().take(4) {
                for swap in 0..2u8 {
                    let (ot, ls, ms) = if swap == 0 {
                        (o, l0, m0)
                    } else {
                        (o.transpose(), m0, l0)
                    };
                    for a in 0..4 {
                        let row = ot.row(a);
                        let mut idx: Vec<usize> = (0..4).filter(|&c| row[c].abs() > 1e-6).collect();
                        if idx.len() != 2 {
                            continue;
                        }
                        idx.sort_by(|&x, &y| row[y].abs().partial_cmp(&row[x].abs()).unwrap());
                        let (j, l) = (idx[0], idx[1]);
                        let ts = row[l] * row[l];
                        let Some((t0, t1)) = nr_interval(ls[a], ms[j], ms[l], &hull) else {
                            continue;
                        };
                        let rec: Rec = (t1 - t0, swap, a as u8, j as u8, l as u8, t0, t1);
                        if ts >= t0 && ts <= t1 {
                            if let Some(h) =
                                try_chart(0, lift, &l0, &m0, tau, &rec, (ts - t0) / (t1 - t0))
                            {
                                return Some(h);
                            }
                        }
                        let rec: Rec = (
                            t1 - t0,
                            swap,
                            a as u8,
                            j as u8,
                            l as u8,
                            (ts - 1.5 * d).max(t0),
                            (ts + 1.5 * d).min(t1),
                        );
                        if rec.5 < rec.6 && budget > 0 {
                            budget -= 1;
                            if let Some(h) = exact_chart(0, lift, &l0, &m0, tau, &rec) {
                                return Some(h);
                            }
                        }
                    }
                }
            }
        }
    }
    if mode != 2 {
        return None;
    }
    // exact selection over every chart, sets in schedule order, records narrowest first
    for (role, lift, l0, m0, tau, recs) in &sets {
        for rec in recs {
            if let Some(h) = exact_chart(*role, *lift, l0, m0, *tau, rec) {
                return Some(h);
            }
        }
    }
    None
}

/// Production certificate behind the standalone solver's cheap rootwise gate: the Sylvester-projector
/// rootwise residual of the master (clusters as blocks, 1e-8, the standalone certificate) must pass
/// before compiler_solution is asked for the frame and the public certificate.
fn certified(
    problem: &super::PreparedSandwich,
    lam: &[C; 4],
    mu: &[C; 4],
    lifts: &[[C; 4]; 2],
    o: &Real4,
    r: f64,
) -> Option<super::Solution> {
    let d1h: [C; 4] = std::array::from_fn(|k| lam[k].sqrt());
    let mut m = Mat4::zeros();
    for a in 0..4 {
        for b in 0..4 {
            let mut v = C::default();
            for k in 0..4 {
                v += mu[k] * (o[(a, k)] * o[(b, k)]);
            }
            m[(a, b)] = d1h[a] * v * d1h[b];
        }
    }
    let rw = lifts
        .iter()
        .map(|lt| rootwise(&m, lt))
        .fold(f64::INFINITY, f64::min);
    if rw < 1e-8 {
        super::compiler_solution(problem, complex(o), super::Rung::Chart, r)
    } else {
        None
    }
}

/// Rootwise certificate (production's public standard): the symmetric unitary master
/// `S = D1^(1/2) O D2 O^T D1^(1/2)` is diagonalized against the KNOWN target roots by real spectral
/// projectors (repeated roots as one block, no divided differences inside a block); the residual of
/// `U^T S U - diag(roots)` is the rootwise error.
fn rootwise(s: &Mat4, roots: &[C; 4]) -> f64 {
    // clusters of target roots (production's cluster gate): one block each
    let mut rep = [0usize; 4];
    let mut ng = 0;
    let mut grp = [0usize; 4];
    for k in 0..4 {
        match (0..ng).find(|&g| (roots[rep[g]] - roots[k]).norm() < 1e-3) {
            Some(g) => grp[k] = g,
            None => {
                rep[ng] = k;
                grp[k] = ng;
                ng += 1;
            }
        }
    }
    let Some(u) = real_eigvecs_grouped(s, roots, &rep[..ng], &grp) else {
        return f64::INFINITY;
    };
    let mut d = Mat4::zeros();
    for r in 0..4 {
        for c in 0..4 {
            let mut v = C::default();
            for a in 0..4 {
                let mut w = C::default();
                for b in 0..4 {
                    w += s[(a, b)] * u[(b, c)];
                }
                v += u[(a, r)] * w;
            }
            d[(r, c)] = v;
        }
    }
    let mut worst = 0.0f64;
    for r in 0..4 {
        for c in 0..4 {
            if grp[r] != grp[c] {
                worst = worst.max(d[(r, c)].norm());
            }
        }
    }
    for g in 0..ng {
        let idx: Vec<usize> = (0..4).filter(|&k| grp[k] == g).collect();
        let m = idx.len();
        // the block is symmetric unitary: eigenvectors of its real part (real symmetric), eigenvalues as
        // Rayleigh quotients of the complex block (validation layer: numerics allowed here)
        let ev: Vec<C> = if m == 1 {
            vec![d[(idx[0], idx[0])]]
        } else {
            // a generic real combination of the real and imaginary parts (both Q cos/sin Theta Q^T) separates
            // eigenvalues whose cosines coincide; the Rayleigh quotient error is second order in any residual mixing
            let re = nalgebra::DMatrix::from_fn(m, m, |r, c| {
                let z = 0.5 * (d[(idx[r], idx[c])] + d[(idx[c], idx[r])]);
                0.8 * z.re + 0.6 * z.im
            });
            let se = re.symmetric_eigen();
            (0..m)
                .map(|j| {
                    let q = se.eigenvectors.column(j);
                    let mut v = C::default();
                    for r in 0..m {
                        for c in 0..m {
                            v += d[(idx[r], idx[c])] * (q[r] * q[c]);
                        }
                    }
                    v
                })
                .collect()
        };
        // best matching of the block's eigenvalues to the cluster's roots
        let mut perm: Vec<usize> = (0..m).collect();
        let mut best = f64::INFINITY;
        loop {
            let e = (0..m)
                .map(|j| (ev[perm[j]] - roots[idx[j]]).norm())
                .fold(0.0, f64::max);
            best = best.min(e);
            // next permutation
            let mut i = m;
            let mut found = false;
            if m >= 2 {
                let mut ii = m - 1;
                while ii > 0 {
                    if perm[ii - 1] < perm[ii] {
                        i = ii - 1;
                        found = true;
                        break;
                    }
                    ii -= 1;
                }
            }
            if !found {
                break;
            }
            let mut j = m - 1;
            while perm[j] <= perm[i] {
                j -= 1;
            }
            perm.swap(i, j);
            perm[i + 1..].reverse();
        }
        worst = worst.max(best);
    }
    worst
}
/// Real orthonormal basis adapted to the given root clusters of the symmetric unitary `s`: the
/// projector of each cluster is the product over the OTHER clusters only, so no small divisor appears.
fn real_eigvecs_grouped(s: &Mat4, d: &[C; 4], rep: &[usize], grp: &[usize; 4]) -> Option<Real4> {
    let ng = rep.len();
    let mut u = Real4::zeros();
    let mut taken = [[0.0f64; 4]; 4];
    let mut nt = 0;
    for g in 0..ng {
        let mut p = Mat4::identity();
        // annihilate EVERY root outside cluster g (not just each other cluster's representative): a root
        // that sits at the cluster's spread from its representative would otherwise leak into this block
        for k in 0..4 {
            if grp[k] != g {
                let mut sh = *s;
                for i in 0..4 {
                    sh[(i, i)] -= d[k];
                }
                p = sh * p;
                let inv = 1.0 / (d[rep[g]] - d[k]);
                p *= inv;
            }
        }
        // within a near-cluster the projector's eigenvalues are 1 + O(spread), complex: keep the real part,
        // whose columns still span the cluster's (real) eigenspace
        let mut cols = [[0.0f64; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                if p[(r, c)].im.abs() > 0.5 {
                    return None;
                }
                cols[c][r] = p[(r, c)].re;
            }
        }
        for k in 0..4 {
            if grp[k] != g {
                continue;
            }
            for v in &taken[..nt] {
                for c in cols.iter_mut() {
                    let dd: f64 = (0..4).map(|r| v[r] * c[r]).sum();
                    for r in 0..4 {
                        c[r] -= dd * v[r];
                    }
                }
            }
            let (bi, bn) = cols
                .iter()
                .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
                .enumerate()
                .fold((0, 0.0), |a, (i, n)| if n > a.1 { (i, n) } else { a });
            if bn < 1e-7 {
                return None;
            }
            let v: [f64; 4] = std::array::from_fn(|r| cols[bi][r] / bn);
            for r in 0..4 {
                u[(r, k)] = v[r];
            }
            taken[nt] = v;
            nt += 1;
        }
    }
    Some(u)
}
/// `3+1` leaf: `O^T W O = m D2 + (m'-m) r r^T D2`, secular equations linear in `r_k^2`.
fn leaf_31(d1: &[C; 4], d2: &[C; 4], lam_t: &[C; 4]) -> Vec<Real4> {
    let mut out = vec![];
    let mut e_idx = None;
    for e in 0..4 {
        let others: Vec<usize> = (0..4).filter(|&k| k != e).collect();
        if (d1[others[0]] - d1[others[1]]).norm() < 1e-6
            && (d1[others[0]] - d1[others[2]]).norm() < 1e-6
            && (d1[e] - d1[others[0]]).norm() > 1e-6
        {
            e_idx = Some(e);
        }
    }
    let Some(e) = e_idx else { return out };
    let others: Vec<usize> = (0..4).filter(|&k| k != e).collect();
    let m = {
        let s = (d1[others[0]] + d1[others[1]] + d1[others[2]]) / 3.0;
        s / s.norm()
    };
    let mp = d1[e];
    let mut a: Vec<[f64; 5]> = vec![];
    for i in 0..4 {
        let mut r = [C::default(); 4];
        let mut ok = true;
        for k in 0..4 {
            let den = lam_t[i] - m * d2[k];
            if den.norm() < 1e-12 {
                ok = false;
                break;
            }
            r[k] = d2[k] * (mp - m) / den;
        }
        if ok {
            a.push([r[0].re, r[1].re, r[2].re, r[3].re, 1.0]);
            a.push([r[0].im, r[1].im, r[2].im, r[3].im, 0.0]);
        }
    }
    a.push([1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut ata = [[0.0; 4]; 4];
    let mut atb = [0.0; 4];
    for r in &a {
        for i in 0..4 {
            for j in 0..4 {
                ata[i][j] += r[i] * r[j];
            }
            atb[i] += r[i] * r[4];
        }
    }
    let mm = nalgebra::Matrix4::<f64>::from_fn(|i, j| ata[i][j]);
    let Some(inv) = mm.try_inverse() else {
        return out;
    };
    let y = inv * nalgebra::Vector4::new(atb[0], atb[1], atb[2], atb[3]);
    if y.iter().any(|&v| v < -1e-7) {
        return out;
    }
    let rv: [f64; 4] = std::array::from_fn(|k| y[k].max(0.0).sqrt());
    for sg in 0..8u32 {
        let rr: [f64; 4] = std::array::from_fn(|k| {
            if k > 0 && (sg >> (k - 1)) & 1 == 1 {
                -rv[k]
            } else {
                rv[k]
            }
        });
        let mut rowsv: Vec<[f64; 4]> = vec![rr];
        for i in 0..4 {
            if rowsv.len() == 4 {
                break;
            }
            let mut v = [0.0; 4];
            v[i] = 1.0;
            for q in &rowsv {
                let d: f64 = (0..4).map(|k| q[k] * v[k]).sum();
                for k in 0..4 {
                    v[k] -= d * q[k];
                }
            }
            let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n > 1e-6 {
                for k in 0..4 {
                    v[k] /= n;
                }
                rowsv.push(v);
            }
        }
        if rowsv.len() < 4 {
            continue;
        }
        let mut o = Real4::zeros();
        o.set_row(e, &RowVector4::from_row_slice(&rowsv[0]));
        for (t, &oi) in others.iter().enumerate() {
            o.set_row(oi, &RowVector4::from_row_slice(&rowsv[t + 1]));
        }
        if o.determinant() < 0.0 {
            o.row_mut(others[0]).neg_mut();
        }
        out.push(o);
    }
    out
}
fn canon_perm(d: &[C; 4], tol: f64) -> Option<(Real4, [C; 4])> {
    for p in 0..4 {
        for q in (p + 1)..4 {
            if (d[p] - d[q]).norm() < tol {
                let rest: Vec<usize> = (0..4).filter(|&k| k != p && k != q).collect();
                let order = [p, q, rest[0], rest[1]];
                let mut pm = Real4::zeros();
                for c in 0..4 {
                    pm[(order[c], c)] = 1.0;
                }
                return Some((pm, [d[p], d[q], d[rest[0]], d[rest[1]]]));
            }
        }
    }
    None
}
fn sqrt_psd2(s: &[[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let (a, b, c) = (s[0][0], s[0][1], s[1][1]);
    let det = a * c - b * b;
    let tr = a + c;
    if det < -1e-9 || tr < -1e-9 {
        return None;
    }
    let sd = det.max(0.0).sqrt();
    let den = (tr + 2.0 * sd).max(0.0).sqrt();
    if den < 1e-14 {
        return Some([[0.0, 0.0], [0.0, 0.0]]);
    }
    Some([[(a + sd) / den, b / den], [b / den, (c + sd) / den]])
}
/// Canonical frame from the 2x2 overlap block `G` (simple eigen-coordinates of both gates at 2,3).
fn frame_from_g(g: &[[f64; 2]; 2]) -> Option<Real4> {
    let gtg = [
        [
            g[0][0] * g[0][0] + g[1][0] * g[1][0],
            g[0][0] * g[0][1] + g[1][0] * g[1][1],
        ],
        [
            g[0][0] * g[0][1] + g[1][0] * g[1][1],
            g[0][1] * g[0][1] + g[1][1] * g[1][1],
        ],
    ];
    let m = [[1.0 - gtg[0][0], -gtg[0][1]], [-gtg[0][1], 1.0 - gtg[1][1]]];
    let h = sqrt_psd2(&m)?;
    let mut o = Real4::zeros();
    for r in 0..2 {
        for c in 0..2 {
            o[(r, 2 + c)] = h[r][c];
            o[(2 + r, 2 + c)] = g[r][c];
        }
    }
    let cols: Vec<[f64; 4]> = vec![
        std::array::from_fn(|r| o[(r, 2)]),
        std::array::from_fn(|r| o[(r, 3)]),
    ];
    let mut comp: Vec<[f64; 4]> = vec![];
    for i in 0..4 {
        if comp.len() == 2 {
            break;
        }
        let mut v = [0.0; 4];
        v[i] = 1.0;
        for q in cols.iter().chain(comp.iter()) {
            let d: f64 = (0..4).map(|k| q[k] * v[k]).sum();
            for k in 0..4 {
                v[k] -= d * q[k];
            }
        }
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-7 {
            for k in 0..4 {
                v[k] /= n;
            }
            comp.push(v);
        }
    }
    if comp.len() < 2 {
        return None;
    }
    for r in 0..4 {
        o[(r, 0)] = comp[0][r];
        o[(r, 1)] = comp[1][r];
    }
    if o.determinant() < 0.0 {
        o.column_mut(0).neg_mut();
    }
    Some(o)
}
/// Canonical `2+1+1 x (2+1+1 or 2+2)` leaf: `e1` linear in `G o G`, `e2 = A + B.s + K det(G)^2`.
/// 2+1+1 x 2+1+1 overlap leaf.  With sigma = s12 fixed, e1 is linear in (s11, s21, s22) and e2 is quadratic
/// on that line; the sign lift squares to a quartic in the line parameter.  sigma is selected by the
/// discriminant events of that quartic (an isolated fold fibre is a double root, hit by the extremum event).
fn overlap_leaf(
    d1: &[C; 4],
    d2: &[C; 4],
    e1t: C,
    e2t: C,
    test: &mut dyn FnMut(Real4) -> bool,
) -> bool {
    let (m, mu2, mu3) = (d1[0], d1[2], d1[3]);
    let (mp, nu2, nu3) = (d2[0], d2[2], d2[3]);
    let c0 = m * (nu2 + nu3) + mp * (mu2 + mu3);
    let cjk = [
        [(m - mu2) * (mp - nu2), (m - mu2) * (mp - nu3)],
        [(m - mu3) * (mp - nu2), (m - mu3) * (mp - nu3)],
    ];
    let ev = |g: &[[f64; 2]; 2]| -> Option<f64> {
        let o = frame_from_g(g)?;
        Some(char_e1e2(d1, d2, &o).1.re)
    };
    let z = [[0.0; 2]; 2];
    let Some(a0) = ev(&z) else { return false };
    let mut bjk = [[0.0; 2]; 2];
    for j in 0..2 {
        for k in 0..2 {
            let mut g = z;
            g[j][k] = 0.5;
            let Some(e) = ev(&g) else { return false };
            bjk[j][k] = (e - a0) * 4.0;
        }
    }
    let kk = {
        let Some(e) = ev(&[[0.5, 0.0], [0.0, 0.5]]) else {
            return false;
        };
        ((e - a0) - (bjk[0][0] + bjk[1][1]) / 4.0) * 16.0
    };
    if (nu2 - nu3).norm() < 1e-9 {
        let (ca, cb) = (cjk[0][0], cjk[1][0]);
        let rhs = e1t - c0;
        let det = ca.re * cb.im - ca.im * cb.re;
        if det.abs() < 1e-14 || kk.abs() < 1e-14 {
            return false;
        }
        let s11 = (rhs.re * cb.im - rhs.im * cb.re) / det;
        let s22 = (ca.re * rhs.im - ca.im * rhs.re) / det;
        if !(-1e-9..=1.0 + 1e-9).contains(&s11) || !(-1e-9..=1.0 + 1e-9).contains(&s22) {
            return false;
        }
        let t12 = (a0 + bjk[0][0] * s11 + bjk[1][0] * s22 + kk * s11 * s22 - e2t.re) / kk;
        if t12 < -1e-9 || t12 > s11 * s22 + 1e-9 {
            return false;
        }
        let s12 = t12.max(0.0).sqrt();
        for sg in [1.0, -1.0] {
            let sm = [
                [s11.clamp(0.0, 1.0), sg * s12],
                [sg * s12, s22.clamp(0.0, 1.0)],
            ];
            if let Some(o) = sqrt_psd2(&sm).and_then(|g| frame_from_g(&g)) {
                if test(o) {
                    return true;
                }
            }
        }
        return false;
    }
    let rows = [
        [cjk[0][0].re, cjk[1][0].re, cjk[1][1].re],
        [cjk[0][0].im, cjk[1][0].im, cjk[1][1].im],
    ];
    let d = [
        rows[0][1] * rows[1][2] - rows[0][2] * rows[1][1],
        rows[0][2] * rows[1][0] - rows[0][0] * rows[1][2],
        rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0],
    ];
    if d.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-13 {
        return false;
    }
    let p = (0..3)
        .max_by(|&a, &b| d[a].abs().partial_cmp(&d[b].abs()).unwrap())
        .unwrap();
    let cols: Vec<usize> = (0..3).filter(|&c| c != p).collect();
    let mdet = rows[0][cols[0]] * rows[1][cols[1]] - rows[0][cols[1]] * rows[1][cols[0]];
    if mdet.abs() < 1e-14 {
        return false;
    }
    let mul = |a: &[f64], b: &[f64]| {
        let mut c = vec![0.0; a.len() + b.len() - 1];
        for i in 0..a.len() {
            for j in 0..b.len() {
                c[i + j] += a[i] * b[j];
            }
        }
        c
    };
    // line x0 + t d through the e1 plane at sigma, the quadratic f(t) = e2 target - e2 on the line, the quartic
    let quartic = |sig: f64| -> ([f64; 5], [f64; 3], [f64; 3]) {
        let rhs = e1t - c0 - cjk[0][1] * sig;
        let r = [rhs.re, rhs.im];
        let mut x0 = [0.0; 3];
        x0[cols[0]] = (r[0] * rows[1][cols[1]] - r[1] * rows[0][cols[1]]) / mdet;
        x0[cols[1]] = (rows[0][cols[0]] * r[1] - rows[1][cols[0]] * r[0]) / mdet;
        let (s11, s21, s22) = ([x0[0], d[0]], [x0[1], d[1]], [x0[2], d[2]]);
        let mut f = [e2t.re - a0 - bjk[0][1] * sig, 0.0, 0.0];
        for (coef, poly) in [(bjk[0][0], s11), (bjk[1][0], s21), (bjk[1][1], s22)] {
            f[0] -= coef * poly[0];
            f[1] -= coef * poly[1];
        }
        let s11s22 = mul(&s11, &s22);
        f[0] -= kk * (s11s22[0] + sig * s21[0]);
        f[1] -= kk * (s11s22[1] + sig * s21[1]);
        f[2] -= kk * s11s22[2];
        let lhs = mul(&f, &f);
        let rhs3 = mul(&mul(&s11, &s21), &s22);
        let mut q = [0.0; 5];
        q.copy_from_slice(&lhs[..5]);
        for i in 0..4 {
            q[i] -= 4.0 * kk * kk * sig * rhs3[i];
        }
        (q, x0, f)
    };
    exact_select(&|sig| quartic_disc(&quartic(sig).0), 0.0, 1.0, 32, |sig| {
        let (q, x0, f) = quartic(sig);
        let (rts, nr) = real_roots(&q);
        for &t in &rts[..nr] {
            let sv = [x0[0] + t * d[0], x0[1] + t * d[1], x0[2] + t * d[2]];
            if sv.iter().any(|&v| !(-1e-7..=1.0 + 1e-7).contains(&v)) {
                continue;
            }
            let sv: [f64; 3] = std::array::from_fn(|i| sv[i].clamp(0.0, 1.0));
            let fval = f[0] + f[1] * t + f[2] * t * t;
            let sign = if kk.abs() > 1e-14 && -fval / (2.0 * kk) < 0.0 {
                -1.0
            } else {
                1.0
            };
            let g = [
                [sv[0].sqrt(), sig.sqrt()],
                [sv[1].sqrt(), sign * sv[2].sqrt()],
            ];
            if let Some(o) = frame_from_g(&g) {
                if test(o) {
                    return true;
                }
            }
        }
        false
    })
}

// Near-3+1 gates: the stabilizer closure.
/// `d2` has three eigenvalues within 1e-4 (not exactly equal).  With `d2 = d2_0 + delta E`, `d2_0` the
/// snapped gate and `O_0` a snapped solution, `S` in the triple's stabilizer commutes with `d2_0`, so
/// `M(O_0 S) = M_0 + delta D1 O_0 (S E S^T) O_0^T` exactly.  The deviations are tangential,
/// `E = i mean diag(theta) + O(delta)`, so with `Y = S diag(theta) S^T` real symmetric the target is affine in
/// `Y` up to `delta^2`: three linear conditions and the trace cut a plane in Sym_3, `tr Y^2` is an ellipse
/// on it and `det Y` a cubic; each intersection gives `S` (eigenvectors of `Y`, closed form) and a frame.
fn near_31_leaf(
    d1: &[C; 4],
    d2: &[C; 4],
    lt: &[C; 4],
    test: &mut dyn FnMut(Real4) -> bool,
) -> bool {
    let mut tr = [0usize; 3];
    let mut found = false;
    for sgl in 0..4 {
        let t: Vec<usize> = (0..4).filter(|&k| k != sgl).collect();
        if (0..3).all(|a| (0..3).all(|b| (d2[t[a]] - d2[t[b]]).norm() < 1e-4))
            && (d2[sgl] - d2[t[0]]).norm() > 1e-3
        {
            tr = [t[0], t[1], t[2]];
            found = true;
        }
    }
    if !found {
        return false;
    }
    let mean = {
        let z = d2[tr[0]] + d2[tr[1]] + d2[tr[2]];
        z / z.norm()
    };
    let theta: [f64; 3] = std::array::from_fn(|j| (mean.conj() * d2[tr[j]]).im); // tangential deviations
    let delta = theta.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if delta < 1e-13 {
        return false;
    }
    let th: [f64; 3] = std::array::from_fn(|j| theta[j] / delta);
    let mut d20 = *d2;
    for &k in &tr {
        d20[k] = mean;
    }
    let scal = C::new(0.0, 1.0) * mean; // E = delta * scal * diag(th); rows and rhs both divided by delta
    let tau = e1e2_of_spectrum(lt);
    let d1m = Mat4::from_diagonal(&nalgebra::Vector4::new(d1[0], d1[1], d1[2], d1[3]));
    let d20m = Mat4::from_diagonal(&nalgebra::Vector4::new(d20[0], d20[1], d20[2], d20[3]));
    let pairs = [(0usize, 0usize), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    for o0 in leaf_31(&d20, d1, lt).into_iter().map(|o| o.transpose()) {
        let om = complex(&o0);
        let m0 = d1m * om * d20m * om.transpose();
        let e10 = m0.trace();
        let e20 = ((e10 * e10) - (m0 * m0).trace()) * 0.5;
        // rows: Re(scal L1), Im(scal L1), Re(scal L2), trace  |  rhs
        let mut rows = [[0.0f64; 7]; 4];
        for (k, &(a, b)) in pairs.iter().enumerate() {
            let mut yt = Mat4::zeros();
            yt[(tr[a], tr[b])] = C::new(1.0, 0.0);
            yt[(tr[b], tr[a])] = C::new(1.0, 0.0);
            let n = d1m * om * yt * om.transpose();
            let l1 = scal * n.trace();
            let l2 = scal * (e10 * n.trace() - (m0 * n).trace());
            rows[0][k] = l1.re;
            rows[1][k] = l1.im;
            rows[2][k] = l2.re;
            rows[3][k] = if a == b { 1.0 } else { 0.0 };
        }
        rows[0][6] = (tau[0] - e10.re) / delta;
        rows[1][6] = (tau[1] - e10.im) / delta;
        rows[2][6] = (tau[2] - e20.re) / delta;
        rows[3][6] = th[0] + th[1] + th[2];
        // particular solution and 2-dim nullspace by elimination with full pivoting
        let mut m = rows;
        let mut piv: Vec<(usize, usize)> = vec![];
        let mut ur = [false; 4];
        let mut uc = [false; 6];
        for _ in 0..4 {
            let mut best = (0.0f64, 0, 0);
            for r in 0..4 {
                if ur[r] {
                    continue;
                }
                for c in 0..6 {
                    if !uc[c] && m[r][c].abs() > best.0 {
                        best = (m[r][c].abs(), r, c);
                    }
                }
            }
            if best.0 < 1e-12 {
                break;
            }
            let (_, pr, pc) = best;
            ur[pr] = true;
            uc[pc] = true;
            piv.push((pr, pc));
            for r in 0..4 {
                if r != pr {
                    let f = m[r][pc] / m[pr][pc];
                    for c in 0..7 {
                        m[r][c] -= f * m[pr][c];
                    }
                }
            }
        }
        if (0..4).any(|r| !ur[r] && m[r][6].abs() > 1e-9) {
            continue;
        }
        let mut y0 = [0.0f64; 6];
        for &(pr, pc) in &piv {
            y0[pc] = m[pr][6] / m[pr][pc];
        }
        let free: Vec<usize> = (0..6).filter(|&c| !uc[c]).collect();
        if free.len() != 2 {
            continue;
        }
        let basis: Vec<[f64; 6]> = free
            .iter()
            .map(|&f| {
                let mut n = [0.0; 6];
                n[f] = 1.0;
                for &(pr, pc) in &piv {
                    n[pc] = -m[pr][f] / m[pr][pc];
                }
                n
            })
            .collect();
        // ellipse tr Y^2 = sum th^2 on the plane y0 + u p + v q  (off-diagonal entries count twice)
        let w = |k: usize| if k < 3 { 1.0 } else { 2.0 };
        let ip = |x: &[f64; 6], z: &[f64; 6]| (0..6).map(|k| w(k) * x[k] * z[k]).sum::<f64>();
        let (p, q) = (basis[0], basis[1]);
        let (gpp, gpq, gqq, gp0, gq0, g00) = (
            ip(&p, &p),
            ip(&p, &q),
            ip(&q, &q),
            ip(&p, &y0),
            ip(&q, &y0),
            ip(&y0, &y0),
        );
        let target2 = th.iter().map(|x| x * x).sum::<f64>();
        // centre: minimize the quadratic -> G c = -g0
        let det = gpp * gqq - gpq * gpq;
        if det.abs() < 1e-18 {
            continue;
        }
        let cu = -(gqq * gp0 - gpq * gq0) / det;
        let cv = -(gpp * gq0 - gpq * gp0) / det;
        let rad2 = target2
            - (g00
                + 2.0 * (gp0 * cu + gq0 * cv)
                + gpp * cu * cu
                + 2.0 * gpq * cu * cv
                + gqq * cv * cv);
        if rad2 <= 0.0 {
            continue;
        }
        // axes of the quadratic form [[gpp, gpq],[gpq, gqq]]
        let (ha, hb) = (0.5 * (gpp + gqq), 0.5 * (gpp - gqq));
        let rr = (hb * hb + gpq * gpq).sqrt();
        let (l1, l2) = (ha + rr, ha - rr);
        if l2 <= 0.0 {
            continue;
        }
        let ang = 0.5 * gpq.atan2(hb);
        let (e1u, e1v) = (ang.cos(), ang.sin());
        let (e2u, e2v) = (-ang.sin(), ang.cos());
        let (ra, rb) = ((rad2 / l1).sqrt(), (rad2 / l2).sqrt());
        let ymat = |phi: f64| -> ([f64; 6], M3) {
            let (u, v) = (
                cu + ra * phi.cos() * e1u + rb * phi.sin() * e2u,
                cv + ra * phi.cos() * e1v + rb * phi.sin() * e2v,
            );
            let y: [f64; 6] = std::array::from_fn(|k| y0[k] + u * p[k] + v * q[k]);
            (
                y,
                M3::new(y[0], y[3], y[4], y[3], y[1], y[5], y[4], y[5], y[2]),
            )
        };
        let g = |phi: f64| ymat(phi).1.determinant() - th[0] * th[1] * th[2];
        // sign changes of the cubic along the ellipse, bisected
        let cells = 256;
        let mut prev = g(0.0);
        for c in 1..=cells {
            let x1 = std::f64::consts::TAU * c as f64 / cells as f64;
            let cur = g(x1);
            if prev.signum() != cur.signum() && prev != 0.0 && cur != 0.0 {
                let (mut lo, mut hi, mut flo) = (
                    std::f64::consts::TAU * (c - 1) as f64 / cells as f64,
                    x1,
                    prev,
                );
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let fm = g(mid);
                    if fm == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if fm.signum() == flo.signum() {
                        lo = mid;
                        flo = fm;
                    } else {
                        hi = mid;
                    }
                }
                let (_, y) = ymat(0.5 * (lo + hi));
                // eigenvectors of Y at the known eigenvalues th: cross products of two rows of Y - th I
                let mut sm = M3::zeros();
                let mut ok = true;
                for k in 0..3 {
                    let z = y - M3::identity() * th[k];
                    let r0 = V3::new(z[(0, 0)], z[(0, 1)], z[(0, 2)]);
                    let r1 = V3::new(z[(1, 0)], z[(1, 1)], z[(1, 2)]);
                    let r2 = V3::new(z[(2, 0)], z[(2, 1)], z[(2, 2)]);
                    let cands = [r0.cross(&r1), r0.cross(&r2), r1.cross(&r2)];
                    let vbest = cands
                        .iter()
                        .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap())
                        .unwrap();
                    if vbest.norm() < 1e-12 {
                        ok = false;
                        break;
                    }
                    let vn = vbest / vbest.norm();
                    for r in 0..3 {
                        sm[(r, k)] = vn[r];
                    }
                }
                if !ok {
                    continue;
                }
                if sm.determinant() < 0.0 {
                    for r in 0..3 {
                        sm[(r, 0)] = -sm[(r, 0)];
                    }
                }
                let mut st = Real4::identity();
                for a in 0..3 {
                    for b in 0..3 {
                        st[(tr[a], tr[b])] = sm[(a, b)];
                    }
                }
                if test(o0 * st) {
                    return true;
                }
            }
            prev = cur;
        }
    }
    false
}

// One paired gate (2+2): the squared-Plucker leaf.
/// `d1 = diag` with a pair {p,q} of equal eigenvalues `a` and the complementary pair equal to `c`
/// (A = cI + sP, s = a - c).  With `R = O^T P O` the projection onto the plane of rows p,q of O, the
/// characteristic polynomial of `(cI + sR) B` is affine in the six squared Plucker coordinates
/// `w_ij = x_ij^2` of that plane: e1 = c e1(b) + s sum_{i<j} (b_i + b_j) w_ij and
/// e2 = c^2 e2(b) + sum_{i<j} w_ij [ s c (b_i (e1(b) - b_i) + b_j (e1(b) - b_j)) + s^2 b_i b_j ].
/// Real image: w >= 0, sum w = 1, Heron H(w12 w34, w13 w24, w14 w23) = 0.  Three target rows and the
/// simplex cut a plane; on each polygon edge w_ij = 0 the quartic is a quadratic (one complementary
/// product vanishes, the other two must be equal), and lines through the interior give quartics.
/// Signs from the Plucker relation, the plane from its largest-minor chart, the rest by completion.
fn pair22_leaf(d1: &[C; 4], d2: &[C; 4], lt: &[C; 4], test: &mut dyn FnMut(Real4) -> bool) -> bool {
    // pairs within 1e-4 are snapped to their means: the frame is then exact for the snapped gate and
    // within the snap distance for the true one, which the one-step correction closes
    let mut pr = None;
    for p in 0..4 {
        for q in (p + 1)..4 {
            let o: Vec<usize> = (0..4).filter(|&k| k != p && k != q).collect();
            if (d1[p] - d1[q]).norm() < 1e-4
                && (d1[o[0]] - d1[o[1]]).norm() < 1e-4
                && (d1[p] - d1[o[0]]).norm() > 1e-3
            {
                pr = Some((p, q));
            }
        }
    }
    let Some((p, q)) = pr else { return false };
    let oth: Vec<usize> = (0..4).filter(|&k| k != p && k != q).collect();
    let unit = |z: C| z / z.norm();
    let (a, c) = (unit(d1[p] + d1[q]), unit(d1[oth[0]] + d1[oth[1]]));
    let sg = a - c;
    let b = *d2;
    let e1b = b[0] + b[1] + b[2] + b[3];
    let mut e2b = C::default();
    for i in 0..4 {
        for j in (i + 1)..4 {
            e2b += b[i] * b[j];
        }
    }
    let pairs = [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let tau = e1e2_of_spectrum(lt);
    let mut rows = [[0.0f64; 7]; 4];
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let c1 = sg * (b[i] + b[j]);
        let c2 = sg * c * (b[i] * (e1b - b[i]) + b[j] * (e1b - b[j])) + sg * sg * b[i] * b[j];
        rows[0][k] = c1.re;
        rows[1][k] = c1.im;
        rows[2][k] = c2.re;
        rows[3][k] = 1.0;
    }
    let k1 = c * e1b;
    let k2 = c * c * e2b;
    rows[0][6] = tau[0] - k1.re;
    rows[1][6] = tau[1] - k1.im;
    rows[2][6] = tau[2] - k2.re;
    rows[3][6] = 1.0;
    // plane y0 + u p + v q
    let mut m = rows;
    let mut piv: Vec<(usize, usize)> = vec![];
    let mut ur = [false; 4];
    let mut uc = [false; 6];
    for _ in 0..4 {
        let mut best = (0.0f64, 0, 0);
        for r in 0..4 {
            if ur[r] {
                continue;
            }
            for cc in 0..6 {
                if !uc[cc] && m[r][cc].abs() > best.0 {
                    best = (m[r][cc].abs(), r, cc);
                }
            }
        }
        if best.0 < 1e-12 {
            break;
        }
        let (_, prow, pc) = best;
        ur[prow] = true;
        uc[pc] = true;
        piv.push((prow, pc));
        for r in 0..4 {
            if r != prow {
                let f = m[r][pc] / m[prow][pc];
                for cc in 0..7 {
                    m[r][cc] -= f * m[prow][cc];
                }
            }
        }
    }
    if (0..4).any(|r| !ur[r] && m[r][6].abs() > 1e-9) {
        return false;
    }
    let mut y0 = [0.0f64; 6];
    for &(prow, pc) in &piv {
        y0[pc] = m[prow][6] / m[prow][pc];
    }
    let free: Vec<usize> = (0..6).filter(|&cc| !uc[cc]).collect();
    if free.len() != 2 {
        return false;
    }
    let basis: Vec<[f64; 6]> = free
        .iter()
        .map(|&f| {
            let mut n = [0.0; 6];
            n[f] = 1.0;
            for &(prow, pc) in &piv {
                n[pc] = -m[prow][f] / m[prow][pc];
            }
            n
        })
        .collect();
    let (bp, bq) = (basis[0], basis[1]);
    let w_at =
        |u: f64, v: f64| -> [f64; 6] { std::array::from_fn(|k| y0[k] + u * bp[k] + v * bq[k]) };
    // complementary products: A0 = w12 w34 (k 0,5), B0 = w13 w24 (k 1,4), C0 = w14 w23 (k 2,3)
    let prods = |w: &[f64; 6]| [w[0] * w[5], w[1] * w[4], w[2] * w[3]];
    let heron = |w: &[f64; 6]| {
        let [a0, b0, c0] = prods(w);
        2.0 * (a0 * b0 + a0 * c0 + b0 * c0) - a0 * a0 - b0 * b0 - c0 * c0
    };
    let mut cands: Vec<[f64; 6]> = vec![];
    // (a) polygon edges: w_k = 0 along the line, quadratic condition on the remaining two products
    for k in 0..6 {
        let (ak, bk, ck) = (y0[k], bp[k], bq[k]);
        // line in (u,v): ak + bk u + ck v = 0; parametrize by t along its direction
        let n = (bk * bk + ck * ck).sqrt();
        if n < 1e-14 {
            continue;
        }
        let (u0, v0) = (-ak * bk / (n * n), -ak * ck / (n * n));
        let (du, dv) = (-ck / n, bk / n);
        let wt = |t: f64| w_at(u0 + t * du, v0 + t * dv);
        // the two products that must be equal: p1(t) - p2(t) = 0, quadratic in t: sample 3 points
        let others = match k {
            0 | 5 => (1usize, 2usize),
            1 | 4 => (0, 2),
            _ => (0, 1),
        };
        let f = |t: f64| {
            let pr = prods(&wt(t));
            pr[others.0] - pr[others.1]
        };
        let (f0, f1, f2) = (f(0.0), f(1.0), f(-1.0));
        let qa = 0.5 * (f1 + f2) - f0;
        let qb = 0.5 * (f1 - f2);
        let qc = f0;
        let mut ts = [0.0f64; 6];
        let mut nt = 0;
        if qa.abs() > 1e-14 {
            quad_roots(qb / qa, qc / qa, &mut ts, &mut nt);
        } else if qb.abs() > 1e-14 {
            ts[0] = -qc / qb;
            nt = 1;
        }
        for &t in &ts[..nt] {
            let mut w = wt(t);
            w[k] = 0.0;
            if w.iter().all(|&x| x > -1e-9) {
                cands.push(w.map(|x| x.max(0.0)));
            }
        }
    }
    // (b) lines through the plane's base point: Heron quartic along each
    for dir in 0..4 {
        let ang = std::f64::consts::PI * dir as f64 / 4.0;
        let (du, dv) = (ang.cos(), ang.sin());
        let wt = |t: f64| w_at(t * du, t * dv);
        // sample the quartic at 5 points and interpolate its coefficients
        let ts5 = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let hs: [f64; 5] = ts5.map(|t| heron(&wt(t)));
        let mut vm = [[0.0f64; 6]; 5];
        for r in 0..5 {
            let t = ts5[r];
            vm[r] = [1.0, t, t * t, t * t * t, t * t * t * t, hs[r]];
        }
        for cc in 0..5 {
            let pv = (cc..5)
                .max_by(|&x, &y| vm[x][cc].abs().partial_cmp(&vm[y][cc].abs()).unwrap())
                .unwrap();
            vm.swap(cc, pv);
            if vm[cc][cc].abs() < 1e-300 {
                continue;
            }
            for r in 0..5 {
                if r != cc {
                    let f = vm[r][cc] / vm[cc][cc];
                    for j in 0..6 {
                        vm[r][j] -= f * vm[cc][j];
                    }
                }
            }
        }
        let h: [f64; 5] = std::array::from_fn(|cc| vm[cc][5] / vm[cc][cc]);
        let (rts, nr) = real_roots(&h);
        for &t in &rts[..nr] {
            let w = wt(t);
            if w.iter().all(|&x| x > -1e-9) {
                cands.push(w.map(|x| x.max(0.0)));
            }
        }
    }
    // signs, chart, frame
    for w in cands {
        let mag: [f64; 6] = w.map(|x| x.sqrt());
        for sgn in 0..32u32 {
            let x: [f64; 6] = std::array::from_fn(|k| {
                if k > 0 && (sgn >> (k - 1)) & 1 == 1 {
                    -mag[k]
                } else {
                    mag[k]
                }
            });
            if (x[0] * x[5] - x[1] * x[4] + x[2] * x[3]).abs() > 1e-9 {
                continue;
            }
            // chart at the largest minor (i,j): v1 = e_i + sum_k alpha_k e_k, v2 = e_j + sum_k beta_k e_k
            let (kmax, _) = x.iter().enumerate().fold((0, 0.0f64), |acc, (k, &v)| {
                if v.abs() > acc.1 {
                    (k, v.abs())
                } else {
                    acc
                }
            });
            let (i, j) = pairs[kmax];
            let xs = |a: usize, b2: usize| -> f64 {
                if a < b2 {
                    x[pairs.iter().position(|&pp| pp == (a, b2)).unwrap()]
                } else {
                    -x[pairs.iter().position(|&pp| pp == (b2, a)).unwrap()]
                }
            };
            let mut v1 = [0.0f64; 4];
            let mut v2 = [0.0f64; 4];
            v1[i] = 1.0;
            v2[j] = 1.0;
            for k in 0..4 {
                if k == i || k == j {
                    continue;
                }
                v2[k] = xs(i, k) / x[kmax];
                v1[k] = -xs(j, k) / x[kmax];
            }
            // Gram-Schmidt rows p,q, then completion for the other two rows
            let mut o = Real4::zeros();
            let mut rowsv: Vec<[f64; 4]> = vec![];
            for v in [v1, v2] {
                let mut vv = v;
                for r in &rowsv {
                    let d: f64 = (0..4).map(|t| r[t] * vv[t]).sum();
                    for t in 0..4 {
                        vv[t] -= d * r[t];
                    }
                }
                let n: f64 = vv.iter().map(|t| t * t).sum::<f64>().sqrt();
                if n < 1e-9 {
                    break;
                }
                rowsv.push(vv.map(|t| t / n));
            }
            if rowsv.len() != 2 {
                continue;
            }
            for e in 0..4 {
                if rowsv.len() == 4 {
                    break;
                }
                let mut vv = [0.0; 4];
                vv[e] = 1.0;
                for r in &rowsv {
                    let d: f64 = (0..4).map(|t| r[t] * vv[t]).sum();
                    for t in 0..4 {
                        vv[t] -= d * r[t];
                    }
                }
                let n: f64 = vv.iter().map(|t| t * t).sum::<f64>().sqrt();
                if n > 1e-7 {
                    rowsv.push(vv.map(|t| t / n));
                }
            }
            if rowsv.len() != 4 {
                continue;
            }
            for (r, v) in [p, q, oth[0], oth[1]].into_iter().zip(&rowsv) {
                o.set_row(r, &RowVector4::from_row_slice(v));
            }
            if o.determinant() < 0.0 {
                o.row_mut(oth[1]).neg_mut();
            }
            if test(o) {
                return true;
            }
        }
    }
    false
}

// ------------------------------------------------ near-scalar gate: Schur-Horn leaf (Vertex class)
/// Real orthogonal `O` with `diag(O diag(theta) O^T) = v` (v majorized by theta), by plane rotations:
/// the largest remaining target is placed by rotating the current max and min diagonal entries.
fn schur_horn(theta: [f64; 4], v: [f64; 4]) -> Option<Real4> {
    let mut d = theta;
    let mut o = Real4::identity();
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|&x, &y| v[y].partial_cmp(&v[x]).unwrap());
    let mut free = [true; 4];
    let mut place = [0usize; 4]; // place[j] = index holding v_j
    for &j in &order[..3] {
        let mut p = usize::MAX;
        let mut q = usize::MAX;
        for i in 0..4 {
            if !free[i] {
                continue;
            }
            if p == usize::MAX || d[i] > d[p] {
                p = i;
            }
            if q == usize::MAX || d[i] < d[q] {
                q = i;
            }
        }
        if p == q {
            place[j] = p;
            free[p] = false;
            continue;
        }
        let c2 = ((v[j] - d[q]) / (d[p] - d[q])).clamp(0.0, 1.0);
        let (c, sn) = (c2.sqrt(), (1.0 - c2).sqrt());
        // rotate rows p, q of O: new row p = c row_p + sn row_q, new row q = -sn row_p + c row_q  (O -> G O)
        let (rp, rq) = (o.row(p).into_owned(), o.row(q).into_owned());
        for t in 0..4 {
            o[(p, t)] = c * rp[t] + sn * rq[t];
            o[(q, t)] = -sn * rp[t] + c * rq[t];
        }
        d[q] = d[p] + d[q] - v[j];
        d[p] = v[j];
        place[j] = p;
        free[p] = false;
    }
    let last = order[3];
    let q = (0..4).find(|&i| free[i])?;
    place[last] = q;
    // rows so that entry j of the diagonal is v_j
    let mut out = Real4::zeros();
    for j in 0..4 {
        out.set_row(j, &o.row(place[j]));
    }
    if out.determinant() < 0.0 {
        out.row_mut(0).neg_mut();
    }
    Some(out)
}
/// `mu` within 1e-2 of a scalar omega I (not exactly): mu_b = omega e^{i phi_b}; to first order the class
/// depends on O only through v_a = sum_b B_ab phi_b (e1 = omega[e1(lam) + i sum_a lam_a v_a] and the e2 analogue),
/// four linear equations fix v, and B is a Schur-Horn problem; the one-step correction removes the O(phi^2) terms.
fn near_scalar_leaf(
    lam: &[C; 4],
    mu: &[C; 4],
    lt: &[C; 4],
    test: &mut dyn FnMut(Real4) -> bool,
) -> bool {
    let spread = (0..4)
        .flat_map(|a| ((a + 1)..4).map(move |b| (a, b)))
        .map(|(a, b)| (mu[a] - mu[b]).norm())
        .fold(0.0, f64::max);
    if !(1e-11..=1e-2).contains(&spread) {
        return false;
    }
    let om = {
        let z = mu[0] + mu[1] + mu[2] + mu[3];
        z / z.norm()
    };
    let phi: [f64; 4] = std::array::from_fn(|b| (mu[b] / om).arg());
    let e1l = lam[0] + lam[1] + lam[2] + lam[3];
    let mut e2l = C::default();
    for a in 0..4 {
        for b in (a + 1)..4 {
            e2l += lam[a] * lam[b];
        }
    }
    let tau = e1e2_of_spectrum(lt);
    let e1t = C::new(tau[0], tau[1]);
    let c1 = e1t / om - e1l; // = i sum_a lam_a v_a
    let c2 = tau[2] - (om * om * e2l).re; // = Re[omega^2 i sum_a (e1l lam_a - lam_a^2) v_a]
    let mut m = [[0.0f64; 5]; 4];
    for a in 0..4 {
        let f1 = C::new(0.0, 1.0) * lam[a];
        let f2 = om * om * C::new(0.0, 1.0) * (e1l * lam[a] - lam[a] * lam[a]);
        m[0][a] = f1.re;
        m[1][a] = f1.im;
        m[2][a] = f2.re;
        m[3][a] = 1.0;
    }
    m[0][4] = c1.re;
    m[1][4] = c1.im;
    m[2][4] = c2;
    m[3][4] = phi.iter().sum();
    for c in 0..4 {
        let piv = (c..4)
            .max_by(|&x, &y| m[x][c].abs().partial_cmp(&m[y][c].abs()).unwrap())
            .unwrap();
        m.swap(c, piv);
        if m[c][c].abs() < 1e-14 {
            return false;
        }
        for r in 0..4 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for t in 0..5 {
                    m[r][t] -= f * m[c][t];
                }
            }
        }
    }
    let v: [f64; 4] = std::array::from_fn(|a| m[a][4] / m[a][a]);
    match schur_horn(phi, v) {
        Some(o) => test(o),
        None => false,
    }
}
/// Overlap leaf on gates with one 2+1+1 pattern each, in the canonical order.
fn degenerate_pair_leaf(
    lam: &[C; 4],
    mu: &[C; 4],
    e1t: C,
    e2t: C,
    test: &mut dyn FnMut(Real4) -> bool,
) -> bool {
    let Some((p1, d1c)) = canon_perm(lam, 1e-4) else {
        return false;
    };
    let Some((p2, d2c)) = canon_perm(mu, 1e-4) else {
        return false;
    };
    // pairs within 1e-4 are snapped to their mean; the one-step correction closes the snap distance
    let snap = |d: [C; 4]| -> [C; 4] {
        let m = d[0] + d[1];
        let m = m / m.norm();
        [m, m, d[2], d[3]]
    };
    let (d1c, d2c) = (snap(d1c), snap(d2c));
    if (d1c[2] - d1c[3]).norm() < 1e-9
        || (d1c[0] - d1c[2]).norm() < 1e-9
        || (d2c[0] - d2c[2]).norm() < 1e-9
    {
        return false;
    }
    overlap_leaf(&d1c, &d2c, e1t, e2t, &mut |oc| {
        test(p1 * oc * p2.transpose())
    })
}

/// The complete cascade under the production certificate: leaves first on
/// clustered spectra, charts (fast, wall rung, exact), leaves after the fast path otherwise, then a second
/// pass with the spectra snapped to their strata on clustered rows.  No linearized corrections anywhere.
pub(crate) fn solve_full(problem: &super::PreparedSandwich) -> Option<super::Solution> {
    let lam = problem.left;
    let mu = problem.right;
    let lifts = problem.target_roots;
    let gap = |l: &[C; 4]| {
        (0..4)
            .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
            .map(|(i, j)| (l[i] - l[j]).norm())
            .fold(f64::INFINITY, f64::min)
    };
    let clustered = gap(&lam).min(gap(&mu)).min(gap(&lifts[0])) < 1e-4;
    let allequal = |l: &[C; 4]| (1..4).all(|k| (l[k] - l[0]).norm() < 1e-9);
    if allequal(&lam) || allequal(&mu) {
        if let Some(h) = check_chart_candidate(problem, Real4::identity()) {
            return Some(h);
        }
    }
    let run_leaves = |lamc: [C; 4], muc: [C; 4], liftsc: [[C; 4]; 2]| -> Option<super::Solution> {
        for lift in 0..2 {
            let lam_t = liftsc[lift];
            let tau = e1e2_of_spectrum(&lam_t);
            let (e1t, e2t) = (C::new(tau[0], tau[1]), C::new(tau[2], 0.0));
            let mut hit: Option<super::Solution> = None;
            {
                let mut t = |o: Real4| -> bool {
                    match check_chart_candidate(problem, o) {
                        Some(h) => {
                            hit = Some(h);
                            true
                        }
                        None => false,
                    }
                };
                let _ = block_leaves(&lamc, &muc, e1t, e2t).into_iter().any(&mut t)
                    || near_scalar_leaf(&lamc, &muc, &lam_t, &mut |o| t(o))
                    || near_scalar_leaf(&muc, &lamc, &lam_t, &mut |o| t(o.transpose()))
                    || pair22_leaf(&lamc, &muc, &lam_t, &mut |o| t(o))
                    || pair22_leaf(&muc, &lamc, &lam_t, &mut |o| t(o.transpose()))
                    || near_31_leaf(&lamc, &muc, &lam_t, &mut |o| t(o))
                    || near_31_leaf(&muc, &lamc, &lam_t, &mut |o| t(o.transpose()))
                    || degenerate_pair_leaf(&lamc, &muc, e1t, e2t, &mut |o| t(o))
                    || degenerate_pair_leaf(&muc, &lamc, e1t, e2t, &mut |o| t(o.transpose()))
                    || overlap_leaf(&lamc, &muc, e1t, e2t, &mut |o| t(o));
            }
            if hit.is_some() {
                return hit;
            }
            let conj4 = |d: &[C; 4]| -> [C; 4] { std::array::from_fn(|q| d[q].conj()) };
            for role in 0..3 {
                let (d1, d2, lt) = match role {
                    0 => (lamc, muc, lam_t),
                    1 => (lam_t, conj4(&muc), lamc),
                    _ => (lam_t, conj4(&lamc), muc),
                };
                for o in leaf_31(&d1, &d2, &lt) {
                    let ob = reconstruct_role(role, &lamc, &muc, &lam_t, &o);
                    if let Some(ob) = ob {
                        if let Some(h) = check_chart_candidate(problem, ob) {
                            return Some(h);
                        }
                    }
                }
            }
        }
        None
    };
    if clustered {
        if let Some(h) = run_leaves(lam, mu, lifts) {
            return Some(h);
        }
    }
    if let Some(h) = solve_charts_gated(problem, 1, lam, mu, lifts, true) {
        return Some(h);
    }
    if !clustered {
        if let Some(h) = run_leaves(lam, mu, lifts) {
            return Some(h);
        }
    }
    if let Some(h) = solve_charts_gated(problem, 2, lam, mu, lifts, true) {
        return Some(h);
    }
    if !clustered {
        return None;
    }
    let snap = |l: [C; 4]| -> [C; 4] {
        let mut out = l;
        let mut done = [false; 4];
        for a in 0..4 {
            if done[a] {
                continue;
            }
            let mut z = C::default();
            let mut idx = vec![];
            for b in 0..4 {
                if !done[b] && (l[a] - l[b]).norm() < 1e-4 {
                    idx.push(b);
                    z += l[b];
                }
            }
            let z = z / z.norm();
            for &b in &idx {
                out[b] = z;
                done[b] = true;
            }
        }
        out
    };
    let (lamc, muc) = (snap(lam), snap(mu));
    let lts = snap(lifts[0]);
    let liftsc = [lts, std::array::from_fn(|k| -lts[k])];
    if let Some(h) = run_leaves(lamc, muc, liftsc) {
        return Some(h);
    }
    solve_charts_snapped(problem, lamc, muc, liftsc)
}

#[cfg(test)]
mod readiness_cubic_tests {
    #[test]
    fn small_nonzero_constant_does_not_add_spurious_real_roots() {
        // x^3 + 4x^2 + 13x + c is strictly increasing for every c:
        // its derivative is 3(x + 4/3)^2 + 23/3.
        for constant in [-1e-12, -1e-15, -1e-20, 1e-20, 1e-15, 1e-12] {
            let (roots, count) = super::real_roots(&[constant, 13.0, 4.0, 1.0, 0.0]);
            assert_eq!(count, 1, "spurious real roots for {constant}: {roots:?}");
            let root = roots[0];
            assert!(root.is_finite());
            let residual = ((root + 4.0) * root + 13.0) * root + constant;
            assert!(residual.abs() < 64.0 * f64::EPSILON * 14.0);
        }
    }

    #[test]
    fn zero_constant_cubic_keeps_the_exact_factor() {
        let cases: [([f64; 5], &[f64]); 5] = [
            ([0.0, 13.0, 4.0, 1.0, 0.0], &[0.0]),
            ([0.0, -2.0, 1.0, 1.0, 0.0], &[-2.0, 0.0, 1.0]),
            ([0.0, 0.0, -2.0, 1.0, 0.0], &[0.0, 0.0, 2.0]),
            ([0.0, 0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 0.0]),
            ([0.0, 4.0, 4.0, 1.0, 0.0], &[-2.0, -2.0, 0.0]),
        ];
        for (h, expected) in cases {
            let (mut roots, count) = super::real_roots(&h);
            assert_eq!(count, expected.len(), "unexpected roots for {h:?}");
            roots[..count].sort_by(f64::total_cmp);
            for (actual, expected) in roots[..count].iter().zip(expected) {
                assert!((actual - expected).abs() < 5e-14, "{h:?}: {roots:?}");
            }
            assert!(roots[..count].contains(&0.0));
        }
    }

    #[test]
    fn small_cubic_lead_keeps_both_finite_roots() {
        // (x - 0.6)(x + 1)(x + 900000): the old largest-root
        // subtraction displaced the finite roots despite a modest interval.
        let h = [-540000.0, 359999.4, 900000.4, 1.0, 0.0];
        let (roots, count) = super::real_roots(&h);
        assert_eq!(count, 3);
        for expected in [0.6, -1.0, -900000.0] {
            let error = roots[..count]
                .iter()
                .map(|r| (r - expected).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(
                error <= 5e-12 * (1.0 + expected.abs()),
                "missing {expected}: {roots:?}"
            );
        }
    }
}
