//! Complete algebraic realization of the routed `1 + 3` endpoint stratum.
//!
//! A fixed row/column pair splits off one target eigenvalue. The residual
//! `SO(3)` problem first tests its nine zero-entry walls, where `e1` is
//! bilinear in two squared Givens cosines. Scalar and `2 + 1` residual spectra
//! use identity and one-Givens rank-drop formulas. For the dense residual, the
//! complex trace cuts the four-dimensional Birkhoff chart to a plane and
//! orthostochasticity is one Heron quartic. After the nine faces are removed,
//! every compact component has a coordinate extremum; its discriminant has
//! degree at most twelve. A Heron kernel line lifts the selected bistochastic
//! point to `SO(3)` with no sign enumeration.

use super::{
    c, compound_residual, frame_metrics, orient_so4, poly_roots, signed_perm, Mat4, ACCEPT, C,
};

// The nominal 6 compatible base permutations x 6 ordered plane pairs contain
// four parameterizations of each zero-entry wall.  Square symmetries identify
// those four; WALL9 retains one representative for every matrix entry.
const BASE3: [[usize; 3]; 3] = [[0, 1, 2], [0, 2, 1], [1, 0, 2]];
const WALL9: [(usize, usize, usize); 9] = [
    (0, 0, 1),
    (0, 0, 2),
    (0, 1, 0),
    (0, 1, 2),
    (0, 2, 0),
    (0, 2, 1),
    (1, 0, 1),
    (1, 1, 0),
    (2, 1, 2),
];

/// Lower-degree part of a routed `1 + 3` split: rank drops and the nine
/// zero-entry faces. The compact plane-quartic completion is `solve_dense`
/// and deliberately runs after the other boundary accelerators.
pub(super) fn solve_walls(
    a: &[C; 4],
    b: &[C; 4],
    routed: &[[u8; 4]; 4],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
) -> Option<(Mat4, f64)> {
    for branch in 0..2 {
        for fixed_row in 0..4 {
            let rows: [usize; 3] = std::array::from_fn(|k| if k < fixed_row { k } else { k + 1 });
            let planes3 = [(rows[0], rows[1]), (rows[0], rows[2]), (rows[1], rows[2])];
            for fixed_col in 0..4 {
                if routed[fixed_row][fixed_col] & (1 << branch) == 0 {
                    continue;
                }
                let cols: [usize; 3] =
                    std::array::from_fn(|k| if k < fixed_col { k } else { k + 1 });
                let aa: [C; 3] = rows.map(|i| a[i]);
                let bb: [C; 3] = cols.map(|j| b[j]);
                let tau = targets[branch][0] - a[fixed_row] * b[fixed_col];
                if let Some(o) = rank_drop_frame(&aa, &bb, tau, fixed_row, fixed_col, rows, cols) {
                    let residual = compound_residual(dc, lam, &o, &targets[branch]);
                    if residual <= ACCEPT {
                        return Some((orient_so4(o), residual));
                    }
                }
                for &(base, p0, p1) in &WALL9 {
                    let mut perm = [0usize; 4];
                    perm[fixed_row] = fixed_col;
                    for k in 0..3 {
                        perm[rows[k]] = cols[BASE3[base][k]];
                    }
                    let planes = [planes3[p0], planes3[p1]];
                    let corners = e1_corners(a, b, planes, perm);
                    let co = [
                        corners[0],
                        corners[1] - corners[0],
                        corners[2] - corners[0],
                        corners[3] - corners[1] - corners[2] + corners[0],
                    ];
                    let residual = [co[0] - targets[branch][0], co[1], co[2], co[3]];
                    let rr: [f64; 4] = residual.map(|z| z.re);
                    let ri: [f64; 4] = residual.map(|z| z.im);

                    let q = [
                        rr[0] * ri[2] - ri[0] * rr[2],
                        rr[0] * ri[3] + rr[1] * ri[2] - ri[0] * rr[3] - ri[1] * rr[2],
                        rr[1] * ri[3] - ri[1] * rr[3],
                    ];
                    let mut candidates = Vec::with_capacity(6);
                    for x in quadratic_unit_roots(q) {
                        recover_y(rr, ri, x, &mut candidates);
                    }
                    // Dependent real equations have identically zero
                    // resultant.  A bilinear zero set in a square meets its
                    // boundary, so these four restrictions are complete.
                    let qscale = q.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    let rscale = rr.iter().chain(&ri).fold(0.0_f64, |m, &v| m.max(v.abs()));
                    if qscale <= 128.0 * f64::EPSILON * rscale * rscale {
                        recover_y(rr, ri, 0.0, &mut candidates);
                        recover_y(rr, ri, 1.0, &mut candidates);
                        for y in [0.0, 1.0] {
                            if let Some(x) = solve_real_lines(
                                rr[0] + rr[2] * y,
                                rr[1] + rr[3] * y,
                                ri[0] + ri[2] * y,
                                ri[1] + ri[3] * y,
                            ) {
                                candidates.push((x, y));
                            }
                        }
                    }
                    for (x, y) in candidates {
                        if !(-1e-8..=1.0 + 1e-8).contains(&x) || !(-1e-8..=1.0 + 1e-8).contains(&y)
                        {
                            continue;
                        }
                        let (x, y) = (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
                        let z =
                            residual[0] + residual[1] * x + residual[2] * y + residual[3] * x * y;
                        if z.norm() > 1e-8 {
                            continue;
                        }
                        let o = frame([x, y], planes, perm);
                        let r = compound_residual(dc, lam, &o, &targets[branch]);
                        if r <= ACCEPT {
                            return Some((orient_so4(o), r));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Complete dense part of the routed `1 + 3` theorem.  This runs after all
/// lower-degree boundary realizers so a degree-at-most-12 selector cannot
/// preempt a cheaper exact witness.
pub(super) fn solve_dense(
    a: &[C; 4],
    b: &[C; 4],
    routed: &[[u8; 4]; 4],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
) -> Option<(Mat4, f64)> {
    for branch in 0..2 {
        for fixed_row in 0..4 {
            let rows: [usize; 3] = std::array::from_fn(|k| if k < fixed_row { k } else { k + 1 });
            for fixed_col in 0..4 {
                if routed[fixed_row][fixed_col] & (1 << branch) == 0 {
                    continue;
                }
                let cols: [usize; 3] =
                    std::array::from_fn(|k| if k < fixed_col { k } else { k + 1 });
                let aa: [C; 3] = rows.map(|i| a[i]);
                let bb: [C; 3] = cols.map(|j| b[j]);
                let tau = targets[branch][0] - a[fixed_row] * b[fixed_col];
                // Birkhoff--von Neumann: every squared orthogonal block is
                // bistochastic, hence its trace is a convex combination of
                // the six permutation traces. This exact necessary gate
                // avoids constructing a plane quartic for an impossible
                // routed action.
                if !trace_in_permutation_hull(&aa, &bb, tau) {
                    continue;
                }
                if let Some(solved) = dense_rank_three_frame(
                    &aa,
                    &bb,
                    tau,
                    fixed_row,
                    fixed_col,
                    rows,
                    cols,
                    &mut |o| {
                        let residual = compound_residual(dc, lam, &o, &targets[branch]);
                        (residual <= ACCEPT).then_some((o, residual))
                    },
                ) {
                    return Some(solved);
                }
            }
        }
    }
    None
}

/// Necessary feasibility gate for a residual trace.  A `3 x 3`
/// orthostochastic matrix lies in the Birkhoff polytope, whose vertices are
/// the six permutation matrices.  In the complex trace plane, membership in
/// their convex hull is equivalent by Caratheodory to membership in one of
/// their twenty triangles.
fn trace_in_permutation_hull(a: &[C; 3], b: &[C; 3], tau: C) -> bool {
    const P3: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let vertex: [C; 6] = P3.map(|permutation| (0..3).map(|i| a[i] * b[permutation[i]]).sum::<C>());
    super::point_in_complex_hull(&vertex, tau)
}

type Poly = Vec<f64>;
type Bi4 = [[f64; 5]; 5];

fn poly_mul(left: &[f64], right: &[f64]) -> Poly {
    let mut product = vec![0.0; left.len() + right.len() - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            product[i + j] += a * b;
        }
    }
    product
}

fn poly_term(out: &mut Poly, coefficient: f64, factors: &[&Poly]) {
    let product = factors
        .iter()
        .fold(vec![1.0], |value, factor| poly_mul(&value, factor));
    if out.len() < product.len() {
        out.resize(product.len(), 0.0);
    }
    for (slot, value) in out.iter_mut().zip(product) {
        *slot += coefficient * value;
    }
}

/// Discriminant of `a*v^4+b*v^3+c*v^2+d*v+e`, coefficientwise in the
/// remaining Birkhoff coordinate.  For a total-degree-four plane curve the
/// result has degree at most twelve.
pub(super) fn quartic_discriminant_poly(f: &[Poly; 5]) -> Option<Poly> {
    let (e, d, c_, b, a) = (&f[0], &f[1], &f[2], &f[3], &f[4]);
    let (a2, a3) = (poly_mul(a, a), poly_mul(&poly_mul(a, a), a));
    let (b2, b3, b4) = {
        let b2 = poly_mul(b, b);
        let b3 = poly_mul(&b2, b);
        let b4 = poly_mul(&b2, &b2);
        (b2, b3, b4)
    };
    let (c2, c3, c4) = {
        let c2 = poly_mul(c_, c_);
        let c3 = poly_mul(&c2, c_);
        let c4 = poly_mul(&c2, &c2);
        (c2, c3, c4)
    };
    let (d2, d3, d4) = {
        let d2 = poly_mul(d, d);
        let d3 = poly_mul(&d2, d);
        let d4 = poly_mul(&d2, &d2);
        (d2, d3, d4)
    };
    let (e2, e3) = {
        let e2 = poly_mul(e, e);
        let e3 = poly_mul(&e2, e);
        (e2, e3)
    };
    let mut out = Vec::new();
    poly_term(&mut out, 256.0, &[&a3, &e3]);
    poly_term(&mut out, -192.0, &[&a2, b, d, &e2]);
    poly_term(&mut out, -128.0, &[&a2, &c2, &e2]);
    poly_term(&mut out, 144.0, &[&a2, c_, &d2, e]);
    poly_term(&mut out, -27.0, &[&a2, &d4]);
    poly_term(&mut out, 144.0, &[a, &b2, c_, &e2]);
    poly_term(&mut out, -6.0, &[a, &b2, &d2, e]);
    poly_term(&mut out, -80.0, &[a, b, &c2, d, e]);
    poly_term(&mut out, 18.0, &[a, b, c_, &d3]);
    poly_term(&mut out, 16.0, &[a, &c4, e]);
    poly_term(&mut out, -4.0, &[a, &c3, &d2]);
    poly_term(&mut out, -27.0, &[&b4, &e2]);
    poly_term(&mut out, 18.0, &[&b3, c_, d, e]);
    poly_term(&mut out, -4.0, &[&b3, &d3]);
    poly_term(&mut out, -4.0, &[&b2, &c3, e]);
    poly_term(&mut out, 1.0, &[&b2, &c2, &d2]);
    let scale = out.iter().fold(0.0f64, |maximum, x| maximum.max(x.abs()));
    while out.len() > 1 && out.last().is_some_and(|x| x.abs() < 1e-12 * scale) {
        out.pop();
    }
    (scale > 0.0 && scale.is_finite() && out.len() <= 13).then_some(out)
}

fn affine_product(left: [f64; 3], right: [f64; 3]) -> [[f64; 3]; 3] {
    const DEGREE: [(usize, usize); 3] = [(0, 0), (1, 0), (0, 1)];
    let mut product = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            product[DEGREE[i].0 + DEGREE[j].0][DEGREE[i].1 + DEGREE[j].1] += left[i] * right[j];
        }
    }
    product
}

fn bi_product(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> Bi4 {
    let mut product = [[0.0; 5]; 5];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    product[i + k][j + l] += left[i][j] * right[k][l];
                }
            }
        }
    }
    product
}

fn bi_add_scaled(out: &mut Bi4, value: &Bi4, scale: f64) {
    for i in 0..5 {
        for j in 0..5 {
            out[i][j] += scale * value[i][j];
        }
    }
}

fn eval_poly(coefficients: &[f64], value: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |result, coefficient| result * value + coefficient)
}

/// Complete dense residual of one routed 1+3 split. The trace plane cuts the
/// 3x3 Birkhoff polytope to two affine coordinates; orthostochasticity is one
/// Heron plane quartic. The nine Birkhoff faces were exhausted immediately
/// above, so every remaining compact component has a coordinate extremum.
#[allow(clippy::too_many_arguments)]
fn dense_rank_three_frame<R>(
    a: &[C; 3],
    b: &[C; 3],
    tau: C,
    fixed_row: usize,
    fixed_col: usize,
    rows: [usize; 3],
    cols: [usize; 3],
    accept: &mut impl FnMut(Mat4) -> Option<R>,
) -> Option<R> {
    // [constant, X, Y, Z, W] for the nine entries of a 3x3 bistochastic
    // matrix in the standard four-coordinate Birkhoff chart.
    const BF: [[f64; 5]; 9] = [
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0, -1.0],
        [1.0, -1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, -1.0, 0.0, -1.0],
        [-1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let mut trace = [C::new(0.0, 0.0); 5];
    for i in 0..3 {
        for j in 0..3 {
            let coefficient = a[i] * b[j];
            for k in 0..5 {
                trace[k] += coefficient * BF[3 * i + j][k];
            }
        }
    }
    let rhs = tau - trace[0];
    let mut pivot = None;
    let mut pivot_scale = 0.0f64;
    for p in 0..4 {
        for q in p + 1..4 {
            let det = trace[p + 1].re * trace[q + 1].im - trace[q + 1].re * trace[p + 1].im;
            if det.abs() > pivot_scale {
                pivot = Some((p, q, det));
                pivot_scale = det.abs();
            }
        }
    }
    let coefficient_scale = trace[1..]
        .iter()
        .fold(0.0f64, |maximum, z| maximum.max(z.norm()));
    let (p, q, determinant) = pivot?;
    if pivot_scale < 1e-12 * coefficient_scale * coefficient_scale {
        return None;
    }
    let free: [usize; 2] = {
        let mut result = [0usize; 2];
        let mut count = 0;
        for k in 0..4 {
            if k != p && k != q {
                result[count] = k;
                count += 1;
            }
        }
        result
    };
    let solve = |z: C| -> [f64; 2] {
        [
            (z.re * trace[q + 1].im - trace[q + 1].re * z.im) / determinant,
            (trace[p + 1].re * z.im - z.re * trace[p + 1].im) / determinant,
        ]
    };
    let base = solve(rhs);
    let du = solve(-trace[free[0] + 1]);
    let dv = solve(-trace[free[1] + 1]);
    let mut coordinate = [[0.0; 3]; 4];
    coordinate[p] = [base[0], du[0], dv[0]];
    coordinate[q] = [base[1], du[1], dv[1]];
    coordinate[free[0]] = [0.0, 1.0, 0.0];
    coordinate[free[1]] = [0.0, 0.0, 1.0];
    let entries: [[f64; 3]; 9] = std::array::from_fn(|entry| {
        let mut form = [BF[entry][0], 0.0, 0.0];
        for k in 0..4 {
            for degree in 0..3 {
                form[degree] += BF[entry][k + 1] * coordinate[k][degree];
            }
        }
        form
    });
    let products: [[[f64; 3]; 3]; 3] =
        std::array::from_fn(|j| affine_product(entries[j], entries[3 + j]));
    let mut heron = [[0.0; 5]; 5];
    for i in 0..3 {
        bi_add_scaled(&mut heron, &bi_product(&products[i], &products[i]), -1.0);
        for j in i + 1..3 {
            bi_add_scaled(&mut heron, &bi_product(&products[i], &products[j]), 2.0);
        }
    }
    let fv: [Poly; 5] = std::array::from_fn(|v_degree| {
        let mut coefficient: Poly = (0..=4 - v_degree)
            .map(|u_degree| heron[u_degree][v_degree])
            .collect();
        while coefficient.len() > 1 && coefficient.last().is_some_and(|x| x.abs() < 1e-15) {
            coefficient.pop();
        }
        coefficient
    });
    let selector = quartic_discriminant_poly(&fv)?;
    let approximate = poly_roots(&selector);
    let approximate_real: Vec<f64> = approximate
        .iter()
        .filter(|root| root.im.abs() < 2e-5 && (-1e-7..=1.0 + 1e-7).contains(&root.re))
        .map(|root| root.re.clamp(0.0, 1.0))
        .collect();
    let mut critical = approximate_real;
    critical.sort_by(f64::total_cmp);
    critical.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
    let mut cover = Vec::with_capacity(2 * critical.len() + 1);
    let mut previous = 0.0;
    for root in critical {
        cover.push(0.5 * (previous + root));
        cover.push(root);
        previous = root;
    }
    cover.push(0.5 * (previous + 1.0));
    cover.sort_by(|left, right| (left - 0.5).abs().total_cmp(&(right - 0.5).abs()));

    let heron_scale = heron
        .iter()
        .flatten()
        .fold(0.0f64, |maximum, value| maximum.max(value.abs()))
        .max(1e-300);
    for u in cover {
        let quartic: Vec<f64> = fv
            .iter()
            .map(|coefficient| eval_poly(coefficient, u))
            .collect();
        let approximate_v = poly_roots(&quartic);
        let approximate_real_v: Vec<f64> = approximate_v
            .iter()
            .filter(|root| root.im.abs() <= 2e-5 && (-1e-7..=1.0 + 1e-7).contains(&root.re))
            .map(|root| root.re.clamp(0.0, 1.0))
            .collect();
        let mut roots_v = approximate_real_v;
        // If F(u,v) is identically zero, this is a vertical component of the
        // plane quartic.  Its Birkhoff slice is an interval, so its midpoint
        // is an exact algebraic section.  Floating evaluation only proposes
        // the midpoint; the Gram and forward certificates remain decisive.
        let quartic_scale = quartic
            .iter()
            .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
        if quartic_scale <= 2e-8 * heron_scale {
            if let Some((lo, hi)) = birkhoff_v_interval(&entries, u) {
                roots_v.push(0.5 * (lo + hi));
            }
        }
        roots_v.sort_by(f64::total_cmp);
        roots_v.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
        for v in roots_v {
            let values: [f64; 9] = entries.map(|form| form[0] + form[1] * u + form[2] * v);
            if values
                .iter()
                .any(|value| *value < -2e-7 || *value > 1.0 + 2e-7)
            {
                continue;
            }
            let row0 = [values[0], values[1], values[2]];
            let row1 = [values[3], values[4], values[5]];
            let Some((first, second)) = heron_kernel_columns(row0, row1) else {
                continue;
            };
            let third = [
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ];
            let block = [first, second, third];
            let mut frame = Mat4::zeros();
            frame[(fixed_row, fixed_col)] = c(1.0, 0.0);
            for i in 0..3 {
                for j in 0..3 {
                    frame[(rows[i], cols[j])] = c(block[i][j], 0.0);
                }
            }
            let frame = orient_so4(frame);
            let Some(metrics) = frame_metrics(&frame) else {
                continue;
            };
            if !metrics.within(1e-11) {
                continue;
            }
            if let Some(solved) = accept(frame) {
                return Some(solved);
            }
        }
    }
    None
}

fn birkhoff_v_interval(entries: &[[f64; 3]; 9], u: f64) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for form in entries {
        let base = form[0] + form[1] * u;
        let slope = form[2];
        if slope.abs() <= 1e-14 {
            if !(-2e-8..=1.0 + 2e-8).contains(&base) {
                return None;
            }
            continue;
        }
        let x0 = -base / slope;
        let x1 = (1.0 - base) / slope;
        lo = lo.max(x0.min(x1));
        hi = hi.min(x0.max(x1));
    }
    (lo <= hi).then_some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)))
}

#[derive(Clone, Copy)]
enum ThreeSpectrum {
    Scalar,
    TwoPlusOne { common: C, singleton: usize },
    Generic,
}

fn three_spectrum(values: &[C; 3]) -> ThreeSpectrum {
    let same = |i: usize, j: usize| (values[i] - values[j]).norm() <= 1e-12;
    if same(0, 1) && same(0, 2) {
        return ThreeSpectrum::Scalar;
    }
    for (singleton, repeated0, repeated1) in [(0, 1, 2), (1, 0, 2), (2, 0, 1)] {
        if same(repeated0, repeated1) && !same(singleton, repeated0) {
            return ThreeSpectrum::TwoPlusOne {
                common: values[repeated0],
                singleton,
            };
        }
    }
    ThreeSpectrum::Generic
}

fn rank_drop_frame(
    a: &[C; 3],
    b: &[C; 3],
    tau: C,
    fixed_row: usize,
    fixed_col: usize,
    rows: [usize; 3],
    cols: [usize; 3],
) -> Option<Mat4> {
    let (ak, bk) = (three_spectrum(a), three_spectrum(b));
    let mut q = [[0.0; 3]; 3];
    match (ak, bk) {
        (ThreeSpectrum::Scalar, _) | (_, ThreeSpectrum::Scalar) => {
            for i in 0..3 {
                q[i][i] = 1.0;
            }
        }
        (
            ThreeSpectrum::TwoPlusOne {
                common: ac,
                singleton: ia,
            },
            ThreeSpectrum::TwoPlusOne {
                common: bc,
                singleton: ib,
            },
        ) => {
            let as_ = a[ia];
            let bs = b[ib];
            let base = ac * bc + ac * bs + as_ * bc;
            let coefficient = (as_ - ac) * (bs - bc);
            if coefficient.norm() <= 1e-14 {
                return None;
            }
            let x = (tau - base) / coefficient;
            if x.im.abs() > 1e-8 || !(-1e-8..=1.0 + 1e-8).contains(&x.re) {
                return None;
            }
            let x = x.re.clamp(0.0, 1.0);
            let mut permutation = [usize::MAX; 3];
            permutation[ia] = ib;
            let remaining_rows = match ia {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            let remaining_cols = match ib {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            for k in 0..2 {
                permutation[remaining_rows[k]] = remaining_cols[k];
            }
            for i in 0..3 {
                q[i][permutation[i]] = 1.0;
            }
            let other = remaining_rows[0];
            let (ct, st) = (x.sqrt(), (1.0 - x).sqrt());
            let old_ia = q[ia];
            let old_other = q[other];
            for j in 0..3 {
                q[ia][j] = ct * old_ia[j] - st * old_other[j];
                q[other][j] = st * old_ia[j] + ct * old_other[j];
            }
        }
        _ => return None,
    }
    let mut o = Mat4::zeros();
    o[(fixed_row, fixed_col)] = c(1.0, 0.0);
    for i in 0..3 {
        for j in 0..3 {
            o[(rows[i], cols[j])] = c(q[i][j], 0.0);
        }
    }
    Some(orient_so4(o))
}

fn recover_y(rr: [f64; 4], ri: [f64; 4], x: f64, out: &mut Vec<(f64, f64)>) {
    let (nr, dr) = (rr[0] + rr[1] * x, rr[2] + rr[3] * x);
    let (ni, di) = (ri[0] + ri[1] * x, ri[2] + ri[3] * x);
    if let Some(y) = solve_real_lines(nr, dr, ni, di) {
        out.push((x, y));
    } else if dr.abs().max(di.abs()) <= 1e-14 && nr.abs().max(ni.abs()) <= 1e-8 {
        out.push((x, 0.0));
        out.push((x, 1.0));
    }
}

fn solve_real_lines(a0: f64, a1: f64, b0: f64, b1: f64) -> Option<f64> {
    let t = if a1.abs() >= b1.abs() && a1.abs() > 1e-14 {
        -a0 / a1
    } else if b1.abs() > 1e-14 {
        -b0 / b1
    } else {
        return None;
    };
    ((a0 + a1 * t).abs().max((b0 + b1 * t).abs()) <= 1e-8).then_some(t)
}

fn frame(xy: [f64; 2], planes: [(usize, usize); 2], perm: [usize; 4]) -> Mat4 {
    let mut o = signed_perm(perm);
    for (v, &(i, j)) in xy.into_iter().zip(&planes) {
        let v = v.clamp(0.0, 1.0);
        let (ct, st) = (v.sqrt(), (1.0 - v).sqrt());
        let mut g = Mat4::identity();
        g[(i, i)] = c(ct, 0.0);
        g[(j, j)] = c(ct, 0.0);
        g[(i, j)] = c(-st, 0.0);
        g[(j, i)] = c(st, 0.0);
        o = g * o;
    }
    o
}

fn e1_corners(a: &[C; 4], b: &[C; 4], planes: [(usize, usize); 2], perm: [usize; 4]) -> [C; 4] {
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].map(|xy| {
        let mut p = perm;
        for (v, &(i, j)) in xy.into_iter().zip(&planes) {
            if v == 0.0 {
                p.swap(i, j);
            }
        }
        (0..4).map(|i| a[i] * b[p[i]]).sum()
    })
}

fn quadratic_unit_roots(q: [f64; 3]) -> Vec<f64> {
    let scale = q.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return Vec::new();
    }
    let eps = 64.0 * f64::EPSILON * scale;
    let mut roots = Vec::with_capacity(2);
    if q[2].abs() <= eps {
        if q[1].abs() > eps {
            roots.push(-q[0] / q[1]);
        }
    } else {
        let disc = q[1] * q[1] - 4.0 * q[2] * q[0];
        let disc_scale = q[1] * q[1] + (4.0 * q[2] * q[0]).abs();
        if disc >= -128.0 * f64::EPSILON * disc_scale {
            let root = disc.max(0.0).sqrt();
            let h = -0.5 * (q[1] + if q[1] < 0.0 { -root } else { root });
            if h == 0.0 {
                roots.push(-q[1] / (2.0 * q[2]));
            } else {
                roots.push(h / q[2]);
                roots.push(q[0] / h);
            }
        }
    }
    roots.retain(|x| x.is_finite() && (-1e-10..=1.0 + 1e-10).contains(x));
    for x in &mut roots {
        *x = x.clamp(0.0, 1.0);
    }
    if roots.len() == 2 && (roots[0] - roots[1]).abs() <= 1e-12 {
        roots.pop();
    }
    roots
}

/// Recover the two signed orthonormal core columns from their squared entries.
///
/// Put `x_i=alpha_i*beta_i`.  On the Heron locus, choosing one nonzero pivot
/// and its positive square root determines the other two signed products by
/// the rank-one kernel of
///
/// `[[2*x_i, x_i+x_j-x_k], [x_i+x_j-x_k, 2*x_j]]`.
///
/// Thus no relative-sign enumeration or Gram--Schmidt projection is needed.
/// The final two norm scalings only remove floating evaluation error; they are
/// identities on the exact locus.
fn heron_kernel_columns(alpha_raw: [f64; 3], beta_raw: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let alpha = alpha_raw.map(|value| value.max(0.0));
    let beta = beta_raw.map(|value| value.max(0.0));
    let x: [f64; 3] = std::array::from_fn(|k| alpha[k] * beta[k]);
    let pivot = (0..3).max_by(|&i, &j| x[i].total_cmp(&x[j])).unwrap_or(0);
    let mut q = [0.0; 3];
    if x[pivot] > 0.0 {
        let j = (pivot + 1) % 3;
        let k = (pivot + 2) % 3;
        q[pivot] = x[pivot].sqrt();
        q[j] = -(x[pivot] + x[j] - x[k]) / (2.0 * q[pivot]);
        // Enforce the exact kernel sum in floating arithmetic.  The Heron
        // identity then gives q[k]^2=x[k].
        q[k] = -q[pivot] - q[j];
    }

    let mut a = [0.0; 3];
    let mut b = [0.0; 3];
    for k in 0..3 {
        // Root the larger squared coordinate and recover the smaller by one
        // division.  This is the stable four-square-root realization of the
        // signed products q[k].
        if alpha[k] >= beta[k] {
            a[k] = alpha[k].sqrt();
            if a[k] > 0.0 {
                b[k] = q[k] / a[k];
            } else if q[k] != 0.0 {
                return None;
            }
        } else {
            b[k] = beta[k].sqrt();
            if b[k] > 0.0 {
                a[k] = q[k] / b[k];
            } else if q[k] != 0.0 {
                return None;
            }
        }
    }

    let an = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    let bn = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !an.is_finite() || !bn.is_finite() || an < 1e-12 || bn < 1e-12 {
        return None;
    }
    for k in 0..3 {
        a[k] /= an;
        b[k] /= bn;
    }
    let dot = (0..3).map(|k| a[k] * b[k]).sum::<f64>();
    (dot.abs() <= 2e-10).then_some((a, b))
}
