//! Constructive solution of the rank-four real multiplicative-Horn problem.
//!
//! The algorithm has one mathematical dispatch, not an empirical cascade:
//! tight quantum-Horn faces recurse into lower-rank blocks; the relative
//! interior is lifted from a fixed exactly certified irreducible rational
//! anchor.  Repeated-spectrum Weyl walls are obtained as regular limits.  In
//! ranks at most four Falbel--Wentworth prove that there are no inner walls, so
//! these cases exhaust every feasible datum.

use super::{
    compound_residual, eigphases, esym4, mmat, orient_so4, rho_weyl, symfn, weyl_from_monodromy,
    Mat4, Rung, Solution, ACCEPT, C,
};
use nalgebra::{SMatrix, SVector};
use std::sync::atomic::{AtomicUsize, Ordering};

static PATH_CALLS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
static CORRECT_CALLS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
static CORRECT_ITERS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
static FACE_EQUALITIES: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
static FACE_HITS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];

fn profile_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("CONSTRUCTIVE_PROFILE").is_some());
    *ENABLED
}

pub fn reset_constructive_profile() {
    for counters in [
        &PATH_CALLS,
        &CORRECT_CALLS,
        &CORRECT_ITERS,
        &FACE_EQUALITIES,
        &FACE_HITS,
    ] {
        for counter in counters {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

pub fn constructive_profile() -> String {
    let load = |a: &[AtomicUsize; 5]| {
        (1..=4)
            .map(|n| a[n].load(Ordering::Relaxed))
            .collect::<Vec<_>>()
    };
    format!(
        "path {:?} correct {:?} iterations {:?} face_equal {:?} face_hit {:?}",
        load(&PATH_CALLS),
        load(&CORRECT_CALLS),
        load(&CORRECT_ITERS),
        load(&FACE_EQUALITIES),
        load(&FACE_HITS),
    )
}

// Quantum Littlewood--Richardson facet labels for QH*(Gr(r,4)).  Masks use
// the chamber order [m0,m1,m2,-sum]; `qlr_mask_indices` converts that order to
// the magic-basis diagonal order used by `eigphases`.  The table is generated
// from the repository's single QLR source (`core/chamber/qlr.rs`).
const QLR4_FACES: [(usize, u8, u8, u8); 72] = [
    (1, 8, 8, 8),
    (1, 8, 4, 4),
    (1, 4, 8, 4),
    (1, 8, 2, 2),
    (1, 2, 8, 2),
    (1, 8, 1, 1),
    (1, 1, 8, 1),
    (1, 4, 4, 2),
    (1, 4, 2, 1),
    (1, 2, 4, 1),
    (1, 4, 1, 8),
    (1, 1, 4, 8),
    (1, 2, 2, 8),
    (1, 2, 1, 4),
    (1, 1, 2, 4),
    (1, 1, 1, 2),
    (2, 12, 12, 12),
    (2, 12, 10, 10),
    (2, 10, 12, 10),
    (2, 12, 6, 6),
    (2, 6, 12, 6),
    (2, 12, 9, 9),
    (2, 9, 12, 9),
    (2, 12, 5, 5),
    (2, 5, 12, 5),
    (2, 12, 3, 3),
    (2, 3, 12, 3),
    (2, 10, 10, 6),
    (2, 10, 10, 9),
    (2, 10, 6, 5),
    (2, 6, 10, 5),
    (2, 10, 9, 5),
    (2, 9, 10, 5),
    (2, 10, 5, 3),
    (2, 5, 10, 3),
    (2, 10, 5, 12),
    (2, 5, 10, 12),
    (2, 10, 3, 10),
    (2, 3, 10, 10),
    (2, 6, 6, 3),
    (2, 6, 9, 12),
    (2, 9, 6, 12),
    (2, 6, 5, 10),
    (2, 5, 6, 10),
    (2, 6, 3, 9),
    (2, 3, 6, 9),
    (2, 9, 9, 3),
    (2, 9, 5, 10),
    (2, 5, 9, 10),
    (2, 9, 3, 6),
    (2, 3, 9, 6),
    (2, 5, 5, 9),
    (2, 5, 5, 6),
    (2, 5, 3, 5),
    (2, 3, 5, 5),
    (2, 3, 3, 12),
    (3, 14, 14, 14),
    (3, 14, 13, 13),
    (3, 13, 14, 13),
    (3, 14, 11, 11),
    (3, 11, 14, 11),
    (3, 14, 7, 7),
    (3, 7, 14, 7),
    (3, 13, 13, 11),
    (3, 13, 11, 7),
    (3, 11, 13, 7),
    (3, 13, 7, 14),
    (3, 7, 13, 14),
    (3, 11, 11, 14),
    (3, 11, 7, 13),
    (3, 7, 11, 13),
    (3, 7, 7, 11),
];

/// Backward-compatible total entry point.  It is now the single constructive
/// algorithm; no algebraic-atlas or global-net fallback is invoked.
pub fn solve_total(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    solve_constructive(c, g, t)
}

/// Canonical-anchor continuation alone, exposed for proof diagnostics and
/// benchmarking.  The rational anchor is irreducible for every pair of input
/// multiplicity patterns for which an irreducible pair can exist in rank four.
/// Outer Horn faces deliberately return `Unsolved` here; their tight equality
/// supplies the lower-rank block decomposition used by the global recursion.
pub fn solve_constructive(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    let eb = eigphases(weyl_from_monodromy(c));
    let ep = eigphases(weyl_from_monodromy(g));
    let wt = weyl_from_monodromy(t);
    let et = [eigphases(wt), eigphases(rho_weyl(wt))];
    let dc_diag: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, eb[k]));
    let lam_diag: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * ep[k]));
    let dc = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&dc_diag));
    let lam = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&lam_diag));
    let target_specs: [[C; 4]; 2] = std::array::from_fn(|branch| {
        std::array::from_fn(|k| C::from_polar(1.0, 2.0 * et[branch][k]))
    });
    let targets: [[C; 4]; 2] = target_specs.map(esym4);
    for branch in 0..2 {
        if let Some(o) = solve_rank_recursive(4, &dc_diag, &lam_diag, &target_specs[branch]) {
            let residual = compound_residual(&dc, &lam, &o, &targets[branch]);
            if residual < ACCEPT {
                return Solution {
                    o: orient_so4(o),
                    rung: Rung::Constructive,
                    residual,
                };
            }
        }
    }
    Solution {
        o: Mat4::identity(),
        rung: Rung::Unsolved,
        residual: f64::INFINITY,
    }
}

/// Compatibility alias retained for the existing benchmark switch.
pub fn solve_hadamard(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    solve_constructive(c, g, t)
}

fn canonical_anchor_frame() -> Mat4 {
    // Cayley transform of the integral skew matrix with upper triangle
    // (-5, 9, -7, -1, -6, 6).  Exact arithmetic gives O^T O=I and det O=1.
    // An exhaustive commutant-rank certificate over all ordered multiplicity
    // patterns of four is recorded in s194_universal_irreducible_anchor.py.
    let d = 595.0;
    Mat4::from_row_slice(&[
        C::new(-521.0 / d, 0.0),
        C::new(-214.0 / d, 0.0),
        C::new(-148.0 / d, 0.0),
        C::new(122.0 / d, 0.0),
        C::new(148.0 / d, 0.0),
        C::new(-428.0 / d, 0.0),
        C::new(299.0 / d, 0.0),
        C::new(244.0 / d, 0.0),
        C::new(242.0 / d, 0.0),
        C::new(-137.0 / d, 0.0),
        C::new(-484.0 / d, 0.0),
        C::new(206.0 / d, 0.0),
        C::new(46.0 / d, 0.0),
        C::new(-326.0 / d, 0.0),
        C::new(-92.0 / d, 0.0),
        C::new(-487.0 / d, 0.0),
    ])
}

fn active_diagonal(values: &[C; 4], n: usize) -> Mat4 {
    let mut out = Mat4::zeros();
    for i in 0..n {
        out[(i, i)] = values[i];
    }
    out
}

fn active_esym(values: &[C; 4], n: usize) -> [C; 4] {
    let mut padded = [C::new(0.0, 0.0); 4];
    padded[..n].copy_from_slice(&values[..n]);
    esym4(padded)
}

fn active_spectrum(m: &Mat4, n: usize) -> Option<[C; 4]> {
    let e = symfn(m);
    let mut coeff = Vec::with_capacity(n + 1);
    coeff.push(C::new(1.0, 0.0));
    for k in 0..n {
        coeff.push(if k % 2 == 0 { -e[k] } else { e[k] });
    }
    let roots = crate::cpoly::roots(&coeff);
    if roots.len() != n {
        return None;
    }
    let mut out = [C::new(1.0, 0.0); 4];
    out[..n].copy_from_slice(&roots);
    Some(out)
}

fn normalized_alcove(values: &[C; 4], n: usize, center: C) -> Option<[f64; 4]> {
    let mut x = [0.0; 4];
    for i in 0..n {
        x[i] = (values[i] / center).arg() / std::f64::consts::TAU;
    }
    x[..n].sort_by(|a, b| b.total_cmp(a));
    let winding = x[..n].iter().sum::<f64>().round() as i32;
    if winding > 0 {
        for item in x.iter_mut().take(winding as usize) {
            *item -= 1.0;
        }
    } else if winding < 0 {
        for item in x[..n].iter_mut().rev().take((-winding) as usize) {
            *item += 1.0;
        }
    }
    x[..n].sort_by(|a, b| b.total_cmp(a));
    (x[..n].iter().sum::<f64>().abs() < 2e-8 && x[0] - x[n - 1] < 1.0 + 2e-8).then_some(x)
}

fn values_from_normalized_alcove(a: &[f64; 4], n: usize, center: C) -> [C; 4] {
    let mut out = [C::new(1.0, 0.0); 4];
    for i in 0..n {
        out[i] = center * C::from_polar(1.0, std::f64::consts::TAU * a[i]);
    }
    out
}

fn rank_anchor_frame(n: usize) -> Mat4 {
    if n == 3 {
        // Quaternion (1,2,3,4), normalized.  This is irreducible whenever one
        // of the two non-scalar rank-three spectra is simple; the remaining
        // pattern pairs are forced onto a face by the two-projection CS split.
        let d = 30.0;
        let mut out = Mat4::identity();
        let q = [[-20.0, 4.0, 22.0], [20.0, -10.0, 20.0], [10.0, 28.0, 4.0]];
        for i in 0..3 {
            for j in 0..3 {
                out[(i, j)] = C::new(q[i][j] / d, 0.0);
            }
        }
        out
    } else {
        canonical_anchor_frame()
    }
}

fn spectral_coords(e: &[C; 4], n: usize, det: C) -> SVector<f64, 3> {
    match n {
        2 => {
            let half_det = C::from_polar(1.0, det.arg() / 2.0);
            SVector::<f64, 3>::new((e[0] / half_det).re, 0.0, 0.0)
        }
        3 => SVector::<f64, 3>::new(e[0].re, e[0].im, 0.0),
        4 => {
            let half_det = C::from_polar(1.0, det.arg() / 2.0);
            SVector::<f64, 3>::new(e[0].re, e[0].im, (e[1] / half_det).re)
        }
        _ => SVector::zeros(),
    }
}

fn solve_small_gram(
    gram: &SMatrix<f64, 3, 3>,
    err: &SVector<f64, 3>,
    m: usize,
) -> Option<SVector<f64, 3>> {
    let mut y = SVector::<f64, 3>::zeros();
    match m {
        1 => {
            if gram[(0, 0)].abs() < 1e-15 {
                return None;
            }
            y[0] = err[0] / gram[(0, 0)];
        }
        2 => {
            let det = gram[(0, 0)] * gram[(1, 1)] - gram[(0, 1)] * gram[(1, 0)];
            if det.abs() < 1e-15 {
                return None;
            }
            y[0] = (gram[(1, 1)] * err[0] - gram[(0, 1)] * err[1]) / det;
            y[1] = (-gram[(1, 0)] * err[0] + gram[(0, 0)] * err[1]) / det;
        }
        3 => return gram.lu().solve(err),
        _ => return None,
    }
    Some(y)
}

/// The two nonconstant spectral invariants needed in ranks at most four,
/// evaluated directly from diagonal spectra and a real orthogonal frame.
/// Cauchy--Binet avoids forming `D O Lambda O^T D` and its matrix powers.
#[inline]
fn direct_e1_e2(n: usize, a: &[C; 4], b: &[C; 4], o: &Mat4) -> (C, C) {
    let mut e1 = C::new(0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            let x = o[(i, j)].re;
            e1 += a[i] * b[j] * (x * x);
        }
    }
    if n < 4 {
        return (e1, C::new(0.0, 0.0));
    }
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut e2 = C::new(0.0, 0.0);
    for &(i0, i1) in &PAIRS {
        let ai = a[i0] * a[i1];
        for &(j0, j1) in &PAIRS {
            let minor = o[(i0, j0)].re * o[(i1, j1)].re - o[(i0, j1)].re * o[(i1, j0)].re;
            e2 += ai * b[j0] * b[j1] * (minor * minor);
        }
    }
    (e1, e2)
}

#[inline]
fn direct_coords(n: usize, a: &[C; 4], b: &[C; 4], o: &Mat4, det: C) -> SVector<f64, 3> {
    let (e1, e2) = direct_e1_e2(n, a, b, o);
    match n {
        3 => SVector::<f64, 3>::new(e1.re, e1.im, 0.0),
        4 => {
            let half_det = C::from_polar(1.0, det.arg() / 2.0);
            SVector::<f64, 3>::new(e1.re, e1.im, (e2 / half_det).re)
        }
        _ => SVector::zeros(),
    }
}

type J36 = SMatrix<f64, 3, 6>;

#[inline]
fn pair_index_sign(x: usize, y: usize) -> Option<(usize, f64)> {
    if x == y {
        return None;
    }
    let (lo, hi, sign) = if x < y { (x, y, 1.0) } else { (y, x, -1.0) };
    let idx = match (lo, hi) {
        (0, 1) => 0,
        (0, 2) => 1,
        (0, 3) => 2,
        (1, 2) => 3,
        (1, 3) => 4,
        (2, 3) => 5,
        _ => return None,
    };
    Some((idx, sign))
}

/// Invariants and their right-Givens Jacobian.  On `Lambda^2 R^4`, a plane
/// rotation mixes only the four compound columns containing exactly one plane
/// index; this is the sparse exterior-square differential of Cauchy--Binet.
#[inline]
fn direct_coords_jacobian(
    n: usize,
    a: &[C; 4],
    b: &[C; 4],
    o: &Mat4,
    det: C,
    planes: &[(usize, usize)],
) -> (SVector<f64, 3>, J36) {
    let mut e1 = C::new(0.0, 0.0);
    let mut de1 = [C::new(0.0, 0.0); 6];
    for i in 0..n {
        for j in 0..n {
            let x = o[(i, j)].re;
            e1 += a[i] * b[j] * (x * x);
        }
        for (k, &(p, q)) in planes.iter().enumerate() {
            de1[k] += 2.0 * a[i] * (b[p] - b[q]) * (o[(i, p)].re * o[(i, q)].re);
        }
    }

    let mut j = J36::zeros();
    if n == 3 {
        for k in 0..planes.len() {
            j[(0, k)] = de1[k].re;
            j[(1, k)] = de1[k].im;
        }
        return (SVector::<f64, 3>::new(e1.re, e1.im, 0.0), j);
    }

    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut minors = [[0.0f64; 6]; 6];
    let mut e2 = C::new(0.0, 0.0);
    for (ri, &(i0, i1)) in PAIRS.iter().enumerate() {
        let ai = a[i0] * a[i1];
        for (cj, &(j0, j1)) in PAIRS.iter().enumerate() {
            let m = o[(i0, j0)].re * o[(i1, j1)].re - o[(i0, j1)].re * o[(i1, j0)].re;
            minors[ri][cj] = m;
            e2 += ai * b[j0] * b[j1] * (m * m);
        }
    }
    let half_det = C::from_polar(1.0, det.arg() / 2.0);
    for (k, &(p, q)) in planes.iter().enumerate() {
        let mut de2 = C::new(0.0, 0.0);
        for (cj, &(j0, j1)) in PAIRS.iter().enumerate() {
            let mapped = if j0 == p && j1 != q {
                pair_index_sign(q, j1)
            } else if j1 == p && j0 != q {
                pair_index_sign(j0, q)
            } else if j0 == q && j1 != p {
                pair_index_sign(p, j1).map(|(idx, s)| (idx, -s))
            } else if j1 == q && j0 != p {
                pair_index_sign(j0, p).map(|(idx, s)| (idx, -s))
            } else {
                None
            };
            let Some((other, sign)) = mapped else {
                continue;
            };
            let bj = b[j0] * b[j1];
            for (ri, &(i0, i1)) in PAIRS.iter().enumerate() {
                de2 += 2.0 * a[i0] * a[i1] * bj * (minors[ri][cj] * sign * minors[ri][other]);
            }
        }
        j[(0, k)] = de1[k].re;
        j[(1, k)] = de1[k].im;
        j[(2, k)] = (de2 / half_det).re;
    }
    (SVector::<f64, 3>::new(e1.re, e1.im, (e2 / half_det).re), j)
}

/// Right-multiply a real frame by a Givens rotation without a general complex
/// matrix multiplication.  Only the two selected columns change.
#[inline]
fn rotate_columns(o: &mut Mat4, a: usize, b: usize, theta: f64, n: usize) {
    let (ct, st) = (theta.cos(), theta.sin());
    for i in 0..n {
        let (x, y) = (o[(i, a)].re, o[(i, b)].re);
        o[(i, a)] = C::new(ct * x + st * y, 0.0);
        o[(i, b)] = C::new(-st * x + ct * y, 0.0);
    }
}

fn correct_target_rank(
    n: usize,
    dc: &Mat4,
    lam: &Mat4,
    target_values: &[C; 4],
    start: &Mat4,
) -> Option<Mat4> {
    if profile_enabled() {
        CORRECT_CALLS[n].fetch_add(1, Ordering::Relaxed);
    }
    let target_e = active_esym(target_values, n);
    let det = target_values[..n].iter().copied().product::<C>();
    let target_coord = spectral_coords(&target_e, n, det);
    let mcoord = n - 1;
    let a: [C; 4] = std::array::from_fn(|i| dc[(i, i)] * dc[(i, i)]);
    let b: [C; 4] = std::array::from_fn(|i| lam[(i, i)]);
    let planes: Vec<(usize, usize)> = super::PLANES
        .iter()
        .copied()
        .filter(|&(a, b)| a < n && b < n)
        .collect();
    let mut o = *start;

    for _ in 0..128 {
        if profile_enabled() {
            CORRECT_ITERS[n].fetch_add(1, Ordering::Relaxed);
        }
        let (current_coord, j) = direct_coords_jacobian(n, &a, &b, &o, det, &planes);
        let err = current_coord - target_coord;
        if err.rows(0, mcoord).amax() < 2e-13 {
            return Some(o);
        }
        let gram = j * j.transpose();
        let y = solve_small_gram(&gram, &err, mcoord).or_else(|| {
            let damped = gram + SMatrix::<f64, 3, 3>::identity() * (1e-14 * gram.amax().max(1.0));
            solve_small_gram(&damped, &err, mcoord)
        })?;
        let mut delta = -(j.transpose() * y);
        let dn = delta.norm();
        if !dn.is_finite() {
            return None;
        }
        if dn > 0.5 {
            delta *= 0.5 / dn;
        }
        let current = err.rows(0, mcoord).amax();
        let mut accepted = false;
        let mut scale = 1.0;
        for _ in 0..12 {
            let mut candidate = o;
            for (k, &(a, b)) in planes.iter().enumerate() {
                rotate_columns(&mut candidate, a, b, scale * delta[k], n);
            }
            let next = direct_coords(n, &a, &b, &candidate, det) - target_coord;
            if next.rows(0, mcoord).amax() < current {
                o = candidate;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            return None;
        }
    }
    let err = direct_coords(n, &a, &b, &o, det) - target_coord;
    (err.rows(0, mcoord).amax() < 2e-10).then_some(o)
}

fn path_lift_rank(
    n: usize,
    dc_values: &[C; 4],
    lam_values: &[C; 4],
    target: &[C; 4],
) -> Option<Mat4> {
    if profile_enabled() {
        PATH_CALLS[n].fetch_add(1, Ordering::Relaxed);
    }
    let repeated_target = (0..n).any(|i| (i + 1..n).any(|j| (target[i] - target[j]).norm() < 1e-9));
    if repeated_target {
        // Two regular approaches to the same Weyl-wall limit.  Gradual endpoint
        // continuation and a direct tight endpoint have complementary numerical
        // conditioning, while both are instances of the proved regular-limit lift.
        path_lift_rank_start(n, dc_values, lam_values, target, 30)
            .or_else(|| path_lift_rank_start(n, dc_values, lam_values, target, 40))
    } else {
        path_lift_rank_start(n, dc_values, lam_values, target, 0)
    }
}

fn path_lift_rank_start(
    n: usize,
    dc_values: &[C; 4],
    lam_values: &[C; 4],
    target: &[C; 4],
    initial_endpoint_power: i32,
) -> Option<Mat4> {
    let dc = active_diagonal(dc_values, n);
    let lam = active_diagonal(lam_values, n);
    let anchor_o = rank_anchor_frame(n);
    let det = target[..n].iter().copied().product::<C>();
    let repeated_target = initial_endpoint_power != 0;
    // The overwhelmingly common regular case closes in the anchor chart itself.
    // It is already the s=1 trial of the continuation below, so do it before
    // paying for the anchor polynomial roots used only by path subdivision.
    if !repeated_target {
        if let Some(o) = correct_target_rank(n, &dc, &lam, target, &anchor_o) {
            return Some(o);
        }
    }
    let anchor_spec = active_spectrum(&mmat(&dc, &lam, &anchor_o), n)?;
    let center = C::from_polar(1.0, det.arg() / n as f64);
    let anchor = normalized_alcove(&anchor_spec, n, center)?;
    let target_a = normalized_alcove(target, n, center)?;
    let target_e = active_esym(target, n);
    let target_coord = spectral_coords(&target_e, n, det);
    // Stay on the regular side of a Weyl wall.  The target invariants are the
    // continuous limit, so this gives arbitrary prescribed accuracy without
    // asking Newton to invert the rank-deficient endpoint differential.
    let mut endpoint_power = initial_endpoint_power;
    let mut s_end = if repeated_target {
        1.0 - 2.0f64.powi(-endpoint_power)
    } else {
        1.0
    };
    let mut o = anchor_o;
    let mut s = 0.0f64;
    // Try the whole regular segment first.  Failed Newton correction merely
    // bisects this same path, preserving the continuation proof while making
    // the overwhelmingly common one-chart case a single solve.
    let mut h = if repeated_target { 1.0 } else { 0.5 };
    loop {
        while s < s_end - 1e-15 {
            let sn = (s + h).min(s_end);
            let mut a = [0.0; 4];
            for i in 0..n {
                a[i] = (1.0 - sn) * anchor[i] + sn * target_a[i];
            }
            let intermediate = values_from_normalized_alcove(&a, n, center);
            let Some(candidate) = correct_target_rank(n, &dc, &lam, &intermediate, &o) else {
                if std::env::var_os("TRACE_PATH").is_some() {
                    eprintln!("path n={n} correction failed s={s:.16e} sn={sn:.16e} h={h:.3e}");
                }
                h *= 0.5;
                if h < 2.0f64.powi(-44) {
                    return None;
                }
                continue;
            };
            o = candidate;
            s = sn;
            // A repeated target spectrum is an endpoint of the alcove chart.  The
            // invariant map remains continuous there even when eigenangle
            // coordinates lose rank, so a sufficiently close interior lift is a
            // certified numerical realization and avoids demanding a singular
            // final Newton solve.
            let got = spectral_coords(&symfn(&mmat(&dc, &lam, &o)), n, det);
            let target_error = (got - target_coord).rows(0, n - 1).amax();
            if std::env::var_os("TRACE_PATH").is_some() {
                eprintln!("path n={n} accepted s={s:.16e} h={h:.3e} target_err={target_error:.3e}");
            }
            if target_error < 2e-11 {
                return Some(o);
            }
            h = (h * 1.5).min(0.25);
        }
        if !repeated_target {
            break;
        }
        endpoint_power += 4;
        if endpoint_power > 46 {
            return None;
        }
        s_end = 1.0 - 2.0f64.powi(-endpoint_power);
        h = h.min((s_end - s).max(2.0f64.powi(-46)));
    }
    correct_target_rank(n, &dc, &lam, target, &o)
}

fn qlr_eig_mask(mask: u8) -> u8 {
    // QLR chamber order -> eigphases order:
    // [m0,m1,m2,-sum] -> [m1,m0,-sum,m2].
    const TO_EIG: [usize; 4] = [1, 0, 3, 2];
    let mut out = 0u8;
    for i in 0..4 {
        if mask & (1 << i) != 0 {
            out |= 1 << TO_EIG[i];
        }
    }
    out
}

fn has_repeated_value(values: &[C; 4], n: usize) -> bool {
    (0..n).any(|i| (i + 1..n).any(|j| (values[i] - values[j]).norm() < 1e-9))
}

#[inline]
fn select_mask(values: &[C; 4], mask: u8, n: usize) -> [C; 4] {
    let mut out = [C::new(1.0, 0.0); 4];
    let mut dst = 0;
    for src in 0..n {
        if mask & (1 << src) != 0 {
            out[dst] = values[src];
            dst += 1;
        }
    }
    out
}

#[inline]
fn product_mask(values: &[C; 4], mask: u8, n: usize) -> C {
    let mut out = C::new(1.0, 0.0);
    for i in 0..n {
        if mask & (1 << i) != 0 {
            out *= values[i];
        }
    }
    out
}

fn rank_three_horn_feasible(dc_values: &[C; 4], lam_values: &[C; 4], target: &[C; 4]) -> bool {
    // Agnihotri--Woodward SU-alcove inequalities.  Choose cube roots for A
    // and B independently and normalize W by their product; then A'B'=W'
    // with all three determinants one and no hidden central shift.
    let mut a_values = [C::new(1.0, 0.0); 4];
    for i in 0..3 {
        a_values[i] = dc_values[i] * dc_values[i];
    }
    let det_a = a_values[..3].iter().copied().product::<C>();
    let det_b = lam_values[..3].iter().copied().product::<C>();
    let center_a = C::from_polar(1.0, det_a.arg() / 3.0);
    let center_b = C::from_polar(1.0, det_b.arg() / 3.0);
    let Some(alpha) = normalized_alcove(&a_values, 3, center_a) else {
        return false;
    };
    let Some(beta) = normalized_alcove(lam_values, 3, center_b) else {
        return false;
    };
    let Some(gamma) = normalized_alcove(target, 3, center_a * center_b) else {
        return false;
    };

    // QH*(Gr(1,3)) and QH*(Gr(2,3)) are both the quantum ring of P^2.
    // For Schubert exponents p,q in 0..2, p*q has exponent (p+q) mod 3
    // and degree floor((p+q)/3).  The subset map is the standard phi map.
    const R1: [[usize; 2]; 3] = [[2, 3], [1, 3], [0, 3]];
    const R2: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];
    for subsets in [R1, R2] {
        for p in 0..3 {
            for q in 0..3 {
                let r = (p + q) % 3;
                let degree = ((p + q) / 3) as f64;
                let sum_subset = |x: &[f64; 4], entry: [usize; 2]| {
                    if entry[1] == 3 {
                        x[entry[0]]
                    } else {
                        x[entry[0]] + x[entry[1]]
                    }
                };
                let lhs = sum_subset(&alpha, subsets[p]) + sum_subset(&beta, subsets[q])
                    - sum_subset(&gamma, subsets[r]);
                if lhs > degree + 2e-8 {
                    return false;
                }
            }
        }
    }
    true
}

fn solve_rank_two(dc_values: &[C; 4], lam_values: &[C; 4], target: &[C; 4]) -> Option<Mat4> {
    let a = [dc_values[0] * dc_values[0], dc_values[1] * dc_values[1]];
    let b = [lam_values[0], lam_values[1]];
    let denom = (a[0] - a[1]) * (b[0] - b[1]);
    if denom.norm() < 1e-12 {
        return Some(Mat4::identity());
    }
    let cross = a[0] * b[1] + a[1] * b[0];
    let xz = (target[0] + target[1] - cross) / denom;
    if xz.im.abs() > 2e-7 || xz.re < -2e-8 || xz.re > 1.0 + 2e-8 {
        return None;
    }
    let x = xz.re.clamp(0.0, 1.0);
    let (c, s) = (x.sqrt(), (1.0 - x).sqrt());
    let mut o = Mat4::identity();
    o[(0, 0)] = C::new(c, 0.0);
    o[(0, 1)] = C::new(-s, 0.0);
    o[(1, 0)] = C::new(s, 0.0);
    o[(1, 1)] = C::new(c, 0.0);
    Some(o)
}

fn try_reducing_masks(
    n: usize,
    dc_values: &[C; 4],
    lam_values: &[C; 4],
    target: &[C; 4],
    rows: u8,
    cols: u8,
    vals: u8,
) -> Option<Mat4> {
    let k = rows.count_ones() as usize;
    if k == 0 || k >= n || cols.count_ones() as usize != k || vals.count_ones() as usize != k {
        return None;
    }
    let all = (1u8 << n) - 1;
    let (rows_c, cols_c, vals_c) = (all ^ rows, all ^ cols, all ^ vals);
    let a_values = dc_values.map(|z| z * z);
    if (product_mask(&a_values, rows, n) * product_mask(lam_values, cols, n)
        - product_mask(target, vals, n))
    .norm()
        > 2e-8
    {
        return None;
    }
    if profile_enabled() {
        FACE_EQUALITIES[n].fetch_add(1, Ordering::Relaxed);
    }
    let left_dc = select_mask(dc_values, rows, n);
    let left_lam = select_mask(lam_values, cols, n);
    let left_w = select_mask(target, vals, n);
    let left = solve_rank_recursive(k, &left_dc, &left_lam, &left_w)?;
    let right_dc = select_mask(dc_values, rows_c, n);
    let right_lam = select_mask(lam_values, cols_c, n);
    let right_w = select_mask(target, vals_c, n);
    let right = solve_rank_recursive(n - k, &right_dc, &right_lam, &right_w)?;

    let mut o = Mat4::zeros();
    let mut ii = 0;
    for i in 0..n {
        if rows & (1 << i) == 0 {
            continue;
        }
        let mut jj = 0;
        for j in 0..n {
            if cols & (1 << j) != 0 {
                o[(i, j)] = left[(ii, jj)];
                jj += 1;
            }
        }
        ii += 1;
    }
    ii = 0;
    for i in 0..n {
        if rows_c & (1 << i) == 0 {
            continue;
        }
        let mut jj = 0;
        for j in 0..n {
            if cols_c & (1 << j) != 0 {
                o[(i, j)] = right[(ii, jj)];
                jj += 1;
            }
        }
        ii += 1;
    }
    for i in n..4 {
        o[(i, i)] = C::new(1.0, 0.0);
    }
    if o.determinant().re < 0.0 {
        for i in 0..n {
            o[(i, 0)] = -o[(i, 0)];
        }
    }
    let det = target[..n].iter().copied().product::<C>();
    let got = direct_coords(n, &a_values, lam_values, &o, det);
    let want = spectral_coords(&active_esym(target, n), n, det);
    if (got - want).rows(0, n - 1).amax() < 2e-7 {
        if profile_enabled() {
            FACE_HITS[n].fetch_add(1, Ordering::Relaxed);
        }
        Some(o)
    } else {
        None
    }
}

fn solve_rank_recursive(
    n: usize,
    dc_values: &[C; 4],
    lam_values: &[C; 4],
    target: &[C; 4],
) -> Option<Mat4> {
    if n == 1 {
        return Some(Mat4::identity());
    }
    if n == 2 {
        return solve_rank_two(dc_values, lam_values, target);
    }
    let horn3 = n != 3 || rank_three_horn_feasible(dc_values, lam_values, target);
    if !horn3 {
        return None;
    }

    // Any outer-wall realization has such a reducing subset.  Conversely, two
    // recursively feasible blocks explicitly prove reducibility, so this test
    // cannot misclassify an interior datum.
    // On a multiplicity stratum, ambient QLR facets can coalesce into
    // lower-dimensional outer faces.  A reducing subspace is still exactly a
    // triple of equal-cardinality spectral subsets with matching determinant.
    // Enumerating those labels only on the coalesced strata is complete and
    // avoids burdening the regular interior with irrelevant coincidences.
    let a_values = dc_values.map(|z| z * z);
    let repeated = has_repeated_value(&a_values, n)
        || has_repeated_value(lam_values, n)
        || has_repeated_value(target, n);
    if repeated {
        // Coalesced strata: determinant-labelled reducing subsets are the
        // confluent limits of QLR facets.  Enumerate them as stack-only masks.
        let all = (1u8 << n) - 1;
        for k in (1..=n / 2).rev() {
            for rows in 1..all {
                if rows.count_ones() as usize != k {
                    continue;
                }
                for cols in 1..all {
                    if cols.count_ones() as usize != k {
                        continue;
                    }
                    for vals in 1..all {
                        if vals.count_ones() as usize == k {
                            if let Some(o) = try_reducing_masks(
                                n, dc_values, lam_values, target, rows, cols, vals,
                            ) {
                                return Some(o);
                            }
                        }
                    }
                }
            }
        }
    } else if n == 4 {
        // Regular rank four: scan the exact 72 QLR facets, balanced 2+2
        // first because both children then close by the quadratic formula.
        for preferred_k in [2usize, 1usize] {
            for &(rank, mut rows, mut cols, mut vals) in &QLR4_FACES {
                if rank == 3 {
                    rows = 15 ^ rows;
                    cols = 15 ^ cols;
                    vals = 15 ^ vals;
                }
                rows = qlr_eig_mask(rows);
                cols = qlr_eig_mask(cols);
                vals = qlr_eig_mask(vals);
                if rows.count_ones() as usize == preferred_k {
                    if let Some(o) =
                        try_reducing_masks(n, dc_values, lam_values, target, rows, cols, vals)
                    {
                        return Some(o);
                    }
                }
            }
        }
    } else {
        // Rank three has 27 possible rank-one labels.  Recursive feasibility
        // of both children is the exact wall certificate.
        for rows in [1u8, 2, 4] {
            for cols in [1u8, 2, 4] {
                for vals in [1u8, 2, 4] {
                    if let Some(o) =
                        try_reducing_masks(n, dc_values, lam_values, target, rows, cols, vals)
                    {
                        return Some(o);
                    }
                }
            }
        }
    }
    path_lift_rank(n, dc_values, lam_values, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exterior_square_jacobian_matches_finite_differences() {
        let o = canonical_anchor_frame();
        let a: [C; 4] = std::array::from_fn(|i| C::from_polar(1.0, 0.17 * (i + 1) as f64));
        let b: [C; 4] = std::array::from_fn(|i| C::from_polar(1.0, -0.23 * (i + 1) as f64));
        let det = a.iter().copied().product::<C>() * b.iter().copied().product::<C>();
        let (_, j) = direct_coords_jacobian(4, &a, &b, &o, det, &super::super::PLANES);
        let h = 1e-6;
        for (k, &(p, q)) in super::super::PLANES.iter().enumerate() {
            let mut plus = o;
            let mut minus = o;
            rotate_columns(&mut plus, p, q, h, 4);
            rotate_columns(&mut minus, p, q, -h, 4);
            let fd = (direct_coords(4, &a, &b, &plus, det) - direct_coords(4, &a, &b, &minus, det))
                / (2.0 * h);
            for r in 0..3 {
                assert!(
                    (fd[r] - j[(r, k)]).abs() < 2e-8,
                    "r={r} k={k} fd={} j={}",
                    fd[r],
                    j[(r, k)]
                );
            }
        }
    }

    #[test]
    fn rational_rank_four_anchor_is_special_orthogonal() {
        let o = canonical_anchor_frame();
        assert!((o.transpose() * o - Mat4::identity()).norm() < 1e-14);
        assert!((o.determinant() - C::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn constructive_solver_closes_repeated_target_endpoint() {
        // feasible_linspace row 502800: simple inputs and a triple target
        // eigenvalue.  This exercises regular-limit continuation at a Weyl wall.
        let sol = solve_constructive(
            [0.375, 0.125, 0.0],
            [0.34375, 0.09375, -0.03125],
            [0.375, 0.375, -0.125],
        );
        assert_eq!(sol.rung, Rung::Constructive);
        assert!(sol.residual < ACCEPT, "residual={:.3e}", sol.residual);
    }

    #[test]
    fn total_entry_point_is_the_single_constructive_solver() {
        let args = (
            [0.28125, 0.09375, -0.03125],
            [0.375, 0.25, -0.125],
            [0.375, 0.375, -0.125],
        );
        let direct = solve_constructive(args.0, args.1, args.2);
        let total = solve_total(args.0, args.1, args.2);
        assert_eq!(direct.rung, Rung::Constructive);
        assert_eq!(total.rung, Rung::Constructive);
        assert!(direct.residual < ACCEPT && total.residual < ACCEPT);
    }
}
