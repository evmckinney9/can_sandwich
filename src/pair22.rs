//! Algebraic realization when one input has spectrum `2 + 2`.
//!
//! If the repeated factor is `(c,c,d,d)`, its orthogonal conjugates are
//! `c I + (d-c) P` for a real rank-two projector `P`.  The target
//! characteristic polynomial is affine-linear in the six squared Pluecker
//! coordinates of `im(P)`.  On the regular rank-four stratum those equations
//! leave an affine plane, and decomposability is one Heron quartic on it.
//!
//! This module deliberately handles only that proved rank-four stratum.  The
//! lower ranks are already complete support formulas: CS decomposition gives
//! a two-Givens face for `2+2` against `2+2`, one principal angle gives an
//! edge for `2+2` against `3+1`, and a scalar factor is a vertex.  It
//! exhausts all six coordinate walls and then samples every regular interior
//! component using the quartic discriminant.  Rank drops, polynomial content,
//! and identically repeated fibres decline to the older confluent constructor.
//! Every proposed frame receives both an explicit Gram check and the original
//! forward spectral certificate.

use super::{c, compound_residual, frame_metrics, poly_roots, Mat4, ACCEPT, C};

const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
type Affine = [f64; 3];
type Bi2 = [[f64; 3]; 3];
type Bi4 = [[f64; 5]; 5];

/// Try the two exact factor orientations.  Swapping the factors transposes the
/// realizing orthogonal frame.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve(
    prefix: &[C; 4],
    gate: &[C; 4],
    target_specs: &[[C; 4]; 2],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
    gate_is_pair22: bool,
    prefix_is_pair22: bool,
    target_is_pair22: [bool; 2],
    include_walls: bool,
    include_dense: bool,
) -> Option<(Mat4, f64)> {
    for (branch, target_spec) in target_specs.iter().enumerate() {
        let expected_det = prefix.iter().product::<C>() * gate.iter().product::<C>();
        if (expected_det - target_spec.iter().product::<C>()).norm() > 1e-8 {
            continue;
        }
        if gate_is_pair22 {
            let mut accept = |o: Mat4| certify(o, dc, lam, &targets[branch]);
            if let Some(hit) = solve_oriented(
                gate,
                prefix,
                target_spec,
                include_walls,
                include_dense,
                &mut accept,
            ) {
                return Some(hit);
            }
        }
        if prefix_is_pair22 {
            let mut accept = |o: Mat4| certify(o.transpose(), dc, lam, &targets[branch]);
            if let Some(hit) = solve_oriented(
                prefix,
                gate,
                target_spec,
                include_walls,
                include_dense,
                &mut accept,
            ) {
                return Some(hit);
            }
        }
        if target_is_pair22[branch] {
            let inverse_target: [C; 4] = target_spec.map(|value| value.conj());
            let inverse_prefix: [C; 4] = prefix.map(|value| value.conj());
            let inverse_gate: [C; 4] = gate.map(|value| value.conj());

            // D_g V W^-1 V^T D_g = S C^-1 S^T.  Inversion gives the
            // role-swapped original sandwich, hence O=S^T.
            let inverse_prefix_diagonal =
                Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&inverse_prefix));
            let mut accept = |v: Mat4| {
                let x = oriented_matrix(gate, &inverse_target, &v);
                let s = super::recover_frame(&x, &inverse_prefix_diagonal);
                certify(s.transpose(), dc, lam, &targets[branch])
            };
            if let Some(hit) = solve_oriented(
                &inverse_target,
                gate,
                &inverse_prefix,
                include_walls,
                include_dense,
                &mut accept,
            ) {
                return Some(hit);
            }

            // D_c V W^-1 V^T D_c = S G^-1 S^T gives O=S directly.
            let inverse_gate_diagonal =
                Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&inverse_gate));
            let mut accept = |v: Mat4| {
                let x = oriented_matrix(prefix, &inverse_target, &v);
                let s = super::recover_frame(&x, &inverse_gate_diagonal);
                certify(s, dc, lam, &targets[branch])
            };
            if let Some(hit) = solve_oriented(
                &inverse_target,
                prefix,
                &inverse_gate,
                include_walls,
                include_dense,
                &mut accept,
            ) {
                return Some(hit);
            }
        }
    }
    None
}

fn oriented_matrix(other: &[C; 4], repeated: &[C; 4], o: &Mat4) -> Mat4 {
    let roots: [C; 4] = other.map(|value| value.sqrt());
    let d = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&roots));
    let lambda = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(repeated));
    d * o * lambda * o.transpose() * d
}

fn certify(mut o: Mat4, dc: &Mat4, lam: &Mat4, target: &[C; 4]) -> Option<(Mat4, f64)> {
    let metrics = frame_metrics(&o)?;
    if !metrics.within(2e-10) {
        return None;
    }
    if metrics.determinant < 0.0 {
        for row in 0..4 {
            o[(row, 0)] = -o[(row, 0)];
        }
    }
    let residual = compound_residual(dc, lam, &o, target);
    (residual <= ACCEPT).then_some((o, residual))
}

fn pair22_groups(values: &[C; 4]) -> Option<(C, C, [usize; 2], [usize; 2])> {
    let same = |x: C, y: C| (x - y).norm_sqr() <= f64::EPSILON;
    let mut first = [usize::MAX; 2];
    let mut second = [usize::MAX; 2];
    let (mut nf, mut ns) = (0usize, 0usize);
    let anchor = values[0];
    for (index, &value) in values.iter().enumerate() {
        if same(value, anchor) {
            if nf >= 2 {
                return None;
            }
            first[nf] = index;
            nf += 1;
        } else {
            if ns >= 2 {
                return None;
            }
            second[ns] = index;
            ns += 1;
        }
    }
    if nf != 2 || ns != 2 || !same(values[second[0]], values[second[1]]) {
        return None;
    }
    let other = values[second[0]];
    ((other - anchor).norm() > 1e-14).then_some((anchor, other, first, second))
}

/// Solve with `repeated` in the gate position and `other=D^2` in the prefix
/// position.  The callback performs role-swap conversion and certification.
fn solve_oriented<R>(
    repeated: &[C; 4],
    other: &[C; 4],
    target: &[C; 4],
    include_walls: bool,
    include_dense: bool,
    accept: &mut impl FnMut(Mat4) -> Option<R>,
) -> Option<R> {
    let (anchor, peel, anchor_positions, peel_positions) = pair22_groups(repeated)?;
    let rho = peel - anchor;
    let delta: [C; 4] = std::array::from_fn(|i| anchor * other[i]);
    let chi_delta = polynomial_from_roots(&delta);
    let chi_target = polynomial_from_roots(target);

    let mut columns = [[C::new(0.0, 0.0); 4]; 6];
    for (edge, &(i, j)) in PAIRS.iter().enumerate() {
        let left = polynomial_excluding(&delta, i, usize::MAX);
        let right = polynomial_excluding(&delta, j, usize::MAX);
        let pair = polynomial_excluding(&delta, i, j);
        for degree in 0..4 {
            columns[edge][degree] = -rho * (other[i] * left[degree] + other[j] * right[degree])
                + rho * rho * other[i] * other[j] * pair[degree];
        }
    }
    let rhs: [C; 4] = std::array::from_fn(|degree| chi_target[degree] - chi_delta[degree]);
    let forms = affine_solution(&columns, &rhs)?;
    let products = complementary_products(&forms);
    let heron = heron_quartic(&products);

    // If the whole affine plane lies in a coordinate wall, Heron is exactly
    // the negative square of the difference of the two surviving products.
    // Work with that primitive conic, never with its repeated square.
    let contained_wall = (0..6).find(|&wall| {
        forms[wall][1].abs().max(forms[wall][2].abs()) <= 1e-12 && forms[wall][0].abs() <= 1e-10
    });
    let primitive_conic = contained_wall.map(|wall| {
        let (left, right) = complementary_survivors(wall);
        quadratic_difference(&products[left], &products[right])
    });
    if primitive_conic
        .as_ref()
        .is_some_and(|curve| bi2_scale(curve) <= 1e-12)
    {
        // The entire affine polygon lies in the Pluecker variety.
        return try_polygon_point(&forms, anchor_positions, peel_positions, accept);
    }

    // Every coordinate wall is a complete lower-dimensional peel.
    for wall in 0..if include_walls { 6 } else { 0 } {
        let form = forms[wall];
        let direction_scale = form[1].abs().max(form[2].abs());
        if direction_scale <= 1e-12 {
            continue;
        }
        let wall_curve = if let Some(curve) = primitive_conic.as_ref() {
            *curve
        } else {
            let (left, right) = complementary_survivors(wall);
            quadratic_difference(&products[left], &products[right])
        };
        let (poly, u_of_s, v_of_s) = restrict_quadratic(&wall_curve, form)?;
        let poly_scale = poly.iter().fold(0.0f64, |m, value| m.max(value.abs()));
        if poly_scale <= 1e-12 * bi2_scale(&wall_curve).max(1e-300) {
            if let Some((lo, hi)) = line_feasible_interval(&forms, u_of_s, v_of_s) {
                let s = 0.5 * (lo + hi);
                if let Some(hit) = try_candidate(
                    &forms,
                    u_of_s[0] + u_of_s[1] * s,
                    v_of_s[0] + v_of_s[1] * s,
                    anchor_positions,
                    peel_positions,
                    accept,
                ) {
                    return Some(hit);
                }
            }
            continue;
        }
        for s in real_unit_roots(&poly) {
            let u = u_of_s[0] + u_of_s[1] * s;
            let v = v_of_s[0] + v_of_s[1] * s;
            if let Some(hit) = try_candidate(&forms, u, v, anchor_positions, peel_positions, accept)
            {
                return Some(hit);
            }
        }
    }
    if !include_dense {
        return None;
    }

    // For a primitive curve, the roots
    // of its v-discriminant divide [0,1] into intervals on which the number of
    // real fibre points is constant.  Sampling each interval and its critical
    // endpoints is therefore complete on this declared regular stratum.
    let coefficients: Vec<Vec<f64>> = if let Some(curve) = primitive_conic.as_ref() {
        (0..3)
            .map(|v_degree| {
                let mut p: Vec<f64> = (0..3).map(|u_degree| curve[u_degree][v_degree]).collect();
                trim_polynomial(&mut p, 1e-14);
                p
            })
            .collect()
    } else {
        (0..5)
            .map(|v_degree| {
                let mut p: Vec<f64> = (0..=4 - v_degree)
                    .map(|u_degree| heron[u_degree][v_degree])
                    .collect();
                trim_polynomial(&mut p, 1e-14);
                p
            })
            .collect()
    };
    let curve_scale = coefficients
        .iter()
        .flatten()
        .fold(0.0f64, |maximum, value| maximum.max(value.abs()))
        .max(1e-300);
    if let Some(hit) = try_vertical_content(
        &coefficients,
        curve_scale,
        &forms,
        anchor_positions,
        peel_positions,
        accept,
    ) {
        return Some(hit);
    }
    let selector = if coefficients.len() == 3 {
        quadratic_discriminant(&coefficients)
    } else {
        let quartic: [Vec<f64>; 5] = std::array::from_fn(|degree| coefficients[degree].clone());
        super::one_plus_three::quartic_discriminant_poly(&quartic)?
    };
    let selector_scale = selector.iter().fold(0.0f64, |m, value| m.max(value.abs()));
    if selector_scale == 0.0 || !selector_scale.is_finite() {
        return None;
    }
    let mut critical: Vec<f64> = poly_roots(&selector)
        .into_iter()
        .filter(|root| root.im.abs() <= 2e-5 && (-1e-8..=1.0 + 1e-8).contains(&root.re))
        .map(|root| root.re.clamp(0.0, 1.0))
        .collect();
    critical.sort_by(f64::total_cmp);
    critical.dedup_by(|left, right| (*left - *right).abs() <= 1e-9);
    let mut cover = Vec::with_capacity(2 * critical.len() + 1);
    let mut previous = 0.0;
    for root in critical {
        if root > previous + 1e-12 {
            cover.push(0.5 * (previous + root));
        }
        cover.push(root);
        previous = root;
    }
    if previous < 1.0 - 1e-12 {
        cover.push(0.5 * (previous + 1.0));
    }
    cover.sort_by(|left, right| (left - 0.5).abs().total_cmp(&(right - 0.5).abs()));

    for u in cover {
        let mut fibre: Vec<f64> = coefficients
            .iter()
            .map(|coefficient| evaluate_polynomial(coefficient, u))
            .collect();
        trim_polynomial(&mut fibre, 1e-13);
        let fibre_scale = fibre.iter().fold(0.0f64, |m, value| m.max(value.abs()));
        if fibre_scale <= 2e-10 * curve_scale {
            if let Some((lo, hi)) = feasible_v_interval(&forms, u) {
                if let Some(hit) = try_candidate(
                    &forms,
                    u,
                    0.5 * (lo + hi),
                    anchor_positions,
                    peel_positions,
                    accept,
                ) {
                    return Some(hit);
                }
            }
            continue;
        }
        for v in real_unit_roots(&fibre) {
            if let Some(hit) = try_candidate(&forms, u, v, anchor_positions, peel_positions, accept)
            {
                return Some(hit);
            }
        }
    }
    None
}

fn polynomial_from_roots(roots: &[C; 4]) -> [C; 5] {
    let mut result = [C::new(0.0, 0.0); 5];
    result[0] = C::new(1.0, 0.0);
    let mut degree = 0usize;
    for &root in roots {
        for k in (0..=degree).rev() {
            result[k + 1] += result[k];
            result[k] *= -root;
        }
        degree += 1;
    }
    result
}

fn polynomial_excluding(roots: &[C; 4], skip0: usize, skip1: usize) -> [C; 4] {
    let mut result = [C::new(0.0, 0.0); 4];
    result[0] = C::new(1.0, 0.0);
    let mut degree = 0usize;
    for (index, &root) in roots.iter().enumerate() {
        if index == skip0 || index == skip1 {
            continue;
        }
        for k in (0..=degree).rev() {
            result[k + 1] += result[k];
            result[k] *= -root;
        }
        degree += 1;
    }
    result
}

/// Complete-pivot fraction-free-in-structure solve of the real coefficient
/// equations.  The returned coordinates are actual free Pluecker squares, so
/// `u=0` and `v=0` are genuine coordinate walls rather than arbitrary kernel
/// coordinates.
fn affine_solution(columns: &[[C; 4]; 6], rhs: &[C; 4]) -> Option<[Affine; 6]> {
    let mut original = [[0.0f64; 7]; 9];
    for degree in 0..4 {
        for edge in 0..6 {
            original[2 * degree][edge] = columns[edge][degree].re;
            original[2 * degree + 1][edge] = columns[edge][degree].im;
        }
        original[2 * degree][6] = rhs[degree].re;
        original[2 * degree + 1][6] = rhs[degree].im;
    }
    for edge in 0..6 {
        original[8][edge] = 1.0;
    }
    original[8][6] = 1.0;

    let mut matrix = original;
    let global_scale = original
        .iter()
        .flatten()
        .fold(0.0f64, |m, value| m.max(value.abs()))
        .max(1e-300);
    for row in &mut matrix {
        let scale = row[..6].iter().fold(0.0f64, |m, value| m.max(value.abs()));
        // Self-inversiveness makes some real/imaginary coefficient rows
        // identically zero.  Do not normalize their floating remnants into
        // fake independent equations.
        if scale <= 2e-12 * global_scale {
            if row[6].abs() > 2e-10 * global_scale {
                return None;
            }
            *row = [0.0; 7];
        } else {
            for value in row {
                *value /= scale;
            }
        }
    }
    let mut permutation = [0, 1, 2, 3, 4, 5];
    let mut rank = 0usize;
    while rank < 6 {
        let mut pivot = (rank, rank, 0.0f64);
        for row in rank..9 {
            for column in rank..6 {
                if matrix[row][column].abs() > pivot.2 {
                    pivot = (row, column, matrix[row][column].abs());
                }
            }
        }
        if pivot.2 <= 2e-11 {
            break;
        }
        matrix.swap(rank, pivot.0);
        for row in &mut matrix {
            row.swap(rank, pivot.1);
        }
        permutation.swap(rank, pivot.1);
        let value = matrix[rank][rank];
        for column in rank..7 {
            matrix[rank][column] /= value;
        }
        for row in 0..9 {
            if row == rank {
                continue;
            }
            let multiplier = matrix[row][rank];
            for column in rank..7 {
                matrix[row][column] -= multiplier * matrix[rank][column];
            }
        }
        rank += 1;
    }
    if rank != 4 {
        return None;
    }
    for row in rank..9 {
        let coefficient = matrix[row][..6]
            .iter()
            .fold(0.0f64, |m, value| m.max(value.abs()));
        if coefficient <= 2e-9 && matrix[row][6].abs() > 2e-8 {
            return None;
        }
    }

    let mut y = [[0.0f64; 3]; 6];
    y[4][1] = 1.0;
    y[5][2] = 1.0;
    for row in 0..4 {
        y[row][0] = matrix[row][6];
        y[row][1] = -matrix[row][4];
        y[row][2] = -matrix[row][5];
    }
    let mut forms = [[0.0f64; 3]; 6];
    for slot in 0..6 {
        forms[permutation[slot]] = y[slot];
    }

    // Reject an ill-conditioned nominal rank-four solve before it can pollute
    // the Heron coefficients.  The old exact-confluence path remains next.
    for coordinate in 0..3 {
        for row in &original {
            let lhs = (0..6)
                .map(|edge| row[edge] * forms[edge][coordinate])
                .sum::<f64>();
            let expected = if coordinate == 0 { row[6] } else { 0.0 };
            let scale = row[..6]
                .iter()
                .fold(row[6].abs(), |m, value| m.max(value.abs()))
                .max(1e-300);
            if scale <= 2e-12 * global_scale {
                continue;
            }
            if (lhs - expected).abs() > 2e-8 * scale {
                return None;
            }
        }
    }
    Some(forms)
}

fn affine_product(left: Affine, right: Affine) -> Bi2 {
    const DEGREE: [(usize, usize); 3] = [(0, 0), (1, 0), (0, 1)];
    let mut product = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            product[DEGREE[i].0 + DEGREE[j].0][DEGREE[i].1 + DEGREE[j].1] += left[i] * right[j];
        }
    }
    product
}

fn bi_product(left: &Bi2, right: &Bi2) -> Bi4 {
    let mut result = [[0.0; 5]; 5];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    result[i + k][j + l] += left[i][j] * right[k][l];
                }
            }
        }
    }
    result
}

fn add_scaled(out: &mut Bi4, value: &Bi4, scale: f64) {
    for i in 0..5 {
        for j in 0..5 {
            out[i][j] += scale * value[i][j];
        }
    }
}

fn complementary_products(forms: &[Affine; 6]) -> [Bi2; 3] {
    [
        affine_product(forms[0], forms[5]),
        affine_product(forms[1], forms[4]),
        affine_product(forms[2], forms[3]),
    ]
}

fn heron_quartic(products: &[Bi2; 3]) -> Bi4 {
    let mut result = [[0.0; 5]; 5];
    for i in 0..3 {
        add_scaled(&mut result, &bi_product(&products[i], &products[i]), -1.0);
        for j in i + 1..3 {
            add_scaled(&mut result, &bi_product(&products[i], &products[j]), 2.0);
        }
    }
    result
}

fn complementary_survivors(wall: usize) -> (usize, usize) {
    match wall {
        0 | 5 => (1, 2),
        1 | 4 => (0, 2),
        2 | 3 => (0, 1),
        _ => unreachable!("six Pluecker coordinates"),
    }
}

fn quadratic_difference(left: &Bi2, right: &Bi2) -> Bi2 {
    std::array::from_fn(|i| std::array::from_fn(|j| left[i][j] - right[i][j]))
}

fn bi2_scale(curve: &Bi2) -> f64 {
    curve
        .iter()
        .flatten()
        .fold(0.0f64, |maximum, value| maximum.max(value.abs()))
}

fn restrict_quadratic(curve: &Bi2, wall: Affine) -> Option<(Vec<f64>, [f64; 2], [f64; 2])> {
    let (u, v) = if wall[2].abs() >= wall[1].abs() && wall[2].abs() > 1e-14 {
        ([0.0, 1.0], [-wall[0] / wall[2], -wall[1] / wall[2]])
    } else if wall[1].abs() > 1e-14 {
        ([-wall[0] / wall[1], -wall[2] / wall[1]], [0.0, 1.0])
    } else {
        return None;
    };
    let mut result = vec![0.0; 5];
    for i in 0..3 {
        let up = affine_power(u, i);
        for j in 0..3 {
            let coefficient = curve[i][j];
            if coefficient == 0.0 {
                continue;
            }
            let vp = affine_power(v, j);
            for (a, &up_value) in up.iter().enumerate() {
                for (b, &vp_value) in vp.iter().enumerate() {
                    result[a + b] += coefficient * up_value * vp_value;
                }
            }
        }
    }
    trim_polynomial(&mut result, 1e-13);
    Some((result, u, v))
}

/// Intersect `lo <= s <= hi` with the affine halfspace `a + b s >= 0`.
fn intersect_halfspace(lo: &mut f64, hi: &mut f64, a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    if b.abs() <= 64.0 * f64::EPSILON * scale {
        return a >= -2e-12 * scale;
    }
    let root = -a / b;
    if b > 0.0 {
        *lo = lo.max(root);
    } else {
        *hi = hi.min(root);
    }
    *lo <= *hi + 2e-12
}

fn line_feasible_interval(forms: &[Affine; 6], u: [f64; 2], v: [f64; 2]) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for form in forms {
        let a = form[0] + form[1] * u[0] + form[2] * v[0];
        let b = form[1] * u[1] + form[2] * v[1];
        if !intersect_halfspace(&mut lo, &mut hi, a, b)
            || !intersect_halfspace(&mut lo, &mut hi, 1.0 - a, -b)
        {
            return None;
        }
    }
    Some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)))
}

fn feasible_v_interval(forms: &[Affine; 6], u: f64) -> Option<(f64, f64)> {
    line_feasible_interval(forms, [u, 0.0], [0.0, 1.0])
}

/// When Heron vanishes on the whole affine plane, any point of the bounded
/// Pluecker-square polygon is a witness.  Enumerating its affine boundary
/// intersections is a finite exact selection rule, not an optimization loop.
fn try_polygon_point<R>(
    forms: &[Affine; 6],
    anchor_positions: [usize; 2],
    peel_positions: [usize; 2],
    accept: &mut impl FnMut(Mat4) -> Option<R>,
) -> Option<R> {
    if let Some(hit) = try_candidate(forms, 0.5, 0.5, anchor_positions, peel_positions, accept) {
        return Some(hit);
    }

    let mut boundaries = Vec::with_capacity(16);
    boundaries.extend([[0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]);
    boundaries.extend([[0.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]);
    for &form in forms {
        boundaries.push(form);
        boundaries.push([form[0] - 1.0, form[1], form[2]]);
    }
    for i in 0..boundaries.len() {
        for j in i + 1..boundaries.len() {
            let left = boundaries[i];
            let right = boundaries[j];
            let determinant = left[1] * right[2] - left[2] * right[1];
            let scale = left[1]
                .abs()
                .max(left[2].abs())
                .max(right[1].abs())
                .max(right[2].abs())
                .max(1.0);
            if determinant.abs() <= 64.0 * f64::EPSILON * scale * scale {
                continue;
            }
            let u = (-left[0] * right[2] + left[2] * right[0]) / determinant;
            let v = (-left[1] * right[0] + left[0] * right[1]) / determinant;
            if let Some(hit) = try_candidate(forms, u, v, anchor_positions, peel_positions, accept)
            {
                return Some(hit);
            }
        }
    }
    None
}

fn polynomial_product(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mut product = vec![0.0; left.len() + right.len() - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            product[i + j] += a * b;
        }
    }
    product
}

fn add_polynomial_scaled(target: &mut Vec<f64>, value: &[f64], scale: f64) {
    if target.len() < value.len() {
        target.resize(value.len(), 0.0);
    }
    for (slot, &coefficient) in target.iter_mut().zip(value) {
        *slot += scale * coefficient;
    }
}

/// Discriminant in `v` of `c(u) + b(u)v + a(u)v^2`.
fn quadratic_discriminant(coefficients: &[Vec<f64>]) -> Vec<f64> {
    debug_assert_eq!(coefficients.len(), 3);
    let mut result = polynomial_product(&coefficients[1], &coefficients[1]);
    let ac = polynomial_product(&coefficients[0], &coefficients[2]);
    add_polynomial_scaled(&mut result, &ac, -4.0);
    trim_polynomial(&mut result, 1e-13);
    result
}

/// A common factor of all fibre coefficients is a vertical component.  It is
/// enough to root the lowest-degree nonzero coefficient and verify common
/// vanishing before choosing any feasible point on that vertical line.
fn try_vertical_content<R>(
    coefficients: &[Vec<f64>],
    curve_scale: f64,
    forms: &[Affine; 6],
    anchor_positions: [usize; 2],
    peel_positions: [usize; 2],
    accept: &mut impl FnMut(Mat4) -> Option<R>,
) -> Option<R> {
    let seed = coefficients
        .iter()
        .filter(|coefficient| {
            coefficient
                .iter()
                .fold(0.0f64, |m, value| m.max(value.abs()))
                > 1e-13 * curve_scale
        })
        .min_by_key(|coefficient| coefficient.len())?;
    for u in real_unit_roots(seed) {
        let common_residual = coefficients
            .iter()
            .map(|coefficient| evaluate_polynomial(coefficient, u).abs())
            .fold(0.0f64, f64::max);
        if common_residual > 2e-7 * curve_scale {
            continue;
        }
        let Some((lo, hi)) = feasible_v_interval(forms, u) else {
            continue;
        };
        if let Some(hit) = try_candidate(
            forms,
            u,
            0.5 * (lo + hi),
            anchor_positions,
            peel_positions,
            accept,
        ) {
            return Some(hit);
        }
    }
    None
}

fn affine_power(value: [f64; 2], exponent: usize) -> Vec<f64> {
    let mut result = vec![1.0];
    for _ in 0..exponent {
        let mut next = vec![0.0; result.len() + 1];
        for (degree, &coefficient) in result.iter().enumerate() {
            next[degree] += coefficient * value[0];
            next[degree + 1] += coefficient * value[1];
        }
        result = next;
    }
    result
}

fn trim_polynomial(polynomial: &mut Vec<f64>, tolerance: f64) {
    let scale = polynomial
        .iter()
        .fold(0.0f64, |m, value| m.max(value.abs()));
    while polynomial.len() > 1
        && polynomial
            .last()
            .is_some_and(|value| value.abs() <= tolerance * scale)
    {
        polynomial.pop();
    }
}

fn evaluate_polynomial(polynomial: &[f64], value: f64) -> f64 {
    polynomial
        .iter()
        .rev()
        .fold(0.0, |result, coefficient| result * value + coefficient)
}

fn real_unit_roots(polynomial: &[f64]) -> Vec<f64> {
    let scale = polynomial
        .iter()
        .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
    let tolerance = 64.0 * f64::EPSILON * scale;
    let candidates: Vec<C> = match polynomial.len() {
        0 | 1 => Vec::new(),
        2 if polynomial[1].abs() > tolerance => {
            vec![C::new(-polynomial[0] / polynomial[1], 0.0)]
        }
        3 if polynomial[2].abs() > tolerance => {
            let discriminant = polynomial[1] * polynomial[1] - 4.0 * polynomial[2] * polynomial[0];
            let discriminant_scale =
                polynomial[1] * polynomial[1] + (4.0 * polynomial[2] * polynomial[0]).abs();
            if discriminant < -128.0 * f64::EPSILON * discriminant_scale {
                Vec::new()
            } else {
                let root = discriminant.max(0.0).sqrt();
                let q = -0.5 * (polynomial[1] + if polynomial[1] < 0.0 { -root } else { root });
                if q == 0.0 {
                    vec![C::new(-polynomial[1] / (2.0 * polynomial[2]), 0.0)]
                } else {
                    vec![
                        C::new(q / polynomial[2], 0.0),
                        C::new(polynomial[0] / q, 0.0),
                    ]
                }
            }
        }
        _ => poly_roots(polynomial),
    };
    let mut roots: Vec<f64> = candidates
        .into_iter()
        .filter(|root| root.im.abs() <= 2e-6 && (-1e-8..=1.0 + 1e-8).contains(&root.re))
        .map(|root| root.re.clamp(0.0, 1.0))
        .collect();
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|left, right| (*left - *right).abs() <= 1e-9);
    roots
}

fn try_candidate<R>(
    forms: &[Affine; 6],
    u: f64,
    v: f64,
    anchor_positions: [usize; 2],
    peel_positions: [usize; 2],
    accept: &mut impl FnMut(Mat4) -> Option<R>,
) -> Option<R> {
    if !u.is_finite()
        || !v.is_finite()
        || !(-1e-7..=1.0 + 1e-7).contains(&u)
        || !(-1e-7..=1.0 + 1e-7).contains(&v)
    {
        return None;
    }
    let mut x = forms.map(|form| form[0] + form[1] * u + form[2] * v);
    if x.iter()
        .any(|value| !value.is_finite() || *value < -2e-7 || *value > 1.0 + 2e-7)
    {
        return None;
    }
    for value in &mut x {
        *value = value.max(0.0);
    }
    let sum = x.iter().sum::<f64>();
    if !sum.is_finite() || (sum - 1.0).abs() > 2e-7 || sum <= 0.0 {
        return None;
    }
    for value in &mut x {
        *value /= sum;
    }
    let frame = plucker_frame(x, anchor_positions, peel_positions)?;
    accept(frame)
}

/// Chart-free lift of nonnegative squared Pluecker coordinates.  Complementary
/// pair signs come from the Heron kernel; `P=-K^2` and `*K` then supply both
/// oriented repeated eigenspaces without a coordinate-chart enumeration.
fn plucker_frame(
    x: [f64; 6],
    anchor_positions: [usize; 2],
    peel_positions: [usize; 2],
) -> Option<Mat4> {
    let squared_products = [x[0] * x[5], x[1] * x[4], x[2] * x[3]];
    let pivot = (0..3).max_by(|&i, &j| squared_products[i].total_cmp(&squared_products[j]))?;
    let mut q = [0.0; 3];
    if squared_products[pivot] > 1e-28 {
        let j = (pivot + 1) % 3;
        let k = (pivot + 2) % 3;
        q[pivot] = squared_products[pivot].sqrt();
        q[j] = -(squared_products[pivot] + squared_products[j] - squared_products[k])
            / (2.0 * q[pivot]);
        q[k] = -q[pivot] - q[j];
    }
    let desired_products = [q[0], -q[1], q[2]];
    let complementary = [(0usize, 5usize), (1, 4), (2, 3)];
    let mut p = [0.0f64; 6];
    for (pair, &(left, right)) in complementary.iter().enumerate() {
        if x[left] >= x[right] {
            p[left] = x[left].sqrt();
            if p[left] > 0.0 {
                p[right] = desired_products[pair] / p[left];
            } else if desired_products[pair].abs() > 1e-12 {
                return None;
            }
        } else {
            p[right] = x[right].sqrt();
            if p[right] > 0.0 {
                p[left] = desired_products[pair] / p[right];
            } else if desired_products[pair].abs() > 1e-12 {
                return None;
            }
        }
    }
    let norm = p.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-14 {
        return None;
    }
    for value in &mut p {
        *value /= norm;
    }
    let plucker_error = (p[0] * p[5] - p[1] * p[4] + p[2] * p[3]).abs();
    if plucker_error > 2e-8 {
        return None;
    }

    let mut k = [[0.0f64; 4]; 4];
    for (edge, &(i, j)) in PAIRS.iter().enumerate() {
        k[i][j] = p[edge];
        k[j][i] = -p[edge];
    }
    let mut projector = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            projector[i][j] = -(0..4).map(|r| k[i][r] * k[r][j]).sum::<f64>();
        }
    }
    let image_pivot = (0..4).max_by(|&i, &j| projector[i][i].total_cmp(&projector[j][j]))?;
    let image_norm = projector[image_pivot][image_pivot].max(0.0).sqrt();
    if image_norm < 1e-12 {
        return None;
    }
    let image0 = std::array::from_fn(|row| projector[row][image_pivot] / image_norm);
    let image1 = std::array::from_fn(|row| -(0..4).map(|j| k[row][j] * image0[j]).sum::<f64>());

    let kernel_pivot =
        (0..4).max_by(|&i, &j| (1.0 - projector[i][i]).total_cmp(&(1.0 - projector[j][j])))?;
    let kernel_norm = (1.0 - projector[kernel_pivot][kernel_pivot])
        .max(0.0)
        .sqrt();
    if kernel_norm < 1e-12 {
        return None;
    }
    let kernel0 = std::array::from_fn(|row| {
        (f64::from(row == kernel_pivot) - projector[row][kernel_pivot]) / kernel_norm
    });
    // The Hodge-dual bivector rotates the orthogonal complement exactly as
    // K rotates the image plane. In PAIRS order 01,02,03,12,13,23 its upper
    // triangle is 23,-13,12,03,-02,01.
    let dual_coordinates = [p[5], -p[4], p[3], p[2], -p[1], p[0]];
    let mut dual = [[0.0f64; 4]; 4];
    for (edge, &(i, j)) in PAIRS.iter().enumerate() {
        dual[i][j] = dual_coordinates[edge];
        dual[j][i] = -dual_coordinates[edge];
    }
    let kernel1 =
        std::array::from_fn(|row| -(0..4).map(|j| dual[row][j] * kernel0[j]).sum::<f64>());

    let mut columns = [[0.0f64; 4]; 4];
    columns[peel_positions[0]] = image0;
    columns[peel_positions[1]] = image1;
    columns[anchor_positions[0]] = kernel0;
    columns[anchor_positions[1]] = kernel1;
    Some(Mat4::from_fn(|row, column| c(columns[column][row], 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_free_lift_handles_the_p12_coordinate_plane() {
        // The former row-0 Pluecker chart rejected this exact plane.
        let mut x = [0.0; 6];
        x[3] = 1.0; // p_12^2 = 1
        let frame = plucker_frame(x, [0, 1], [2, 3]).expect("coordinate plane");
        let real = frame.map(|value| value.re);
        let gram = real.transpose() * real;
        let residual = (gram - nalgebra::Matrix4::identity())
            .iter()
            .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
        assert!(residual < 1e-13);
        assert!((real.determinant().abs() - 1.0).abs() < 1e-13);

        let image = [2usize, 3usize];
        let recovered: [f64; 6] = std::array::from_fn(|edge| {
            let (i, j) = PAIRS[edge];
            let minor = real[(i, image[0])] * real[(j, image[1])]
                - real[(j, image[0])] * real[(i, image[1])];
            minor * minor
        });
        for edge in 0..6 {
            assert!((recovered[edge] - x[edge]).abs() < 1e-13);
        }
    }

    #[test]
    fn planted_pair22_target_is_recovered_by_the_plane_heron_solver() {
        let phase = |angle: f64| C::from_polar(1.0, angle);
        let other = [phase(0.17), phase(0.61), phase(-0.43), phase(-0.35)];
        let repeated = [phase(0.29), phase(0.29), phase(-0.29), phase(-0.29)];
        let planted = super::super::givens(0, 1, 0.31)
            * super::super::givens(1, 3, -0.47)
            * super::super::givens(0, 2, 0.22);
        let d = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(
            &other.map(|value| value.sqrt()),
        ));
        let lambda = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&repeated));
        let target_matrix = d * planted * lambda * planted.transpose() * d;
        let target = super::super::eig4(&target_matrix);
        let target_esym = super::super::esym4(target);
        let mut accept = |o: Mat4| {
            let residual = compound_residual(&d, &lambda, &o, &target_esym);
            (residual < 1e-9).then_some(residual)
        };
        let residual = solve_oriented(&repeated, &other, &target, true, true, &mut accept)
            .expect("regular Pair22 witness");
        assert!(residual < 1e-9);
    }
}
