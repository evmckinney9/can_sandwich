//! Algebraic Cauer reduction of the residual axis fibre.
//!
//! In one labelled chart the six off-diagonal mixed brackets close exactly on
//! `span(P(t),t(t-1))`.  Binary-quartic invariants therefore descend to one
//! degree-ten selector `T(u)`.  Each selected `u` lifts by one cubic Cauer
//! turnover, followed by linear recovery and the Heron-kernel frame.  The old
//! degree-30 eliminant is only the pullback of this tower and is not
//! constructed in production.  Loss of the cubic carrier is exactly a
//! repeated prefix phase and belongs to the earlier multiplicity dispatcher.
//!
//! A compact curve component has a pendant-coordinate extremum; components
//! meeting the positive boundary are owned by six zero-weight resultants.
//! No sampled section bank or high-degree eliminant remains.

use super::{
    compound_residual, frame_metrics, point_in_complex_hull, poly_roots, Mat4, ACCEPT, C, PERMS24,
};
use nalgebra::{Matrix3, SMatrix, Vector3};

/// `(axis row, pinned gate pair, pendant gate label, reference gate label)`.
pub(super) type Chart = (usize, (usize, usize), usize, usize);
const COLUMN_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
pub(super) const ORBIT_LEN: usize = 24;

#[inline]
pub(super) fn orbit_chart(index: usize) -> Chart {
    let axis = index / COLUMN_PAIRS.len();
    let pair = COLUMN_PAIRS[index % COLUMN_PAIRS.len()];
    let mut rest = [usize::MAX; 2];
    let mut nrest = 0;
    for column in 0..4 {
        if column != pair.0 && column != pair.1 {
            rest[nrest] = column;
            nrest += 1;
        }
    }
    (axis, pair, rest[0], rest[1])
}

fn conditioned_charts(delta: &[C; 4], lam: &[C; 4]) -> [(Chart, f64); ORBIT_LEN] {
    // The chart denominator bound is the minimum of one row-complement area
    // and two column-pair gaps.  There are only four distinct areas and six
    // distinct pair/complement scores in the 24-label orbit.  Compute those
    // ten geometric invariants once; a comparison sort must not rebuild a
    // 3x3 determinant O(24 log 24) times.
    let areas: [f64; 4] = std::array::from_fn(|axis| {
        let mut core = [C::new(0.0, 0.0); 3];
        let mut count = 0;
        for (index, &value) in delta.iter().enumerate() {
            if index != axis {
                core[count] = value;
                count += 1;
            }
        }
        Matrix3::new(
            1.0, 1.0, 1.0, core[0].re, core[1].re, core[2].re, core[0].im, core[1].im, core[2].im,
        )
        .determinant()
        .abs()
    });
    let pair_scores: [f64; 6] = std::array::from_fn(|pair_index| {
        let pair = COLUMN_PAIRS[pair_index];
        let mut rest = [usize::MAX; 2];
        let mut count = 0;
        for column in 0..4 {
            if column != pair.0 && column != pair.1 {
                rest[count] = column;
                count += 1;
            }
        }
        let pinned_gap = 0.5 * (lam[pair.0] - lam[pair.1]).norm();
        let pendant_gap = (lam[rest[0]] - lam[rest[1]]).norm();
        pinned_gap.min(pendant_gap)
    });
    std::array::from_fn(|index| {
        let chart = orbit_chart(index);
        (chart, areas[chart.0].min(pair_scores[index % 6]))
    })
}

#[inline]
fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[inline]
fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

/// Outward-only separation from the convex hull of twelve points in R^3.
///
/// Every facet plane contains at least three vertices. We enumerate those
/// triples and reject only when all twelve vertices lie on one side while the
/// query lies beyond the opposite side by a scale-aware margin. Degenerate
/// triples and numerically marginal planes abstain, so this can lose pruning
/// but cannot turn a near-boundary realization into a decline.
fn separated_from_joint_hull<const N: usize>(vertices: &[[f64; 3]; N], point: [f64; 3]) -> bool {
    let coordinate_scale = vertices
        .iter()
        .flat_map(|vertex| vertex.iter())
        .chain(point.iter())
        .fold(1.0f64, |scale, value| scale.max(value.abs()));
    for i in 0..N.saturating_sub(2) {
        for j in i + 1..N.saturating_sub(1) {
            for k in j + 1..N {
                let p = vertices[i];
                let ab = std::array::from_fn(|d| vertices[j][d] - p[d]);
                let ac = std::array::from_fn(|d| vertices[k][d] - p[d]);
                let normal = cross3(ab, ac);
                let normal_scale = dot3(normal, normal).sqrt();
                if normal_scale <= 1e-12 * coordinate_scale * coordinate_scale {
                    continue;
                }
                // ACCEPT is 1e-9 in these same characteristic coordinates.
                // The factor 16 also absorbs construction and plane-rounding
                // error before a support exclusion becomes authoritative.
                let tolerance = 16.0 * ACCEPT * normal_scale * coordinate_scale;
                let mut minimum = f64::INFINITY;
                let mut maximum = f64::NEG_INFINITY;
                for vertex in vertices {
                    let side = dot3(normal, std::array::from_fn(|d| vertex[d] - p[d]));
                    minimum = minimum.min(side);
                    maximum = maximum.max(side);
                }
                let query = dot3(normal, std::array::from_fn(|d| point[d] - p[d]));
                if (minimum >= -tolerance && query < -tolerance)
                    || (maximum <= tolerance && query > tolerance)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Necessary joint fundamental-character support for one labelled axis chart.
///
/// The chart imposes `O[axis,pair.0]=O[axis,pair.1]=0`.  Therefore the
/// orthostochastic matrix `B=O⊙O` lies in the corresponding face of the
/// Birkhoff polytope.  That face has exactly the twelve permutation vertices
/// whose axis row avoids the forbidden pair.
///
/// More strongly, after moving the axis row and forbidden columns to
/// `(0,{0,1})`, every chart frame is parameterized by `x=cos(theta)^2` and one
/// `Q in O(3)`. Both `e1` and `e2` are affine in the same doubly-stochastic
/// matrix `Q⊙Q`; for fixed `x`, Birkhoff therefore expresses their joint
/// value as a convex combination of six permutation values. Each such value
/// is affine in `x`, hence a combination of its two endpoints. Those twelve
/// endpoints are precisely the permitted four-dimensional permutations.
fn character_supported(
    delta: &[C; 4],
    lam: &[C; 4],
    target: &[C; 4],
    chart: Chart,
    include_second_character: bool,
) -> bool {
    let (axis, pair, _, _) = chart;
    let mut traces = [C::default(); 12];
    let mut vertices = [[0.0; 3]; 12];
    let mut count = 0usize;
    for permutation in PERMS24.iter() {
        if permutation[axis] == pair.0 || permutation[axis] == pair.1 {
            continue;
        }
        let roots: [C; 4] = std::array::from_fn(|row| delta[row] * lam[permutation[row]]);
        let e1: C = roots.iter().copied().sum();
        let e2 = roots[0] * roots[1]
            + roots[0] * roots[2]
            + roots[0] * roots[3]
            + roots[1] * roots[2]
            + roots[1] * roots[3]
            + roots[2] * roots[3];
        traces[count] = e1;
        vertices[count] = [e1.re, e1.im, e2.re];
        count += 1;
    }
    debug_assert_eq!(count, vertices.len());
    point_in_complex_hull(&traces, target[0])
        && (!include_second_character
            || !separated_from_joint_hull(&vertices, [target[0].re, target[0].im, target[1].re]))
}

/// Necessary joint fundamental-character support for one zero-weight boundary.
///
/// Besides the chart zeros in the axis row, `alpha_i=0` forces
/// `O[core[i],pair.0]=0`, while `beta_i=0` forces
/// `O[core[i],pair.1]=0`.  The corresponding face of the Birkhoff polytope
/// has eight permutation vertices.  Since the target trace is linear in
/// `O⊙O`, exclusion from their planar hull proves that this entire
/// zero-weight family is empty before its sextic resultant is constructed.
fn zero_weight_character_supported(
    delta: &[C; 4],
    lam: &[C; 4],
    target: &[C; 4],
    chart: Chart,
    zero_row: usize,
    zero_column: usize,
    include_second_character: bool,
) -> bool {
    let (axis, pair, _, _) = chart;
    let mut traces = [C::default(); 8];
    let mut vertices = [[0.0; 3]; 8];
    let mut count = 0usize;
    for permutation in PERMS24.iter() {
        if permutation[axis] == pair.0
            || permutation[axis] == pair.1
            || permutation[zero_row] == zero_column
        {
            continue;
        }
        let roots: [C; 4] = std::array::from_fn(|row| delta[row] * lam[permutation[row]]);
        let e1: C = roots.iter().copied().sum();
        let e2 = roots[0] * roots[1]
            + roots[0] * roots[2]
            + roots[0] * roots[3]
            + roots[1] * roots[2]
            + roots[1] * roots[3]
            + roots[2] * roots[3];
        traces[count] = e1;
        vertices[count] = [e1.re, e1.im, e2.re];
        count += 1;
    }
    debug_assert_eq!(count, vertices.len());
    point_in_complex_hull(&traces, target[0])
        && (!include_second_character
            || !separated_from_joint_hull(&vertices, [target[0].re, target[0].im, target[1].re]))
}

#[derive(Clone)]
pub(super) struct AxisLine {
    core: [usize; 3],
    axis: usize,
    pair: (usize, usize),
    pendant_label: usize,
    reference_label: usize,
    pendant: f64,
    /// Affine chain in tau (every recover intermediate is affine): alpha,
    /// beta as value-at-0 + tau-derivative, and the consistency remainder.
    a0: [f64; 3],
    ad: [f64; 3],
    b0: [f64; 3],
    bd: [f64; 3],
    rem0: C,
    remd: C,
}

struct Recovered {
    heron: f64,
    alpha: [f64; 3],
    beta: [f64; 3],
}

#[inline]
fn peval<const N: usize>(p: &[C; N], z: C) -> C {
    p.iter().fold(C::new(0.0, 0.0), |a, &x| a * z + x)
}

pub(super) fn target_poly(et: &[f64; 4]) -> [C; 5] {
    let w: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * et[k]));
    let e1 = w.iter().copied().sum::<C>();
    let e2 = w[0] * w[1] + w[0] * w[2] + w[0] * w[3] + w[1] * w[2] + w[1] * w[3] + w[2] * w[3];
    let e3 = w[0] * w[1] * w[2] + w[0] * w[1] * w[3] + w[0] * w[2] * w[3] + w[1] * w[2] * w[3];
    [C::new(1.0, 0.0), -e1, e2, -e3, w.iter().product()]
}

fn axis_line(
    delta: [C; 4],
    lam: [C; 4],
    chi: [C; 5],
    chart: (usize, (usize, usize), usize, usize),
    pendant: f64,
) -> Option<AxisLine> {
    if !(0.0 < pendant && pendant < 1.0) {
        return None;
    }
    let (axis, pair, pendant_label, reference_label) = chart;
    let mut cv = [0usize; 3];
    let mut n = 0;
    for k in 0..4 {
        if k != axis {
            cv[n] = k;
            n += 1;
        }
    }
    let dc = [delta[cv[0]], delta[cv[1]], delta[cv[2]]];
    let (la, lb, lc, lr) = (
        lam[pair.0],
        lam[pair.1],
        lam[pendant_label],
        lam[reference_label],
    );
    let aa = (la + lb) / 2.0;
    let bb = (la - lb) / 2.0;
    if bb.norm() < 1e-12 {
        return None;
    }
    let product = dc.iter().product::<C>() * la * lb * lr;
    let mu = lr * delta[axis];
    let rho = lc - lr;
    if rho.norm() < 1e-12 {
        return None;
    }

    // q_s(mu)=desired is one real affine equation in s=x+iy.  The complex
    // 2x2 representation has rank one by self-inversive reciprocity.
    let desired = -peval(&chi, mu) / (rho * delta[axis] * pendant);
    let constant = mu * mu * mu - product;
    let coeff_x = -mu * mu + product * mu;
    let coeff_y = C::new(0.0, -1.0) * (mu * mu + product * mu);
    let rhs = desired - constant;
    let rows = [
        ([coeff_x.re, coeff_y.re], rhs.re),
        ([coeff_x.im, coeff_y.im], rhs.im),
    ];
    let norms = [
        rows[0].0[0].hypot(rows[0].0[1]),
        rows[1].0[0].hypot(rows[1].0[1]),
    ];
    let ir = usize::from(norms[1] > norms[0]);
    if norms[ir] < 1e-12 {
        return None;
    }
    let normal = rows[ir].0;
    let nn = normal[0] * normal[0] + normal[1] * normal[1];
    let point = [normal[0] * rows[ir].1 / nn, normal[1] * rows[ir].1 / nn];
    let direction = [-normal[1] / nn.sqrt(), normal[0] / nn.sqrt()];
    for (row, b) in rows {
        if (row[0] * point[0] + row[1] * point[1] - b).abs() > 2e-7 {
            return None;
        }
    }

    let tdelta = Matrix3::new(
        1.0, 1.0, 1.0, dc[0].re, dc[1].re, dc[2].re, dc[0].im, dc[1].im, dc[2].im,
    );
    if tdelta.determinant().abs() < 1e-10 {
        return None;
    }
    // The whole recover chain is affine in tau: compose it once at
    // (value, derivative) instead of re-solving per pendant root.
    let one = C::new(1.0, 0.0);
    let zero = C::new(0.0, 0.0);
    let s0 = C::new(point[0], point[1]);
    let sd = C::new(direction[0], direction[1]);
    let q0v = [one, -s0, product * s0.conj(), -product];
    let qdv = [zero, -sd, product * sd.conj(), zero];
    let c2_0 = [
        q0v[0],
        q0v[1] - mu * q0v[0],
        q0v[2] - mu * q0v[1],
        q0v[3] - mu * q0v[2],
        -mu * q0v[3],
    ];
    let c2_d = [
        qdv[0],
        qdv[1] - mu * qdv[0],
        qdv[2] - mu * qdv[1],
        qdv[3] - mu * qdv[2],
        -mu * qdv[3],
    ];
    let qc_0: [C; 4] = std::array::from_fn(|k| (c2_0[k + 1] - chi[k + 1]) / rho);
    let qc_d: [C; 4] = std::array::from_fn(|k| c2_d[k + 1] / rho);
    let da = delta[axis];
    let num_0: [C; 4] = std::array::from_fn(|k| qc_0[k] - pendant * da * q0v[k]);
    let num_d: [C; 4] = std::array::from_fn(|k| qc_d[k] - pendant * da * qdv[k]);
    let omt = 1.0 - pendant;
    let qn2_0 = num_0[0] / omt;
    let qn2_d = num_d[0] / omt;
    let rem0 = num_0[3] + mu * (num_0[2] + mu * (num_0[1] + mu * num_0[0]));
    let remd = num_d[3] + mu * (num_d[2] + mu * (num_d[1] + mu * num_d[0]));
    let lu = tdelta.lu();
    let u0 = lu.solve(&Vector3::new(1.0, qn2_0.re, qn2_0.im))?;
    let ud = lu.solve(&Vector3::new(0.0, qn2_d.re, qn2_d.im))?;
    let sum_dc = dc[0] + dc[1] + dc[2];
    let vd0 = (s0 - aa * sum_dc - (lr - aa) * qn2_0) / bb;
    let vdd = (sd - (lr - aa) * qn2_d) / bb;
    let v0 = lu.solve(&Vector3::new(0.0, vd0.re, vd0.im))?;
    let vd_ = lu.solve(&Vector3::new(0.0, vdd.re, vdd.im))?;
    let a0: [f64; 3] = std::array::from_fn(|k| (1.0 - u0[k] + v0[k]) / 2.0);
    let ad: [f64; 3] = std::array::from_fn(|k| (-ud[k] + vd_[k]) / 2.0);
    let b0: [f64; 3] = std::array::from_fn(|k| (1.0 - u0[k] - v0[k]) / 2.0);
    let bd: [f64; 3] = std::array::from_fn(|k| (-ud[k] - vd_[k]) / 2.0);
    Some(AxisLine {
        core: cv,
        axis,
        pair,
        pendant_label,
        reference_label,
        pendant,
        a0,
        ad,
        b0,
        bd,
        rem0,
        remd,
    })
}

impl AxisLine {
    /// Interval where all six squared coordinates can be nonnegative.
    /// Every realizable root must lie here, so an empty intersection is an
    /// exact decline certificate before constructing or rooting the Heron
    /// quartic.  The slack matches `frames`' existing boundary convention.
    fn positive_tau_interval(&self) -> Option<(f64, f64)> {
        const SLACK: f64 = 2e-6;
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        for (&value, &slope) in self
            .a0
            .iter()
            .zip(&self.ad)
            .chain(self.b0.iter().zip(&self.bd))
        {
            if slope > 0.0 {
                lo = lo.max((-SLACK - value) / slope);
            } else if slope < 0.0 {
                hi = hi.min((-SLACK - value) / slope);
            } else if value < -SLACK {
                return None;
            }
        }
        (lo <= hi).then_some((lo, hi))
    }

    fn recover(&self, tau: f64) -> Option<Recovered> {
        let rem = self.rem0 + self.remd * tau;
        if rem.norm() > 2e-6 {
            return None;
        }
        let alpha: [f64; 3] = std::array::from_fn(|k| self.a0[k] + self.ad[k] * tau);
        let beta: [f64; 3] = std::array::from_fn(|k| self.b0[k] + self.bd[k] * tau);
        let x: [f64; 3] = std::array::from_fn(|k| alpha[k] * beta[k]);
        let sx = x[0] + x[1] + x[2];
        let heron = sx * sx - 2.0 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        if !heron.is_finite() {
            return None;
        }
        Some(Recovered { heron, alpha, beta })
    }

    fn frame(&self, r: &Recovered) -> Option<Mat4> {
        if r.alpha.iter().chain(r.beta.iter()).any(|&x| x < -2e-6) {
            return None;
        }
        let Some((a, b)) = heron_kernel_columns(r.alpha, r.beta) else {
            return None;
        };
        let n = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let cp = self.pendant.sqrt();
        let sp = (1.0 - self.pendant).sqrt();
        let mut o = Mat4::zeros();
        for k in 0..3 {
            o[(self.core[k], self.pair.0)] = C::new(a[k], 0.0);
            o[(self.core[k], self.pair.1)] = C::new(b[k], 0.0);
            o[(self.core[k], self.pendant_label)] = C::new(sp * n[k], 0.0);
            o[(self.core[k], self.reference_label)] = C::new(cp * n[k], 0.0);
        }
        o[(self.axis, self.pendant_label)] = C::new(cp, 0.0);
        o[(self.axis, self.reference_label)] = C::new(-sp, 0.0);
        let metrics = frame_metrics(&o)?;
        if !metrics.within(1e-11) {
            return None;
        }
        if metrics.determinant < 0.0 {
            for row in 0..4 {
                o[(row, self.reference_label)] = -o[(row, self.reference_label)];
            }
        }
        Some(o)
    }
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
pub(super) fn heron_kernel_columns(
    alpha_raw: [f64; 3],
    beta_raw: [f64; 3],
) -> Option<([f64; 3], [f64; 3])> {
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

/// All four roots of a real quartic in radicals (Ferrari / resolvent
/// cubic) -- the section construction's own algebra, replacing a general
/// companion eigensolve per section. Complex pairs are returned (the
/// caller's small-imaginary filter is semantic). Returns None on a
/// degenerate leading coefficient or non-finite intermediates; the
/// caller falls back to the eigensolve rooter, so completeness never
/// depends on this path.
pub(super) fn quartic_roots(q: &[f64; 5], out: &mut [C; 4]) -> Option<usize> {
    let scale = q.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if !(scale.is_finite()) || scale == 0.0 || q[4].abs() < 1e-12 * scale {
        return None;
    }
    let (p3, p2, p1, p0) = (q[3] / q[4], q[2] / q[4], q[1] / q[4], q[0] / q[4]);
    // depressed quartic y^4 + p y^2 + qq y + r, x = y - p3/4
    let sh = p3 / 4.0;
    let p = p2 - 3.0 * p3 * p3 / 8.0;
    let qq = p1 - p3 * p2 / 2.0 + p3 * p3 * p3 / 8.0;
    let r = p0 - p3 * p1 / 4.0 + p3 * p3 * p2 / 16.0 - 3.0 * p3.powi(4) / 256.0;
    if !(p.is_finite() && qq.is_finite() && r.is_finite()) {
        return None;
    }
    // resolvent cubic z^3 + 2p z^2 + (p^2-4r) z - qq^2: value at 0 is
    // -qq^2 <= 0, so its largest real root is >= 0.
    let (ca, cb, cc) = (2.0 * p, p * p - 4.0 * r, -qq * qq);
    let z0 = {
        // depress t = z - ca/3: t^3 + cp t + cq
        let cp = cb - ca * ca / 3.0;
        let cq = 2.0 * ca.powi(3) / 27.0 - ca * cb / 3.0 + cc;
        let disc = -4.0 * cp.powi(3) - 27.0 * cq * cq;
        let t = if disc >= 0.0 && cp < 0.0 {
            // three real roots: take the largest (k = 0 branch)
            let m = 2.0 * (-cp / 3.0).sqrt();
            let arg = (3.0 * cq / (cp * m)).clamp(-1.0, 1.0);
            m * (arg.acos() / 3.0).cos()
        } else {
            // one real root (Cardano)
            let s = (cq * cq / 4.0 + cp.powi(3) / 27.0).max(0.0).sqrt();
            let u = (-cq / 2.0 + s).cbrt();
            let v = (-cq / 2.0 - s).cbrt();
            u + v
        };
        (t - ca / 3.0).max(0.0)
    };
    let s = z0.sqrt();
    // factor y^4 + p y^2 + qq y + r = (y^2 + s y + g0)(y^2 - s y + h0)
    let (g0, h0) = if s > 1e-150 * (1.0 + p.abs()).sqrt() {
        ((p + z0 - qq / s) / 2.0, (p + z0 + qq / s) / 2.0)
    } else {
        // biquadratic: y^2 = (-p +- sqrt(p^2 - 4r))/2
        let d = p * p - 4.0 * r;
        let sq = C::new(d.max(0.0).sqrt(), (-d).max(0.0).sqrt());
        let y2a = (C::new(-p, 0.0) + sq) * 0.5;
        let y2b = (C::new(-p, 0.0) - sq) * 0.5;
        let mut n = 0;
        for y2 in [y2a, y2b] {
            let y = y2.sqrt();
            out[n] = y - C::new(sh, 0.0);
            out[n + 1] = -y - C::new(sh, 0.0);
            n += 2;
        }
        return Some(4);
    };
    if !(g0.is_finite() && h0.is_finite()) {
        return None;
    }
    let mut n = 0;
    for (b, c0) in [(s, g0), (-s, h0)] {
        let d = b * b - 4.0 * c0;
        if d >= 0.0 {
            // cancellation-free real pair
            let sd = d.sqrt();
            let t1 = if b >= 0.0 {
                (-b - sd) / 2.0
            } else {
                (-b + sd) / 2.0
            };
            let (r1, r2) = if t1.abs() > 0.0 {
                (t1, c0 / t1)
            } else {
                (0.0, -b)
            };
            out[n] = C::new(r1 - sh, 0.0);
            out[n + 1] = C::new(r2 - sh, 0.0);
        } else {
            let im = (-d).sqrt() / 2.0;
            out[n] = C::new(-b / 2.0 - sh, im);
            out[n + 1] = C::new(-b / 2.0 - sh, -im);
        }
        n += 2;
    }
    Some(n)
}

/// Re-root one quartic with the exact companion rooter and return the
/// root nearest `tau`: the radical sweep locates candidates (its error is
/// three orders below every admissibility filter), the eigensolve grade
/// is restored on the rare accepted root. One bounded call, no iteration.
fn exact_nearest_tau(q: &[f64; 5], tau: f64) -> f64 {
    poly_roots(q)
        .into_iter()
        .filter(|z| z.im.abs() < 3e-3)
        .map(|z| z.re)
        .min_by(|a, b| (a - tau).abs().total_cmp(&(b - tau).abs()))
        .unwrap_or(tau)
}

fn quartic_coefficients(line: &AxisLine, cleared: bool) -> Option<[f64; 5]> {
    // Exact composition: x_k(tau) = alpha_k beta_k is quadratic, and
    // heron = (sum x)^2 - 2 sum x^2 is the quartic -- no sampling, no
    // interpolation, one LU already done at line construction.
    let scale = if cleared {
        (line.pendant * (1.0 - line.pendant)).powi(4)
    } else {
        1.0
    };
    let mut ssum = [0.0f64; 3];
    let mut qsum = [0.0f64; 5];
    for k in 0..3 {
        let (a0, ad, b0, bd) = (line.a0[k], line.ad[k], line.b0[k], line.bd[k]);
        let xq = [a0 * b0, a0 * bd + ad * b0, ad * bd];
        for i in 0..3 {
            ssum[i] += xq[i];
        }
        for i in 0..3 {
            for j in 0..3 {
                qsum[i + j] += xq[i] * xq[j];
            }
        }
    }
    let mut f = [0.0f64; 5];
    for i in 0..3 {
        for j in 0..3 {
            f[i + j] += ssum[i] * ssum[j];
        }
    }
    for i in 0..5 {
        f[i] = (f[i] - 2.0 * qsum[i]) * scale;
        if !f[i].is_finite() {
            return None;
        }
    }
    Some(f)
}

/// Realize one pendant value selected by the Cauer critical cover: solve its
/// quartic exactly, reconstruct the Heron-kernel frame, and forward-certify.
#[allow(clippy::too_many_arguments)]
fn realize_pendant(
    delta: [C; 4],
    lam: [C; 4],
    chi: [C; 5],
    chart: (usize, (usize, usize), usize, usize),
    prep: Option<&ChartPrep>,
    pendant: f64,
    tau_tol: f64,
    dc: &Mat4,
    lam_m: &Mat4,
    target: &[C; 4],
) -> Option<(Mat4, f64)> {
    let line = if let (Some(p), true) = (prep, ChartPrep::in_range(pendant)) {
        p.line_at(pendant)
    } else {
        axis_line(delta, lam, chi, chart, pendant)
    };
    let line = line?;
    let (tau_lo, tau_hi) = line.positive_tau_interval()?;
    let q = quartic_coefficients(&line, false)?;
    let mut qbuf = [C::default(); 4];
    let try_roots = |taus: &[C]| {
        for &tau in taus {
            if tau.im.abs() > tau_tol || tau.re < tau_lo || tau.re > tau_hi {
                continue;
            }
            let Some(r) = line.recover(tau.re) else {
                continue;
            };
            if r.heron.abs() > 2e-5 {
                continue;
            }
            let Some(o) = line.frame(&r) else {
                continue;
            };
            if compound_residual(dc, lam_m, &o, target) >= 1e-2 {
                continue;
            }
            // Restore eigensolve-grade tau before the residual verdict. This
            // is one bounded algebraic solve, never an iterative correction.
            let tex = exact_nearest_tau(&q, tau.re);
            let r = line.recover(tex).unwrap_or(r);
            let Some(o) = line.frame(&r) else {
                continue;
            };
            let residual = compound_residual(dc, lam_m, &o, target);
            if residual < ACCEPT {
                return Some((o, residual));
            }
        }
        None
    };
    if let Some(n) = quartic_roots(&q, &mut qbuf) {
        if let Some(hit) = try_roots(&qbuf[..n]) {
            return Some(hit);
        }
        // If the radical chart returned finite but inaccurate roots, enumerate
        // the same quartic once with the bounded companion kernel.  This does
        // not change the selected pendant or introduce an iterative solve.
        if tau_tol >= 1e-3 {
            return try_roots(&poly_roots(&q));
        }
        return None;
    }
    try_roots(&poly_roots(&q))
}

/// One (branch, orientation) attempt context of the axis tier.
pub(super) struct Combo {
    pub(super) delta: [C; 4],
    pub(super) lam: [C; 4],
    pub(super) chi: [C; 5],
    pub(super) dc: Mat4,
    pub(super) lam_m: Mat4,
    pub(super) target: [C; 4],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum AxisPass {
    /// Bounded degree-ten Cauer candidates from critical pendant values.
    CriticalFast,
    /// Certified isolation of the same selector and its exact boundaries.
    CriticalCertified,
}

/// Evaluate the exact critical projection of the labelled axis fibre across
/// both target branches and both factor orientations. Every label evaluates
/// the same six-bracket/Cauer formula; labels differ only by the finite
/// row/zero-pair action.
///
/// No fixed pendant samples occur here.  A compact component projects to an
/// interval whose boundary consists of critical or zero-weight values, so the
/// selector roots and the interval cover in `try_discriminant_cover` own
/// completeness.  `accept` applies the caller's orientation fix-up and final
/// spectral certificate; rejection resumes the exact enumeration.
pub(super) fn solve_all<R>(
    combos: &[Combo],
    pass: AxisPass,
    mut accept: impl FnMut(usize, Mat4, f64, bool) -> Option<R>,
) -> Option<R> {
    debug_assert!(matches!(combos.len(), 2 | 4));
    // Order the two factor orientations by their closest spectral gap.
    // This is a permutation of the same exact enumeration.
    let minimum_gap = |z: &[C; 4]| -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..4 {
            for j in i + 1..4 {
                let d = (z[i] * z[j].conj()).arg().abs();
                best = best.min(d);
            }
        }
        best
    };
    let gd = minimum_gap(&combos[0].delta);
    let gs = minimum_gap(&combos[1].delta);
    let order = if combos.len() == 2 {
        if gs < gd {
            [1, 0, usize::MAX, usize::MAX]
        } else {
            [0, 1, usize::MAX, usize::MAX]
        }
    } else if gs < gd {
        [1, 3, 0, 2]
    } else {
        [0, 2, 1, 3]
    };
    let ord = &order[..combos.len()];
    #[cfg(feature = "diagnostics")]
    let mut attempted = 0usize;
    // Every chart divides by the pinned gap, the pendant/reference gap, and
    // the core-circle area.  Their minimum is a coordinate-free lower bound
    // on the three denominators, so trying the largest bound first is the
    // natural conditioning order.  It changes neither candidates nor
    // coverage.
    for &i in ord {
        let cb = &combos[i];
        let mut charts = conditioned_charts(&cb.delta, &cb.lam);
        charts.sort_by(|left, right| right.1.total_cmp(&left.1));
        for &(chart, _score) in &charts {
            // The planar trace test is sufficient for the floating
            // acceleration pass. The stronger joint-character certificate is
            // reserved for the complete pass, where rejecting one chart
            // avoids certified decic isolation and dominates its fixed hull
            // cost.
            if !character_supported(
                &cb.delta,
                &cb.lam,
                &cb.target,
                chart,
                pass == AxisPass::CriticalCertified,
            ) {
                super::prof::hit(64);
                continue;
            }
            #[cfg(feature = "diagnostics")]
            {
                attempted += 1;
                if std::env::var_os("AXIS_TRACE").is_some() {
                    eprintln!(
                        "AXIS_TRACE try pass={} attempt={attempted} combo={i} chart={chart:?} score={_score:.6e}",
                        if pass == AxisPass::CriticalCertified {
                            "certified"
                        } else {
                            "fast"
                        }
                    );
                }
            }
            let mut certify = |o, r, zero_weight_boundary| accept(i, o, r, zero_weight_boundary);
            if let Some(hit) = disc_attempt(
                cb.delta,
                cb.lam,
                cb.chi,
                chart,
                &cb.dc,
                &cb.lam_m,
                &cb.target,
                pass == AxisPass::CriticalCertified,
                &mut certify,
            ) {
                #[cfg(feature = "diagnostics")]
                if std::env::var_os("AXIS_TRACE").is_some() {
                    eprintln!(
                        "AXIS_TRACE pass={} attempts={attempted} combo={i} chart={chart:?}",
                        if pass == AxisPass::CriticalCertified {
                            "certified"
                        } else {
                            "fast"
                        }
                    );
                }
                return Some(hit);
            }
        }
    }
    None
}

/// Evaluate one chart's regular decic--cubic tower and, on the complete pass,
/// its certified roots and exact singular/boundary strata.
#[allow(clippy::too_many_arguments)]
fn disc_attempt<R>(
    delta: [C; 4],
    lam: [C; 4],
    chi: [C; 5],
    chart: Chart,
    dc: &Mat4,
    lam_m: &Mat4,
    target: &[C; 4],
    certified: bool,
    accept: &mut impl FnMut(Mat4, f64, bool) -> Option<R>,
) -> Option<R> {
    let started = super::prof::start();
    let prep = chart_prep(delta, lam, chi, chart);
    super::prof::rec(59, started);
    let started = super::prof::start();
    let pullback = prep.as_ref().and_then(weighted_invariant_pullback);
    super::prof::rec(60, started);
    if let Some(pullback) = pullback {
        if certified {
            let started = super::prof::start();
            if let Some(roots) = certified_cubic_pullback_roots(&pullback) {
                super::prof::rec(62, started);
                let started = super::prof::start();
                if let Some(hit) = try_discriminant_cover(
                    roots,
                    delta,
                    lam,
                    chi,
                    chart,
                    prep.as_ref(),
                    dc,
                    lam_m,
                    target,
                    accept,
                ) {
                    super::prof::rec(63, started);
                    return Some(hit);
                }
                super::prof::rec(63, started);
            } else {
                super::prof::rec(62, started);
            }
            if let Some(prep) = prep.as_ref() {
                let roots = zero_weight_pendants(prep, delta, lam, target, true);
                let started = super::prof::start();
                let hit = try_pendant_values(
                    &roots,
                    true,
                    delta,
                    lam,
                    chi,
                    chart,
                    Some(prep),
                    dc,
                    lam_m,
                    target,
                    accept,
                );
                super::prof::rec(63, started);
                return hit;
            }
        } else {
            let started = super::prof::start();
            let roots = cubic_pullback_roots(&pullback);
            super::prof::rec(61, started);
            let started = super::prof::start();
            let hit = try_discriminant_cover(
                roots,
                delta,
                lam,
                chi,
                chart,
                prep.as_ref(),
                dc,
                lam_m,
                target,
                accept,
            );
            super::prof::rec(63, started);
            if hit.is_some() {
                return hit;
            }
        }
    } else if certified {
        if let Some(prep) = prep.as_ref() {
            if let Some(roots) = singular_subresultant_pendants(prep) {
                if let Some(hit) = try_discriminant_cover(
                    roots,
                    delta,
                    lam,
                    chi,
                    chart,
                    Some(prep),
                    dc,
                    lam_m,
                    target,
                    accept,
                ) {
                    return Some(hit);
                }
            }
            let roots = zero_weight_pendants(prep, delta, lam, target, true);
            return try_pendant_values(
                &roots,
                true,
                delta,
                lam,
                chi,
                chart,
                Some(prep),
                dc,
                lam_m,
                target,
                accept,
            );
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn try_discriminant_cover<R>(
    mut troots: Vec<f64>,
    delta: [C; 4],
    lam: [C; 4],
    chi: [C; 5],
    chart: (usize, (usize, usize), usize, usize),
    prep: Option<&ChartPrep>,
    dc: &Mat4,
    lam_m: &Mat4,
    target: &[C; 4],
    accept: &mut impl FnMut(Mat4, f64, bool) -> Option<R>,
) -> Option<R> {
    // Exact cover completion: a compact component's t-projection is an
    // interval whose endpoints are fold (discriminant) roots or the
    // strip boundary, so the midpoint of some consecutive fold interval
    // lies strictly inside it. Trying every such midpoint alongside the
    // fold points themselves makes this pass a provably complete cover
    // of the chart's compact components (given the certified-degree
    // rooting), retiring the fixed-section sweep's coverage caveat.
    troots.sort_by(|a, b| a.total_cmp(b));
    let mut cover = Vec::with_capacity(2 * troots.len() + 1);
    let mut prev = 0.0f64;
    for &r in &troots {
        let mid = 0.5 * (prev + r);
        if mid > 1e-7 && mid < 1.0 - 1e-7 {
            cover.push(mid);
        }
        cover.push(r);
        prev = r;
    }
    let mid = 0.5 * (prev + 1.0);
    if mid > 1e-7 && mid < 1.0 - 1e-7 {
        cover.push(mid);
    }
    cover.sort_by(|a, b| (a - 0.5).abs().total_cmp(&(b - 0.5).abs()));
    try_pendant_values(
        &cover, false, delta, lam, chi, chart, prep, dc, lam_m, target, accept,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_pendant_values<R>(
    pendants: &[f64],
    zero_weight_boundary: bool,
    delta: [C; 4],
    lam: [C; 4],
    chi: [C; 5],
    chart: (usize, (usize, usize), usize, usize),
    prep: Option<&ChartPrep>,
    dc: &Mat4,
    lam_m: &Mat4,
    target: &[C; 4],
    accept: &mut impl FnMut(Mat4, f64, bool) -> Option<R>,
) -> Option<R> {
    for &pendant in pendants {
        if let Some((o, residual)) = realize_pendant(
            delta, lam, chi, chart, prep, pendant, 3e-3, dc, lam_m, target,
        ) {
            if let Some(hit) = accept(o, residual, zero_weight_boundary) {
                return Some(hit);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------
// s187: compile an axis chart once. The pendant dependence of the whole
// affine chain is Laurent: s0(t) = s_m1/t + s00 with a t-independent
// direction, every later step is linear over R in (s0, conj s0, sd),
// and only the documented /t and /(1-t) divisions occur. The prep runs
// the EXISTING chain arithmetic once on the Laurent coefficient level
// (coefficientwise, no sampling), so line_at(t) rebuilds the exact
// AxisLine with scalar Horner + three precomputed tdelta^-1 columns --
// no per-section LU, no solves, no chi evaluation.

/// Laurent triple a/t + b + c*t.
#[derive(Clone, Copy)]
struct L3(C, C, C);

impl L3 {
    #[inline]
    fn at(&self, t: f64) -> C {
        self.0 / t + self.1 + self.2 * t
    }
}

pub(super) struct ChartPrep {
    chart: (usize, (usize, usize), usize, usize),
    core: [usize; 3],
    /// tdelta^-1 e1, e2, e3
    tcols: [Vector3<f64>; 3],
    /// other-row consistency residual c_m1/t + c_0 (|.| <= 2e-7 gate)
    cons: (f64, f64),
    /// num_0[0](t) Laurent (qn2_0 = this / (1-t))
    n00: L3,
    /// num_d[0](t) = nd0.0 + nd0.1 * t (qn2_d = this / (1-t))
    nd0: (C, C),
    /// rem0(t), remd(t) Laurent
    rem0: L3,
    remd: (C, C),
    /// vd0(t) = (s0(t) - aa*sum_dc)/bb - kv * qn2_0(t); vdd = sd/bb - kv*qn2_d
    v0a: (C, C),
    kv: C,
    sdb: C,
}

pub(super) fn chart_prep(
    delta: [C; 4],
    lam: [C; 4],
    chi: [C; 5],
    chart: (usize, (usize, usize), usize, usize),
) -> Option<ChartPrep> {
    let (axis, pair, pendant_label, reference_label) = chart;
    let mut cv = [0usize; 3];
    let mut n = 0;
    for k in 0..4 {
        if k != axis {
            cv[n] = k;
            n += 1;
        }
    }
    let dc = [delta[cv[0]], delta[cv[1]], delta[cv[2]]];
    let (la, lb, lc, lr) = (
        lam[pair.0],
        lam[pair.1],
        lam[pendant_label],
        lam[reference_label],
    );
    let aa = (la + lb) / 2.0;
    let bb = (la - lb) / 2.0;
    if bb.norm() < 1e-12 {
        return None;
    }
    let product = dc.iter().product::<C>() * la * lb * lr;
    let mu = lr * delta[axis];
    let rho = lc - lr;
    if rho.norm() < 1e-12 {
        return None;
    }
    let da = delta[axis];
    // desired(t) = d_m1 / t
    let d_m1 = -peval(&chi, mu) / (rho * da);
    let constant = mu * mu * mu - product;
    let coeff_x = -mu * mu + product * mu;
    let coeff_y = C::new(0.0, -1.0) * (mu * mu + product * mu);
    let rows = [[coeff_x.re, coeff_y.re], [coeff_x.im, coeff_y.im]];
    let (dm1_r, cst_r) = ([d_m1.re, d_m1.im], [constant.re, constant.im]);
    let norms = [rows[0][0].hypot(rows[0][1]), rows[1][0].hypot(rows[1][1])];
    let ir = usize::from(norms[1] > norms[0]);
    if norms[ir] < 1e-12 {
        return None;
    }
    let io = 1 - ir;
    let normal = rows[ir];
    let nn = normal[0] * normal[0] + normal[1] * normal[1];
    let w = C::new(normal[0], normal[1]) / nn;
    // rhs_ir(t) = dm1_r[ir]/t - cst_r[ir]; point(t) = w * rhs_ir(t)
    let s_m1 = w * dm1_r[ir];
    let s00 = -w * cst_r[ir];
    let direction = [-normal[1] / nn.sqrt(), normal[0] / nn.sqrt()];
    let sd = C::new(direction[0], direction[1]);
    // other-row residual: g_o * rhs_ir(t) - rhs_o(t), Laurent (c_m1, c_0)
    let g_o = rows[io][0] * w.re + rows[io][1] * w.im;
    let cons = (g_o * dm1_r[ir] - dm1_r[io], -g_o * cst_r[ir] + cst_r[io]);
    let tdelta = Matrix3::new(
        1.0, 1.0, 1.0, dc[0].re, dc[1].re, dc[2].re, dc[0].im, dc[1].im, dc[2].im,
    );
    if tdelta.determinant().abs() < 1e-10 {
        return None;
    }
    let lu = tdelta.lu();
    let tcols = [
        lu.solve(&Vector3::new(1.0, 0.0, 0.0))?,
        lu.solve(&Vector3::new(0.0, 1.0, 0.0))?,
        lu.solve(&Vector3::new(0.0, 0.0, 1.0))?,
    ];
    // the chain on Laurent (m1, z0) pairs; q0v[k] = (m1_k/t + z0_k)
    let zero = C::new(0.0, 0.0);
    let one = C::new(1.0, 0.0);
    let q0v = [
        (zero, one),
        (-s_m1, -s00),
        (product * s_m1.conj(), product * s00.conj()),
        (zero, -product),
    ];
    let c2_0 = [
        q0v[0],
        (q0v[1].0 - mu * q0v[0].0, q0v[1].1 - mu * q0v[0].1),
        (q0v[2].0 - mu * q0v[1].0, q0v[2].1 - mu * q0v[1].1),
        (q0v[3].0 - mu * q0v[2].0, q0v[3].1 - mu * q0v[2].1),
        (-mu * q0v[3].0, -mu * q0v[3].1),
    ];
    let qc_0: [(C, C); 4] =
        std::array::from_fn(|k| (c2_0[k + 1].0 / rho, (c2_0[k + 1].1 - chi[k + 1]) / rho));
    // num_0[k](t) = qc_0[k] - t*da*q0v[k]  (Laurent m1, z0, p1)
    let num_0: [L3; 4] =
        std::array::from_fn(|k| L3(qc_0[k].0, qc_0[k].1 - da * q0v[k].0, -da * q0v[k].1));
    // d-side: qdv constant
    let qdv = [zero, -sd, product * sd.conj(), zero];
    let c2_d = [
        qdv[0],
        qdv[1] - mu * qdv[0],
        qdv[2] - mu * qdv[1],
        qdv[3] - mu * qdv[2],
        -mu * qdv[3],
    ];
    let qc_d: [C; 4] = std::array::from_fn(|k| c2_d[k + 1] / rho);
    // num_d[k](t) = qc_d[k] - t*da*qdv[k]  (z0, p1)
    let num_d: [(C, C); 4] = std::array::from_fn(|k| (qc_d[k], -da * qdv[k]));
    // rem Horner over Laurent components
    let h3 = |v: &[L3; 4]| -> L3 {
        let mut acc = v[3];
        for k in [2usize, 1, 0] {
            acc = L3(
                v[k].0 + mu * acc.0,
                v[k].1 + mu * acc.1,
                v[k].2 + mu * acc.2,
            );
        }
        // Horner order in axis_line is num[3] + mu(num[2] + mu(num[1] + mu num[0]))
        acc
    };
    // careful: axis_line's Horner nests num_0[0] innermost; replicate exactly
    let rem0 = {
        let mut acc = num_0[0];
        for k in [1usize, 2, 3] {
            acc = L3(
                num_0[k].0 + mu * acc.0,
                num_0[k].1 + mu * acc.1,
                num_0[k].2 + mu * acc.2,
            );
        }
        acc
    };
    let _ = h3;
    let remd = {
        let mut acc = num_d[0];
        for k in [1usize, 2, 3] {
            acc = (num_d[k].0 + mu * acc.0, num_d[k].1 + mu * acc.1);
        }
        acc
    };
    let sum_dc = dc[0] + dc[1] + dc[2];
    let v0a = ((s_m1) / bb, (s00 - aa * sum_dc) / bb);
    let kv = (lr - aa) / bb;
    let sdb = sd / bb;
    Some(ChartPrep {
        chart,
        core: cv,
        tcols,
        cons,
        n00: num_0[0],
        nd0: num_d[0],
        rem0,
        remd,
        v0a,
        kv,
        sdb,
    })
}

impl ChartPrep {
    /// The compiled evaluator's well-conditioned range: outside it the 1/t
    /// and 1/(1-t) Laurent amplification reaches ~1e-6 relative against the
    /// reference chain (shadow census), enough to flip borderline precision
    /// classes; callers fall back to axis_line there (rare fold-cover
    /// sections only).
    pub(super) fn in_range(pendant: f64) -> bool {
        (0.01..=0.99).contains(&pendant)
    }

    pub(super) fn line_at(&self, pendant: f64) -> Option<AxisLine> {
        if !(0.0 < pendant && pendant < 1.0) {
            return None;
        }
        let t = pendant;
        if (self.cons.0 / t + self.cons.1).abs() > 2e-7 {
            return None;
        }
        let omt = 1.0 - t;
        let qn2_0 = self.n00.at(t) / omt;
        let qn2_d = (self.nd0.0 + self.nd0.1 * t) / omt;
        let comb = |a: f64, b: f64| -> Vector3<f64> { self.tcols[1] * a + self.tcols[2] * b };
        let u0 = self.tcols[0] + comb(qn2_0.re, qn2_0.im);
        let ud = comb(qn2_d.re, qn2_d.im);
        let vd0 = self.v0a.0 / t + self.v0a.1 - self.kv * qn2_0;
        let vdd = self.sdb - self.kv * qn2_d;
        let v0 = comb(vd0.re, vd0.im);
        let vd_ = comb(vdd.re, vdd.im);
        let a0: [f64; 3] = std::array::from_fn(|k| (1.0 - u0[k] + v0[k]) / 2.0);
        let ad: [f64; 3] = std::array::from_fn(|k| (-ud[k] + vd_[k]) / 2.0);
        let b0: [f64; 3] = std::array::from_fn(|k| (1.0 - u0[k] - v0[k]) / 2.0);
        let bd: [f64; 3] = std::array::from_fn(|k| (-ud[k] - vd_[k]) / 2.0);
        let (axis, pair, pendant_label, reference_label) = self.chart;
        Some(AxisLine {
            core: self.core,
            axis,
            pair,
            pendant_label,
            reference_label,
            pendant,
            a0,
            ad,
            b0,
            bd,
            rem0: self.rem0.at(t),
            remd: self.remd.0 + self.remd.1 * t,
        })
    }
}

// Every polynomial in the axis selector has degree at most 36.  Keeping one
// bounded stack representation makes the exact low-degree algebra visible
// and removes the heap traffic that formerly dominated direct-decic charts.
const POLY_CAPACITY: usize = 37;

#[derive(Clone, Copy)]
struct Poly {
    coefficients: [f64; POLY_CAPACITY],
    len: usize,
}

impl Poly {
    fn zero() -> Self {
        Self {
            coefficients: [0.0; POLY_CAPACITY],
            len: 1,
        }
    }

    fn from_slice(values: &[f64]) -> Self {
        assert!(!values.is_empty() && values.len() <= POLY_CAPACITY);
        let mut result = Self::zero();
        result.len = values.len();
        result.coefficients[..values.len()].copy_from_slice(values);
        result
    }

    fn as_slice(&self) -> &[f64] {
        &self.coefficients[..self.len]
    }

    fn scale(self, factor: f64) -> Self {
        let mut result = self;
        for value in &mut result.coefficients[..result.len] {
            *value *= factor;
        }
        result
    }

    fn add_scaled(mut self, other: &Self, factor: f64) -> Self {
        self.len = self.len.max(other.len);
        for (slot, &value) in self.coefficients.iter_mut().zip(other.as_slice()) {
            *slot += factor * value;
        }
        self
    }

    fn mul(self, other: &Self) -> Self {
        let len = self.len + other.len - 1;
        assert!(len <= POLY_CAPACITY);
        let mut result = Self::zero();
        result.len = len;
        for (i, &left) in self.as_slice().iter().enumerate() {
            for (j, &right) in other.as_slice().iter().enumerate() {
                result.coefficients[i + j] += left * right;
            }
        }
        result
    }

    fn max_abs(&self) -> f64 {
        self.as_slice()
            .iter()
            .fold(0.0f64, |value, x| value.max(x.abs()))
    }
}

fn pscale(a: &Poly, scale: f64) -> Poly {
    (*a).scale(scale)
}

fn padd(a: &Poly, b: &Poly) -> Poly {
    (*a).add_scaled(b, 1.0)
}

fn psub(a: &Poly, b: &Poly) -> Poly {
    (*a).add_scaled(b, -1.0)
}

fn pmul(a: &Poly, b: &Poly) -> Poly {
    (*a).mul(b)
}

type WeightNumerators = ([Poly; 3], [Poly; 3], [Poly; 3], [Poly; 3]);

/// Numerators of the six core weights after `sigma=t*tau`:
/// `alpha_i=(a0_i(t)+sigma*a1_i(t))/(t(1-t))`, and likewise for beta.
fn weight_numerators(prep: &ChartPrep) -> Option<WeightNumerators> {
    let den = Poly::from_slice(&[0.0, 1.0, -1.0]);
    let den_c = [0.0, 1.0, -1.0].map(|x| C::new(x, 0.0));
    let n0 = [prep.n00.0, prep.n00.1, prep.n00.2];
    let nd = [C::new(0.0, 0.0), prep.nd0.0, prep.nd0.1];
    let one_minus_t = [C::new(1.0, 0.0), C::new(-1.0, 0.0), C::new(0.0, 0.0)];
    let v0_num: [C; 3] = std::array::from_fn(|k| {
        prep.v0a.0 * one_minus_t[k] + prep.v0a.1 * den_c[k] - prep.kv * n0[k]
    });
    let vd_num: [C; 3] = std::array::from_fn(|k| prep.sdb * den_c[k] - prep.kv * nd[k]);
    let combine = |z: &[C; 3], row: usize| -> Poly {
        let values: [f64; 3] =
            std::array::from_fn(|k| prep.tcols[1][row] * z[k].re + prep.tcols[2][row] * z[k].im);
        Poly::from_slice(&values)
    };

    let mut a0 = [Poly::zero(); 3];
    let mut a1 = [Poly::zero(); 3];
    let mut b0 = [Poly::zero(); 3];
    let mut b1 = [Poly::zero(); 3];
    for row in 0..3 {
        let u0 = padd(&pscale(&den, prep.tcols[0][row]), &combine(&n0, row));
        let ud = combine(&nd, row);
        let v0 = combine(&v0_num, row);
        let vd = combine(&vd_num, row);
        a0[row] = pscale(&padd(&psub(&den, &u0), &v0), 0.5);
        b0[row] = pscale(&psub(&psub(&den, &u0), &v0), 0.5);
        let ad = pscale(&padd(&pscale(&ud, -1.0), &vd), 0.5);
        let bd = pscale(&psub(&pscale(&ud, -1.0), &vd), 0.5);
        let scale = ad.max_abs().max(bd.max_abs()).max(1e-300);
        if ad.coefficients[0].abs().max(bd.coefficients[0].abs()) > 1e-11 * scale {
            return None;
        }
        a1[row] = Poly::from_slice(&ad.as_slice()[1..]);
        b1[row] = Poly::from_slice(&bd.as_slice()[1..]);
    }
    Some((a0, a1, b0, b1))
}

/// Coefficients in `sigma` of the denominator-cleared Heron equation after
/// `sigma=t*tau`. Each returned vector is a power polynomial in the pendant
/// coordinate `t`. This is the exact coefficient-level counterpart of
/// sampling `quartic_coefficients(line, true)` and dividing coefficient `k`
/// by `t^k`.
fn sigma_heron_from_weights(weights: &WeightNumerators) -> [Poly; 5] {
    let (a0, a1, b0, b1) = weights;
    let x: [[Poly; 3]; 3] = std::array::from_fn(|i| {
        [
            pmul(&a0[i], &b0[i]),
            padd(&pmul(&a0[i], &b1[i]), &pmul(&a1[i], &b0[i])),
            pmul(&a1[i], &b1[i]),
        ]
    });
    // Heron is the determinant of the symmetric rank-one packet
    //
    //   K = [[2*x0, x0+x1-x2], [x0+x1-x2, 2*x1]].
    //
    // With K(sigma)=K0+sigma*K1+sigma^2*K2, polarizing this 2x2
    // determinant constructs the five quartic coefficients with 15
    // polynomial products instead of the 36 products in the expanded
    // `(sum x)^2-2*sum(x^2)` expression.
    let packet: [[Poly; 3]; 3] = std::array::from_fn(|degree| {
        [
            pscale(&x[0][degree], 2.0),
            psub(&padd(&x[0][degree], &x[1][degree]), &x[2][degree]),
            pscale(&x[1][degree], 2.0),
        ]
    });
    let det = |left: &[Poly; 3]| psub(&pmul(&left[0], &left[2]), &pmul(&left[1], &left[1]));
    let polar = |left: &[Poly; 3], right: &[Poly; 3]| {
        psub(
            &padd(&pmul(&left[0], &right[2]), &pmul(&left[2], &right[0])),
            &pscale(&pmul(&left[1], &right[1]), 2.0),
        )
    };
    [
        det(&packet[0]),
        polar(&packet[0], &packet[1]),
        padd(&polar(&packet[0], &packet[2]), &det(&packet[1])),
        polar(&packet[1], &packet[2]),
        det(&packet[2]),
    ]
}

#[cfg(test)]
fn sigma_heron_power(prep: &ChartPrep) -> Option<[Poly; 5]> {
    let weights = weight_numerators(prep)?;
    Some(sigma_heron_from_weights(&weights))
}

/// Direct power coefficients of the sigma-quartic discriminant. This avoids
/// reconstructing a known polynomial from 49 floating samples. Boundary
/// saturation is deliberately kept separate so its exact multiplicity can be
/// checked before division.
fn sigma_invariants_from_weights(weights: &WeightNumerators) -> (Poly, Poly) {
    let h = sigma_heron_from_weights(weights);
    sigma_invariants_from_heron(&h)
}

fn sigma_invariants_from_heron(h: &[Poly; 5]) -> (Poly, Poly) {
    let (e, d, c, b, a) = (&h[0], &h[1], &h[2], &h[3], &h[4]);
    let invariant_i = padd(
        &psub(&pscale(&pmul(a, e), 12.0), &pscale(&pmul(b, d), 3.0)),
        &pmul(c, c),
    );
    let invariant_j = {
        let mut value = pscale(&pmul(&pmul(a, c), e), 72.0);
        value = padd(&value, &pscale(&pmul(&pmul(b, c), d), 9.0));
        value = psub(&value, &pscale(&pmul(a, &pmul(d, d)), 27.0));
        value = psub(&value, &pscale(&pmul(&pmul(b, b), e), 27.0));
        psub(&value, &pscale(&pmul(&pmul(c, c), c), 2.0))
    };
    (invariant_i, invariant_j)
}

/// Critical pendant values when the quartic discriminant vanishes identically.
///
/// Over `k(t)`, the first nonzero subresultant of `F` and `F_sigma` is their
/// generic gcd.  If that gcd is linear `L(t)*sigma+M(t)`, its square-free
/// cubic loses another distinct root exactly where `L=M=0`.  If the generic
/// gcd is quadratic, the corresponding condition is `A=B=C=0`.  FLINT takes
/// these coefficient gcds exactly over the dyadic rationals and Arb isolates
/// their common real roots.  A cubic gcd leaves a linear square-free curve
/// and has no interior fold values.
fn singular_subresultant_roots(heron: &[Poly; 5]) -> Option<Vec<f64>> {
    let (e, d, c, b, a) = (&heron[0], &heron[1], &heron[2], &heron[3], &heron[4]);
    let aa = psub(&pscale(&pmul(a, c), 8.0), &pscale(&pmul(b, b), 3.0));
    let bb = psub(&pscale(&pmul(a, d), 12.0), &pscale(&pmul(b, c), 2.0));
    let cc = psub(&pscale(&pmul(a, e), 16.0), &pmul(b, d));

    let p3 = pscale(a, 4.0);
    let p2 = pscale(b, 3.0);
    let p1 = pscale(c, 2.0);
    let p0 = *d;
    let aa2 = pmul(&aa, &aa);
    let bb2 = pmul(&bb, &bb);

    let l_terms = [
        pmul(&aa2, &p1),
        pmul(&pmul(&aa, &p3), &cc),
        pmul(&pmul(&aa, &p2), &bb),
        pmul(&p3, &bb2),
    ];
    let l = l_terms[0]
        .add_scaled(&l_terms[1], -1.0)
        .add_scaled(&l_terms[2], -1.0)
        .add_scaled(&l_terms[3], 1.0);
    let m_terms = [
        pmul(&aa2, &p0),
        pmul(&pmul(&aa, &p2), &cc),
        pmul(&pmul(&p3, &bb), &cc),
    ];
    let m = m_terms[0]
        .add_scaled(&m_terms[1], -1.0)
        .add_scaled(&m_terms[2], 1.0);

    let l_scale = l_terms
        .iter()
        .fold(0.0f64, |scale, term| scale.max(term.max_abs()));
    let m_scale = m_terms
        .iter()
        .fold(0.0f64, |scale, term| scale.max(term.max_abs()));
    let linear_small =
        l.max_abs() <= 2e-10 * l_scale.max(1e-300) && m.max_abs() <= 2e-10 * m_scale.max(1e-300);
    if !linear_small {
        return super::arb_roots::common_real_roots_power(&[l.as_slice(), m.as_slice()]);
    }

    let quadratic_scale = [
        pmul(a, c).max_abs(),
        pmul(b, b).max_abs(),
        pmul(a, d).max_abs(),
        pmul(b, c).max_abs(),
        pmul(a, e).max_abs(),
        pmul(b, d).max_abs(),
    ]
    .into_iter()
    .fold(0.0f64, f64::max)
    .max(1e-300);
    let quadratic_small =
        aa.max_abs().max(bb.max_abs()).max(cc.max_abs()) <= 2e-10 * quadratic_scale;
    if quadratic_small {
        return Some(Vec::new());
    }
    super::arb_roots::common_real_roots_power(&[aa.as_slice(), bb.as_slice(), cc.as_slice()])
}

fn singular_subresultant_pendants(prep: &ChartPrep) -> Option<Vec<f64>> {
    let weights = weight_numerators(prep)?;
    let heron = sigma_heron_from_weights(&weights);
    let (invariant_i, invariant_j) = sigma_invariants_from_heron(&heron);
    let left = pscale(&pmul(&pmul(&invariant_i, &invariant_i), &invariant_i), 4.0);
    let right = pmul(&invariant_j, &invariant_j);
    let discriminant = psub(&left, &right);
    let scale = left.max_abs().max(right.max_abs()).max(1e-300);
    if discriminant.max_abs() > 2e-10 * scale {
        return None;
    }
    let mut roots = singular_subresultant_roots(&heron)?;
    roots.retain(|&root| root > 1e-7 && root < 1.0 - 1e-7);
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|left, right| (*left - *right).abs() < 1e-12);
    Some(roots)
}

/// Recover the common cubic carrier directly from the six mixed brackets.
/// The proved physical-axis factor law is
///
/// `c_ij(t) = p_ij (t^3-a*t+c) + q_ij t(t-1)` for `i != j`.
///
/// Thus one nonzero cubic leading coefficient gives `c`, `e=1-a+c`, and
/// `a` from its values at 0, 1, and infinity.  Unlike logarithmic jets of a
/// quartic invariant, this remains regular when c=0 or e=0; those endpoint
/// walls merely contribute a filtered t=0 or t=1 root to the cubic lift.
fn mixed_brackets(weights: &WeightNumerators) -> [Poly; 6] {
    let (a0, a1, b0, b1) = weights;
    let mut brackets = [Poly::zero(); 6];
    let mut count = 0;
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            brackets[count] = psub(&pmul(&a0[i], &b1[j]), &pmul(&a1[i], &b0[j]));
            count += 1;
        }
    }
    brackets
}

fn mixed_bracket_parameters(brackets: &[Poly; 6]) -> Option<(f64, f64)> {
    let carrier = brackets.iter().max_by(|left, right| {
        let score =
            |polynomial: &Poly| polynomial.coefficients[3].abs() / polynomial.max_abs().max(1e-300);
        score(left).total_cmp(&score(right))
    })?;
    let scale = carrier.max_abs();
    let lead = carrier.coefficients[3];
    if scale == 0.0 || !scale.is_finite() || lead.abs() < 1e-11 * scale {
        return None;
    }
    let c = carrier.coefficients[0] / lead;
    let e = carrier.as_slice().iter().sum::<f64>() / lead;
    let a = 1.0 + c - e;
    if !a.is_finite() || !c.is_finite() {
        return None;
    }

    let mut relative_remainder = 0.0f64;
    for bracket in brackets {
        let bracket_scale = bracket.max_abs();
        if bracket_scale <= 1e-14 * scale {
            continue;
        }
        let d = &bracket.coefficients;
        relative_remainder = relative_remainder
            .max((d[0] - c * d[3]).abs() / bracket_scale)
            .max((d[1] + d[2] + a * d[3]).abs() / bracket_scale);
    }
    (relative_remainder < 2e-6).then_some((a, c))
}

/// Recover `F_w(u)` from
///
/// `f(t) = [t(t-1)]^w F_w((t^3-a*t+c)/(t(t-1)))`.
///
/// The basis element for `u^k` is monic of degree `2w+k`, so one descending
/// triangular sweep recovers every coefficient.  The full remainder is a
/// decline-only certificate of the weighted descent.
fn weighted_descend(polynomial: &Poly, weight: usize, a: f64, c: f64) -> Option<(Poly, f64)> {
    let scale = polynomial.max_abs();
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    let numerator = Poly::from_slice(&[c, -a, 0.0, 1.0]);
    let pole = Poly::from_slice(&[0.0, -1.0, 1.0]);
    let mut numerator_powers = [Poly::zero(); 7];
    let mut pole_powers = [Poly::zero(); 7];
    numerator_powers[0] = Poly::from_slice(&[1.0]);
    pole_powers[0] = Poly::from_slice(&[1.0]);
    for exponent in 1..=weight {
        numerator_powers[exponent] = numerator_powers[exponent - 1].mul(&numerator);
        pole_powers[exponent] = pole_powers[exponent - 1].mul(&pole);
    }
    let mut remainder = *polynomial;
    let mut quotient = Poly::zero();
    quotient.len = weight + 1;
    for k in (0..=weight).rev() {
        let basis = pmul(&numerator_powers[k], &pole_powers[weight - k]);
        debug_assert_eq!(basis.len, 2 * weight + k + 1);
        let coefficient = remainder.coefficients[2 * weight + k];
        quotient.coefficients[k] = coefficient;
        for (slot, &value) in remainder.coefficients.iter_mut().zip(basis.as_slice()) {
            *slot -= coefficient * value;
        }
    }
    let relative_remainder = remainder.max_abs() / scale;
    quotient
        .as_slice()
        .iter()
        .all(|value| value.is_finite())
        .then_some((quotient, relative_remainder))
}

fn weighted_invariant_pullback(prep: &ChartPrep) -> Option<CauerPullback> {
    let weights = weight_numerators(prep)?;
    let brackets = mixed_brackets(&weights);
    let (invariant_i, invariant_j) = sigma_invariants_from_weights(&weights);
    let (a, c) = mixed_bracket_parameters(&brackets)?;
    let (i4, i_remainder) = weighted_descend(&invariant_i, 4, a, c)?;
    let (j6, j_remainder) = weighted_descend(&invariant_j, 6, a, c)?;
    let numerator = Poly::from_slice(&[c, -a, 0.0, 1.0]);
    let selector_full = psub(&pscale(&pmul(&pmul(&i4, &i4), &i4), 4.0), &pmul(&j6, &j6));
    let scale = selector_full.max_abs();
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    let high_remainder = selector_full
        .as_slice()
        .iter()
        .skip(11)
        .fold(0.0f64, |value, x| value.max(x.abs()))
        / scale;
    let mut selector = selector_full.scale(1.0 / scale);
    selector.len = 11;
    let relative_remainder = i_remainder.max(j_remainder).max(high_remainder);
    (relative_remainder < 2e-6).then_some(CauerPullback {
        selector,
        numerator,
    })
}

#[derive(Clone)]
struct CauerPullback {
    /// T(u), ascending power coefficients, of degree at most ten.
    selector: Poly,
    /// Numerator P(t) of the rational carrier R=P/[t(t-1)].
    numerator: Poly,
}

fn lifted_carrier_polynomial(pullback: &CauerPullback, u: f64) -> [f64; 4] {
    let mut polynomial = [0.0; 4];
    polynomial[..pullback.numerator.len].copy_from_slice(pullback.numerator.as_slice());
    // P(t)-u*t(t-1) = P(t)+u*t-u*t^2.
    polynomial[1] += u;
    polynomial[2] -= u;
    polynomial
}

/// Roots of the theorem-owned selector using its compile-time degree.  The
/// dynamic companion remains the exact degree-drop path; the regular decic
/// avoids heap allocation and dynamic matrix dispatch.
fn selector_roots(selector: &Poly) -> Vec<C> {
    let scale = selector.max_abs().max(1e-300);
    match selector.len - 1 {
        10 if selector.coefficients[10].abs() >= 1e-13 * scale => {
            let lead = selector.coefficients[10];
            let companion = SMatrix::<f64, 10, 10>::from_fn(|i, j| {
                if j == 9 {
                    -selector.coefficients[i] / lead
                } else if i == j + 1 {
                    1.0
                } else {
                    0.0
                }
            });
            companion.complex_eigenvalues().iter().copied().collect()
        }
        _ => poly_roots(selector.as_slice()),
    }
}

fn cubic_pullback_roots(pullback: &CauerPullback) -> Vec<f64> {
    let mut roots = Vec::new();
    for u in selector_roots(&pullback.selector) {
        if u.im.abs() >= 2e-5 {
            continue;
        }
        let cubic = lifted_carrier_polynomial(pullback, u.re);
        let mut lifted = [0.0; 3];
        let count = super::real_roots_cubic(&cubic, &mut lifted);
        roots.extend(
            lifted[..count]
                .iter()
                .copied()
                .filter(|&root| root.is_finite() && root > 1e-7 && root < 1.0 - 1e-7),
        );
    }
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|left, right| (*left - *right).abs() < 1e-8);
    roots
}

/// Certified real roots of the exact generic tower
///
/// `T_10(u) = 0` and `t^3 - u t^2 + (u-a)t + c = 0`.
///
/// FLINT first removes repeated factors exactly over the dyadic rationals;
/// Arb owns square-free root isolation at both levels.  Floating roots are
/// seeds only.  A return of `None` means certified isolation itself failed,
/// not that the selector or lift had multiplicity.
fn certified_cubic_pullback_roots(pullback: &CauerPullback) -> Option<Vec<f64>> {
    let decic_seeds: Vec<(f64, f64)> = selector_roots(&pullback.selector)
        .into_iter()
        .map(|root| (root.re, root.im))
        .collect();
    let u_roots =
        super::arb_roots::real_roots_power_seeded(pullback.selector.as_slice(), &decic_seeds)?;

    let mut roots = Vec::new();
    for u in u_roots {
        let cubic = lifted_carrier_polynomial(pullback, u);
        let cubic_seeds: Vec<(f64, f64)> = poly_roots(&cubic)
            .into_iter()
            .map(|root| (root.re, root.im))
            .collect();
        let lifted = super::arb_roots::real_roots_power_seeded(&cubic, &cubic_seeds)?;
        roots.extend(
            lifted
                .into_iter()
                .filter(|&root| root > 1e-7 && root < 1.0 - 1e-7),
        );
    }
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|left, right| (*left - *right).abs() < 1e-12);
    Some(roots)
}

/// Pendant values at which one core weight vanishes on the Heron curve.
/// If weight i is zero, Heron is `-(x_j-x_k)^2`, so the remaining condition is
/// exactly `x_j=x_k`. Eliminating sigma between that quadratic and the chosen
/// affine weight gives `q0*l1^2-q1*l0*l1+q2*l0^2`.
fn zero_weight_pendants(
    prep: &ChartPrep,
    delta: [C; 4],
    lam: [C; 4],
    target: &[C; 4],
    certified: bool,
) -> Vec<f64> {
    let Some((a0, a1, b0, b1)) = weight_numerators(prep) else {
        return Vec::new();
    };
    let x: [[Poly; 3]; 3] = std::array::from_fn(|i| {
        [
            pmul(&a0[i], &b0[i]),
            padd(&pmul(&a0[i], &b1[i]), &pmul(&a1[i], &b0[i])),
            pmul(&a1[i], &b1[i]),
        ]
    });
    let mut roots = Vec::new();
    let pair = prep.chart.1;
    for family in 0..2 {
        for i in 0..3 {
            let zero_column = if family == 0 { pair.0 } else { pair.1 };
            if !zero_weight_character_supported(
                &delta,
                &lam,
                target,
                prep.chart,
                prep.core[i],
                zero_column,
                certified,
            ) {
                super::prof::hit(65);
                continue;
            }
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            let (l0, l1) = if family == 0 {
                (&a0[i], &a1[i])
            } else {
                (&b0[i], &b1[i])
            };
            let q: [Poly; 3] = std::array::from_fn(|s| psub(&x[j][s], &x[k][s]));
            let resultant = padd(
                &psub(&pmul(&q[0], &pmul(l1, l1)), &pmul(&q[1], &pmul(l0, l1))),
                &pmul(&q[2], &pmul(l0, l0)),
            );
            let scale = resultant.max_abs();
            if scale == 0.0 || !scale.is_finite() {
                continue;
            }
            let candidates = poly_roots(resultant.as_slice());
            if certified {
                let seeds: Vec<(f64, f64)> =
                    candidates.iter().map(|root| (root.re, root.im)).collect();
                if let Some(isolated) =
                    super::arb_roots::real_roots_power_seeded(resultant.as_slice(), &seeds)
                {
                    roots.extend(
                        isolated
                            .into_iter()
                            .filter(|&root| root > 1e-7 && root < 1.0 - 1e-7),
                    );
                    continue;
                }
            }
            for root in candidates {
                if root.im.abs() < 2e-6 && root.re > 1e-7 && root.re < 1.0 - 1e-7 {
                    roots.push(root.re);
                }
            }
        }
    }
    roots.sort_by(|a, b| (a - 0.5).abs().total_cmp(&(b - 0.5).abs()));
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    roots
}

#[cfg(test)]
mod coefficient_tests {
    use super::*;

    #[test]
    fn every_axis_face_vertex_passes_its_joint_character_support_gate() {
        let delta = [0.13, 0.47, 1.01, 1.73].map(|angle| C::from_polar(1.0, angle));
        let lam = [0.22, 0.61, 1.19, 2.03].map(|angle| C::from_polar(1.0, angle));
        for index in 0..ORBIT_LEN {
            let chart = orbit_chart(index);
            let forbidden = chart.1;
            for permutation in PERMS24.iter() {
                if permutation[chart.0] == forbidden.0 || permutation[chart.0] == forbidden.1 {
                    continue;
                }
                let roots: [C; 4] = std::array::from_fn(|row| delta[row] * lam[permutation[row]]);
                let target = super::target_poly(&roots.map(|root| root.arg() / 2.0));
                let coefficients = [-target[1], target[2], -target[3], target[4]];
                assert!(character_supported(
                    &delta,
                    &lam,
                    &coefficients,
                    chart,
                    true,
                ));
            }
        }
    }

    #[test]
    fn dense_axis_frames_pass_the_joint_character_support_gate() {
        let delta = [0.13, 0.47, 1.01, 1.73].map(|angle| C::from_polar(1.0, angle));
        let lam = [0.22, 0.61, 1.19, 2.03].map(|angle| C::from_polar(1.0, angle));
        let chart = (0, (0, 1), 2, 3);
        for (theta, a, b, g) in [
            (0.19_f64, 0.31_f64, -0.73_f64, 1.07_f64),
            (0.43, -0.81, 0.27, 0.62),
            (1.11, 0.58, 0.91, -0.36),
        ] {
            let rotation = |i: usize, j: usize, angle: f64| {
                let mut value = Matrix3::identity();
                let (co, si) = (angle.cos(), angle.sin());
                value[(i, i)] = co;
                value[(j, j)] = co;
                value[(i, j)] = -si;
                value[(j, i)] = si;
                value
            };
            let q = rotation(0, 1, a) * rotation(0, 2, b) * rotation(1, 2, g);
            let (co, si) = (theta.cos(), theta.sin());
            let mut o = [[0.0; 4]; 4];
            o[0] = [0.0, 0.0, co, si];
            for row in 0..3 {
                o[row + 1] = [
                    q[(row, 0)],
                    q[(row, 1)],
                    -si * q[(row, 2)],
                    co * q[(row, 2)],
                ];
            }
            let e1: C = (0..4)
                .flat_map(|row| (0..4).map(move |column| (row, column)))
                .map(|(row, column)| delta[row] * lam[column] * o[row][column].powi(2))
                .sum();
            let mut e2 = C::default();
            for i in 0..3 {
                for k in i + 1..4 {
                    for j in 0..3 {
                        for m in j + 1..4 {
                            let minor = o[i][j] * o[k][m] - o[i][m] * o[k][j];
                            e2 += delta[i] * delta[k] * lam[j] * lam[m] * minor.powi(2);
                        }
                    }
                }
            }
            assert!(character_supported(
                &delta,
                &lam,
                &[e1, e2, C::default(), C::new(1.0, 0.0)],
                chart,
                true,
            ));
        }
    }

    #[test]
    fn every_zero_weight_face_vertex_passes_its_trace_support_gate() {
        let delta = [0.13, 0.47, 1.01, 1.73].map(|angle| C::from_polar(1.0, angle));
        let lam = [0.22, 0.61, 1.19, 2.03].map(|angle| C::from_polar(1.0, angle));
        for index in 0..ORBIT_LEN {
            let chart = orbit_chart(index);
            let pair = chart.1;
            let core: Vec<usize> = (0..4).filter(|&row| row != chart.0).collect();
            for family in 0..2 {
                let zero_column = if family == 0 { pair.0 } else { pair.1 };
                for &zero_row in &core {
                    let mut count = 0usize;
                    for permutation in PERMS24.iter() {
                        if permutation[chart.0] == pair.0
                            || permutation[chart.0] == pair.1
                            || permutation[zero_row] == zero_column
                        {
                            continue;
                        }
                        count += 1;
                        let roots: [C; 4] =
                            std::array::from_fn(|row| delta[row] * lam[permutation[row]]);
                        let target = super::target_poly(&roots.map(|root| root.arg() / 2.0));
                        let coefficients = [-target[1], target[2], -target[3], target[4]];
                        assert!(zero_weight_character_supported(
                            &delta,
                            &lam,
                            &coefficients,
                            chart,
                            zero_row,
                            zero_column,
                            true,
                        ));
                    }
                    assert_eq!(count, 8);
                }
            }
        }
    }

    #[test]
    fn mixed_brackets_cover_endpoint_drops() {
        let q = Poly::from_slice(&[0.0, -1.0, 1.0]);
        for (a, c) in [(0.73, 0.0), (1.19, 0.19)] {
            let p = Poly::from_slice(&[c, -a, 0.0, 1.0]);
            let brackets: [Poly; 6] =
                std::array::from_fn(|k| p.scale(0.3 + k as f64).add_scaled(&q, -0.2 * k as f64));
            let (recovered_a, recovered_c) =
                mixed_bracket_parameters(&brackets).expect("cubic bracket carrier");
            assert!((recovered_a - a).abs() < 2e-14);
            assert!((recovered_c - c).abs() < 2e-14);
        }
    }

    #[test]
    fn quartic_subresultants_select_every_square_free_degree() {
        // (sigma-t)^2 (sigma-1)(sigma+1).  The generic square-free degree is
        // three and drops at t=-1,+1.
        let linear_gcd = [
            Poly::from_slice(&[0.0, 0.0, -1.0]),
            Poly::from_slice(&[0.0, 2.0]),
            Poly::from_slice(&[-1.0, 0.0, 1.0]),
            Poly::from_slice(&[0.0, -2.0]),
            Poly::from_slice(&[1.0]),
        ];
        let roots = singular_subresultant_roots(&linear_gcd).expect("linear gcd roots");
        assert_eq!(roots.len(), 2);
        assert!((roots[0] + 1.0).abs() < 1e-13);
        assert!((roots[1] - 1.0).abs() < 1e-13);

        // (sigma^2-t^2)^2.  The generic square-free degree is two and its two
        // roots collide at t=0.
        let quadratic_gcd = [
            Poly::from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0]),
            Poly::zero(),
            Poly::from_slice(&[0.0, 0.0, -2.0]),
            Poly::zero(),
            Poly::from_slice(&[1.0]),
        ];
        let roots = singular_subresultant_roots(&quadratic_gcd).expect("quadratic gcd roots");
        assert_eq!(roots.len(), 1);
        assert!(roots[0].abs() < 1e-13);

        // (sigma-t)^4 has a linear square-free part and therefore no fold
        // values after global multiplicity is removed.
        let cubic_gcd = [
            Poly::from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0]),
            Poly::from_slice(&[0.0, 0.0, 0.0, -4.0]),
            Poly::from_slice(&[0.0, 0.0, 6.0]),
            Poly::from_slice(&[0.0, -4.0]),
            Poly::from_slice(&[1.0]),
        ];
        let roots = singular_subresultant_roots(&cubic_gcd).expect("cubic gcd roots");
        assert!(roots.is_empty());
    }

    #[test]
    fn heron_kernel_recovers_regular_and_boundary_frames() {
        let cases = [
            (
                [1.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0],
                [4.0 / 9.0, 1.0 / 9.0, 4.0 / 9.0],
            ),
            ([0.0, 0.5, 0.5], [0.0, 0.5, 0.5]),
            ([1.0, 0.0, 0.0], [0.0, 0.36, 0.64]),
        ];
        for (alpha, beta) in cases {
            let (a, b) = heron_kernel_columns(alpha, beta).expect("Heron frame");
            let aa = a.iter().map(|value| value * value).sum::<f64>();
            let bb = b.iter().map(|value| value * value).sum::<f64>();
            let ab = (0..3).map(|k| a[k] * b[k]).sum::<f64>();
            assert!((aa - 1.0).abs() < 2e-15);
            assert!((bb - 1.0).abs() < 2e-15);
            assert!(ab.abs() < 2e-15);
            for k in 0..3 {
                assert!((a[k] * a[k] - alpha[k]).abs() < 2e-15);
                assert!((b[k] * b[k] - beta[k]).abs() < 2e-15);
            }
        }
    }

    fn evaluate(polynomial: &[f64], x: f64) -> f64 {
        polynomial
            .iter()
            .rev()
            .fold(0.0, |value, &coefficient| value * x + coefficient)
    }

    #[test]
    fn sigma_coefficients_match_chart_evaluation() {
        let phases =
            |monodromy| super::super::eigphases(super::super::weyl_from_monodromy(monodromy));
        let eb = phases([0.39018406812566, 0.33535840603455, -0.16714093500979]);
        let ep = phases([0.26190622019265, 0.2480406935928, -0.04889170690693]);
        let et = phases([0.3436597844367, 0.07856425613504, -0.01947571687131]);
        let delta = eb.map(|phase| C::from_polar(1.0, 2.0 * phase));
        let lam = ep.map(|phase| C::from_polar(1.0, 2.0 * phase));
        let chi = target_poly(&et);
        let sample_t = [0.17, 0.37, 0.63, 0.83];
        let (prep, chart) = [(0, (0, 1), 2, 3)]
            .into_iter()
            .find_map(|chart| {
                let prep = chart_prep(delta, lam, chi, chart)?;
                sample_t
                    .iter()
                    .all(|&t| prep.line_at(t).is_some())
                    .then_some((prep, chart))
            })
            .expect("generic prepared axis chart");
        let heron = sigma_heron_power(&prep).expect("coefficient construction");
        for t in sample_t {
            let line = prep
                .line_at(t)
                .unwrap_or_else(|| axis_line(delta, lam, chi, chart, t).expect("axis line"));
            let cleared = quartic_coefficients(&line, true).expect("quartic");
            let mut sigma = cleared;
            let mut power = 1.0;
            for coefficient in &mut sigma {
                *coefficient /= power;
                power *= t;
            }
            for degree in 0..5 {
                let direct = evaluate(heron[degree].as_slice(), t);
                let scale = direct.abs().max(sigma[degree].abs()).max(1e-12);
                assert!((direct - sigma[degree]).abs() < 2e-8 * scale);
            }
        }
    }
}
