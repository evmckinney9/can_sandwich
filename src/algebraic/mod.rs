//! Algebraic candidate search: support, multiplicity formulas, and three-Givens charts.
use crate::diagnostics::prof;
pub(crate) use crate::problem::{
    PERMS24, PLANES, Problem, SpectrumKind, StratumSignature, compensated_sum, esym4,
};
use crate::{C, ComplexMatrix as Mat4};
pub(crate) mod interior;
mod klein;
mod radical;
mod support_strata;
mod three_givens;
use crate::problem;
pub(crate) use interior::{
    bernstein_variations, perm_vertex_residual, recover_frame, solve_boundary_accelerators,
};

/// Reorient a candidate after swapping the two input spectra.
fn apply_transpose(frame: Mat4, transpose: bool) -> Mat4 {
    if transpose { frame.transpose() } else { frame }
}

/// Check a constructed frame before using its coefficient residual.
fn certify_frame_against_targets(
    mut frame: Mat4,
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]],
) -> Option<(Mat4, f64)> {
    if frame
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.im.abs() > FRAME_ACCEPT)
    {
        return None;
    }
    let real = frame.map(|z| z.re);
    let determinant = real.determinant();
    let gram = (real.transpose() * real - nalgebra::Matrix4::identity()).amax();
    if !determinant.is_finite()
        || !gram.is_finite()
        || gram > FRAME_ACCEPT
        || (determinant.abs() - 1.0).abs() > FRAME_ACCEPT
    {
        return None;
    }
    if determinant < 0.0 {
        frame.column_mut(0).neg_mut();
    }
    let residual = targets
        .iter()
        .map(|target| compound_residual(dc, lam, &frame, target))
        .fold(f64::INFINITY, f64::min);
    (residual <= ACCEPT).then_some((frame, residual))
}

/// Normalize the orientation, then check the frame once against the original
/// roots. Keep the verified eigenbasis for endpoint reconstruction.
pub(crate) fn compiler_solution(
    problem: &Problem,
    o: Mat4,
    rung: Rung,
    residual: f64,
) -> Option<Solution> {
    if !residual.is_finite()
        || o.iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.im.abs() > FRAME_ACCEPT)
    {
        return None;
    }
    let mut real = o.map(|z| z.re);
    let determinant = real.determinant();
    if !determinant.is_finite()
        || (determinant.abs() - 1.0).abs() > FRAME_ACCEPT
        || (real.transpose() * real - nalgebra::Matrix4::identity()).amax() > FRAME_ACCEPT
    {
        return None;
    }
    if determinant < 0.0 {
        real.column_mut(0).neg_mut();
    }
    let certify = |real| {
        let state = crate::spectral::verify(problem, &real)?;
        Some(Solution {
            o: real,
            rung,
            residual: state.error,
            state,
        })
    };
    if let Some(solution) = certify(real) {
        return Some(solution);
    }
    // Refine a constructed witness before discarding it under the strict
    // spectral tolerance. Rotations preserve its existing frame accuracy.
    if let Some(refined) = problem.iterate(real, 0.0)
        && let Some(solution) = certify(refined)
    {
        return Some(solution);
    }
    let roots = problem.target_roots[0];
    if !(0..4).any(|i| (i + 1..4).any(|j| (roots[i] - roots[j]).norm() < 1e-3)) {
        return None;
    }
    let o = real.map(|x| C::new(x, 0.0));
    let master = Mat4::from_fn(|i, j| {
        let mut entry = C::default();
        for k in 0..4 {
            entry += (o[(i, k)] * problem.right[k]) * o[(j, k)];
        }
        problem.dc[(i, i)] * entry * problem.dc[(j, j)]
    });
    let inverse_dc = problem.dc.map(|z| z.conj());
    for roots in &problem.target_roots {
        let Some(target_master) = klein::retarget_symmetric(&master, roots) else {
            continue;
        };
        let peeled = inverse_dc * target_master * inverse_dc;
        let Some(mut real) = klein::takagi_real(&peeled, &problem.right) else {
            continue;
        };
        if real.determinant() < 0.0 {
            real.column_mut(0).neg_mut();
        }
        if let Some(solution) = certify(real) {
            return Some(solution);
        }
    }
    None
}

/// Try the bounded algebraic constructions before numerical recovery.
pub(crate) fn solve(problem: &Problem) -> Option<Solution> {
    solve_scalar_factor(problem)
        .or_else(|| solve_rank_one_31(problem))
        .or_else(|| solve_prefix(problem))
}

#[inline]
fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

/// One signed permutation frame `P ∈ SO(4)`: the permutation matrix `P_{i,p_i}=1`
/// with row 0 negated when needed to force `det = +1`. The sign squares away in the
/// spectrum (`PΛPᵀ` diagonal = `Λ` permuted), so it matters only for the emitted frame.
pub(crate) fn signed_perm(p: [usize; 4]) -> Mat4 {
    let mut m = Mat4::zeros();
    for (i, &pi) in p.iter().enumerate() {
        m[(i, pi)] = c(1.0, 0.0);
    }
    // det of a permutation matrix = sign = (-1)^inversions; negate row 0 if odd to force +1.
    let inversions = (0..4)
        .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
        .filter(|&(i, j)| p[i] > p[j])
        .count();
    if inversions % 2 == 1 {
        for j in 0..4 {
            m[(0, j)] = -m[(0, j)];
        }
    }
    m
}

/// Left-multiply by a Givens rotation, touching only the two affected rows.
pub(crate) fn rotate_rows(o: &mut Mat4, i: usize, j: usize, cosine: f64, sine: f64) {
    for column in 0..4 {
        let (x, y) = (o[(i, column)], o[(j, column)]);
        o[(i, column)] = cosine * x - sine * y;
        o[(j, column)] = sine * x + cosine * y;
    }
}

/// Elementary symmetric functions `e₁..e₄` of `eig(A)` via Newton's identities on
/// traces -- no eigendecomposition, so it does not floor at spectrum degeneracy.
/// Production uses the matrix-free `compound_residual`; this is the test-side
/// reference implementation.
#[cfg(test)]
fn symfn(a: &Mat4) -> [C; 4] {
    let a2 = a * a;
    let a3 = a2 * a;
    let (p1, p2, p3, p4) = (a.trace(), a2.trace(), a3.trace(), (a3 * a).trace());
    let e1 = p1;
    let e2 = (e1 * p1 - p2) / 2.0;
    let e3 = (e2 * p1 - e1 * p2 + p3) / 3.0;
    let e4 = (e3 * p1 - e2 * p2 + e1 * p3 - p4) / 4.0;
    [e1, e2, e3, e4]
}

/// The sandwich Makhlin matrix `M = D_C·O·Λ·Oᵀ·D_C` for a real frame `O` (passed as
/// `Mat4` with zero imaginary part). `dc = mb(Can(C))`, `lam = mb(Can(G))²`.
#[cfg(test)]
fn mmat(dc: &Mat4, lam: &Mat4, o: &Mat4) -> Mat4 {
    dc * o * lam * o.transpose() * dc
}

/// Cauchy–Binet moment residual for a real orthogonal frame.
/// In dimension four, Jacobi's identity pairs complementary squared minors:
/// 18 minors suffice for e2. See docs/research.md#complementary-minors-in-dimension-four.
pub(crate) fn compound_residual(dc: &Mat4, lam: &Mat4, o: &Mat4, target: &[C; 4]) -> f64 {
    // a_i = dc_{ii}^2 (complex), lv_j = lam_{jj} (complex), O real (zero imag entries).
    let a: [C; 4] = std::array::from_fn(|i| {
        let d = dc[(i, i)];
        d * d
    });
    let lv: [C; 4] = std::array::from_fn(|j| lam[(j, j)]);
    let or_: [[f64; 4]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| o[(i, j)].re));

    // e1 = sum_{i,j} a_i * or[i][j]^2 * lv_j  (16 terms)
    let mut e1_terms = [C::default(); 16];
    let mut term_index = 0;
    for i in 0..4 {
        for j in 0..4 {
            e1_terms[term_index] = a[i] * lv[j] * (or_[i][j] * or_[i][j]);
            term_index += 1;
        }
    }
    let e1 = compensated_sum(e1_terms);

    // PLANES[5-i] complements PLANES[i]. Choose the three row pairs
    // containing row 0; each represents one complementary pair of terms.
    let a_pairs = PLANES.map(|(i, j)| a[i] * a[j]);
    let b_pairs = PLANES.map(|(i, j)| lv[i] * lv[j]);
    let e2_terms: [C; 18] = std::array::from_fn(|n| {
        let (i, j) = (n / 6, n % 6);
        let (i0, i1) = PLANES[i];
        let (j0, j1) = PLANES[j];
        let minor = or_[i0][j0] * or_[i1][j1] - or_[i0][j1] * or_[i1][j0];
        let weight = a_pairs[i] * b_pairs[j] + a_pairs[5 - i] * b_pairs[5 - j];
        weight * (minor * minor)
    });
    let e2 = compensated_sum(e2_terms);

    // Reduced certificate: max(|Δe1|, |Re(Δe2/s)|). Equal to direct evaluation because
    // |e4|=1 forces |De3|=|De1| and De4=0, and e2/s ∈ R so |Re(De2/s)| = |De2| exactly.
    let de1 = e1 - target[0];
    let s = target[3].sqrt();
    let de2_s = (e2 - target[1]) / s;
    de1.norm().max(de2_s.re.abs())
}

/// 3-Givens chart frame `O = G(planes₂,θ_z)·G(planes₁,θ_y)·G(planes₀,θ_x)·P`, with
/// `θ(v)=arccos(√v)`, `v=cos²θ ∈ [0,1]` (s13 chart). `e₁,e₂,e₃` of `M` are trilinear
/// in `(x,y,z)` here -- the structure the deg-6 companion eigensolve exploits.
pub(crate) fn chart_o(xyz: [f64; 3], planes: [(usize, usize); 3], perm: [usize; 4]) -> Mat4 {
    let mut o = signed_perm(perm);
    for (k, &(i, j)) in planes.iter().enumerate() {
        let theta = xyz[k].clamp(0.0, 1.0).sqrt().acos();
        let (sine, cosine) = theta.sin_cos();
        rotate_rows(&mut o, i, j, cosine, sine);
    }
    o
}

/// Direct-sqrt chart frame: `cos θ = √v`, `sin θ = √(1−v)` for `θ = arccos(√v)`,
/// two square roots per coordinate, with rotations applied directly to rows.
pub(crate) fn chart_o_sqrt(xyz: [f64; 3], planes: [(usize, usize); 3], perm: [usize; 4]) -> Mat4 {
    let mut o = signed_perm(perm);
    for (k, &(i, j)) in planes.iter().enumerate() {
        let v = xyz[k].clamp(0.0, 1.0);
        let ct = v.sqrt();
        let st = (1.0 - v).sqrt();
        rotate_rows(&mut o, i, j, ct, st);
    }
    o
}

/// All complex roots of a real polynomial via companion-matrix eigenvalues
/// (nalgebra). Used by bounded algebraic constructions outside the hot
/// three-Givens atlas. `coeffs` are low-to-high; trailing near-zero leading
/// terms are trimmed.
pub(crate) fn poly_roots(coeffs: &[f64]) -> Vec<C> {
    // Trim leading coeffs that are tiny RELATIVE to the largest -- a near-zero lead (e.g. the sin2
    // closure factor 1−γ² → 0 at γ=±1) would otherwise make the companion entries blow up.
    let scale = coeffs
        .iter()
        .fold(0.0_f64, |m, &c| m.max(c.abs()))
        .max(1e-300);
    let mut hi = coeffs.len();
    while hi > 1 && coeffs[hi - 1].abs() < 1e-13 * scale {
        hi -= 1;
    }
    let a = &coeffs[..hi];
    if a.len() < 2 {
        return vec![];
    }
    let n = a.len() - 1;
    let lead = a[n];
    // companion: subdiagonal 1, last column = -a_i/lead -> char poly = a(x)/lead.
    let comp = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| {
        if j == n - 1 {
            -a[i] / lead
        } else if i == j + 1 {
            1.0
        } else {
            0.0
        }
    });
    nalgebra::Schur::try_new(comp, f64::EPSILON, 64)
        .map(|schur| schur.complex_eigenvalues().iter().copied().collect())
        .unwrap_or_default()
}

/// Which bounded construction produced the certified frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    /// Signed permutation or scalar-factor solution.
    Vertex,
    /// Rotation in one coordinate plane.
    Edge,
    /// Rotations in two disjoint coordinate planes.
    Face,
    /// Exact rank-one spectral-measure selector when either factor is `3+1`.
    RankOne31,
    /// Certified candidate from the near-`3+1` rank-one limit. The formula is
    /// not a theorem on this stratum; the unchanged forward certificate is
    /// mandatory.
    NearRankOne31,
    /// Three-Givens chart, including inverse-factor transport.
    Interior,
    /// Klein-circulant one-sided chart: `e₁` linear in the orthostochastic
    /// diagonal pins a line in the simplex, `e₂` collapses to one real quadratic.
    Klein,
    /// Repeated spectrum solved by rank-secular formulas.
    Radical,
    /// Certified result from bounded Levenberg–Marquardt refinement or restarts.
    Numerical,
}

/// A verified real frame, its construction, and its spectral error.
#[derive(Clone, Copy)]
pub struct Solution {
    pub o: nalgebra::Matrix4<f64>,
    /// Construction that produced the verified frame.
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub rung: Rung,
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub residual: f64,
    pub(crate) state: crate::spectral::State,
}

/// Polynomial residual screen. Final acceptance also requires the rootwise
/// spectral check; clustered roots can amplify coefficient errors.
pub(crate) const ACCEPT: f64 = 1e-13;

/// Public-frame tolerance: the fast compound residual is valid only for a real
/// orthogonal frame.  A rejected accelerator candidate must fall through to
/// the next construction rather than escape the black box.
const FRAME_ACCEPT: f64 = 1e-12;

/// Use the reduced trilinear score only well below coefficient acceptance.
/// Every candidate must still pass the final rootwise verifier.
pub(crate) const FAST_ACCEPT: f64 = 1e-14;

fn solve_scalar_factor(problem: &Problem) -> Option<Solution> {
    let (scalar, other) = if problem.strata.c == SpectrumKind::Scalar4 {
        (problem.left[0], problem.right)
    } else if problem.strata.g == SpectrumKind::Scalar4 {
        (problem.right[0], problem.left)
    } else {
        return None;
    };
    let expected: [C; 4] = other.map(|z| scalar * z);
    let error = problem.match_roots(expected);
    (error <= ACCEPT)
        .then(|| compiler_solution(problem, Mat4::identity(), Rung::Vertex, error))
        .flatten()
}

/// Direct exact `3+1` selector (R0028).
///
/// When `d=diag(a,a,a,c)`, the other factor `B` enters only through the
/// grouped masses of `u=O^T e_4`.  Differentiating the rank-one determinant
/// identity at `a*beta_g` recovers those masses linearly.  This routine is
/// deliberately before the generic cascade: it has no chart variables,
/// root search, or continuous selection.
fn solve_rank_one_31(problem: &Problem) -> Option<Solution> {
    use problem::SpectrumProximity::{Exact, Near};
    for (proximity, tolerance, rung) in [
        (Exact, 64.0 * f64::EPSILON, Rung::RankOne31),
        (Near, 1e-7, Rung::NearRankOne31),
    ] {
        for (distinguished, other, kind, distance, transpose) in [
            (
                problem.left,
                problem.right,
                problem.strata.c,
                problem.strata.c_proximity,
                false,
            ),
            (
                problem.right,
                problem.left,
                problem.strata.g,
                problem.strata.g_proximity,
                true,
            ),
        ] {
            if kind == SpectrumKind::Triple31
                && distance == proximity
                && let Some(hit) =
                    solve_rank_one_side(problem, distinguished, other, transpose, tolerance, rung)
            {
                return Some(hit);
            }
        }
    }
    None
}

fn solve_rank_one_side(
    problem: &Problem,
    distinguished: [C; 4],
    other: [C; 4],
    transpose_result: bool,
    group_tolerance: f64,
    rung: Rung,
) -> Option<Solution> {
    let mut singleton = None;
    let mut repeated = None;
    for i in 0..4 {
        let count = (0..4)
            .filter(|&j| (distinguished[j] - distinguished[i]).norm() <= group_tolerance)
            .count();
        if count == 1 {
            singleton = Some(i);
        } else if count == 3 {
            repeated = Some(i);
        }
    }
    let (singleton, repeated) = (singleton?, repeated?);
    let a = distinguished[repeated];
    let c = distinguished[singleton];
    let delta = c - a;
    if delta.norm() <= 64.0 * f64::EPSILON {
        return None;
    }

    // Group equal eigenvalues of the arbitrary factor.  The characteristic
    // polynomial depends on a group only through the sum of squared masses.
    let mut groups: Vec<(C, Vec<usize>)> = Vec::new();
    for j in 0..4 {
        if let Some((_, indices)) = groups
            .iter_mut()
            .find(|(beta, _)| (other[j] - *beta).norm() <= group_tolerance)
        {
            indices.push(j);
        } else {
            groups.push((other[j], vec![j]));
        }
    }

    let mut candidate = None;
    for target in &problem.targets {
        let p = [
            target[3],
            -target[2],
            target[1],
            -target[0],
            C::new(1.0, 0.0),
        ];
        let mut masses = Vec::with_capacity(groups.len());
        let mut valid = true;
        for (g, (beta, indices)) in groups.iter().enumerate() {
            let z = a * *beta;
            let order = indices.len() - 1;
            let derivative = polynomial_derivative_at(&p, z, order);
            let mut denominator = -delta * *beta;
            for q in 1..=order {
                denominator *= C::new(q as f64, 0.0);
            }
            for (h, (other_beta, other_indices)) in groups.iter().enumerate() {
                if h == g {
                    continue;
                }
                for _ in 0..other_indices.len() {
                    denominator *= a * (*beta - *other_beta);
                }
            }
            if denominator.norm() <= 1e-14 {
                valid = false;
                break;
            }
            let mass = derivative / denominator;
            let scale = 1.0 + mass.norm();
            if mass.im.abs() > 2e-7 * scale || mass.re < -2e-7 || mass.re > 1.0 + 2e-7 {
                valid = false;
                break;
            }
            masses.push(mass.re.clamp(0.0, 1.0));
        }
        if !valid || (masses.iter().sum::<f64>() - 1.0).abs() > 2e-6 {
            continue;
        }

        let mut u = [0.0; 4];
        for ((_, indices), mass) in groups.iter().zip(masses) {
            u[indices[0]] = mass.sqrt();
        }
        let mut pmap = [0usize; 4];
        let mut next = 0usize;
        for row in 0..4 {
            if row == singleton {
                pmap[row] = 3;
            } else {
                pmap[row] = next;
                next += 1;
            }
        }
        let pi = signed_perm(pmap);
        let mut w = [0.0; 4];
        w[3] = 1.0 - u[3];
        for i in 0..3 {
            w[i] = -u[i];
        }
        let mut h = Mat4::identity();
        let denom = w.iter().map(|x| x * x).sum::<f64>();
        if denom <= 1e-14 {
            h = Mat4::identity();
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    let base = if i == j { 1.0 } else { 0.0 };
                    h[(i, j)] = C::new(base - 2.0 * w[i] * w[j] / denom, 0.0);
                }
            }
        }
        let mut jfix = Mat4::identity();
        jfix[(0, 0)] = C::new(-1.0, 0.0);
        let normalized = jfix * h;
        let mut o = pi * normalized;
        if transpose_result {
            o = o.transpose();
        }
        let residual = compound_residual(&problem.dc, &problem.lam, &o, target);
        if residual <= ACCEPT {
            candidate = compiler_solution(problem, o, rung, residual);
            if candidate.is_some() {
                return candidate;
            }
        }
    }
    candidate
}

fn polynomial_derivative_at(coeff: &[C; 5], z: C, order: usize) -> C {
    let mut value = C::default();
    for k in order..=4 {
        let mut factor = 1.0;
        for q in 0..order {
            factor *= (k - q) as f64;
        }
        value += coeff[k] * C::new(factor, 0.0) * z.powu((k - order) as u32);
    }
    value
}

/// Try support, Klein, confluent, and three-Givens constructions in order.
fn solve_prefix(problem: &Problem) -> Option<Solution> {
    let tpre = prof::start();
    // A vertex or one-Givens edge preserves routed eigenvalues a_i*g_j. Test
    // that necessary spectral signature before enumerating support incidences.
    // The gate carries its branch masks into the edge solve, so the certificate
    // is computed only once.
    let teg = prof::start();
    let edge_gate = support_strata::edge_gate(&problem.routed, &problem.target_roots);
    prof::rec(prof::SEG_EDGEGATE, teg);
    let tvx = prof::start();
    if let Some(viable) = edge_gate.as_ref() {
        for p in *PERMS24 {
            // A vertex needs all four routed roots on the same target branch.
            let branch_mask = (0..4).fold(0b11u8, |mask, k| mask & viable[k][p[k]]);
            if branch_mask == 0 {
                continue;
            }
            let r = perm_vertex_residual(&problem.left, &problem.right, &problem.targets, &p);
            if r < ACCEPT
                && let Some(solution) = compiler_solution(problem, signed_perm(p), Rung::Vertex, r)
            {
                return Some(solution);
            }
        }
    }
    prof::rec(prof::SEG_VERTEX, tvx);
    prof::rec(prof::PRELUDE, tpre);
    // Exhaust the lower support strata before a broader section can cannibalize
    // their cheaper, better-conditioned formulas.
    if let Some(viable) = edge_gate.as_ref() {
        let tp = prof::start();
        let hit = support_strata::solve_edge(
            &problem.left,
            &problem.right,
            &problem.dc,
            &problem.lam,
            &problem.targets,
            &problem.routed,
            viable,
            PERMS24,
        );
        prof::rec(prof::EDGE, tp);
        if let Some((o, r)) = hit
            && let Some(solution) = compiler_solution(problem, o, Rung::Edge, r)
        {
            return Some(solution);
        }
    }
    let tp = prof::start();
    let hit = support_strata::solve_face(problem);
    prof::rec(prof::FACE, tp);
    if let Some(solution) = hit {
        return Some(solution);
    }
    // Klein-circulant acceleration: a one-sided section of the same realization
    // relation, with e1 linear in the orthostochastic diagonal and e2 reduced
    // to a real quartic.  This is not a separate completeness branch; it runs
    // here because it cheaply owns a large part of both the generic and
    // confluent populations.
    let tk = prof::start();
    let klein_hit = klein::solve(
        &problem.left,
        &problem.right,
        &problem.routed,
        &problem.targets,
    );
    prof::rec(prof::KLEIN_TOTAL, tk);
    if let Some((o, r)) = klein_hit
        && let Some(solution) = compiler_solution(problem, o, Rung::Klein, r)
    {
        return Some(solution);
    }
    // Try multiplicity formulas after the cheaper support and Klein sections.
    if problem.strata.has_confluence() {
        let started = prof::start();
        let hit = support_strata::solve_radical(
            &problem.left,
            &problem.right,
            &problem.target_roots,
            &problem.dc,
            &problem.lam,
            &problem.targets,
            problem.strata,
        );
        prof::rec(prof::RADICAL_TOTAL, started);
        if let Some((o, r)) = hit
            && let Some(solution) = compiler_solution(problem, o, Rung::Radical, r)
        {
            return Some(solution);
        }
    }
    if let Some((o, residual, rung)) = solve_boundary_accelerators(problem)
        && let Some(solution) = compiler_solution(problem, o, rung, residual)
    {
        return Some(solution);
    }
    None
}

#[test]
fn rejects_invalid_solutions() {
    let problem = Problem::new([0.0; 3], [0.0; 3], [0.0; 3]);
    let mut nonorthogonal = Mat4::identity();
    nonorthogonal[(0, 0)].re += 1e-6;
    let mut complex = Mat4::identity();
    complex[(0, 0)].im = 1e-6;
    let mut nan_imaginary = Mat4::identity();
    nan_imaginary[(0, 0)].im = f64::NAN;
    for frame in [
        nonorthogonal,
        complex,
        nan_imaginary,
        Mat4::repeat(C::new(f64::NAN, 0.0)),
    ] {
        assert!(compiler_solution(&problem, frame, Rung::Interior, 0.0).is_none());
    }
    assert!(compiler_solution(&problem, Mat4::identity(), Rung::Interior, 0.0).is_some());
    let wrong_target = Problem::new([0.0; 3], [0.0; 3], [0.125, 0.0, 0.0]);
    assert!(compiler_solution(&wrong_target, Mat4::identity(), Rung::Interior, 0.0).is_none());
}

#[test]
fn compound_moments_match_matrix_traces() {
    for sample in 0..128 {
        let x = sample as f64;
        let problem = Problem::new(
            [0.25 * x.sin(), 0.0, 0.125 * (2.0 * x).sin()],
            [0.25 * (3.0 * x).sin(), 0.125 * x.sin(), 0.0],
            [0.125 * (5.0 * x).sin(), 0.0, 0.0],
        );
        let mut o = signed_perm(PERMS24[sample % 24]);
        for (k, (i, j)) in PLANES.into_iter().enumerate() {
            let angle = (x * (k + 1) as f64).sin();
            let (sine, cosine) = angle.sin_cos();
            rotate_rows(&mut o, i, j, cosine, sine);
        }
        let actual = symfn(&mmat(&problem.dc, &problem.lam, &o));
        // Newton's trace identities do not use complementary minors.
        for mut target in problem.targets {
            // Isolate e2 so a larger trace residual cannot hide a bad formula.
            target[0] = actual[0];
            let expected = ((actual[1] - target[1]) / target[3].sqrt()).re.abs();
            let reduced = compound_residual(&problem.dc, &problem.lam, &o, &target);
            assert!((reduced - expected).abs() < 2e-13, "sample {sample}");
        }
    }
}
