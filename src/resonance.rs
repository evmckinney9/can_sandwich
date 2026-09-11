//! Double-confluence resonance construction: the inner spectrum carries a
//! doubled value AND the target spectrum carries a doubled value.
//!
//! With `Λ = (λa, μ, μ, λb)` the frame enters only through its two
//! simple-value columns `u, v`:
//! `OΛOᵀ = μI + α·uuᵀ + β·vvᵀ` with `α = λa−μ`, `β = λb−μ`.  A doubled target
//! value `ω` forces `S − ω` to have a two-dimensional kernel, and because
//! `S − ω = (μA² − ω) + α(Au)(Au)ᵀ + β(Av)(Av)ᵀ` with the diagonal part
//! invertible, the whole 2×2 interaction matrix must vanish.  That yields the
//! three resonance conditions
//! `α·Σ pᵢcᵢ/(ω−μcᵢ) = 1`, `β·Σ qᵢcᵢ/(ω−μcᵢ) = 1`, `Σ rᵢcᵢ/(ω−μcᵢ) = 0`,
//! which are LINEAR in `(p, q, r) = (u∘u, v∘v, u∘v)`.  On the unit circle the
//! conjugate of each condition is an identity (`conj F = −μ²F − μ` given
//! `Σp = 1`), so together with the norms, orthogonality and the trace the
//! stratum is a rank-8 linear system in R¹²; what remains is the rank-one
//! compatibility `r∘r = p∘q`.  A zero-pattern section (`r_k = q_k = 0` or
//! `r_k = p_k = 0`) cuts the affine family to two dimensions where two
//! compatibility conics meet in a quartic resultant: radicals end to end.
//! When `ω` collides with a diagonal value `μc_k`, `e_k` is an eigenvector
//! for free (`u_k = v_k = 0`) and the identical construction runs one
//! dimension lower, where the family is a line and each conic a quadratic.
//!
//! Confluence is detected with the same `sqrt(machine-epsilon)` separation as
//! the dispatch spine: algebraically coincident inputs, never near-degeneracy.

use super::{poly_roots, Mat4, ACCEPT, C};
use nalgebra::{DMatrix, DVector};

const COINCIDE: f64 = 1.5e-8;

fn doubled_pairs(s: &[C]) -> Vec<(usize, usize)> {
    let n = s.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if (s[i] - s[j]).norm() < COINCIDE {
                out.push((i, j));
            }
        }
    }
    out
}

/// The resonance linear system on `(p, q, r)` over the active coordinates.
/// Returns a particular solution and a null-space basis, or `None` when the
/// data does not lie on the stratum (inconsistent system).
struct Family {
    base: DVector<f64>,
    null: Vec<DVector<f64>>,
}

fn family(outer: &[C], alpha: C, beta: C, mu: C, omegas: &[C], tau: C) -> Option<Family> {
    let n = outer.len();
    let cols = 3 * n;
    let mut rows: Vec<(Vec<C>, C)> = Vec::with_capacity(4 + 3 * omegas.len());
    let sum_row = |block: usize| {
        let mut v = vec![C::default(); cols];
        for slot in v.iter_mut().skip(block * n).take(n) {
            *slot = C::new(1.0, 0.0);
        }
        v
    };
    rows.push((sum_row(0), C::new(1.0, 0.0)));
    rows.push((sum_row(1), C::new(1.0, 0.0)));
    rows.push((sum_row(2), C::default()));
    for &om in omegas {
        let mut den = vec![C::default(); n];
        for i in 0..n {
            den[i] = om - mu * outer[i];
            if den[i].norm() < 1e-10 {
                return None; // diagonal collision: handled by the reduction
            }
        }
        for (block, scale) in [(0, alpha), (1, beta), (2, C::new(1.0, 0.0))] {
            let mut v = vec![C::default(); cols];
            for i in 0..n {
                v[block * n + i] = scale * outer[i] / den[i];
            }
            let rhs = if block == 2 {
                C::default()
            } else {
                C::new(1.0, 0.0)
            };
            rows.push((v, rhs));
        }
    }
    let mut trace = vec![C::default(); cols];
    for i in 0..n {
        trace[i] = alpha * outer[i];
        trace[n + i] = beta * outer[i];
    }
    rows.push((trace, tau));

    let m = 2 * rows.len();
    let mut a = DMatrix::<f64>::zeros(m, cols);
    let mut b = DVector::<f64>::zeros(m);
    for (ri, (v, rhs)) in rows.iter().enumerate() {
        for (ci, z) in v.iter().enumerate() {
            a[(2 * ri, ci)] = z.re;
            a[(2 * ri + 1, ci)] = z.im;
        }
        b[2 * ri] = rhs.re;
        b[2 * ri + 1] = rhs.im;
    }
    let svd = a.clone().svd(true, true);
    let base = svd.solve(&b, 1e-9).ok()?;
    if (&a * &base - &b).amax() > 1e-8 {
        return None; // off the stratum
    }
    let vt = svd.v_t.as_ref()?;
    let mut null = Vec::new();
    for (k, &sv) in svd.singular_values.iter().enumerate() {
        if sv <= 1e-9 {
            null.push(vt.row(k).transpose());
        }
    }
    // Columns beyond the number of computed singular values (m >= cols here,
    // so the thin SVD already spans all of R^cols).
    if null.is_empty() {
        return None; // a zero-dimensional family cannot satisfy 4 quadrics
    }
    Some(Family { base, null })
}

/// `(r_i² − p_i q_i)` restricted to `x = base + Σ y_a dir_a`, as coefficients
/// on the monomials `1, y_0.., y_a y_b (a<=b)`.
fn conic(base: &DVector<f64>, dirs: &[DVector<f64>], n: usize, i: usize) -> Vec<f64> {
    let p = |v: &DVector<f64>| v[i];
    let q = |v: &DVector<f64>| v[n + i];
    let r = |v: &DVector<f64>| v[2 * n + i];
    let mut out = vec![r(base) * r(base) - p(base) * q(base)];
    for d in dirs {
        out.push(2.0 * r(base) * r(d) - (p(d) * q(base) + p(base) * q(d)));
    }
    for a in 0..dirs.len() {
        for b in a..dirs.len() {
            let (da, db) = (&dirs[a], &dirs[b]);
            if a == b {
                out.push(r(da) * r(da) - p(da) * q(da));
            } else {
                out.push(2.0 * r(da) * r(db) - (p(da) * q(db) + p(db) * q(da)));
            }
        }
    }
    out
}

fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

fn polysub(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = a.to_vec();
    out.resize(out.len().max(b.len()), 0.0);
    for (i, &y) in b.iter().enumerate() {
        out[i] -= y;
    }
    out
}

fn polyval(p: &[f64], x: f64) -> f64 {
    p.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Real intersection points of two plane conics by the degenerate-pencil
/// split: `det(G1 + λG2) = 0` (cubic), decompose the degenerate member into
/// two lines, intersect each with `G2`.  Conic coefficient order is
/// `[1, s, t, s², st, t²]` on homogeneous `(1, s, t)`.
fn pencil_points(c1: &[f64], c2: &[f64]) -> Vec<Vec<f64>> {
    let gm = |c: &[f64]| -> [[f64; 3]; 3] {
        [
            [c[0], 0.5 * c[1], 0.5 * c[2]],
            [0.5 * c[1], c[3], 0.5 * c[4]],
            [0.5 * c[2], 0.5 * c[4], c[5]],
        ]
    };
    let (g1, g2) = (gm(c1), gm(c2));
    let det3 = |m: &[[f64; 3]; 3]| -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    // det(G1 + λ G2) as a cubic in λ by four-point interpolation.
    let mut vals = [0.0f64; 4];
    for (k, lam) in [-1.0f64, 0.0, 1.0, 2.0].iter().enumerate() {
        let mut m = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = g1[i][j] + lam * g2[i][j];
            }
        }
        vals[k] = det3(&m);
    }
    // Newton forward differences on nodes -1, 0, 1, 2.
    let c0 = vals[1];
    let a = vals[2] - vals[1];
    let b = vals[1] - vals[0];
    let cc = (vals[3] - 3.0 * vals[2] + 3.0 * vals[1] - vals[0]) / 6.0;
    // p(λ) = c0 + ((a+b)/2)λ + ((a-b)/2)λ² + cc·(λ³ - λ)
    let coeffs = [c0, 0.5 * (a + b) - cc, 0.5 * (a - b), cc];
    let scale = coeffs.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return Vec::new();
    }
    let roots = poly_roots(&coeffs);
    let mut out = Vec::new();
    for lam in roots {
        if lam.im.abs() > 1e-8 * (1.0 + lam.re.abs()) {
            continue;
        }
        let l = lam.re;
        let mut d = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                d[i][j] = g1[i][j] + l * g2[i][j];
            }
        }
        // Split the degenerate conic D into two lines: B = adj(D) = -ppᵀ,
        // recover the double point p, then D + skew(p) has rank 1 with rows
        // proportional to one line and columns to the other.
        let adj = |m: &[[f64; 3]; 3], i: usize, j: usize| -> f64 {
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            m[j1][i1] * m[j2][i2] - m[j1][i2] * m[j2][i1]
        };
        let mut bdiag = [0.0f64; 3];
        for i in 0..3 {
            bdiag[i] = adj(&d, i, i);
        }
        let (pi, _) = bdiag
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
            .unwrap();
        let lines: [[f64; 3]; 2] = if bdiag[pi].abs() > 1e-13 {
            if -bdiag[pi] < 0.0 {
                // adj(D) = -ppᵀ needs -B_ii >= 0; wrong sign means this
                // pencil member is not a real line pair.
                continue;
            }
            let beta = (-bdiag[pi]).sqrt();
            let p = [
                adj(&d, pi, 0) / beta,
                adj(&d, pi, 1) / beta,
                adj(&d, pi, 2) / beta,
            ];
            let cmat = [
                [d[0][0], d[0][1] + p[2], d[0][2] - p[1]],
                [d[1][0] - p[2], d[1][1], d[1][2] + p[0]],
                [d[2][0] + p[1], d[2][1] - p[0], d[2][2]],
            ];
            let (mut bi, mut bj, mut best) = (0usize, 0usize, 0.0f64);
            for i in 0..3 {
                for j in 0..3 {
                    if cmat[i][j].abs() > best {
                        best = cmat[i][j].abs();
                        bi = i;
                        bj = j;
                    }
                }
            }
            if best < 1e-13 {
                continue;
            }
            [cmat[bi], [cmat[0][bj], cmat[1][bj], cmat[2][bj]]]
        } else {
            // Rank-one member: a double line, read off the largest row.
            let (bi, _) = (0..3)
                .map(|i| (i, d[i][0].abs() + d[i][1].abs() + d[i][2].abs()))
                .max_by(|x, y| x.1.total_cmp(&y.1))
                .unwrap();
            [d[bi], d[bi]]
        };
        for line in &lines {
            // Intersect a·1 + b·s + c·t = 0 with conic c2 (quadratic).
            let (la, lb, lc) = (line[0], line[1], line[2]);
            let (q0, q1, q2, q3, q4, q5) = (c2[0], c2[1], c2[2], c2[3], c2[4], c2[5]);
            let cands: Vec<(f64, f64)> = if lc.abs() >= lb.abs() && lc.abs() > 1e-13 {
                // t = -(a + b s)/c; substitute: quadratic in s.
                let (t0, t1) = (-la / lc, -lb / lc);
                let aa = q3 + q4 * t1 + q5 * t1 * t1;
                let bb = q1 + q2 * t1 + q4 * t0 + 2.0 * q5 * t0 * t1;
                let ccq = q0 + q2 * t0 + q5 * t0 * t0;
                quad_roots(aa, bb, ccq)
                    .into_iter()
                    .map(|sv| (sv, t0 + t1 * sv))
                    .collect()
            } else if lb.abs() > 1e-13 {
                let (s0, s1) = (-la / lb, -lc / lb);
                let aa = q5 + q4 * s1 + q3 * s1 * s1;
                let bb = q2 + q1 * s1 + q4 * s0 + 2.0 * q3 * s0 * s1;
                let ccq = q0 + q1 * s0 + q3 * s0 * s0;
                quad_roots(aa, bb, ccq)
                    .into_iter()
                    .map(|tv| (s0 + s1 * tv, tv))
                    .collect()
            } else {
                continue; // the line at infinity: no affine points
            };
            for (sv, tv) in cands {
                out.push(vec![sv, tv]);
            }
        }
        if !out.is_empty() {
            break; // one real degenerate member carries every common point
        }
    }
    out
}

/// Real roots of `a x² + b x + c`, degenerating to the linear root.
fn quad_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < 1e-13 * (b.abs() + c.abs()).max(1.0) {
        if b.abs() < 1e-300 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    let sq = disc.sqrt();
    // Citardauq for the small root: avoids cancellation.
    let q = -0.5 * (b + b.signum() * sq);
    let mut out = vec![q / a];
    if q.abs() > 1e-300 {
        out.push(c / q);
    }
    out
}

/// Points of the slice satisfying the compatibility conics: dimension 1 uses
/// one quadratic, dimension 2 the two-conic quartic resultant.
fn slice_candidates(
    base: &DVector<f64>,
    dirs: &[DVector<f64>],
    n: usize,
    idxs: &[usize],
) -> Vec<Vec<f64>> {
    match dirs.len() {
        0 => vec![vec![]],
        1 => {
            for &i in idxs {
                let c = conic(base, dirs, n, i); // [c0, c1, c2]
                if c[2].abs() < 1e-13 && c[1].abs() < 1e-13 {
                    continue;
                }
                return poly_roots(&[c[0], c[1], c[2]])
                    .into_iter()
                    .filter(|z| z.im.abs() < 1e-8)
                    .map(|z| vec![z.re])
                    .collect();
            }
            vec![vec![]]
        }
        _ => {
            let (i0, i1) = (idxs[0], idxs[1]);
            let c1 = conic(base, dirs, n, i0); // [1, s, t, s², st, t²]
            let c2 = conic(base, dirs, n, i1);
            // The classical conic-pencil intersection: pick a degenerate
            // member det(G1 + λG2) = 0 (one cubic), split it into two lines,
            // intersect each line with the second conic (one quadratic per
            // line).  Every step is a closed form on machine-accurate
            // coefficients, so the candidates land at ~1e-14 where the
            // resultant's linear back-substitution left ~1e-10 of debris;
            // the resultant path below stays as the structural fallback.
            let mut out = pencil_points(&c1, &c2);
            if !out.is_empty() {
                // Also add the mirrored role (degenerate member split against
                // the FIRST conic) when the primary split found nothing real.
                return out;
            }
            out = pencil_points(&c2, &c1);
            if !out.is_empty() {
                return out;
            }
            // As quadratics in t with polynomial-in-s coefficients.
            let coefs = |c: &[f64]| {
                (
                    vec![c[5]],             // t²
                    vec![c[2], c[4]],       // t
                    vec![c[0], c[1], c[3]], // 1
                )
            };
            let (a2, a1, a0) = coefs(&c1);
            let (b2, b1, b0) = coefs(&c2);
            let d20 = polysub(&polymul(&a2, &b0), &polymul(&a0, &b2));
            let d21 = polysub(&polymul(&a2, &b1), &polymul(&a1, &b2));
            let d10 = polysub(&polymul(&a1, &b0), &polymul(&a0, &b1));
            let res = polysub(&polymul(&d20, &d20), &polymul(&d21, &d10));
            let mut out = Vec::new();
            for z in poly_roots(&res) {
                if z.im.abs() > 1e-8 {
                    continue;
                }
                let s = z.re;
                let den = polyval(&d21, s);
                if den.abs() < 1e-12 {
                    continue;
                }
                out.push(vec![s, -polyval(&d20, s) / den]);
            }
            out
        }
    }
}

/// Assemble the real frame from a compatible `(p, q, r)` point: `u = √p`,
/// `v = r/u`, the doubled-value columns are an orthonormal completion.
fn frame_from(
    x: &DVector<f64>,
    active: &[usize],
    u_slot: usize,
    v_slot: usize,
    mu_slots: (usize, usize),
) -> Option<Mat4> {
    let n = active.len();
    let mut u = [0.0f64; 4];
    let mut v = [0.0f64; 4];
    for (a, &coord) in active.iter().enumerate() {
        let (p, q, r) = (x[a], x[n + a], x[2 * n + a]);
        if p < -1e-9 || q < -1e-9 || (r * r - p * q).abs() > 1e-7 {
            return None;
        }
        let ui = p.max(0.0).sqrt();
        u[coord] = ui;
        v[coord] = if ui > 1e-8 { r / ui } else { q.max(0.0).sqrt() };
    }
    // Deterministic orthonormal completion of {u, v} by Gram-Schmidt over the
    // coordinate vectors.
    let mut cols: Vec<[f64; 4]> = vec![u, v];
    for k in 0..4 {
        if cols.len() == 4 {
            break;
        }
        let mut w = [0.0f64; 4];
        w[k] = 1.0;
        for c in &cols {
            let dot: f64 = (0..4).map(|i| w[i] * c[i]).sum();
            for i in 0..4 {
                w[i] -= dot * c[i];
            }
        }
        let norm: f64 = (0..4).map(|i| w[i] * w[i]).sum::<f64>().sqrt();
        if norm > 1e-4 {
            for wi in &mut w {
                *wi /= norm;
            }
            cols.push(w);
        }
    }
    if cols.len() < 4 {
        return None;
    }
    let mut o = [[0.0f64; 4]; 4];
    for i in 0..4 {
        o[i][u_slot] = cols[0][i];
        o[i][v_slot] = cols[1][i];
        o[i][mu_slots.0] = cols[2][i];
        o[i][mu_slots.1] = cols[3][i];
    }
    // det via explicit 4×4 expansion on the real array.
    let det = {
        let m = &o;
        let det3 = |r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize| {
            m[r0][c0] * (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1])
                - m[r0][c1] * (m[r1][c0] * m[r2][c2] - m[r1][c2] * m[r2][c0])
                + m[r0][c2] * (m[r1][c0] * m[r2][c1] - m[r1][c1] * m[r2][c0])
        };
        m[0][0] * det3(1, 2, 3, 1, 2, 3) - m[0][1] * det3(1, 2, 3, 0, 2, 3)
            + m[0][2] * det3(1, 2, 3, 0, 1, 3)
            - m[0][3] * det3(1, 2, 3, 0, 1, 2)
    };
    if det < 0.0 {
        for row in &mut o {
            row[mu_slots.0] = -row[mu_slots.0];
        }
    }
    Some(Mat4::from_fn(|i, j| C::new(o[i][j], 0.0)))
}

/// One oriented instance: `spec(A O Λ Oᵀ A) = w` with `A² = diag(outer)`,
/// `Λ = diag(inner)`.  Yields every section candidate to `accept`.
fn construct(
    outer4: &[C; 4],
    inner4: &[C; 4],
    w4: &[C; 4],
    accept: &mut impl FnMut(Mat4) -> bool,
) -> bool {
    let inner_pairs = doubled_pairs(inner4);
    let w_pairs = doubled_pairs(w4);
    if inner_pairs.is_empty() || w_pairs.is_empty() {
        return false;
    }
    for &(m0, m1) in &inner_pairs {
        let mu = inner4[m0];
        let rest: Vec<usize> = (0..4).filter(|&k| k != m0 && k != m1).collect();
        let (la, lb) = (inner4[rest[0]], inner4[rest[1]]);
        let (alpha, beta) = (la - mu, lb - mu);
        if alpha.norm() < 1e-10 || beta.norm() < 1e-10 {
            continue; // triple confluence: owned by the 1+3 formulas
        }
        let tau4: C = w4.iter().sum::<C>() - mu * outer4.iter().sum::<C>();

        // Diagonal collisions ω = μ·c_k: e_k is an eigenvector for free and
        // the construction descends to the complementary 3-space.
        for &(w0, _) in &w_pairs {
            let om = w4[w0];
            for k in 0..4 {
                if (om - mu * outer4[k]).norm() >= COINCIDE {
                    continue;
                }
                let active: Vec<usize> = (0..4).filter(|&i| i != k).collect();
                let mut wrem: Vec<C> = Vec::with_capacity(3);
                for (i, &wv) in w4.iter().enumerate() {
                    if i != w0 {
                        wrem.push(wv);
                    }
                }
                let rem_pairs = doubled_pairs(&wrem);
                let Some(&(r0, _)) = rem_pairs.first() else {
                    continue;
                };
                let outer3: Vec<C> = active.iter().map(|&i| outer4[i]).collect();
                let tau3: C = wrem.iter().sum::<C>() - mu * outer3.iter().sum::<C>();
                let omegas = [wrem[r0]];
                if let Some(fam) = family(&outer3, alpha, beta, mu, &omegas, tau3) {
                    if run_sections(&fam, &active, rest[0], rest[1], (m0, m1), accept) {
                        return true;
                    }
                }
            }
        }

        // Generic branch: resonance at every doubled target value at once.
        let omegas: Vec<C> = w_pairs.iter().map(|&(w0, _)| w4[w0]).collect();
        let active: Vec<usize> = (0..4).collect();
        let mut omega_sets: Vec<Vec<C>> = vec![omegas.clone()];
        if omegas.len() > 1 {
            // A doubly-doubled target may still be reachable through a single
            // resonance if the joint system is (numerically) inconsistent.
            omega_sets.extend(omegas.iter().map(|&om| vec![om]));
        }
        for oms in &omega_sets {
            if let Some(fam) = family(outer4, alpha, beta, mu, oms, tau4) {
                if run_sections(&fam, &active, rest[0], rest[1], (m0, m1), accept) {
                    return true;
                }
            }
        }
    }
    false
}

/// Enumerate the bounded zero-pattern sections of one affine family and hand
/// every algebraic candidate to `accept`.
fn run_sections(
    fam: &Family,
    active: &[usize],
    u_slot: usize,
    v_slot: usize,
    mu_slots: (usize, usize),
    accept: &mut impl FnMut(Mat4) -> bool,
) -> bool {
    let n = active.len();
    let nd = fam.null.len();
    // Each section: extra linear conditions on y, then the compatibility
    // conics of the remaining coordinates.
    let mut sections: Vec<(Vec<usize>, Vec<usize>)> = Vec::new(); // (zeroed x-cols, conic idxs)
    if nd <= 2 {
        sections.push((vec![], (0..n).collect()));
    } else {
        for k in 0..n {
            for which in [n, 0] {
                // r_k = 0 plus q_k = 0 (or p_k = 0)
                sections.push((
                    vec![2 * n + k, which + k],
                    (0..n).filter(|&i| i != k).collect(),
                ));
            }
        }
    }
    for (zeroed, idxs) in &sections {
        let (base, dirs) = if zeroed.is_empty() {
            (fam.base.clone(), fam.null.clone())
        } else {
            let m = zeroed.len();
            let mut cmat = DMatrix::<f64>::zeros(m, nd);
            let mut rhs = DVector::<f64>::zeros(m);
            for (ri, &col) in zeroed.iter().enumerate() {
                for (ci, nv) in fam.null.iter().enumerate() {
                    cmat[(ri, ci)] = nv[col];
                }
                rhs[ri] = -fam.base[col];
            }
            let svd = cmat.clone().svd(true, true);
            let Ok(yp) = svd.solve(&rhs, 1e-10) else {
                continue;
            };
            if (&cmat * &yp - &rhs).amax() > 1e-8 {
                continue;
            }
            // Kernel of the wide section matrix = orthocomplement of its row
            // space (Gram-Schmidt over the coordinate directions).
            let mut span: Vec<DVector<f64>> = Vec::new();
            for ri in 0..cmat.nrows() {
                let mut r = cmat.row(ri).transpose();
                for s in &span {
                    let d = r.dot(s);
                    r -= s * d;
                }
                if r.norm() > 1e-10 {
                    let nrm = r.norm();
                    span.push(r / nrm);
                }
            }
            let mut w_dirs: Vec<DVector<f64>> = Vec::new();
            for k in 0..nd {
                let mut e = DVector::<f64>::zeros(nd);
                e[k] = 1.0;
                for s in &span {
                    let d = e.dot(s);
                    e -= s * d;
                }
                if e.norm() > 1e-6 {
                    let nrm = e.norm();
                    let e = e / nrm;
                    span.push(e.clone());
                    w_dirs.push(e);
                }
            }
            let mut base = fam.base.clone();
            for (ci, nv) in fam.null.iter().enumerate() {
                base += nv * yp[ci];
            }
            let dirs: Vec<DVector<f64>> = w_dirs
                .iter()
                .take(2)
                .map(|wd| {
                    let mut d = DVector::<f64>::zeros(3 * n);
                    for (ci, nv) in fam.null.iter().enumerate() {
                        d += nv * wd[ci];
                    }
                    d
                })
                .collect();
            (base, dirs)
        };
        for cand in slice_candidates(&base, &dirs, n, idxs) {
            let mut x = base.clone();
            for (c, d) in cand.iter().zip(&dirs) {
                x += d * *c;
            }
            if let Some(o) = frame_from(&x, active, u_slot, v_slot, mu_slots) {
                if accept(o) {
                    return true;
                }
            }
        }
    }
    false
}

/// Entry point mirroring `solve_radical`: both target branches, both factor
/// orientations, forward-certified.
/// Tripled-target dual construction: a target value of multiplicity three
/// makes the symmetric compound `omega·I + (sigma-omega)·zzᵀ` a rank-one
/// perturbation of a scalar, so `O Λ Oᵀ = omega·A⁻² + rho·yyᵀ` is a rank-one
/// update of a DIAGONAL: the classical Cauchy machinery pointed at the
/// TARGET side.  Weights `y_i² = -chi_Λ(d_i)/(rho·chi'_D(d_i))` are closed
/// form, the frame columns are Cauchy vectors `y/(d - λ_k)` (a diagonal
/// collision `d_i = λ_k` contributes the coordinate column for free), and
/// reality is a bounded eight-way sign enumeration judged per candidate.
fn construct_triple(
    outer4: &[C; 4],
    inner4: &[C; 4],
    w4: &[C; 4],
    accept: &mut impl FnMut(Mat4) -> bool,
) -> bool {
    // Identify the tripled value and its simple partner.
    let mut om = None;
    for i in 0..4 {
        let m = (0..4)
            .filter(|&j| (w4[j] - w4[i]).norm() < COINCIDE)
            .count();
        if m == 3 {
            om = Some(w4[i]);
            break;
        }
    }
    let Some(omega) = om else { return false };
    let sigma = *w4
        .iter()
        .find(|&&v| (v - omega).norm() >= COINCIDE)
        .unwrap_or(&omega);
    let rho = sigma - omega;
    if rho.norm() < 1e-10 {
        return false; // scalar target: not this construction's stratum
    }
    let d: [C; 4] = std::array::from_fn(|i| omega / outer4[i]);
    // Distinct outer diagonal is required by the divided-difference weights.
    for i in 0..4 {
        for j in i + 1..4 {
            if (d[i] - d[j]).norm() < COINCIDE {
                return false;
            }
        }
    }
    let chi_inner = |x: C| -> C { inner4.iter().map(|&l| x - l).product() };
    let mut y0 = [C::default(); 4];
    for i in 0..4 {
        let dprime: C = (0..4).filter(|&j| j != i).map(|j| d[i] - d[j]).product();
        let t = -chi_inner(d[i]) / (rho * dprime);
        y0[i] = t.sqrt();
    }
    // Eight sign classes (a global flip leaves every normalized column fixed).
    for mask in 0..8u8 {
        let y: [C; 4] = std::array::from_fn(|i| {
            if i > 0 && mask >> (i - 1) & 1 == 1 {
                -y0[i]
            } else {
                y0[i]
            }
        });
        let mut o = [[0.0f64; 4]; 4];
        let mut imag = 0.0f64;
        let mut ok = true;
        for k in 0..4 {
            let mut coll = None;
            for i in 0..4 {
                if (d[i] - inner4[k]).norm() < 1e-9 {
                    coll = Some(i);
                    break;
                }
            }
            let col: [C; 4] = if let Some(i0) = coll {
                std::array::from_fn(|i| {
                    if i == i0 {
                        C::new(1.0, 0.0)
                    } else {
                        C::default()
                    }
                })
            } else {
                let v: [C; 4] = std::array::from_fn(|i| y[i] / (d[i] - inner4[k]));
                let n2: C = v.iter().map(|z| z * z).sum();
                if n2.norm() < 1e-13 {
                    ok = false;
                    break;
                }
                let n = n2.sqrt();
                std::array::from_fn(|i| v[i] / n)
            };
            for i in 0..4 {
                imag = imag.max(col[i].im.abs());
                o[i][k] = col[i].re;
            }
        }
        if !ok || imag > 1e-7 {
            continue;
        }
        // Modified Gram-Schmidt over the assembled columns: the Cauchy
        // vectors are exactly orthogonal in exact arithmetic, so this
        // closed-form orthonormalization (finitely many square roots and
        // divisions) only removes the float debris of the assembly chain
        // before the unchanged 2e-10 frame gate judges the frame.
        let mut ortho = true;
        for k in 0..4 {
            for j in 0..k {
                let dot: f64 = (0..4).map(|i| o[i][k] * o[i][j]).sum();
                for i in 0..4 {
                    o[i][k] -= dot * o[i][j];
                }
            }
            let norm: f64 = (0..4).map(|i| o[i][k] * o[i][k]).sum::<f64>().sqrt();
            if norm < 1e-6 {
                ortho = false;
                break;
            }
            for i in 0..4 {
                o[i][k] /= norm;
            }
        }
        if !ortho {
            continue;
        }
        if accept(Mat4::from_fn(|i, j| C::new(o[i][j], 0.0))) {
            return true;
        }
    }
    false
}

pub(crate) fn solve(
    c_in: &[C; 4],
    g_in: &[C; 4],
    target_specs: &[[C; 4]; 2],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
) -> Option<(Mat4, f64)> {
    solve_with(c_in, g_in, target_specs, dc, lam, targets, |o, residual| {
        (residual <= ACCEPT).then_some((o, residual))
    })
}

/// Enumerate the finite double-confluence sections with the caller's direct
/// certificate inside the enumeration.  Coefficient residuals are useful for
/// ordering candidates, but are ill-conditioned at the repeated roots this
/// chart owns and therefore cannot decide whether enumeration stops.
pub(crate) fn solve_with<R>(
    c_in: &[C; 4],
    g_in: &[C; 4],
    target_specs: &[[C; 4]; 2],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
    mut finalize: impl FnMut(Mat4, f64) -> Option<R>,
) -> Option<R> {
    let pcg: C = c_in.iter().product::<C>() * g_in.iter().product::<C>();
    for (bi, w) in target_specs.iter().enumerate() {
        if (pcg - w.iter().product::<C>()).norm() > 1e-8 {
            continue;
        }
        let mut hit: Option<R> = None;
        let mut verify = |o: Mat4, transpose: bool| -> bool {
            let oriented = if transpose { o.transpose() } else { o };
            if let Some((o, residual)) =
                super::certify_frame_candidate(oriented, dc, lam, &targets[bi])
            {
                if hit.is_none() {
                    hit = finalize(o, residual);
                }
            }
            hit.is_some()
        };
        // Direct: inner = gate spectrum; transpose: inner = prefix spectrum
        // (`u` solves (C,G,T) iff `uᵀ` solves (G,C,T)).
        if construct(c_in, g_in, w, &mut |o| verify(o, false)) {
            return hit;
        }
        if construct(g_in, c_in, w, &mut |o| verify(o, true)) {
            return hit;
        }
        if construct_triple(c_in, g_in, w, &mut |o| verify(o, false)) {
            return hit;
        }
        if construct_triple(g_in, c_in, w, &mut |o| verify(o, true)) {
            return hit;
        }
    }
    None
}
