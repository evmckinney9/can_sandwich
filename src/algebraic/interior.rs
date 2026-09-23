//! Direct three-Givens charts, their sextic equations and root isolation,
//! including inverse-factor Klein and transported three-Givens constructions.
use super::{
    ACCEPT, C, FAST_ACCEPT, Mat4, PERMS24, Problem, Rung, c, chart_o, chart_o_sqrt,
    compound_residual, esym4, klein, prof, three_givens,
};

/// Monomial corners `(x,y,z) ∈ {0,1}³` in the order used by the
/// trilinear Möbius transform below.
pub(crate) const CORNERS: [[f64; 3]; 8] = [
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 1., 0.],
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 1.],
];

/// Certified separation test: is `t` at distance > `margin` from
/// conv(corners)? A few Frank-Wolfe steps toward the closest hull point
/// propose a separating direction d; exclusion is declared only on the
/// verified certificate  min_i <c_i - t, d> > margin*|d|  (8 dot products,
/// checked exactly every round). Soundness never depends on the search:
/// a missed separation just means no pruning. Bounded 12 rounds, no
/// tolerance semantics beyond the caller's margin.
pub(crate) fn hull_excludes(corners: &[[f64; 3]], t: [f64; 3], margin: f64) -> bool {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut v = [0.0f64; 3];
    let wn = 1.0 / corners.len() as f64;
    for c in corners.iter() {
        for k in 0..3 {
            v[k] += c[k] * wn;
        }
    }
    for _ in 0..12 {
        let d = sub(v, t);
        let dn = dot(d, d).sqrt();
        if dn <= margin {
            return false; // estimate already near t: no separation
        }
        // support of the hull in direction -d, and the certificate check
        let mut smin = f64::INFINITY;
        let mut si = 0;
        for (i, c) in corners.iter().enumerate() {
            let val = dot(sub(*c, t), d);
            if val < smin {
                smin = val;
                si = i;
            }
        }
        if smin > margin * dn {
            return true; // VERIFIED separating hyperplane
        }
        // move v to the closest point to t on segment [v, corners[si]]
        let w = sub(corners[si], v);
        let ww = dot(w, w);
        if ww <= 1e-300 {
            return false;
        }
        let step = (-dot(sub(v, t), w) / ww).clamp(0.0, 1.0);
        if step <= 0.0 {
            return false; // no progress possible: t effectively inside
        }
        for k in 0..3 {
            v[k] += step * w[k];
        }
    }
    false
}

/// The eight corners of every three-Givens chart lie in the same 24-point
/// Weyl orbit. Cache that orbit lazily by the packed permutation itself: a
/// short successful scan pays exactly for the vertices it visits, while an
/// exhaustive scan evaluates each spectral vertex at most once.
pub(crate) fn chart_corners_cached(
    a2: &[C; 4],
    lam: &[C; 4],
    planes: [(usize, usize); 3],
    perm: [usize; 4],
    vertices: &mut [Option<(C, C)>; 256],
) -> ([C; 8], [C; 8]) {
    let mut e1c = [C::new(0.0, 0.0); 8];
    let mut e2c = [C::new(0.0, 0.0); 8];
    for (idx, &xyz) in CORNERS.iter().enumerate() {
        // corner permutation: start from `perm`, apply a transposition for each plane whose
        // angle is π/2 (corner coord == 0, since x=cos²θ). Signs square away in the spectrum.
        let mut c = perm;
        for (k, &(i, j)) in planes.iter().enumerate() {
            if xyz[k] == 0.0 {
                c.swap(i, j);
            }
        }
        let code = c[0] | c[1] << 2 | c[2] << 4 | c[3] << 6;
        let (e1, e2) = *vertices[code].get_or_insert_with(|| {
            let diag: [C; 4] = std::array::from_fn(|jj| a2[jj] * lam[c[jj]]);
            let es = esym4(diag);
            (es[0], es[1])
        });
        e1c[idx] = e1;
        e2c[idx] = e2;
    }
    (e1c, e2c)
}

/// Möbius transform of the Boolean cube in CORNERS order.
pub(crate) fn mobius8(vals: &[C; 8]) -> [C; 8] {
    [
        vals[0],
        vals[1] - vals[0],
        vals[2] - vals[0],
        vals[3] - vals[0],
        vals[4] - vals[1] - vals[2] + vals[0],
        vals[5] - vals[1] - vals[3] + vals[0],
        vals[6] - vals[2] - vals[3] + vals[0],
        vals[7] - vals[4] - vals[5] - vals[6] + vals[1] + vals[2] + vals[3] - vals[0],
    ]
}

/// Real roots of a real cubic in radicals (Cardano/Viete), with exact degree
/// drops. No iteration: the trigonometric branch is closed form.
pub(crate) fn real_roots_cubic(c: &[f64; 4], out: &mut [f64; 3]) -> usize {
    let scale = c.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale == 0.0 {
        return 0;
    }
    let eps = 1e-13 * scale;
    if c[3].abs() <= eps {
        let (a, b, k) = (c[2], c[1], c[0]);
        if a.abs() <= eps {
            if b.abs() <= eps {
                return 0;
            }
            out[0] = -k / b;
            return 1;
        }
        let d = b * b - 4.0 * a * k;
        if d < 0.0 {
            return 0;
        }
        let q = -0.5 * (b + if b >= 0.0 { d.sqrt() } else { -d.sqrt() });
        out[0] = q / a;
        out[1] = if q.abs() > 0.0 { k / q } else { out[0] };
        return 2;
    }
    let (a2, a1, a0) = (c[2] / c[3], c[1] / c[3], c[0] / c[3]);
    let sh = a2 / 3.0;
    let p = a1 - a2 * a2 / 3.0;
    let q = 2.0 * a2 * a2 * a2 / 27.0 - a2 * a1 / 3.0 + a0;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    if disc > 0.0 {
        let sd = disc.sqrt();
        out[0] = (-q / 2.0 + sd).cbrt() + (-q / 2.0 - sd).cbrt() - sh;
        1
    } else {
        let r = (-p / 3.0).max(0.0).sqrt();
        let th = if r > 0.0 {
            (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };
        for k in 0..3 {
            out[k] = 2.0 * r * ((th + 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos() - sh;
        }
        3
    }
}

/// The components of the ADMISSIBLE region `{x ∈ [0,1] : y(x), z(x) ∈ [0,1]}`.
///
/// The kernel of `A(x)` is the vector of signed maximal minors -- the `n` that
/// `eliminate` already computes -- so recovery is RATIONAL in x:
/// `y = n₁/n₀`, `z = n₂/n₀`, ratios of cubics. Hence the admissible set is cut
/// by the five cubics `n₀, n₁, n₀−n₁, n₂, n₀−n₂`, whose roots are radical.
/// Restricting the sextic's root test to this set instead of all of `[0,1]` is
/// what turns a 12.7% decline rate into 66.7%.
pub(crate) fn admissible_components(n: &[[f64; 4]; 4], out: &mut [(f64, f64); 8]) -> usize {
    let sub = |a: &[f64; 4], b: &[f64; 4]| -> [f64; 4] { std::array::from_fn(|i| a[i] - b[i]) };
    // The box must be EXACTLY the one `recover` accepts downstream: production
    // keeps y,z in [-1e-6, 1.0001], and cutting the region at [0,1] instead
    // silently rejected five real linspace solutions (caught by the
    // transported cyclic-boundary charts picking them up).
    const LO: f64 = -1e-6;
    const HI: f64 = 1.0001;
    let scal = |a: &[f64; 4], t: f64| -> [f64; 4] { std::array::from_fn(|i| a[i] * t) };
    let mut bp = [0.0f64; 24];
    let (mut nb, mut r) = (2usize, [0.0f64; 3]);
    bp[0] = 0.0;
    bp[1] = 1.0;
    for c in [
        n[0],
        sub(&n[1], &scal(&n[0], LO)),
        sub(&n[1], &scal(&n[0], HI)),
        sub(&n[2], &scal(&n[0], LO)),
        sub(&n[2], &scal(&n[0], HI)),
    ]
    .iter()
    {
        let k = real_roots_cubic(c, &mut r);
        for &v in r[..k].iter() {
            if v > 1e-13 && v < 1.0 - 1e-13 && nb < 24 {
                bp[nb] = v;
                nb += 1;
            }
        }
    }
    bp[..nb].sort_by(f64::total_cmp);
    let ev = |c: &[f64; 4], t: f64| c[0] + t * (c[1] + t * (c[2] + t * c[3]));
    let mut m = 0usize;
    for w in 0..nb - 1 {
        let (lo, hi) = (bp[w], bp[w + 1]);
        if hi - lo < 1e-13 {
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let d0 = ev(&n[0], mid);
        let keep = if d0.abs() < 1e-12 {
            true // indeterminate: keep, so the gate stays one-sided
        } else {
            let (y, z) = (ev(&n[1], mid) / d0, ev(&n[2], mid) / d0);
            (LO - 1e-9..=HI + 1e-9).contains(&y) && (LO - 1e-9..=HI + 1e-9).contains(&z)
        };
        if !keep {
            continue;
        }
        if m > 0 && lo - out[m - 1].1 < 1e-13 {
            out[m - 1].1 = hi;
        } else if m < 8 {
            out[m] = (lo, hi);
            m += 1;
        }
    }
    m
}

/// One-sided root exclusion on `[0,1]`: `false` PROVES the polynomial has no
/// root there, so this can gate a chart without ever rejecting a real solution.
///
/// The Bernstein coefficients on `[0,1]` are `b_j = Σ_{k≤j} C(j,k)/C(n,k)·a_k`,
/// and the variation-diminishing property bounds the roots in `(0,1)` by the
/// sign changes of `b` -- a constant sign is a PROOF of absence. Unlike a Sturm
/// remainder sequence (which is exact only in exact arithmetic and mis-rejects
/// badly in f64 on these eliminants) the Bernstein basis is well conditioned on
/// the unit interval, and a one-sided certificate is all a gate needs.
pub(crate) fn bernstein_variations(a0: &[f64], lo: f64, hi: f64, rel_tol: f64) -> usize {
    const BIN: [[f64; 7]; 7] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 4.0, 6.0, 4.0, 1.0, 0.0, 0.0],
        [1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 0.0],
        [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
    ];
    if !(hi > lo) {
        return 0;
    }
    // restrict to [lo,hi]: Taylor-shift by lo, then scale by (hi-lo); Bernstein
    // on [0,1] of the result is Bernstein on [lo,hi] of the original, so the
    // variation-diminishing proof carries over unchanged.
    let len = a0.len().min(7);
    let mut a = [0.0f64; 7];
    for j in 0..len {
        let mut acc = 0.0;
        let mut pw = 1.0;
        for k in j..len {
            acc += BIN[k][j] * pw * a0[k];
            pw *= lo;
        }
        a[j] = acc;
    }
    let d = hi - lo;
    let mut pw = 1.0;
    for item in a.iter_mut().take(len) {
        *item *= pw;
        pw *= d;
    }
    let scale = a.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale == 0.0 {
        return 1;
    }
    let n = match (0..len).rev().find(|&k| a[k].abs() > rel_tol * scale) {
        Some(k) if k >= 1 => k,
        _ => return 1,
    };
    let mut b = [0.0f64; 7];
    let mut bmax = 0.0f64;
    for j in 0..=n {
        let mut acc = 0.0;
        for k in 0..=j {
            acc += BIN[j][k] / BIN[n][k] * a[k];
        }
        b[j] = acc;
        bmax = bmax.max(acc.abs());
    }
    if bmax == 0.0 {
        return 1;
    }
    let eps = rel_tol * bmax;
    let (mut last, mut changes) = (0.0f64, 0usize);
    for j in 0..=n {
        if b[j].abs() <= eps {
            continue;
        }
        if last != 0.0 && last * b[j] < 0.0 {
            changes += 1;
        }
        last = b[j];
    }
    // #roots in (lo,hi) <= changes, and == changes (mod 2). So changes == 0
    // PROVES no root, and an ODD count PROVES at least one exists.
    changes
}

pub(crate) fn may_have_unit_root(a0: &[f64; 7], lo: f64, hi: f64) -> usize {
    bernstein_variations(a0, lo, hi, 1e-14)
}

/// Try direct charts, inverse-factor Klein, and transported charts.
/// These bounded searches can decline; numerical recovery follows them.
pub(crate) fn solve_boundary_accelerators(problem: &Problem) -> Option<(Mat4, f64, Rung)> {
    let started = prof::start();
    let hit = solve_charts(
        &problem.left,
        &problem.right,
        &problem.target_roots,
        &problem.dc,
        &problem.lam,
        &BOUNDARY_HEAD,
    );
    prof::rec(prof::INTERIOR_TOTAL, started);
    if let Some((o, residual)) = hit {
        return Some((o, residual, Rung::Interior));
    }
    // The transported Klein section solves (conj(C), W -> G), then a real
    // Takagi factor transports its symmetric-unitary output back to C/G.
    // The anchor is a chart choice, not a class datum: both target lifts are
    // enumerated here so the section is representative-independent (the
    // orbit census showed the rho pass was silently doing this enumeration).
    let inverse_prefix: [C; 4] = std::array::from_fn(|j| problem.left[j].conj());
    let inverse_prefix_root: [C; 4] =
        std::array::from_fn(|k| C::from_polar(1.0, -problem.left_phases[k]));
    let anchored_targets = [esym4(problem.right), {
        let e = esym4(problem.right);
        [-e[0], e[1], -e[2], e[3]]
    }];
    let anchor_dc = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&inverse_prefix_root));
    for anchor_roots in &problem.target_roots {
        let anchor_lam = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(anchor_roots));
        let anchor_routed: [[C; 4]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|j| inverse_prefix[i] * anchor_roots[j]));
        if let Some((p, _)) = klein::solve(
            &inverse_prefix,
            anchor_roots,
            &anchor_routed,
            &anchored_targets,
        ) {
            let symmetric = anchor_dc * p * anchor_lam * p.transpose() * anchor_dc;
            if let Some(frame) = klein::takagi_real(&symmetric, &problem.right) {
                let o = Mat4::from_fn(|i, j| C::new(frame[(i, j)], 0.0));
                let residual = problem
                    .targets
                    .iter()
                    .map(|target| compound_residual(&problem.dc, &problem.lam, &o, target))
                    .fold(f64::INFINITY, f64::min);
                if residual < 1e-12 {
                    return Some((o, residual, Rung::Klein));
                }
            }
        }
    }

    solve_transported(problem).map(|(o, residual)| (o, residual, Rung::Interior))
}

/// Reanchor the same three-spectrum equation when the direct charts are
/// ill-conditioned or do not contain a solution. Both target lifts and both
/// input orientations are needed; the recovered frame is checked in the
/// original coordinates.
fn solve_transported(problem: &Problem) -> Option<(Mat4, f64)> {
    let distinct = |s: &[C; 4]| (0..4).all(|i| (0..i).all(|j| (s[i] - s[j]).norm() > 1e-6));
    if !distinct(&problem.left) || !distinct(&problem.right) {
        return None;
    }
    let product = problem.left.iter().product::<C>() * problem.right.iter().product::<C>();
    for w in &problem.target_roots {
        if !distinct(w) || (product - w.iter().product::<C>()).norm() > 1e-8 {
            continue;
        }
        let inverse_w = w.map(|z| z.conj());
        for (phase, target, transpose) in [
            (&problem.right_phases, problem.left.map(|z| z.conj()), true),
            (&problem.left_phases, problem.right.map(|z| z.conj()), false),
        ] {
            let d = phase.map(|p| C::from_polar(1.0, p));
            let left = d.map(|z| z * z);
            let dc = Mat4::from_diagonal(&nalgebra::Vector4::from(d));
            let lam = Mat4::from_diagonal(&nalgebra::Vector4::from(inverse_w));
            let trace = esym4(target)[0];
            let mut ranked = std::array::from_fn::<_, 24, _>(|i| {
                let value: C = (0..4).map(|j| left[j] * inverse_w[PERMS24[i][j]]).sum();
                (i, (value - trace).norm())
            });
            ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
            let order: Vec<_> = ranked.iter().flat_map(|&(i, _)| [(i, 0), (i, 1)]).collect();
            let Some((v, _)) = solve_charts(&left, &inverse_w, &[target], &dc, &lam, &order) else {
                continue;
            };
            let symmetric = dc * v * lam * v.transpose() * dc;
            let diagonal = Mat4::from_diagonal(&nalgebra::Vector4::from(target));
            let frame = recover_frame(&symmetric, &diagonal);
            let o = if transpose { frame.transpose() } else { frame };
            let residual = problem
                .targets
                .iter()
                .map(|target| compound_residual(&problem.dc, &problem.lam, &o, target))
                .fold(f64::INFINITY, f64::min);
            if residual <= ACCEPT {
                return Some((o, residual));
            }
        }
    }
    None
}

const PHI: f64 = 0.618_033_988_75;

/// Real orthonormal eigenvectors of a symmetric unitary `U₁` (columns), via the real symmetric
/// `Re(U₁)+φ·Im(U₁)`. The theorem: each spectral projector of a symmetric unitary is Hermitian AND
/// symmetric, hence real -> the eigenvectors are real (`[[secular-conditioning-win-realness-free]]`).
pub(crate) fn extract_o(u1: &Mat4) -> nalgebra::Matrix4<f64> {
    let sym = nalgebra::Matrix4::<f64>::from_fn(|i, j| u1[(i, j)].re + PHI * u1[(i, j)].im);
    nalgebra::SymmetricEigen::new(sym).eigenvectors
}

/// Recover the emitted real frame `O ∈ SO(4)` from `U₁`: take its real eigenvectors (`extract_o`),
/// reorder the columns so column `k` carries eigenvalue `lam[k]` (so `O·Λ·Oᵀ = U₁` against the FIXED
/// G gate Λ, not a permuted copy), and sign-fix to `det = +1` (a column sign flip leaves `O·Λ·Oᵀ`
/// invariant since Λ is diagonal). Λ's spectrum (G data) is generically distinct, so the match is a
/// clean bijection even when the TARGET w is degenerate.
// A Sylvester-projector extraction (each column as a rank-one projector, no
// eigensolve) is faster but measurably less accurate on 4x4 unitary folds: the
// runtime gain sits inside the noise floor while machine-precise rows drop.
// The same trade holds in `klein::takagi_real`.  Never weaken tolerances.
pub(crate) fn recover_frame(u1: &Mat4, lam: &Mat4) -> Mat4 {
    let v = extract_o(u1);
    // eigenvalue carried by column k: μ_k = v_kᵀ U₁ v_k (v_k real).
    let mu: [C; 4] = std::array::from_fn(|k| {
        let mut acc = c(0.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                acc += c(v[(i, k)] * v[(j, k)], 0.0) * u1[(i, j)];
            }
        }
        acc
    });
    let mut used = [false; 4];
    let mut o = Mat4::zeros();
    for t in 0..4 {
        let target = lam[(t, t)];
        let mut best = (f64::INFINITY, 0usize);
        for k in 0..4 {
            if !used[k] && (mu[k] - target).norm() < best.0 {
                best = ((mu[k] - target).norm(), k);
            }
        }
        used[best.1] = true;
        for i in 0..4 {
            o[(i, t)] = c(v[(i, best.1)], 0.0);
        }
    }
    if o.map(|z| z.re).determinant() < 0.0 {
        for i in 0..4 {
            o[(i, 0)] = -o[(i, 0)];
        }
    }
    o
}

/// Full reduced spectral distance for a permutation frame. For unit-modulus,
/// determinant-consistent spectra, e3 is determined by e1 and e4 is fixed.
pub(crate) fn perm_vertex_residual(
    a2: &[C; 4],
    lam: &[C; 4],
    targets: &[[C; 4]],
    p: &[usize; 4],
) -> f64 {
    let pr: [C; 4] = std::array::from_fn(|j| a2[j] * lam[p[j]]);
    let e1 = pr[0] + pr[1] + pr[2] + pr[3];
    let p2 = pr[0] * pr[0] + pr[1] * pr[1] + pr[2] * pr[2] + pr[3] * pr[3];
    let e2 = (e1 * e1 - p2) * 0.5;
    targets
        .iter()
        .map(|t| (e1 - t[0]).norm().max((e2 - t[1]).norm()))
        .fold(f64::INFINITY, f64::min)
}

pub(crate) fn ev_poly(co: &[f64], x: f64) -> f64 {
    co.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Certify that a power-basis polynomial has no zero on `[0,1]` by converting
/// it to Bernstein form. A Bernstein polynomial is a convex combination of
/// its coefficients on the unit interval, so coefficients of one strict sign
/// exclude a root. The machine-epsilon guard makes inconclusive cases fall
/// through to sign-change isolation.
pub(crate) fn bernstein_excludes_unit(p: &[f64]) -> bool {
    let mut n = p.len().saturating_sub(1);
    let scale = p.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    if scale == 0.0 {
        return false;
    }
    while n > 0 && p[n].abs() <= 32.0 * f64::EPSILON * scale {
        n -= 1;
    }
    debug_assert!(n <= 8, "detx degree bound");
    // Pascal triangle up to row n (additions only): the naive
    // choose(k,i)/choose(n,i) form costs about 250 float divisions per call.
    let mut ch = [[0.0f64; 9]; 9];
    for k in 0..=n {
        ch[k][0] = 1.0;
        for i in 1..=k {
            ch[k][i] = ch[k - 1][i - 1] + if i < k { ch[k - 1][i] } else { 0.0 };
        }
    }
    let mut w = [0.0f64; 9];
    for i in 0..=n {
        w[i] = p[i] / ch[n][i];
    }
    let mut pos = true;
    let mut neg = true;
    let guard = 128.0 * f64::EPSILON * scale;
    for k in 0..=n {
        let b: f64 = (0..=k).map(|i| ch[k][i] * w[i]).sum();
        pos &= b > guard;
        neg &= b < -guard;
        if !pos && !neg {
            return false;
        }
    }
    pos || neg
}

/// Certified single-root extraction on a bracket, by bounded bisection.
///
/// An ODD Bernstein sign count on `[lo,hi]` proves an odd number of roots there,
/// hence `p(lo)·p(hi) < 0`, hence a bracket already exists -- the sign change IS
/// the certificate. `real_roots_unit` scans a fixed 64-point grid and can miss
/// such a root whenever the component is narrower than a cell, or the cell holds
/// a second root and the two crossings cancel. Recovering it here is what makes
/// "Bernstein proves existence ⟹ the isolator returns a root" actually hold,
/// without weakening any tolerance or substituting a more permissive rooter.
pub(crate) fn refine_bracket(p: &[f64], mut a: f64, mut b: f64) -> Option<f64> {
    let ev = |x: f64| p.iter().rev().fold(0.0f64, |acc, &c| acc * x + c);
    let (mut fa, fb) = (ev(a), ev(b));
    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    if fa.signum() == fb.signum() {
        return None;
    }
    for _ in 0..80 {
        let m = 0.5 * (a + b);
        let fm = ev(m);
        if fm == 0.0 || (b - a) <= 1e-16 * (1.0 + a.abs()) {
            return Some(m);
        }
        if fa.signum() != fm.signum() {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    Some(0.5 * (a + b))
}

pub(crate) fn real_roots_unit(p: &[f64]) -> Vec<f64> {
    if bernstein_excludes_unit(p) {
        return vec![];
    }
    const N: usize = 64;
    let mut out = Vec::new();
    // Batch Horner across the whole grid: the per-coefficient pass vectorizes,
    // where the pointwise fold's serial fma chain costs about 1 us per call.
    // Same evaluation order, bit-identical values.
    let mut xs = [0.0f64; N + 1];
    for (k, x) in xs.iter_mut().enumerate() {
        *x = k as f64 * (1.0 / N as f64); // exact: N a power of two
    }
    let mut vals = [0.0f64; N + 1];
    for &c in p.iter().rev() {
        for k in 0..=N {
            vals[k] = vals[k] * xs[k] + c;
        }
    }
    let (mut xa, mut pa) = (0.0, vals[0]);
    for k in 1..=N {
        let xb = xs[k];
        let pb = vals[k];
        if pa == 0.0 {
            out.push(xa);
        } else if pa.signum() != pb.signum() {
            // Bracketed refinement by Illinois regula falsi: keeps pure
            // bisection's bracket certificate and termination width with
            // about 5x fewer polynomial evaluations.
            // Re-evaluate the bracket endpoints with scalar Horner. The batch
            // grid has a different rounding path; reusing its endpoint values
            // preserves the sign certificate but degrades the final root
            // (92.70% vs 97.57% machine-precise on full Haar).
            let (ra, rb) = (ev_poly(p, xa), ev_poly(p, xb));
            let (mut flo, mut fhi) = if ra != 0.0 && ra.signum() != rb.signum() {
                (ra, rb)
            } else {
                (pa, pb)
            };
            let (mut lo, mut hi) = (xa, xb);
            let mut side = 0i8;
            let mut prev = f64::NAN;
            for _it in 0..80 {
                if hi - lo <= f64::EPSILON * hi.abs().max(0.5) {
                    break;
                }
                let denom = fhi - flo;
                let mut x = if denom != 0.0 {
                    (lo * fhi - hi * flo) / denom
                } else {
                    0.5 * (lo + hi)
                };
                if !(x > lo && x < hi) {
                    x = 0.5 * (lo + hi);
                }
                // The estimate converges long before the bracket width does
                // (regula falsi moves one end slowly); a stationary iterate
                // inside a sign-change bracket IS the converged root.
                if (x - prev).abs() <= f64::EPSILON * x.abs().max(0.5) {
                    lo = x;
                    hi = x;
                    break;
                }
                prev = x;
                let fx = ev_poly(p, x);
                if fx == 0.0 {
                    lo = x;
                    hi = x;
                    break;
                }
                if fx.signum() == flo.signum() {
                    lo = x;
                    flo = fx;
                    if side < 0 {
                        fhi *= 0.5;
                    }
                    side = -1;
                } else {
                    hi = x;
                    fhi = fx;
                    if side > 0 {
                        flo *= 0.5;
                    }
                    side = 1;
                }
            }
            out.push(0.5 * (lo + hi));
        }
        xa = xb;
        pa = pb;
    }
    out
}

/// Plane words used by the direct chart search.
const INTERIOR_PLANES: [[(usize, usize); 3]; 16] = [
    [(0, 1), (1, 2), (2, 3)],
    [(0, 1), (0, 2), (0, 3)],
    [(0, 1), (1, 2), (1, 3)],
    [(0, 2), (0, 3), (1, 2)],
    [(0, 2), (1, 3), (1, 2)],
    [(0, 3), (1, 2), (1, 3)],
    [(0, 2), (0, 3), (1, 3)],
    [(0, 1), (0, 2), (1, 3)],
    [(0, 1), (0, 2), (2, 3)],
    [(0, 1), (0, 3), (1, 2)],
    [(0, 1), (0, 3), (2, 3)],
    [(0, 1), (1, 3), (2, 3)],
    [(0, 2), (1, 2), (2, 3)],
    [(0, 2), (1, 3), (2, 3)],
    [(0, 3), (1, 2), (2, 3)],
    [(0, 3), (1, 3), (2, 3)],
];

#[allow(clippy::too_many_arguments)]
pub(crate) fn rescue_root(
    eliminant: &three_givens::ChartPolynomial,
    res: &[[f64; 8]; 3],
    dc: &Mat4,
    lam: &Mat4,
    target: &[C; 4],
    planes: [(usize, usize); 3],
    perm: [usize; 4],
    o: Mat4,
    r: f64,
) -> (Mat4, f64) {
    if r < 1e-14 {
        return (o, r);
    }
    let Some(roots) = eliminant.qz6_roots() else {
        return (o, r);
    };
    let mut best = (o, r);
    for x in roots {
        for (yv, zv) in three_givens::recover(res, x) {
            if !(-1e-6..=1.0001).contains(&yv) || !(-1e-6..=1.0001).contains(&zv) {
                continue;
            }
            let (y, z) = (yv.clamp(0.0, 1.0), zv.clamp(0.0, 1.0));
            let m8 = [1.0, x, y, z, x * y, x * z, y * z, x * y * z];
            let re1: f64 = res[0].iter().zip(&m8).map(|(a, b)| a * b).sum();
            let im1: f64 = res[1].iter().zip(&m8).map(|(a, b)| a * b).sum();
            let re3: f64 = res[2].iter().zip(&m8).map(|(a, b)| a * b).sum();
            let rs = re1.hypot(im1).max(re3.abs());
            if rs >= best.1 {
                continue;
            }
            let (oc, rc) = if rs < FAST_ACCEPT {
                let oc = chart_o_sqrt([x, y, z], planes, perm);
                (oc, rs)
            } else {
                let oc = chart_o([x, y, z], planes, perm);
                let rc = compound_residual(dc, lam, &oc, target);
                (oc, rc)
            };
            if rc < best.1 {
                best = (oc, rc);
            }
        }
    }
    best
}

/// Direct charts ordered by measured coverage. The schedule is an accelerator,
/// with no completeness claim. Entries index PERMS24 and INTERIOR_PLANES.
pub(crate) const BOUNDARY_HEAD: [(usize, usize); 16] = [
    (14, 7),
    (6, 2),
    (18, 7),
    (7, 3),
    (1, 13),
    (6, 13),
    (3, 3),
    (0, 4),
    (20, 8),
    (0, 8),
    (12, 0),
    (10, 8),
    (21, 3),
    (16, 1),
    (6, 4),
    (13, 6),
];

/// Search an ordered chart schedule. The direct and transported schedules
/// have no repeated cube classes, so they need no dynamic deduplication.
fn solve_charts(
    left: &[C; 4],
    right: &[C; 4],
    targets: &[[C; 4]],
    dc: &Mat4,
    lam: &Mat4,
    order: &[(usize, usize)],
) -> Option<(Mat4, f64)> {
    let mut spectral_vertices = [None; 256];
    for w in targets {
        let detw = w[0] * w[1] * w[2] * w[3];
        let sq = detw.sqrt();
        let w1 = w[0] + w[1] + w[2] + w[3];
        let w2 = w[0] * w[1] + w[0] * w[2] + w[0] * w[3] + w[1] * w[2] + w[1] * w[3] + w[2] * w[3];
        for &(pi, wi) in order {
            let perm = PERMS24[pi];
            let planes = INTERIOR_PLANES[wi];
            let tp = prof::start();
            let (e1c, e2c) =
                chart_corners_cached(left, right, planes, perm, &mut spectral_vertices);
            // HULL GATE (proven necessary, lossless): e1, e2 are multilinear
            // on [0,1]^3, and a multilinear map sends the cube into the convex
            // hull of its 8 corner images (affine in each variable, induction).
            // An accepted frame has |e_k(M) - e_k(target)| <= ACCEPT, so a
            // target outside the corner bounding box, expanded by
            // ACCEPT + trilinear-model FP slack (~1e-11, the trilinearity
            // test's observed bound), cannot be reached from this class.
            const HULL_M: f64 = 1e-8;
            let (mut lo1, mut hi1) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut lo2, mut hi2) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut lo3, mut hi3) = (f64::INFINITY, f64::NEG_INFINITY);
            for i in 0..8 {
                let v1 = e1c[i].re;
                let v2 = e1c[i].im;
                let v3 = (e2c[i] / sq).re;
                lo1 = lo1.min(v1);
                hi1 = hi1.max(v1);
                lo2 = lo2.min(v2);
                hi2 = hi2.max(v2);
                lo3 = lo3.min(v3);
                hi3 = hi3.max(v3);
            }
            let t3 = (w2 / sq).re;
            if w1.re < lo1 - HULL_M
                || w1.re > hi1 + HULL_M
                || w1.im < lo2 - HULL_M
                || w1.im > hi2 + HULL_M
                || t3 < lo3 - HULL_M
                || t3 > hi3 + HULL_M
            {
                prof::rec(prof::CHART_COEFFS, tp);
                prof::rec(prof::GATE_BOX, prof::start());
                continue;
            }
            // Exact-hull separation (same lemma, sharp set): measured on
            // the hard rows it excludes 54-100% of classes where the box
            // excludes none. Certificate-verified, hence lossless.
            let cpts: [[f64; 3]; 8] =
                std::array::from_fn(|i| [e1c[i].re, e1c[i].im, (e2c[i] / sq).re]);
            if hull_excludes(&cpts, [w1.re, w1.im, t3], HULL_M) {
                prof::rec(prof::CHART_COEFFS, tp);
                prof::rec(prof::GATE_HULL, prof::start());
                continue;
            }
            prof::rec(prof::GATE_PASS, prof::start());
            let (a1, a2) = (mobius8(&e1c), mobius8(&e2c));
            prof::rec(prof::CHART_COEFFS, tp);
            // 3 real trilinear residuals; monomial order [1,x,y,z,xy,xz,yz,xyz].
            let mut r1: [f64; 8] = std::array::from_fn(|i| a1[i].re);
            r1[0] -= w1.re;
            let mut r2: [f64; 8] = std::array::from_fn(|i| a1[i].im);
            r2[0] -= w1.im;
            let a2n: [C; 8] = std::array::from_fn(|i| a2[i] / sq);
            let mut r3: [f64; 8] = std::array::from_fn(|i| a2n[i].re);
            r3[0] -= (w2 / sq).re;
            let tp = prof::start();
            // For fixed x, the original residuals are three planes in
            // [1,y,z,yz]. Their cofactor kernel lies on the Segre quadric
            // iff the chart has a solution. This constructs the genuine
            // sextic directly, without the old degree-8 resultant and its
            // numerically fragile quadratic deflation.
            let eliminant = three_givens::eliminate(&[r1, r2, r3]);
            let sextic = eliminant.coefficients;
            prof::rec(prof::ELIM, tp);
            if sextic.iter().all(|c| c.abs() < 1e-12) || !sextic.iter().all(|c| c.is_finite()) {
                continue;
            }
            // Exact eigensolve-free decline: all three eliminants must admit
            // a root in [0,1]. One-sided, so a real solution is never
            // rejected; it only skips the 6x6 QZ on charts that provably
            // cannot hold one.
            // Cheapest decline first: the whole-interval test reuses the
            // sextic already in hand, so the ~13% it rejects never pay for
            // the cubic breakpoints that the sharper region test needs.
            if may_have_unit_root(&sextic, 0.0, 1.0) == 0 {
                continue;
            }
            let mut comps = [(0.0f64, 0.0f64); 8];
            let ncomp = admissible_components(&eliminant.kernel, &mut comps);
            if ncomp == 0 {
                continue;
            }
            // An ODD sign count on a component PROVES a root there; if the
            // rooter then finds none, that is a computational failure, not
            // mathematical nonexistence.
            let mut guaranteed = false;
            let mut possible = false;
            for &(lo, hi) in comps[..ncomp].iter() {
                let ch = may_have_unit_root(&sextic, lo, hi);
                possible |= ch > 0;
                guaranteed |= ch % 2 == 1;
            }
            if !possible {
                continue;
            }

            let tp = prof::start();
            // Taking the proven odd-count bracket first (bisection instead
            // of the degree-6 rooter) is slower on every corpus and degrades
            // the worst residual by 12x: bisection is exact in existence but
            // not in accuracy, so the general rooter stays the primitive.
            let mut roots = real_roots_unit(&sextic);
            prof::rec(prof::ROOTS, tp);
            if roots.is_empty() {
                if !guaranteed {
                    continue;
                }
                // Existence is proven on some component; recover it from the
                // bracket the proof supplies rather than declaring absence.
                for &(lo, hi) in comps[..ncomp].iter() {
                    if may_have_unit_root(&sextic, lo, hi) % 2 == 1
                        && let Some(x) = refine_bracket(&sextic, lo, hi)
                    {
                        roots.push(x);
                    }
                }
                if roots.is_empty() {
                    continue;
                }
            }
            for x in roots.iter().copied() {
                for (yv, zv) in three_givens::recover(&[r1, r2, r3], x) {
                    if !(-1e-6..=1.0001).contains(&yv) || !(-1e-6..=1.0001).contains(&zv) {
                        continue;
                    }
                    let y = yv.clamp(0.0, 1.0);
                    let z = zv.clamp(0.0, 1.0);
                    // Trilinear reduced certificate at the candidate root.
                    // r1/r2/r3 are the already-built [f64;8] monomial coefficient vectors
                    // (order [1,x,y,z,xy,xz,yz,xyz]); evaluating them is exactly the three-real
                    // spectral residual max(hypot(Re De1, Im De1), |Re De2/sq|).
                    let (xy, xz, yz) = (x * y, x * z, y * z);
                    let m8 = [1.0, x, y, z, xy, xz, yz, xy * z];
                    let re1: f64 = r1.iter().zip(&m8).map(|(a, b)| a * b).sum();
                    let im1: f64 = r2.iter().zip(&m8).map(|(a, b)| a * b).sum();
                    let re3: f64 = r3.iter().zip(&m8).map(|(a, b)| a * b).sum();
                    let rs = re1.hypot(im1).max(re3.abs());
                    if rs >= ACCEPT {
                        continue;
                    }
                    let o = if rs < FAST_ACCEPT {
                        chart_o_sqrt([x, y, z], planes, perm)
                    } else {
                        chart_o([x, y, z], planes, perm)
                    };
                    let residual = if rs < FAST_ACCEPT {
                        rs
                    } else {
                        compound_residual(dc, lam, &o, &esym4(*w))
                    };
                    if residual < ACCEPT {
                        return Some(rescue_root(
                            &eliminant,
                            &[r1, r2, r3],
                            dc,
                            lam,
                            &esym4(*w),
                            planes,
                            perm,
                            o,
                            residual,
                        ));
                    }
                }
            }
        }
    }
    None
}
