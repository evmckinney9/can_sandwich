//! T8: the axis-stratum chord construction (spec ch. 3).
//!
//! Solutions on the axis stratum have the exact-zero support form
//! `O = (1 ⊕ O₃) · R_kl(θ)`: one row of `O` supported on a column pair, the
//! complement a 3×3 rotation.  Writing `x = cos²θ` (the chord parameter) and
//! `B̃ = O₃ ∘ O₃` (the 3×3 orthostochastic block, an affine 4-parameter
//! Birkhoff family), the three independent spectral conditions are AFFINE in
//! `B̃` for fixed `x`, so per chart the construction is:
//!   1. the 6×4 real affine system `M(x)·p = b(x)` (closed form below);
//!   2. its rank-3 solution line `p = p₀ + u·d` by a FIXED-PIVOT 3×3 solve
//!      (the pivot pattern is chosen deterministically at one probe point;
//!      the consistency residual of the remaining rows is the chart's
//!      decline gate — cheap for non-axis inputs);
//!   3. the orthostochasticity defect along the line, a known-degree QUARTIC
//!      in `u`, recovered by exact 5-point interpolation (an integer
//!      finite-difference stencil, not a fit);
//!   4. candidate chord values `x*` where the quartic's discriminant changes
//!      sign between adjacent Chebyshev nodes, refined by bracketed
//!      bisection on the actual function (the refine_bracket primitive);
//!   5. per candidate: closed-form quartic roots, `B̃` positivity, row
//!      square-roots with sign matching, and the forward certificate.
//! Every accepted output is certificate-checked; the enumeration is bounded
//! (2 roles × 4 row slots × 6 column supports × 64 nodes).
//!
//! Validated design: s380_chord_exact_design.py, 37/37 on the axis corpus at
//! 8.9e-16..1.2e-13 where the descent tier floors at ~1e-10 (its pullback
//! coefficient assembly).  This module is the additive pre-pass; the descent
//! tier remains the fallback.

use super::{compound_residual, poly_roots, Mat4, ACCEPT, C};

const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
const NCHEB: usize = 64;
/// Consistency gate on the rank-3 line: the remaining rows of the affine
/// system must be satisfied to fp scale, else the chart declines.
const LINE_CONSISTENT: f64 = 1e-7;

/// The three spectral conditions `E(p; x, w)` for the canonical chart
/// (row 0 supported on columns (0, 2)), evaluated at Birkhoff parameters `p`.
fn spectral_conditions(aph: &[C; 4], lam: &[C; 4], wt: &[C; 3], x: f64, p: &[f64; 4]) -> [C; 3] {
    let inv = |z: C| z.conj(); // unimodular inverse
    let l = [inv(lam[0]), inv(lam[1]), inv(lam[2]), inv(lam[3])];
    let ap = [aph[1], aph[2], aph[3]];
    let mu1 = l[0] * x + l[2] * (1.0 - x);
    let mu3 = l[0] * (1.0 - x) + l[2] * x;
    let nu2 = (l[2] - l[0]) * (l[2] - l[0]) * (x * (1.0 - x));
    let (la, lb, lc) = (l[1], mu3, l[3]);
    let (b11, b12, b21, b22) = (p[0], p[1], p[2], p[3]);
    let bt = [
        [b11, b12, 1.0 - b11 - b12],
        [b21, b22, 1.0 - b21 - b22],
        [
            1.0 - b11 - b21,
            1.0 - b12 - b22,
            b11 + b12 + b21 + b22 - 1.0,
        ],
    ];
    let mut diag = [C::default(); 3];
    for m in 0..3 {
        for r in 0..3 {
            diag[m] += ap[r] * bt[r][m];
        }
    }
    let mut m2 = [C::default(); 3];
    for (n, &(i, j)) in PAIRS.iter().enumerate() {
        let mm = 3 - i - j;
        for &(k, kl) in &PAIRS {
            m2[n] += ap[k] * ap[kl] * bt[3 - k - kl][mm];
        }
    }
    let det3 = ap[0] * ap[1] * ap[2];
    let mut out = [C::default(); 3];
    for (n, &w) in wt.iter().enumerate() {
        let d = la * lb * lc - w * (diag[0] * lb * lc + diag[1] * la * lc + diag[2] * la * lb)
            + w * w * (m2[0] * lc + m2[1] * lb + m2[2] * la)
            - w * w * w * det3;
        let cf = la * lc - w * (la * diag[2] + lc * diag[0]) + w * w * m2[1];
        out[n] = (mu1 - w * aph[0]) * d - nu2 * cf;
    }
    out
}

/// The 6×4 realified affine system `M·p = b` at chord `x`.
fn build_system(aph: &[C; 4], lam: &[C; 4], wt: &[C; 3], x: f64) -> ([[f64; 4]; 6], [f64; 6]) {
    let base = spectral_conditions(aph, lam, wt, x, &[0.0; 4]);
    let mut m = [[0.0f64; 4]; 6];
    let mut b = [0.0f64; 6];
    for j in 0..4 {
        let mut e = [0.0f64; 4];
        e[j] = 1.0;
        let col = spectral_conditions(aph, lam, wt, x, &e);
        for n in 0..3 {
            m[2 * n][j] = (col[n] - base[n]).re;
            m[2 * n + 1][j] = (col[n] - base[n]).im;
        }
    }
    for n in 0..3 {
        b[2 * n] = -base[n].re;
        b[2 * n + 1] = -base[n].im;
    }
    (m, b)
}

/// Deterministic rank-3 pivot pattern by greedy elimination on the probe
/// system.  `None` when the system is rank-deficient at the probe (chart
/// degenerate for this data).
fn pick_pivots(m: &[[f64; 4]; 6]) -> Option<([usize; 3], [usize; 3])> {
    let mut a = *m;
    let mut rows = [0usize; 3];
    let mut cols = [0usize; 3];
    for step in 0..3 {
        let (mut br, mut bc, mut bv) = (0usize, 0usize, 0.0f64);
        for (r, row) in a.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v.abs() > bv {
                    (br, bc, bv) = (r, c, v.abs());
                }
            }
        }
        if bv < 1e-12 {
            return None;
        }
        rows[step] = br;
        cols[step] = bc;
        let piv = a[br][bc];
        for r in 0..6 {
            let f = a[r][bc] / piv;
            if r != br {
                for c in 0..4 {
                    a[r][c] -= f * a[br][c];
                }
            }
        }
        for c in 0..4 {
            a[br][c] = 0.0;
        }
    }
    Some((rows, cols))
}

/// Solve the fixed 3×3 pivot subsystem: the line `p = p0 + u·d`, plus the
/// full-system consistency residual.
fn line_from_pivots(
    m: &[[f64; 4]; 6],
    b: &[f64; 6],
    rows: &[usize; 3],
    cols: &[usize; 3],
) -> Option<([f64; 4], [f64; 4], f64)> {
    let free = (0..4).find(|c| !cols.contains(c)).unwrap();
    let mut a3 = [[0.0f64; 3]; 3];
    let mut b3 = [0.0f64; 3];
    let mut f3 = [0.0f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            a3[i][j] = m[rows[i]][cols[j]];
        }
        b3[i] = b[rows[i]];
        f3[i] = -m[rows[i]][free];
    }
    let det = |a: &[[f64; 3]; 3]| -> f64 {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d0 = det(&a3);
    if d0.abs() < 1e-14 {
        return None;
    }
    let solve3 = |rhs: &[f64; 3]| -> [f64; 3] {
        let mut out = [0.0f64; 3];
        for c in 0..3 {
            let mut ac = a3;
            for r in 0..3 {
                ac[r][c] = rhs[r];
            }
            out[c] = det(&ac) / d0;
        }
        out
    };
    let s0 = solve3(&b3);
    let sf = solve3(&f3);
    let mut p0 = [0.0f64; 4];
    let mut d = [0.0f64; 4];
    for k in 0..3 {
        p0[cols[k]] = s0[k];
        d[cols[k]] = sf[k];
    }
    d[free] = 1.0;
    let mut res = 0.0f64;
    for r in 0..6 {
        let mut acc = -b[r];
        for c in 0..4 {
            acc += m[r][c] * p0[c];
        }
        res = res.max(acc.abs());
    }
    Some((p0, d, res))
}

fn birkhoff(p: &[f64; 4]) -> [[f64; 3]; 3] {
    let (b11, b12, b21, b22) = (p[0], p[1], p[2], p[3]);
    [
        [b11, b12, 1.0 - b11 - b12],
        [b21, b22, 1.0 - b21 - b22],
        [
            1.0 - b11 - b21,
            1.0 - b12 - b22,
            b11 + b12 + b21 + b22 - 1.0,
        ],
    ]
}

/// The orthostochasticity defect along the line is a quartic in `u`;
/// recover its (descending) coefficients by the exact 5-point integer
/// finite-difference stencil at u ∈ {-2,-1,0,1,2}.
fn defect_quartic(p0: &[f64; 4], d: &[f64; 4]) -> [f64; 5] {
    let val = |u: f64| -> f64 {
        let p = [
            p0[0] + u * d[0],
            p0[1] + u * d[1],
            p0[2] + u * d[2],
            p0[3] + u * d[3],
        ];
        let bt = birkhoff(&p);
        let urow = [
            bt[0][0] * bt[1][0],
            bt[0][1] * bt[1][1],
            bt[0][2] * bt[1][2],
        ];
        2.0 * (urow[0] * urow[1] + urow[0] * urow[2] + urow[1] * urow[2])
            - (urow[0] * urow[0] + urow[1] * urow[1] + urow[2] * urow[2])
    };
    let (f_m2, f_m1, f_0, f_1, f_2) = (val(-2.0), val(-1.0), val(0.0), val(1.0), val(2.0));
    [
        (f_m2 - 4.0 * f_m1 + 6.0 * f_0 - 4.0 * f_1 + f_2) / 24.0,
        (-f_m2 + 2.0 * f_m1 - 2.0 * f_1 + f_2) / 12.0,
        (-f_m2 + 16.0 * f_m1 - 30.0 * f_0 + 16.0 * f_1 - f_2) / 24.0,
        (f_m2 - 8.0 * f_m1 + 8.0 * f_1 - f_2) / 12.0,
        f_0,
    ]
}

/// Discriminant of the quartic `a u⁴ + b u³ + c u² + d u + e`.
fn quartic_disc(q: &[f64; 5]) -> f64 {
    let (a, b, c, d, e) = (q[0], q[1], q[2], q[3], q[4]);
    256.0 * a.powi(3) * e.powi(3)
        - 192.0 * a.powi(2) * b * d * e.powi(2)
        - 128.0 * a.powi(2) * c.powi(2) * e.powi(2)
        + 144.0 * a.powi(2) * c * d.powi(2) * e
        - 27.0 * a.powi(2) * d.powi(4)
        + 144.0 * a * b.powi(2) * c * e.powi(2)
        - 6.0 * a * b.powi(2) * d.powi(2) * e
        - 80.0 * a * b * c.powi(2) * d * e
        + 18.0 * a * b * c * d.powi(3)
        + 16.0 * a * c.powi(4) * e
        - 4.0 * a * c.powi(3) * d.powi(2)
        - 27.0 * b.powi(4) * e.powi(2)
        + 18.0 * b.powi(3) * c * d * e
        - 4.0 * b.powi(3) * d.powi(3)
        - 4.0 * b.powi(2) * c.powi(3) * e
        + b.powi(2) * c.powi(2) * d.powi(2)
    // (standard closed form; sign conventions cancel in the sign-change test)
}

/// Candidate `O₃` rows from a Birkhoff point: exact square roots with the
/// sign class matched by orthogonality; `None` off the orthogonal locus.
fn rows_from_birkhoff(bt: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut btc = *bt;
    for row in btc.iter_mut() {
        for v in row.iter_mut() {
            if *v < -1e-8 || *v > 1.0 + 1e-8 {
                return None;
            }
            *v = v.clamp(0.0, 1.0);
        }
    }
    let r0 = [btc[0][0].sqrt(), btc[0][1].sqrt(), btc[0][2].sqrt()];
    let m1 = [btc[1][0].sqrt(), btc[1][1].sqrt(), btc[1][2].sqrt()];
    // Sign class by MINIMAL row overlap under a routing bound: the forward
    // certificate is the arbiter, so selection only routes.
    let mut r1 = None;
    let mut best = 1e-3f64;
    for signs in [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
    ] {
        let cand = [signs[0] * m1[0], signs[1] * m1[1], signs[2] * m1[2]];
        let dot = (r0[0] * cand[0] + r0[1] * cand[1] + r0[2] * cand[2]).abs();
        if dot < best {
            best = dot;
            r1 = Some(cand);
        }
    }
    let r1 = r1?;
    let r2 = [
        r0[1] * r1[2] - r0[2] * r1[1],
        r0[2] * r1[0] - r0[0] * r1[2],
        r0[0] * r1[1] - r0[1] * r1[0],
    ];
    Some([r0, r1, r2])
}

/// Assemble the chart-frame `O = (1 ⊕ O₃) · R₀₂(θ)` with `cos²θ = x`.
fn assemble_chart_frame(o3: &[[f64; 3]; 3], x: f64, s_sign: f64) -> [[f64; 4]; 4] {
    let (cth, sth) = (x.sqrt(), s_sign * (1.0 - x).sqrt());
    let mut ot = [[0.0f64; 4]; 4];
    ot[0][0] = 1.0;
    for i in 0..3 {
        for j in 0..3 {
            ot[i + 1][j + 1] = o3[i][j];
        }
    }
    let mut r4 = [[0.0f64; 4]; 4];
    (r4[1][1], r4[3][3]) = (1.0, 1.0);
    (r4[0][0], r4[2][2]) = (cth, cth);
    (r4[0][2], r4[2][0]) = (sth, -sth);
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for (k, r4row) in r4.iter().enumerate() {
                out[i][j] += ot[i][k] * r4row[j];
            }
        }
    }
    out
}

struct ChartData<'a> {
    aph: [C; 4],
    lam: [C; 4],
    wt: &'a [C; 3],
}

/// One chart's bounded solve: probe → pivots → Chebyshev sign scan →
/// bracketed bisection → per-candidate exact construction.  Emits raw
/// chart-frame candidates; the caller maps them back and certifies.
fn solve_chart(data: &ChartData, out: &mut Vec<([[f64; 4]; 4], f64)>) {
    let (mp, bp) = build_system(&data.aph, &data.lam, data.wt, 0.37);
    let Some((rows, cols)) = pick_pivots(&mp) else {
        return;
    };
    // Cheap chart decline before any scan: non-axis inputs are GROSSLY
    // inconsistent (residual O(0.1)) on every chart, so a coarse routing
    // bound suffices here; the per-node LINE_CONSISTENT gate below does the
    // fine work.  A tight probe gate was measured to kill a valid chart
    // whose consistency dips near a pivot degeneracy at the probe x.
    match line_from_pivots(&mp, &bp, &rows, &cols) {
        Some((_, _, res)) if res <= 1e-3 => {}
        _ => return,
    }
    // The system entries are polynomial of degree <= 2 in x (Cf is
    // x-free, D linear, the prefactors linear/quadratic), so a 4-node
    // Newton table is an EXACT representation; the scan evaluates it
    // instead of rebuilding the spectral conditions per node.
    let mut mt = [([[0.0f64; 4]; 6], [0.0f64; 6]); 4];
    for (t, slot) in mt.iter_mut().enumerate() {
        *slot = build_system(&data.aph, &data.lam, data.wt, t as f64 / 3.0);
    }
    let newton = |f: [f64; 4]| -> [f64; 4] {
        let d1 = [f[1] - f[0], f[2] - f[1], f[3] - f[2]];
        let d2 = [d1[1] - d1[0], d1[2] - d1[1]];
        [f[0], d1[0], d2[0] / 2.0, (d2[1] - d2[0]) / 6.0]
    };
    let mut m_nf = [[[0.0f64; 4]; 4]; 6];
    let mut b_nf = [[0.0f64; 4]; 6];
    for r in 0..6 {
        for c in 0..4 {
            m_nf[r][c] = newton([mt[0].0[r][c], mt[1].0[r][c], mt[2].0[r][c], mt[3].0[r][c]]);
        }
        b_nf[r] = newton([mt[0].1[r], mt[1].1[r], mt[2].1[r], mt[3].1[r]]);
    }
    let eval_mb = |x: f64| -> ([[f64; 4]; 6], [f64; 6]) {
        let t = 3.0 * x;
        let horner = |n: &[f64; 4]| n[0] + t * (n[1] + (t - 1.0) * (n[2] + (t - 2.0) * n[3]));
        let mut m = [[0.0f64; 4]; 6];
        let mut b = [0.0f64; 6];
        for r in 0..6 {
            for c in 0..4 {
                m[r][c] = horner(&m_nf[r][c]);
            }
            b[r] = horner(&b_nf[r]);
        }
        (m, b)
    };
    let geval = |x: f64| -> Option<f64> {
        let (m, b) = eval_mb(x);
        let (p0, d, res) = line_from_pivots(&m, &b, &rows, &cols)?;
        if res > LINE_CONSISTENT {
            return None;
        }
        // The pivot minor's determinant clears the Cramer poles from the
        // sign-change test (any positive even power works; the candidates
        // are re-derived and certified at x*, so only the sign structure
        // matters here).
        let mut a3 = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a3[i][j] = m[rows[i]][cols[j]];
            }
        }
        let den = a3[0][0] * (a3[1][1] * a3[2][2] - a3[1][2] * a3[2][1])
            - a3[0][1] * (a3[1][0] * a3[2][2] - a3[1][2] * a3[2][0])
            + a3[0][2] * (a3[1][0] * a3[2][1] - a3[1][1] * a3[2][0]);
        Some(quartic_disc(&defect_quartic(&p0, &d)) * den.powi(12))
    };
    let mut xs = [0.0f64; NCHEB];
    let mut gs = [f64::NAN; NCHEB];
    for k in 0..NCHEB {
        // ascending Chebyshev nodes of [0, 1]
        let theta = (2 * (NCHEB - 1 - k) + 1) as f64 * std::f64::consts::PI / (2 * NCHEB) as f64;
        xs[k] = 0.5 + 0.5 * theta.cos();
        gs[k] = match geval(xs[k]) {
            Some(v) => v,
            None => return, // inconsistent anywhere => not this chart's stratum
        };
    }
    let mut cands: Vec<f64> = Vec::new();
    for k in 0..NCHEB - 1 {
        if gs[k] * gs[k + 1] > 0.0 {
            continue;
        }
        let (mut lo, mut hi, mut flo) = (xs[k], xs[k + 1], gs[k]);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let Some(fm) = geval(mid) else { break };
            if flo * fm <= 0.0 {
                hi = mid;
            } else {
                (lo, flo) = (mid, fm);
            }
        }
        cands.push(0.5 * (lo + hi));
    }
    for xv in cands {
        // full-precision rebuild at the candidate (the scan used the table)
        let (m, b) = build_system(&data.aph, &data.lam, data.wt, xv);
        let Some((p0, d, res)) = line_from_pivots(&m, &b, &rows, &cols) else {
            continue;
        };
        if res > LINE_CONSISTENT {
            continue;
        }
        let q = defect_quartic(&p0, &d);
        for root in poly_roots(&[q[4], q[3], q[2], q[1], q[0]]) {
            if root.im.abs() > 1e-5 {
                continue;
            }
            let u = root.re;
            let p = [
                p0[0] + u * d[0],
                p0[1] + u * d[1],
                p0[2] + u * d[2],
                p0[3] + u * d[3],
            ];
            let Some(o3) = rows_from_birkhoff(&birkhoff(&p)) else {
                continue;
            };
            for s_sign in [1.0, -1.0] {
                out.push((assemble_chart_frame(&o3, xv, s_sign), xv));
            }
        }
    }
}

/// The chord pre-pass over the finite chart orbit.  Returns the first
/// candidate whose forward certificate passes at machine-adjacent scale.
pub(crate) fn solve(
    left: &[C; 4],
    right: &[C; 4],
    target_roots: &[[C; 4]; 2],
    dc: &Mat4,
    lam_m: &Mat4,
    targets: &[[C; 4]; 2],
) -> Option<(Mat4, f64)> {
    let wt: [C; 3] = [
        target_roots[0][0].conj(),
        target_roots[0][1].conj(),
        target_roots[0][2].conj(),
    ];
    let inv = |z: C| z.conj();
    let mut raw: Vec<([[f64; 4]; 4], f64)> = Vec::new();
    // Machine-scale candidates return immediately; a merely-certified one is
    // held while the bounded enumeration finishes (the same hold rule as the
    // dispatch orbit: first-hit-wins must not mask an exact candidate).
    let mut held: Option<(Mat4, f64)> = None;
    for role in 0..2 {
        let (a_role, l_role): ([C; 4], [C; 4]) = if role == 0 {
            (*left, *right)
        } else {
            (
                [inv(right[0]), inv(right[1]), inv(right[2]), inv(right[3])],
                [inv(left[0]), inv(left[1]), inv(left[2]), inv(left[3])],
            )
        };
        for rsel in 0..4 {
            let mut perm_r = [rsel, 0, 0, 0];
            let mut idx = 1;
            for i in 0..4 {
                if i != rsel {
                    perm_r[idx] = i;
                    idx += 1;
                }
            }
            for (k, l) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                let rest: Vec<usize> = (0..4).filter(|i| *i != k && *i != l).collect();
                let perm_c = [k, rest[0], l, rest[1]];
                let aph = std::array::from_fn(|i| a_role[perm_r[i]]);
                let lam = std::array::from_fn(|i| l_role[perm_c[i]]);
                raw.clear();
                solve_chart(&ChartData { aph, lam, wt: &wt }, &mut raw);
                for (oc, _x) in &raw {
                    // map back: Ob[perm_r[i]][perm_c[j]] = Oc[i][j]; role 1
                    // additionally transposes.
                    let mut ob = [[0.0f64; 4]; 4];
                    for i in 0..4 {
                        for j in 0..4 {
                            ob[perm_r[i]][perm_c[j]] = oc[i][j];
                        }
                    }
                    let o = Mat4::from_fn(|i, j| {
                        C::new(if role == 0 { ob[i][j] } else { ob[j][i] }, 0.0)
                    });
                    let residual = targets
                        .iter()
                        .map(|tg| compound_residual(dc, lam_m, &o, tg))
                        .fold(f64::INFINITY, f64::min);
                    if residual < 1e-12 {
                        return Some((o, residual));
                    }
                    if residual < ACCEPT && held.as_ref().is_none_or(|(_, h)| residual < *h) {
                        held = Some((o, residual));
                    }
                }
            }
        }
    }
    held
}
