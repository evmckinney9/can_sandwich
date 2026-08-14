//! Klein-circulant rung: the one-sided radical construction.
//!
//! Restrict the frame to `O = L_q·P`, `L_q` left quaternion multiplication by a
//! unit `q ∈ S³` and `P` a coset representative of `S₄/V` (`V` = the Klein group
//! of ⊕-shifts, normal in `S₄`; right multiplication by `V` is absorbed by
//! relabelling `q`, so these six charts exhaust the family).
//!
//! `L_q` is LINEAR in `q`, so with `x_i = q_i²` the orthostochastic matrix is a
//! Klein convolution `B_jk = x_{j⊕π[k]}` and
//!
//!   `e₁ = Σ_m C_m·x_m`,   `C_m = Σ_k a2[π[k]⊕m]·g2[k]`,
//!
//! is linear in `x`. Together with `Σx = 1` that pins `x` to a LINE in the
//! simplex; the chart admits the target iff the line meets the positive orthant.
//! On the line the only non-polynomial content of `e₂` is the single monomial
//! `q₀q₁q₂q₃ = σ·√D₄(s)`:
//!
//!   `e₂ = P₂(s) + κ·σ·√D₄(s)`,   `P₂` quadratic, `κ` constant, `D₄ = Πx_i(s)`.
//!
//! `M` is unitary with `e₄` pinned by the data, so its characteristic polynomial
//! is self-inversive and `e₂/√e₄` is real for EVERY `O` -- matching `e₂` is ONE
//! real equation, not two. Dividing through by `√e₄` makes `P₂` and `κ` real:
//!
//!   `p(s) = −k·σ·√D₄(s)`,  `p = Re((P₂ − e₂ᵗ)/√e₄)`,  `k = Re(κ/√e₄)`,
//!
//! so squaring gives a REAL quartic `p(s)² = k²·D₄(s)`. `σ` is then read off from
//! the sign of `p·k`, not searched.
use super::{axis_quartic::quartic_roots, Mat4, ACCEPT, C};

/// `L_q[j][m] = SGN[j][m] · q[XOR[j][m]]` -- the left-multiplication matrix of
/// the quaternion `q = q₀ + q₁i + q₂j + q₃k`.
const XOR: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]];
const SGN: [[f64; 4]; 4] = [
    [1.0, -1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0, 1.0],
];

/// One representative per coset of `S₄/V`, with its parity. The parity picks the
/// column that is negated so `O ∈ SO(4)`; that negation leaves
/// `M = D·O·Λ·Oᵀ·D` untouched (a diagonal `±1` commutes with `Λ` and squares
/// away), so it plays no part in the gate or the eliminant.
const REPS: [([usize; 4], bool); 6] = [
    ([0, 1, 2, 3], false),
    ([0, 1, 3, 2], true),
    ([0, 2, 1, 3], true),
    ([0, 2, 3, 1], false),
    ([0, 3, 1, 2], false),
    ([0, 3, 2, 1], true),
];
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

#[inline]
fn build_o(q: &[f64; 4], pi: &[usize; 4], odd: bool) -> [[f64; 4]; 4] {
    std::array::from_fn(|j| {
        std::array::from_fn(|k| {
            let m = pi[k];
            let e = if odd && k == 0 { -1.0 } else { 1.0 };
            e * SGN[j][m] * q[XOR[j][m]]
        })
    })
}

/// Cauchy-Binet split of `e₂(M)` in the Klein chart, exactly: the quadratic form
/// `Q` with `P₂(x) = Σ Q_ab·x_a·x_b`, and the coefficient `κ` of `q₀q₁q₂q₃`.
///
/// Each 2×2 minor is `±q_u q_v ∓ q_u' q_v'` with `u⊕v = u'⊕v' = δ`. Its square
/// contributes `x_u x_v + x_u' x_v'` plus a cross term `−2T·q_u q_v q_u' q_v'`.
/// For `δ ≠ 0` the two pairs are complementary, so the cross term IS `q₀q₁q₂q₃`
/// and lands in `κ`; for `δ = 0` it is `x_u x_u'` and stays in `Q`. Nothing else
/// in `e₂` is odd in any `q_i`, so the split is complete.
fn pair_weights(a2: &[C; 4], g2: &[C; 4]) -> [[C; 6]; 6] {
    std::array::from_fn(|ai| {
        let (i, ip) = PAIRS[ai];
        std::array::from_fn(|gj| {
            let (j, jp) = PAIRS[gj];
            // Preserve the historical left association exactly.
            ((a2[i] * a2[ip]) * g2[j]) * g2[jp]
        })
    })
}

fn split_e2(weights: &[[C; 6]; 6], pi: &[usize; 4]) -> ([[C; 4]; 4], C) {
    let mut q = [[C::new(0.0, 0.0); 4]; 4];
    let mut kap = C::new(0.0, 0.0);
    for (ai, &(i, ip)) in PAIRS.iter().enumerate() {
        for (gj, &(j, jp)) in PAIRS.iter().enumerate() {
            let (pj, pjp) = (pi[j], pi[jp]);
            let w = weights[ai][gj];
            let t = SGN[i][pj] * SGN[ip][pjp] * SGN[i][pjp] * SGN[ip][pj];
            let (u, v) = (i ^ pj, ip ^ pjp);
            let (up, vp) = (i ^ pjp, ip ^ pj);
            q[u][v] += w;
            q[up][vp] += w;
            if u ^ v == 0 {
                q[u][up] -= 2.0 * t * w;
            } else {
                kap -= 2.0 * t * w;
            }
        }
    }
    (q, kap)
}

/// Real roots of a real quartic, machine grade, on the stack.
///
/// The 4×4 Frobenius companion's Schur form: unlike the radical (Ferrari) route
/// it does not lose half its digits on the squared eliminant, and unlike the
/// crate's general `poly_roots` it allocates nothing -- both matter here, where
/// the rooter runs once per chart on the hot path. A degenerate leading
/// coefficient (the companion would blow up) falls back to the general rooter.
fn quartic_real_roots(q: &[f64; 5]) -> impl Iterator<Item = f64> {
    let scale = q.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let (mut fixed, mut spill) = ([0.0f64; 4], Vec::new());
    let mut n = 0usize;
    let finite = q.iter().all(|v| v.is_finite());
    if finite && scale > 0.0 && q[4].abs() >= 1e-12 * scale {
        let comp = nalgebra::Matrix4::from_fn(|i, j| {
            if j == 3 {
                -q[i] / q[4]
            } else if i == j + 1 {
                1.0
            } else {
                0.0
            }
        });
        // BOUND THE QR SWEEPS. `complex_eigenvalues` calls `Schur::new`, which
        // passes max_niter = 0 -- unlimited -- and spins forever on a companion
        // that will not converge. feasible_linspace row 672488 hung on exactly
        // that, invisible at stride 40. A non-converging companion simply
        // yields no roots; the caller then declines the chart, which is
        // correct, because an unconverged root is not a certificate.
        if let Some(sch) = nalgebra::Schur::try_new(comp, f64::EPSILON, 64) {
            for ev in sch.complex_eigenvalues().iter() {
                if ev.im.abs() < 1e-7 {
                    fixed[n] = ev.re;
                    n += 1;
                }
            }
        }
    } else if finite && scale > 0.0 {
        spill = super::poly_roots(q)
            .into_iter()
            .filter(|z| z.im.abs() < 1e-7)
            .map(|z| z.re)
            .collect();
    }
    fixed.into_iter().take(n).chain(spill)
}

/// Is `z` in the convex hull of four planar points? The EXACT necessary gate:
/// `e₁ = Σ C_m·x_m` with `x` in the simplex, so a chart admits a target only if
/// `e₁ᵗ ∈ conv{C_m}`. Testing it here costs twelve 2×2 determinants and skips
/// the cross products, the LU and the `e₂` split on the two thirds of charts
/// that cannot possibly hold the target (s262 decline census).
#[inline]
fn in_hull(c: &[C; 4], z: C) -> bool {
    let cross = |o: C, a: C, b: C| (a.re - o.re) * (b.im - o.im) - (a.im - o.im) * (b.re - o.re);
    const TRI: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    TRI.iter().any(|t| {
        let (p, q, r) = (c[t[0]], c[t[1]], c[t[2]]);
        let (d0, d1, d2) = (cross(p, q, z), cross(q, r, z), cross(r, p, z));
        let e = 1e-12;
        (d0 >= -e && d1 >= -e && d2 >= -e) || (d0 <= e && d1 <= e && d2 <= e)
    })
}

/// Real Takagi with a KNOWN spectrum: given a symmetric unitary `y` equal to
/// `O·diag(g)·Oᵀ` for some real orthogonal `O`, recover `O`.
///
/// `y = A + iB` with `A`, `B` real symmetric and COMMUTING (that is exactly
/// `y·ȳ = I`), so they share `O` as eigenvectors. A real combination `A + tB`
/// has known eigenvalues `Re g + t·Im g`; choosing `t` to separate them makes
/// one 4×4 real-symmetric eigendecomposition return the columns, matched to
/// the known spectrum rather than searched for. Column signs are free because
/// `O·diag(g)·Oᵀ` is invariant under them.
pub fn takagi_real(y: &Mat4, g: &[C; 4]) -> Option<nalgebra::Matrix4<f64>> {
    let a = nalgebra::Matrix4::<f64>::from_fn(|i, j| y[(i, j)].re);
    let b = nalgebra::Matrix4::<f64>::from_fn(|i, j| y[(i, j)].im);
    for &t in &[0.0f64, 1.0, -1.0, 0.5, 2.0, -0.37] {
        let lam: [f64; 4] = std::array::from_fn(|k| g[k].re + t * g[k].im);
        if (0..4).any(|i| ((i + 1)..4).any(|j| (lam[i] - lam[j]).abs() < 1e-7)) {
            continue;
        }
        // Sylvester's formula, not an eigensolver: with the eigenvalues KNOWN
        // and separated, the spectral projector Π_{j≠k}(M−λⱼ)/(λₖ−λⱼ) is rank
        // one, so its largest column IS the eigenvector. Three 4×4 products
        // per column, no iteration -- an iterative symmetric_eigen here costs
        // ~2 us and eats the whole rung's migration gain.
        let m = a + b * t;
        let mut o = nalgebra::Matrix4::<f64>::zeros();
        let mut ok = true;
        for k in 0..4 {
            let mut p = nalgebra::Matrix4::<f64>::identity();
            for j in 0..4 {
                if j != k {
                    let mut f = m;
                    for i in 0..4 {
                        f[(i, i)] -= lam[j];
                    }
                    p *= f / (lam[k] - lam[j]);
                }
            }
            let mut best = (0.0f64, 0usize);
            for cidx in 0..4 {
                let n = p.column(cidx).norm();
                if n > best.0 {
                    best = (n, cidx);
                }
            }
            if best.0 < 1e-8 {
                ok = false;
                break;
            }
            o.set_column(k, &(p.column(best.1) / best.0));
        }
        if ok {
            return Some(o);
        }
    }
    None
}

/// One chart's solved geometry: the line `x(s) = x0 + s·v` in the simplex and
/// the real eliminant data `p(s) = b₀ + b₁s + b₂s²`, `k`, on it.
struct Line {
    x0: [f64; 4],
    v: [f64; 4],
    cm: [C; 4],
    e1_target: C,
    pi: [usize; 4],
    odd: bool,
    b: [f64; 3],
    k: f64,
}

impl Line {
    /// Residual of the one unsquared scalar equation left after `e1` fixes the
    /// simplex line.  On the admissible interval all four coordinates are
    /// nonnegative, and the sign of `sigma` is forced by `p=-k*sigma*sqrt(D4)`.
    /// This ranks quartic roots without constructing four candidate matrices.
    fn equation_residual(&self, s: f64) -> f64 {
        let p = self.b[0] + self.b[1] * s + self.b[2] * s * s;
        let sigma = -(self.k * p).signum();
        let d4 = (0..4)
            .map(|i| (self.x0[i] + s * self.v[i]).max(0.0))
            .product::<f64>();
        (p + self.k * sigma * d4.sqrt()).abs()
    }

    /// The complete reduced characteristic-polynomial certificate specialized
    /// to the Klein chart.  This is exactly `compound_residual`: `e1` is the
    /// Klein convolution already used to define the line, while the real
    /// normalized `e2` error is the unsquared scalar equation above.  `e3` and
    /// `e4` then follow from self-inversiveness and the determinant constraint.
    fn residual(&self, s: f64) -> f64 {
        let x: [f64; 4] = std::array::from_fn(|i| (self.x0[i] + s * self.v[i]).max(0.0));
        let e1 = (0..4).map(|i| self.cm[i] * x[i]).sum::<C>();
        (e1 - self.e1_target).norm().max(self.equation_residual(s))
    }

    /// The frame at parameter `s` and its chart-native exact residual. `σ` is
    /// read off from `p = −k·σ·√D₄` (with `√D₄ > 0`), never searched.
    fn frame_at(&self, s: f64) -> (Mat4, f64) {
        let sigma = -(self.k * (self.b[0] + self.b[1] * s + self.b[2] * s * s)).signum();
        let mut q: [f64; 4] = std::array::from_fn(|i| (self.x0[i] + s * self.v[i]).max(0.0).sqrt());
        q[3] *= sigma;
        let om = build_o(&q, &self.pi, self.odd);
        let o = Mat4::from_fn(|i, j| C::new(om[i][j], 0.0));
        let r = self.residual(s);
        (o, r)
    }
}

/// Try every Klein chart in both target branches. Returns the certified frame
/// and its residual against the ORIGINAL sandwich.
pub fn solve(
    a2: &[C; 4],
    g2v: &[C; 4],
    products: &[[C; 4]; 4],
    targets: &[[C; 4]; 2],
) -> Option<(Mat4, f64)> {
    let mut weights: Option<[[C; 6]; 6]> = None;
    // FROBENIUS PRE-GATE: DERIVED, SOUND, AND TOO LOOSE TO PAY (2026-08-09).
    // From the singular-value form sigma(cosΘ_c·P·cosΘ_g − sinΘ_c·Q·sinΘ_g) =
    // |cos(φʷ_a/2)|: ‖G‖²_F = ‖APB‖² − 2<APB,N> + ‖N‖², with ‖APB‖² linear in
    // the doubly stochastic P∘P (so between its permutation extremes) and N
    // monomial on every Klein chart (so the cross term is bounded by |P|<=1).
    // Unioning over the six permutations gives a row-level necessary test in
    // ~40 flops needing no chart<->pattern map. It is CORRECT -- coverage stayed
    // 100.0000% with rung counts bit-identical on both corpora -- but the union
    // plus the |P|<=1 slack makes it so loose it rejects essentially nothing:
    // interleaved A/B, 5 reps, min statistic, haar 5.79 vs 5.79 (neutral),
    // linspace 4.39 -> 4.43 (+0.9%, pure overhead). Removed.
    // TIGHTENING ATTEMPTED AND IT IS WRONG AS WRITTEN (2026-08-09). Using
    // sigma(a) = pi[0]^pi[a] makes the test FIRE but it rejects charts that DO
    // solve (haar Klein 213204 -> 212678, 526 rows pushed to Interior; coverage
    // survives only by fall-through). A necessary condition may never do that.
    // THE DIAGNOSIS, corrected: it is NOT sigma-vs-sigma^-1 -- both the ||N||^2
    // and cross-term sums are symmetric in a, so those two give an IDENTICAL
    // bound. The real error is which SO(3) is frozen. For O = L_q F the compound
    // splits as Lambda^2 O = rho(q) rho_+(F)  (+)  rho_-(F), so the FREE factor
    // is P = rho(q) rho_+(F) and the FROZEN one is Q = rho_-(F): the monomial
    // pattern of N = sin(Theta_c) Q sin(Theta_g) is that of the ANTI-SELF-DUAL
    // image of the frozen signed permutation, which the naive XOR map is not.
    // Fix: compute rho_-(F) once per chart rep (6 of them, compile-time), take
    // its support, and VERIFY the bound against ||G||_F^2 evaluated directly at
    // a known solution before re-measuring. The further exact tightening is to
    // replace |P|<=1 in the cross term with the diagonal-of-a-rotation
    // tetrahedron (4 vertices, not 8 sign patterns), which needs N's signs.
    // Chart outer, branch inner: the constants C_m, the null direction, the LU of
    // the linear system, and the e₂ split are all properties of the CHART. Only
    // the right-hand side and e₂ᵗ change with the target branch.
    for (pi, odd) in REPS.iter() {
        {
            let cm: [C; 4] =
                std::array::from_fn(|m| (0..4).map(|k| products[pi[k] ^ m][k]).sum::<C>());

            if !targets.iter().any(|tg| in_hull(&cm, tg[0])) {
                continue;
            }

            // e₁ and Σx = 1: a 3×4 real system. `v` = the null direction
            // (generalized cross product of the three rows), `x0` = the
            // min-norm particular solution.
            let cross = |u: C, w: C| u.re * w.im - u.im * w.re;
            let det3 = |c: [usize; 3]| cross(cm[c[1]] - cm[c[0]], cm[c[2]] - cm[c[0]]);
            let vraw: [f64; 4] = [
                -det3([1, 2, 3]),
                det3([0, 2, 3]),
                -det3([0, 1, 3]),
                det3([0, 1, 2]),
            ];
            // Unit direction: the cross product's own scale is arbitrary, and it
            // would otherwise spread the quartic's coefficients over decades.
            let vn = vraw.iter().map(|z| z * z).sum::<f64>().sqrt();
            if vn < 1e-12 {
                continue;
            }
            let v: [f64; 4] = std::array::from_fn(|i| vraw[i] / vn);
            // A particular solution needs no generic 4x4 solve.  Delete the
            // coordinate whose cofactor has largest magnitude; the remaining
            // three C_m form the best-conditioned triangle.  Barycentric area
            // ratios solve e1=sum C_m*x_m, sum x_m=1.  Orthogonally projecting
            // that point along the null direction gives the same minimum-norm
            // anchor x0 that the former appended-row LU computed.
            let omitted = (0..4)
                .max_by(|&i, &j| vraw[i].abs().total_cmp(&vraw[j].abs()))
                .unwrap_or(0);
            let kept: [usize; 3] = {
                let mut out = [0usize; 3];
                let mut n = 0;
                for index in 0..4 {
                    if index != omitted {
                        out[n] = index;
                        n += 1;
                    }
                }
                out
            };
            let (ci, cj, ck) = (cm[kept[0]], cm[kept[1]], cm[kept[2]]);
            let denominator = cross(cj - ci, ck - ci);
            if denominator.abs() < 1e-12 {
                continue;
            }
            // `split_e2` is 36 complex quadruple products, the dominant per-chart
            // cost, and it depends only on (a2, g2, pi) -- never on the target.
            // Two thirds of charts are rejected by the x >= 0 gate below (the
            // exact hull condition e1ᵗ ∈ conv{C_m}), so it is computed lazily,
            // on the first branch that survives the gate.
            let mut e2split: Option<([[C; 4]; 4], C)> = None;

            for tg in targets.iter() {
                let (e1t, e2t) = (tg[0], tg[1]);
                let u = cross(e1t - ci, ck - ci) / denominator;
                let w = cross(cj - ci, e1t - ci) / denominator;
                let mut particular = [0.0f64; 4];
                particular[kept[0]] = 1.0 - u - w;
                particular[kept[1]] = u;
                particular[kept[2]] = w;
                let shift = (0..4).map(|i| particular[i] * v[i]).sum::<f64>();
                let x0: [f64; 4] = std::array::from_fn(|i| particular[i] - shift * v[i]);

                // The chart admits the target iff x0 + s·v ≥ 0 has a solution.
                let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
                for i in 0..4 {
                    if v[i].abs() < 1e-14 {
                        if x0[i] < 0.0 {
                            lo = 1.0;
                            hi = -1.0;
                        }
                        continue;
                    }
                    let b = -x0[i] / v[i];
                    if v[i] > 0.0 {
                        lo = lo.max(b);
                    } else {
                        hi = hi.min(b);
                    }
                }
                if !(hi - lo > 1e-12) {
                    continue;
                }

                let spectral_weights = weights.get_or_insert_with(|| pair_weights(a2, g2v));
                let (qf, kap) = *e2split.get_or_insert_with(|| split_e2(spectral_weights, pi));
                // p(s) = Re((P₂(s) − e₂ᵗ)/√e₄) and k = Re(κ/√e₄), both real; the
                // branch of √e₄ flips p and k together, so p² = k²D₄ does not see it.
                let s4 = tg[3].sqrt();
                let mut b = [0.0f64; 3];
                for a in 0..4 {
                    for c in 0..4 {
                        let w = qf[a][c] / s4;
                        b[0] += (w * x0[a] * x0[c]).re;
                        b[1] += (w * (x0[a] * v[c] + v[a] * x0[c])).re;
                        b[2] += (w * v[a] * v[c]).re;
                    }
                }
                b[0] -= (e2t / s4).re;
                let k = (kap / s4).re;

                // D₄(s) = Π(x0_i + s·v_i), ascending.
                let mut d4c = [0.0f64, 0.0, 0.0, 0.0, 0.0];
                d4c[0] = 1.0;
                for i in 0..4 {
                    for n in (0..4).rev() {
                        d4c[n + 1] += d4c[n] * v[i];
                        d4c[n] *= x0[i];
                    }
                }
                // p(s)² − k²·D₄(s), ascending.
                let k2 = k * k;
                let quart = [
                    b[0] * b[0] - k2 * d4c[0],
                    2.0 * b[0] * b[1] - k2 * d4c[1],
                    b[1] * b[1] + 2.0 * b[0] * b[2] - k2 * d4c[2],
                    2.0 * b[1] * b[2] - k2 * d4c[3],
                    b[2] * b[2] - k2 * d4c[4],
                ];
                // Squaring p = −kσ√D₄ worsens the quartic's conditioning. The
                // radical rooter still locates every root inside the admissible
                // interval; a companion solve is needed only when the selected
                // forward certificate is not already at machine precision.
                let line = Line {
                    x0,
                    v,
                    cm,
                    e1_target: e1t,
                    pi: *pi,
                    odd: *odd,
                    b,
                    k,
                };
                // The radical rooter SCANS: it locates every root of the squared
                // eliminant well inside the admissible interval, at a fraction
                // of an eigensolve. Every candidate is scored and the best kept
                // -- squaring p = −kσ√D₄ admits the other sign's roots too, and
                // a spurious one can sit just inside ACCEPT and mask the true
                // root. The accepted root is re-rooted only when its forward
                // residual is above the machine-grade threshold.
                let mut rt = [C::new(0.0, 0.0); 4];
                let Some(nr) = quartic_roots(&quart, &mut rt) else {
                    continue;
                };
                let roots: [Option<f64>; 4] = std::array::from_fn(|index| {
                    (index < nr)
                        .then_some(rt[index])
                        .filter(|z| z.im.abs() < 1e-7)
                        .map(|z| z.re)
                        .filter(|&s| s >= lo - 1e-9 && s <= hi + 1e-9)
                });
                let Some(mut s) = roots.iter().flatten().copied().min_by(|&x, &y| {
                    line.equation_residual(x)
                        .total_cmp(&line.equation_residual(y))
                }) else {
                    continue;
                };
                let (mut o, mut r) = line.frame_at(s);
                if r >= ACCEPT {
                    // Numerical fallback only. Algebraically every admissible
                    // root satisfying the unsquared equation is a solution;
                    // retain the former full scoring if f64 disagrees.
                    let first = s;
                    for other in roots.iter().flatten().copied().filter(|&x| x != first) {
                        let (candidate, residual) = line.frame_at(other);
                        if residual < r {
                            (o, r) = (candidate, residual);
                            s = other;
                        }
                    }
                    if r >= ACCEPT {
                        continue;
                    }
                }
                // The radical root is already a complete forward certificate.
                // The companion reroot exists only to restore machine-grade
                // digits, so it has no work to do when those digits are already
                // present.
                if r < 1e-12 {
                    return Some((o, r));
                }
                let exact = quartic_real_roots(&quart)
                    .min_by(|x, y| (x - s).abs().total_cmp(&(y - s).abs()))
                    .unwrap_or(s);
                let (oe, re) = line.frame_at(exact);
                let (o, r) = if re < r { (oe, re) } else { (o, r) };
                // Klein DECLINES rather than return a frame looser than the
                // rungs below it would have produced. It owns ~70% of the haar
                // corpora, so its worst row sets the corpus maximum: accepting
                // at ACCEPT pushed fresh_haar_500k's worst residual from
                // 2.149e-10 to 4.843e-10. Only 56 of 351,954 Klein rows exceed
                // FAST_ACCEPT, so the cost is 0.016% of its share and those
                // rows simply fall through to the interior atlas.
                if r < super::FAST_ACCEPT {
                    return Some((o, r));
                }
                continue;
            }
        }
    }
    None
}
