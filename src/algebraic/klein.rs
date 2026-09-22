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
use super::{ACCEPT, C, Mat4, PERMS24};

/// Diagonalize a nearly unitary normal matrix through a Hermitian projection.
///
/// A generic complex eigensolver can lose several digits on a clustered
/// unit-circle spectrum even though the matrix is normal.  For unitary `y`,
///
/// `H(phi) = Re(exp(-i phi) y)`
///
/// is Hermitian and has the same eigenspaces as `y`.  We choose a projection
/// direction that separates the requested target roots, use the backward-
/// stable self-adjoint eigensolver, and recover the complex roots as Rayleigh
/// quotients of `y`.  The returned error includes every off-diagonal entry in
/// that basis, so callers do not have to assume that the input was exactly
/// normal in floating point.
pub(crate) fn unitary_eigenvalues(y: &Mat4, target: &[C; 4]) -> Option<([C; 4], f64)> {
    // This must agree with the compiler's repeated-root cluster. Attempting to
    // separate two roots that the public boundary treats as one block makes
    // the projection direction chase their O(1e-12) split and can collapse
    // the gap between the actual eigenspaces.
    const ROOT_CLUSTER: f64 = 1e-8;
    let mut best_phi = 0.0;
    let mut best_gap = -1.0f64;
    for step in 0..64 {
        let phi = std::f64::consts::TAU * step as f64 / 64.0;
        let phase = C::from_polar(1.0, -phi);
        let projected = target.map(|root| (phase * root).re);
        let mut gap = f64::INFINITY;
        let mut has_distinct_pair = false;
        for i in 0..4 {
            for j in i + 1..4 {
                if (target[i] - target[j]).norm() >= ROOT_CLUSTER {
                    has_distinct_pair = true;
                    gap = gap.min((projected[i] - projected[j]).abs());
                }
            }
        }
        if !has_distinct_pair {
            gap = f64::INFINITY;
        }
        if gap > best_gap {
            best_gap = gap;
            best_phi = phi;
        }
    }

    let diagonalize = |phi: f64| -> Option<([C; 4], f64)> {
        let phase = C::from_polar(1.0, -phi);
        let hermitian = faer::Mat::<C>::from_fn(4, 4, |row, column| {
            (phase * y[(row, column)] + phase.conj() * y[(column, row)].conj()) * 0.5
        });
        let decomposition = hermitian.self_adjoint_eigen(faer::Side::Lower).ok()?;
        let vectors = decomposition.U();
        let diagonalized = Mat4::from_fn(|row, column| {
            let mut value = C::default();
            for i in 0..4 {
                for j in 0..4 {
                    value += vectors[(i, row)].conj() * y[(i, j)] * vectors[(j, column)];
                }
            }
            value
        });
        let off_diagonal = (0..4)
            .flat_map(|row| (0..4).map(move |column| (row, column)))
            .filter(|(row, column)| row != column)
            .map(|(row, column)| diagonalized[(row, column)].norm())
            .fold(0.0f64, f64::max);
        let roots = std::array::from_fn(|i| diagonalized[(i, i)]);
        Some((roots, off_diagonal))
    };

    let mut best = diagonalize(best_phi)?;
    // At an exact target collision the target-optimal projection may itself
    // be degenerate.  Only then, try a bounded set of transverse Hermitian
    // projections and retain the basis that most nearly diagonalizes `y`.
    if best.1 > 1e-9 {
        for step in 1..8 {
            let candidate = diagonalize(best_phi + std::f64::consts::PI * step as f64 / 8.0)?;
            if candidate.1 < best.1 {
                best = candidate;
            }
        }
    }
    Some(best)
}

/// Replace the spectrum of a nearby symmetric-unitary candidate in its real
/// eigenframe. Symmetric unitaries have commuting real and imaginary parts,
/// so one real self-adjoint eigendecomposition supplies that frame. The target
/// assignment is a finite permutation, and the caller must still reconstruct
/// and verify the original sandwich.
pub(crate) fn retarget_symmetric(y: &Mat4, target: &[C; 4]) -> Option<(Mat4, Mat4)> {
    const ROOT_CLUSTER: f64 = 1e-8;
    let mut best_phi = 0.0;
    let mut best_gap = -1.0f64;
    for step in 0..64 {
        let phi = std::f64::consts::TAU * step as f64 / 64.0;
        let phase = C::from_polar(1.0, -phi);
        let projected = target.map(|root| (phase * root).re);
        let mut gap = f64::INFINITY;
        let mut distinct = false;
        for i in 0..4 {
            for j in i + 1..4 {
                if (target[i] - target[j]).norm() >= ROOT_CLUSTER {
                    distinct = true;
                    gap = gap.min((projected[i] - projected[j]).abs());
                }
            }
        }
        if !distinct {
            gap = f64::INFINITY;
        }
        if gap > best_gap {
            best_gap = gap;
            best_phi = phi;
        }
    }
    let phase = C::from_polar(1.0, -best_phi);
    let projected = nalgebra::Matrix4::<f64>::from_fn(|row, column| {
        let forward = (phase * y[(row, column)]).re;
        let reverse = (phase * y[(column, row)]).re;
        0.5 * (forward + reverse)
    });
    let q = projected.symmetric_eigen().eigenvectors;
    let actual: [C; 4] = std::array::from_fn(|column| {
        let mut value = C::default();
        for row in 0..4 {
            for inner in 0..4 {
                value += C::new(q[(row, column)] * q[(inner, column)], 0.0) * y[(row, inner)];
            }
        }
        value
    });
    let permutation = PERMS24.iter().min_by(|left, right| {
        let score = |p: &&[usize; 4]| {
            (0..4)
                .map(|i| (actual[i] - target[p[i]]).norm_sqr())
                .sum::<f64>()
        };
        score(left).total_cmp(&score(right))
    })?;
    let mut ordered = Mat4::zeros();
    for column in 0..4 {
        for row in 0..4 {
            ordered[(row, permutation[column])] = C::new(q[(row, column)], 0.0);
        }
    }
    let diagonal = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(target));
    let rebuilt = ordered * diagonal * ordered.transpose();
    Some((rebuilt, ordered))
}

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

/// Is `z` in the convex hull of four planar points? The exact necessary gate:
/// `e₁ = Σ C_m·x_m` with `x` in the simplex, so a chart admits a target only if
/// `e₁ᵗ ∈ conv{C_m}`. Testing it here costs twelve 2×2 determinants and skips
/// the cross products, the LU and the `e₂` split on the two thirds of charts
/// that cannot possibly hold the target.
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
pub(crate) fn takagi_real(y: &Mat4, g: &[C; 4]) -> Option<nalgebra::Matrix4<f64>> {
    let a = nalgebra::Matrix4::<f64>::from_fn(|i, j| y[(i, j)].re);
    let b = nalgebra::Matrix4::<f64>::from_fn(|i, j| y[(i, j)].im);
    for &t in &[0.0f64, 1.0, -1.0, 0.5, 2.0, -0.37] {
        let lam: [f64; 4] = std::array::from_fn(|k| g[k].re + t * g[k].im);
        if (0..4).any(|i| ((i + 1)..4).any(|j| (lam[i] - lam[j]).abs() < 1e-7)) {
            continue;
        }
        // Sylvester's formula, not an eigensolver: with the eigenvalues KNOWN
        // and separated, the spectral projector Π_{j≠k}(M−λⱼ)/(λₖ−λⱼ) is rank
        // one, so its largest column IS the eigenvector. Horner form: the
        // numerator is the divided characteristic m³−s₁m²+s₂m−s₃I with sᵢ the
        // elementary symmetric functions of the three OTHER eigenvalues, so
        // m² and m³ are shared across all four columns -- two 4×4 products
        // total instead of twelve, no iteration.
        let m = a + b * t;
        let m2 = m * m;
        let m3 = m2 * m;
        let mut o = nalgebra::Matrix4::<f64>::zeros();
        let mut ok = true;
        for k in 0..4 {
            let mut others = [0.0f64; 3];
            let mut idx = 0;
            let mut d = 1.0f64;
            for j in 0..4 {
                if j != k {
                    others[idx] = lam[j];
                    idx += 1;
                    d *= lam[k] - lam[j];
                }
            }
            let s1 = others[0] + others[1] + others[2];
            let s2 = others[0] * others[1] + others[0] * others[2] + others[1] * others[2];
            let s3 = others[0] * others[1] * others[2];
            let mut p = (m3 - m2 * s1 + m * s2) / d;
            for i in 0..4 {
                p[(i, i)] -= s3 / d;
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
            let oc = Mat4::from_fn(|r, k| C::new(o[(r, k)], 0.0));
            let diagonalized = oc.transpose() * y * oc;
            let mut error = 0.0f64;
            for r in 0..4 {
                for c in 0..4 {
                    let want = if r == c { g[r] } else { C::default() };
                    error = error.max((diagonalized[(r, c)] - want).norm());
                }
            }
            if error < 1e-8 {
                return Some(o);
            }
        }
    }
    // Near a target collision every fixed projection above can have a small
    // eigenvalue gap, making divided projectors unusable even though the real
    // Takagi frame itself is well conditioned as a subspace.  A symmetric
    // eigensolve supplies an orthonormal basis of that subspace; enumerate its
    // finite column order and accept only direct complex reconstruction.
    for &t in &[0.0f64, 1.0, -1.0, 0.5, 2.0, -0.37] {
        let m = a + b * t;
        let eigenvectors = m.symmetric_eigen().eigenvectors;
        for permutation in *PERMS24 {
            let o = nalgebra::Matrix4::<f64>::from_fn(|r, k| eigenvectors[(r, permutation[k])]);
            let oc = Mat4::from_fn(|r, k| C::new(o[(r, k)], 0.0));
            let diagonalized = oc.transpose() * y * oc;
            let mut error = 0.0f64;
            for r in 0..4 {
                for c in 0..4 {
                    let want = if r == c { g[r] } else { C::default() };
                    error = error.max((diagonalized[(r, c)] - want).norm());
                }
            }
            if error < 1e-8 {
                return Some(o);
            }
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
pub(crate) fn solve(
    a2: &[C; 4],
    g2v: &[C; 4],
    products: &[[C; 4]; 4],
    targets: &[[C; 4]; 2],
) -> Option<(Mat4, f64)> {
    let mut weights: Option<[[C; 6]; 6]> = None;
    // A Frobenius-norm pre-gate (from the singular-value form of the compound)
    // is sound but too loose to pay: unioned over the six permutations it
    // rejects essentially nothing. Tightening it by freezing the wrong SO(3)
    // factor rejects charts that do solve; the free factor is
    // P = rho(q) rho_+(F) and the frozen one is Q = rho_-(F), whose monomial
    // pattern is the anti-self-dual image of the signed permutation. Not used.
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

/// All four roots of a real quartic in radicals (Ferrari / resolvent
/// cubic) -- the section construction's own algebra, replacing a general
/// companion eigensolve per section. Complex pairs are returned (the
/// caller's small-imaginary filter is semantic). Returns None on a
/// degenerate leading coefficient or non-finite intermediates; the
/// caller falls back to the eigensolve rooter, so completeness never
/// depends on this path.
pub(crate) fn quartic_roots(q: &[f64; 5], out: &mut [C; 4]) -> Option<usize> {
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
