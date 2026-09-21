//! The can-sandwich, from scratch: given monodromy coords (C, G, T), find the local
//! `u` (as a real frame `O ∈ SO(4)` in the magic basis) with
//! `weyl(Can(C)·u·Can(G)) = weyl(T)`.
//!
//! One master object: `M = D_C·O·Λ·Oᵀ·D_C`, the Makhlin matrix of the sandwich in
//! the magic basis, with `D_C = mb(Can(C))`, `Λ = mb(Can(G))²` both diagonal and
//! `O = mb(u) ∈ SO(4)`. Reachability/matching is the spectrum of `M` (the real
//! multiplicative-Horn image); we match it by its symmetric functions `e₁..e₄`
//! (traces, no eig -- polynomial, does not floor at degeneracy).
use nalgebra::{Complex, Matrix4};
use std::f64::consts::PI;

type C = Complex<f64>;

#[path = "certificate.rs"]
mod certificate;
#[path = "chart_precision.rs"]
mod chart_precision;
#[path = "charts.rs"]
mod charts;
#[path = "interior.rs"]
mod interior;
#[path = "klein.rs"]
mod klein;
#[path = "one_plus_three.rs"]
mod one_plus_three;
#[path = "problem.rs"]
mod problem;
#[path = "resonance.rs"]
mod resonance;
#[path = "support_strata.rs"]
mod support_strata;
#[path = "three_givens.rs"]
mod three_givens;
#[path = "two_plus_two.rs"]
mod two_plus_two;
use certificate::*;
// Keep the interior algebra available to sibling rungs through this module,
// but do not re-export its entire implementation surface within the crate.
pub(crate) use interior::bernstein_variations;
use interior::*;
pub type Mat4 = Matrix4<C>;

/// One accepted interior chart: frame, residual, row permutation, plane word.
type InteriorHit = (Mat4, f64, [usize; 4], [(usize, usize); 3]);

#[cfg(test)]
use problem::spectrum_kind;
use problem::{PreparedSandwich, SpectrumKind, StratumSignature};

/// PROF=1 instrumentation: per-stage aggregate ns/calls across a corpus run.
/// Compiled out unless the `diagnostics` feature is enabled.
pub mod prof {
    #[cfg(feature = "diagnostics")]
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    #[cfg(feature = "diagnostics")]
    pub const N: usize = 53;
    /// Slot indices, one per stage or event counter.
    pub const EDGE: usize = 0;
    pub const CHART_COEFFS: usize = 1;
    pub const ELIM: usize = 2;
    pub const ROOTS: usize = 3;
    pub const SCORE: usize = 4;
    pub const RADICAL_ORIENTED: usize = 5;
    pub const RADICAL_FRAME: usize = 6;
    pub const INTERIOR_TOTAL: usize = 7;
    pub const RADICAL_TOTAL: usize = 8;
    pub const PRELUDE: usize = 9;
    pub const FACE: usize = 10;
    pub const SW_RANK1: usize = 11;
    pub const SW_TWO_STEP: usize = 12;
    pub const SW_MIRROR: usize = 13;
    pub const SW_VERIFY: usize = 14;
    pub const SW_PAIR_ROOTS: usize = 15;
    pub const SW_WORD_GATE: usize = 16;
    pub const GATE_BOX: usize = 17;
    pub const GATE_HULL: usize = 18;
    pub const GATE_PASS: usize = 19;
    pub const TIER_REANCHOR_FAST: usize = 20;
    pub const N_TRY_MU: usize = 21;
    pub const RJ_BASE: usize = 22;
    pub const RJ_SKEL: usize = 23;
    pub const RJ_UREAL: usize = 24;
    pub const RJ_USUM: usize = 25;
    pub const N_ACCEPT: usize = 26;
    pub const RJ_V_IMAG: usize = 27;
    pub const RJ_V_NEG: usize = 28;
    pub const RJ_V_SUM: usize = 29;
    pub const SW_BASE: usize = 30;
    pub const ARC_PAIR_SKIP: usize = 31;
    pub const ARC_CAND_SKIP: usize = 32;
    pub const URAY_SKIP: usize = 33;
    pub const SKEL_SKIP: usize = 34;
    pub const BASE_FAIL_FORCED: usize = 35;
    pub const BASE_FAIL_SKEL: usize = 36;
    pub const BASE_FAIL_PAIR: usize = 37;
    pub const DEV_LT_1EM6: usize = 38;
    pub const DEV_MID: usize = 39;
    pub const DEV_GT_1EM2: usize = 40;
    pub const PAIR_GATE_SOME: usize = 41;
    pub const PAIR_GATE_NONE: usize = 42;
    pub const INH_SKIP: usize = 43;
    pub const RED_SKIP: usize = 44;
    pub const SW_HEADER: usize = 45;
    pub const KLEIN_TOTAL: usize = 46;
    pub const SEG_PREPARE: usize = 47;
    pub const SEG_EDGEGATE: usize = 48;
    pub const SEG_VERTEX: usize = 49;
    pub const CHART_TIER: usize = 50;
    pub const CHART_PHASEA: usize = 51;
    pub const CERT_FAST: usize = 52;
    #[cfg(feature = "diagnostics")]
    pub const NAMES: [&str; N] = [
        "edge",
        "chart_coeffs",
        "elim",
        "roots",
        "score",
        "radical_oriented",
        "radical_frame",
        "interior_total",
        "radical_total",
        "prelude",
        "face",
        "sw_rank1",
        "sw_two_step",
        "sw_mirror",
        "sw_verify",
        "sw_pair_roots",
        "sw_word_gate",
        "gate_box",
        "gate_hull",
        "gate_pass",
        "tier_reanchor_fast",
        "n_try_mu",
        "rj_base",
        "rj_skel",
        "rj_ureal",
        "rj_usum",
        "n_accept",
        "rj_v_imag",
        "rj_v_neg",
        "rj_v_sum",
        "sw_base",
        "arc_pair_skip",
        "arc_cand_skip",
        "uray_skip",
        "skel_skip",
        "base_fail_forced",
        "base_fail_skel",
        "base_fail_pair",
        "dev_lt_1em6",
        "dev_mid",
        "dev_gt_1em2",
        "pair_gate_some",
        "pair_gate_none",
        "inh_skip",
        "red_skip",
        "sw_header",
        "klein_total",
        "seg_prepare",
        "seg_edgegate",
        "seg_vertex",
        "chart_tier",
        "chart_phaseA",
        "cert_fast",
    ];
    #[cfg(feature = "diagnostics")]
    pub static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    #[cfg(feature = "diagnostics")]
    pub static CT: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    #[cfg(feature = "diagnostics")]
    #[inline]
    fn on() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| std::env::var_os("PROF").is_some())
    }
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub fn start() -> Option<std::time::Instant> {
        on().then(std::time::Instant::now)
    }
    #[cfg(not(feature = "diagnostics"))]
    #[inline(always)]
    pub fn start() -> Option<std::time::Instant> {
        None
    }
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub fn rec(i: usize, t0: Option<std::time::Instant>) {
        if let Some(t) = t0 {
            NS[i].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            CT[i].fetch_add(1, Relaxed);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    #[inline(always)]
    pub fn rec(_i: usize, _t0: Option<std::time::Instant>) {}
    /// Count-only event (no timing): gate/rejection census under PROF=1.
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub fn hit(i: usize) {
        if on() {
            CT[i].fetch_add(1, Relaxed);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    #[inline(always)]
    pub fn hit(_i: usize) {}
    #[cfg(feature = "diagnostics")]
    #[allow(clippy::print_stderr)]
    pub fn dump() {
        if !on() {
            return;
        }
        for i in 0..N {
            let (ns, ct) = (NS[i].load(Relaxed), CT[i].load(Relaxed));
            if ct > 0 {
                eprintln!(
                    "[prof] {:>16}: {:>9.1} ms  {:>9} calls  {:>8.2} us/call",
                    NAMES[i],
                    ns as f64 / 1e6,
                    ct,
                    ns as f64 / 1e3 / ct as f64
                );
            }
        }
    }
}

#[inline]
fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

/// The diagonal phase matrix `D = mb(Can(w))` (canonical gates are diagonal in the
/// magic basis). Built directly as `diag(exp(i*eigphases(w)))` in the ordering
/// documented by [`eigphases`].
#[cfg(any(test, feature = "diagnostics"))]
pub(super) fn dphase(w: [f64; 3]) -> Mat4 {
    let ph = eigphases(w);
    let d: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, ph[k]));
    Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&d))
}

/// Return the canonical right endpoint frame of a certified child.
/// This is the exact frame needed to insert the local layer between two
/// adjacent entanglers in a factorization chain.
#[cfg(feature = "diagnostics")]
pub fn endpoint_right_gauge(c: [f64; 3], g: [f64; 3], t: [f64; 3], frame: &Mat4) -> Option<Mat4> {
    let problem = PreparedSandwich::new(c, g, t);
    certificate::canonical_right_endpoint_gauge(&problem, frame)
        .or_else(|| certificate::endpoint_factorization(&problem, frame).map(|(_, r)| r))
}

#[cfg(feature = "diagnostics")]
pub fn endpoint_gauge_residual(c: [f64; 3], g: [f64; 3], t: [f64; 3], frame: &Mat4) -> f64 {
    let problem = PreparedSandwich::new(c, g, t);
    certificate::endpoint_factorization_residual(&problem, frame)
}

/// Check whether two certified child endpoint gauges collapse a three-factor
/// entangler word to the requested gate class.  If the child identities are
/// `D_C O₁ D_X = L₁ D_M R₁`, etc., the forced interstitial locals are
/// `V₁=R₁ᵀ`, `V₂=R₂ᵀ`; no optimization remains.  The residual compares the
/// spectrum of `(D_X V₁ D_Y V₂ D_Z)(...)ᵀ` with the canonical spectrum of G.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "diagnostics")]
pub fn factorized_gate_collapse_residual(
    g: [f64; 3],
    x: [f64; 3],
    y: [f64; 3],
    z: [f64; 3],
    r1: &Mat4,
    middle_frame: &Mat4,
    r2: &Mat4,
    final_frame: &Mat4,
) -> f64 {
    let word = dphase(x)
        * r1.transpose()
        * *middle_frame
        * dphase(y)
        * r2.transpose()
        * *final_frame
        * dphase(z);
    let actual = symfn(&(word * word.transpose()));
    let problem = PreparedSandwich::new([0.0; 3], g, [0.0; 3]);
    (0..2)
        .map(|branch| {
            let want = esym4(problem.target_roots[branch]);
            actual
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

/// Monodromy triple -> Weyl coords (gulps convention `[m₀+m₁, m₀+m₂, m₁+m₂]`).
pub(super) fn weyl_from_monodromy(m: [f64; 3]) -> [f64; 3] {
    [m[0] + m[1], m[0] + m[2], m[1] + m[2]]
}

/// Rho-reflected Weyl coords `[1−c₁, c₂, −c₃]`: same Makhlin invariants, different
/// eigenvalue set -- a target can match in either orientation, so rungs try both.
pub(super) fn rho_weyl(w: [f64; 3]) -> [f64; 3] {
    [1.0 - w[0], w[1], -w[2]]
}

#[inline]
/// The 4 magic-basis eigenphases of `Can(w)` in closed form (no matrix): the diagonal
/// of `mb(Can(w))`, i.e. `D_C = diag(exp(i·eigphases))`. Same order as `dphase`.
pub(super) fn eigphases(w: [f64; 3]) -> [f64; 4] {
    let h = PI / 2.0;
    [
        h * (w[0] - w[1] + w[2]),
        h * (w[0] + w[1] - w[2]),
        h * (-w[0] - w[1] - w[2]),
        h * (-w[0] + w[1] + w[2]),
    ]
}

/// Elementary symmetric functions `e₁..e₄` of four scalars (the diagonal-spectrum
/// fast path: `symfn` without forming any matrix).
pub(super) fn esym4(s: [C; 4]) -> [C; 4] {
    let e1 = compensated_sum(s);
    let e2 = compensated_sum([
        s[0] * s[1],
        s[0] * s[2],
        s[0] * s[3],
        s[1] * s[2],
        s[1] * s[3],
        s[2] * s[3],
    ]);
    let e3 = compensated_sum([
        s[0] * s[1] * s[2],
        s[0] * s[1] * s[3],
        s[0] * s[2] * s[3],
        s[1] * s[2] * s[3],
    ]);
    let e4 = s[0] * s[1] * s[2] * s[3];
    [e1, e2, e3, e4]
}

#[inline]
pub(super) fn compensated_sum<const N: usize>(terms: [C; N]) -> C {
    let mut sum = C::default();
    let mut correction = C::default();
    for term in terms {
        let adjusted = term - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

/// One signed permutation frame `P ∈ SO(4)`: the permutation matrix `P_{i,p_i}=1`
/// with row 0 negated when needed to force `det = +1`. The sign squares away in the
/// spectrum (`PΛPᵀ` diagonal = `Λ` permuted), so it matters only for the emitted frame.
pub(super) fn signed_perm(p: [usize; 4]) -> Mat4 {
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

/// Givens rotation in plane `(i,j)` by `θ`: identity except `g[i,i]=g[j,j]=cosθ`,
/// `g[i,j]=−sinθ`, `g[j,i]=sinθ` (s66 convention).
pub(super) fn givens(i: usize, j: usize, theta: f64) -> Mat4 {
    let (ct, st) = (theta.cos(), theta.sin());
    let mut g = Mat4::identity();
    g[(i, i)] = c(ct, 0.0);
    g[(j, j)] = c(ct, 0.0);
    g[(i, j)] = c(-st, 0.0);
    g[(j, i)] = c(st, 0.0);
    g
}

/// The 6 Givens planes (= the 6 transpositions / permutohedron edge directions).
pub(super) const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// The 24 permutations of `(0,1,2,3)`, computed once (hot-path: no per-call alloc).
pub(super) static PERMS24: std::sync::LazyLock<[[usize; 4]; 24]> = std::sync::LazyLock::new(|| {
    let mut v = Vec::with_capacity(24);
    let mut a = [0usize, 1, 2, 3];
    heap(&mut a, 4, &mut v);
    std::array::from_fn(|i| v[i])
});

fn heap(a: &mut [usize; 4], k: usize, out: &mut Vec<[usize; 4]>) {
    if k == 1 {
        out.push(*a);
        return;
    }
    for i in 0..k {
        heap(a, k - 1, out);
        let swap = if k.is_multiple_of(2) { i } else { 0 };
        a.swap(swap, k - 1);
    }
}

/// Elementary symmetric functions `e₁..e₄` of `eig(A)` via Newton's identities on
/// traces -- no eigendecomposition, so it does not floor at spectrum degeneracy.
/// Production uses the matrix-free `compound_residual`; this is the test-side
/// reference implementation.
pub(super) fn symfn(a: &Mat4) -> [C; 4] {
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
pub(super) fn mmat(dc: &Mat4, lam: &Mat4, o: &Mat4) -> Mat4 {
    dc * o * lam * o.transpose() * dc
}

#[cfg(test)]
/// Smooth (non-flooring) residual: `‖symfn(M(O)) − symfn(target)‖∞`. The reach
/// certificate; near degeneracy this is the metric, not weyl coords.
pub(super) fn smooth_residual(dc: &Mat4, lam: &Mat4, o: &Mat4, target: &[C; 4]) -> f64 {
    let got = symfn(&mmat(dc, lam, o));
    (0..4)
        .map(|i| (got[i] - target[i]).norm())
        .fold(0.0, f64::max)
}

/// Cauchy-Binet compound-moment residual `max(|Δe₁|, |Re(Δe₂/s)|)` with
/// `s = √e₄(target)`, evaluated without forming `M` or any matrix power (16 + 36
/// scalar terms instead of 7 matrix products). Equal to `smooth_residual` in
/// exact arithmetic; the numerical discrepancy is about 1e-14.
pub(super) fn compound_residual(dc: &Mat4, lam: &Mat4, o: &Mat4, target: &[C; 4]) -> f64 {
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

    // e2 = sum_{i0<i1, j0<j1} a[i0]*a[i1] * det(O_{IJ})^2 * lv[j0]*lv[j1]  (36 terms)
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut e2_terms = [C::default(); 36];
    term_index = 0;
    for &(i0, i1) in &PAIRS {
        let ai = a[i0] * a[i1];
        for &(j0, j1) in &PAIRS {
            let m = or_[i0][j0] * or_[i1][j1] - or_[i0][j1] * or_[i1][j0];
            e2_terms[term_index] = ai * lv[j0] * lv[j1] * (m * m);
            term_index += 1;
        }
    }
    let e2 = compensated_sum(e2_terms);

    // Reduced certificate: max(|Δe1|, |Re(Δe2/s)|). Equal to smooth_residual because
    // |e4|=1 forces |De3|=|De1| and De4=0, and e2/s ∈ R so |Re(De2/s)| = |De2| exactly.
    let de1 = e1 - target[0];
    let s = target[3].sqrt();
    let de2_s = (e2 - target[1]) / s;
    de1.norm().max(de2_s.re.abs())
}

/// 3-Givens chart frame `O = G(planes₂,θ_z)·G(planes₁,θ_y)·G(planes₀,θ_x)·P`, with
/// `θ(v)=arccos(√v)`, `v=cos²θ ∈ [0,1]` (s13 chart). `e₁,e₂,e₃` of `M` are trilinear
/// in `(x,y,z)` here -- the structure the deg-6 companion eigensolve exploits.
pub(super) fn chart_o(xyz: [f64; 3], planes: [(usize, usize); 3], perm: [usize; 4]) -> Mat4 {
    let mut o = signed_perm(perm);
    for (k, &(i, j)) in planes.iter().enumerate() {
        let theta = xyz[k].clamp(0.0, 1.0).sqrt().acos();
        o = givens(i, j, theta) * o;
    }
    o
}

/// Direct-sqrt chart frame: `cos θ = √v`, `sin θ = √(1−v)` for `θ = arccos(√v)`,
/// three square roots per coordinate instead of three transcendentals. Equal to
/// `chart_o` in exact arithmetic.
pub(super) fn chart_o_sqrt(xyz: [f64; 3], planes: [(usize, usize); 3], perm: [usize; 4]) -> Mat4 {
    let mut o = signed_perm(perm);
    for (k, &(i, j)) in planes.iter().enumerate() {
        let v = xyz[k].clamp(0.0, 1.0);
        let ct = v.sqrt();
        let st = (1.0 - v).sqrt();
        let mut g = Mat4::identity();
        g[(i, i)] = c(ct, 0.0);
        g[(j, j)] = c(ct, 0.0);
        g[(i, j)] = c(-st, 0.0);
        g[(j, i)] = c(st, 0.0);
        o = g * o;
    }
    o
}

/// Berkeley (2+2) CS masses of an orthogonal frame.  These are the squared
/// singular-value invariants of the leading 2x2 block; unlike a frame gauge,
/// they survive the left/right O(2)xO(2) actions.
#[cfg(any(test, feature = "diagnostics"))]
pub(super) fn cs_masses(o: &Mat4) -> (f64, f64) {
    let a00 = o[(0, 0)].re;
    let a01 = o[(0, 1)].re;
    let a10 = o[(1, 0)].re;
    let a11 = o[(1, 1)].re;
    let s = a00 * a00 + a01 * a01 + a10 * a10 + a11 * a11;
    let det = a00 * a11 - a01 * a10;
    (s, det * det)
}

/// Multilinear coefficient table for the lifted CS masses on an atlas word.
/// Coefficients are indexed by the 3-bit corner `(x=1,y=1,z=1)` mask.  The
/// table is an exact algebraic representation of the corner interpolant; the
/// atlas test below certifies that it equals the completed-frame masses.
#[cfg(test)]
pub(super) fn cs_mass_coeffs(planes: [(usize, usize); 3], perm: [usize; 4]) -> [[f64; 8]; 2] {
    let mut coeffs = [[0.0; 8]; 2];
    for mask in 0..8 {
        let q = [
            if mask & 1 != 0 { 1.0 } else { 0.0 },
            if mask & 2 != 0 { 1.0 } else { 0.0 },
            if mask & 4 != 0 { 1.0 } else { 0.0 },
        ];
        let (s, d) = cs_masses(&chart_o(q, planes, perm));
        coeffs[0][mask] = s;
        coeffs[1][mask] = d;
    }
    coeffs
}

#[inline]
#[cfg(test)]
fn eval_multilinear(coeffs: &[f64; 8], p: [f64; 3]) -> f64 {
    (0..8)
        .map(|mask| {
            let weight = (0..3).fold(1.0, |acc, k| {
                let bit = (mask >> k) & 1;
                acc * if bit == 1 { p[k] } else { 1.0 - p[k] }
            });
            weight * coeffs[mask]
        })
        .sum()
}

/// All complex roots of a real polynomial via companion-matrix eigenvalues
/// (faer). Used by bounded algebraic constructions outside the hot
/// three-Givens atlas. `coeffs` are low-to-high; trailing near-zero leading
/// terms are trimmed.
pub(super) fn poly_roots(coeffs: &[f64]) -> Vec<C> {
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
    let comp = faer::Mat::<f64>::from_fn(n, n, |i, j| {
        if j == n - 1 {
            -a[i] / lead
        } else if i == j + 1 {
            1.0
        } else {
            0.0
        }
    });
    // A degenerate γ can still yield a non-convergent eigensolve; treat as "no roots here" (the
    // per-γ scan skips it) rather than panicking.
    let Ok(ev) = comp.eigenvalues() else {
        return vec![];
    };
    (0..n).map(|i| C::new(ev[i].re, ev[i].im)).collect()
}

/// Eigenvalues of a 4×4 (faer). Test-only: the solver gets the target spectrum in closed
/// form (`exp(2i·eigphases)`) and matches via `symfn`.
#[cfg(test)]
pub(super) fn eig4(m: &Mat4) -> [C; 4] {
    let fm = faer::Mat::<C>::from_fn(4, 4, |i, j| m[(i, j)]);
    let ev = fm.eigenvalues().expect("eig4");
    std::array::from_fn(|i| ev[i])
}

/// Evaluate a trilinear form (coeffs in monomial order) at `(x,y,z)`. Test-only: the solver
/// uses `ev_ml` (the multilinear+Y chart); this is the roundtrip harness's reference form.
#[cfg(test)]
pub(super) fn ev_trilinear(co: &[C; 8], x: f64, y: f64, z: f64) -> C {
    let m = [1.0, x, y, z, x * y, x * z, y * z, x * y * z];
    (0..8).map(|k| co[k] * m[k]).sum()
}

// ---- polynomial helpers for the interior elimination (real coeffs, low→high) ----

/// Which bounded construction produced the certified frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    /// A frame transported from an adjacent exact stratum and certified
    /// directly against the original problem. This is a finite algebraic
    /// transition chart, not a numerical correction.
    Vertex,
    Edge,
    Face,
    /// Routed `1 + 3` peel: rank-drop formulas, nine quadratic zero-entry
    /// walls, and the dense residual selector.
    OnePlusThree,
    /// Exact rank-one spectral-measure selector when either factor is `3+1`.
    RankOne31,
    /// Certified candidate from the near-`3+1` rank-one limit. The formula is
    /// not a theorem on this stratum; the unchanged forward certificate is
    /// mandatory.
    NearRankOne31,
    Interior,
    /// Klein-circulant one-sided chart: `e₁` linear in the orthostochastic
    /// diagonal pins a line in the simplex, `e₂` collapses to one real quadratic.
    Klein,
    /// One factor has multiplicity `2 + 2`: linear target matching in six
    /// squared Pluecker coordinates followed by one Heron plane quartic.
    Pair22,
    /// Input-side rank-secular closed form ((3,1)-degenerate base or gate).
    /// Radical strata of the distilled solver (skeleton / pin-pair /
    /// split-pair theta characteristics), tried in all four orientations and
    /// re-gated on the original smooth residual.
    Radical,
    /// The row-in-plane chart atlas: one row of `O` in a coordinate 2-plane, the
    /// inner solve a line cut by the Heron quartic, `t'` from the chart's own
    /// discriminant. Closed-form frames only.
    Chart,
    Unsolved,
}

/// Stable, endpoint-computable branch bin used by corpus census tooling.
/// This deliberately records predicates independently of the first successful
/// rung, so a new branch can be measured as a routing change rather than only
/// as a new success count.
#[cfg(feature = "diagnostics")]
pub fn branch_signature(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> String {
    let p = PreparedSandwich::new(c, g, t);
    format!(
        "C={:?};G={:?};T0={:?};T1={:?};Cp={:?};Gp={:?};Tp0={:?};Tp1={:?};routed_collision={}",
        p.strata.c,
        p.strata.g,
        p.strata.target[0],
        p.strata.target[1],
        p.strata.c_proximity,
        p.strata.g_proximity,
        p.strata.target_proximity[0],
        p.strata.target_proximity[1],
        p.strata.routed_collision
    )
}

#[derive(Clone)]
pub struct Solution {
    pub o: Mat4,
    /// Construction that produced the certified frame, or `Unsolved`.
    pub rung: Rung,
    pub residual: f64,
}

/// Accept threshold on the smooth residual. A true reach is ~1e-13; this is loose
/// enough to absorb FP while rejecting non-reaches (whose residual is O(0.1+)).
pub(super) const ACCEPT: f64 = 1e-9;

/// Public-frame tolerance.  This is deliberately tighter than the spectral
/// acceptance threshold: the fast compound residual is valid only for a real
/// orthogonal frame.  A rejected accelerator candidate must fall through to
/// the next construction rather than escape the black box.
const FRAME_ACCEPT: f64 = 2e-10;

#[inline]
#[cfg(test)]
fn certified_solution(o: Mat4, rung: Rung, residual: f64) -> Option<Solution> {
    if residual > ACCEPT {
        return None;
    }
    framed_solution(o, rung, residual)
}

#[inline]
fn unsolved_solution() -> Solution {
    Solution {
        o: Mat4::identity(),
        rung: Rung::Unsolved,
        residual: f64::INFINITY,
    }
}

/// Fast-accept threshold for the reduced trilinear score. Candidates with
/// `rs < FAST_ACCEPT` have `smooth_residual` below `FAST_ACCEPT` plus the score
/// discrepancy (about 6e-15 over the full corpora), so they sit well below
/// `ACCEPT`, which is 100x larger. The reduced score also rejects candidates at
/// or above `ACCEPT` before frame construction; shadow gates over the full Haar
/// and linspace corpora found no decision disagreements.
pub(crate) const FAST_ACCEPT: f64 = 1e-11;

/// How many ranked perms the edge/interior rungs try. The vertex-residual rank ORDERS the
/// perms so the right chart is hit first (fast early-exit), but truncating it drops the right
/// perm region-dependently and costs coverage for ~no perf gain (perf is eig-bound, not
/// perm-bound) -- so we keep all 24, ranked.
pub(super) const CAND_K: usize = 24;

/// Force every lazily built table (perm list, interior quotient). One-time
/// setup work; call before timing loops so a corpus max measures the solver,
/// not the first row's table construction (measured: the linspace corpus max
/// was the first interior-rung row paying the 216-class quotient build).
#[cfg(feature = "diagnostics")]
pub fn init_tables() {
    let _ = &*PERMS24;
    let _ = &*INTERIOR_QUOTIENT;
}

pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    solve_inner(c, g, t)
}

/// Enumerate the finite ordered interior chart witnesses.  A generic child
/// realization has a continuous fiber; these are distinct closed-form
/// transversal witnesses exposed by the atlas, rather than repeated calls to
/// the first-hit production selector.
#[cfg(any(test, feature = "diagnostics"))]
pub fn ordered_chart_solutions(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Vec<Solution> {
    let problem = PreparedSandwich::new(c, g, t);
    let mut out = Vec::new();
    let mut vertices = [None; 256];
    for &order in &interior::BOUNDARY_HEAD {
        let Some((o, residual, _, _)) = interior::solve_interior_cover(
            &problem.left,
            &problem.right,
            &problem.target_roots,
            &problem.dc,
            &problem.lam,
            &problem.targets,
            &[order],
            &mut vertices,
        ) else {
            continue;
        };
        let Some(solution) = certificate::compiler_solution(&problem, o, Rung::Interior, residual)
        else {
            continue;
        };
        if out.iter().all(|old: &Solution| {
            (old.o - solution.o)
                .iter()
                .map(|z| z.norm())
                .fold(0.0, f64::max)
                > 1e-8
        }) {
            out.push(solution);
        }
    }
    if out.is_empty() {
        let solution = solve_inner(c, g, t);
        if solution.rung != Rung::Unsolved {
            out.push(solution);
        }
    }
    out
}

/// Factor a target canonical gate through the fixed Berkeley entangler:
/// `B * V * B ~ target`, where the returned frame is the local middle gate
/// `V` in the magic basis.  Since both outer factors are 2+2, this dispatches
/// directly to the Pair22/Heron selector rather than the generic tail.
#[cfg(any(test, feature = "diagnostics"))]
pub fn factor_through_berkeley(target: [f64; 3]) -> Option<Mat4> {
    // Berkeley canonical coordinates are (1/2,1/4,0); convert through the
    // package's monodromy convention c=(m0+m1,m0+m2,m1+m2).
    const B: [f64; 3] = [0.375, 0.125, -0.125];
    let solution = solve(B, B, target);
    (solution.rung != Rung::Unsolved).then_some(solution.o)
}

/// Reduce a generic right entangler to the fixed Berkeley factor.  In canonical
/// coordinates choose `G = B + R` and `M = T - R`; because all canonical
/// factors are diagonal in the magic basis, the same local frame that realizes
/// `C · U · B ~ M` realizes `C · U · G ~ T`.  This is the recursive waypoint
/// reduction with the waypoint eliminated analytically.
#[cfg(feature = "diagnostics")]
pub fn solve_via_fixed_berkeley(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Solution> {
    const B: [f64; 3] = [0.375, 0.125, -0.125];
    solve_via_fixed_factor(c, g, t, B)
}

/// General fixed-factor form of the recursive reduction. `h` is a canonical
/// monodromy triple whose realization chart is known or separately certified.
#[cfg(feature = "diagnostics")]
pub fn solve_via_fixed_factor(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    h: [f64; 3],
) -> Option<Solution> {
    let gw = weyl_from_monodromy(g);
    let tw = weyl_from_monodromy(t);
    let bw = weyl_from_monodromy(h);
    let residual = [
        tw[0] - gw[0] + bw[0],
        tw[1] - gw[1] + bw[1],
        tw[2] - gw[2] + bw[2],
    ];
    let middle = [
        (residual[0] + residual[1] - residual[2]) * 0.5,
        (residual[0] - residual[1] + residual[2]) * 0.5,
        (-residual[0] + residual[1] + residual[2]) * 0.5,
    ];
    let child = solve(c, h, middle);
    if child.rung == Rung::Unsolved {
        return None;
    }
    let problem = PreparedSandwich::new(c, g, t);
    certificate::compiler_solution(&problem, child.o, child.rung, child.residual)
}

/// Re-certify an externally selected frame against the original sandwich.
/// This is intentionally strict: a frame found in a transformed factor chart
/// is useful only if it survives the caller's representatives.
#[cfg(feature = "diagnostics")]
pub fn certify_frame(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    o: Mat4,
    rung: Rung,
) -> Option<Solution> {
    let problem = PreparedSandwich::new(c, g, t);
    certificate::compiler_solution(&problem, o, rung, ACCEPT)
}

/// Evaluate one proposed waypoint for the virtual factorization
/// `G = B V B`. The caller supplies `M` (for example from a reachable-polytope
/// intersection); this routine realizes both B-children and applies the exact
/// endpoint-Grassmannian compatibility test. A small residual means the two
/// child certificates can be stitched with the fixed middle local `V`.
#[cfg(any(test, feature = "diagnostics"))]
pub fn solve_factorized_waypoint(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    waypoint: [f64; 3],
) -> Option<(Solution, Solution, Mat4, f64)> {
    const B: [f64; 3] = [0.375, 0.125, -0.125];
    let middle = factor_through_berkeley(g)?;
    let first = solve(c, B, waypoint);
    let second = solve(waypoint, B, t);
    let mut candidates = Vec::new();
    if first.rung != Rung::Unsolved && second.rung != Rung::Unsolved {
        candidates.push((first, second));
    }
    // The realization fiber is nontrivial.  If the production witnesses do
    // not glue, try the finite ordered chart transversal before declaring the
    // waypoint impossible.  This keeps the outer waypoint search unchanged
    // while making orientation selection explicit and deterministic.
    if candidates
        .first()
        .is_none_or(|(a, b)| a.rung == Rung::Unsolved || b.rung == Rung::Unsolved)
    {
        let fs = ordered_chart_solutions(c, B, waypoint);
        let ss = ordered_chart_solutions(waypoint, B, t);
        for a in fs {
            for b in &ss {
                candidates.push((a.clone(), b.clone()));
            }
        }
    }
    let mut best = None;
    for (first, second) in candidates {
        let first_problem = PreparedSandwich::new(c, B, waypoint);
        let Some(endpoint) = certificate::canonical_right_endpoint_gauge(&first_problem, &first.o)
        else {
            continue;
        };
        let expected = endpoint * middle;
        let mut residual = f64::INFINITY;
        for mask in 0..16 {
            let signs = [0, 1, 2, 3].map(|i| if (mask >> i) & 1 == 0 { 1.0 } else { -1.0 });
            if signs.iter().product::<f64>() < 0.0 {
                continue;
            }
            let s = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(
                &signs.map(|x| C::new(x, 0.0)),
            ));
            residual = residual.min(certificate::endpoint_plane_residual(
                &(s * expected),
                &second.o,
            ));
        }
        if best.as_ref().is_none_or(|(_, _, _, r)| residual < *r) {
            best = Some((first, second, middle, residual));
        }
        if residual <= ACCEPT {
            break;
        }
    }
    if best.as_ref().is_none_or(|(_, _, _, r)| *r > ACCEPT) {
        let fs = ordered_chart_solutions(c, B, waypoint);
        let ss = ordered_chart_solutions(waypoint, B, t);
        for first in fs {
            let first_problem = PreparedSandwich::new(c, B, waypoint);
            let Some(endpoint) =
                certificate::canonical_right_endpoint_gauge(&first_problem, &first.o)
            else {
                continue;
            };
            let expected = endpoint * middle;
            for second in &ss {
                let mut residual = f64::INFINITY;
                for mask in 0..16 {
                    let signs = [0, 1, 2, 3].map(|i| if (mask >> i) & 1 == 0 { 1.0 } else { -1.0 });
                    if signs.iter().product::<f64>() < 0.0 {
                        continue;
                    }
                    let s = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(
                        &signs.map(|x| C::new(x, 0.0)),
                    ));
                    residual = residual.min(certificate::endpoint_plane_residual(
                        &(s * expected),
                        &second.o,
                    ));
                }
                if best.as_ref().is_none_or(|(_, _, _, r)| residual < *r) {
                    best = Some((first.clone(), second.clone(), middle, residual));
                }
                if residual <= ACCEPT {
                    return Some((first, second.clone(), middle, residual));
                }
            }
        }
    }
    best
}

/// Diagnostic only: compare the two CS masses of the relative endpoint frame
/// against the Berkeley middle frame.  This is a necessary B-double-coset
/// check, not a generic waypoint certificate (the left stabilizer of a generic
/// waypoint is K_M, not K_B).
#[cfg(feature = "diagnostics")]
pub fn factorized_waypoint_mass_residual(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    waypoint: [f64; 3],
) -> Option<(f64, f64)> {
    let (first, second, middle, _) = solve_factorized_waypoint(c, g, t, waypoint)?;
    let first_problem = PreparedSandwich::new(c, [0.375, 0.125, -0.125], waypoint);
    let endpoint = certificate::canonical_right_endpoint_gauge(&first_problem, &first.o)?;
    let relative = endpoint.transpose() * second.o;
    let (s, d) = cs_masses(&relative);
    let (sm, dm) = cs_masses(&middle);
    Some(((s - sm).abs(), (d - dm).abs()))
}

/// Collapse a compatible factorized waypoint to a direct certified frame for
/// the original gate. Endpoint gauges are rephased onto the canonical target
/// diagonal before the fixed Berkeley middle frame is applied.
#[cfg(feature = "diagnostics")]
pub fn solve_factorized_waypoint_direct(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    waypoint: [f64; 3],
) -> Option<Solution> {
    let (first, _second, middle, compatibility) = solve_factorized_waypoint(c, g, t, waypoint)?;
    // The Plücker chart is a coordinate certificate, not the final spectral
    // certificate.  Near a chart boundary its subtraction error is amplified
    // by the endpoint Takagi gauge; allow a small numerical band here, while
    // retaining the strict original compiler certificate below.
    if compatibility > 1e-7 {
        return None;
    }
    const B: [f64; 3] = [0.375, 0.125, -0.125];
    let factor_problem = PreparedSandwich::new(B, B, g);
    let problem = PreparedSandwich::new(c, g, t);
    for branch in 0..2 {
        let Some((left, _right)) =
            certificate::endpoint_factorization_branch(&factor_problem, &middle, branch)
        else {
            continue;
        };
        let _left = if branch == 0 {
            left
        } else {
            let (sp, _) = certificate::rho_transport_for_collapse();
            left * sp
        };
        // `first.o` is already the frame multiplying the original right
        // entangler: the endpoint equation is `O₂ = R₁ V`, so
        // `C O₁ B V B ~ T`.  Multiplying O₁ by the factorization's left
        // Takagi frame double-counts the virtual BVB collapse and was the
        // reason even the planted B,B,B case failed.
        let candidate = first.o;
        for candidate in [candidate] {
            if let Some(solution) =
                certificate::compiler_solution(&problem, candidate, Rung::Chart, compatibility)
            {
                return Some(solution);
            }
        }
    }
    None
}

/// The chart tier alone: `PreparedSandwich` then `charts::solve_full`, with no
/// prefix rung, representative orbit, hold or adjacent-stratum lattice.
/// Diagnostics only: measures whether everything above the tail is load-bearing.
#[cfg(feature = "diagnostics")]
pub fn solve_charts_only(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    let problem = PreparedSandwich::new(c, g, t);
    charts::solve_full(&problem).unwrap_or_else(unsolved_solution)
}

/// Minimum within-pair root gap over recognized nonscalar paired inputs.
/// This is a floating-point applicability label, not a theorem-domain test.
#[cfg(feature = "diagnostics")]
pub fn paired_edge_scope(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<f64> {
    let problem = PreparedSandwich::new(c, g, t);
    [
        two_plus_two::paired_gap(&problem.left),
        two_plus_two::paired_gap(&problem.right),
    ]
    .into_iter()
    .flatten()
    .reduce(f64::min)
}

/// Candidate-only paired-input wall selector with R0266 backward transport.
/// Uses the public certificate; unsupported ranks and numerical misses return
/// `Unsolved`. No chart tail, dense Heron selector, or optimizer is called.
#[cfg(feature = "diagnostics")]
pub fn solve_paired_edges(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    solve_paired_edges_inner(c, g, t, true)
}

/// Ablation of `solve_paired_edges` retaining only original input roles.
#[cfg(feature = "diagnostics")]
pub fn solve_paired_edges_forward(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    solve_paired_edges_inner(c, g, t, false)
}

#[cfg(feature = "diagnostics")]
fn solve_paired_edges_inner(c: [f64; 3], g: [f64; 3], t: [f64; 3], backward: bool) -> Solution {
    let problem = PreparedSandwich::new(c, g, t);
    two_plus_two::solve_paired_edges_with(&problem, backward, |o, residual| {
        compiler_solution(&problem, o, Rung::Pair22, residual)
    })
    .unwrap_or_else(unsolved_solution)
}

fn solve_inner(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    let tp = prof::start();
    let problem = PreparedSandwich::new(c, g, t);
    prof::rec(prof::SEG_PREPARE, tp);
    // A scalar factor makes the realization fibre empty or all of SO(4):
    // `alpha O B O^T` is similar to `alpha B`, and likewise on the right.
    // This includes the identity edge without a separate tolerance branch.
    if let Some(solution) = solve_scalar_factor(&problem) {
        return solution;
    }
    // Exact 3+1 factors have a rank-one spectral-measure selector.
    // Keep it ahead of the generic multiplicity cascade.
    if let Some(solution) = solve_rank_one_31(&problem) {
        return solution;
    }
    // The exact-stratum prefix (walls, paired and 3+1 spectra, the two
    // one-sided accelerators), then the chart tail.  The first certified
    // frame is returned; the certificate is the acceptance test, so a better
    // residual is never hunted.
    if let Some(solution) = solve_prefix(&problem) {
        return solution;
    }
    // The tail: the standalone cascade under the production certificate: charts
    // (fast probes, wall rung, exact selection over the canonical atlas), the exact
    // leaves, and a second pass on spectra snapped to their strata.  Closed-form
    // frames only.
    let tct = prof::start();
    let out = charts::solve_full(&problem).unwrap_or_else(unsolved_solution);
    prof::rec(prof::CHART_TIER, tct);
    out
}

fn solve_scalar_factor(problem: &PreparedSandwich) -> Option<Solution> {
    let (scalar, other) = if problem.strata.c == SpectrumKind::Scalar4 {
        (problem.left[0], problem.right)
    } else if problem.strata.g == SpectrumKind::Scalar4 {
        (problem.right[0], problem.left)
    } else {
        return None;
    };
    let expected: [C; 4] = other.map(|z| scalar * z);
    let o = Mat4::identity();
    let mut best = f64::INFINITY;
    for target in &problem.target_roots {
        for permutation in *PERMS24 {
            let error = (0..4)
                .map(|i| (expected[i] - target[permutation[i]]).norm())
                .fold(0.0, f64::max);
            best = best.min(error);
        }
    }
    (best <= ACCEPT)
        .then(|| compiler_solution(problem, o, Rung::Vertex, best))
        .flatten()
}

/// Direct exact `3+1` selector (R0028).
///
/// When `d=diag(a,a,a,c)`, the other factor `B` enters only through the
/// grouped masses of `u=O^T e_4`.  Differentiating the rank-one determinant
/// identity at `a*beta_g` recovers those masses linearly.  This routine is
/// deliberately before the generic cascade: it has no chart variables,
/// root search, or continuous selection.
fn solve_rank_one_31(problem: &PreparedSandwich) -> Option<Solution> {
    if problem.strata.c == SpectrumKind::Triple31
        && problem.strata.c_proximity == problem::SpectrumProximity::Exact
    {
        if let Some(hit) = solve_rank_one_side(
            problem,
            problem.left,
            problem.right,
            false,
            64.0 * f64::EPSILON,
            Rung::RankOne31,
        ) {
            return Some(hit);
        }
    }
    if problem.strata.g == SpectrumKind::Triple31
        && problem.strata.g_proximity == problem::SpectrumProximity::Exact
    {
        if let Some(hit) = solve_rank_one_side(
            problem,
            problem.right,
            problem.left,
            true,
            64.0 * f64::EPSILON,
            Rung::RankOne31,
        ) {
            return Some(hit);
        }
    }
    if problem.strata.c == SpectrumKind::Triple31
        && problem.strata.c_proximity == problem::SpectrumProximity::Near
    {
        if let Some(hit) = solve_rank_one_side(
            problem,
            problem.left,
            problem.right,
            false,
            1.0e-7,
            Rung::NearRankOne31,
        ) {
            return Some(hit);
        }
    }
    if problem.strata.g == SpectrumKind::Triple31
        && problem.strata.g_proximity == problem::SpectrumProximity::Near
    {
        if let Some(hit) = solve_rank_one_side(
            problem,
            problem.right,
            problem.left,
            true,
            1.0e-7,
            Rung::NearRankOne31,
        ) {
            return Some(hit);
        }
    }
    None
}

fn solve_rank_one_side(
    problem: &PreparedSandwich,
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

/// Convert a swapped-orientation solution back to the original edge:
/// `u` solves `(C,G,T)` iff `uᵀ` solves `(G,C,T)`; in the magic basis
/// `mb(uᵀ) = E·mb(u)ᵀ·E` with `E = diag(1,-1,1,-1)`.  The converted frame is
/// re-certified against the original lift; a conversion that fails
/// certification falls through to the next representative.
/// Convert a certificate built with central-rho gate representatives back to
/// the caller's canonical representatives.  In the magic basis,
///
/// `D(rho(w)) = i L D(w) R`,
///
/// Every microsecond-scale exact stage: support strata, Klein, the
/// multiplicity formulas, the 1+3 walls, the boundary accelerators, and the
/// float pass of the dense 1+3 selector.
fn solve_prefix(problem: &PreparedSandwich) -> Option<Solution> {
    let tpre = prof::start();
    // A vertex or one-Givens edge preserves routed eigenvalues a_i*g_j. Test
    // that necessary spectral signature before enumerating support incidences.
    // The gate carries its branch masks into the edge solve, so the certificate
    // is computed only once.  An interior stratum certificate kills every
    // facet-tied support construction (vertex/edge/1+3 via the empty gate,
    // face via its own guard) outright.
    let teg = prof::start();
    let edge_gate = support_strata::edge_gate(&problem.routed, &problem.target_roots);
    prof::rec(prof::SEG_EDGEGATE, teg);
    let tvx = prof::start();
    let mut support_perms = None;
    if let Some(viable) = edge_gate.edge.as_ref() {
        let cand = *PERMS24;
        for p in cand {
            // A vertex needs all four routed roots on the same target branch.
            let branch_mask = (0..4).fold(0b11u8, |mask, k| mask & viable[k][p[k]]);
            if branch_mask == 0 {
                continue;
            }
            let r = perm_vertex_residual(&problem.left, &problem.right, &problem.targets, &p);
            if r < ACCEPT {
                if let Some(solution) = compiler_solution(problem, signed_perm(p), Rung::Vertex, r)
                {
                    return Some(solution);
                }
            }
        }
        support_perms = Some(cand);
    }
    prof::rec(prof::SEG_VERTEX, tvx);
    prof::rec(prof::PRELUDE, tpre);
    // Exhaust the lower support strata before a broader section can cannibalize
    // their cheaper, better-conditioned formulas.
    if let (Some(viable), Some(cand)) = (edge_gate.edge.as_ref(), support_perms.as_ref()) {
        let tp = prof::start();
        let hit = support_strata::solve_edge(
            &problem.left,
            &problem.right,
            &problem.dc,
            &problem.lam,
            &problem.targets,
            &problem.routed,
            viable,
            cand,
        );
        prof::rec(prof::EDGE, tp);
        if let Some((o, r)) = hit {
            if let Some(solution) = compiler_solution(problem, o, Rung::Edge, r) {
                return Some(solution);
            }
        }
    }
    let tp = prof::start();
    let hit = support_strata::solve_face(
        &problem.left,
        &problem.right,
        &problem.dc,
        &problem.lam,
        &problem.target_roots,
        &problem.targets,
    );
    prof::rec(prof::FACE, tp);
    if let Some((o, r)) = hit {
        if let Some(solution) = compiler_solution(problem, o, Rung::Face, r) {
            return Some(solution);
        }
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
    if let Some((o, r)) = klein_hit {
        if let Some(solution) = compiler_solution(problem, o, Rung::Klein, r) {
            return Some(solution);
        }
    }
    let target_repeated = problem.strata.target.iter().any(|kind| kind.is_repeated());
    let mut resonance_best = None;
    if problem.strata.has_confluence() && target_repeated {
        let exact = resonance::solve_with(
            &problem.left,
            &problem.right,
            &problem.target_roots,
            &problem.dc,
            &problem.lam,
            &problem.targets,
            |o, residual| {
                let solution = compiler_solution(problem, o, Rung::Radical, residual)?;
                if solution.residual < 1e-12 {
                    return Some(solution);
                }
                if resonance_best
                    .as_ref()
                    .is_none_or(|candidate: &Solution| solution.residual < candidate.residual)
                {
                    resonance_best = Some(solution);
                }
                None
            },
        );
        if let Some(solution) = exact {
            return Some(solution);
        }
    }
    // Multiplicity formulas own only exact confluent signatures and run after
    // the cheaper support and Klein sections have declined.
    if let Some((o, r, rung)) = support_strata::solve_confluent(
        &problem.left,
        &problem.right,
        &problem.target_roots,
        &problem.dc,
        &problem.lam,
        &problem.targets,
        problem.strata,
    ) {
        if let Some(solution) = compiler_solution(problem, o, rung, r) {
            return Some(solution);
        }
    }
    if let Some(solution) = resonance_best {
        return Some(solution);
    }
    let has_one_plus_three = edge_gate.exact.iter().flatten().any(|&mask| mask != 0);
    if has_one_plus_three {
        if let Some((o, residual)) = one_plus_three::solve_walls(
            &problem.left,
            &problem.right,
            &edge_gate.exact,
            &problem.dc,
            &problem.lam,
            &problem.targets,
        ) {
            if let Some(solution) = compiler_solution(problem, o, Rung::OnePlusThree, residual) {
                return Some(solution);
            }
        }
    }
    if let Some((o, residual, rung)) = solve_boundary_accelerators(problem) {
        if let Some(solution) = compiler_solution(problem, o, rung, residual) {
            return Some(solution);
        }
    }
    if has_one_plus_three {
        if let Some((o, residual)) = one_plus_three::solve_dense(
            &problem.left,
            &problem.right,
            &edge_gate.exact,
            &problem.dc,
            &problem.lam,
            &problem.targets,
        ) {
            if let Some(solution) = compiler_solution(problem, o, Rung::OnePlusThree, residual) {
                return Some(solution);
            }
        }
    }
    None
}

// =================== RANK RUNG: the degenerate-w spectral inverse (symmetric_horn s13/s16) ===================
// A symmetric unitary S = O·Λ·Oᵀ has REAL eigenvectors, so U₁ = O·Λ·Oᵀ = V diag(spec) Vᵀ with V real.
// A DEGENERATE target w (the sliver) collapses the unknown V to a diagonal-plus-low-rank inverse-
// eigenvalue problem (rank 4−m₀, m₀ = top w-multiplicity); the Givens "fold" was a chart artifact.
// Closed form per multiplicity pattern -- no GN, no search. Proven in symmetric_horn s12-s16
// ([[sliver-rank-structured-inverse-solved]]). Tried after the cheap chart cascade declines.

// Test-only refinement counter for the bracketed real-root solver.
#[cfg(test)]
thread_local! {
    static REFINE_ITERS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// ===================== SLIVER RUNG: the peel ('none'-gauge) =====================
// For a 'none'-gauge 4-Givens word the gauge γ=cos(2φ_g) is slaved (degree 1) and the exact-
// degenerate sliver is the fold of a 1-dim family, located by root conditions (eliminant double
// root + e3-invariant tangent). Construction = companion-eigenvalue roots + linear
// back-substitution + a small parabola fit.

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stderr)]
    use super::*;
    include!("sparse_candidate_tests.rs");

    #[test]
    fn clustered_chart_conditioning_retains_original_input_roots() {
        // Exact constructor triples, including corpus roundoff. The first
        // exercises a narrow feasible quartic interval; the others exercise
        // cancellation of the unit/determinant relations.
        let cases = [
            (
                [
                    0.22275108585431175,
                    0.034783033435165096,
                    -0.034783033434665094,
                ],
                [
                    3.940164785237606e-5,
                    3.719643390931126e-5,
                    -1.5999729614063347e-5,
                ],
                [0.465164848419121, 0.27726681794701724, -0.2772149763792984],
            ),
            (
                [0.2500000203855839, 0.24999997790254228, 0.24999997209745775],
                [
                    3.162039039450218e-5,
                    2.875089443661313e-5,
                    8.008324774382537e-6,
                ],
                [
                    0.25002876924039463,
                    0.25000799499260795,
                    0.24993161651888982,
                ],
            ),
            (
                [0.2500000125082198, 0.24999999305716747, 0.24999995694283256],
                [0.2500061412329326, 0.2500036050596355, 0.24994639494036455],
                [
                    4.387932868452071e-5,
                    6.155433609517091e-6,
                    3.6078516658932802e-6,
                ],
            ),
            (
                [0.46986120830703, 0.46986120819175, -0.40958362757526],
                [0.24787741830247, 0.24762903238808998, 0.24737096761191002],
                [0.226980135397151, 0.21769777836797044, 0.2174181051570272],
            ),
        ];
        for (index, (c, g, t)) in cases.into_iter().enumerate() {
            let problem = PreparedSandwich::new(c, g, t);
            let solution = solve(c, g, t);
            assert_ne!(solution.rung, Rung::Unsolved, "case {index}");
            assert!(frame_metrics(&solution.o).unwrap().within(1e-8));
            assert!(solution.o.determinant().re > 0.0);
            // Independent direct diagonalization against the ORIGINAL roots,
            // not the normalized candidate-generation coefficients.
            let actual = eig4(&sandwich_master(&problem, &solution.o));
            let error = problem
                .target_roots
                .iter()
                .flat_map(|target| {
                    PERMS24.iter().map(move |permutation| {
                        (0..4)
                            .map(|i| (actual[i] - target[permutation[i]]).norm())
                            .fold(0.0f64, f64::max)
                    })
                })
                .fold(f64::INFINITY, f64::min);
            assert!(error < 1e-8, "case {index}: original root error {error:e}");
        }
    }

    #[test]
    fn eighth_cx_structured_vertex_returns_without_the_dense_tail() {
        let solution = solve(
            [0.03125, 0.03125, -0.03125],
            [0.34375, -0.09375, -0.09375],
            [0.375, -0.125, -0.125],
        );
        assert_eq!(solution.rung, Rung::Vertex);
        assert!(solution.residual < 1e-12);
    }

    #[test]
    fn near_swap_repeated_gate_witness_is_realized() {
        // Public-pipeline regression: the fast atlas declines this feasible
        // sandwich because C is near SWAP and G has an exact repeated Weyl
        // coordinate.  The target was formed by an explicit local sandwich.
        let solution = solve(
            [
                0.251_567_175_907_25,
                0.248_432_824_092_75,
                0.246_902_391_208_76,
            ],
            [
                0.033_635_662_418_36,
                0.026_914_048_264_11,
                0.026_914_048_264_11,
            ],
            [
                0.278_977_270_160_45,
                0.277_417_719_732_49,
                0.160_565_542_174_26,
            ],
        );
        assert_ne!(solution.rung, Rung::Unsolved);
        assert!(
            solution.residual < 1e-8,
            "residual={:.3e}",
            solution.residual
        );
    }

    #[test]
    fn near_identity_gate_roundoff_is_realized() {
        let c = [
            0.423_503_315_564_57,
            0.290_670_418_709_1,
            -0.137_677_049_838_24,
        ];
        let g = [3.8e-13, 1.5e-13, 1.0e-13];
        let t = [
            0.423_503_315_564_27,
            0.290_670_418_709_24,
            -0.137_677_049_838_22,
        ];
        let problem = PreparedSandwich::new(c, g, t);
        let mut minimum = f64::INFINITY;
        let mut certified = 0usize;
        for permutation in *PERMS24 {
            let residual = perm_vertex_residual(
                &problem.left,
                &problem.right,
                &problem.targets,
                &permutation,
            );
            minimum = minimum.min(residual);
            certified += usize::from(
                compiler_solution(&problem, signed_perm(permutation), Rung::Vertex, residual)
                    .is_some(),
            );
        }
        let solution = solve(c, g, t);
        assert_ne!(
            solution.rung,
            Rung::Unsolved,
            "minimum vertex residual={minimum:.3e}, certified={certified}",
        );
    }

    #[test]
    fn near_scalar_triple_collision_uses_verified_takagi_fallback() {
        // A projector frame built from the known target eigenvalues can be
        // nonzero but wrong at this near-triple collision. Takagi extraction
        // must verify that frame and continue to the eigensolver fallback.
        let solution = solve(
            [
                0.076_301_851_551_263_62,
                0.076_301_851_551_263_62,
                0.076_301_851_551_263_62,
            ],
            [
                4.853_548_865_645_266_6e-8,
                8.227_351_897_810_43e-9,
                -5.298_329_210_715_759e-9,
            ],
            [
                0.076_301_900_013_802_94,
                0.076_301_850_032_083_01,
                0.076_301_803_530_515_3,
            ],
        );
        assert_ne!(solution.rung, Rung::Unsolved);
        assert!(solution.residual < 1e-8);
    }

    #[test]
    fn public_solution_gate_rejects_nonframes() {
        let mut nonorthogonal = Mat4::identity();
        nonorthogonal[(0, 0)] = C::new(1.0 + 1e-6, 0.0);
        assert!(certified_solution(nonorthogonal, Rung::Interior, 0.0).is_none());

        let mut complex = Mat4::identity();
        complex[(0, 0)].im = 1e-6;
        assert!(certified_solution(complex, Rung::Interior, 0.0).is_none());

        assert!(certified_solution(Mat4::identity(), Rung::Interior, ACCEPT * 2.0).is_none());
        let accepted = certified_solution(Mat4::identity(), Rung::Interior, 0.0)
            .expect("a real SO(4) frame with zero residual must pass");
        assert_eq!(accepted.rung, Rung::Interior);
    }

    #[test]
    fn identity_right_factor_is_exact_edge() {
        let c = [0.17, 0.04, -0.09];
        let hit = solve(c, [0.0, 0.0, 0.0], c);
        assert_eq!(hit.rung, Rung::Vertex);
        assert!(hit.residual < 1e-12);
        let miss = solve(c, [0.0, 0.0, 0.0], [0.21, 0.03, -0.08]);
        assert_eq!(miss.rung, Rung::Unsolved);
    }

    #[test]
    fn berkeley_factorization_routes_generic_targets_through_pair22() {
        let targets = [
            [0.25, 0.25, -0.25],
            [0.31, 0.17, -0.09],
            [0.22, 0.08, 0.03],
            [0.41, 0.19, -0.12],
        ];
        for target in targets {
            let middle = factor_through_berkeley(target)
                .expect("B-V-B factorization should have a Pair22 witness");
            let problem =
                PreparedSandwich::new([0.375, 0.125, -0.125], [0.375, 0.125, -0.125], target);
            let residual =
                compound_residual(&problem.dc, &problem.lam, &middle, &problem.targets[0]).min(
                    compound_residual(&problem.dc, &problem.lam, &middle, &problem.targets[1]),
                );
            assert!(residual < 1e-8, "target={target:?}, residual={residual:e}");
        }
    }

    #[test]
    fn berkeley_factorization_grid_has_no_generic_pair22_holes() {
        // Valid Weyl points sampled in the interior and near all three walls,
        // converted to the package's monodromy coordinates.
        let weyl_points = [
            [0.10, 0.07, 0.02],
            [0.20, 0.13, 0.04],
            [0.31, 0.17, 0.09],
            [0.42, 0.21, 0.08],
            [0.49, 0.24, 0.01],
            [0.26, 0.26, 0.12],
            [0.38, 0.30, 0.18],
            [0.50, 0.25, 0.20],
        ];
        for c in weyl_points {
            let target = [
                0.5 * (c[0] + c[1] - c[2]),
                0.5 * (c[0] + c[2] - c[1]),
                0.5 * (c[1] + c[2] - c[0]),
            ];
            let middle = factor_through_berkeley(target)
                .expect("interior Weyl target should factor through Berkeley B");
            let problem =
                PreparedSandwich::new([0.375, 0.125, -0.125], [0.375, 0.125, -0.125], target);
            let residual = problem
                .targets
                .iter()
                .map(|tau| compound_residual(&problem.dc, &problem.lam, &middle, tau))
                .fold(f64::INFINITY, f64::min);
            assert!(residual < 1e-8, "weyl={c:?}, residual={residual:e}");
        }
    }

    #[test]
    fn right_endpoint_gauge_reconstructs_an_orthogonal_frame() {
        let c = [0.5, 0.25, -0.25];
        let b = [0.375, 0.125, -0.125];
        let m = [0.25, 0.25, -0.25];
        let solution = solve(c, b, m);
        let problem = PreparedSandwich::new(c, b, m);
        let gauge = certificate::right_endpoint_gauge(&problem, &solution.o)
            .expect("certified child must expose its endpoint gauge");
        let real = gauge.map(|z| z.re);
        assert!((real.transpose() * real - nalgebra::Matrix4::<f64>::identity()).norm() < 1e-8);
        assert!((real.determinant().abs() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn factorized_middle_constraint_is_exact_after_endpoint_gauge() {
        let c = [0.5, 0.25, -0.25];
        let b = [0.375, 0.125, -0.125];
        let m = [0.25, 0.25, -0.25];
        let first = PreparedSandwich::new(c, b, m);
        let first_solution = solve(c, b, m);
        let endpoint = certificate::canonical_right_endpoint_gauge(&first, &first_solution.o)
            .expect("first child must expose endpoint gauge");
        let middle = Mat4::identity();
        let second_frame = endpoint * middle;
        let residual = certificate::factorized_middle_residual(
            &first,
            &first_solution.o,
            &second_frame,
            &middle,
        )
        .expect("gauge extraction must succeed");
        assert!(residual < 1e-8, "residual={residual:e}");
        let plane_residual = certificate::factorized_middle_plane_residual(
            &first,
            &first_solution.o,
            &second_frame,
            &middle,
        )
        .expect("gauge extraction must succeed");
        assert!(plane_residual < 1e-8, "plane residual={plane_residual:e}");
    }

    #[test]
    fn factorized_waypoint_api_evaluates_a_proposed_point() {
        let b = [0.375, 0.125, -0.125];
        let candidate = solve_factorized_waypoint(b, b, b, b)
            .expect("the B-B-B waypoint is inside the Pair22 child domains");
        assert!(candidate.3.is_finite());
        let middle = factor_through_berkeley(b).unwrap();
        let fp = PreparedSandwich::new(b, b, b);
        let (left, right) = certificate::endpoint_factorization(&fp, &middle).unwrap();
        let db = dphase(weyl_from_monodromy(b));
        let h = db * middle * db;
        let factor_error = (h - left * db * right)
            .iter()
            .map(|z| z.norm())
            .fold(0.0, f64::max);
        assert!(factor_error < 1e-8);

        let first_problem = PreparedSandwich::new(b, b, b);
        let endpoint = certificate::canonical_right_endpoint_gauge(&first_problem, &candidate.0.o)
            .expect("canonical endpoint gauge");
        let second_frame = endpoint * middle;
        let virtual_master = mmat(&db, &(db * db), &second_frame);
        let direct_master = mmat(&db, &(db * db), &(candidate.0.o * left));
        let error = symfn(&virtual_master)
            .iter()
            .zip(symfn(&direct_master).iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(error < 1e-8, "collapse error={error:e}");
    }

    #[test]
    fn rho_transport_identity_matches_canonical_diagonals() {
        let w = [0.41, 0.17, 0.06];
        let (sp, ps) = certificate::rho_transport_for_collapse();
        let lhs = dphase(rho_weyl(w));
        let rhs = (sp * dphase(w) * ps).map(|entry| C::new(0.0, 1.0) * entry);
        let error = (lhs - rhs)
            .iter()
            .map(|entry| entry.norm())
            .fold(0.0, f64::max);
        assert!(error < 1e-12, "rho transport error={error:e}");
    }

    #[test]
    fn scalar_left_factor_is_exact_edge() {
        let g = [0.17, 0.04, -0.09];
        let hit = solve([0.0, 0.0, 0.0], g, g);
        assert_eq!(hit.rung, Rung::Vertex);
        assert!(hit.residual < 1e-12);
        let miss = solve([0.0, 0.0, 0.0], g, [0.21, 0.03, -0.08]);
        assert_eq!(miss.rung, Rung::Unsolved);
    }

    #[test]
    fn exact_right_triple31_has_a_certified_identity_witness() {
        // g=(1/10,1/10,1/10) maps to Weyl (1/5,1/5,1/5), whose magic-basis
        // spectrum has multiplicity 3+1.  The target is the diagonal product
        // of c=(1/20,1/50,-1/100) and this g, so O=I is a known witness.
        let problem =
            PreparedSandwich::new([0.05, 0.02, -0.01], [0.1, 0.1, 0.1], [0.15, 0.12, 0.09]);
        assert_eq!(problem.strata.g, SpectrumKind::Triple31);
        let solution = solve([0.05, 0.02, -0.01], [0.1, 0.1, 0.1], [0.15, 0.12, 0.09]);
        assert_ne!(solution.rung, Rung::Unsolved);
        assert!(
            solution.residual < 1e-12,
            "residual {:.3e}",
            solution.residual
        );
    }

    #[test]
    fn rank_one_selector_recovers_a_nontrivial_right_triple31_frame() {
        // Build the target from a genuine noncommuting frame, then replace the
        // dummy target in PreparedSandwich by its exact known spectrum.  This
        // isolates the new selector from Weyl-coordinate inversion and tests
        // the mass inverse, Householder completion, right-side transpose, and
        // public characteristic certificate together.
        let mut problem =
            PreparedSandwich::new([0.05, 0.02, -0.01], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]);
        assert_eq!(problem.strata.g, SpectrumKind::Triple31);
        let mut planted = Mat4::identity();
        let theta = 0.37f64;
        let (s, c) = theta.sin_cos();
        planted[(0, 0)] = C::new(c, 0.0);
        planted[(0, 2)] = C::new(-s, 0.0);
        planted[(2, 0)] = C::new(s, 0.0);
        planted[(2, 2)] = C::new(c, 0.0);
        let target_matrix = sandwich_master(&problem, &planted);
        let roots = eig4(&target_matrix);
        let coefficients = esym4(roots);
        problem.target_roots = [roots, roots];
        problem.targets = [coefficients, coefficients];
        let solution = solve_rank_one_31(&problem).expect("rank-one 3+1 selector");
        assert_eq!(solution.rung, Rung::RankOne31);
        assert!(
            solution.residual < 1e-10,
            "residual {:.3e}",
            solution.residual
        );
    }

    #[test]
    fn rank_one_selector_recovers_a_nontrivial_left_triple31_frame() {
        let mut problem =
            PreparedSandwich::new([0.1, 0.1, 0.1], [0.05, 0.02, -0.01], [0.0, 0.0, 0.0]);
        assert_eq!(problem.strata.c, SpectrumKind::Triple31);
        let mut planted = Mat4::identity();
        let theta = 0.29f64;
        let (s, c) = theta.sin_cos();
        planted[(0, 0)] = C::new(c, 0.0);
        planted[(0, 2)] = C::new(-s, 0.0);
        planted[(2, 0)] = C::new(s, 0.0);
        planted[(2, 2)] = C::new(c, 0.0);
        let target_matrix = sandwich_master(&problem, &planted);
        let roots = eig4(&target_matrix);
        let coefficients = esym4(roots);
        problem.target_roots = [roots, roots];
        problem.targets = [coefficients, coefficients];
        let solution = solve_rank_one_31(&problem).expect("rank-one 3+1 selector");
        assert_eq!(solution.rung, Rung::RankOne31);
        assert!(
            solution.residual < 1e-10,
            "residual {:.3e}",
            solution.residual
        );
    }

    #[test]
    fn exact_dense_rational_counterexample_requires_complete_fallback() {
        // This is the exact fixture from
        // realization_problem/docs/exact_finite_axis_counterexample.md, encoded with one
        // consistent monodromy convention and then projected into the production alcove.
        // The older disabled regression accidentally conjugated C and G but not T; folding
        // that malformed triple produced an unrelated easy Interior case.
        let solution = solve(
            [
                0.320_387_857_027_5,
                0.265_146_172_188_37,
                -0.066_006_359_790_49,
            ],
            [
                0.293_629_518_344_32,
                0.236_744_755_386_73,
                -0.151_493_215_321_9,
            ],
            [
                0.455_019_768_623_71,
                0.049_468_366_219_53,
                0.029_722_997_536_99,
            ],
        );
        assert_ne!(solution.rung, Rung::Unsolved);
        assert!(solution.residual < ACCEPT);
    }

    #[test]
    fn exact_dense_rational_counterexample_is_never_falsely_accepted() {
        let solution = solve(
            [
                0.320_387_857_027_5,
                0.265_146_172_188_37,
                -0.066_006_359_790_49,
            ],
            [
                0.293_629_518_344_32,
                0.236_744_755_386_73,
                -0.151_493_215_321_9,
            ],
            [
                0.455_019_768_623_71,
                0.049_468_366_219_53,
                0.029_722_997_536_99,
            ],
        );
        if solution.rung == Rung::Unsolved {
            assert!(solution.residual.is_infinite());
            return;
        }
        assert!(solution.residual < ACCEPT);
        let gram = solution.o.transpose() * solution.o - Mat4::identity();
        assert!(gram.norm() < 1e-10);
        assert!((solution.o.determinant().re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn radical_residue_frame_is_orthogonal_before_public_acceptance() {
        // This exact dyadic corpus row used to return a nominal Radical frame
        // with Gram residual 8.27e-6.  Its compound spectral residual looked
        // small only because that formula assumes orthogonality.  The radical
        // callback must reject the raw residue columns and recover the real
        // eigenframe of the already-constructed symmetric sandwich matrix.
        let solution = solve(
            [0.40625, 0.34375, -0.28125],
            [0.5, 0.25, -0.25],
            [0.21875, 0.03125, 0.03125],
        );
        assert_eq!(solution.rung, Rung::Radical);
        assert!(solution.residual < ACCEPT);
        let real = solution.o.map(|value| value.re);
        let gram = real.transpose() * real - nalgebra::Matrix4::identity();
        let gram_residual = gram
            .iter()
            .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
        assert!(gram_residual < 2e-10, "Gram residual={gram_residual:.3e}");
        assert!((real.determinant() - 1.0).abs() < 1e-10);
    }

    #[test]
    #[ignore = "tail benchmark for the consistently folded exact dense fixture"]
    fn benchmark_exact_dense_rational_fixture() {
        let input = (
            [
                0.320_387_857_027_5,
                0.265_146_172_188_37,
                -0.066_006_359_790_49,
            ],
            [
                0.293_629_518_344_32,
                0.236_744_755_386_73,
                -0.151_493_215_321_9,
            ],
            [
                0.455_019_768_623_71,
                0.049_468_366_219_53,
                0.029_722_997_536_99,
            ],
        );
        let cold_started = std::time::Instant::now();
        std::hint::black_box(solve(input.0, input.1, input.2));
        let cold = cold_started.elapsed().as_nanos() as f64 / 1000.0;
        let mut samples = Vec::with_capacity(31);
        for _ in 0..31 {
            let started = std::time::Instant::now();
            let solution = solve(input.0, input.1, input.2);
            std::hint::black_box(solution);
            samples.push(started.elapsed().as_nanos() as f64 / 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        let average = samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "exact dense production: cold={cold:.3} us average={average:.3} us slowest={:.3} us median={:.3} us",
            samples[samples.len() - 1],
            samples[samples.len() / 2],
        );
    }

    #[test]
    fn hardened_resonance_owns_the_retired_selector_unique_rows() {
        // The three rows whose unique coverage once justified the 1+3 dense
        // selector (feasible_linspace 107415/117216/619511): doubled-input x
        // doubled-target pairs served by the conic-pencil sections, and the
        // C = G tripled-target row served by the dual rank-one construction.
        for (c, g, t) in [
            (
                [0.21875, 0.21875, -0.03125],
                [0.40625, 0.28125, -0.21875],
                [0.375, -0.0625, -0.0625],
            ),
            (
                [0.25, 0.0625, -0.0625],
                [0.25, 0.0625, -0.0625],
                [0.375, -0.125, -0.125],
            ),
            (
                [0.40625, 0.28125, -0.21875],
                [0.21875, 0.21875, -0.03125],
                [0.375, -0.0625, -0.0625],
            ),
        ] {
            let solution = solve(c, g, t);
            assert_ne!(solution.rung, Rung::Unsolved, "{c:?} {g:?} {t:?}");
            assert!(
                solution.residual < 1e-12,
                "residual {:.3e}",
                solution.residual
            );
        }
    }

    #[test]
    fn routed_zero_wall_row_survives_the_retired_dense_selector() {
        // feasible_linspace row 329583: one routed eigenvalue splits off and
        // the remaining dense SO(3) block lies on a two-Givens wall.  The
        // retired 1+3 dense selector once owned this row; the hardened
        // resonance/secular dispatch must keep it machine-precise.
        let solution = solve(
            [0.3125, 0.1875, -0.1875],
            [0.375, 0.3125, -0.1875],
            [0.375, 0.125, 0.0],
        );
        assert_ne!(solution.rung, Rung::Unsolved);
        assert!(solution.residual < 1e-12);
    }

    #[test]
    fn every_weyl_vertex_passes_the_routed_support_gate() {
        let a2 = std::array::from_fn(|k| C::from_polar(1.0, [0.13, 0.47, 1.01, 1.73][k]));
        let g2 = std::array::from_fn(|k| C::from_polar(1.0, [0.22, 0.61, 1.19, 2.03][k]));
        let routed = std::array::from_fn(|i| std::array::from_fn(|j| a2[i] * g2[j]));
        for p in PERMS24.iter() {
            let spec = std::array::from_fn(|k| a2[k] * g2[p[k]]);
            let targets = [spec, spec];
            let support = support_strata::edge_gate(&routed, &targets);
            let viable = support.edge.expect("a vertex is a special edge");
            assert_ne!((0..4).fold(0b11u8, |mask, k| mask & viable[k][p[k]]), 0);
            assert!(perm_vertex_residual(&a2, &g2, &[esym4(spec), esym4(spec)], p) < 1e-12);
        }
    }

    #[test]
    fn radical_dispatch_distinguishes_multiplicity_from_near_degeneracy() {
        let exact = [
            c(1.0, 0.0),
            c(1.0, 0.0),
            C::from_polar(1.0, 0.7),
            C::from_polar(1.0, 1.4),
        ];
        let near = [
            c(1.0, 0.0),
            C::from_polar(1.0, 1e-3),
            C::from_polar(1.0, 0.7),
            C::from_polar(1.0, 1.4),
        ];
        assert_eq!(spectrum_kind(&exact), SpectrumKind::Pair211);
        assert_eq!(spectrum_kind(&near), SpectrumKind::Distinct);
    }

    /// The master object `M = D_C·O·Λ·Oᵀ·D_C` reduces (conjugate by D_C⁻¹) to the multiplicative-Horn
    /// product `U₁·U₂`, with `U₁ = O·Λ·Oᵀ` a SYMMETRIC unitary (the Takagi / U(4)/O(4) locus) and
    /// `U₂ = D_C²` diagonal. spec(U₁)=spec(Λ) (G data), spec(U₂)=spec(D_C²) (C data), spec(U₁U₂)=target.
    /// This pins the problem's real home: the symmetric-space multiplicative Horn -- not generic SO(4).
    #[test]
    fn master_object_is_symmetric_unitary_product() {
        let dc = dphase([0.31, 0.19, 0.11]); // D_C
        let lam = {
            let dg = dphase([0.37, 0.23, 0.13]);
            dg * dg
        }; // Λ = D_G²
        let o = givens(0, 1, 0.7) * givens(2, 3, 1.1) * givens(1, 2, 0.4) * givens(0, 3, 0.9);
        assert!(
            (o.map(|z| z.re).determinant() - 1.0).abs() < 1e-12,
            "O not SO(4)"
        );
        assert!(o.iter().all(|z| z.im.abs() < 1e-14), "O not real");
        let m = mmat(&dc, &lam, &o);
        let u1 = o * lam * o.transpose(); // O Λ Oᵀ
        let u2 = dc * dc; // D_C²
        let sort = |mut v: [C; 4]| {
            v.sort_by(|a, b| a.arg().partial_cmp(&b.arg()).unwrap());
            v
        };
        let maxerr =
            |a: [C; 4], b: [C; 4]| (0..4).map(|i| (a[i] - b[i]).norm()).fold(0.0, f64::max);
        // (1) the reduction: spec(M) == spec(U₁U₂).
        let d = maxerr(sort(eig4(&m)), sort(eig4(&(u1 * u2))));
        assert!(d < 1e-10, "spec(M) != spec(U₁U₂): {d:.2e}");
        // (2) U₁ is a SYMMETRIC unitary (the real-structure constraint -- the open part of the construction).
        let sym = (u1 - u1.transpose()).norm();
        let uni = (u1.adjoint() * u1 - Mat4::identity()).norm();
        assert!(
            sym < 1e-12 && uni < 1e-12,
            "U₁ not symmetric-unitary: sym={sym:.2e} uni={uni:.2e}"
        );
        // (3) spec(U₁) == diag(Λ): U₁ is the SO(4)-orbit of Λ (eigenvalues preserved by real conjugation).
        let dl = maxerr(sort(std::array::from_fn(|i| lam[(i, i)])), sort(eig4(&u1)));
        assert!(dl < 1e-10, "spec(U₁) != diag(Λ): {dl:.2e}");
    }

    #[test]
    fn reanchor_inverse_recovers_original_sandwich() {
        let dc = dphase([0.31, 0.19, 0.11]);
        let dg = dphase([0.37, 0.23, 0.13]);
        let a = dc * dc;
        let g = dg * dg;
        let s = givens(0, 1, 0.7) * givens(2, 3, 1.1) * givens(1, 2, 0.4) * givens(0, 3, 0.9);
        let diagonalize = |m: &Mat4| {
            let v = extract_o(m);
            let w: [C; 4] = std::array::from_fn(|k| {
                let mut z = C::new(0.0, 0.0);
                for i in 0..4 {
                    for j in 0..4 {
                        z += C::new(v[(i, k)] * v[(j, k)], 0.0) * m[(i, j)];
                    }
                }
                z
            });
            (v.map(|x| C::new(x, 0.0)), w)
        };

        // X_g=D_g V W^-1 V^T D_g has spectrum A^-1.  Its matched real
        // eigenframe S gives the original orientation as O=S^T.
        let n = dg * s * a * s.transpose() * dg;
        let (vg, wg) = diagonalize(&n);
        let wgi: [C; 4] = std::array::from_fn(|k| C::new(1.0, 0.0) / wg[k]);
        let wrg = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&wgi));
        let xg = dg * vg * wrg * vg.transpose() * dg;
        let av: [C; 4] = std::array::from_fn(|k| C::new(1.0, 0.0) / a[(k, k)]);
        let ai = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&av));
        let og = recover_frame(&xg, &ai).transpose();
        assert!(smooth_residual(&dc, &g, &og, &esym4(wg)) < 1e-10);

        // X_c=D_c V W^-1 V^T D_c has spectrum G^-1.  This reanchor is
        // already in the original orientation, so O=S.
        let m = dc * s * g * s.transpose() * dc;
        let (vc, wc) = diagonalize(&m);
        let wci: [C; 4] = std::array::from_fn(|k| C::new(1.0, 0.0) / wc[k]);
        let wrc = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&wci));
        let xc = dc * vc * wrc * vc.transpose() * dc;
        let gv: [C; 4] = std::array::from_fn(|k| C::new(1.0, 0.0) / g[(k, k)]);
        let gi = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&gv));
        let oc = recover_frame(&xc, &gi);
        assert!(smooth_residual(&dc, &g, &oc, &esym4(wc)) < 1e-10);
    }

    #[test]
    fn reanchored_charts_close_former_generic_tail() {
        // These were the only two misses in the complete fast-only census.
        // Before target reanchoring they entered cl3/peel3 for seconds to
        // minutes; each is a regular 3-Givens point from another vertex of
        // the spectral triangle.
        let rows = [
            (
                [0.34375, 0.09375, -0.03125],
                [0.3125, 0.25, -0.125],
                [0.25, 0.1875, -0.1875],
            ),
            (
                [0.40625, 0.15625, -0.09375],
                [0.25, 0.1875, -0.0625],
                [0.3125, 0.25, -0.25],
            ),
        ];
        for (c, g, t) in rows {
            let problem = PreparedSandwich::new(c, g, t);
            let (o, residual) = CyclicThreeGivens::new(&problem)
                .and_then(|atlas| atlas.solve_planes(0, 16, false))
                .expect("reanchored chart witness");
            assert!(residual < 1e-12, "residual={residual:.3e}");
            assert!((o.map(|z| z.re).determinant() - 1.0).abs() < 1e-10);
        }
    }

    /// Isolated micro-benchmark for the interior per-chart kernel (ignored;
    /// run with --ignored --nocapture). WSL wall-clock noise swamps stride
    /// benches at the 10% level; this loop gives a clean per-call number on
    /// REAL detx polynomials from the actual elimination.
    #[test]
    #[ignore = "micro-benchmark; run with --ignored --nocapture"]
    fn bench_interior_kernel() {
        let eb = eigphases([0.31, 0.19, 0.11]);
        let ep = eigphases([0.37, 0.23, 0.13]);
        let et = eigphases([0.41, 0.17, 0.05]);
        let prefix_diag: [C; 4] = std::array::from_fn(|j| C::from_polar(1.0, 2.0 * eb[j]));
        let gate: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * ep[k]));
        let w: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * et[k]));
        let w1: C = w.iter().copied().sum();
        let w2 = w[0] * w[1] + w[0] * w[2] + w[0] * w[3] + w[1] * w[2] + w[1] * w[3] + w[2] * w[3];
        let sq = (w[0] * w[1] * w[2] * w[3]).sqrt();
        let mut polys: Vec<Vec<f64>> = vec![];
        for &planes in INTERIOR_PLANES.iter() {
            for &perm in PERMS24.iter() {
                let (a1, a2) = chart_coeffs_diag(&prefix_diag, &gate, planes, perm);
                let mut r1: [f64; 8] = std::array::from_fn(|i| a1[i].re);
                r1[0] -= w1.re;
                let mut r2: [f64; 8] = std::array::from_fn(|i| a1[i].im);
                r2[0] -= w1.im;
                let a2n: [C; 8] = std::array::from_fn(|i| a2[i] / sq);
                let mut r3: [f64; 8] = std::array::from_fn(|i| a2n[i].re);
                r3[0] -= (w2 / sq).re;
                let split = |r: &[f64; 8]| ([r[0], r[1], r[2], r[4]], [r[3], r[5], r[6], r[7]]);
                let (p1, q1) = split(&r1);
                let (p2, q2) = split(&r2);
                let (p3, q3) = split(&r3);
                let sub = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| -> [[f64; 3]; 3] {
                    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] - b[i][j]))
                };
                let f: [[f64; 3]; 3] = sub(prod_bil(p2, q1), prod_bil(p1, q2));
                let g: [[f64; 3]; 3] = sub(prod_bil(p3, q1), prod_bil(p1, q3));
                let col = |m: &[[f64; 3]; 3], k: usize| vec![m[0][k], m[1][k], m[2][k]];
                let fy = [col(&f, 0), col(&f, 1), col(&f, 2)];
                let gy = [col(&g, 0), col(&g, 1), col(&g, 2)];
                polys.push(resultant_y(&fy, &gy));
            }
        }
        eprintln!("polys: {}", polys.len());
        let reps = 300;
        let t0 = std::time::Instant::now();
        let mut sink = 0usize;
        for _ in 0..reps {
            for p in &polys {
                sink += real_roots_unit(p).len();
            }
        }
        let el = t0.elapsed().as_nanos() as f64;
        eprintln!(
            "real_roots_unit: {:.1} ns/call (sink {sink}, refine iters/call {:.2})",
            el / (reps * polys.len()) as f64,
            refine_iters() as f64 / (reps * polys.len()) as f64
        );
        // component timings: bernstein alone, scan-only floor
        let t2 = std::time::Instant::now();
        let mut se = 0usize;
        for _ in 0..reps {
            for p in &polys {
                se += bernstein_excludes_unit(p) as usize;
            }
        }
        eprintln!(
            "bernstein alone: {:.1} ns/call ({se})",
            t2.elapsed().as_nanos() as f64 / (reps * polys.len()) as f64
        );
        let t3 = std::time::Instant::now();
        let mut sv = 0.0f64;
        for _ in 0..reps {
            for p in &polys {
                const N: usize = 64;
                let mut vals = [0.0f64; N + 1];
                let mut xs = [0.0f64; N + 1];
                for (k, x) in xs.iter_mut().enumerate() {
                    *x = k as f64 / N as f64;
                }
                for &c in p.iter().rev() {
                    for k in 0..=N {
                        vals[k] = vals[k] * xs[k] + c;
                    }
                }
                sv += vals[37];
            }
        }
        eprintln!(
            "scan alone: {:.1} ns/call ({sv:.2})",
            t3.elapsed().as_nanos() as f64 / (reps * polys.len()) as f64
        );
        // and the excluded/scanned split
        let excl = polys.iter().filter(|p| bernstein_excludes_unit(p)).count();
        eprintln!("bernstein-excluded: {excl}/{}", polys.len());
        // elimination kernel cost (chart_coeffs + resultant), same loop
        let t1 = std::time::Instant::now();
        let mut sink2 = 0.0f64;
        for _ in 0..reps {
            for &planes in INTERIOR_PLANES.iter() {
                for &perm in PERMS24.iter() {
                    let (a1, _a2) = chart_coeffs_diag(&prefix_diag, &gate, planes, perm);
                    sink2 += a1[0].re;
                }
            }
        }
        let el1 = t1.elapsed().as_nanos() as f64;
        eprintln!(
            "chart_coeffs_diag: {:.1} ns/call (sink {sink2})",
            el1 / (reps * polys.len()) as f64
        );
    }

    #[test]
    fn poly_roots_recovers_known_roots() {
        let roots = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95];
        // build the monic polynomial Π(x − r) as coeffs (low→high)
        let mut co = vec![1.0];
        for r in roots {
            let mut next = vec![0.0; co.len() + 1];
            for (i, &cv) in co.iter().enumerate() {
                next[i] += -r * cv;
                next[i + 1] += cv;
            }
            co = next;
        }
        let got = poly_roots(&co);
        // each true root must be matched by some computed root
        for r in roots {
            let best = got
                .iter()
                .map(|z| (z - C::new(r, 0.0)).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-9, "root {r} not recovered (best {best})");
        }
    }

    #[test]
    fn interior_elimination_roundtrip() {
        // Manufacture a target from a known chart point; the elimination must recover an
        // O reaching that spectrum (verified in symfn, exact -- the deg-6 companion path).
        let dc = dphase([0.31, 0.19, 0.07]);
        let lam = {
            let dg = dphase([0.27, 0.13, 0.05]);
            dg * dg
        };
        let planes = [(0, 1), (1, 2), (2, 3)];
        let perm = [0, 1, 2, 3];
        let o_true = chart_o([0.4137, 0.6291, 0.1733], planes, perm); // off any scan grid
        let m = mmat(&dc, &lam, &o_true);
        let target = symfn(&m);
        let w = eig4(&m);
        let et: [f64; 4] = std::array::from_fn(|k| w[k].arg() / 2.0);
        let eb = eigphases([0.31, 0.19, 0.07]);
        let ep = eigphases([0.27, 0.13, 0.05]);
        let (_o, r_cf, _, _) = solve_interior(&eb, &ep, &dc, &lam, &[et], &[target], &PERMS24[..])
            .expect("interior found no root");

        // Exact-vs-scan gate: the companion solve must hit a true root (machine
        // precision), unreachable by a fine grid scan, proving it is not a bisection
        // in disguise. A 40³ scan's best is orders of magnitude worse.
        let planes_s = planes;
        let mut best_scan = f64::INFINITY;
        let n = 40;
        for ix in 0..=n {
            for iy in 0..=n {
                for iz in 0..=n {
                    let p = [
                        ix as f64 / n as f64,
                        iy as f64 / n as f64,
                        iz as f64 / n as f64,
                    ];
                    let o = chart_o(p, planes_s, perm);
                    best_scan = best_scan.min(smooth_residual(&dc, &lam, &o, &target));
                }
            }
        }
        assert!(r_cf < 1e-9, "interior not a true root: r_cf={r_cf}");
        assert!(
            r_cf < best_scan / 1e4,
            "closed-form not beating scan (would be a scan-in-disguise): r_cf={r_cf} best_scan={best_scan}"
        );
    }

    #[test]
    fn e1_e2_are_trilinear_in_the_chart() {
        // The corner-extracted trilinear model must reproduce e1,e2 of M at interior
        // points (s12: e1,e2,e3 are multilinear in x,y,z = cos² of the 3 Givens angles).
        let dc = dphase([0.31, 0.19, 0.07]);
        let lam = {
            let dg = dphase([0.27, 0.13, 0.05]);
            dg * dg
        };
        let planes = [(0, 1), (1, 2), (2, 3)];
        let perm = [0, 1, 2, 3];
        let eb = eigphases([0.31, 0.19, 0.07]);
        let ep = eigphases([0.27, 0.13, 0.05]);
        let (a1, a2) = chart_coeffs(&eb, &ep, planes, perm);
        // a few deterministic interior points
        let pts = [[0.3, 0.6, 0.1], [0.8, 0.2, 0.5], [0.15, 0.95, 0.42]];
        let mut worst = 0.0f64;
        for p in pts {
            let m = mmat(&dc, &lam, &chart_o(p, planes, perm));
            let s = symfn(&m);
            let d1 = (ev_trilinear(&a1, p[0], p[1], p[2]) - s[0]).norm();
            let d2 = (ev_trilinear(&a2, p[0], p[1], p[2]) - s[1]).norm();
            worst = worst.max(d1).max(d2);
        }
        assert!(worst < 1e-11, "e1/e2 not trilinear in chart: worst={worst}");
    }

    #[test]
    fn identity_frame_reproduces_canonical_spectrum() {
        // O = I  =>  M = D_C·Λ·D_C = D_C²·Λ, and for C=G=T it must match target_symfn(T).
        let w = [0.31, 0.21, 0.11];
        let dc = dphase(w);
        let lam = {
            let dg = dphase(w);
            dg * dg
        };
        let o = Mat4::identity();
        // Sandwich Can(w)·I·Can(w) has weyl 2w folded; instead check the bare identity-frame
        // matches the C=I sandwich: with C=identity weyl, M = Λ and symfn = target of G's square.
        let res = smooth_residual(&dc, &lam, &o, &symfn(&mmat(&dc, &lam, &o)));
        assert!(res < 1e-12, "self-consistency: {res}");
    }

    #[test]
    fn compound_residual_matches_smooth_residual() {
        // The Cauchy-Binet compound evaluator must equal smooth_residual within 1e-12
        // on several (dc, lam, o) triples, including interior chart points and identity.
        let dc = dphase([0.31, 0.19, 0.07]);
        let lam = {
            let dg = dphase([0.27, 0.13, 0.05]);
            dg * dg
        };
        let planes = [(0, 1), (1, 2), (2, 3)];
        let perm = [0, 1, 2, 3];
        // Target from the same dc/lam at a known interior point.
        let o_ref = chart_o([0.4137, 0.6291, 0.1733], planes, perm);
        let target = symfn(&mmat(&dc, &lam, &o_ref));
        let pts = [
            [0.3, 0.6, 0.1],
            [0.8, 0.2, 0.5],
            [0.15, 0.95, 0.42],
            [0.4137, 0.6291, 0.1733], // the exact root: both should be ~0
        ];
        let mut worst = 0.0f64;
        for p in pts {
            let o = chart_o(p, planes, perm);
            let sr = smooth_residual(&dc, &lam, &o, &target);
            let cr = compound_residual(&dc, &lam, &o, &target);
            worst = worst.max((sr - cr).abs());
        }
        // Identity frame: target = its own spectrum, so residual = 0.
        let o_id = Mat4::identity();
        let target_id = symfn(&mmat(&dc, &lam, &o_id));
        let sr_id = smooth_residual(&dc, &lam, &o_id, &target_id);
        let cr_id = compound_residual(&dc, &lam, &o_id, &target_id);
        worst = worst.max((sr_id - cr_id).abs()).max(cr_id);
        assert!(
            worst < 1e-12,
            "compound_residual mismatch: worst={worst:.3e}"
        );
    }

    #[test]
    fn chart_lifted_masses_are_affine_on_two_givens() {
        // Shared planes: G01(x)G12(y) has s=1+y and d=y exactly.
        for &(x, y) in &[(0.13, 0.27), (0.41, 0.82), (0.93, 0.06)] {
            let o = chart_o([x, y, 1.0], [(0, 1), (1, 2), (2, 3)], [0, 1, 2, 3]);
            let (s, d) = cs_masses(&o);
            assert!((s - (1.0 + y)).abs() < 2e-12, "s={s}, y={y}");
            assert!((d - y).abs() < 2e-12, "d={d}, y={y}");
        }

        // Disjoint planes: the first rotation is entirely inside the leading
        // block, so that block remains orthogonal; hence s=2 and d=1.
        for &(x, y) in &[(0.13, 0.27), (0.41, 0.82), (0.93, 0.06)] {
            let o = chart_o([x, y, 1.0], [(0, 1), (2, 3), (1, 2)], [0, 1, 2, 3]);
            let (s, d) = cs_masses(&o);
            assert!((s - 2.0).abs() < 2e-12, "s={s}, x={x}");
            assert!((d - 1.0).abs() < 2e-12, "d={d}, s={s}, x={x}");
        }
    }

    #[test]
    fn chart_lifted_masses_are_trilinear_on_three_givens() {
        // The production three-Givens chart uses squared cosines.  Test the
        // stronger statement needed by the selector: s and d are recovered by
        // multilinear interpolation from the eight chart corners, with no
        // residual square-root dependence.
        let planes = [(0, 1), (1, 2), (2, 3)];
        let interp = |which: usize, p: [f64; 3]| -> f64 {
            let mut out = 0.0;
            for mask in 0..8 {
                let q = [
                    if mask & 1 != 0 { 1.0 } else { 0.0 },
                    if mask & 2 != 0 { 1.0 } else { 0.0 },
                    if mask & 4 != 0 { 1.0 } else { 0.0 },
                ];
                let w = (0..3).fold(1.0, |acc, k| {
                    acc * if q[k] == 1.0 { p[k] } else { 1.0 - p[k] }
                });
                let o = chart_o(q, planes, [0, 1, 2, 3]);
                let v = cs_masses(&o);
                out += w * if which == 0 { v.0 } else { v.1 };
            }
            out
        };
        for p in [[0.17, 0.43, 0.79], [0.61, 0.22, 0.36], [0.93, 0.08, 0.54]] {
            let o = chart_o(p, planes, [0, 1, 2, 3]);
            let (s, d) = cs_masses(&o);
            assert!(
                (s - interp(0, p)).abs() < 2e-11,
                "s={s}, fit={}",
                interp(0, p)
            );
            assert!(
                (d - interp(1, p)).abs() < 2e-11,
                "d={d}, fit={}",
                interp(1, p)
            );
        }
    }

    #[test]
    fn lifted_masses_are_multilinear_on_entire_interior_atlas() {
        // Exhaust the actual 16 plane words and 24 SO(4) permutations used by
        // the exact interior solver.  Corner interpolation is an exact test
        // for a multilinear polynomial in x=cos²(theta), not a numerical fit.
        let points = [[0.173, 0.431, 0.792], [0.614, 0.227, 0.361]];
        let mut worst = 0.0;
        let mut worst_label = ([0usize; 4], [(0usize, 0usize); 3]);
        for &planes in INTERIOR_PLANES.iter() {
            for &perm in PERMS24.iter() {
                let coeffs = cs_mass_coeffs(planes, perm);
                for p in points {
                    let o = chart_o(p, planes, perm);
                    let actual = cs_masses(&o);
                    let fit = [
                        eval_multilinear(&coeffs[0], p),
                        eval_multilinear(&coeffs[1], p),
                    ];
                    let err = (actual.0 - fit[0]).abs().max((actual.1 - fit[1]).abs());
                    if err > worst {
                        worst = err;
                        worst_label = (perm, planes);
                    }
                }
            }
        }
        assert!(
            worst < 3e-10,
            "non-multilinear lifted mass: err={worst:.3e}, perm={:?}, planes={:?}",
            worst_label.0,
            worst_label.1
        );
    }
}
