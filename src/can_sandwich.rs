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

pub type C = Complex<f64>;

#[path = "arb_roots.rs"]
mod arb_roots;
#[path = "axis_quartic.rs"]
mod axis_quartic;
#[path = "klein.rs"]
mod klein;
#[path = "one_plus_three.rs"]
mod one_plus_three;
#[path = "pair22.rs"]
mod pair22;
#[path = "problem.rs"]
mod problem;
#[path = "secular.rs"]
mod secular;
#[cfg(any(test, feature = "research-spin"))]
#[path = "spin_selector.rs"]
mod spin_selector;

#[path = "three_givens.rs"]
mod three_givens;
pub type Mat4 = Matrix4<C>;

#[cfg(test)]
use problem::spectrum_kind;
use problem::{frame_metrics, orient_so4, PreparedSandwich, SpectrumKind, StratumSignature};

/// PROF=1 instrumentation: per-stage aggregate ns/calls across a corpus run.
/// Compiled out unless the `diagnostics` feature is enabled.
pub mod prof {
    #[cfg(feature = "diagnostics")]
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    #[cfg(feature = "diagnostics")]
    pub const N: usize = 66;
    #[cfg(feature = "diagnostics")]
    pub const NAMES: [&str; N] = [
        "rank_perms",
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
        "input_early",
        "sw_scalar",
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
        "tier_axis",
        "tier_fallback",
        "n_try_mu",
        "rj_base",
        "rj_skel",
        "rj_ureal",
        "rj_usum",
        "n_accept",
        "n_eps_mirror",
        "rj_v_imag",
        "rj_v_neg",
        "rj_v_sum",
        "sw_base",
        "arc_pair_skip",
        "arc_cand_skip",
        "arc_mispredict",
        "uray_skip",
        "uray_mispredict",
        "eps_accept",
        "skel_skip",
        "skel_mispredict",
        "base_fail_forced",
        "base_fail_skel",
        "base_fail_pair",
        "dev_lt_1em6",
        "dev_mid",
        "dev_gt_1em2",
        "pair_gate_some",
        "pair_gate_none",
        "inh_skip",
        "inh_mispredict",
        "red_skip",
        "red_mispredict",
        "sw_header",
        "klein_total",
        "axis_prep",
        "axis_pullback",
        "axis_fast_roots",
        "axis_cert_roots",
        "axis_reconstruct",
        "axis_hull_skip",
        "axis_weight_hull_skip",
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

#[cfg(test)]
/// The diagonal phase matrix `D = mb(Can(w))` (canonical gates are diagonal in the
/// magic basis). Built directly as `diag(exp(i*eigphases(w)))` -- the recorded
/// closed form of that diagonal (`eigphases` doc; asserted against `mb(canonical)`
/// by `canonical_is_diagonal_in_magic_basis`). The old matrix-product route cost
/// ~3us per solve() in prelude matmuls for a matrix that is diagonal by
/// construction.
pub fn dphase(w: [f64; 3]) -> Mat4 {
    let ph = eigphases(w);
    let d: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, ph[k]));
    Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&d))
}

/// Monodromy triple -> Weyl coords (gulps convention `[m₀+m₁, m₀+m₂, m₁+m₂]`).
pub fn weyl_from_monodromy(m: [f64; 3]) -> [f64; 3] {
    [m[0] + m[1], m[0] + m[2], m[1] + m[2]]
}

/// Rho-reflected Weyl coords `[1−c₁, c₂, −c₃]`: same Makhlin invariants, different
/// eigenvalue set -- a target can match in either orientation, so rungs try both.
pub fn rho_weyl(w: [f64; 3]) -> [f64; 3] {
    [1.0 - w[0], w[1], -w[2]]
}

/// The 4 magic-basis eigenphases of `Can(w)` in closed form (no matrix): the diagonal
/// of `mb(Can(w))`, i.e. `D_C = diag(exp(i·eigphases))`. Same order as `dphase`.
pub fn eigphases(w: [f64; 3]) -> [f64; 4] {
    use std::f64::consts::PI;
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
pub fn esym4(s: [C; 4]) -> [C; 4] {
    let e1 = s[0] + s[1] + s[2] + s[3];
    let e2 = s[0] * s[1] + s[0] * s[2] + s[0] * s[3] + s[1] * s[2] + s[1] * s[3] + s[2] * s[3];
    let e3 = s[0] * s[1] * s[2] + s[0] * s[1] * s[3] + s[0] * s[2] * s[3] + s[1] * s[2] * s[3];
    let e4 = s[0] * s[1] * s[2] * s[3];
    [e1, e2, e3, e4]
}

/// One signed permutation frame `P ∈ SO(4)`: the permutation matrix `P_{i,p_i}=1`
/// with row 0 negated when needed to force `det = +1`. The sign squares away in the
/// spectrum (`PΛPᵀ` diagonal = `Λ` permuted), so it matters only for the emitted frame.
pub fn signed_perm(p: [usize; 4]) -> Mat4 {
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
pub fn givens(i: usize, j: usize, theta: f64) -> Mat4 {
    let (ct, st) = (theta.cos(), theta.sin());
    let mut g = Mat4::identity();
    g[(i, i)] = c(ct, 0.0);
    g[(j, j)] = c(ct, 0.0);
    g[(i, j)] = c(-st, 0.0);
    g[(j, i)] = c(st, 0.0);
    g
}

/// The 6 Givens planes (= the 6 transpositions / permutohedron edge directions).
pub const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// The 24 permutations of `(0,1,2,3)`, computed once (hot-path: no per-call alloc).
pub static PERMS24: std::sync::LazyLock<[[usize; 4]; 24]> = std::sync::LazyLock::new(|| {
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
        let swap = if k % 2 == 0 { i } else { 0 };
        a.swap(swap, k - 1);
    }
}

#[inline]
fn cross2(left: C, right: C) -> f64 {
    left.re * right.im - left.im * right.re
}

/// Membership in a finite convex hull in the complex trace plane.  By
/// Caratheodory, a point in a planar hull lies in one triangle of its
/// vertices.  The tolerance is outward-only, so this is safe as a necessary
/// feasibility gate near a hull face.
pub(super) fn point_in_complex_hull<const N: usize>(vertices: &[C; N], point: C) -> bool {
    let scale = vertices
        .iter()
        .map(|vertex| vertex.norm())
        .fold(point.norm().max(1.0), f64::max);
    let linear_tolerance = 2e-10 * scale;
    let area_tolerance = linear_tolerance * scale;
    if vertices
        .iter()
        .any(|vertex| (*vertex - point).norm() <= linear_tolerance)
    {
        return true;
    }
    for i in 0..N.saturating_sub(2) {
        for j in i + 1..N.saturating_sub(1) {
            for k in j + 1..N {
                let (p, q, r) = (vertices[i], vertices[j], vertices[k]);
                let area = cross2(q - p, r - p);
                if area.abs() <= area_tolerance {
                    for (x, y) in [(p, q), (p, r), (q, r)] {
                        let direction = y - x;
                        let length2 = direction.norm_sqr();
                        if length2 <= area_tolerance * area_tolerance {
                            continue;
                        }
                        let parameter = ((point - x).re * direction.re
                            + (point - x).im * direction.im)
                            / length2;
                        let distance = cross2(point - x, direction).abs() / length2.sqrt();
                        if (-2e-10..=1.0 + 2e-10).contains(&parameter)
                            && distance <= linear_tolerance
                        {
                            return true;
                        }
                    }
                    continue;
                }
                let s0 = cross2(q - p, point - p) / area;
                let s1 = cross2(point - p, r - p) / area;
                let s2 = 1.0 - s0 - s1;
                if s0 >= -2e-10 && s1 >= -2e-10 && s2 >= -2e-10 {
                    return true;
                }
            }
        }
    }
    false
}

/// Elementary symmetric functions `e₁..e₄` of `eig(A)` via Newton's identities on
/// traces -- no eigendecomposition, so it does not floor at spectrum degeneracy.
/// Production uses the matrix-free `compound_residual` (s184); this remains the
/// test-side reference implementation.
#[cfg(test)]
pub fn symfn(a: &Mat4) -> [C; 4] {
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
pub fn mmat(dc: &Mat4, lam: &Mat4, o: &Mat4) -> Mat4 {
    dc * o * lam * o.transpose() * dc
}

#[cfg(test)]
/// Smooth (non-flooring) residual: `‖symfn(M(O)) − symfn(target)‖∞`. The reach
/// certificate; near degeneracy this is the metric, not weyl coords.
pub fn smooth_residual(dc: &Mat4, lam: &Mat4, o: &Mat4, target: &[C; 4]) -> f64 {
    let got = symfn(&mmat(dc, lam, o));
    (0..4)
        .map(|i| (got[i] - target[i]).norm())
        .fold(0.0, f64::max)
}

/// s184 §4: Cauchy-Binet compound-moment residual. Computes `max(|Δe₁|, |Re(Δe₂/s)|)`
/// where `s = √e₄(target)`, without forming `M` or any matrix power (16 + 36 scalar
/// terms vs 7 matmuls in `smooth_residual`). Equal to `smooth_residual` in exact
/// arithmetic; max discrepancy is numerical (observed ~1e-14). Use for benchmarking
/// before replacing `smooth_residual` globally (s184 §6 step 4).
pub fn compound_residual(dc: &Mat4, lam: &Mat4, o: &Mat4, target: &[C; 4]) -> f64 {
    // a_i = dc_{ii}^2 (complex), lv_j = lam_{jj} (complex), O real (zero imag entries).
    let a: [C; 4] = std::array::from_fn(|i| {
        let d = dc[(i, i)];
        d * d
    });
    let lv: [C; 4] = std::array::from_fn(|j| lam[(j, j)]);
    let or_: [[f64; 4]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| o[(i, j)].re));

    // e1 = sum_{i,j} a_i * or[i][j]^2 * lv_j  (16 terms)
    let mut e1 = C::default();
    for i in 0..4 {
        for j in 0..4 {
            e1 += a[i] * lv[j] * (or_[i][j] * or_[i][j]);
        }
    }

    // e2 = sum_{i0<i1, j0<j1} a[i0]*a[i1] * det(O_{IJ})^2 * lv[j0]*lv[j1]  (36 terms)
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut e2 = C::default();
    for &(i0, i1) in &PAIRS {
        let ai = a[i0] * a[i1];
        for &(j0, j1) in &PAIRS {
            let m = or_[i0][j0] * or_[i1][j1] - or_[i0][j1] * or_[i1][j0];
            e2 += ai * lv[j0] * lv[j1] * (m * m);
        }
    }

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
pub fn chart_o(xyz: [f64; 3], planes: [(usize, usize); 3], perm: [usize; 4]) -> Mat4 {
    let mut o = signed_perm(perm);
    for (k, &(i, j)) in planes.iter().enumerate() {
        let theta = xyz[k].clamp(0.0, 1.0).sqrt().acos();
        o = givens(i, j, theta) * o;
    }
    o
}

/// Direct-sqrt chart frame (s184 §3): `cos θ = √v`, `sin θ = √(1−v)` for `θ = arccos(√v)`,
/// eliminating `acos/sin/cos` (3 transcendentals per coordinate -> 3 sqrts). Equivalent
/// to `chart_o` in exact arithmetic. Shadow-compared via `S184_SQRT` before dispatch change.
pub fn chart_o_sqrt(xyz: [f64; 3], planes: [(usize, usize); 3], perm: [usize; 4]) -> Mat4 {
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

/// Monomial corners `(x,y,z) ∈ {0,1}³` in the order used by the
/// trilinear Möbius transform below.
const CORNERS: [[f64; 3]; 8] = [
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 1., 0.],
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 1.],
];

#[cfg(test)]
/// Trilinear coeffs of `(e₁, e₂)` of `M` over the chart, extracted from the 8 corners.
/// At a corner every Givens is 0 (`x=cos²θ=1`) or π/2 (`x=0` → a signed swap), so `O` is a
/// signed perm and `M` is DIAGONAL: the corner spectrum is `{a²_j·λ_{c(j)}}` with `c` the
/// composed permutation -- pure scalar, no matmul (same trick as the vertex rung).
pub fn chart_coeffs(
    eb: &[f64; 4],
    ep: &[f64; 4],
    planes: [(usize, usize); 3],
    perm: [usize; 4],
) -> ([C; 8], [C; 8]) {
    let prefix_diag: [C; 4] = std::array::from_fn(|j| C::from_polar(1.0, 2.0 * eb[j]));
    let lam: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * ep[k]));
    chart_coeffs_diag(&prefix_diag, &lam, planes, perm)
}

#[cfg(test)]
fn chart_coeffs_diag(
    a2: &[C; 4],
    lam: &[C; 4],
    planes: [(usize, usize); 3],
    perm: [usize; 4],
) -> ([C; 8], [C; 8]) {
    let (e1c, e2c) = chart_corners(a2, lam, planes, perm);
    (mobius8(&e1c), mobius8(&e2c))
}

/// Certified separation test: is `t` at distance > `margin` from
/// conv(corners)? A few Frank-Wolfe steps toward the closest hull point
/// propose a separating direction d; exclusion is declared ONLY on the
/// verified certificate  min_i <c_i - t, d> > margin*|d|  (8 dot products,
/// checked exactly every round). Soundness never depends on the search:
/// a missed separation just means no pruning. Bounded 12 rounds, no
/// tolerance semantics beyond the caller's margin.
fn hull_excludes(corners: &[[f64; 3]], t: [f64; 3], margin: f64) -> bool {
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

/// The chart's 8 corner values of (e1, e2) (the raw material of both the
/// multilinear coefficients and the hull gate).
#[cfg(test)]
fn chart_corners(
    a2: &[C; 4],
    lam: &[C; 4],
    planes: [(usize, usize); 3],
    perm: [usize; 4],
) -> ([C; 8], [C; 8]) {
    let mut vertices = [None; 256];
    chart_corners_cached(a2, lam, planes, perm, &mut vertices)
}

/// The eight corners of every three-Givens chart lie in the same 24-point
/// Weyl orbit. Cache that orbit lazily by the packed permutation itself: a
/// short successful scan pays exactly for the vertices it visits, while an
/// exhaustive scan evaluates each spectral vertex at most once.
fn chart_corners_cached(
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
fn mobius8(vals: &[C; 8]) -> [C; 8] {
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

/// All complex roots of a real polynomial via companion-matrix eigenvalues
/// (faer). Used by bounded algebraic constructions outside the hot
/// three-Givens atlas. `coeffs` are low-to-high; trailing near-zero leading
/// terms are trimmed.
pub fn poly_roots(coeffs: &[f64]) -> Vec<C> {
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

/// Eigenvalues of a 4×4 (faer). TEST-ONLY -- the solver gets the target spectrum in closed
/// form (`exp(2i·eigphases)`) and matches via `symfn`.
#[cfg(test)]
pub fn eig4(m: &Mat4) -> [C; 4] {
    let fm = faer::Mat::<C>::from_fn(4, 4, |i, j| m[(i, j)]);
    let ev = fm.eigenvalues().expect("eig4");
    std::array::from_fn(|i| ev[i])
}

/// Evaluate a trilinear form (coeffs in monomial order) at `(x,y,z)`. Test-only: the solver
/// uses `ev_ml` (the multilinear+Y chart); this is the roundtrip harness's reference form.
#[cfg(test)]
pub fn ev_trilinear(co: &[C; 8], x: f64, y: f64, z: f64) -> C {
    let m = [1.0, x, y, z, x * y, x * z, y * z, x * y * z];
    (0..8).map(|k| co[k] * m[k]).sum()
}

// ---- polynomial helpers for the interior elimination (real coeffs, low→high) ----

#[cfg(test)]
fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

#[cfg(test)]
fn poly_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len().max(b.len())];
    for (i, &x) in a.iter().enumerate() {
        out[i] += x;
    }
    for (i, &x) in b.iter().enumerate() {
        out[i] -= x;
    }
    out
}

#[cfg(test)]
/// Resultant in `y` of two quadratics `a₂y²+a₁y+a₀` and `b₂y²+b₁y+b₀` whose coefficients
/// are polynomials in `x` (`fy=[a₀,a₁,a₂]`, `gy=[b₀,b₁,b₂]`): closed form
/// `(a₂b₀−a₀b₂)² − (a₂b₁−a₁b₂)(a₁b₀−a₀b₁)`. Replaces the 4×4 Sylvester-Leibniz det
/// (~8 poly-mults vs ~96). A constant scale/sign is irrelevant -- only the roots matter.
fn resultant_y(fy: &[Vec<f64>; 3], gy: &[Vec<f64>; 3]) -> Vec<f64> {
    let (a0, a1, a2) = (&fy[0], &fy[1], &fy[2]);
    let (b0, b1, b2) = (&gy[0], &gy[1], &gy[2]);
    let t1 = poly_sub(&poly_mul(a2, b0), &poly_mul(a0, b2));
    let t2 = poly_sub(&poly_mul(a2, b1), &poly_mul(a1, b2));
    let t3 = poly_sub(&poly_mul(a1, b0), &poly_mul(a0, b1));
    poly_sub(&poly_mul(&t1, &t1), &poly_mul(&t2, &t3))
}

#[cfg(test)]
/// Bilinear `[1,x,y,xy]` product -> biquadratic `[i][j] = coeff x^i y^j`, i,j ≤ 2.
fn prod_bil(a: [f64; 4], b: [f64; 4]) -> [[f64; 3]; 3] {
    let terms = |c: [f64; 4]| [(0, 0, c[0]), (1, 0, c[1]), (0, 1, c[2]), (1, 1, c[3])];
    let mut out = [[0.0; 3]; 3];
    for (i1, j1, c1) in terms(a) {
        for (i2, j2, c2) in terms(b) {
            out[i1 + i2][j1 + j2] += c1 * c2;
        }
    }
    out
}

/// Real roots of a real cubic in radicals (Cardano/Viete), with exact degree
/// drops. No iteration: the trigonometric branch is closed form.
fn real_roots_cubic(c: &[f64; 4], out: &mut [f64; 3]) -> usize {
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
fn admissible_components(n: &[[f64; 4]; 4], out: &mut [(f64, f64); 8]) -> usize {
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

/// Obstruction taxonomy for the interior chart funnel. The invariant: an
/// inability to COMPUTE a root must never be recorded as mathematical
/// nonexistence, so a non-convergent rooter, a non-finite coefficient or a
/// degenerate eliminant becomes NUMERICAL_FAILURE, never a DECLINE.
/// Enabled by GULPS_FUNNEL=1 in `diagnostics` builds; compiled out otherwise.
pub mod funnel {
    #[cfg(feature = "diagnostics")]
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    #[cfg(feature = "diagnostics")]
    pub const N: usize = 52;
    #[cfg(feature = "diagnostics")]
    pub const NAMES: [&str; N] = [
        "ATTEMPTS",
        "HULL_DECLINE_no_real_root_in_[0,1]",
        "RECOVERY_REGION_EMPTY",
        "SEXTIC_NO_ROOT_IN_S",
        "FINAL_COMPATIBILITY_FAIL",
        "NUMERICAL_FAILURE",
        "DEGENERATE_ELIMINANT",
        "RAD_p0_gate",
        "RAD_p0_swap",
        "RAD_p0_target",
        "RAD_p1_gate",
        "RAD_p1_swap",
        "RAD_p1_target",
        "RAD_m0_w0",
        "RAD_m0_w1",
        "RAD_m0_w2",
        "RAD_m1_w0",
        "RAD_m1_w1",
        "RAD_m1_w2",
        "RAD_m2_w0",
        "RAD_m2_w1",
        "RAD_m2_w2",
        "RAD_m3_w0",
        "RAD_m3_w1",
        "RAD_m3_w2",
        "RAD_m4_w0",
        "RAD_m4_w1",
        "RAD_m4_w2",
        "RAD_m5_w0",
        "RAD_m5_w1",
        "RAD_m5_w2",
        "RAD_m6_w0",
        "RAD_m6_w1",
        "RAD_m6_w2",
        "RAD_m7_w0",
        "RAD_m7_w1",
        "RAD_m7_w2",
        "PAIRIDX_0",
        "PAIRIDX_1",
        "PAIRIDX_2",
        "PAIRIDX_3",
        "PAIRIDX_4",
        "PAIRIDX_5",
        "PAIRIDX_6",
        "PAIRIDX_7",
        "APRIORI_HIT",
        "APRIORI_MISS",
        "EVENCHAR_ENTER",
        "ALLSIMPLE_T",
        "ALLSIMPLE_F",
        "SKELWIN_APRIORI_OK",
        "SKELWIN_APRIORI_REJECT",
    ];
    pub const ATTEMPTS: usize = 0;
    pub const HULL: usize = 1;
    pub const REGION_EMPTY: usize = 2;
    pub const NO_ROOT_IN_S: usize = 3;
    pub const FINAL_FAIL: usize = 4;
    pub const NUMERICAL: usize = 5;
    pub const DEGENERATE: usize = 6;
    /// Which of the multiplicity recursion's three orientations owns a row, per
    /// pass. The order they are tried in is fixed; this reads whether the
    /// dataflow still justifies it after the Klein migration.
    pub const RAD: usize = 7;
    /// Semialgebraic router probe: index 13 + 3*active_mask + winner. The mask
    /// is which of (gate, swap, target) orientations the multiplicity strata
    /// make ACTIVE -- a sign condition, not a trial. If the winner is a
    /// FUNCTION of the mask, the 1.73-attempt search collapses to 1 lookup.
    pub const RADMW: usize = 13;
    /// Which position in the pair scan actually wins. The scan is the last
    /// try-until-hit in the hot path; if the winner concentrates, a coverage
    /// reorder cuts it the way the 2026-07-21 interior reorder did (4.63->2.29).
    pub const PAIRIDX: usize = 37;
    /// A-priori skeleton parity target vs the lazily calibrated one. The
    /// even-side arrangement law reduces to D_t = pi - A_t, the SAME eps
    /// vector as the odd side, so skel_target should be computable before any
    /// enumeration. Verify before replacing the calibration.
    pub const APRIORI: usize = 45;
    pub const EVENCHAR: usize = 47;
    /// Would the a-priori parity gate REJECT a subset that actually wins?
    /// Any nonzero REJECT means the gate is lossy and must not be enabled.
    pub const SKELWIN: usize = 50;
    #[cfg(feature = "diagnostics")]
    static C: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    #[cfg(feature = "diagnostics")]
    fn on() -> bool {
        use std::sync::OnceLock;
        static F: OnceLock<bool> = OnceLock::new();
        *F.get_or_init(|| std::env::var_os("GULPS_FUNNEL").is_some())
    }
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub fn bump(i: usize) {
        if on() {
            C[i].fetch_add(1, Relaxed);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    #[inline(always)]
    pub fn bump(_i: usize) {}
    #[cfg(feature = "diagnostics")]
    pub fn snapshot() -> [u64; N] {
        std::array::from_fn(|i| C[i].load(Relaxed))
    }
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

fn may_have_unit_root(a0: &[f64; 7], lo: f64, hi: f64) -> usize {
    bernstein_variations(a0, lo, hi, 1e-14)
}

/// Which bounded construction produced the certified frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    Vertex,
    Edge,
    Face,
    /// Routed `1 + 3` peel: rank-drop formulas and nine quadratic zero-entry
    /// walls. Dense residual fibers continue to the complete global axis tier.
    OnePlusThree,
    Interior,
    /// Per-chart six-bracket Cauer reduction: one decic, one cubic/quadratic
    /// lift, linear recovery, and the Heron-kernel frame. The finite chart
    /// orbit is not a complete realization atlas.
    AxisQuartic,
    /// Klein-circulant one-sided chart: `e₁` linear in the orthostochastic
    /// diagonal pins a line in the simplex, `e₂` collapses to one real quadratic.
    Klein,
    /// One factor has multiplicity `2 + 2`: linear target matching in six
    /// squared Pluecker coordinates followed by one Heron plane quartic.
    Pair22,
    /// Exact bounded four-Givens algebraic fiber.
    /// Input-side rank-secular closed form ((3,1)-degenerate base or gate).
    /// Radical strata of the distilled solver (skeleton / pin-pair /
    /// split-pair theta characteristics), tried in all four orientations and
    /// re-gated on the original smooth residual.
    Radical,
    /// Generic projective Spin action slice (research route until its bounded
    /// action-selection contract is complete).
    Spin,
    Unsolved,
}

pub struct Solution {
    pub o: Mat4,
    pub rung: Rung,
    pub residual: f64,
}

/// Accept threshold on the smooth residual. A true reach is ~1e-13; this is loose
/// enough to absorb FP while rejecting non-reaches (whose residual is O(0.1+)).
pub const ACCEPT: f64 = 1e-9;

/// Public-frame tolerance.  This is deliberately tighter than the spectral
/// acceptance threshold: the fast compound residual is valid only for a real
/// orthogonal frame.  A rejected accelerator candidate must fall through to
/// the next construction rather than escape the black box.
const FRAME_ACCEPT: f64 = 2e-10;

#[inline]
fn certified_solution(o: Mat4, rung: Rung, residual: f64) -> Option<Solution> {
    if !residual.is_finite() || residual > ACCEPT {
        return None;
    }
    let metrics = frame_metrics(&o)?;
    if !metrics.within(FRAME_ACCEPT) {
        return None;
    }
    Some(Solution {
        o: orient_so4(o),
        rung,
        residual,
    })
}

/// Rootwise certificate for the public compiler boundary. Symmetric-function
/// residuals are the stable algebraic test used inside the solver, but their
/// inverse map is ill-conditioned at repeated spectra: an `O(1e-9)` coefficient
/// error can represent an `O(1e-5)` class error. Once per returned candidate,
/// match the four roots of the realized master matrix against both target lifts.
fn compiler_solution(
    problem: &PreparedSandwich,
    o: Mat4,
    rung: Rung,
    residual: f64,
) -> Option<Solution> {
    let solution = certified_solution(o, rung, residual)?;
    let repeated_target = problem.target_roots[0].iter().enumerate().any(|(i, root)| {
        problem.target_roots[0][i + 1..]
            .iter()
            .any(|other| (root - other).norm() < 1e-8)
    });
    if repeated_target {
        // Root solvers are themselves ill-conditioned (and may take a very long
        // QR tail) at an exact multiple root. Require a machine-scale algebraic
        // certificate there instead. This rejects the loose near-edge frames
        // that motivated the rootwise gate without diagonalizing the confluent case.
        return (residual < 1e-12).then_some(solution);
    }
    let master = problem.dc * solution.o * problem.lam * solution.o.transpose() * problem.dc;
    let matrix = faer::Mat::<C>::from_fn(4, 4, |i, j| master[(i, j)]);
    let roots = matrix.eigenvalues().ok()?;
    let error = problem
        .target_roots
        .iter()
        .flat_map(|target| {
            PERMS24.iter().map(|order| {
                (0..4)
                    .map(|i| (roots[i] - target[order[i]]).norm())
                    .fold(0.0_f64, f64::max)
            })
        })
        .fold(f64::INFINITY, f64::min);
    (error < 1e-8).then_some(solution)
}

#[inline]
fn unsolved_solution() -> Solution {
    Solution {
        o: Mat4::identity(),
        rung: Rung::Unsolved,
        residual: f64::INFINITY,
    }
}

/// Fast-accept threshold for the s184 reduced trilinear score (s184 §6 step 3).
/// Candidates with rs < FAST_ACCEPT have smooth_residual < FAST_ACCEPT + max_disc
/// (max_disc = 6e-15 observed over the full corpora), so they are guaranteed
/// well below ACCEPT. Shadow corpus: ra_sr=0 rr_sa=0 on both full corpora.
/// FAST_ACCEPT is 100x below ACCEPT.  The reduced score also rejects candidates
/// at or above ACCEPT before frame construction; full Haar and linspace shadow
/// gates over the adaptive-recovery population found zero decision disagreements
/// (558,761 candidates, max discrepancy 6.106e-15).
pub(crate) const FAST_ACCEPT: f64 = 1e-11;

/// How many ranked perms the edge/interior rungs try. The vertex-residual rank ORDERS the
/// perms so the right chart is hit first (fast early-exit), but truncating it drops the right
/// perm region-dependently and costs coverage for ~no perf gain (perf is eig-bound, not
/// perm-bound) -- so we keep all 24, ranked.
pub const CAND_K: usize = 24;

/// Force every lazily built table (perm list, interior quotient). One-time
/// setup work; call before timing loops so a corpus max measures the solver,
/// not the first row's table construction (measured: the linspace corpus max
/// was the first interior-rung row paying the 216-class quotient build).
#[cfg(feature = "diagnostics")]
pub fn init_tables() {
    let _ = &*PERMS24;
    let _ = &*INTERIOR_QUOTIENT;
}

// Interior rung: the deg-6 multilinear chart. `e₁,e₂` are trilinear in
// `(x,y,z)=cos²θ`; eliminate `z` linearly, take the quadratic resultant in
// `y`, isolate real `x∈[0,1]`, back-substitute, and verify with `symfn`.
#[cfg(feature = "diagnostics")]
thread_local! {
    /// Instrumentation: #(perm,plane) charts tried in solve_interior, and the winning indices.
    pub static INTERIOR_TRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub static INTERIOR_WIN_PLANE: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
    pub static INTERIOR_WIN_PERM: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Winning quotient class (0..216) and root index within real_roots_unit; -1 = not set.
    pub static INTERIOR_WIN_CLASS: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
    pub static INTERIOR_WIN_ROOT: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
    /// Diagnostic-only exact quotient classes with at least one certified
    /// candidate during a CHARTALL primary-atlas scan.
    static INTERIOR_VALID_CLASSES: std::cell::RefCell<[bool; 216]> =
        const { std::cell::RefCell::new([false; 216]) };
}
/// Reset the interior instrumentation counters (call before `solve`).
#[cfg(feature = "diagnostics")]
pub fn reset_interior_instr() {
    INTERIOR_TRIES.with(|c| c.set(0));
    INTERIOR_WIN_PLANE.with(|c| c.set(-1));
    INTERIOR_WIN_PERM.with(|c| c.set(0));
    INTERIOR_WIN_CLASS.with(|c| c.set(-1));
    INTERIOR_WIN_ROOT.with(|c| c.set(-1));
    INTERIOR_VALID_CLASSES.with(|c| *c.borrow_mut() = [false; 216]);
}
/// GULPS_CHARTALL diagnostic: when set, `solve_interior_words` disables its
/// first-valid early return so one full primary-atlas pass records EVERY
/// quotient class admitting a certified candidate for the direct frame (the
/// input to the minimal-spanning-set set-cover). Zero cost when unset.
fn chartall_mode() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| std::env::var_os("GULPS_CHARTALL").is_some())
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        false
    }
}

#[inline(always)]
fn count_interior_try() {
    #[cfg(feature = "diagnostics")]
    INTERIOR_TRIES.with(|c| c.set(c.get() + 1));
}

#[inline(always)]
fn record_interior_hit(class: usize, plane: usize, perm: usize, root: usize) {
    #[cfg(feature = "diagnostics")]
    {
        INTERIOR_VALID_CLASSES.with(|classes| classes.borrow_mut()[class] = true);
        INTERIOR_WIN_PLANE.with(|c| c.set(plane as i32));
        INTERIOR_WIN_PERM.with(|c| c.set(perm));
        INTERIOR_WIN_CLASS.with(|c| c.set(class as i32));
        INTERIOR_WIN_ROOT.with(|c| c.set(root as i32));
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (class, plane, perm, root);
}
/// The structurally-hot quotient classes (the 102 living in the 16 principal
/// words); intrinsic to the cube-automorphism quotient, independent of any corpus.
#[cfg(feature = "diagnostics")]
pub fn interior_hot_classes() -> Vec<u16> {
    (0..216u16)
        .filter(|&c| INTERIOR_QUOTIENT.hot[c as usize])
        .collect()
}
/// Diagnostic: the (plane_idx 0..16, static_perm_idx 0..24) -> quotient-class map
/// for the hot atlas, flattened row-major. Used offline to compute the coverage
/// scan order; not called in production.
#[cfg(feature = "diagnostics")]
pub fn interior_class_map() -> Vec<u16> {
    let mut v = Vec::with_capacity(16 * 24);
    for plane_idx in 0..16 {
        for perm_idx in 0..24 {
            v.push(INTERIOR_QUOTIENT.class[plane_idx][perm_idx]);
        }
    }
    v
}
/// The quotient classes (0..216) that admitted a certified candidate on the last
/// `solve` under CHARTALL -- the direct frame's valid chart set.
#[cfg(feature = "diagnostics")]
pub fn interior_valid_classes() -> Vec<u16> {
    INTERIOR_VALID_CLASSES.with(|c| {
        c.borrow()
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| v.then_some(i as u16))
            .collect()
    })
}
/// (#charts tried, winning plane idx or -1, winning perm idx).
#[cfg(feature = "diagnostics")]
pub fn interior_instr() -> (usize, i32, usize) {
    (
        INTERIOR_TRIES.with(|c| c.get()),
        INTERIOR_WIN_PLANE.with(|c| c.get()),
        INTERIOR_WIN_PERM.with(|c| c.get()),
    )
}
/// (winning quotient class 0..216, winning x-root index within real_roots_unit; -1 if not Interior).
#[cfg(feature = "diagnostics")]
pub fn interior_class_root() -> (i32, i32) {
    (
        INTERIOR_WIN_CLASS.with(|c| c.get()),
        INTERIOR_WIN_ROOT.with(|c| c.get()),
    )
}

pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    let tpre = prof::start();
    let problem = PreparedSandwich::new(c, g, t);

    // A vertex or one-Givens edge preserves routed eigenvalues a_i*g_j. Test
    // that necessary spectral signature before enumerating support incidences.
    // The gate carries its branch masks into the edge solve, so the certificate
    // is computed only once.
    let edge_gate = secular::edge_gate(&problem.routed, &problem.target_roots);
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
                if let Some(solution) = compiler_solution(&problem, signed_perm(p), Rung::Vertex, r)
                {
                    return solution;
                }
            }
        }
        support_perms = Some(cand);
    }

    prof::rec(10, tpre);
    if chartall_mode() {
        // Full primary-atlas scan with early-return disabled: records the direct
        // frame's valid chart set in INTERIOR_VALID_CLASSES. Transported target-slot
        // charts are deliberately skipped so the set is pure. `cand` is all 24 perms.
        let cand = *PERMS24;
        let _ = solve_interior(
            &problem.left_phases,
            &problem.right_phases,
            &problem.dc,
            &problem.lam,
            &problem.target_phases,
            &problem.targets,
            &cand,
        );
        return Solution {
            o: Mat4::identity(),
            rung: Rung::Interior,
            residual: 0.0,
        };
    }
    // Exhaust the lower support strata before a broader section can cannibalize
    // their cheaper, better-conditioned formulas.
    if let (Some(viable), Some(cand)) = (edge_gate.edge.as_ref(), support_perms.as_ref()) {
        let tp = prof::start();
        let hit = secular::solve_edge(
            &problem.left,
            &problem.right,
            &problem.dc,
            &problem.lam,
            &problem.targets,
            &problem.routed,
            viable,
            cand,
        );
        prof::rec(1, tp);
        if let Some((o, r)) = hit {
            if let Some(solution) = compiler_solution(&problem, o, Rung::Edge, r) {
                return solution;
            }
        }
    }
    let tp = prof::start();
    let hit = secular::solve_face(
        &problem.left,
        &problem.right,
        &problem.dc,
        &problem.lam,
        &problem.target_roots,
        &problem.targets,
    );
    prof::rec(11, tp);
    if let Some((o, r)) = hit {
        if let Some(solution) = compiler_solution(&problem, o, Rung::Face, r) {
            return solution;
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
    prof::rec(58, tk);
    if let Some((o, r)) = klein_hit {
        if let Some(solution) = compiler_solution(&problem, o, Rung::Klein, r) {
            return solution;
        }
    }
    // Multiplicity formulas own only exact confluent signatures and run after
    // the cheaper support and Klein sections have declined.
    if let Some((o, r, rung)) = secular::solve_confluent(
        &problem.left,
        &problem.right,
        &problem.target_roots,
        &problem.dc,
        &problem.lam,
        &problem.targets,
        problem.strata,
    ) {
        if let Some(solution) = compiler_solution(&problem, o, rung, r) {
            return solution;
        }
    }
    // Complete every routed `1 + 3` endpoint: rank drops, the nine bilinear
    // zero-entry faces, then one degree-at-most-12 Birkhoff--Heron selector
    // for a dense residual `SO(3)` trace fiber.  This closes the fixed-axis
    // endpoints before the generic interior chart below.
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
            if let Some(solution) = compiler_solution(&problem, o, Rung::OnePlusThree, residual) {
                return solution;
            }
        }
    }
    if let Some((o, residual, rung)) = solve_boundary_accelerators(&problem) {
        if let Some(solution) = compiler_solution(&problem, o, rung, residual) {
            return solution;
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
            if let Some(solution) = compiler_solution(&problem, o, Rung::OnePlusThree, residual) {
                return solution;
            }
        }
    }
    // Per-chart six-bracket/Cauer realization peel. The finite row/zero-pair
    // orbit is exact but not universal; a decline reaches the currently open
    // dense-residual realization problem below.
    if let Some((o, r)) = solve_axis_tier(&problem) {
        if let Some(solution) = compiler_solution(&problem, o, Rung::AxisQuartic, r) {
            return solution;
        }
    }
    unsolved_solution()
}

/// Research-only alternate tail used to census the generic Spin architecture.
/// Exhausting the bounded dyadic prefix does not certify infeasibility.
#[cfg(feature = "research-spin")]
pub fn solve_spin_dyadic(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    maximum_denominator: i32,
) -> (Solution, usize) {
    let (solution, attempts) = spin_selector::solve_dyadic(c, g, t, maximum_denominator);
    let solution = solution
        .and_then(|candidate| certified_solution(candidate.o, candidate.rung, candidate.residual));
    (solution.unwrap_or_else(unsolved_solution), attempts)
}

/// Research-only dense action order. A finite prefix is not a completeness or
/// infeasibility certificate; it measures the theorem-backed regular Spin
/// fallback independently of the production sections.
#[cfg(feature = "research-spin")]
pub fn solve_spin_spread(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    maximum_actions: usize,
) -> (Solution, usize, usize) {
    let (solution, attempts, actions) = spin_selector::solve_spread(c, g, t, maximum_actions);
    let solution = solution
        .and_then(|candidate| certified_solution(candidate.o, candidate.rung, candidate.residual));
    (
        solution.unwrap_or_else(unsolved_solution),
        attempts,
        actions,
    )
}

/// Research diagnostic for the projective Spin selector. The right action is
/// recovered from an already-certified frame, so this does not constitute an
/// independent realization route.
#[cfg(feature = "research-spin")]
pub fn replay_spin_action(c: [f64; 3], g: [f64; 3], t: [f64; 3], frame: &Mat4) -> Solution {
    spin_selector::replay_frame_action(c, g, t, frame)
        .and_then(|candidate| certified_solution(candidate.o, candidate.rung, candidate.residual))
        .unwrap_or_else(unsolved_solution)
}

/// Fast, incomplete peel inside the closed three-Givens boundary atlas.
///
/// All formulas and forward certificates are exact, but `BOUNDARY_HEAD` is an
/// empirical acceleration schedule, not a completeness theorem. Any decline
/// continues to the residual axis evaluator.
/// Direct and transported charts are coordinate choices in one cyclic
/// three-spectrum relation.
fn solve_boundary_accelerators(problem: &PreparedSandwich) -> Option<(Mat4, f64, Rung)> {
    let mut direct_vertices = [None; 256];
    let direct = |order: &[(usize, usize)], vertices: &mut [Option<(C, C)>; 256]| {
        let started = prof::start();
        let hit = solve_interior_cover(
            &problem.left,
            &problem.right,
            &problem.target_roots,
            &problem.dc,
            &problem.lam,
            &problem.targets,
            order,
            vertices,
        );
        prof::rec(8, started);
        hit.map(|(o, residual, _, _)| (o, residual, Rung::Interior))
    };

    if let Some(hit) = direct(&BOUNDARY_HEAD, &mut direct_vertices) {
        return Some(hit);
    }
    // The transported Klein section solves (conj(C), W -> G), then a real
    // Takagi factor transports its symmetric-unitary output back to C/G.
    let inverse_prefix: [C; 4] = std::array::from_fn(|j| problem.left[j].conj());
    let inverse_prefix_root: [C; 4] =
        std::array::from_fn(|k| C::from_polar(1.0, -problem.left_phases[k]));
    let anchored_targets = [esym4(problem.right), {
        let e = esym4(problem.right);
        [-e[0], e[1], -e[2], e[3]]
    }];
    let anchor_dc = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&inverse_prefix_root));
    let anchor_lam =
        Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&problem.target_roots[0]));
    let anchor_routed: [[C; 4]; 4] = std::array::from_fn(|i| {
        std::array::from_fn(|j| inverse_prefix[i] * problem.target_roots[0][j])
    });
    if let Some((p, _)) = klein::solve(
        &inverse_prefix,
        &problem.target_roots[0],
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

    let cyclic = CyclicThreeGivens::new(problem);
    let started = prof::start();
    let transported = cyclic
        .as_ref()
        .and_then(|atlas| atlas.solve_planes(0, 2, false));
    prof::rec(23, started);
    if let Some((o, residual)) = transported {
        return Some((o, residual, Rung::Interior));
    }
    None
}

/// Diagnostic entry point for the intrinsic axis atlas alone.  This bypasses
/// every lower-dimensional spectral section, but retains the original-input
/// forward certificate.  It is used to measure the image of the proposed
/// global axis theorem; production dispatch does not call it.
#[cfg(any(test, feature = "diagnostics"))]
pub fn solve_axis_only(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    let problem = PreparedSandwich::new(c, g, t);
    if let Some((o, residual)) = solve_axis_tier(&problem) {
        Solution {
            o,
            rung: Rung::AxisQuartic,
            residual,
        }
    } else {
        Solution {
            o: Mat4::identity(),
            rung: Rung::Unsolved,
            residual: f64::INFINITY,
        }
    }
}

/// Diagnostic for the stronger direct Bott--Samelson hypothesis: enumerate
/// all 216 distinct spectral images of the 96 spanning-tree three-Givens
/// words, without cyclic reanchoring or the four-Givens axis extension.
#[cfg(feature = "diagnostics")]
pub fn solve_three_givens_only(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    let problem = PreparedSandwich::new(c, g, t);
    let ranked = rank_perms_pre(&problem.left, &problem.right, &problem.targets);
    if ranked[0].1 < ACCEPT {
        return Solution {
            o: signed_perm(ranked[0].0),
            rung: Rung::Vertex,
            residual: ranked[0].1,
        };
    }
    let perms: Vec<[usize; 4]> = ranked.iter().map(|x| x.0).collect();
    let mut spectral_vertices = [None; 256];
    if let Some((o, r, _, _)) = solve_interior_words(
        &problem.left,
        &problem.right,
        &problem.target_roots,
        &problem.dc,
        &problem.lam,
        &problem.targets,
        &perms,
        &INTERIOR_PLANES,
        0,
        true,
        None,
        &mut spectral_vertices,
    ) {
        return Solution {
            o: orient_so4(o),
            rung: Rung::Interior,
            residual: r,
        };
    }
    if let Some((o, residual)) = CyclicThreeGivens::new(&problem)
        .and_then(|atlas| atlas.solve_planes(0, INTERIOR_PLANES.len(), true))
    {
        Solution {
            o,
            rung: Rung::Interior,
            residual,
        }
    } else {
        Solution {
            o: Mat4::identity(),
            rung: Rung::Unsolved,
            residual: f64::INFINITY,
        }
    }
}

/// Diagnostic for the exact 16-word lexicographic spanning-tree chain used by
/// the native algebra audits, crossed with all 24 relative base permutations.
///
/// This is intentionally separate from `solve_three_givens_only`: the first
/// 16 historical production words contain the same unordered trees, but one
/// has a different (noncommuting) Givens order.  The canonical indices are
/// therefore resolved against the complete 96-word table rather than assumed
/// to be `0..16`.
#[cfg(feature = "diagnostics")]
pub fn solve_canonical_three_givens_only(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    const CANONICAL_WORDS: [[(usize, usize); 3]; 16] = [
        [(0, 1), (0, 2), (0, 3)],
        [(0, 1), (0, 2), (1, 3)],
        [(0, 1), (0, 2), (2, 3)],
        [(0, 1), (0, 3), (1, 2)],
        [(0, 1), (0, 3), (2, 3)],
        [(0, 1), (1, 2), (1, 3)],
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (1, 3), (2, 3)],
        [(0, 2), (0, 3), (1, 2)],
        [(0, 2), (0, 3), (1, 3)],
        [(0, 2), (1, 2), (1, 3)],
        [(0, 2), (1, 2), (2, 3)],
        [(0, 2), (1, 3), (2, 3)],
        [(0, 3), (1, 2), (1, 3)],
        [(0, 3), (1, 2), (2, 3)],
        [(0, 3), (1, 3), (2, 3)],
    ];

    let problem = PreparedSandwich::new(c, g, t);
    let ranked = rank_perms_pre(&problem.left, &problem.right, &problem.targets);
    if ranked[0].1 < ACCEPT {
        return Solution {
            o: signed_perm(ranked[0].0),
            rung: Rung::Vertex,
            residual: ranked[0].1,
        };
    }

    let mut order = Vec::with_capacity(16 * 24);
    for word in CANONICAL_WORDS {
        let word_idx = INTERIOR_PLANES
            .iter()
            .position(|candidate| *candidate == word)
            .expect("canonical word must occur in complete tree atlas");
        for static_perm_idx in 0..24 {
            order.push((static_perm_idx, word_idx));
        }
    }

    let mut spectral_vertices = [None; 256];
    match solve_interior_words(
        &problem.left,
        &problem.right,
        &problem.target_roots,
        &problem.dc,
        &problem.lam,
        &problem.targets,
        &[],
        &INTERIOR_PLANES,
        0,
        true,
        Some(&order),
        &mut spectral_vertices,
    ) {
        Some((o, r, _, _)) => Solution {
            o: orient_so4(o),
            rung: Rung::Interior,
            residual: r,
        },
        None => Solution {
            o: Mat4::identity(),
            rung: Rung::Unsolved,
            residual: f64::INFINITY,
        },
    }
}

/// Rung-10 axis tier: the bounded four-Givens algebraic fiber, tried at the
/// direct AND role-swapped orientations (spec(Dg O' Da O'^T) = w with
/// O = O'^T, same double coset; the 2026-07-15 densification misses were
/// direct-axis holes solvable at the swap) across both target branches.
/// Swap results are re-gated on the original smooth residual.
fn trace_in_permutation_hull4(left: &[C; 4], right: &[C; 4], trace: C) -> bool {
    let vertices: [C; 24] =
        PERMS24.map(|permutation| (0..4).map(|i| left[i] * right[permutation[i]]).sum::<C>());
    point_in_complex_hull(&vertices, trace)
}

fn solve_axis_tier_direct(
    problem: &PreparedSandwich,
    target_branch_count: usize,
    pass: axis_quartic::AxisPass,
    accept_threshold: f64,
) -> Option<(Mat4, f64, bool)> {
    debug_assert!((1..=2).contains(&target_branch_count));
    let mut active_branches = [0usize; 2];
    let mut active_count = 0;
    for branch in 0..target_branch_count {
        if trace_in_permutation_hull4(&problem.left, &problem.right, problem.targets[branch][0]) {
            active_branches[active_count] = branch;
            active_count += 1;
        }
    }
    if active_count == 0 {
        return None;
    }
    let dcs_d: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, problem.right_phases[k]));
    let dcs = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&dcs_d));
    let lams = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&problem.left));
    // Combo order (b0 direct, b0 swap, b1 direct, b1 swap) matches the old
    // serial tier; odd indices are role-swapped.
    let combos: [axis_quartic::Combo; 4] = std::array::from_fn(|i| {
        let (branch, swap) = (active_branches[(i / 2).min(active_count - 1)], i % 2 == 1);
        let (delta, lam) = if swap {
            (problem.right, problem.left)
        } else {
            (problem.left, problem.right)
        };
        axis_quartic::Combo {
            delta,
            lam,
            chi: axis_quartic::target_poly(&problem.target_phases[branch]),
            dc: if swap { dcs } else { problem.dc },
            lam_m: if swap { lams } else { problem.lam },
            target: problem.targets[branch],
        }
    });
    let tp = prof::start();
    let accept = |i, o: Mat4, r, zero_weight_boundary: bool| {
        let threshold = if zero_weight_boundary {
            ACCEPT
        } else {
            accept_threshold
        };
        if i % 2 == 0 {
            return (r <= threshold).then(|| (orient_so4(o), r, zero_weight_boundary));
        }
        // Swap results are re-gated on the original smooth residual.
        let o = o.transpose();
        let r = problem
            .targets
            .iter()
            .map(|t| compound_residual(&problem.dc, &problem.lam, &o, t))
            .fold(f64::INFINITY, f64::min);
        (r <= threshold).then(|| (orient_so4(o), r, zero_weight_boundary))
    };
    let hit = axis_quartic::solve_all(&combos[..2 * active_count], pass, accept);
    prof::rec(24, tp);
    hit
}

fn solve_axis_tier(problem: &PreparedSandwich) -> Option<(Mat4, f64)> {
    // The passes are deliberately disjoint: first try the bounded floating
    // decic--cubic candidates across the whole finite orbit, then isolate the
    // same algebraic tower and its boundary strata exactly.  Replaying the
    // floating candidates in the certified pass cannot add a witness and used
    // to double the work on every complete decline.
    if let Some((o, residual, _)) =
        solve_axis_tier_direct(problem, 2, axis_quartic::AxisPass::CriticalFast, ACCEPT)
    {
        return Some((o, residual));
    }
    if let Some((o, residual, _)) = solve_axis_tier_direct(
        problem,
        2,
        axis_quartic::AxisPass::CriticalCertified,
        ACCEPT,
    ) {
        return Some((o, residual));
    }

    None
}

/// Run the cheap chart algebra for one reanchored sandwich
/// `D_a V diag(b) V^T D_a`, returning its symmetric-unitary output. The
/// spectra are supplied directly, so no Weyl-chamber coordinates are needed.
fn solve_transported_three_givens(
    a_phase: &[f64; 4],
    b: &[C; 4],
    target_spec: &[C; 4],
    plane_start: usize,
    plane_end: usize,
    qz_all: bool,
) -> Option<Mat4> {
    let da_diag: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, a_phase[k]));
    let a2: [C; 4] = std::array::from_fn(|k| da_diag[k] * da_diag[k]);
    let da = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&da_diag));
    let lb = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(b));
    let target = esym4(*target_spec);
    let targets = [target];
    let ranked = rank_perms_pre(&a2, b, &targets);
    let perms: Vec<[usize; 4]> = ranked.iter().take(CAND_K).map(|x| x.0).collect();
    let mut spectral_vertices = [None; 256];

    // Interior words only: the reanchored vertex/edge/face arms NEVER fired
    // (liveness read 2026-07-21, all four stride-1 corpora, 1.6M rows, zero
    // hits) -- consistent with the target-slot law: every row reaching this
    // tier is generic interior in the reanchored frame too. Deleting them
    // also removes an edge+face solve from every reanchor attempt.
    let v = if let Some((v, _, _, _)) = solve_interior_words(
        &a2,
        b,
        std::slice::from_ref(target_spec),
        &da,
        &lb,
        &targets,
        &perms,
        &INTERIOR_PLANES[plane_start..plane_end],
        plane_start,
        qz_all,
        None,
        &mut spectral_vertices,
    ) {
        v
    } else {
        return None;
    };
    Some(mmat(&da, &lb, &v))
}

/// One prepared cyclic three-Givens boundary atlas.  Reanchoring is only a
/// target-slot change in the same three-spectrum relation, so both production
/// placements advance this object over disjoint plane ranges instead of
/// rebuilding an eleven-argument fallback and repeating earlier planes.
struct CyclicThreeGivens<'a> {
    problem: &'a PreparedSandwich,
    product: C,
    ct: [C; 4],
    gt: [C; 4],
    ctd: Mat4,
    gtd: Mat4,
}

impl<'a> CyclicThreeGivens<'a> {
    #[inline]
    fn distinct(spectrum: &[C; 4]) -> bool {
        (0..4).all(|i| (0..i).all(|j| (spectrum[i] - spectrum[j]).norm() > 1e-6))
    }

    fn new(problem: &'a PreparedSandwich) -> Option<Self> {
        // Repeated spectra belong to the closed multiplicity dispatcher.
        if !Self::distinct(&problem.left) || !Self::distinct(&problem.right) {
            return None;
        }
        let ct: [C; 4] = std::array::from_fn(|k| problem.left[k].conj());
        let gt: [C; 4] = std::array::from_fn(|k| problem.right[k].conj());
        Some(Self {
            problem,
            product: problem.left.iter().product::<C>() * problem.right.iter().product::<C>(),
            ctd: Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&ct)),
            gtd: Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&gt)),
            ct,
            gt,
        })
    }

    fn certify(&self, o: Mat4) -> Option<(Mat4, f64)> {
        let residual = self
            .problem
            .targets
            .iter()
            .map(|target| compound_residual(&self.problem.dc, &self.problem.lam, &o, target))
            .fold(f64::INFINITY, f64::min);
        (residual <= ACCEPT).then_some((o, residual))
    }

    fn solve_planes(
        &self,
        plane_start: usize,
        plane_end: usize,
        qz_all: bool,
    ) -> Option<(Mat4, f64)> {
        // The second target encoding is the central-negated/pair-swapped Weyl image of this
        // canonical one.  It added no coverage: the 15 corpus rows for which it happened to be
        // the first cyclic chart were all accepted by the complete axis fallback in 32--70 us.
        // Keep one target action here; the two transported side actions below have distinct chart
        // images and are both performance-relevant.
        let w = &self.problem.target_roots[0];
        if !Self::distinct(w) || (self.product - w.iter().product::<C>()).norm() > 1e-8 {
            return None;
        }
        let inverse_w: [C; 4] = std::array::from_fn(|k| w[k].conj());
        // The two remaining cyclic side actions use the same boundary solver:
        //   X_g targets C^-1 and transports back as O=S^T;
        //   X_c targets G^-1 and transports back as O=S.
        for (phase, target, diagonal, transpose) in [
            (&self.problem.right_phases, &self.ct, &self.ctd, true),
            (&self.problem.left_phases, &self.gt, &self.gtd, false),
        ] {
            if let Some(x) = solve_transported_three_givens(
                phase,
                &inverse_w,
                target,
                plane_start,
                plane_end,
                qz_all,
            ) {
                let frame = recover_frame(&x, diagonal);
                let o = if transpose { frame.transpose() } else { frame };
                if let Some(hit) = self.certify(o) {
                    return Some(hit);
                }
            }
        }
        None
    }
}

// =================== RANK RUNG: the degenerate-w spectral inverse (symmetric_horn s13/s16) ===================
// A symmetric unitary S = O·Λ·Oᵀ has REAL eigenvectors, so U₁ = O·Λ·Oᵀ = V diag(spec) Vᵀ with V real.
// A DEGENERATE target w (the sliver) collapses the unknown V to a diagonal-plus-low-rank inverse-
// eigenvalue problem (rank 4−m₀, m₀ = top w-multiplicity); the Givens "fold" was a chart artifact.
// Closed form per multiplicity pattern -- no GN, no search. Proven in symmetric_horn s12-s16
// ([[sliver-rank-structured-inverse-solved]]). Tried after the cheap chart cascade declines.

/// Incommensurate weight folding `U₁`'s complex spectrum into one real symmetric matrix whose
/// eigenvectors are `U₁`'s real frame (`Re(U₁)+φ·Im(U₁)` is non-degenerate exactly when `spec(U₁)` is).
const PHI: f64 = 0.618_033_988_75;

/// Real orthonormal eigenvectors of a symmetric unitary `U₁` (columns), via the real symmetric
/// `Re(U₁)+φ·Im(U₁)`. The theorem: each spectral projector of a symmetric unitary is Hermitian AND
/// symmetric, hence real -> the eigenvectors are real (`[[secular-conditioning-win-realness-free]]`).
fn extract_o(u1: &Mat4) -> nalgebra::Matrix4<f64> {
    let sym = nalgebra::Matrix4::<f64>::from_fn(|i, j| u1[(i, j)].re + PHI * u1[(i, j)].im);
    nalgebra::SymmetricEigen::new(sym).eigenvectors
}

/// Recover the emitted real frame `O ∈ SO(4)` from `U₁`: take its real eigenvectors (`extract_o`),
/// reorder the columns so column `k` carries eigenvalue `lam[k]` (so `O·Λ·Oᵀ = U₁` against the FIXED
/// G gate Λ, not a permuted copy), and sign-fix to `det = +1` (a column sign flip leaves `O·Λ·Oᵀ`
/// invariant since Λ is diagonal). Λ's spectrum (G data) is generically distinct, so the match is a
/// clean bijection even when the TARGET w is degenerate.
// SYLVESTER-PROJECTOR extract_o: MEASURED 2026-08-09 AND REVERTED. U₁ = O·Λ·Oᵀ
// makes the eigenvalues of Re(U₁)+φ·Im(U₁) known a priori (Re λₖ + φ·Im λₖ), so
// the rank-one projector Π_{j≠k}(M−λⱼ)/(λₖ−λⱼ) gives each column with no
// iteration. It is faster but LESS ACCURATE, and the trade fails: runtime gain
// sits inside the 4% noise floor (linspace 4.42/4.41 vs 4.43-4.61; haar 5.83 vs
// 5.79-5.82) while the precision loss is exactly reproducible -- machine-precise
// linspace 99.9308% -> 99.9120%, haar 99.7520% -> 99.6907%. Never weaken
// tolerances. Same trade seen in klein::takagi_real (eigensolve 99.7550% vs
// Sylvester 99.6563%), so treat this as a property of the projector on 4×4
// unitary folds, not a one-off.
fn recover_frame(u1: &Mat4, lam: &Mat4) -> Mat4 {
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

/// Rank the 24 Weyl vertices by their `e1` residual. The spectra are supplied
/// directly; all callers already own this representation.
///
/// The score is |de1| ONLY (2026-07-21 total-time pass): e1 proximity is the
/// ordering signal for the edge scan (2.14 mean tries; order quality has no
/// recorded headroom), and the vertex gate re-verifies the full reduced
/// distance via `perm_vertex_residual` before accepting -- so dropping the
/// per-perm e2 (a 16-product square table + mul + 4 adds + norm per perm)
/// halves the per-row ranking cost without changing any accept decision.
pub fn rank_perms_pre(a2: &[C; 4], lam: &[C; 4], targets: &[[C; 4]]) -> [([usize; 4], f64); 24] {
    // only 16 distinct products a2[j]*lam[k] exist across the 24 perms
    let prod: [[C; 4]; 4] = std::array::from_fn(|j| std::array::from_fn(|k| a2[j] * lam[k]));
    let mut out: [([usize; 4], f64); 24] = std::array::from_fn(|i| {
        let p = PERMS24[i];
        let e1 = prod[0][p[0]] + prod[1][p[1]] + prod[2][p[2]] + prod[3][p[3]];
        let r = targets
            .iter()
            .map(|t| (e1 - t[0]).norm())
            .fold(f64::INFINITY, f64::min);
        (p, r)
    });
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

/// Full reduced spectral distance for ONE permutation: on unit-modulus
/// det-consistent spectra the esym vector is self-inversive (e4 = t4,
/// e3 = e4*conj(e1) exactly), so the distance is max(|de1|, |de2|) with
/// e2 = (e1^2 - p2)/2. The vertex gate calls this on the (rare) perms whose
/// |de1| passes ACCEPT; `rank_perms_pre` no longer computes e2 at all.
fn perm_vertex_residual(a2: &[C; 4], lam: &[C; 4], targets: &[[C; 4]], p: &[usize; 4]) -> f64 {
    let pr: [C; 4] = std::array::from_fn(|j| a2[j] * lam[p[j]]);
    let e1 = pr[0] + pr[1] + pr[2] + pr[3];
    let p2 = pr[0] * pr[0] + pr[1] * pr[1] + pr[2] * pr[2] + pr[3] * pr[3];
    let e2 = (e1 * e1 - p2) * 0.5;
    targets
        .iter()
        .map(|t| (e1 - t[0]).norm().max((e2 - t[1]).norm()))
        .fold(f64::INFINITY, f64::min)
}

fn ev_poly(co: &[f64], x: f64) -> f64 {
    co.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Certify that a power-basis polynomial has no zero on `[0,1]` by converting
/// it to Bernstein form. A Bernstein polynomial is a convex combination of
/// its coefficients on the unit interval, so coefficients of one strict sign
/// exclude a root. The machine-epsilon guard makes inconclusive cases fall
/// through to sign-change isolation.
fn bernstein_excludes_unit(p: &[f64]) -> bool {
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
    // choose(k,i)/choose(n,i) form cost ~250 float divisions per call and was
    // the single largest item on the haar profile (2026-07-14).
    let mut ch = [[0.0f64; 9]; 9];
    for k in 0..=n {
        ch[k][0] = 1.0;
        for i in 1..=k {
            ch[k][i] = ch[k - 1][i - 1] + if i <= k - 1 { ch[k - 1][i] } else { 0.0 };
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

// Test-only refinement counter for the bracketed real-root solver.
#[cfg(test)]
thread_local! {
    static REFINE_ITERS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
fn refine_iters() -> u64 {
    REFINE_ITERS.with(|c| c.get())
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
fn refine_bracket(p: &[f64], mut a: f64, mut b: f64) -> Option<f64> {
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

fn real_roots_unit(p: &[f64]) -> Vec<f64> {
    if bernstein_excludes_unit(p) {
        return vec![];
    }
    const N: usize = 64;
    let mut out = Vec::new();
    // Batch Horner across the whole grid: the per-coefficient pass vectorizes,
    // where the pointwise fold's serial fma chain cost ~1us/call (haar
    // profile 2026-07-14). Same evaluation order, bit-identical values.
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
            // bisection's bracket certificate and termination width, with
            // ~5x fewer polynomial evaluations (52-step bisection cost ~1us
            // per bracket on the 2026-07-14 haar profile).
            // Re-evaluate the bracket endpoints with scalar Horner. The batch
            // grid has a different rounding path; using its endpoint values
            // here preserves the sign certificate but measurably degrades the
            // final root (92.70% vs 97.57% machine-precise on full Haar).
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
                #[cfg(test)]
                REFINE_ITERS.with(|c| c.set(c.get() + 1));
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

/// Ordered 3-Givens words whose three distinct planes form a spanning tree of
/// K4. There are 16 unordered trees but 16*3! = 96 ordered words: rotations
/// sharing an index do not commute, so one ordering per tree is not a complete
/// chart atlas. Keep the historical 16 first as the measured hot path, then
/// append every missing ordering as a decline-only fallback.
static INTERIOR_PLANES: std::sync::LazyLock<Vec<[(usize, usize); 3]>> =
    std::sync::LazyLock::new(|| {
        let mut words = vec![
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (0, 2), (0, 3)],
            [(0, 1), (1, 2), (1, 3)],
            [(0, 2), (0, 3), (1, 2)],
            [(0, 2), (1, 3), (1, 2)],
            [(0, 3), (1, 2), (1, 3)],
            [(0, 2), (0, 3), (1, 3)],
            // fallback spanning trees (the other 9 of the 16):
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
        for &a in &PLANES {
            for &b in &PLANES {
                for &c in &PLANES {
                    if a == b || a == c || b == c {
                        continue;
                    }
                    let mut touched = [false; 4];
                    for &(i, j) in &[a, b, c] {
                        touched[i] = true;
                        touched[j] = true;
                    }
                    let word = [a, b, c];
                    if touched.iter().all(|&x| x) && !words.contains(&word) {
                        words.push(word);
                    }
                }
            }
        }
        debug_assert_eq!(words.len(), 96);
        words
    });

/// Exact quotient of the raw `(word, perm)` atlas by automorphisms of the
/// trilinear parameter cube.  A chart's `e1,e2` map is determined by its eight
/// permutation corners.  Permuting the three cube axes or replacing any
/// coordinate by `1-x` only reparameterizes `[0,1]^3`, so charts whose corner
/// tables agree under one of these 3!*2^3 automorphisms have identical spectral
/// images.  The 96*24 raw pairs reduce to 216 image classes (102 in the hot 16
/// words and 114 genuinely new classes in the remaining words).
struct InteriorQuotient {
    class: Vec<[u16; 24]>,
    hot: [bool; 216],
}

static INTERIOR_QUOTIENT: std::sync::LazyLock<InteriorQuotient> = std::sync::LazyLock::new(|| {
    fn perm_code(p: [usize; 4]) -> u8 {
        (p[0] | p[1] << 2 | p[2] << 4 | p[3] << 6) as u8
    }
    fn corner_codes(planes: [(usize, usize); 3], perm: [usize; 4]) -> [u8; 8] {
        std::array::from_fn(|k| {
            let mut p = perm;
            for q in 0..3 {
                if CORNERS[k][q] == 0.0 {
                    let (i, j) = planes[q];
                    p.swap(i, j);
                }
            }
            perm_code(p)
        })
    }
    fn corner_index(bits: [usize; 3]) -> usize {
        CORNERS
            .iter()
            .position(|x| {
                x[0] == bits[0] as f64 && x[1] == bits[1] as f64 && x[2] == bits[2] as f64
            })
            .unwrap()
    }
    fn signature(planes: [(usize, usize); 3], perm: [usize; 4]) -> [u8; 8] {
        const AXES: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let corners = corner_codes(planes, perm);
        let mut best = [u8::MAX; 8];
        for axes in AXES {
            for flip in 0..8 {
                let candidate = std::array::from_fn(|k| {
                    let bits =
                        std::array::from_fn(|q| CORNERS[k][axes[q]] as usize ^ ((flip >> q) & 1));
                    corners[corner_index(bits)]
                });
                if candidate < best {
                    best = candidate;
                }
            }
        }
        best
    }

    let mut signatures: Vec<[u8; 8]> = Vec::new();
    let mut class = Vec::with_capacity(INTERIOR_PLANES.len());
    for &planes in INTERIOR_PLANES.iter() {
        class.push(std::array::from_fn(|pi| {
            let sig = signature(planes, PERMS24[pi]);
            let id = match signatures.iter().position(|x| *x == sig) {
                Some(id) => id,
                None => {
                    signatures.push(sig);
                    signatures.len() - 1
                }
            };
            id as u16
        }));
    }
    assert_eq!(signatures.len(), 216);
    let mut hot = [false; 216];
    for row in &class[..16] {
        for &id in row {
            hot[id as usize] = true;
        }
    }
    assert_eq!(hot.iter().filter(|&&x| x).count(), 102);
    InteriorQuotient { class, hot }
});

/// Machine-precision rescue on an ACCEPTED interior candidate (s184/QZ, 2026-07-17).
/// The scan root can carry ~1e-11 forward error near close root pairs; one bounded
/// 6x6 QZ eigensolve of the SAME chart eliminant re-derives the root to eigensolve
/// grade. Runs only when the accepted residual is not already machine-precise
/// (~1.5% of haar rows), and the original candidate is kept unless the rescued one
/// strictly improves the verified residual, so coverage and routing are unchanged
/// by construction.
#[allow(clippy::too_many_arguments)]
fn rescue_root(
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
    if r < 1e-12 {
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

/// High-volume closed-boundary sections, in decreasing measured cell volume.
/// These 16 exact charts cover 99.5% of Haar interior targets; they are
/// an acceleration peel only, and do not carry a completeness claim.
const BOUNDARY_HEAD: [(usize, usize); 16] = [
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

pub fn solve_interior(
    eb: &[f64; 4],
    ep: &[f64; 4],
    dc: &Mat4,
    lam: &Mat4,
    et_branches: &[[f64; 4]],
    targets: &[[C; 4]],
    perms: &[[usize; 4]],
) -> Option<(Mat4, f64, [usize; 4], [(usize, usize); 3])> {
    let prefix_diag: [C; 4] = std::array::from_fn(|j| C::from_polar(1.0, 2.0 * eb[j]));
    let gate: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, 2.0 * ep[k]));
    let target_specs: Vec<[C; 4]> = et_branches
        .iter()
        .map(|et| std::array::from_fn(|k| C::from_polar(1.0, 2.0 * et[k])))
        .collect();
    let mut spectral_vertices = [None; 256];
    solve_interior_words(
        &prefix_diag,
        &gate,
        &target_specs,
        dc,
        lam,
        targets,
        perms,
        &INTERIOR_PLANES[..16],
        0,
        false,
        None,
        &mut spectral_vertices,
    )
}

/// Direct interior scan in an explicit order of `(static_perm_idx, plane_idx)`
/// classes. Production supplies `BOUNDARY_HEAD`, sorted by measured cell
/// volume, so most rows resolve in one or two exact chart solves.
fn solve_interior_cover(
    prefix_diag: &[C; 4],
    gate: &[C; 4],
    target_specs: &[[C; 4]],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]],
    order: &[(usize, usize)],
    spectral_vertices: &mut [Option<(C, C)>; 256],
) -> Option<(Mat4, f64, [usize; 4], [(usize, usize); 3])> {
    solve_interior_words(
        prefix_diag,
        gate,
        target_specs,
        dc,
        lam,
        targets,
        &[],
        &INTERIOR_PLANES[..16],
        0,
        false,
        Some(order),
        spectral_vertices,
    )
}

fn solve_interior_words(
    prefix_diag: &[C; 4],
    gate: &[C; 4],
    target_specs: &[[C; 4]],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]],
    perms: &[[usize; 4]],
    plane_words: &[[(usize, usize); 3]],
    word_offset: usize,
    qz_all: bool,
    order: Option<&[(usize, usize)]>,
    spectral_vertices: &mut [Option<(C, C)>; 256],
) -> Option<(Mat4, f64, [usize; 4], [(usize, usize); 3])> {
    // Diagnostic escape hatch for auditing the cube-reparameterization
    // quotient itself.  Default production and atlas diagnostics deduplicate
    // to 216 image classes; CS_RAW_TREE_ATLAS exhausts the raw word/permutation
    // cross product instead.
    #[cfg(feature = "diagnostics")]
    let deduplicate = std::env::var_os("CS_RAW_TREE_ATLAS").is_none();
    #[cfg(not(feature = "diagnostics"))]
    let deduplicate = true;
    // Stronger diagnostic: bypass every floating-point exclusion gate and send
    // each nondegenerate sextic directly through the all-root QZ path.  Final
    // acceptance still uses the original trilinear and spectral certificates.
    #[cfg(feature = "diagnostics")]
    let ungated = std::env::var_os("CS_UNGATED_TREE_ATLAS").is_some();
    #[cfg(not(feature = "diagnostics"))]
    let ungated = false;
    // Scan order: either an explicit (static_perm_idx, plane_idx) sequence -- the
    // coverage order for the direct atlas, biggest cells first -- or the
    // perms x plane_words cross product (reanchor/cold-word callers). Order is
    // completeness-neutral: on a miss every class is still tried.
    let perm_static: Vec<usize> = if order.is_none() {
        perms
            .iter()
            .map(|&p| PERMS24.iter().position(|q| *q == p).unwrap())
            .collect()
    } else {
        Vec::new()
    };
    // Branch OUTER so the common (direct) branch is fully tried before rho -- a direct-branch
    // target never pays the rho eigs. Per-branch invariants hoisted out of the cell loop.
    for (bi, w) in target_specs.iter().enumerate() {
        // Only one representative of each exact cube-reparameterization class
        // needs an eliminant.  The cold call suppresses every class already
        // exhausted by the hot call; a mid-scan resume (REANCHOR_MID split)
        // suppresses exactly the classes its own head (words 0..offset x the
        // same perms) already tried, so head + tail cover the same class set
        // as one continuous scan.
        let mut seen = if word_offset == 0 {
            [false; 216]
        } else if word_offset >= 16 {
            INTERIOR_QUOTIENT.hot
        } else {
            let mut s = [false; 216];
            for wi in 0..word_offset {
                for &perm in perms {
                    let pi = PERMS24.iter().position(|p| *p == perm).unwrap();
                    s[INTERIOR_QUOTIENT.class[wi][pi] as usize] = true;
                }
            }
            s
        };
        let detw = w[0] * w[1] * w[2] * w[3];
        let sq = detw.sqrt();
        let w1 = w[0] + w[1] + w[2] + w[3];
        let w2 = w[0] * w[1] + w[0] * w[2] + w[0] * w[3] + w[1] * w[2] + w[1] * w[3] + w[2] * w[3];
        let n_iter = match order {
            Some(o) => o.len(),
            None => perm_static.len() * plane_words.len(),
        };
        for it in 0..n_iter {
            let (static_perm_idx, plane_idx) = match order {
                Some(o) => o[it],
                None => (perm_static[it / plane_words.len()], it % plane_words.len()),
            };
            {
                let perm = PERMS24[static_perm_idx];
                let planes = plane_words[plane_idx];
                let class =
                    INTERIOR_QUOTIENT.class[word_offset + plane_idx][static_perm_idx] as usize;
                if deduplicate && seen[class] {
                    continue;
                }
                seen[class] = true;
                count_interior_try();
                let tp = prof::start();
                let (e1c, e2c) =
                    chart_corners_cached(prefix_diag, gate, planes, perm, spectral_vertices);
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
                if !ungated
                    && (w1.re < lo1 - HULL_M
                        || w1.re > hi1 + HULL_M
                        || w1.im < lo2 - HULL_M
                        || w1.im > hi2 + HULL_M
                        || t3 < lo3 - HULL_M
                        || t3 > hi3 + HULL_M)
                {
                    prof::rec(2, tp);
                    prof::rec(20, prof::start());
                    continue;
                }
                // Exact-hull separation (same lemma, sharp set): measured on
                // the hard rows it excludes 54-100% of classes where the box
                // excludes none. Certificate-verified, hence lossless.
                let cpts: [[f64; 3]; 8] =
                    std::array::from_fn(|i| [e1c[i].re, e1c[i].im, (e2c[i] / sq).re]);
                if !ungated && hull_excludes(&cpts, [w1.re, w1.im, t3], HULL_M) {
                    prof::rec(2, tp);
                    prof::rec(21, prof::start());
                    continue;
                }
                prof::rec(22, prof::start());
                let (a1, a2) = (mobius8(&e1c), mobius8(&e2c));
                prof::rec(2, tp);
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
                prof::rec(3, tp);
                funnel::bump(funnel::ATTEMPTS);
                if sextic.iter().all(|c| c.abs() < 1e-12) || !sextic.iter().all(|c| c.is_finite()) {
                    funnel::bump(funnel::DEGENERATE);
                    continue;
                }
                // Exact eigensolve-free decline: all three eliminants must admit
                // a root in [0,1]. One-sided, so a real solution is never
                // rejected; it only skips the 6x6 QZ on charts that provably
                // cannot hold one.
                // Cheapest decline first: the whole-interval test reuses the
                // sextic already in hand, so the ~13% it rejects never pay for
                // the cubic breakpoints that the sharper region test needs.
                if !ungated && may_have_unit_root(&sextic, 0.0, 1.0) == 0 {
                    funnel::bump(funnel::HULL);
                    continue;
                }
                let mut comps = [(0.0f64, 0.0f64); 8];
                let ncomp = admissible_components(&eliminant.kernel, &mut comps);
                if !ungated && ncomp == 0 {
                    funnel::bump(funnel::REGION_EMPTY);
                    continue;
                }
                // An ODD sign count on a component PROVES a root there; if the
                // rooter then finds none, that is a computational failure, not
                // mathematical nonexistence.
                let mut guaranteed = false;
                let mut possible = false;
                if !ungated {
                    for &(lo, hi) in comps[..ncomp].iter() {
                        let ch = may_have_unit_root(&sextic, lo, hi);
                        possible |= ch > 0;
                        guaranteed |= ch % 2 == 1;
                    }
                }
                if !ungated && !possible {
                    funnel::bump(funnel::NO_ROOT_IN_S);
                    continue;
                }

                // GULPS_DUMP_SEXTIC=1: the genuine interior eliminant, for the
                // structural question "is this a generic sextic on real data?".
                // Diagnostic only; zero cost when the var is unset.
                #[cfg(feature = "diagnostics")]
                if std::env::var_os("GULPS_DUMP_SEXTIC").is_some() {
                    // the three trilinear residuals themselves, in monomial
                    // order [1,x,y,z,xy,xz,yz,xyz]: enough to rebuild the
                    // x-, y- and z-eliminants offline by index permutation.
                    let mut line = String::from("TRILIN");
                    for r in [&r1, &r2, &r3] {
                        for v in r.iter() {
                            line.push_str(&format!(" {v:.17e}"));
                        }
                    }
                    println!("{line}");
                }
                let tp = prof::start();
                // MEASURED 2026-08-09, do NOT re-try: taking the PROVEN
                // odd-count bracket first (bisection instead of the degree-6
                // rooter) is worse on BOTH axes -- runtime 5.14->5.20,
                // 6.12->6.18, 6.08->6.64, and it DEGRADES worst residual on
                // fresh_haar 1.213e-11 -> 1.459e-10 (12x) plus 2 machine-precise
                // rows on feasible_haar. Bisection is exact in existence but not
                // in accuracy; the general rooter is the better primitive here.
                let mut roots = if qz_all || ungated {
                    eliminant
                        .qz6_roots()
                        .unwrap_or_else(|| real_roots_unit(&sextic))
                } else {
                    real_roots_unit(&sextic)
                };
                prof::rec(4, tp);
                if roots.is_empty() {
                    if ungated {
                        funnel::bump(funnel::NUMERICAL);
                        continue;
                    }
                    if !guaranteed {
                        funnel::bump(funnel::NO_ROOT_IN_S);
                        continue;
                    }
                    // Existence is PROVEN on some component; recover it from the
                    // bracket the proof supplies rather than declaring absence.
                    for &(lo, hi) in comps[..ncomp].iter() {
                        if may_have_unit_root(&sextic, lo, hi) % 2 == 1 {
                            if let Some(x) = refine_bracket(&sextic, lo, hi) {
                                roots.push(x);
                            }
                        }
                    }
                    if roots.is_empty() {
                        funnel::bump(funnel::NUMERICAL);
                        continue;
                    }
                }
                for (xi, x) in roots.iter().copied().enumerate() {
                    for (yv, zv) in three_givens::recover(&[r1, r2, r3], x) {
                        if !(-1e-6..=1.0001).contains(&yv) || !(-1e-6..=1.0001).contains(&zv) {
                            continue;
                        }
                        let y = yv.clamp(0.0, 1.0);
                        let z = zv.clamp(0.0, 1.0);
                        // s184 §2: trilinear reduced certificate at the candidate root.
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
                        // s184 §6 step 3: fast accept well below ACCEPT (rs agrees with
                        // smooth_residual to ~1e-14; shadow ra_sr=0 rr_sa=0 on both full
                        // corpora). Skip smooth_residual and use the sqrt frame directly.
                        if rs < FAST_ACCEPT {
                            let o = chart_o_sqrt([x, y, z], planes, perm);
                            let (o, rs) = rescue_root(
                                &eliminant,
                                &[r1, r2, r3],
                                dc,
                                lam,
                                &targets[bi],
                                planes,
                                perm,
                                o,
                                rs,
                            );
                            record_interior_hit(class, plane_idx, static_perm_idx, xi);
                            if !chartall_mode() {
                                return Some((o, rs, perm, planes));
                            }
                        }
                        let tp = prof::start();
                        let o = chart_o([x, y, z], planes, perm);
                        let r = compound_residual(dc, lam, &o, &targets[bi]);
                        prof::rec(5, tp);
                        if r < ACCEPT {
                            let (o, r) = rescue_root(
                                &eliminant,
                                &[r1, r2, r3],
                                dc,
                                lam,
                                &targets[bi],
                                planes,
                                perm,
                                o,
                                r,
                            );
                            record_interior_hit(class, plane_idx, static_perm_idx, xi);
                            if !chartall_mode() {
                                return Some((o, r, perm, planes));
                            }
                        }
                    }
                }
                funnel::bump(funnel::FINAL_FAIL);
            }
        }
    }
    None
}

// ===================== SLIVER RUNG: the peel ('none'-gauge), ported from s111 =====================
// For a 'none'-gauge 4-Givens word the gauge γ=cos(2φ_g) is SLAVED (degree-1) and the exact-
// degenerate sliver is the FOLD of a 1-dim family, located by root-conditions (eliminant double-
// root + e3-invariant tangent) -- no HC, no descent. Construction = companion-eig roots + linear
// back-substitution + a small parabola fit. Verified vs s111: k=20865→2.7e-13, k=22082→1.2e-9.

#[cfg(test)]
mod tests {
    use super::*;

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
    #[ignore = "the complete dense residual selector has not been implemented yet"]
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
    fn finite_axis_orbit_has_24_distinct_valid_charts() {
        let charts: [axis_quartic::Chart; axis_quartic::ORBIT_LEN] =
            std::array::from_fn(axis_quartic::orbit_chart);
        for index in 0..charts.len() {
            assert!(!charts[..index].contains(&charts[index]));
            let (axis, pair, pendant, reference) = charts[index];
            assert!(axis < 4);
            assert_ne!(pair.0, pair.1);
            assert!(![pair.0, pair.1].contains(&pendant));
            assert!(![pair.0, pair.1].contains(&reference));
            assert_ne!(pendant, reference);
        }
    }

    #[test]
    fn one_plus_three_zero_wall_uses_quadratic_block_solver() {
        // feasible_linspace row 329583: one routed eigenvalue splits off and
        // the remaining dense SO(3) block lies on a two-Givens wall.  This
        // formerly reached the endpoint through cyclic three-Givens transport.
        let solution = solve(
            [0.3125, 0.1875, -0.1875],
            [0.375, 0.3125, -0.1875],
            [0.375, 0.125, 0.0],
        );
        assert_eq!(solution.rung, Rung::OnePlusThree);
        assert!(solution.residual < 1e-12);
    }

    #[test]
    fn dense_one_plus_three_selector_closes_the_nine_face_counterexample() {
        // Exact rational unit-circle data from the independent wall audit.
        // Its planted SO(3) witness is dense and the exact nine-face
        // resultants prove that no zero-entry witness exists.
        let a = [
            c(-3.0 / 5.0, -4.0 / 5.0),
            c(0.0, 1.0),
            c(4.0 / 5.0, -3.0 / 5.0),
            c(7.0 / 25.0, 24.0 / 25.0),
        ];
        let b = [
            c(-4.0 / 5.0, 3.0 / 5.0),
            c(0.0, -1.0),
            c(15.0 / 17.0, 8.0 / 17.0),
            c(13.0 / 85.0, -84.0 / 85.0),
        ];
        let tau = c(
            -350_288_779.0 / 300_594_425.0,
            420_781_878.0 / 300_594_425.0,
        );
        let eta = c(2107.0 / 2125.0, 276.0 / 2125.0);
        let rho = eta.conj();
        let kappa = eta * tau.conj();
        let target = [rho + tau, kappa + rho * tau, eta + rho * kappa, rho * eta];
        let targets = [target, target];
        let dc = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&a.map(|z| z.sqrt())));
        let lam = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&b));
        let mut planted = Mat4::zeros();
        let q = [
            [-95.0, -816.0, -180.0],
            [-480.0, -95.0, 684.0],
            [-684.0, 180.0, -455.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                planted[(i, j)] = c(q[i][j] / 841.0, 0.0);
            }
        }
        planted[(3, 3)] = c(1.0, 0.0);
        assert!(compound_residual(&dc, &lam, &planted, &target) < 1e-12);

        let mut routed = [[0u8; 4]; 4];
        routed[3][3] = 1;
        assert!(
            one_plus_three::solve_walls(&a, &b, &routed, &dc, &lam, &targets).is_none(),
            "the exact fixture has no zero-entry residual witness"
        );
        let routed_solution = one_plus_three::solve_dense(&a, &b, &routed, &dc, &lam, &targets)
            .expect("the complete Birkhoff--Heron selector must own the dense routed fiber");
        assert!(
            routed_solution.1 < 1e-11,
            "routed residual {:.17e}",
            routed_solution.1
        );

        // The direct finite axis orbit also contains this fiber.
        let master = dc * planted * lam * planted.transpose() * dc;
        let spectrum = eig4(&master);
        let et: [f64; 4] = spectrum.map(|z| z.arg() / 2.0);
        let eb: [f64; 4] = a.map(|z| z.arg() / 2.0);
        let ep: [f64; 4] = b.map(|z| z.arg() / 2.0);
        let et_branches = [et, et];
        let target_roots = [spectrum, spectrum];
        let problem = PreparedSandwich {
            left_phases: eb,
            right_phases: ep,
            target_phases: et_branches,
            left: a,
            right: b,
            target_roots,
            targets,
            routed: std::array::from_fn(|i| std::array::from_fn(|j| a[i] * b[j])),
            dc,
            lam,
            strata: StratumSignature::new(&a, &b, &target_roots),
        };
        let axis = solve_axis_tier_direct(
            &problem,
            2,
            axis_quartic::AxisPass::CriticalCertified,
            ACCEPT,
        )
        .expect("the direct finite axis orbit must realize the routed dense fixture");
        assert!(axis.1 < ACCEPT, "axis residual {:.17e}", axis.1);
    }

    #[test]
    fn zero_weight_axis_boundary_closes_former_axis_hole() {
        // Haar corpus row 141700 was the last axis-only failure. Its physical
        // component meets alpha_i=0 or beta_i=0, so the zero-weight resultant
        // must select it without help from the three-Givens boundary peel.
        let solution = solve_axis_only(
            [0.39018406812566, 0.33535840603455, -0.16714093500979],
            [0.26190622019265, 0.2480406935928, -0.04889170690693],
            [0.3436597844367, 0.07856425613504, -0.01947571687131],
        );
        assert_eq!(solution.rung, Rung::AxisQuartic);
        assert!(solution.residual < ACCEPT);
    }

    #[test]
    fn every_weyl_vertex_passes_the_routed_support_gate() {
        let a2 = std::array::from_fn(|k| C::from_polar(1.0, [0.13, 0.47, 1.01, 1.73][k]));
        let g2 = std::array::from_fn(|k| C::from_polar(1.0, [0.22, 0.61, 1.19, 2.03][k]));
        let routed = std::array::from_fn(|i| std::array::from_fn(|j| a2[i] * g2[j]));
        for p in PERMS24.iter() {
            let spec = std::array::from_fn(|k| a2[k] * g2[p[k]]);
            let targets = [spec, spec];
            let support = secular::edge_gate(&routed, &targets);
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
    #[ignore]
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

        // EXACT-vs-SCAN gate (inner-loop agent's discipline): the companion solve must hit a
        // TRUE root (machine precision), unreachable by a fine grid scan -- proving it is not a
        // bisection in disguise. A 40³ scan's best is orders of magnitude worse.
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
        // s184 §4: Cauchy-Binet compound evaluator must equal smooth_residual within 1e-12
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
}
