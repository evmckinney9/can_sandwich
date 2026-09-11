//! The distilled can-sandwich solver: one curve, two strata, one
//! residue law.
//!
//! Every instance defines a real genus-3 curve. Gate-degenerate rows
//! are the HYPERELLIPTIC stratum: solutions are theta characteristics,
//! computed in radicals -- skeletons (even characteristics: 4+4 data
//! splits with the det-pin product), pin pairs (odd characteristics:
//! the beta-cubic closure), and split pairs (the quartic closure). The
//! all-generic stratum is the SMOOTH QUARTIC, where radicals are
//! impossible by the classical bitangent theorem, and the certified
//! bounded eigensolve (the production dispatcher) is the sanctioned
//! form. Gates are recovered by ONE confluent residue law applied to
//! both factors (v from chi_mu with rho1, u from chi_mu' with rho2),
//! signs by the ray/involution structure, one forward verify.
use nalgebra::{Complex, Matrix4};

pub type C = Complex<f64>;
pub type Mat4 = Matrix4<C>;

pub const CTOL: f64 = 2e-3;
pub const XTOL: f64 = 1e-9;

/// Frame ingredients of a radical-stratum solution: `m = assemble(c, a,
/// &peels)`, so the realizing orthogonal frame has the peel vectors as
/// columns (gate values `c + rho`) completed orthonormally on the
/// c-eigenspace. `None` for the certified pass-through, whose `m` is
/// already a production frame.
pub struct Frame {
    pub c: C,
    pub peels: Vec<([f64; 4], C)>,
}

pub struct Solved {
    pub m: Mat4,
    pub frame: Option<Frame>,
}

/// Stack-allocated partition of exactly 4 indices into non-empty clusters.
/// Labels 0..k are assigned in first-seen (smallest index) order, so
/// cluster_rep(c) is always the smallest index in that cluster.
#[derive(Clone, Copy)]
struct Clusters4 {
    cluster_of: [u8; 4],
    k: u8,
}
impl Clusters4 {
    /// Size of cluster c.
    #[inline]
    fn cluster_len(&self, c: u8) -> usize {
        self.cluster_of.iter().filter(|&&x| x == c).count()
    }
    /// Smallest (first) index in cluster c.
    #[inline]
    fn cluster_rep(&self, c: u8) -> usize {
        self.cluster_of.iter().position(|&x| x == c).unwrap()
    }
    /// Second-smallest index in cluster c (panics if cluster size < 2).
    #[inline]
    fn cluster_second(&self, c: u8) -> usize {
        let rep = self.cluster_rep(c);
        (rep + 1..4).find(|&j| self.cluster_of[j] == c).unwrap()
    }
}

fn clusters4(z: &[C; 4], tol: f64) -> Clusters4 {
    let tol2 = tol * tol;
    let mut cluster_of = [u8::MAX; 4];
    let mut k = 0u8;
    for i in 0..4 {
        if cluster_of[i] == u8::MAX {
            cluster_of[i] = k;
            for j in (i + 1)..4 {
                if cluster_of[j] == u8::MAX && (z[i] - z[j]).norm_sqr() < tol2 {
                    cluster_of[j] = k;
                }
            }
            k += 1;
        }
    }
    Clusters4 { cluster_of, k }
}

/// The confluent inheritance theorem at the candidate boundary.  A delta
/// cluster of size `s` contributes `s - 1` copies of its root to every
/// admissible intermediate spectrum.  Test this before deflation: removing
/// the "nearest" unrelated root and relying on a later residue failure both
/// obscures the normalized lower-genus problem and wastes reconstruction
/// work.
#[inline]
fn inherits_clusters(mu: &[C; 4], delta: &[C; 4], dcl: &Clusters4) -> bool {
    (0..dcl.k).all(|cluster| {
        let multiplicity = dcl.cluster_len(cluster);
        multiplicity < 2
            || mu
                .iter()
                .filter(|&&root| (root - delta[dcl.cluster_rep(cluster)]).norm_sqr() < 1e-8)
                .count()
                >= multiplicity - 1
    })
}

pub fn sdist(x: &[C; 4], y: &[C; 4]) -> f64 {
    fn heap(k: usize, idx: &mut [usize; 4], x: &[C; 4], y: &[C; 4], best: &mut f64) {
        if k == 1 {
            let d = (0..4)
                .map(|i| (x[i] - y[idx[i]]).norm())
                .fold(0.0f64, f64::max);
            *best = best.min(d);
            return;
        }
        for i in 0..k {
            heap(k - 1, idx, x, y, best);
            if k.is_multiple_of(2) {
                idx.swap(i, k - 1)
            } else {
                idx.swap(0, k - 1)
            }
        }
    }
    let mut idx = [0usize, 1, 2, 3];
    let mut best = f64::INFINITY;
    heap(4, &mut idx, x, y, &mut best);
    best
}

fn eig4(m: &Mat4) -> Option<[C; 4]> {
    if (0..4).any(|i| (0..4).any(|j| !m[(i, j)].re.is_finite() || !m[(i, j)].im.is_finite())) {
        return None;
    }
    let fm = faer::Mat::<C>::from_fn(4, 4, |i, j| m[(i, j)]);
    let ev = fm.eigenvalues().ok()?;
    Some(std::array::from_fn(|i| ev[i]))
}

fn assemble(c: C, a: &[C; 4], peels: &[([f64; 4], C)]) -> Mat4 {
    let d: [C; 4] = std::array::from_fn(|j| C::from_polar(1.0, a[j].arg() / 2.0));
    let mut m = Mat4::from_fn(|i, j| if i == j { c * a[i] } else { C::default() });
    for (vec, rho) in peels {
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] += *rho * d[i] * d[j] * C::new(vec[i] * vec[j], 0.0);
            }
        }
    }
    m
}

/// Monic quartic from four roots, highest-first, no allocation.
fn chi4(r: &[C; 4]) -> [C; 5] {
    let mut c = [C::default(); 5];
    c[0] = C::new(1.0, 0.0);
    for (k, &t) in r.iter().enumerate() {
        for i in (1..=k + 1).rev() {
            let prev = c[i - 1];
            c[i] -= t * prev;
        }
    }
    c
}

/// Closed-form roots for degree <= 3 (leading-trimmed like
/// cpoly::roots): quadratic formula and complex Cardano. The beta
/// polynomial is rooted once per pin pair, so this removes the hot
/// path's companion eigensolve entirely.
fn roots_small(p: &[C]) -> ([C; 4], usize) {
    let zero = C::default();
    let maxc = p.iter().map(|c| c.norm()).fold(0.0f64, f64::max);
    if maxc == 0.0 {
        return ([zero; 4], 0);
    }
    let mut i0 = 0;
    while i0 < p.len() && p[i0].norm() < 1e-13 * maxc {
        i0 += 1;
    }
    let p = &p[i0..];
    match p.len() {
        0 | 1 => ([zero; 4], 0),
        2 => {
            let mut out = [zero; 4];
            out[0] = -p[1] / p[0];
            (out, 1)
        }
        3 => {
            let disc = (p[1] * p[1] - p[0] * p[2] * 4.0).sqrt();
            // Branch-matched quadratic: the large root from the non-cancelling
            // numerator, the small root from the product c/a -- the naive
            // (-b +- disc)/2a form loses the small root's digits when
            // b ~ +-disc (the closure chain's wall data).
            let (np_, nm_) = (-p[1] + disc, -p[1] - disc);
            let big = if np_.norm() >= nm_.norm() { np_ } else { nm_ } / (p[0] * 2.0);
            let mut out = [zero; 4];
            out[0] = big;
            out[1] = if big.norm() > 1e-300 {
                p[2] / (p[0] * big)
            } else {
                big
            };
            (out, 2)
        }
        4 => {
            let bb = p[1] / p[0];
            let cc = p[2] / p[0];
            let dd = p[3] / p[0];
            let pd = cc - bb * bb / 3.0;
            let qd = bb * bb * bb * (2.0 / 27.0) - bb * cc / 3.0 + dd;
            let sq = (qd * qd / 4.0 + pd * pd * pd / 27.0).sqrt();
            // Cardano branch by magnitude: the smaller branch is the
            // half-cancelled one whenever qd/2 ~ +-sq, not only at exact zero.
            let (cp, cm) = (-qd / 2.0 + sq, -qd / 2.0 - sq);
            let mut cst = if cp.norm() >= cm.norm() { cp } else { cm }.powf(1.0 / 3.0);
            let _ = &mut cst;
            let mut out = [zero; 4];
            if cst.norm() < 1e-30 {
                let r = -bb / 3.0;
                out[0] = r;
                out[1] = r;
                out[2] = r;
                return (out, 3);
            }
            let om = C::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
            for k in 0usize..3 {
                let ck = cst * om.powu(k as u32);
                out[k] = ck - pd / (ck * 3.0) - bb / 3.0;
            }
            (out, 3)
        }
        _ => ([zero; 4], 0), // unreachable from free_pair_roots (bd <= 4)
    }
}

/// The confluent residue law: v_k^2 from mu, cluster-aware. Per
/// delta-cluster of size s, the inheritance theorem puts (s-1) matched
/// copies of delta_k in mu; drop them (nearest match -- exact for pinned
/// data values), then the plain product formula. Cluster members beyond
/// the representative carry v = 0 (pass-through).
fn residues_v(
    mu: &[C; 4],
    delta: &[C; 4],
    dcl: &Clusters4,
    rho1: C,
    a: &[C; 4],
) -> Option<[f64; 4]> {
    let mut vv = [0.0f64; 4];
    for c in 0..dcl.k {
        let k = dcl.cluster_rep(c);
        let clen = dcl.cluster_len(c);
        let mut used = [false; 4];
        for _ in 1..clen {
            let mut mi = usize::MAX;
            let mut md = f64::INFINITY;
            for (i, &m) in mu.iter().enumerate() {
                if !used[i] {
                    let d = (m - delta[k]).norm();
                    if d < md {
                        md = d;
                        mi = i;
                    }
                }
            }
            used[mi] = true;
        }
        // A kept mu coinciding with the node makes this residue zero BY
        // THEOREM (the completion's pass-through, applied here too): the
        // numerical product would instead yield sqrt-of-noise dirt in a
        // component the zero-strata law says is exactly absent.
        if mu
            .iter()
            .enumerate()
            .any(|(i, &m)| !used[i] && (m - delta[k]).norm_sqr() < XTOL * XTOL)
        {
            vv[k] = 0.0;
            continue;
        }
        let mut num = C::new(1.0, 0.0);
        for (i, &m) in mu.iter().enumerate() {
            if !used[i] {
                num *= m - delta[k];
            }
        }
        let mut dden = C::new(1.0, 0.0);
        for l in 0..4 {
            if dcl.cluster_of[l] != c {
                dden *= delta[l] - delta[k];
            }
        }
        let val = num / (rho1 * dden * a[k]);
        if !val.re.is_finite() || !val.im.is_finite() || val.im.abs() > 1e-6 || val.re < -1e-7 {
            return None;
        }
        vv[k] = val.re.max(0.0).sqrt();
    }
    Some(vv)
}

/// Free-pair closure for the mu-pins (t1, t2): with both pins at data
/// values, q = gamma (z - t1)(z - t2) is forced, and eliminating gamma^2
/// between its two lift conditions at the free conjugate pair (m, pp/m),
/// pp = pin/(t1 t2), gives the squared closure
///   F(m) = pp^2 P(m) Qt(m)^2 - Pt(m) Q(m)^2,   deg 12,
/// with Pt, Qt the pp-reversals of P, Q. F is anti-self-inversive
/// (R[F] = -pp^6 F), so its 12 roots split exactly as: the four trivial
/// roots {t1, t2, pp/t1, pp/t2} (P vanishes at the pins; the mirror
/// symmetry supplies their images), the two collision points +-sqrt(pp)
/// (where the free pair degenerates to a double), and THREE mirror pairs
/// carried by a CUBIC in the Joukowski variable beta = m + pp/m:
/// deflate the trivial roots, split off (m^2 - pp), and the remaining
/// self-inversive sextic V satisfies
/// V/m^3 = v0(beta^3 - 3 pp beta) + v1(beta^2 - 2 pp) + v2 beta + v3. All radicals-grade: synthetic
/// divisions, one cubic rootfind, one quadratic per beta -- no
/// companion eigensolve. Every root is projected to the circle (genuine
/// mu is unitary; the ray-reality lemma makes circle-exact candidates
/// exactly testable and extraneous sign-class roots die at the gates).
#[derive(Clone, Copy)]
struct BetaClosure {
    coefficients: [C; 4],
    len: usize,
    double_root: Option<C>,
}

/// One-sided root exclusion for the regular beta factor on an admissible
/// mirror interval. With h^2=pp and beta=m+pp/m, unit-circle candidates have
/// beta=2 h x, x in [-1,1]. After this substitution the closure coefficients
/// share one phase and define a real polynomial of degree at most three.
/// Constant-sign Bernstein coefficients prove that it has no root on [lo,hi].
/// Marginal phase alignment or signs abstain, so this is only a rejection
/// certificate; the rooter remains the fallback.
fn beta_interval_variations(beta: BetaClosure, pp: C, lo: f64, hi: f64) -> Option<usize> {
    if !(lo < hi) || lo < -1.000_001 || hi > 1.000_001 {
        return None;
    }
    let degree = beta.len - 1;
    let mut h = pp.sqrt();
    let hn = h.norm();
    if hn < 1e-12 {
        return None;
    }
    h /= C::new(hn, 0.0);
    let two_h = h * 2.0;
    let mut transformed = [C::default(); 4]; // low power first
    let mut scale = 0.0f64;
    for (j, &bj) in beta.coefficients[..beta.len].iter().enumerate() {
        let power = degree - j;
        let co = bj * two_h.powu(power as u32);
        transformed[power] = co;
        scale = scale.max(co.norm());
    }
    if scale < 1e-24 || !scale.is_finite() {
        return None;
    }
    let anchor = transformed[..=degree]
        .iter()
        .copied()
        .max_by(|a, b| a.norm_sqr().total_cmp(&b.norm_sqr()))
        .unwrap();
    let phase = anchor.conj() / anchor.norm();
    let mut power = [0.0f64; 4];
    for k in 0..=degree {
        let z = transformed[k] * phase / scale;
        if !z.re.is_finite() || z.im.abs() > 1e-9 * (1.0 + z.re.abs()) {
            return None;
        }
        power[k] = z.re;
    }
    let eval = |x: f64| power[..=degree].iter().rev().fold(0.0, |v, &a| v * x + a);
    if eval(lo).abs() <= 1e-8 || eval(hi).abs() <= 1e-8 {
        return None;
    }
    Some(crate::cascade::bernstein_variations(
        &power[..=degree],
        lo,
        hi,
        1e-8,
    ))
}

#[inline]
fn free_pair_beta(
    p_poly: &[C],
    t1: C,
    t2: C,
    m1c: usize,
    m2c: usize,
    pp: C,
) -> Option<BetaClosure> {
    let zero = C::default();
    let pp2 = pp * pp;
    let (k1, k2) = (m1c.min(2), m2c.min(2));
    // The closure F = pp^2 P qt^2 - Pt q^2 carries its trivial roots (the
    // pins and their mirrors, at data multiplicity) in EACH TERM separately:
    // P and its reversal vanish at the pins/mirrors as exact data roots, and
    // the q/qt factors carry them symbolically.  Deflate PER TERM, exactly,
    // BEFORE the subtraction -- deflating the assembled difference instead
    // divides the subtraction's cancellation noise by near-zero factors and
    // was the margin band's closure floor.
    let deflate = |poly: &mut [C; 9], len: &mut usize, r: C| {
        // synthetic division by (m - r) at an exact root, high-first
        for i in 1..(*len - 1) {
            let prev = poly[i - 1];
            poly[i] += prev * r;
        }
        *len -= 1;
    };
    // P deflated at the pins; Pt (the pp-reversal) deflated at the mirrors.
    let mut pdef = [zero; 9];
    pdef.copy_from_slice(&p_poly[..9]);
    let mut plen = 9usize;
    for _ in 0..k1 {
        deflate(&mut pdef, &mut plen, t1);
    }
    for _ in 0..k2 {
        deflate(&mut pdef, &mut plen, t2);
    }
    let mut pt = [zero; 9];
    let mut ppi = C::new(1.0, 0.0);
    for (i, slot) in pt.iter_mut().enumerate() {
        *slot = p_poly[8 - i] * ppi;
        ppi *= pp;
    }
    let mut ptlen = 9usize;
    for _ in 0..k1 {
        deflate(&mut pt, &mut ptlen, pp / t1);
    }
    for _ in 0..k2 {
        deflate(&mut pt, &mut ptlen, pp / t2);
    }
    // remaining symbolic factors: q^2 keeps (m-t1)^{2-k1}(m-t2)^{2-k2};
    // qt^2 = t1^2 t2^2 (m-pp/t1)^2 (m-pp/t2)^2 keeps the mirror complement.
    let poly_from = |factors: &[(C, usize)], scale: C| -> ([C; 5], usize) {
        let mut out = [zero; 5];
        out[0] = scale;
        let mut n = 1usize;
        for &(r, count) in factors {
            for _ in 0..count {
                for i in (1..=n).rev() {
                    let prev = out[i - 1];
                    out[i] -= r * prev;
                }
                n += 1;
            }
        }
        (out, n)
    };
    let (q2d, q2len) = poly_from(&[(t1, 2 - k1), (t2, 2 - k2)], C::new(1.0, 0.0));
    let (qt2d, qt2len) = poly_from(&[(pp / t1, 2 - k1), (pp / t2, 2 - k2)], t1 * t1 * t2 * t2);
    // f = pp^2 * Pdef * qt2def - Ptdef * q2def  (both conditioned now)
    let mut f = [zero; 13];
    let len = plen + qt2len - 1;
    debug_assert_eq!(len, ptlen + q2len - 1);
    for i in 0..plen {
        for j in 0..qt2len {
            f[i + j] += pdef[i] * qt2d[j] * pp2;
        }
    }
    for i in 0..ptlen {
        for j in 0..q2len {
            f[i + j] -= pt[i] * q2d[j];
        }
    }
    // divide by m^2 - pp
    let vlen = len - 2;
    let mut v = [zero; 11];
    for i in 0..vlen {
        v[i] = f[i] + if i >= 2 { pp * v[i - 2] } else { zero };
    }
    // v is self-inversive of degree 2d (weight pp^d); the beta polynomial
    // (beta = m + pp/m) for the three possible degrees
    if vlen < 3 || vlen.is_multiple_of(2) {
        return None;
    }
    let d = (vlen - 1) / 2;
    let mut b = [zero; 4];
    let bd = match d {
        3 => {
            b[0] = v[0];
            b[1] = v[1];
            b[2] = v[2] - pp * v[0] * 3.0;
            b[3] = v[3] - pp * v[1] * 2.0;
            4
        }
        2 => {
            b[0] = v[0];
            b[1] = v[1];
            b[2] = v[2] - pp * v[0] * 2.0;
            3
        }
        1 => {
            b[0] = v[0];
            b[1] = v[1];
            2
        }
        _ => return None,
    };
    // rational double root of a cubic (companion/Cardano lose sqrt(eps))
    let bdd = if bd == 4 {
        let den = b[1] * b[1] - b[0] * b[2] * 3.0;
        if den.norm() > 1e-300 {
            Some((b[0] * b[3] * 9.0 - b[1] * b[2]) / (den * 2.0))
        } else {
            None
        }
    } else {
        None
    };
    Some(BetaClosure {
        coefficients: b,
        len: bd,
        double_root: bdd,
    })
}

#[inline]
fn lift_beta_roots(beta: BetaClosure, pp: C) -> ([C; 8], usize) {
    let mut out = [C::default(); 8];
    let mut n = 0;
    let (beta_arr, n_beta) = roots_small(&beta.coefficients[..beta.len]);
    for be in beta_arr[..n_beta].iter().copied().chain(beta.double_root) {
        // m^2 - beta m + pp = 0; one representative per mirror pair --
        // taking the NON-CANCELLING branch (the two roots are the mirror
        // pair m, pp/m, so either serves; be ~ -disc cancels the naive one).
        let disc = (be * be - pp * 4.0).sqrt();
        let (np_, nm_) = (be + disc, be - disc);
        let m = if np_.norm() >= nm_.norm() { np_ } else { nm_ } / 2.0;
        if m.norm() > 1e-3 {
            out[n] = m / m.norm();
            n += 1;
        }
    }
    // the collision points (free pair degenerates to a double)
    let sq = pp.sqrt();
    out[n] = sq;
    out[n + 1] = -sq;
    (out, n + 2)
}

#[inline]
pub fn free_pair_roots(
    p_poly: &[C],
    t1: C,
    t2: C,
    m1c: usize,
    m2c: usize,
    pp: C,
) -> ([C; 8], usize) {
    free_pair_beta(p_poly, t1, t2, m1c, m2c, pp)
        .map(|beta| lift_beta_roots(beta, pp))
        .unwrap_or(([C::default(); 8], 0))
}

/// Scale times the monic polynomial with the given roots, highest-first.
fn poly_from_roots(scale: C, rs: &[C]) -> [C; 9] {
    let mut p = [C::default(); 9];
    p[0] = scale;
    for (k, &r) in rs.iter().enumerate() {
        for i in (1..=k + 1).rev() {
            let prev = p[i - 1];
            p[i] -= r * prev;
        }
    }
    p
}

/// The pairing quadratic q of a candidate, in the two forms the rungs
/// produce it: the corner root form gamma (z - t1)(z - t2) (pins stay
/// exact under the R-removal) or a general coefficient quadratic (the
/// sign-class fiber; declines confluent data).
enum Ghat {
    Corner { g2: C, t1: C, t2: C },
}

/// The V-side admissibility of a pin-pair candidate, exactly. For unit-
/// modulus data and pins, each first-peel residue is LINEAR in the
/// mirror-quotient coordinate x = cos(psi - arg(pp)/2) of the free root
/// m = e^{i psi}: with theta_i = arg(delta_i),
///   V_i(x) = kappa_i (x - c_i) / 2,   c_i = cos(theta_i - arg(pp)/2),
/// kappa_i a real per-pair constant (the det-pin phase telescope). The
/// admissible set is therefore ONE x-interval; a pair whose interval is
/// empty has no on-circle candidate, and off-circle candidates fail the
/// residue reality gate. Margins are four decades wider than the exact
/// gate's thresholds, and every uncertain case falls through to the
/// exact evaluation, so this changes which candidates are *examined*,
/// never which are accepted.
struct PairGate {
    half_product: C,
    xlo: f64,
    xhi: f64,
    /// conj(W^2)/|W|^2 at a calibration node, W = d (d-t1)(d-t2)/(a C):
    /// all nodes share arg(W) (det-pin telescope with the z-origin root),
    /// so U_i = zeta r_i^2 / V_i with zeta = g2 * ray. For pairs with no
    /// pin at a data node the production reality dichotomy is EXACTLY
    /// "|Im zeta| > |Re zeta|", and all-negative U is "Re zeta < 0".
    ray: Option<C>,
    /// Telescope violated (kappa not real): every on-circle candidate of the
    /// pair fails residue reality identically -- the pair is dead.
    dead: bool,
    /// Magnitude-law data: U-sum = zeta*S(x), S = sum 2 r_i^2/(kappa_i (x-c_i)).
    kap: [f64; 4],
    cc: [f64; 4],
    r2: [f64; 4],
    mag_ok: bool,
}

impl PairGate {
    #[allow(clippy::too_many_arguments)]
    fn new(t1: C, t2: C, pp: C, delta: &[C; 4], ac: &[C; 4], rho1: C) -> Option<PairGate> {
        let mut m1 = pp.sqrt();
        let m1_norm = m1.norm();
        if m1_norm < 1e-12 {
            return None;
        }
        m1 /= C::new(m1_norm, 0.0);
        // m1=sqrt(pp), m2=-m1: the two endpoints of the mirror quotient.
        // For unit-circle m: pp/m = pp*conj(m), eliminating 4 complex divisions in vex.
        let m2 = -m1;
        let pp_m1 = pp * m1.conj(); // pp / m1 (unit-circle: 1/m = conj(m))
        let pp_m2 = -pp_m1; // pp / m2 = pp / (-m1) = -(pp / m1)
                            // dt12[k] = (delta[k]-t1)*(delta[k]-t2): pair-specific, computed here.
                            // ac[k] = a[k]*cis[k] is loop-invariant across all pairs; precomputed by caller.
        let dt12: [C; 4] = std::array::from_fn(|k| (delta[k] - t1) * (delta[k] - t2));
        // exact residue of the candidate pair (m, pp/m) at node k
        let vex = |m: C, pp_m: C, k: usize| -> C {
            let d = delta[k];
            -(dt12[k] * (d - m) * (d - pp_m)) / (rho1 * ac[k])
        };
        let (mut xlo, mut xhi) = (-1.0f64, 1.0f64);
        // U-ray calibration: any node clear of the pins works (shared phase)
        let mut ray = None;
        let mut best_w = 1e-6;
        let mut r2 = [0.0f64; 4];
        for k in 0..4 {
            let d = delta[k];
            let w = d * dt12[k] / ac[k];
            let n2 = w.norm_sqr();
            r2[k] = n2;
            if n2 > best_w {
                best_w = n2;
                let w2 = w * w;
                // |w2| = |w|^2 = n2, so w2/|w2| = w2/n2 (avoids hypot)
                ray = Some(w2 / n2);
            }
        }
        // a pin AT a data node breaks the pure zeta form of the dichotomy
        // use norm_sqr > 1e-18 to avoid 8 sqrt calls (equivalent to norm > 1e-9)
        let unpinned = delta
            .iter()
            .all(|&d| (d - t1).norm_sqr() > 1e-18 && (d - t2).norm_sqr() > 1e-18);
        if !unpinned {
            ray = None;
        }
        let mut kaps = [0.0f64; 4];
        let mut ccs = [0.0f64; 4];
        let mut mag_ok = ray.is_some();
        let mut dead = false;
        for k in 0..4 {
            let ci = (delta[k] * m1.conj()).re;
            // anchor kappa at whichever probe is far from this node's zero
            let (v, xs) = if (1.0 - ci).abs() > 0.5 {
                (vex(m1, pp_m1, k), 1.0)
            } else {
                (vex(m2, pp_m2, k), -1.0)
            };
            let kap = v * C::new(2.0 / (xs - ci), 0.0);
            if !kap.re.is_finite() {
                return None;
            }
            if kap.im.abs() > 1e-2 * (1.0 + kap.re.abs()) {
                // strongly non-real kappa: reality fails on the whole pair
                dead = true;
            } else if kap.im.abs() > 1e-9 * (1.0 + kap.re.abs()) {
                return None; // marginal telescope: abstain entirely
            }
            let k_re = kap.re;
            kaps[k] = k_re;
            ccs[k] = ci;
            if k_re.abs() < 1e-9 {
                mag_ok = false; // pin-like node: S incomplete, no magnitude gate
            }
            let margin = 1e-3 * (1.0 + k_re.abs()) / k_re.abs().max(1e-30);
            if k_re > 1e-9 {
                xlo = xlo.max(ci - margin);
            } else if k_re < -1e-9 {
                xhi = xhi.min(ci + margin);
            }
            // |kappa| ~ 0: the node never certifies a rejection
        }
        Some(PairGate {
            half_product: m1,
            xlo,
            xhi,
            dead,
            ray,
            kap: kaps,
            cc: ccs,
            r2,
            mag_ok,
        })
    }

    fn hopeless(&self) -> bool {
        self.dead || self.xlo > self.xhi
    }

    /// True only when the exact V-gate provably rejects this candidate.
    /// Near-circle roots (the float splitting of anti-self-inversive
    /// mirror pairs, radial ~1e-5) are gated by their PROJECTED angle:
    /// the laws depend only on arg(m) and the 1e-3 margins dominate the
    /// radial perturbation by two orders.
    fn rejects(&self, m: C) -> bool {
        if (m.norm_sqr() - 1.0).abs() > 1e-4 {
            return false; // far off-circle: leave to the exact reality gate
        }
        let x = (m * self.half_product.conj()).re / m.norm();
        x < self.xlo || x > self.xhi
    }

    /// The regular beta roots plus the four explicitly appended boundary/pin
    /// candidates all miss the admissible residue interval.
    fn excludes_closure(&self, beta: BetaClosure, pp: C, t1: C, t2: C) -> bool {
        self.rejects(self.half_product)
            && self.rejects(-self.half_product)
            && self.rejects(t1)
            && self.rejects(t2)
            && beta_interval_variations(beta, pp, self.xlo, self.xhi) == Some(0)
    }
}

/// Reduced interval gate (confluent three laws, leading (2,1,1)
/// case): with the cluster copy dropped, muhat = {t_surv, m, pf/m} and
/// every rep residue is linear in x = cos(arg m - arg(pf)/2). Same
/// margins and fall-through discipline as PairGate. The U-side clauses
/// descend too (laws 2-3): with ghat = g/R, all reduced What_rep
/// share one phase, so U-reality is zetahat = gamma^2 * rayhat real
/// positive, and the det-pin sum is zetahat*Shat(x) + That(x) = 1 with
/// the ONE cluster term That = K/(alpha + beta x) rational deg-(0/1)
/// (Phat(L) = chi_w(L) prod_other != 0 at the cluster rep; the term is
/// exactly the cluster block's Cauchy-Schwarz slack).
struct PairGateR {
    half_product: C,
    xlo: f64,
    xhi: f64,
    nreps: usize,
    reps: [usize; 4],
    ray: Option<C>,
    kap: [f64; 4],
    cc: [f64; 4],
    r2: [f64; 4],
    /// cluster terms That_i(x) = tk_i / (ta_i + tb_i*x), one per 2-cluster
    /// ((2,1,1): one; (2,2): two)
    ncl: usize,
    tk: [C; 2],
    ta: [C; 2],
    tb: [C; 2],
    mag_ok: bool,
}

impl PairGateR {
    /// `pins` = the surviving (non-letter) pins of the pair after the
    /// inheritance copies are dropped: one for the (2,1,1) leading case,
    /// none for the (2,2) both-letters case. `cl2` = the 2-cluster rep
    /// indices (the letters).
    #[allow(clippy::too_many_arguments)]
    fn new(
        pins: &[C],
        pf: C,
        reps: &[usize],
        delta: &[C; 4],
        cis: &[C; 4],
        rho1: C,
        rho2: C,
        a: &[C; 4],
        w: &[C; 4],
        cl2: &[usize],
    ) -> Option<PairGateR> {
        let mut mhalf = pf.sqrt();
        let half_norm = mhalf.norm();
        if half_norm < 1e-12 {
            return None;
        }
        mhalf /= C::new(half_norm, 0.0);
        let pinprod = |d: C| pins.iter().fold(C::new(1.0, 0.0), |acc, &p| acc * (d - p));
        let vex = |m: C, k: usize| -> C {
            let d = delta[k];
            -(pinprod(d) * (d - m) * (d - pf / m)) / (rho1 * a[k] * cis[k])
        };
        let (m1, m2) = (mhalf, -mhalf);
        let (mut xlo, mut xhi) = (-1.0f64, 1.0f64);
        // reduced U-ray calibration: What = d * prod(d - pins) / (a C-hat);
        // a surviving pin AT a rep node breaks the pure zetahat form
        let mut ray = None;
        let mut best_w = 1e-6;
        let mut r2 = [0.0f64; 4];
        let unpinned = reps
            .iter()
            .all(|&k| pins.iter().all(|&p| (delta[k] - p).norm() > 1e-9));
        for &k in reps {
            let d = delta[k];
            let wq = d * pinprod(d) / (a[k] * cis[k]);
            let n2 = wq.norm_sqr();
            r2[k] = n2;
            if unpinned && n2 > best_w {
                best_w = n2;
                let w2 = wq * wq;
                ray = Some(w2 / w2.norm());
            }
        }
        let mut kaps = [0.0f64; 4];
        let mut ccs = [0.0f64; 4];
        let mut mag_ok = ray.is_some();
        for &k in reps {
            let ci = (delta[k] * mhalf.conj()).re;
            let (v, xs) = if (1.0 - ci).abs() > 0.5 {
                (vex(m1, k), 1.0)
            } else {
                (vex(m2, k), -1.0)
            };
            let kap = v * C::new(2.0 / (xs - ci), 0.0);
            if !kap.re.is_finite() || kap.im.abs() > 1e-9 * (1.0 + kap.re.abs()) {
                return None; // reduced telescope violated: abstain
            }
            let k_re = kap.re;
            kaps[k] = k_re;
            ccs[k] = ci;
            if k_re.abs() < 1e-9 {
                mag_ok = false; // pin-like node: S incomplete
            }
            let margin = 1e-3 * (1.0 + k_re.abs()) / k_re.abs().max(1e-30);
            if k_re > 1e-9 {
                xlo = xlo.max(ci - margin);
            } else if k_re < -1e-9 {
                xhi = xhi.min(ci + margin);
            }
        }
        // cluster term constants per 2-cluster rep L: Phat(L) = chi_w(L) *
        // prod_{SIMPLE reps}(L - d_k) (the R^2-divided identity: other
        // 2-clusters contribute exponent s-2 = 0), and Bhat(L) =
        // prod(L - pins) * (alpha + beta x).
        let mut g = PairGateR {
            half_product: mhalf,
            xlo,
            xhi,
            nreps: 0,
            reps: [0; 4],
            ray,
            kap: kaps,
            cc: ccs,
            r2,
            ncl: 0,
            tk: [C::default(); 2],
            ta: [C::default(); 2],
            tb: [C::default(); 2],
            mag_ok,
        };
        for &l_idx in cl2.iter().take(2) {
            let l = delta[l_idx];
            let chi_w_l = w.iter().fold(C::new(1.0, 0.0), |acc, &wj| acc * (l - wj));
            let prod_simple = reps
                .iter()
                .filter(|&&k| k != l_idx && !cl2.contains(&k))
                .fold(C::new(1.0, 0.0), |acc, &k| acc * (l - delta[k]));
            let denom = rho2 * a[l_idx] * cis[l_idx] * pinprod(l);
            if denom.norm() < 1e-12 {
                g.mag_ok = false;
                continue;
            }
            g.tk[g.ncl] = -chi_w_l * prod_simple / denom;
            g.ta[g.ncl] = l * l + pf;
            g.tb[g.ncl] = -2.0 * l * mhalf;
            g.ncl += 1;
        }
        g.nreps = reps.len().min(4);
        g.reps[..g.nreps].copy_from_slice(&reps[..g.nreps]);
        Some(g)
    }

    fn rejects(&self, m: C) -> bool {
        if (m.norm_sqr() - 1.0).abs() > 1e-4 {
            return false;
        }
        let x = (m * self.half_product.conj()).re / m.norm();
        x < self.xlo || x > self.xhi
    }

    /// Reduced ray + magnitude clauses (candidate-level): same factor-2
    /// ray dichotomy and 1e-2 det-pin margin as the simple gate, with
    /// the cluster's rational term added to the magnitude sum.
    fn rejects_u(&self, m: C, pf: C, g2v: C, delta: &[C; 4]) -> bool {
        let Some(ray) = self.ray else { return false };
        // validity domain: free pair clear of the rep nodes (else the
        // deflated Bhat* is not Nhat/Bhat and zetahat does not factor)
        let m2f = pf / m;
        let clear = self.reps[..self.nreps]
            .iter()
            .all(|&k| (m - delta[k]).norm() > 1e-4 && (m2f - delta[k]).norm() > 1e-4);
        if !clear {
            return false;
        }
        let z = g2v * ray;
        if z.re <= 0.0 || z.im.abs() > 2.0 * z.re {
            return true;
        }
        if self.mag_ok && (m.norm_sqr() - 1.0).abs() < 1e-6 {
            let x = (m * self.half_product.conj()).re / m.norm();
            let s_of_x: f64 = self.reps[..self.nreps]
                .iter()
                .map(|&k| 2.0 * self.r2[k] / (self.kap[k] * (x - self.cc[k])))
                .sum();
            let mut t_re = 0.0;
            for i in 0..self.ncl {
                let bl = self.ta[i] + self.tb[i] * x;
                if bl.norm() < 1e-6 {
                    return false; // near-confluent cluster term: abstain
                }
                t_re += (self.tk[i] / bl).re;
            }
            return (z.re * s_of_x + t_re - 1.0).abs() > 1e-2;
        }
        false
    }
}

/// Candidate-independent data of one two-step problem: the distinct-node
/// Vandermonde factors C_i and the P-hat root multiset. Depends only on
/// (delta, dcl, w), so it is built once per problem, not once per
/// candidate -- the same common-factor extraction MirrorBase performs
/// across sign-class lifts, applied one level up.
struct ProblemBase {
    cis: [C; 4],
    ph: [C; 8],
    np: usize,
}

/// Given a monic degree-2D polynomial N and the D roots of a monic factor B,
/// recover the monic complementary factor Q from N=BQ. Coefficient k of Q
/// depends only on coefficients 0..k, so no root ordering or polynomial
/// rootfinder enters the quotient.
fn monic_complement(n: &[C], roots: &[C]) -> Option<([C; 9], usize)> {
    let d = roots.len();
    if n.len() != 2 * d + 1 || d > 8 {
        return None;
    }
    let b = poly_from_roots(C::new(1.0, 0.0), roots);
    let mut q = [C::default(); 9];
    q[0] = n[0];
    for k in 1..=d {
        q[k] = n[k];
        for j in 1..=k {
            q[k] -= b[j] * q[k - j];
        }
    }
    Some((q, d + 1))
}

fn problem_base(delta: &[C; 4], dcl: &Clusters4, w: &[C; 4]) -> ProblemBase {
    let mut cis = [C::default(); 4];
    for c in 0..dcl.k {
        let k0 = dcl.cluster_rep(c);
        let d = delta[k0];
        let mut ci = C::new(1.0, 0.0);
        for c2 in 0..dcl.k {
            if c2 != c {
                ci *= d - delta[dcl.cluster_rep(c2)];
            }
        }
        cis[k0] = ci;
    }
    // P-hat roots: the simple data values plus w minus its s-2
    // inheritance copies (2 nd values in total).
    let mut ph = [C::default(); 8];
    let mut np = 0;
    let mut wused = [false; 4];
    for c in 0..dcl.k {
        let clen = dcl.cluster_len(c);
        let k0 = dcl.cluster_rep(c);
        if clen == 1 {
            ph[np] = delta[k0];
            np += 1;
        }
        for _ in 2..clen {
            let (mut wi, mut wd) = (usize::MAX, f64::INFINITY);
            for (j, &wj) in w.iter().enumerate() {
                if !wused[j] {
                    let dd = (wj - delta[k0]).norm();
                    if dd < wd {
                        wd = dd;
                        wi = j;
                    }
                }
            }
            wused[wi] = true;
        }
    }
    for (j, &wj) in w.iter().enumerate() {
        if !wused[j] {
            ph[np] = wj;
            np += 1;
        }
    }
    ProblemBase { cis, ph, np }
}

/// Roots of a small monic complex polynomial (degree <= 4), coefficients
/// highest-first, by closed-form radicals (quadratic / Cardano / Ferrari).
/// Bounded and deterministic -- the mirror law's mu' solve: N-hat = B B*
/// and mu' = roots(B*) is the swapped-peel-order intermediate spectrum,
/// on the unit circle by theorem, so the solve is well-conditioned there.
fn complex_monic_roots_small(p: &[C], out: &mut [C; 8]) -> Option<usize> {
    let deg = p.len().checked_sub(1)?;
    let quad = |b: C, c0: C, o: &mut [C]| {
        let s = (b * b - c0 * 4.0).sqrt();
        let q = if (b + s).norm_sqr() >= (b - s).norm_sqr() {
            -(b + s) * 0.5
        } else {
            -(b - s) * 0.5
        };
        o[0] = q;
        o[1] = if q.norm_sqr() > 0.0 { c0 / q } else { -b - q };
    };
    let cbrt = |z: C| -> C {
        if z.norm_sqr() == 0.0 {
            C::default()
        } else {
            C::from_polar(z.norm().cbrt(), z.arg() / 3.0)
        }
    };
    let cardano = |a: C, b: C, c0: C, o: &mut [C]| {
        // monic z^3 + a z^2 + b z + c0, depressed t = z + a/3
        let p1 = b - a * a / 3.0;
        let q1 = a * a * a * (2.0 / 27.0) - a * b / 3.0 + c0;
        let dsc = (q1 * 0.5) * (q1 * 0.5) + (p1 / 3.0) * (p1 / 3.0) * (p1 / 3.0);
        let s = dsc.sqrt();
        let e1 = -q1 * 0.5 + s;
        let e2 = -q1 * 0.5 - s;
        let u = cbrt(if e1.norm_sqr() >= e2.norm_sqr() {
            e1
        } else {
            e2
        });
        let w1 = C::new(-0.5, 0.75f64.sqrt());
        if u.norm_sqr() < 1e-300 {
            let t = cbrt(-q1);
            for (k, ok) in o.iter_mut().take(3).enumerate() {
                *ok = t * if k == 0 {
                    C::new(1.0, 0.0)
                } else if k == 1 {
                    w1
                } else {
                    w1 * w1
                } - a / 3.0;
            }
            return;
        }
        let v = -p1 / (u * 3.0);
        let mut rot = C::new(1.0, 0.0);
        for ok in o.iter_mut().take(3) {
            *ok = u * rot + v * rot.conj() - a / 3.0;
            rot *= w1;
        }
    };
    match deg {
        0 => Some(0),
        1 => {
            out[0] = -p[1];
            Some(1)
        }
        2 => {
            quad(p[1], p[2], &mut out[..2]);
            Some(2)
        }
        3 => {
            cardano(p[1], p[2], p[3], &mut out[..3]);
            Some(3)
        }
        4 => {
            let (a, b, c0, d0) = (p[1], p[2], p[3], p[4]);
            // depress: z = y - a/4
            let p1 = b - a * a * (3.0 / 8.0);
            let q1 = c0 - a * b * 0.5 + a * a * a * 0.125;
            let r1 = d0 - a * c0 * 0.25 + a * a * b / 16.0 - a * a * a * a * (3.0 / 256.0);
            let shift = -a * 0.25;
            let scale = p1.norm().max(q1.norm()).max(r1.norm()).max(1.0);
            if q1.norm() < 1e-14 * scale {
                // biquadratic: y^2 solves t^2 + p1 t + r1
                let mut t2 = [C::default(); 2];
                quad(p1, r1, &mut t2);
                for (k, &t) in t2.iter().enumerate() {
                    let s = t.sqrt();
                    out[2 * k] = s + shift;
                    out[2 * k + 1] = -s + shift;
                }
                return Some(4);
            }
            // resolvent 8m^3 + 8 p1 m^2 + (2 p1^2 - 8 r1) m - q1^2 = 0
            let mut mr = [C::default(); 3];
            cardano(p1, p1 * p1 * 0.25 - r1, -q1 * q1 * 0.125, &mut mr);
            let m = *mr
                .iter()
                .max_by(|x, y| x.norm_sqr().partial_cmp(&y.norm_sqr()).unwrap())
                .unwrap();
            let s = (m * 2.0).sqrt();
            if s.norm_sqr() < 1e-300 {
                return None;
            }
            let t = (p1 + m * 2.0 - q1 / s) * 0.5;
            let u = (p1 + m * 2.0 + q1 / s) * 0.5;
            let mut o1 = [C::default(); 2];
            let mut o2 = [C::default(); 2];
            quad(s, t, &mut o1);
            quad(-s, u, &mut o2);
            out[0] = o1[0] + shift;
            out[1] = o1[1] + shift;
            out[2] = o2[0] + shift;
            out[3] = o2[1] + shift;
            Some(4)
        }
        _ => None,
    }
}

/// Quantities fixed by `mu` across every sign-class lift of the same
/// characteristic.  Building this once is the algebraic analogue of
/// deflating the common factor before evaluating its different cofactors.
struct MirrorBase {
    muhat: [C; 4],
    nm: usize,
    vv: [f64; 4],
}

fn mirror_base(
    mu: &[C; 4],
    delta: &[C; 4],
    dcl: &Clusters4,
    pre_rho1_ac: &[C; 4],
) -> Option<MirrorBase> {
    // mu-hat: drop the s-1 inheritance copies per cluster (nearest
    // match, the same matching residues_v performs on the v side)
    let mut used = [false; 4];
    for c in 0..dcl.k {
        let clen = dcl.cluster_len(c);
        let k0 = dcl.cluster_rep(c);
        for _ in 1..clen {
            let (mut mi, mut md) = (usize::MAX, f64::INFINITY);
            for (i, &m) in mu.iter().enumerate() {
                if !used[i] {
                    let dd = (m - delta[k0]).norm();
                    if dd < md {
                        md = dd;
                        mi = i;
                    }
                }
            }
            used[mi] = true;
        }
    }
    let mut muhat = [C::default(); 4];
    let mut nm = 0;
    for (i, &m) in mu.iter().enumerate() {
        if !used[i] {
            muhat[nm] = m;
            nm += 1;
        }
    }

    // v-side of the same law: V_i = -B(d_i)/(rho1 a_i C_i) per block,
    // gated by reality, positivity, and the det-pin norm sum V = 1.
    let mut vv = [0.0f64; 4];
    let mut nv = 0.0f64;
    for c in 0..dcl.k {
        let k0 = dcl.cluster_rep(c);
        let d = delta[k0];
        // Same exact-zero pass-through as residues_v / the completion: a kept
        // mu-hat at the node makes V here zero by theorem.
        if muhat[..nm]
            .iter()
            .any(|&m| (m - d).norm_sqr() < XTOL * XTOL)
        {
            vv[k0] = 0.0;
            continue;
        }
        let mut bd = C::new(1.0, 0.0);
        for &m in &muhat[..nm] {
            bd *= d - m;
        }
        let vi = -bd / pre_rho1_ac[k0];
        if !vi.re.is_finite() || !vi.im.is_finite() || vi.im.abs() > 1e-6 {
            crate::cascade::prof::hit(crate::cascade::prof::RJ_V_IMAG);
            return None;
        }
        if vi.re < -1e-7 {
            crate::cascade::prof::hit(crate::cascade::prof::RJ_V_NEG);
            return None;
        }
        vv[k0] = vi.re.max(0.0).sqrt();
        nv += vi.re.max(0.0);
    }
    if (nv - 1.0).abs() > 1e-4 {
        crate::cascade::prof::hit(crate::cascade::prof::RJ_V_SUM);
        return None;
    }

    Some(MirrorBase { muhat, nm, vv })
}

/// THE REDUCED MIRROR LAW (the general confluent evaluation). All
/// coincidence handling collapses at the reduced level: strip the
/// inheritance copies (a delta-cluster of size s puts s-1 matched
/// copies in mu and s-2 in w -- exact multiset removals in root form),
/// and the SIMPLE-NODE law holds verbatim at the D distinct nodes,
/// where per node the block invariants are
///   V_i = -B(d_i)  / (rho1 a_i C_i),   B  = prod_l (x - mu-hat_l),
///   U_i = -B*(d_i) / (rho2 a_i C_i),   B* = (P-hat + rho1 rho2 g-hat^2)/B,
///   X_i = g-hat(d_i) / (a_i C_i),      C_i = prod_{j != i} (d_i - d_j),
/// with P-hat = chi_w chi-hat_delta / R and g-hat = g / R (R | g is a
/// theorem), and B* by the monic coefficient quotient N-hat/B: a pin is
/// an ordinary polynomial root, never a 0/0, so the old arm list (generic ratio,
/// P'-pin, structural zero, cluster-g', level-2 member) is this one
/// law at particular coincidence patterns. X_i is SIGNED -- no square
/// root -- so per-block relative signs are determined, and coordinate
/// flips where v_k = 0 are diag(+-1) conjugation gauge: no sign
/// enumeration exists. Simple nodes satisfy X_i^2 = V_i U_i exactly
/// (P-hat(d_i) = 0); cluster blocks reconstruct in the canonical gauge
/// v || e1, u in span(e1, e2), the e2 magnitude being the
/// Cauchy-Schwarz slack -P-hat(d_i)/(rho2 a_i C_i B(d_i)).
#[allow(clippy::too_many_arguments)]
fn mirror_completion(
    base: &MirrorBase,
    pb: &ProblemBase,
    gh: Ghat,
    c: C,
    rho1: C,
    rho2: C,
    a: &[C; 4],
    delta: &[C; 4],
    dcl: &Clusters4,
    w: &[C; 4],
    pre_rho2_ac: &[C; 4],
    pre_ac: &[C; 4],
    _method: &'static str,
) -> Option<Solved> {
    let (muhat, nm, vv) = (&base.muhat, base.nm, base.vv);
    let (ph, np) = (&pb.ph, pb.np);
    // g-hat = g / R: the corner form removes the R-copies of its pair
    // roots exactly (R | g is a theorem: a candidate whose pair roots
    // miss a repeated data value has no completion); the fiber's
    // coefficient form declines confluent data (its interpolation
    // assumes separated nodes)
    let mut gam = C::default();
    let mut gr = [C::default(); 3];
    let mut ng = 0;
    let mut gcoef = [C::default(); 9];
    let mut ghlen = 0;
    let mut skeleton = false;
    match gh {
        Ghat::Corner { g2, t1, t2 } => {
            if g2 == C::default() {
                skeleton = true;
            } else {
                gam = g2.sqrt();
                ng = 1; // the root at z = 0 (v-u orthogonality)
                let ts = [t1, t2];
                let mut tused = [false; 2];
                for c_ in 0..dcl.k {
                    let clen = dcl.cluster_len(c_);
                    let k0 = dcl.cluster_rep(c_);
                    for _ in 1..clen {
                        let (mut ti, mut td) = (usize::MAX, f64::INFINITY);
                        for (j, &t) in ts.iter().enumerate() {
                            if !tused[j] {
                                let dd = (t - delta[k0]).norm();
                                if dd < td {
                                    td = dd;
                                    ti = j;
                                }
                            }
                        }
                        if ti == usize::MAX || td > XTOL {
                            return None;
                        }
                        tused[ti] = true;
                    }
                }
                for (j, &t) in ts.iter().enumerate() {
                    if !tused[j] {
                        gr[ng] = t;
                        ng += 1;
                    }
                }
                gcoef = poly_from_roots(gam, &gr[..ng]);
                ghlen = ng + 1;
            }
        }
    }
    // THE SYMMETRIC LAW: both factors of N-hat = B B* in root form.
    // mu' = roots(B*) is the swapped-peel-order intermediate spectrum
    // (unit circle by theorem); one bounded radical solve per candidate.
    // Every node value below is then a product of gaps: a pin is an
    // ordinary zero factor on either side, and the exact-zero
    // pass-through is one uniform on-circle gap test -- the former
    // coefficient-Horner arms (k-fold derivative quotient, per-branch
    // zero detections) were cancellation patches for the asymmetric
    // representation and are gone.
    let mut mup = [C::default(); 8];
    let mut nmup = 0;
    if skeleton {
        let mut phused = [false; 8];
        for l in 0..nm {
            let (mut pi, mut pd) = (usize::MAX, f64::INFINITY);
            for j in 0..np {
                if !phused[j] {
                    let dd = (ph[j] - muhat[l]).norm();
                    if dd < pd {
                        pd = dd;
                        pi = j;
                    }
                }
            }
            if pi == usize::MAX || pd > XTOL {
                crate::cascade::prof::hit(crate::cascade::prof::RJ_SKEL);
                return None; // gamma = 0 forces mu into the data multiset
            }
            phused[pi] = true;
        }
        for j in 0..np {
            if !phused[j] {
                mup[nmup] = ph[j];
                nmup += 1;
            }
        }
    } else {
        let mut nhat = poly_from_roots(C::new(1.0, 0.0), &ph[..np]);
        let plen = np + 1;
        let rr = rho1 * rho2;
        for i in 0..ghlen {
            for j in 0..ghlen {
                nhat[plen - 2 * ghlen + 1 + i + j] += rr * gcoef[i] * gcoef[j];
            }
        }
        let (bstar, bslen) = monic_complement(&nhat[..plen], &muhat[..nm])?;
        nmup = complex_monic_roots_small(&bstar[..bslen], &mut mup)?;
    }
    // the block invariants at each distinct node
    let mut u2 = [C::default(); 4];
    let mut xr = [0.0f64; 4];
    for c_ in 0..dcl.k {
        let k0 = dcl.cluster_rep(c_);
        let d = delta[k0];
        // root form: pins are exact zeros, never sqrt-of-noise (ghlen > 0
        // implies ng > 0, so no coefficient fallback exists here)
        let gd = if ng > 0 {
            let mut g = gam;
            for &r in &gr[..ng] {
                g *= d - r;
            }
            g
        } else {
            C::default()
        };
        let pinned = muhat[..nm]
            .iter()
            .any(|&m| (m - d).norm_sqr() < XTOL * XTOL);
        let wmatch = w.iter().any(|&wj| (wj - d).norm_sqr() < XTOL * XTOL);
        let ui = if pinned && wmatch {
            C::default() // pass-through one level up: zero by theorem (exact data test)
        } else {
            // U_i = -prod(d - mu') / (rho2 a_i C_i). An exact B*-root at
            // the node -- any provenance: complementary collision, w-side
            // coincidence, deep pass-through -- is the ONE uniform on-circle
            // gap test, so u is zero by theorem instead of sqrt-of-noise.
            let mut acc = C::new(1.0, 0.0);
            let mut zero = false;
            for &r in &mup[..nmup] {
                let gap = d - r;
                if gap.norm_sqr() < XTOL * XTOL {
                    zero = true;
                    break;
                }
                acc *= gap;
            }
            if zero {
                C::default()
            } else {
                -acc / pre_rho2_ac[k0]
            }
        };
        let xi = gd / pre_ac[k0];
        if !ui.re.is_finite() || !ui.im.is_finite() || !xi.re.is_finite() {
            return None;
        }
        u2[k0] = ui;
        xr[k0] = xi.re;
    }
    // the exact reality dichotomy, positivity, and the mu'-side det-pin
    let sre: f64 = u2.iter().map(|z| z.re * z.re).sum();
    let simm: f64 = u2.iter().map(|z| z.im * z.im).sum();
    if simm > sre || u2.iter().any(|z| z.re < 0.0 && z.re * z.re > simm) {
        crate::cascade::prof::hit(crate::cascade::prof::RJ_UREAL);
        return None;
    }
    let s2: f64 = (0..dcl.k)
        .map(|c_| u2[dcl.cluster_rep(c_)].re.max(0.0))
        .sum();
    if (s2 - 1.0).abs() >= 1e-4 {
        crate::cascade::prof::hit(crate::cascade::prof::RJ_USUM);
        return None;
    }
    // canonical-gauge reconstruction: one u, one verify, no enumeration
    let mut uc = [0.0f64; 4];
    for c_ in 0..dcl.k {
        let k0 = dcl.cluster_rep(c_);
        let clen = dcl.cluster_len(c_);
        let ui = u2[k0].re.max(0.0);
        if vv[k0] <= 0.0 {
            uc[k0] = ui.sqrt();
        } else if clen == 1 {
            // sign from the chord product, magnitude from U (X^2 = VU)
            uc[k0] = xr[k0].signum() * ui.sqrt();
        } else {
            uc[k0] = xr[k0] / vv[k0];
            let head = uc[k0] * uc[k0];
            let rem = ui - head;
            // Full cancellation means the cluster's second component is
            // exactly zero (the same zero-strata pass-through as the node
            // formulas); sqrt of the noise difference would inject ~1e-8
            // dirt into an exactly-absent component.
            uc[dcl.cluster_second(c_)] = if rem.abs() < XTOL * (ui.abs() + head) {
                0.0
            } else {
                rem.max(0.0).sqrt()
            };
        }
    }
    let out = verify(c, a, &vv, &uc, rho1, rho2, w);
    if out.is_some() {
        crate::cascade::prof::hit(crate::cascade::prof::N_ACCEPT);
    }
    out
}

/// One candidate: mu (+ gamma^2 and the pair roots) -> v by residues ->
/// u by the reduced mirror law -> det gate -> one eigensolve; then the
/// sign-class fiber over mu (kernel-selector replacement pending).
#[allow(clippy::too_many_arguments)]
fn try_mu(
    mu: &[C; 4],
    gq: Option<(C, C, C)>,
    c: C,
    rho1: C,
    rho2: C,
    a: &[C; 4],
    delta: &[C; 4],
    dcl: &Clusters4,
    pb: &ProblemBase,
    _p_poly: &[C],
    w: &[C; 4],
    pre_rho1_ac: &[C; 4],
    pre_rho2_ac: &[C; 4],
    pre_ac: &[C; 4],
    method: &'static str,
) -> Option<Solved> {
    let (g2, t1, t2) = gq?;
    crate::cascade::prof::hit(crate::cascade::prof::N_TRY_MU);
    let tb = crate::cascade::prof::start();
    let mb0 = mirror_base(mu, delta, dcl, pre_rho1_ac);
    crate::cascade::prof::rec(crate::cascade::prof::SW_BASE, tb);
    let Some(mb) = mb0 else {
        crate::cascade::prof::hit(crate::cascade::prof::RJ_BASE);
        crate::cascade::prof::hit(match method {
            "forced" => crate::cascade::prof::BASE_FAIL_FORCED,
            "skel" => crate::cascade::prof::BASE_FAIL_SKEL,
            _ => crate::cascade::prof::BASE_FAIL_PAIR,
        });
        if method == "pair" {
            let dev = (mu[2].norm_sqr() - 1.0).abs();
            crate::cascade::prof::hit(if dev < 1e-6 {
                crate::cascade::prof::DEV_LT_1EM6
            } else if dev < 1e-2 {
                crate::cascade::prof::DEV_MID
            } else {
                crate::cascade::prof::DEV_GT_1EM2
            });
        }
        return None;
    };
    let tp = crate::cascade::prof::start();
    let mc = mirror_completion(
        &mb,
        pb,
        Ghat::Corner { g2, t1, t2 },
        c,
        rho1,
        rho2,
        a,
        delta,
        dcl,
        w,
        pre_rho2_ac,
        pre_ac,
        method,
    );
    crate::cascade::prof::rec(crate::cascade::prof::SW_MIRROR, tp);
    if let Some(s) = mc {
        return Some(s);
    }
    if g2 == C::default() {
        return None;
    }
    // The sign-class fiber over mu (the old 4-class eps loop) is gone:
    // sign bits are gauge (reduced-mirror-law ledger), corpus-confirmed --
    // with the fiber disabled both complete corpora stay 100% with
    // bit-identical rung ownership and residuals (its 2 accepts in 423,432
    // evaluations were independently owned by later candidates).
    None
}

fn verify(
    c: C,
    a: &[C; 4],
    vv: &[f64; 4],
    uc: &[f64; 4],
    rho1: C,
    rho2: C,
    w: &[C; 4],
) -> Option<Solved> {
    let tp = crate::cascade::prof::start();
    let r = verify_inner(c, a, vv, uc, rho1, rho2, w);
    crate::cascade::prof::rec(crate::cascade::prof::SW_VERIFY, tp);
    r
}

#[allow(clippy::too_many_arguments)]
fn verify_inner(
    c: C,
    a: &[C; 4],
    vv: &[f64; 4],
    uc: &[f64; 4],
    rho1: C,
    rho2: C,
    w: &[C; 4],
) -> Option<Solved> {
    let m = assemble(c, a, &[(*vv, rho1), (*uc, rho2)]);
    // spectrum match via Newton's identities on traces -- the crate's
    // master metric (no eigensolve; does not floor at degeneracy). This
    // removes the last eigensolve from the radical accept path; the
    // final frame_to_o / smooth_residual re-gate is unchanged.
    let m2 = m * m;
    let m3 = m2 * m;
    let m4 = m2 * m2;
    let tr = |x: &Mat4| (0..4).map(|i| x[(i, i)]).sum::<C>();
    let (p1, p2, p3, p4) = (tr(&m), tr(&m2), tr(&m3), tr(&m4));
    let e1 = p1;
    let e2 = (e1 * p1 - p2) / 2.0;
    let e3 = (e2 * p1 - e1 * p2 + p3) / 3.0;
    let e4 = (e3 * p1 - e2 * p2 + e1 * p3 - p4) / 4.0;
    // det(wj*I - m) = chi_m(wj) = wj^4 - e1*wj^3 + e2*wj^2 - e3*wj + e4;
    // Horner evaluation costs 4 muls per point (vs ~40 for a 4x4 determinant).
    let chi_max = w
        .iter()
        .map(|&wj| ((((wj - e1) * wj + e2) * wj - e3) * wj + e4).norm())
        .fold(0.0f64, f64::max);
    if chi_max > 1e-7 {
        return None;
    }
    let ew1 = w[0] + w[1] + w[2] + w[3];
    let ew2 = w[0] * w[1] + w[0] * w[2] + w[0] * w[3] + w[1] * w[2] + w[1] * w[3] + w[2] * w[3];
    let ew3 = w[0] * w[1] * w[2] + w[0] * w[1] * w[3] + w[0] * w[2] * w[3] + w[1] * w[2] * w[3];
    let ew4 = w[0] * w[1] * w[2] * w[3];
    let r = [
        (e1 - ew1).norm(),
        (e2 - ew2).norm(),
        (e3 - ew3).norm(),
        (e4 - ew4).norm(),
    ]
    .into_iter()
    .fold(0.0f64, f64::max);
    (r <= 1e-9).then_some(Solved {
        m,
        frame: Some(Frame {
            c,
            peels: vec![(*vv, rho1), (*uc, rho2)],
        }),
    })
}

/// The strict cyclic T4 word shared by every pin characteristic of one
/// two-step problem. Constructing it means sorting the two data spectra and
/// counting target letters in the four delta gaps; neither operation depends
/// on the pin pair.
struct StrictWord {
    /// Distinct delta angles, sorted; `gaps` of them.  A clustered delta
    /// contributes ONE node here: its forced mu copy sits AT the node (the
    /// confluent limit of the interlacing law -- the collapsing gap carries
    /// its mu into the forced copy), and the remaining `gaps` free values
    /// obey the same one-per-gap word law over the distinct nodes.
    delta: [f64; 4],
    gaps: usize,
    /// Nodes carrying a forced (clustered) copy: a pin at such a node is the
    /// forced letter and imposes NO word constraint.
    forced: [bool; 4],
    target: [f64; 4],
    target_count: [i32; 4],
    offset: [i32; 4],
    x0_lo: i32,
    x0_hi: i32,
}

impl StrictWord {
    fn angle(z: C) -> f64 {
        use std::f64::consts::TAU;
        let a = z.arg();
        if a < 0.0 {
            a + TAU
        } else {
            a
        }
    }

    fn cyclic_distance(x: f64, y: f64) -> f64 {
        use std::f64::consts::TAU;
        let d = (x - y).abs();
        d.min(TAU - d)
    }

    fn arc_in(delta: &[f64], p: f64) -> usize {
        let gaps = delta.len();
        (delta.iter().filter(|&&x| x <= p).count() + gaps - 1) % gaps
    }

    fn new(delta: &[C; 4], w: &[C; 4]) -> Option<Self> {
        let mut all = delta.map(Self::angle);
        all.sort_by(f64::total_cmp);
        // Cluster coincident delta angles: one node per cluster (the forced
        // copies live AT their nodes and are not free letters).
        let mut dn = [0.0f64; 4];
        let mut forced = [false; 4];
        let mut gaps = 0usize;
        for &a_ in &all {
            if gaps == 0 || Self::cyclic_distance(dn[gaps - 1], a_) >= XTOL {
                dn[gaps] = a_;
                gaps += 1;
            } else {
                forced[gaps - 1] = true;
            }
        }
        if gaps >= 2 && Self::cyclic_distance(dn[0], dn[gaps - 1]) < XTOL {
            gaps -= 1; // wrap-around cluster
            forced[0] = true;
        }
        if gaps < 2 {
            return None; // no word structure to prune with
        }
        let wl = w.map(Self::angle);
        for i in 0..4 {
            for j in (i + 1)..4 {
                if Self::cyclic_distance(wl[i], wl[j]) < XTOL {
                    return None;
                }
            }
        }
        for &node in &dn[..gaps] {
            for &target in &wl {
                if Self::cyclic_distance(node, target) < XTOL {
                    return None;
                }
            }
        }
        let mut target_count = [0i32; 4];
        for &target in &wl {
            target_count[Self::arc_in(&dn[..gaps], target)] += 1;
        }
        let mut offset = [0i32; 4];
        let (mut prefix, mut x0_lo, mut x0_hi) = (0i32, 0i32, 4i32);
        for j in 0..gaps {
            offset[j] = j as i32 - prefix;
            x0_lo = x0_lo.max(-offset[j]);
            x0_hi = x0_hi.min(target_count[j] - offset[j]);
            prefix += target_count[j];
        }
        Some(Self {
            delta: dn,
            gaps,
            forced,
            target: wl,
            target_count,
            offset,
            x0_lo,
            x0_hi,
        })
    }

    fn arc(&self, p: f64) -> usize {
        Self::arc_in(&self.delta[..self.gaps], p)
    }

    fn coordinate(p: f64, lo: f64) -> f64 {
        use std::f64::consts::TAU;
        let c = p - lo;
        if c < 0.0 {
            c + TAU
        } else {
            c
        }
    }

    /// Each option is `(delta_gap, fixed x_in_that_gap)`. There are exactly
    /// two closure conventions for a pin that is itself a strict data letter.
    fn pin_options(&self, t: C) -> Option<[(usize, i32); 2]> {
        let p = Self::angle(t);
        if let Some(k) = (0..4).find(|&k| Self::cyclic_distance(p, self.target[k]) < XTOL) {
            let g = self.arc(self.target[k]);
            let rank = (0..4)
                .filter(|&j| {
                    self.arc(self.target[j]) == g
                        && Self::coordinate(self.target[j], self.delta[g])
                            < Self::coordinate(self.target[k], self.delta[g])
                })
                .count() as i32;
            return Some([(g, rank), (g, rank + 1)]);
        }
        if let Some(k) = (0..self.gaps).find(|&k| Self::cyclic_distance(p, self.delta[k]) < XTOL) {
            if self.forced[k] {
                // The pin is the forced copy at a clustered node: it is not a
                // free letter and imposes no word constraint.
                return Some([(usize::MAX, 0), (usize::MAX, 0)]);
            }
            let before = (k + self.gaps - 1) % self.gaps;
            return Some([(k, 0), (before, self.target_count[before])]);
        }
        None
    }

    fn accepts(&self, t1: C, t2: C) -> bool {
        let Some(o1) = self.pin_options(t1) else {
            return false;
        };
        let Some(o2) = self.pin_options(t2) else {
            return false;
        };
        // The recurrence has the closed form x_j=x_0+j-prefix_j.  Intersect
        // its four box constraints once; each pin option then fixes x_0
        // outright.  No enumeration of the five possible x_0 values remains.
        for &(g1, fixed1) in &o1 {
            for &(g2, fixed2) in &o2 {
                let v1 = g1 == usize::MAX;
                let v2 = g2 == usize::MAX;
                if v1 && v2 {
                    return true; // both pins forced: no word constraint at all
                }
                if v1 || v2 {
                    // One vacuous pin: the other alone must fit the box.
                    let (g, fixed) = if v1 { (g2, fixed2) } else { (g1, fixed1) };
                    let x0 = fixed - self.offset[g];
                    if self.x0_lo <= x0 && x0 <= self.x0_hi {
                        return true;
                    }
                    continue;
                }
                if g1 == g2 {
                    continue; // one mu in every strict delta gap
                }
                let x0 = fixed1 - self.offset[g1];
                if x0 == fixed2 - self.offset[g2] && self.x0_lo <= x0 && x0 <= self.x0_hi {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Clone, Copy)]
struct PinPair {
    t1: C,
    t2: C,
    /// Position in the legacy raw enumeration.  The shallow/full scheduler
    /// partitions this index, so removing a dead characteristic must not
    /// silently promote a different one into the hot pass.
    raw_index: usize,
}

/// Enumerate the distinct algebraic odd characteristics of a branch
/// multiset.  Off-diagonal pairs always exist; a diagonal `{x,x}` exists iff
/// the branch value `x` has multiplicity at least two.  `slots[0..2]` is kept
/// as the preferred first pair when present, but its later duplicate is
/// removed.  The returned raw indices preserve the previous pass partition.
fn odd_characteristic_pairs(vals: &[(C, usize)], slots: &[C]) -> ([PinPair; 40], usize) {
    let empty = PinPair {
        t1: C::default(),
        t2: C::default(),
        raw_index: 0,
    };
    let mut pairs = [empty; 40];
    let mut npairs = 0usize;
    let mut raw_index = 0usize;
    let mut preferred = None;
    if slots.len() >= 2 {
        pairs[npairs] = PinPair {
            t1: slots[0],
            t2: slots[1],
            raw_index,
        };
        preferred = Some((slots[0], slots[1]));
        npairs += 1;
        raw_index += 1;
    }
    for i in 0..vals.len() {
        for j in i..vals.len() {
            let pair_raw_index = raw_index;
            raw_index += 1;
            if i == j && vals[i].1 < 2 {
                continue;
            }
            if preferred.is_some_and(|(x, y)| {
                ((vals[i].0 - x).norm_sqr() < XTOL * XTOL
                    && (vals[j].0 - y).norm_sqr() < XTOL * XTOL)
                    || ((vals[i].0 - y).norm_sqr() < XTOL * XTOL
                        && (vals[j].0 - x).norm_sqr() < XTOL * XTOL)
            }) {
                continue;
            }
            pairs[npairs] = PinPair {
                t1: vals[i].0,
                t2: vals[j].0,
                raw_index: pair_raw_index,
            };
            npairs += 1;
        }
    }
    (pairs, npairs)
}

/// T4 word gate for a pin pair (instrumentation increment; env WGATE).
/// On a strict eight-letter word, positivity of both residue steps is
/// equivalent to one `mu` in every cyclic delta gap and one `w` in every
/// cyclic `mu` gap.  If `w_j` is the number of target letters in delta gap
/// `j` and `x_j` counts those below `mu_j`, the latter condition is exactly
///
/// `x_{j+1} = x_j + 1 - w_j`.
///
/// A pin at a target letter fixes `x_j` immediately before or after that
/// letter. A pin at a delta letter belongs to either adjacent closed gap and
/// fixes `x` at that gap's endpoint. We enumerate those at most four boundary
/// conventions and the five possible integer values of `x_0`. `Some(false)`
/// therefore certifies that the characteristic cell has no positive real
/// placement. Coincident data letters remain outside the theorem's domain and
/// return `None`, so the gate never excludes them.
pub fn two_step<R>(
    c: C,
    d1: C,
    d2: C,
    a: &[C; 4],
    w: &[C; 4],
    cap: usize,
    accept: &mut impl FnMut(Solved) -> Option<R>,
) -> Option<R> {
    let th = crate::cascade::prof::start();
    let (rho1v, rho2v) = (d1 - c, d2 - c);
    if rho1v.norm() < CTOL || rho2v.norm() < CTOL {
        return None;
    }
    let delta: [C; 4] = std::array::from_fn(|k| c * a[k]);
    let dcl = clusters4(&delta, CTOL);
    let strict_word = StrictWord::new(&delta, w);
    let prod_a: C = a.iter().product();
    let pb = problem_base(&delta, &dcl, w);
    let all_simple = dcl.k == 4;
    let chi_w = chi4(w);
    let chi_d = chi4(&delta);
    let mut p_poly = [C::default(); 9];
    for i in 0..5 {
        for j in 0..5 {
            p_poly[i + j] += chi_w[i] * chi_d[j];
        }
    }
    let mut vals_buf = [(C::default(), 0usize); 8];
    let mut nvals = 0usize;
    for &z in w.iter().chain(delta.iter()) {
        match vals_buf[..nvals]
            .iter_mut()
            .find(|(y, _)| (*y - z).norm_sqr() < XTOL * XTOL)
        {
            Some(e) => e.1 += 1,
            None => {
                vals_buf[nvals] = (z, 1);
                nvals += 1;
            }
        }
    }
    let vals = &vals_buf[..nvals];
    let mut slots_buf = [C::default(); 7];
    let mut nslots = 0usize;
    for &(z, m) in vals {
        for _ in 1..m {
            slots_buf[nslots] = z;
            nslots += 1;
        }
    }
    let slots = &slots_buf[..nslots];
    // Delta angles are loop-invariant across both orders and all pair cells;
    // the strict-word masks below consume them once.
    let delta_args: [f64; 4] = std::array::from_fn(|k| delta[k].arg());
    // ac[k] = a[k] * pb.cis[k] is loop-invariant across both orders and all pairs;
    // precompute once to remove 4 complex muls from each PairGate::new call.
    let pre_ac: [C; 4] = std::array::from_fn(|k| a[k] * pb.cis[k]);
    // val angles and wildcard masks are loop-invariant across both orders; precompute
    // once to avoid 2x nvals atan2 + 2x nvals*4 norm() calls inside the orders loop.
    let mut pre_angs = [0.0f64; 8];
    let mut pre_masks = [0u8; 8];
    if all_simple {
        for vi_ in 0..nvals {
            let v = vals[vi_].0;
            let a_ = v.arg();
            pre_angs[vi_] = a_;
            let mut wild = false;
            let mut m = 0u8;
            for t in 0..4 {
                if (v - delta[t]).norm_sqr() < 1e-8 {
                    wild = true;
                }
                if a_ > delta_args[t] {
                    m |= 1 << t;
                }
            }
            pre_masks[vi_] = if wild { 0xFF } else { m };
        }
    }
    // The shallow constructions are asymmetric: both peel orders run.
    let orders = [
        (rho1v, rho2v, c * c * c * d1 * prod_a),
        (rho2v, rho1v, c * c * c * d2 * prod_a),
    ];
    crate::cascade::prof::rec(crate::cascade::prof::SW_HEADER, th);
    // On a (2,2) gate the two off-cluster values are equal. Exchanging the
    // two rank-one peels then leaves rho1, rho2, the determinant pin, every
    // characteristic, and the reconstructed matrix unchanged. Quotient that
    // exact gauge symmetry instead of evaluating the same fiber twice.
    let order_count = if (d1 - d2).norm_sqr() < XTOL * XTOL {
        1
    } else {
        2
    };
    for &(rho1, rho2, pin) in &orders[..order_count] {
        // The skeleton arrangement law: with the
        // det-pin closing the phase telescope, V_i of a 4-subset S is
        // rho-tilde_i * prod_S sin((theta_i - phi_s)/2), so admissibility
        // is FOUR PARITY conditions on S's above-node counts -- an XOR of
        // per-value 4-bit masks against a branch-calibrated target (the
        // half-angle branch = lifted angle sum mod 4pi). Wildcards (values
        // at a node) and uncalibrated branches fall through to the exact
        // gate, so the accept set is unchanged by construction.
        // pre_angs/pre_masks precomputed above; only argp = pin.arg() varies per order.
        let skel_masks: Option<f64> = if all_simple { Some(pin.arg()) } else { None };
        let mut skel_target: [Option<u8>; 2] = [None, None];
        // (rho * a[k]) * pb.cis[k]: left-to-right order matches the
        // per-cluster-rep denominator computation in mirror_base /
        // mirror_completion so the result is bit-identical.
        let pre_rho1_ac: [C; 4] = std::array::from_fn(|k| rho1 * a[k] * pb.cis[k]);
        let pre_rho2_ac: [C; 4] = std::array::from_fn(|k| rho2 * a[k] * pb.cis[k]);
        // Single-step cells: v = e_k exactly. The first peel is
        // then diagonal, so mu is data-explicit with one shifted value
        // a_k (c + rho1) that must pass through to a target root -- an exact
        // data gate -- and the remainder is one n=3 rank-one update with
        // classical Cauchy weights. Closed form, no cell search; the mirror
        // variant (u = e_k) arrives via this same block on the swapped peel
        // order. Restores the rho-lift equivariance of the cell family: the
        // two margin rows reached this cell only through the orbit
        // re-encoding (their direct encodings carry the pass-through match
        // at 2e-16).
        for k in 0..4 {
            let mstar = a[k] * (c + rho1);
            let (mut wm, mut best) = (usize::MAX, f64::INFINITY);
            for (j, &wj) in w.iter().enumerate() {
                let dd = (wj - mstar).norm_sqr();
                if dd < best {
                    best = dd;
                    wm = j;
                }
            }
            if wm == usize::MAX || best >= XTOL * XTOL {
                continue;
            }
            // n=3 Cauchy weights on the complementary block:
            // u_j^2 = -prod_{t != wm}(c a_j - w_t)
            //         / (rho2 a_j prod_{l != j,k}(c a_j - c a_l))
            let mut vv = [0.0f64; 4];
            vv[k] = 1.0;
            let mut uc = [0.0f64; 4];
            let mut ok = true;
            for j in 0..4 {
                if j == k {
                    continue;
                }
                let caj = c * a[j];
                let mut num = C::new(1.0, 0.0);
                for (t, &wt) in w.iter().enumerate() {
                    if t != wm {
                        num *= caj - wt;
                    }
                }
                let mut den = rho2 * a[j];
                for (l, &al) in a.iter().enumerate() {
                    if l != j && l != k {
                        den *= caj - c * al;
                    }
                }
                let uj2 = -num / den;
                if !uj2.re.is_finite() || uj2.im.abs() > 1e-6 || uj2.re < -1e-7 {
                    ok = false;
                    break;
                }
                uc[j] = uj2.re.max(0.0).sqrt();
            }
            if !ok {
                continue;
            }
            if let Some(s) = verify(c, a, &vv, &uc, rho1, rho2, w) {
                if let Some(hit) = accept(s) {
                    return Some(hit);
                }
            }
        }
        {
            // fully-forced mu (3+ forced slots pin the intermediate);
            // subset rungs run once, in the shallow pass only
            if cap != usize::MAX && slots.len() >= 3 {
                let mut cands: Vec<[C; 4]> = Vec::new();
                if slots.len() >= 4 {
                    cands.push([slots[0], slots[1], slots[2], slots[3]]);
                }
                cands.push([
                    slots[0],
                    slots[1],
                    slots[2],
                    pin / (slots[0] * slots[1] * slots[2]),
                ]);
                for mu in cands {
                    let prod: C = mu.iter().product();
                    if (prod - pin).norm_sqr() < 1e-12 && inherits_clusters(&mu, &delta, &dcl) {
                        if let Some(s) = try_mu(
                            &mu,
                            Some((C::default(), slots[0], slots[1])),
                            c,
                            rho1,
                            rho2,
                            a,
                            &delta,
                            &dcl,
                            &pb,
                            &p_poly,
                            w,
                            &pre_rho1_ac,
                            &pre_rho2_ac,
                            &pre_ac,
                            "forced",
                        ) {
                            if let Some(hit) = accept(s) {
                                return Some(hit);
                            }
                        }
                    }
                }
            }
            // Even characteristics: the skeleton 4+4 splits (shallow pass
            // only; the full pass would rescan identically).  Complementing a
            // split and exchanging rho1/rho2 is one exact characteristic, but
            // both numerical orientations remain here until a certified
            // conditioning rule can select between their residue formulas.
            // gamma = 0 is codimension 1, so these are the non-generic case, yet
            // deferring them to the full pass measures no change: the tries are
            // cheap and the cost sits in the try_mu calls of the pin path.  Gating
            // this sweep off moves a row between rungs and is slower, so it owns
            // at least one row through another success path; keep it.
            if cap != usize::MAX {
                let mut av = [C::default(); 8];
                for (t, &(z, _)) in vals.iter().enumerate() {
                    av[t] = z;
                }
                let n = nvals;
                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            for l in (k + 1)..n {
                                if (av[i] * av[j] * av[k] * av[l] - pin).norm_sqr() < 1e-12 {
                                    let mu = [av[i], av[j], av[k], av[l]];
                                    if !inherits_clusters(&mu, &delta, &dcl) {
                                        crate::cascade::prof::hit(crate::cascade::prof::INH_SKIP);
                                        continue;
                                    }
                                    let mut gate_known = false;
                                    let mut gate_ok = true;
                                    if let Some(argp) = &skel_masks {
                                        let ms = [
                                            pre_masks[i],
                                            pre_masks[j],
                                            pre_masks[k],
                                            pre_masks[l],
                                        ];
                                        if ms.iter().all(|&m| m != 0xFF) {
                                            let pv = (ms[0] ^ ms[1] ^ ms[2] ^ ms[3]) & 0x0F;
                                            let ssum = pre_angs[i]
                                                + pre_angs[j]
                                                + pre_angs[k]
                                                + pre_angs[l];
                                            let br = ((((ssum - argp)
                                                / (2.0 * std::f64::consts::PI))
                                                .round()
                                                as i64)
                                                & 1)
                                                as usize;
                                            match skel_target[br] {
                                                Some(tgt) => {
                                                    gate_known = true;
                                                    gate_ok = pv == tgt;
                                                }
                                                None => {
                                                    // calibrate on this subset's exact V signs
                                                    let mut neg = 0u8;
                                                    let mut clean = true;
                                                    for t in 0..4 {
                                                        let d = delta[t];
                                                        let mut bd = C::new(1.0, 0.0);
                                                        for &m_ in &mu {
                                                            bd *= d - m_;
                                                        }
                                                        let vi_ = -bd / (rho1 * a[t] * pb.cis[t]);
                                                        // sign-trustworthy: real part dominates
                                                        // the pin-tolerance imaginary echo
                                                        if vi_.re.abs() < 1e-6
                                                            || vi_.im.abs() > 0.1 * vi_.re.abs()
                                                        {
                                                            clean = false;
                                                            break;
                                                        }
                                                        if vi_.re < 0.0 {
                                                            neg |= 1 << t;
                                                        }
                                                    }
                                                    if clean {
                                                        let calibrated = pv ^ neg;
                                                        skel_target[br] = Some(calibrated);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if gate_known && !gate_ok {
                                        crate::cascade::prof::hit(crate::cascade::prof::SKEL_SKIP);
                                        continue;
                                    }
                                    if let Some(s) = try_mu(
                                        &mu,
                                        Some((C::default(), av[i], av[j])),
                                        c,
                                        rho1,
                                        rho2,
                                        a,
                                        &delta,
                                        &dcl,
                                        &pb,
                                        &p_poly,
                                        w,
                                        &pre_rho1_ac,
                                        &pre_rho2_ac,
                                        &pre_ac,
                                        "skel",
                                    ) {
                                        if let Some(hit) = accept(s) {
                                            return Some(hit);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // odd characteristics: pin pairs, the beta-cubic closure
            let (pairs, npairs) = odd_characteristic_pairs(vals, slots);
            let mult_of = |t: C| -> usize {
                vals.iter()
                    .find(|(z, _)| (*z - t).norm_sqr() < XTOL * XTOL)
                    .map(|&(_, m)| m)
                    .unwrap_or(1)
            };
            // These are PIN pairs, and a pin sits at a REPEATED value -- that is
            // what makes it a pin. So order by descending multiplicity weight
            // rather than by construction index: an exact structural criterion
            // (mult_of is already the multiplicity), not a fit. Stable, so ties
            // keep the previous order and the shallow pass still takes 6.
            // Ordering these pin pairs by descending multiplicity globally cuts
            // p99 by 30% but demotes low-multiplicity pins past the cap; sorting
            // within each pass loses the entire win, so the effect is pass
            // membership, not order.  The open question is which pin is correct.
            for pair in pairs[..npairs].iter().filter(|pair| {
                if cap == usize::MAX {
                    pair.raw_index >= 6
                } else {
                    pair.raw_index < cap
                }
            }) {
                let (t1, t2) = (pair.t1, pair.t2);
                let tp = crate::cascade::prof::start();
                let wg = strict_word.as_ref().map(|word| word.accepts(t1, t2));
                crate::cascade::prof::rec(crate::cascade::prof::SW_WORD_GATE, tp);
                if wg == Some(false) {
                    continue;
                }
                let pp = pin / (t1 * t2);
                let arc = if all_simple {
                    let g = PairGate::new(t1, t2, pp, &delta, &pre_ac, rho1);
                    crate::cascade::prof::hit(if g.is_some() {
                        crate::cascade::prof::PAIR_GATE_SOME
                    } else {
                        crate::cascade::prof::PAIR_GATE_NONE
                    });
                    g
                } else {
                    None
                };
                // reduced gate (confluent descent): the (2,1,1) leading
                // case (one 2-cluster, one pin at the letter) and the (2,2)
                // both-letters case (two 2-clusters, both pins at the two
                // distinct letters; no surviving pin, two cluster terms).
                let arc_r: Option<PairGateR> = if !all_simple
                    && (0..dcl.k).all(|c| dcl.cluster_len(c) <= 2)
                {
                    // two_buf: reps of size-2 clusters; reps_buf: all cluster reps
                    let mut two_buf = [0usize; 4];
                    let mut n_two = 0usize;
                    let mut reps_buf = [0usize; 4];
                    for ci in 0..dcl.k {
                        reps_buf[ci as usize] = dcl.cluster_rep(ci);
                        if dcl.cluster_len(ci) == 2 {
                            two_buf[n_two] = dcl.cluster_rep(ci);
                            n_two += 1;
                        }
                    }
                    let n_reps = dcl.k as usize;
                    {
                        // Drop pins that ARE inheritance copies (at 2-cluster
                        // letters); the survivors feed the generic reduced
                        // gate.  A pair putting BOTH pins on the SAME letter
                        // is a multiplicity-ambiguous configuration and stays
                        // ungated, as before.
                        let at_letter =
                            |t: C| (0..n_two).find(|&i| (t - delta[two_buf[i]]).norm_sqr() < 1e-8);
                        let (l1, l2) = (at_letter(t1), at_letter(t2));
                        if l1.is_some() && l1 == l2 {
                            None
                        } else {
                            let mut pins_buf = [C::default(); 2];
                            let mut npins = 0usize;
                            if l1.is_none() {
                                pins_buf[npins] = t1;
                                npins += 1;
                            }
                            if l2.is_none() {
                                pins_buf[npins] = t2;
                                npins += 1;
                            }
                            PairGateR::new(
                                &pins_buf[..npins],
                                pp,
                                &reps_buf[..n_reps],
                                &delta,
                                &pb.cis,
                                rho1,
                                rho2,
                                a,
                                w,
                                &two_buf[..n_two],
                            )
                        }
                    }
                } else {
                    None
                };
                if let Some(g) = &arc {
                    if g.hopeless() {
                        crate::cascade::prof::hit(crate::cascade::prof::ARC_PAIR_SKIP);
                        continue;
                    }
                }
                let (m1c, m2c) = if (t1 - t2).norm_sqr() < XTOL * XTOL {
                    (1, 1)
                } else {
                    (mult_of(t1), mult_of(t2))
                };
                let tp = crate::cascade::prof::start();
                let beta1 = free_pair_beta(&p_poly, t1, t2, 1, 1, pp);
                if beta1.is_some_and(|beta| {
                    arc.as_ref()
                        .is_some_and(|g| g.excludes_closure(beta, pp, t1, t2))
                }) {
                    crate::cascade::prof::hit(crate::cascade::prof::ARC_PAIR_SKIP);
                    continue;
                }
                let (r1, nr1) = beta1
                    .map(|beta| lift_beta_roots(beta, pp))
                    .unwrap_or(([C::default(); 8], 0));
                crate::cascade::prof::rec(crate::cascade::prof::SW_PAIR_ROOTS, tp);
                let (r2, nr2) = if m1c > 1 || m2c > 1 {
                    free_pair_roots(&p_poly, t1, t2, m1c, m2c, pp)
                } else {
                    ([C::default(); 8], 0)
                };
                for &m_free_raw in r1[..nr1]
                    .iter()
                    .chain(r2[..nr2].iter())
                    .chain([t1, t2].iter())
                {
                    // the true intermediate spectrum is unit-modulus
                    // (spectrum of a unitary); the radial component of a
                    // near-circle root is rooter float noise, and projecting
                    // it out restores the exact phase telescope so the
                    // three-law gate applies (unprojected, V.im = kappa *
                    // radial defeats the reality gate spuriously).
                    let m_free = {
                        let n = m_free_raw.norm();
                        if (n - 1.0).abs() < 1e-4 {
                            m_free_raw / n
                        } else {
                            m_free_raw
                        }
                    };
                    let arc_reject = arc
                        .as_ref()
                        .is_some_and(|g| g.hopeless() || g.rejects(m_free));
                    if arc_reject {
                        crate::cascade::prof::hit(crate::cascade::prof::ARC_CAND_SKIP);
                        continue;
                    }
                    let mu = [t1, t2, m_free, pp / m_free];
                    // The inheritance gate (confluent three laws): an
                    // admissible mu on clustered data must carry each cluster
                    // value with multiplicity s-1 (proven); candidates without
                    // the copies are theorem-dead and currently burn a full
                    // residue evaluation before the reality gate kills them.
                    let inh_reject = !all_simple && !inherits_clusters(&mu, &delta, &dcl);
                    if inh_reject {
                        crate::cascade::prof::hit(crate::cascade::prof::INH_SKIP);
                        continue;
                    }
                    let red_reject =
                        !inh_reject && arc_r.as_ref().is_some_and(|g| g.rejects(m_free));
                    if red_reject {
                        crate::cascade::prof::hit(crate::cascade::prof::RED_SKIP);
                        continue;
                    }
                    let m2f = mu[3]; // == pp/m_free; already computed above for mu
                    let q_a = (m_free - t1) * (m_free - t2);
                    let q_b = (m2f - t1) * (m2f - t2);
                    let (x, qx) = if q_a.norm_sqr() >= q_b.norm_sqr() {
                        (m_free, q_a)
                    } else {
                        (m2f, q_b)
                    };
                    let on_data = |z: C| vals.iter().any(|&(y, _)| (y - z).norm_sqr() < 1e-18);
                    let gq = if on_data(m_free) && on_data(m2f) {
                        Some((C::default(), t1, t2))
                    } else if qx.norm_sqr() > 1e-12 {
                        let px = p_poly.iter().fold(C::default(), |acc, &co| acc * x + co);
                        Some((-px / (rho1 * rho2 * x * x * qx * qx), t1, t2))
                    } else {
                        None
                    };
                    // layer-2 gate: U-side reality is one ray condition on
                    // zeta = g2 * ray (derived from the ray theorem; the pair
                    // dependence cancels). Factor-2 margins; shadow-verified.
                    let uray_reject = match (&gq, &arc) {
                        (Some((g2v, _, _)), Some(g)) => match g.ray {
                            Some(ray) if *g2v != C::default() => {
                                // validity domain: the zeta factorization needs
                                // B(d_i) != 0, i.e. the free pair clear of the
                                // data nodes (else the deflated B* is not N/B)
                                let clear = delta.iter().all(|&d| {
                                    (m_free - d).norm_sqr() > 1e-8 && (m2f - d).norm_sqr() > 1e-8
                                });
                                if clear {
                                    let z = *g2v * ray;
                                    if z.re <= 0.0 || z.im.abs() > 2.0 * z.re {
                                        true
                                    } else if g.mag_ok && (m_free.norm_sqr() - 1.0).abs() < 1e-6 {
                                        // magnitude law: U-sum = zeta S(x) = 1
                                        let x = (m_free * g.half_product.conj()).re / m_free.norm();
                                        let s_of_x: f64 = (0..4)
                                            .map(|k| 2.0 * g.r2[k] / (g.kap[k] * (x - g.cc[k])))
                                            .sum();
                                        (z.re * s_of_x - 1.0).abs() > 1e-2
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        },
                        // reduced U-side (laws 2-3): clustered pairs whose
                        // interval gate exists get the same ray dichotomy +
                        // det-pin magnitude with the cluster's rational term
                        (Some((g2v, _, _)), None) if *g2v != C::default() => arc_r
                            .as_ref()
                            .is_some_and(|g| !red_reject && g.rejects_u(m_free, pp, *g2v, &delta)),
                        _ => false,
                    };
                    if uray_reject {
                        crate::cascade::prof::hit(crate::cascade::prof::URAY_SKIP);
                        continue;
                    }
                    if let Some(s) = try_mu(
                        &mu,
                        gq,
                        c,
                        rho1,
                        rho2,
                        a,
                        &delta,
                        &dcl,
                        &pb,
                        &p_poly,
                        w,
                        &pre_rho1_ac,
                        &pre_rho2_ac,
                        &pre_ac,
                        "pair",
                    ) {
                        if let Some(hit) = accept(s) {
                            return Some(hit);
                        }
                    }
                }
            }
        }
    }
    None
}

fn rank_one(c: C, d: C, a: &[C; 4], w: &[C; 4]) -> Option<Solved> {
    let rho = d - c;
    let delta: [C; 4] = std::array::from_fn(|k| c * a[k]);
    let dcl = clusters4(&delta, CTOL);
    let v = residues_v(w, &delta, &dcl, rho, a);
    let v = v?;
    let m = assemble(c, a, &[(v, rho)]);
    let spec = eig4(&m)?;
    let r = sdist(&spec, w);
    (r <= 1e-9).then_some(Solved {
        m,
        frame: Some(Frame {
            c,
            peels: vec![(v, rho)],
        }),
    })
}

pub fn solve_oriented<R>(
    gate: &[C; 4],
    a: &[C; 4],
    w: &[C; 4],
    cap: usize,
    accept: &mut impl FnMut(Solved) -> Option<R>,
) -> Option<R> {
    let cls = clusters4(gate, CTOL);
    let k = cls.k as usize;
    // Sorted cluster sizes (at most 4, no heap alloc).
    let mut sizes = [0usize; 4];
    for c in 0..cls.k {
        sizes[c as usize] = cls.cluster_len(c);
    }
    sizes[..k].sort_unstable();
    match &sizes[..k] {
        // A scalar gate makes the output spectrum independent of the frame.
        // Every feasible instance is therefore a vertex, exhausted before
        // multiplicity recursion enters this function.
        [4] => None,
        [1, 3] => {
            let tri_rep = (0..cls.k)
                .find(|&c| cls.cluster_len(c) == 3)
                .map(|c| cls.cluster_rep(c))
                .unwrap();
            let sng_rep = (0..cls.k)
                .find(|&c| cls.cluster_len(c) == 1)
                .map(|c| cls.cluster_rep(c))
                .unwrap();
            let tp = crate::cascade::prof::start();
            let r = rank_one(gate[tri_rep], gate[sng_rep], a, w);
            crate::cascade::prof::rec(crate::cascade::prof::SW_RANK1, tp);
            r.and_then(accept)
        }
        [2, 2] | [1, 1, 2] => {
            for pair_c in 0..cls.k {
                if cls.cluster_len(pair_c) != 2 {
                    continue;
                }
                let pair_rep = cls.cluster_rep(pair_c);
                // "sing" = all indices NOT in this pair cluster
                let mut sing = [0usize; 2];
                let mut ns = 0;
                for i in 0..4 {
                    if cls.cluster_of[i] != pair_c {
                        sing[ns] = i;
                        ns += 1;
                    }
                }
                let (c, d1, d2) = (gate[pair_rep], gate[sing[0]], gate[sing[1]]);
                let tp = crate::cascade::prof::start();
                let hit = two_step(c, d1, d2, a, w, cap, accept);
                crate::cascade::prof::rec(crate::cascade::prof::SW_TWO_STEP, tp);
                if let Some(s) = hit {
                    return Some(s);
                }
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod beta_interval_tests {
    use super::*;

    #[test]
    fn odd_characteristics_are_multiset_pairs_without_scheduler_promotion() {
        let simple: [(C, usize); 8] = std::array::from_fn(|i| (C::new(i as f64 + 1.0, 0.0), 1));
        let (strict, n_strict) = odd_characteristic_pairs(&simple, &[]);
        assert_eq!(n_strict, 28);
        // Raw index zero was the dead {0,0} diagonal.  Deleting it must not
        // move a sixth characteristic into the raw-index < 6 hot pass.
        assert_eq!(
            strict[..n_strict]
                .iter()
                .filter(|p| p.raw_index < 6)
                .count(),
            5
        );

        let one_node: [(C, usize); 7] =
            std::array::from_fn(|i| (C::new(i as f64 + 1.0, 0.0), 1 + usize::from(i == 0)));
        let (_, n_one_node) = odd_characteristic_pairs(&one_node, &[]);
        assert_eq!(n_one_node, 22); // C(7,2) plus the one genuine diagonal

        let two_nodes: [(C, usize); 6] =
            std::array::from_fn(|i| (C::new(i as f64 + 1.0, 0.0), 1 + usize::from(i < 2)));
        let slots = [two_nodes[0].0, two_nodes[1].0];
        let (pairs, n_two_nodes) = odd_characteristic_pairs(&two_nodes, &slots);
        assert_eq!(n_two_nodes, 17); // C(6,2) plus two genuine diagonals
        for i in 0..n_two_nodes {
            for j in (i + 1)..n_two_nodes {
                let same = (pairs[i].t1 == pairs[j].t1 && pairs[i].t2 == pairs[j].t2)
                    || (pairs[i].t1 == pairs[j].t2 && pairs[i].t2 == pairs[j].t1);
                assert!(!same, "duplicate characteristic at {i},{j}");
            }
        }
    }

    fn closure_from_real_power(power: &[f64], h: C, phase: C) -> BetaClosure {
        let degree = power.len() - 1;
        let two_h = h * 2.0;
        let mut coefficients = [C::default(); 4];
        for (k, &ak) in power.iter().enumerate() {
            coefficients[degree - k] = phase * ak / two_h.powu(k as u32);
        }
        BetaClosure {
            coefficients,
            len: degree + 1,
            double_root: None,
        }
    }

    #[test]
    fn beta_interval_certificate_never_excludes_a_planted_real_root() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut sample = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64 / ((1u64 << 53) as f64)) * 1.8 - 0.9
        };
        for _ in 0..10_000 {
            let mut roots = [sample(), sample(), sample()];
            roots.sort_by(f64::total_cmp);
            let power = [
                -roots[0] * roots[1] * roots[2],
                roots[0] * roots[1] + roots[0] * roots[2] + roots[1] * roots[2],
                -(roots[0] + roots[1] + roots[2]),
                1.0,
            ];
            let angle = sample() * std::f64::consts::PI;
            let h0 = C::new(angle.cos(), angle.sin());
            let h = (h0 * h0).sqrt();
            let phase_angle = sample() * std::f64::consts::PI;
            let phase = C::new(phase_angle.cos(), phase_angle.sin());
            let beta = closure_from_real_power(&power, h, phase);
            for &root in &roots {
                let lo = (root - 1e-6).max(-1.0);
                let hi = (root + 1e-6).min(1.0);
                assert!(
                    beta_interval_variations(beta, h * h, lo, hi) != Some(0),
                    "roots={roots:?} root={root} interval=({lo},{hi}) h={h:?} phase={phase:?} power={power:?}"
                );
            }
        }
    }

    #[test]
    fn beta_interval_certificate_proves_a_root_free_interval() {
        // (x+0.7)(x-0.1)(x-0.8) has no root on [-0.55,-0.25].
        let roots = [-0.7, 0.1, 0.8];
        let power = [
            -roots[0] * roots[1] * roots[2],
            roots[0] * roots[1] + roots[0] * roots[2] + roots[1] * roots[2],
            -(roots[0] + roots[1] + roots[2]),
            1.0,
        ];
        let h = C::new(0.6, 0.8);
        let phase = C::new(-0.8, 0.6);
        let beta = closure_from_real_power(&power, h, phase);
        assert_eq!(beta_interval_variations(beta, h * h, -0.55, -0.25), Some(0));
        assert_ne!(beta_interval_variations(beta, h * h, -0.7, -0.25), Some(0));
    }

    #[test]
    fn complex_monic_roots_small_recovers_unit_circle_roots() {
        // the production population: monic deg 1..4 with unit-modulus roots
        // (mu' is a spectrum), incl. near-coincident pairs
        let seeds: [&[C]; 5] = [
            &[C::new(0.6, 0.8)],
            &[C::new(0.6, 0.8), C::new(-0.28, -0.96)],
            &[C::new(0.6, 0.8), C::new(-0.28, -0.96), C::new(0.96, -0.28)],
            &[
                C::new(0.6, 0.8),
                C::new(-0.28, -0.96),
                C::new(0.96, -0.28),
                C::new(-0.8, 0.6),
            ],
            &[
                C::new(0.6, 0.8),
                C::new(0.6000001, 0.7999999),
                C::new(0.96, -0.28),
                C::new(-0.8, 0.6),
            ],
        ];
        for roots in seeds {
            let p = poly_from_roots(C::new(1.0, 0.0), roots);
            let mut out = [C::default(); 8];
            let n = complex_monic_roots_small(&p[..=roots.len()], &mut out)
                .expect("solver must handle deg <= 4");
            assert_eq!(n, roots.len());
            for &r in roots.iter() {
                let best = out[..n]
                    .iter()
                    .map(|&z| (z - r).norm())
                    .fold(f64::INFINITY, f64::min);
                assert!(best < 5e-8, "root {r} recovered at {best:e}");
            }
        }
    }

    #[test]
    fn monic_complement_recovers_the_other_factor_without_root_order() {
        let left = [
            C::new(0.6, 0.8),
            C::new(-0.8, 0.6),
            C::new(-0.28, -0.96),
            C::new(0.96, -0.28),
        ];
        let right = [
            C::new(0.0, 1.0),
            C::new(-0.6, -0.8),
            C::new(0.8, -0.6),
            C::new(-1.0, 0.0),
        ];
        for d in 1..=4 {
            let b = poly_from_roots(C::new(1.0, 0.0), &left[..d]);
            let q = poly_from_roots(C::new(1.0, 0.0), &right[..d]);
            let mut n = [C::default(); 17];
            for i in 0..=d {
                for j in 0..=d {
                    n[i + j] += b[i] * q[j];
                }
            }
            let (got, len) = monic_complement(&n[..=2 * d], &left[..d]).unwrap();
            assert_eq!(len, d + 1);
            for k in 0..=d {
                assert!((got[k] - q[k]).norm() < 2e-14, "d={d} k={k}");
            }
        }
    }
}
