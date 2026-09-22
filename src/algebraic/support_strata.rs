//! Closed support strata and exact-confluence dispatch.
//!
//! Vertex/edge/face are block-support strata of the same spectral update.
//! `edge_gate` proves, before support-incidence enumeration, whether a vertex or edge
//! can meet the acceptance contract. The dispatcher then calls the linear
//! edge/face formulas here. Exact multiplicities enter `solve_confluent` only
//! after those cheaper strata and the Klein section have declined.

use super::{
    ACCEPT, C, Mat4, PLANES, SpectrumKind, StratumSignature, compound_residual, givens,
    recover_frame, signed_perm,
};

const PAIR6: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

fn complement(i: usize, j: usize) -> (usize, usize) {
    let mut o = (0..4).filter(|&k| k != i && k != j);
    (o.next().unwrap(), o.next().unwrap())
}

/// Face rung: `O = givens(a,b)·givens(c,d)·P` with `{a,b}∩{c,d}=∅`. `M` is block-diagonal in the
/// two 2×2 blocks (disjoint planes -> NO orthostochastic cross-term); each block has θ-free det and
/// trace affine in `cos2θ`. Partition the 4 target eigenvalues into the two block pairs by the θ-free
/// det, then solve each block's `cos2θ` LINEARLY -- two independent edge solves. Closed-form, no eig,
/// machine-precise; well-conditioned exactly where the deg-6 chart degrades (the near-face tail).
pub(crate) fn solve_face(
    d2: &[C; 4],
    lam4: &[C; 4],
    dc: &Mat4,
    lam: &Mat4,
    target_specs: &[[C; 4]],
    targets: &[[C; 4]],
) -> Option<(Mat4, f64)> {
    let half = C::new(0.5, 0.0);
    // Every block det d²_a d²_b λ_x λ_y depends only on the UNORDERED pairs
    // {a,b} and {x,y}, so the det gate lives on PAIR6 x PAIR6, not on ranked
    // permutations: 6x6x6 checks replace the old 864-iteration perm sweep,
    // and the order/sign choices inside a block are enumerated only on a det
    // match (the common decline never pays them). PAIR6 is ordered so that
    // the complement of pair t is pair 5-t.
    let d2p: [C; 6] = std::array::from_fn(|t| d2[PAIR6[t].0] * d2[PAIR6[t].1]);
    let lamp: [C; 6] = std::array::from_fn(|t| lam4[PAIR6[t].0] * lam4[PAIR6[t].1]);
    for (bi, w) in target_specs.iter().enumerate() {
        let wp: [C; 6] = std::array::from_fn(|t| w[PAIR6[t].0] * w[PAIR6[t].1]);
        // p6 ranges over the pairs containing index 0: one representative per
        // partition (p6 <-> 5-p6 with q6/t6 swapped is the same configuration).
        for p6 in 0..3 {
            let (a, b) = PAIR6[p6];
            let (c, d) = PAIR6[5 - p6];
            for q6 in 0..6 {
                let det_ab = d2p[p6] * lamp[q6];
                let det_cd = d2p[5 - p6] * lamp[5 - q6];
                for (t6, &(i, j)) in PAIR6.iter().enumerate() {
                    // norm_sqr vs (1e-7)^2: same predicate, no sqrt.
                    if (wp[t6] - det_ab).norm_sqr() > 1e-14 {
                        continue; // this target pair isn't block {a,b}'s spectrum
                    }
                    if (wp[5 - t6] - det_cd).norm_sqr() > 1e-14 {
                        continue;
                    }
                    let (k, l) = complement(i, j);
                    let (x, y) = PAIR6[q6];
                    let (z, v) = PAIR6[5 - q6];
                    // block trace = A + B·cos2θ; swapping the λ assignment
                    // within a block flips B, so both orders are tried.
                    for s1 in 0..2 {
                        for s2 in 0..2 {
                            let (xa, xb) = if s1 == 0 { (x, y) } else { (y, x) };
                            let (za, zb) = if s2 == 0 { (z, v) } else { (v, z) };
                            let a_ab = half * (d2[a] + d2[b]) * (lam4[xa] + lam4[xb]);
                            let b_ab = half * (lam4[xa] - lam4[xb]) * (d2[a] - d2[b]);
                            let a_cd = half * (d2[c] + d2[d]) * (lam4[za] + lam4[zb]);
                            let b_cd = half * (lam4[za] - lam4[zb]) * (d2[c] - d2[d]);
                            if b_ab.norm() < 1e-12 || b_cd.norm() < 1e-12 {
                                continue;
                            }
                            let u1 = (w[i] + w[j] - a_ab) / b_ab;
                            let u2 = (w[k] + w[l] - a_cd) / b_cd;
                            if u1.im.abs() > 1e-6 || u2.im.abs() > 1e-6 {
                                continue;
                            }
                            if !(-1.0001..=1.0001).contains(&u1.re)
                                || !(-1.0001..=1.0001).contains(&u2.re)
                            {
                                continue;
                            }
                            let th1 = u1.re.clamp(-1.0, 1.0).acos() / 2.0;
                            let th2 = u2.re.clamp(-1.0, 1.0).acos() / 2.0;
                            let mut perm = [0usize; 4];
                            perm[a] = xa;
                            perm[b] = xb;
                            perm[c] = za;
                            perm[d] = zb;
                            let o = givens(a, b, th1) * givens(c, d, th2) * signed_perm(perm);
                            let r = compound_residual(dc, lam, &o, &targets[bi]);
                            if r < ACCEPT {
                                return Some((o, r));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Lossless row-level gate for every vertex and one-Givens edge.
///
/// An edge leaves two routed eigenvalues `d2[k] * lam[m]` unchanged.  If a
/// routed value is farther than `DELTA` from every target root, evaluation of
/// the target characteristic polynomial at that value gives
/// `4 * residual >= DELTA^4`.  The chosen delta therefore excludes only
/// candidates whose residual is already larger than `ACCEPT`.  A vertex is a
/// special edge and necessarily passes the same gate.
pub(crate) struct RoutedSupport {
    pub(crate) edge: Option<[[u8; 4]; 4]>,
    pub(crate) exact: [[u8; 4]; 4],
}

pub(crate) fn edge_gate(routed: &[[C; 4]; 4], target_specs: &[[C; 4]]) -> RoutedSupport {
    const DELTA_SQ: f64 = 1e-4;
    let mut viable = [[0u8; 4]; 4];
    let mut exact = [[0u8; 4]; 4];
    let (mut rows, mut cols) = ([0u8; 8], [0u8; 8]);
    for k in 0..4 {
        for m in 0..4 {
            let prod = routed[k][m];
            for (b, spec) in target_specs.iter().enumerate() {
                let distance = spec
                    .iter()
                    .map(|w| (prod - w).norm_sqr())
                    .fold(f64::INFINITY, f64::min);
                if distance <= 1e-16 {
                    exact[k][m] |= 1 << b;
                }
                if distance <= DELTA_SQ {
                    viable[k][m] |= 1 << b;
                    rows[b] |= 1 << k;
                    cols[b] |= 1 << m;
                }
            }
        }
    }
    let edge = (0..target_specs.len())
        .any(|b| rows[b].count_ones() >= 2 && cols[b].count_ones() >= 2)
        .then_some(viable);
    RoutedSupport { edge, exact }
}

/// Edge rung: `O = givens(i,j,θ)·P`. `M` differs from diagonal only in the `(i,j)`
/// block; `e₁(u)=A+B·u` is affine in `u=cos2θ`, `det` is θ-free. Solve `u` linearly per
/// `(perm, plane)`, require it real and in `[-1,1]`, then verify the full `symfn`.
/// Bounded `24×6` enumeration + a linear solve -- closed-form, not a scan.
///
/// SPECTRAL GATE (exact, proven lossless): an edge frame leaves the two
/// off-plane diagonal entries `μ = d²_k·λ'_k` untouched, and those are then
/// eigenvalues of `M` EXACTLY (the block decouples). If such a `μ` has
/// distance > δ from EVERY target eigenvalue, then `|p_w(μ)| = Π_j|μ−w_j| >
/// δ⁴`, while `|p_w(μ)| = |p_w(μ) − p_M(μ)| ≤ 4·max_k|Δe_k|` (|μ| = 1), so the
/// smooth residual exceeds δ⁴/4. With δ = 1e-2 that bound is 2.5e-9 > ACCEPT:
/// skipping the pair cannot lose an acceptable frame. `target_specs` are the
/// target EIGENVALUE sets per branch (same order as `targets`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_edge(
    d2: &[C; 4],
    lam4: &[C; 4],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]],
    routed: &[[C; 4]; 4],
    viable: &[[u8; 4]; 4],
    perms: &[[usize; 4]],
) -> Option<(Mat4, f64)> {
    let half = C::new(0.5, 0.0);
    for &p in perms.iter() {
        let lp: [C; 4] = std::array::from_fn(|k| lam4[p[k]]); // λ'_k
        for &(i, j) in PLANES.iter() {
            let (k1, k2) = complement(i, j);
            let mask = viable[k1][p[k1]] & viable[k2][p[k2]];
            if mask == 0 {
                continue; // both fixed eigenvalues provably miss every branch
            }
            // e₁(u) = A + B·u : A = Σ_{k∉{i,j}} d²λ' + ½(d²_i+d²_j)(λ'_i+λ'_j),
            //                   B = ½(λ'_i−λ'_j)(d²_i−d²_j).
            let mut a_coef = half * (d2[i] + d2[j]) * (lp[i] + lp[j]);
            for k in 0..4 {
                if k != i && k != j {
                    a_coef += routed[k][p[k]];
                }
            }
            let b_coef = half * (lp[i] - lp[j]) * (d2[i] - d2[j]);
            if b_coef.norm() < 1e-12 {
                continue; // block has no θ-dependence (degenerate); not an edge here
            }
            for (b, t) in targets.iter().enumerate() {
                if mask & (1 << b) == 0 {
                    continue;
                }
                let u = (t[0] - a_coef) / b_coef;
                if u.im.abs() > 1e-3 || u.re < -1.0001 || u.re > 1.0001 {
                    continue; // not a real in-range cos2θ -> target isn't on this edge
                }
                let theta = u.re.clamp(-1.0, 1.0).acos() / 2.0;
                let o = givens(i, j, theta) * signed_perm(p);
                let r = compound_residual(dc, lam, &o, t);
                if r < ACCEPT {
                    return Some((o, r));
                }
            }
        }
    }
    None
}

/// Radical strata of the distilled solver in all four triangle orientations.
/// Direct and role-swapped frames are reconstructed from peel data;
/// target-reanchored frames use exact inverse-eigenframe transport.  Every
/// candidate is re-gated on the original smooth residual.
#[derive(Clone, Copy)]
enum RadicalReconstruction {
    Sandwich { transpose: bool },
    Reanchor { transpose: bool },
}

#[derive(Clone, Copy)]
struct RadicalOrientation {
    gate: [C; 4],
    prefix: [C; 4],
    output: [C; 4],
    eigenvalues: [C; 4],
    owner: SpectrumKind,
    reconstruction: RadicalReconstruction,
}

/// Evaluate one triangle orientation of the same confluent inverse problem.
/// The four production actions differ only in their spectra and in how the
/// resulting symmetric sandwich is transported back to the original edge.
fn solve_radical_orientation(
    orientation: &RadicalOrientation,
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
    cap: usize,
) -> Option<(Mat4, f64)> {
    let mut accept = |candidate: crate::algebraic::radical::Solved| match orientation.reconstruction
    {
        RadicalReconstruction::Sandwich { transpose } => {
            if let Some(frame) = candidate.frame.as_ref() {
                let started = super::prof::start();
                let hit = frame_to_o(frame, dc, lam, targets, &orientation.eigenvalues, transpose);
                super::prof::rec(super::prof::RADICAL_FRAME, started);
                if hit.is_some() {
                    return hit;
                }
            }
            matrix_to_o(
                &candidate.m,
                dc,
                lam,
                targets,
                &orientation.eigenvalues,
                transpose,
            )
        }
        RadicalReconstruction::Reanchor { transpose } => {
            let spectrum =
                Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&orientation.eigenvalues));
            let recovered = recover_frame(&candidate.m, &spectrum);
            let o = super::apply_transpose(recovered, transpose);
            super::certify_frame_against_targets(o, dc, lam, targets)
        }
    };
    let started = super::prof::start();
    let hit = crate::algebraic::radical::solve_oriented(
        &orientation.gate,
        &orientation.prefix,
        &orientation.output,
        cap,
        &mut accept,
    );
    super::prof::rec(super::prof::RADICAL_ORIENTED, started);
    hit
}

/// The reachable-trace hull of the sandwich: m1 = tr(D_c O Lam_g O^T D_c)
/// = a^T B g with a = diag(D_c)^2, g = diag(Lam_g), and B = O∘O doubly
/// stochastic, so over EVERY admissible frame the trace ranges exactly over
/// the convex hull of the 24 permutation sums sum_k a_k g_pi(k) (Birkhoff).
/// The hull depends only on (a, g): build once, query per rho branch.
struct TraceHull {
    hull: [[f64; 2]; 26],
    m: usize,
}

fn cross2(o: [f64; 2], p: [f64; 2], q: [f64; 2]) -> f64 {
    (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
}

impl TraceHull {
    fn new(a: &[C; 4], g: &[C; 4]) -> TraceHull {
        let mut pts = [[0.0f64; 2]; 24];
        let mut n = 0;
        for i in 0..4 {
            for j in 0..4 {
                if j == i {
                    continue;
                }
                for k in 0..4 {
                    if k == i || k == j {
                        continue;
                    }
                    let l = 6 - i - j - k;
                    let s = a[0] * g[i] + a[1] * g[j] + a[2] * g[k] + a[3] * g[l];
                    pts[n] = [s.re, s.im];
                    n += 1;
                }
            }
        }
        pts.sort_unstable_by(|p, q| p.partial_cmp(q).expect("finite trace sums"));
        // Andrew monotone chain; boundary counterclockwise.
        let mut hull = [[0.0f64; 2]; 26];
        let mut m = 0;
        for &pt in &pts[..n] {
            while m >= 2 && cross2(hull[m - 2], hull[m - 1], pt) <= 0.0 {
                m -= 1;
            }
            hull[m] = pt;
            m += 1;
        }
        let lower = m + 1;
        for &pt in pts[..n - 1].iter().rev() {
            while m >= lower && cross2(hull[m - 2], hull[m - 1], pt) <= 0.0 {
                m -= 1;
            }
            hull[m] = pt;
            m += 1;
        }
        m -= 1; // the closing point repeats the first
        TraceHull { hull, m }
    }

    /// True when the trace provably lies OUTSIDE the reachable set: no
    /// doubly stochastic B (a fortiori no frame) attains this spectrum.
    /// Sign-exact; no tolerance.
    fn refutes(&self, trace: C) -> bool {
        let q = [trace.re, trace.im];
        match self.m {
            0 => false,
            1 => q != self.hull[0],
            2 => {
                let (p0, p1) = (self.hull[0], self.hull[1]);
                let (ex, ey) = (p1[0] - p0[0], p1[1] - p0[1]);
                let e2 = ex * ex + ey * ey;
                if e2 < 1e-300 {
                    return q != p0;
                }
                let t = (((q[0] - p0[0]) * ex + (q[1] - p0[1]) * ey) / e2).clamp(0.0, 1.0);
                let foot = [p0[0] + t * ex, p0[1] + t * ey];
                (q[0] - foot[0]).hypot(q[1] - foot[1]) > 0.0
            }
            m => (0..m).any(|i| cross2(self.hull[i], self.hull[(i + 1) % m], q) < 0.0),
        }
    }
}

pub(crate) fn solve_radical(
    c_in: &[C; 4],
    g_in: &[C; 4],
    target_specs: &[[C; 4]; 2],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
    strata: StratumSignature,
) -> Option<(Mat4, f64)> {
    let pcg: C = c_in.iter().product::<C>() * g_in.iter().product::<C>();
    // Demote a rho branch whose trace the Birkhoff hull REFUTES (provably no
    // frame attains that spectrum): first-hit then skips its dead orientation
    // sweep whenever the other branch solves.  Refutation only -- when both
    // branches are hull-feasible the hull carries no winner information and
    // the natural order stands (deeper-clearance ordering was measured to
    // misroute the generic linspace band 5x at p99.9).  A permutation of the
    // search cannot change coverage; the sign test needs no tolerance.
    let a2: [C; 4] = std::array::from_fn(|k| dc[(k, k)] * dc[(k, k)]);
    let lamd: [C; 4] = std::array::from_fn(|k| lam[(k, k)]);
    let branch_order = {
        let hull = TraceHull::new(&a2, &lamd);
        if hull.refutes(target_specs[0].iter().sum()) && !hull.refutes(target_specs[1].iter().sum())
        {
            [1usize, 0]
        } else {
            [0, 1]
        }
    };
    // The compiler boundary demands a machine-scale (< 1e-12) certificate at a
    // repeated target.  A first-branch hit below that bar must not end the
    // branch/orientation search -- it would mask a machine-precise candidate in
    // the other lift (the orbit census traced every "rho-unique" Radical win to
    // exactly this short-circuit).  Hold it as the fallback instead.
    let boundary_exact = strata.target.iter().any(|kind| kind.is_repeated());
    let mut held: Option<(Mat4, f64)> = None;
    for bi in branch_order {
        let w = &target_specs[bi];
        if (pcg - w.iter().product::<C>()).norm() > 1e-8 {
            continue; // det-inconsistent rho branch
        }
        // Radical formulas own exact multiplicity strata. `sandwich` uses a
        // much wider clustering radius internally to condition confluent
        // formulas, but that radius must not be used as the dispatch rule:
        // collapsing a merely close pair cannot pass the 1e-9 forward
        // certificate and caused millisecond generic-row declines. On the
        // unit circle, sqrt(machine epsilon) cleanly separates algebraically
        // coincident input eigenvalues from ordinary near-degeneracy.
        if !strata.c.is_repeated() && !strata.g.is_repeated() && !strata.target[bi].is_repeated() {
            continue;
        }
        let wg: [C; 4] = std::array::from_fn(|k| w[k].conj());
        let ct: [C; 4] = std::array::from_fn(|k| c_in[k].conj());
        let gt: [C; 4] = std::array::from_fn(|k| g_in[k].conj());
        let dcd = a2;
        let orientations = [
            RadicalOrientation {
                gate: *g_in,
                prefix: *c_in,
                output: *w,
                eigenvalues: lamd,
                owner: strata.g,
                reconstruction: RadicalReconstruction::Sandwich { transpose: false },
            },
            RadicalOrientation {
                gate: *c_in,
                prefix: *g_in,
                output: *w,
                eigenvalues: dcd,
                owner: strata.c,
                reconstruction: RadicalReconstruction::Sandwich { transpose: true },
            },
            RadicalOrientation {
                gate: wg,
                prefix: *g_in,
                output: ct,
                eigenvalues: ct,
                owner: strata.target[bi],
                reconstruction: RadicalReconstruction::Reanchor { transpose: true },
            },
            RadicalOrientation {
                gate: wg,
                prefix: *c_in,
                output: gt,
                eigenvalues: gt,
                owner: strata.target[bi],
                reconstruction: RadicalReconstruction::Reanchor { transpose: false },
            },
        ];
        // The hyperelliptic characteristic set is already exhausted by the
        // 36 even skeletons and 28 odd pin pairs. The old split quartic is an
        // alternate boundary representation, not an additional characteristic;
        // after vertex/edge/face decline it has no independent ownership. Thus
        // production needs only the shallow pin-pair family: six hot cells,
        // followed by the remaining odd characteristics.
        // The unbounded second pass owns only 0.49% of the rows, but removing
        // it drops coverage (14 unsolved on the locked corpus), degrades the
        // worst residual by 6x, and costs 4x mean latency as the fallthrough
        // hits the interior rungs.
        const PASS_MAJOR: [(usize, usize); 8] = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
        ];
        const TARGET_COMPLETE_FIRST: [(usize, usize); 4] = [(0, 2), (1, 2), (0, 3), (1, 3)];
        let target_only =
            strata.target[bi].is_rank_two() && !strata.g.is_repeated() && !strata.c.is_repeated();
        let schedule: &[(usize, usize)] = if target_only {
            &TARGET_COMPLETE_FIRST
        } else {
            &PASS_MAJOR
        };
        let mut deferred_target: Option<(Mat4, f64)> = None;
        for &(pass, action) in schedule {
            let cap = if pass == 0 { 6 } else { usize::MAX };
            // Scalar and rank-one spectra have one closed formula. Repeating
            // it in the full pass is provably identical work; only rank two
            // has remaining odd characteristics.
            let active =
                |kind: SpectrumKind| kind.is_rank_two() || (pass == 0 && kind.is_repeated());
            let orientation = &orientations[action];
            if !active(orientation.owner) {
                continue;
            }
            if let Some(mut hit) = solve_radical_orientation(orientation, dc, lam, targets, cap) {
                if action == 2 && target_only && pass == 1 && hit.1 >= 1e-12 {
                    deferred_target = Some(hit);
                    continue;
                }
                if action == 3 {
                    hit = deferred_target
                        .take()
                        .filter(|old| old.1 < hit.1)
                        .unwrap_or(hit);
                }
                if boundary_exact && hit.1 >= 1e-12 {
                    if held.as_ref().is_none_or(|old| hit.1 < old.1) {
                        held = Some(hit);
                    }
                    break; // this lift's best is boundary-doomed; try the other lift
                }
                return Some(hit);
            }
        }
        if let Some(hit) = deferred_target {
            if boundary_exact && hit.1 >= 1e-12 {
                if held.as_ref().is_none_or(|old| hit.1 < old.1) {
                    held = Some(hit);
                }
                continue;
            }
            return Some(hit);
        }
    }
    held
}

/// Recover the physical frame from the symmetric sandwich matrix already
/// produced by the radical algebra.
///
/// In the direct orientation, `m = D_c O Lambda_g O^T D_c`; in the swapped
/// orientation the same identity holds with `D_g` on the outside and the
/// recovered frame is transposed back.  Stripping the known diagonal factor
/// leaves a symmetric unitary whose real eigenvectors are the desired frame.
/// This avoids treating approximate residue vectors as if they were already
/// orthonormal.
fn matrix_to_o(
    m: &Mat4,
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
    eigenvalues: &[C; 4],
    transpose: bool,
) -> Option<(Mat4, f64)> {
    let outer = if transpose {
        let roots: [C; 4] = std::array::from_fn(|index| lam[(index, index)].sqrt());
        Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&roots))
    } else {
        *dc
    };
    let inverse = Mat4::from_diagonal(&nalgebra::Vector4::from_fn(|index, _| {
        outer[(index, index)].conj()
    }));
    let symmetric = inverse * m * inverse;
    let spectrum = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(eigenvalues));
    let frame = recover_frame(&symmetric, &spectrum);
    let o = super::apply_transpose(frame, transpose);
    super::certify_frame_against_targets(o, dc, lam, targets)
}

/// Rebuild the production frame from radical-stratum ingredients. Each peel
/// column must sit at a `vald` position carrying its peel value `c + rho`
/// (vald = the lam diagonal for the production orientation, the prefix
/// diagonal for the role-swapped one, where the built frame is transposed);
/// the remaining positions must carry the anchor `c` and take the real
/// orthonormal completion of the peel span. Small exact-match assignment
/// enumeration (multiplicity ties are gauge); accept on the production
/// smooth residual.
fn frame_to_o(
    fr: &crate::algebraic::radical::Frame,
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
    vald: &[C; 4],
    transpose: bool,
) -> Option<(Mat4, f64)> {
    let np = fr.peels.len();
    // candidate positions per peel (by value), and the c-positions
    let pos_ok =
        |val: C| -> Vec<usize> { (0..4).filter(|&j| (vald[j] - val).norm() < 1e-6).collect() };
    let cpos: Vec<usize> = pos_ok(fr.c);
    if cpos.len() < 4 - fr.peels.len() {
        return None;
    }
    let cands: Vec<Vec<usize>> = fr
        .peels
        .iter()
        .map(|&(_, rho)| pos_ok(fr.c + rho))
        .collect();
    if cands.iter().any(|c| c.is_empty()) {
        return None;
    }
    // real orthonormal completion of the peel span (Gram-Schmidt on e_k)
    let mut basis: Vec<[f64; 4]> = fr.peels.iter().map(|&(v, _)| v).collect();
    for k in 0..4 {
        if basis.len() == 4 {
            break;
        }
        let mut v = [0.0; 4];
        v[k] = 1.0;
        for b in &basis {
            let d: f64 = (0..4).map(|i| b[i] * v[i]).sum();
            for i in 0..4 {
                v[i] -= d * b[i];
            }
        }
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-8 {
            for x in &mut v {
                *x /= n;
            }
            basis.push(v);
        }
    }
    if basis.len() != 4 {
        return None;
    }
    // Column placement and the optional transpose preserve the Gram matrix,
    // so certify the four real basis vectors once before enumerating repeated-
    // eigenvalue assignments.  This is both cheaper and stronger than
    // rechecking a 4x4 matrix after every assignment.
    let mut gram_residual = 0.0f64;
    for i in 0..4 {
        for j in 0..4 {
            let dot: f64 = (0..4).map(|row| basis[i][row] * basis[j][row]).sum();
            gram_residual = gram_residual.max((dot - f64::from(i == j)).abs());
        }
    }
    if !gram_residual.is_finite() || gram_residual > 2e-10 {
        return None;
    }
    // enumerate injective peel->position assignments (<= 12 for np <= 2)
    let mut idx = vec![0usize; np];
    loop {
        let chosen: Vec<usize> = (0..np).map(|i| cands[i][idx[i]]).collect();
        let distinct = (0..np).all(|i| (0..i).all(|j| chosen[j] != chosen[i]));
        if distinct {
            let mut cols = [[0.0f64; 4]; 4];
            let mut used = [false; 4];
            for i in 0..np {
                cols[chosen[i]] = basis[i];
                used[chosen[i]] = true;
            }
            let mut nfree = np;
            for j in 0..4 {
                if !used[j] {
                    if !cpos.contains(&j) {
                        nfree = usize::MAX; // a non-c position left uncovered
                        break;
                    }
                    cols[j] = basis[nfree];
                    nfree += 1;
                }
            }
            if nfree != usize::MAX {
                let raw = Mat4::from_fn(|i, j| C::new(cols[j][i], 0.0));
                let o = super::apply_transpose(raw, transpose);
                if let Some(hit) = super::certify_frame_against_targets(o, dc, lam, targets) {
                    return Some(hit);
                }
            }
        }
        // odometer over cands
        let mut k = 0;
        loop {
            if k == np {
                return None;
            }
            idx[k] += 1;
            if idx[k] < cands[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// The multiplicity recursion, split out so the CHEAP exact charts (Klein and
/// its dual) get a look at the confluent rows first. Routing those rows to the
/// interior chart scans instead costs 525 machine-precise rows; Klein
/// behind the same FAST_ACCEPT gate, so precision is protected by construction
/// and only the ordering changes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_confluent(
    c_in: &[C; 4],
    g_in: &[C; 4],
    target_specs: &[[C; 4]; 2],
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]; 2],
    strata: StratumSignature,
) -> Option<(Mat4, f64, super::Rung)> {
    if strata.g == SpectrumKind::Pair22 || strata.c == SpectrumKind::Pair22 {
        let tp = super::prof::start();
        let hit = super::two_plus_two::solve(
            c_in,
            g_in,
            target_specs,
            dc,
            lam,
            targets,
            strata.g == SpectrumKind::Pair22,
            strata.c == SpectrumKind::Pair22,
            [false; 2],
            true,
            false,
        );
        super::prof::rec(super::prof::RADICAL_TOTAL, tp);
        if let Some((o, r)) = hit {
            return Some((o, r, super::Rung::Pair22));
        }
    }
    let tp = super::prof::start();
    let rad_hit = if strata.has_confluence() {
        solve_radical(c_in, g_in, target_specs, dc, lam, targets, strata)
    } else {
        None
    };
    super::prof::rec(super::prof::RADICAL_TOTAL, tp);
    let target_repeated = strata.target.iter().any(|kind| kind.is_repeated());
    if let Some((o, r)) = rad_hit {
        // At a repeated target the compiler boundary demands a machine-scale
        // certificate; a looser radical hit can never pass it, so it must not
        // mask the degenerate-limit formula below.
        if !target_repeated || r < 1e-12 {
            return Some((o, r, super::Rung::Radical));
        }
    }
    // Double confluence (a repeated inner value AND a repeated target value):
    // a doubled target value collides two branch points of the radical
    // machinery's curve, degenerating its pin-pair characteristics -- and the
    // same collision forces a rank-two kernel whose vanishing interaction
    // matrix turns the spectral conditions linear.  The resonance construction
    // is that degenerate limit; it runs exactly on the radical decline set, so
    // the population the curve formulas own pays nothing.
    let target_deep = strata.target.iter().any(|kind| {
        matches!(
            kind,
            super::SpectrumKind::Triple31 | super::SpectrumKind::Scalar4
        )
    });
    if (strata.c.is_repeated() || strata.g.is_repeated() || target_deep) && target_repeated {
        let tp = super::prof::start();
        let hit = super::resonance::solve(c_in, g_in, target_specs, dc, lam, targets);
        super::prof::rec(super::prof::RADICAL_TOTAL, tp);
        if let Some((o, r)) = hit
            && r < 1e-12
        {
            return Some((o, r, super::Rung::Radical));
        }
    }
    if let Some((o, r)) = rad_hit {
        // Preserve the pre-resonance fall-through: the boundary will judge it.
        return Some((o, r, super::Rung::Radical));
    }
    if strata.g == SpectrumKind::Pair22 || strata.c == SpectrumKind::Pair22 {
        let tp = super::prof::start();
        let hit = super::two_plus_two::solve(
            c_in,
            g_in,
            target_specs,
            dc,
            lam,
            targets,
            strata.g == SpectrumKind::Pair22,
            strata.c == SpectrumKind::Pair22,
            [false; 2],
            false,
            true,
        );
        super::prof::rec(super::prof::RADICAL_TOTAL, tp);
        if let Some((o, r)) = hit {
            return Some((o, r, super::Rung::Pair22));
        }
    }
    if strata.target.contains(&SpectrumKind::Pair22) {
        let tp = super::prof::start();
        let hit = super::two_plus_two::solve(
            c_in,
            g_in,
            target_specs,
            dc,
            lam,
            targets,
            false,
            false,
            strata.target.map(|kind| kind == SpectrumKind::Pair22),
            true,
            true,
        );
        super::prof::rec(super::prof::RADICAL_TOTAL, tp);
        if let Some((o, r)) = hit {
            return Some((o, r, super::Rung::Pair22));
        }
    }
    None
}
