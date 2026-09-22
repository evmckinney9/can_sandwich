//! Research entry points and optional stage timing.
#[cfg(feature = "diagnostics")]
use crate::algebraic::{
    self, ACCEPT, Problem, Rung, Solution, certificate, charts, compiler_solution, esym4, interior,
    symfn, two_plus_two, unsolved_solution,
};
#[cfg(feature = "diagnostics")]
use crate::problem::{eigphases, weyl_from_monodromy};
#[cfg(feature = "diagnostics")]
use crate::{C, ComplexMatrix as Mat4};
/// The diagonal phase matrix `D = mb(Can(w))` (canonical gates are diagonal in the
/// magic basis). Built directly as `diag(exp(i*eigphases(w)))` in the ordering
/// documented by [`eigphases`].
#[cfg(feature = "diagnostics")]
pub(crate) fn dphase(w: [f64; 3]) -> Mat4 {
    let ph = eigphases(w);
    let d: [C; 4] = std::array::from_fn(|k| C::from_polar(1.0, ph[k]));
    Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&d))
}

/// Return the canonical right endpoint frame of a certified child.
/// This is the exact frame needed to insert the local layer between two
/// adjacent entanglers in a factorization chain.
#[cfg(feature = "diagnostics")]
pub fn endpoint_right_gauge(c: [f64; 3], g: [f64; 3], t: [f64; 3], frame: &Mat4) -> Option<Mat4> {
    let problem = Problem::new(c, g, t);
    certificate::canonical_right_endpoint_gauge(&problem, frame)
        .or_else(|| certificate::endpoint_factorization(&problem, frame).map(|(_, r)| r))
}

#[cfg(feature = "diagnostics")]
pub fn endpoint_gauge_residual(c: [f64; 3], g: [f64; 3], t: [f64; 3], frame: &Mat4) -> f64 {
    let problem = Problem::new(c, g, t);
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
    let problem = Problem::new([0.0; 3], g, [0.0; 3]);
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

/// Berkeley (2+2) CS masses of an orthogonal frame.  These are the squared
/// singular-value invariants of the leading 2x2 block; unlike a frame gauge,
/// they survive the left/right O(2)xO(2) actions.
#[cfg(feature = "diagnostics")]
pub(crate) fn cs_masses(o: &Mat4) -> (f64, f64) {
    let a00 = o[(0, 0)].re;
    let a01 = o[(0, 1)].re;
    let a10 = o[(1, 0)].re;
    let a11 = o[(1, 1)].re;
    let s = a00 * a00 + a01 * a01 + a10 * a10 + a11 * a11;
    let det = a00 * a11 - a01 * a10;
    (s, det * det)
}

/// Stable, endpoint-computable branch bin used by corpus census tooling.
/// This deliberately records predicates independently of the first successful
/// rung, so a new branch can be measured as a routing change rather than only
/// as a new success count.
#[cfg(feature = "diagnostics")]
pub fn branch_signature(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> String {
    let p = Problem::new(c, g, t);
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

/// Build the interior quotient before timing the solver.
#[cfg(feature = "diagnostics")]
pub fn init_tables() {
    let _ = &*interior::INTERIOR_QUOTIENT;
}

#[cfg(feature = "diagnostics")]
fn solve_report(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    crate::solve_using(c, g, t, |_, solution, _| Some(solution)).unwrap_or_else(unsolved_solution)
}

/// Enumerate the finite ordered interior chart witnesses.  A generic child
/// realization has a continuous fiber; these are distinct closed-form
/// transversal witnesses exposed by the atlas, rather than repeated calls to
/// the first-hit production selector.
#[cfg(feature = "diagnostics")]
pub fn ordered_chart_solutions(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Vec<Solution> {
    let problem = Problem::new(c, g, t);
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
    if out.is_empty()
        && let Some(solution) = algebraic::solve(&problem)
    {
        out.push(solution);
    }
    out
}

/// Factor a target canonical gate through the fixed Berkeley entangler:
/// `B * V * B ~ target`, where the returned frame is the local middle gate
/// `V` in the magic basis.  Since both outer factors are 2+2, this dispatches
/// directly to the Pair22/Heron selector rather than the generic tail.
#[cfg(feature = "diagnostics")]
pub fn factor_through_berkeley(target: [f64; 3]) -> Option<Mat4> {
    // Berkeley canonical coordinates are (1/2,1/4,0); convert through the
    // package's monodromy convention c=(m0+m1,m0+m2,m1+m2).
    const B: [f64; 3] = [0.375, 0.125, -0.125];
    let solution = solve_report(B, B, target);
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
    let child = solve_report(c, h, middle);
    if child.rung == Rung::Unsolved {
        return None;
    }
    let problem = Problem::new(c, g, t);
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
    let problem = Problem::new(c, g, t);
    certificate::compiler_solution(&problem, o, rung, ACCEPT)
}

/// Evaluate one proposed waypoint for the virtual factorization
/// `G = B V B`. The caller supplies `M` (for example from a reachable-polytope
/// intersection); this routine realizes both B-children and applies the exact
/// endpoint-Grassmannian compatibility test. A small residual means the two
/// child certificates can be stitched with the fixed middle local `V`.
#[cfg(feature = "diagnostics")]
pub fn solve_factorized_waypoint(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    waypoint: [f64; 3],
) -> Option<(Solution, Solution, Mat4, f64)> {
    const B: [f64; 3] = [0.375, 0.125, -0.125];
    let middle = factor_through_berkeley(g)?;
    let first = solve_report(c, B, waypoint);
    let second = solve_report(waypoint, B, t);
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
                candidates.push((a, *b));
            }
        }
    }
    let mut best = None;
    for (first, second) in candidates {
        let first_problem = Problem::new(c, B, waypoint);
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
            let first_problem = Problem::new(c, B, waypoint);
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
                    best = Some((first, *second, middle, residual));
                }
                if residual <= ACCEPT {
                    return Some((first, *second, middle, residual));
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
    let first_problem = Problem::new(c, [0.375, 0.125, -0.125], waypoint);
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
    let factor_problem = Problem::new(B, B, g);
    let problem = Problem::new(c, g, t);
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

/// The chart tier alone: `Problem` then `charts::solve_full`, with no
/// prefix rung, representative orbit, hold or adjacent-stratum lattice.
/// Diagnostics only: measures whether everything above the tail is load-bearing.
#[cfg(feature = "diagnostics")]
pub fn solve_charts_only(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Solution {
    let problem = Problem::new(c, g, t);
    charts::solve_full(&problem).unwrap_or_else(unsolved_solution)
}

/// Minimum within-pair root gap over recognized nonscalar paired inputs.
/// This is a floating-point applicability label, not a theorem-domain test.
#[cfg(feature = "diagnostics")]
pub fn paired_edge_scope(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<f64> {
    let problem = Problem::new(c, g, t);
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
    let problem = Problem::new(c, g, t);
    two_plus_two::solve_paired_edges_with(&problem, backward, |o, residual| {
        compiler_solution(&problem, o, Rung::Pair22, residual)
    })
    .unwrap_or_else(unsolved_solution)
}

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
    #[cfg(feature = "diagnostics")]
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
