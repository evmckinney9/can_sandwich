//! Route reports, candidate verification, and optional stage timing.
#[cfg(feature = "diagnostics")]
use crate::{
    ComplexMatrix as Mat4, Rung, Solution,
    algebraic::{ACCEPT, compiler_solution},
    problem::Problem,
};

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

/// Solve and report the accepted construction and its spectral error.
#[cfg(feature = "diagnostics")]
pub fn solve_report(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Solution> {
    crate::witness(c, g, t).ok().map(|(_, solution)| solution)
}

/// Re-certify an externally selected frame against the original sandwich.
/// A frame that fails the spectral check can come back refined or retargeted,
/// as in the solver's own acceptance path.
#[cfg(feature = "diagnostics")]
pub fn certify_frame(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    o: Mat4,
    rung: Rung,
) -> Option<Solution> {
    let problem = Problem::new(c, g, t);
    compiler_solution(&problem, o, rung, ACCEPT)
}

/// PROF=1 instrumentation: per-stage aggregate ns/calls across a corpus run.
/// Compiled out unless the `diagnostics` feature is enabled.
pub mod prof {
    #[cfg(feature = "diagnostics")]
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    #[cfg(feature = "diagnostics")]
    pub const N: usize = 44;
    /// Slot indices, one per stage or event counter.
    pub const EDGE: usize = 0;
    pub const CHART_COEFFS: usize = 1;
    pub const ELIM: usize = 2;
    pub const ROOTS: usize = 3;
    pub const RADICAL_ORIENTED: usize = 4;
    pub const RADICAL_FRAME: usize = 5;
    pub const INTERIOR_TOTAL: usize = 6;
    pub const RADICAL_TOTAL: usize = 7;
    pub const PRELUDE: usize = 8;
    pub const FACE: usize = 9;
    pub const SW_RANK1: usize = 10;
    pub const SW_TWO_STEP: usize = 11;
    pub const SW_MIRROR: usize = 12;
    pub const SW_PAIR_ROOTS: usize = 13;
    pub const SW_WORD_GATE: usize = 14;
    pub const GATE_BOX: usize = 15;
    pub const GATE_HULL: usize = 16;
    pub const GATE_PASS: usize = 17;
    pub const N_TRY_MU: usize = 18;
    pub const RJ_BASE: usize = 19;
    pub const RJ_SKEL: usize = 20;
    pub const RJ_UREAL: usize = 21;
    pub const RJ_USUM: usize = 22;
    pub const N_CONSTRUCTED: usize = 23;
    pub const RJ_V_IMAG: usize = 24;
    pub const RJ_V_NEG: usize = 25;
    pub const RJ_V_SUM: usize = 26;
    pub const SW_BASE: usize = 27;
    pub const ARC_PAIR_SKIP: usize = 28;
    pub const ARC_CAND_SKIP: usize = 29;
    pub const BASE_FAIL_FORCED: usize = 30;
    pub const BASE_FAIL_SKEL: usize = 31;
    pub const BASE_FAIL_PAIR: usize = 32;
    pub const DEV_LT_1EM6: usize = 33;
    pub const DEV_MID: usize = 34;
    pub const DEV_GT_1EM2: usize = 35;
    pub const PAIR_GATE_SOME: usize = 36;
    pub const PAIR_GATE_NONE: usize = 37;
    pub const INH_SKIP: usize = 38;
    pub const SW_HEADER: usize = 39;
    pub const KLEIN_TOTAL: usize = 40;
    pub const SEG_PREPARE: usize = 41;
    pub const SEG_EDGEGATE: usize = 42;
    pub const SEG_VERTEX: usize = 43;
    #[cfg(feature = "diagnostics")]
    pub const NAMES: [&str; N] = [
        "edge",
        "chart_coeffs",
        "elim",
        "roots",
        "radical_oriented",
        "radical_frame",
        "interior_total",
        "radical_total",
        "prelude",
        "face",
        "sw_rank1",
        "sw_two_step",
        "sw_mirror",
        "sw_pair_roots",
        "sw_word_gate",
        "gate_box",
        "gate_hull",
        "gate_pass",
        "n_try_mu",
        "rj_base",
        "rj_skel",
        "rj_ureal",
        "rj_usum",
        "n_constructed",
        "rj_v_imag",
        "rj_v_neg",
        "rj_v_sum",
        "sw_base",
        "arc_pair_skip",
        "arc_cand_skip",
        "base_fail_forced",
        "base_fail_skel",
        "base_fail_pair",
        "dev_lt_1em6",
        "dev_mid",
        "dev_gt_1em2",
        "pair_gate_some",
        "pair_gate_none",
        "inh_skip",
        "sw_header",
        "klein_total",
        "seg_prepare",
        "seg_edgegate",
        "seg_vertex",
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
